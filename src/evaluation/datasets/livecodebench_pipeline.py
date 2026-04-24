#!/usr/bin/env python3
"""LiveCodeBench dataset helpers used by unified adapters."""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

from evaluation.pipeline.cache_paths import canonical_cache_path

# Use LCB's official code extraction function.
from lcb_runner.utils.extraction_utils import extract_code


def _iter_entropy_results(result_dir: str):
    """
    Iterate over entries in entropy_results, trying multiple formats.

    Tries in order:
    1. JSONL file (entropy_results.jsonl)
    2. JSON file with ijson streaming
    3. JSON file line-by-line with NaN sanitization
    """
    jsonl_path = os.path.join(result_dir, "entropy_results.jsonl")
    json_path = os.path.join(result_dir, "entropy_results.json")

    if os.path.exists(jsonl_path):
        print(f"  Using JSONL format: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Neither entropy_results.jsonl nor entropy_results.json found in {result_dir}"
        )

    try:
        import ijson

        print("  Using ijson streaming on JSON file...")
        with open(json_path, "rb") as f:
            parser = ijson.items(f, "item")
            first = next(parser)
            yield first
            for item in parser:
                yield item
        return
    except ImportError:
        pass
    except Exception as e:
        if "NaN" in str(e) or "lexical error" in str(e):
            print(f"  ijson failed (likely NaN values in JSON): {e}")
            print("  Falling back to NaN-sanitized line-by-line parsing...")
        else:
            raise

    print("  Using NaN-sanitized line-by-line parsing (may be slower)...")
    nan_pattern = re.compile(r"\bNaN\b")
    inf_pattern = re.compile(r"\bInfinity\b")
    ninf_pattern = re.compile(r"-Infinity\b")

    with open(json_path, "r", encoding="utf-8") as f:
        depth = 0
        buf = []
        in_string = False
        escape_next = False

        for line in f:
            for ch in line:
                if escape_next:
                    escape_next = False
                    if depth > 0:
                        buf.append(ch)
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    if depth > 0:
                        buf.append(ch)
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                if not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                if depth > 0 or (depth == 0 and ch == "}"):
                    buf.append(ch)

                if depth == 0 and buf and buf[-1] == "}":
                    raw = "".join(buf)
                    buf = []
                    raw = nan_pattern.sub("null", raw)
                    raw = ninf_pattern.sub("null", raw)
                    raw = inf_pattern.sub("null", raw)
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        continue


def convert_trajectories(result_dir: str) -> str:
    """
    Stream entropy results and extract code from each trajectory.

    Produces livecodebench_extracted.json with extracted code per trajectory.
    """
    output_path = os.path.join(result_dir, "livecodebench_extracted.json")

    for fname in ["entropy_results.jsonl", "entropy_results.json"]:
        fpath = os.path.join(result_dir, fname)
        if os.path.exists(fpath):
            file_size_gb = os.path.getsize(fpath) / (1024**3)
            print(f"  Found {fname} ({file_size_gb:.1f}GB)")

    print("Extracting code from entropy results...")

    count = 0
    failed_extract = 0
    temp_path = output_path + ".tmp"

    try:
        with open(temp_path, "w") as out_f:
            out_f.write("[")
            first = True

            for sample in _iter_entropy_results(result_dir):
                traj_id = sample.get("id", "")
                original_id = sample.get("original_id", "")
                traj_index = sample.get("trajectory_index", 0)

                generated_text = sample.get("generated_text", "")
                code = extract_code(generated_text, None)
                if not code:
                    failed_extract += 1
                    code = "# Failed to extract code\npass"

                entry = {
                    "traj_id": traj_id,
                    "original_id": original_id,
                    "trajectory_index": traj_index,
                    "code": code,
                }

                if not first:
                    out_f.write(",\n")
                json.dump(entry, out_f)
                first = False

                count += 1
                if count % 5000 == 0:
                    print(f"    Extracted {count} trajectories...")

            out_f.write("]")

        os.replace(temp_path, output_path)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise

    print(f"  Converted {count} trajectories to {output_path}")
    print(f"    Failed extraction: {failed_extract}")
    return output_path


def run_livecodebench_evaluation(
    result_dir: str,
    release_version: str = "release_v5",
    num_process: int = 16,
    timeout: int = 6,
    k_list: Optional[List[int]] = None,
) -> Dict:
    """Run LiveCodeBench evaluation against test cases."""
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    def _load_code_generation_dataset(release_version, start_date=None, end_date=None):
        from datetime import datetime

        from datasets import load_dataset

        try:
            dataset = load_dataset(
                "livecodebench/code_generation_lite",
                split="test",
                version_tag=release_version,
                trust_remote_code=True,
            )
        except TypeError:
            dataset = load_dataset(
                "livecodebench/code_generation_lite",
                split="test",
                revision=release_version,
            )
        dataset = [CodeGenerationProblem(**p) for p in dataset]
        if start_date is not None:
            p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
            dataset = [e for e in dataset if p_start_date <= e.contest_date]
        if end_date is not None:
            p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
            dataset = [e for e in dataset if e.contest_date <= p_end_date]
        print(f"Loaded {len(dataset)} problems")
        return dataset

    if k_list is None:
        k_list = [1, 5, 10, 20, 32, 50, 100]

    extracted_path = os.path.join(result_dir, "livecodebench_extracted.json")
    if not os.path.exists(extracted_path):
        raise FileNotFoundError(
            f"livecodebench_extracted.json not found in {result_dir}. "
            "Run conversion first."
        )

    print(f"Loading extracted code from {extracted_path}...")
    print(f"Loading LiveCodeBench problems (version: {release_version})...")
    problems = _load_code_generation_dataset(release_version=release_version)

    problem_map = {}
    for problem in problems:
        problem_map[f"lcb_{problem.question_id}"] = problem

    code_by_problem = defaultdict(list)
    traj_order_by_problem = defaultdict(list)

    def _iter_extracted_entries(path: str):
        try:
            import ijson

            with open(path, "rb") as f:
                for item in ijson.items(f, "item"):
                    yield item
        except ImportError:
            with open(path, "r") as f:
                for item in json.load(f):
                    yield item

    seen_traj_ids = set()
    for entry in _iter_extracted_entries(extracted_path):
        traj_id = entry["traj_id"]
        if traj_id in seen_traj_ids:
            continue
        seen_traj_ids.add(traj_id)

        original_id = entry["original_id"]
        code_by_problem[original_id].append(entry["code"])
        traj_order_by_problem[original_id].append(traj_id)

    traj_counts = {pid: len(codes) for pid, codes in code_by_problem.items()}
    unique_counts = set(traj_counts.values())
    if len(unique_counts) > 1:
        from collections import Counter

        count_freq = Counter(traj_counts.values())
        expected_n = count_freq.most_common(1)[0][0]

        anomalies = {pid: cnt for pid, cnt in traj_counts.items() if cnt != expected_n}
        print("\n  WARNING: Inconsistent trajectory counts detected!")
        print(
            f"  Expected {expected_n} trajectories per problem, "
            f"but {len(anomalies)} problems differ:"
        )
        for pid, cnt in sorted(anomalies.items()):
            print(f"    {pid}: {cnt} trajectories -> truncating to {expected_n}")
            code_by_problem[pid] = code_by_problem[pid][:expected_n]
            traj_order_by_problem[pid] = traj_order_by_problem[pid][:expected_n]

    problem_ids_ordered = []
    samples_list = []
    generations_list = []
    skipped_ids = []

    for original_id in sorted(code_by_problem.keys()):
        if original_id not in problem_map:
            skipped_ids.append(original_id)
            continue
        problem = problem_map[original_id]
        eval_sample = problem.get_evaluation_sample()
        problem_ids_ordered.append(original_id)
        samples_list.append(eval_sample)
        generations_list.append(code_by_problem[original_id])

    if skipped_ids:
        print(
            f"\n  WARNING: {len(skipped_ids)} problems from inference not found "
            f"in LCB {release_version} dataset!"
        )
        print(
            "  This usually means --release_version does not match the version "
            "used during inference."
        )
        print(f"  Skipped IDs (first 10): {skipped_ids[:10]}")
        if len(skipped_ids) > 10:
            print(f"  ... and {len(skipped_ids) - 10} more")

    print(
        f"\nEvaluating {len(samples_list)} problems, "
        f"{sum(len(g) for g in generations_list)} total generations..."
    )

    metrics, results, final_metadata = codegen_metrics(
        samples_list=samples_list,
        generations_list=generations_list,
        k_list=k_list,
        num_process_evaluate=num_process,
        timeout=timeout,
    )
    _ = final_metadata

    print("\n--- Pass@k Results ---")
    for key, value in sorted(metrics.items()):
        if key != "detail":
            print(f"  {key}: {value:.4f}")

    eval_output = {
        "metrics": {k: v for k, v in metrics.items() if k != "detail"},
        "detail": metrics.get("detail", {}),
        "problem_ids": problem_ids_ordered,
        "results": {str(k): v for k, v in results.items()},
        "traj_order": {pid: traj_order_by_problem[pid] for pid in problem_ids_ordered},
    }

    eval_path = os.path.join(result_dir, "livecodebench_eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\n  Saved evaluation results to: {eval_path}")
    return eval_output


def build_evaluation_cache(result_dir: str) -> Dict:
    """Build evaluation_cache.json from LiveCodeBench evaluation results."""
    eval_path = os.path.join(result_dir, "livecodebench_eval_results.json")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"livecodebench_eval_results.json not found in {result_dir}. "
            "Run evaluation first."
        )

    print("\nBuilding evaluation_cache.json...")
    with open(eval_path, "r") as f:
        eval_output = json.load(f)

    problem_ids = eval_output["problem_ids"]
    results = eval_output["results"]
    traj_order = eval_output["traj_order"]

    trajectories = {}
    total_correct = 0
    total_count = 0

    for prob_idx, problem_id in enumerate(problem_ids):
        prob_results = results.get(str(prob_idx), [])
        traj_ids = traj_order.get(problem_id, [])

        for traj_idx, test_results in enumerate(prob_results):
            if traj_idx < len(traj_ids):
                traj_id = traj_ids[traj_idx]
            else:
                traj_id = f"{problem_id}_traj_{traj_idx}"

            is_correct = all((r is True) or (r == 1) for r in test_results) if test_results else False
            status = "pass" if is_correct else "fail"

            trajectories[traj_id] = {
                "is_correct": is_correct,
                "original_id": problem_id,
                "status": status,
            }

            if is_correct:
                total_correct += 1
            total_count += 1

    cache = {
        "version": 2,
        "task_type": "livecodebench",
        "trajectories": trajectories,
    }

    cache_path = canonical_cache_path(result_dir)
    temp_path = cache_path + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(temp_path, cache_path)
        print(f"  Saved evaluation cache: {cache_path}")
        print(f"    Total trajectories: {total_count}")
        if total_count > 0:
            print(f"    Correct: {total_correct} ({total_correct / total_count * 100:.2f}%)")
    except Exception as e:
        print(f"  Warning: Failed to save cache: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return cache


def detect_release_version(result_dir: str) -> Optional[str]:
    """
    Auto-detect LCB release version from the result directory path.

    Looks for patterns like 'release_v6'.
    """
    match = re.search(r"release_v\d+", result_dir)
    if match:
        return match.group(0)
    return None
