#!/usr/bin/env python3
"""BigCodeBench dataset helpers used by unified adapters."""

import json
import os
import re
from typing import Dict, Optional

from evaluation.pipeline.cache_paths import canonical_cache_path


def _extract_code_block_from_text(generated_text: str) -> Optional[str]:
    """
    Extract a Python code block from a thinking model's generated text.

    This only handles "text -> code block" extraction. It does NOT attempt
    to validate or sanitize code. Use code_extract() afterwards.
    """
    if not generated_text:
        return None

    pattern = r"```[Pp]ython\s*\n(.*?)```"
    matches = re.findall(pattern, generated_text, re.DOTALL)
    if matches:
        valid_blocks = []
        for match in matches:
            code = match.strip()
            if len(code) > 20:
                valid_blocks.append(code)
        if valid_blocks:
            return valid_blocks[-1]
        return matches[-1].strip()

    pattern2 = r"```\s*\n(.*?)```"
    matches2 = re.findall(pattern2, generated_text, re.DOTALL)
    if matches2:
        for match in reversed(matches2):
            code = match.strip()
            if len(code) > 20:
                return code

    lines = generated_text.split("\n")
    code_start = -1
    code_end = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ")):
            if code_start == -1:
                code_start = i
            code_end = i
        elif code_start >= 0 and stripped and not stripped.startswith("#"):
            if line.startswith((" ", "\t")):
                code_end = i
            elif "=" in stripped or "(" in stripped:
                code_end = i
    if code_start >= 0:
        return "\n".join(lines[code_start : code_end + 1])

    return None


def extract_python_code_from_thinking(generated_text: str) -> Optional[str]:
    """
    Extract Python code from a thinking model's generated text, then refine
    it with BigCodeBench's code_extract() to keep the longest valid snippet.
    """
    from bigcodebench.sanitize import code_extract

    raw_code = _extract_code_block_from_text(generated_text)
    if not raw_code:
        return None
    try:
        refined = code_extract(raw_code)
        return refined.strip() if refined else raw_code.strip()
    except Exception:
        return raw_code.strip()


def convert_to_bigcodebench_jsonl(
    result_dir: str,
    output_path: str = None,
    select_strategy: str = "all",
    trajectory_index: int = 0,
) -> str:
    """
    Stream-convert entropy_results.json to BigCodeBench JSONL format.

    Uses ijson for streaming to handle very large files (100GB+).
    """
    import ijson

    entropy_file = os.path.join(result_dir, "entropy_results.json")
    if not os.path.exists(entropy_file):
        raise FileNotFoundError(f"entropy_results.json not found in {result_dir}")

    if output_path is None:
        output_path = os.path.join(result_dir, "bigcodebench_samples.jsonl")

    file_size_gb = os.path.getsize(entropy_file) / (1024**3)
    print(f"Converting entropy_results.json ({file_size_gb:.1f}GB) to BigCodeBench JSONL...")
    print(f"  Strategy: {select_strategy}")

    count = 0
    skipped = 0
    failed_extract = 0
    failed_ids = []

    with open(output_path, "w") as out_f:
        with open(entropy_file, "rb") as f:
            parser = ijson.items(f, "item")

            for sample in parser:
                traj_id = sample.get("id", "")
                original_id = sample.get("original_id", "")
                traj_index = sample.get("trajectory_index", 0)

                if select_strategy == "first" and traj_index != 0:
                    skipped += 1
                    continue
                if select_strategy == "index" and traj_index != trajectory_index:
                    skipped += 1
                    continue

                generated_text = sample.get("generated_text", "")
                code = extract_python_code_from_thinking(generated_text)
                if code is None:
                    failed_extract += 1
                    if traj_id:
                        failed_ids.append(traj_id)
                    code = "# Failed to extract code\npass"

                task_id = original_id if original_id else traj_id.split("_traj_")[0]
                entry = {
                    "task_id": task_id,
                    "solution": code,
                    "_identifier": f"{task_id} (trajectory {traj_index})",
                }
                out_f.write(json.dumps(entry) + "\n")
                count += 1

                if count % 5000 == 0:
                    print(f"    Converted {count} trajectories...")

    print(f"  ✓ Converted {count} trajectories to {output_path}")
    print(f"    Skipped: {skipped}, Failed extraction: {failed_extract}")
    if failed_ids:
        failed_path = os.path.join(result_dir, "failed_extraction_ids.txt")
        with open(failed_path, "w") as f:
            for tid in failed_ids:
                f.write(tid + "\n")
        print(f"    Saved failed extraction IDs to: {failed_path}")

    return output_path


def sanitize_samples(samples_path: str, calibrate: bool = False) -> str:
    """Run BigCodeBench sanitizer on the JSONL file."""
    from bigcodebench.data import get_bigcodebench, load_solutions, write_jsonl
    from bigcodebench.sanitize import sanitize as sanitize_code
    from bigcodebench.sanitize import script as sanitize_script

    print(f"Sanitizing samples: {samples_path}")
    if calibrate:
        sanitized_path = samples_path.replace(".jsonl", "-sanitized-calibrated.jsonl")
    else:
        sanitized_path = samples_path.replace(".jsonl", "-sanitized.jsonl")

    try:
        sanitize_script(samples=samples_path, calibrate=calibrate)
        if os.path.exists(sanitized_path):
            print(f"  ✓ Sanitized file: {sanitized_path}")
            return sanitized_path
    except Exception as e:
        print(f"  Warning: official sanitize failed ({e})")
        print("  Falling back to resilient sanitize mode (continue on per-sample errors)...")

    dataset = get_bigcodebench()
    entry_point = {task_id: problem["entry_point"] for task_id, problem in dataset.items()}

    good_solutions = []
    failed_records = []
    total = 0
    for sample in load_solutions(samples_path):
        total += 1
        task_id = sample.get("task_id")
        identifier = sample.get("_identifier", f"line-{total}")

        try:
            if not task_id or task_id not in dataset:
                raise ValueError(f"invalid or unknown task_id: {task_id}")

            old_code = sample.get("solution")
            if old_code is None:
                completion = sample.get("completion")
                if completion is None:
                    raise ValueError("missing both solution and completion")
                old_code = dataset[task_id]["complete_prompt"] + "\n" + completion
            elif calibrate:
                old_code = old_code.replace(
                    "```python\n    ",
                    "```python\n" + dataset[task_id]["complete_prompt"] + "    ",
                )

            new_code = sanitize_code(code=old_code, entrypoint=entry_point[task_id])
            good_solutions.append({"task_id": task_id, "solution": new_code})
        except Exception as sample_err:
            failed_records.append(
                {
                    "task_id": task_id,
                    "identifier": identifier,
                    "error": repr(sample_err),
                }
            )

    write_jsonl(sanitized_path, good_solutions)

    if failed_records:
        failed_path = os.path.join(
            os.path.dirname(samples_path),
            "sanitize_failed_task_ids.txt",
        )
        with open(failed_path, "w") as f:
            for rec in failed_records:
                f.write(
                    json.dumps(
                        {
                            "task_id": rec["task_id"],
                            "identifier": rec["identifier"],
                            "error": rec["error"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  Warning: {len(failed_records)} samples failed during sanitize")
        print(f"  Saved failure report: {failed_path}")

    print(
        f"  ✓ Resilient sanitize completed: "
        f"{len(good_solutions)}/{total} samples written to {sanitized_path}"
    )
    return sanitized_path


def run_bigcodebench_evaluation(
    samples_path: str,
    result_dir: str,
    split: str = "instruct",
    subset: str = "full",
    execution: str = "local",
    pass_k: str = "1",
    calibrated: bool = True,
    parallel: int = -1,
    min_time_limit: float = 1,
    no_gt: bool = False,
) -> Optional[Dict]:
    """Run BigCodeBench evaluation and return results."""
    from bigcodebench.evaluate import evaluate

    print("\nRunning BigCodeBench evaluation...")
    print(f"  Samples: {samples_path}")
    print(f"  Split: {split}, Subset: {subset}")
    print(f"  Execution: {execution}")
    print(f"  Pass@k: {pass_k}")
    print(f"  Calibrated: {calibrated}")

    if os.path.isdir(samples_path):
        eval_results_path = os.path.join(samples_path, "eval_results.json")
    else:
        eval_results_path = samples_path.replace(".jsonl", "_eval_results.json")
    pass_at_k_path = eval_results_path.replace("eval_results.json", "pass_at_k.json")

    for existing_file in [eval_results_path, pass_at_k_path]:
        if os.path.exists(existing_file):
            backup = existing_file + ".bak"
            print(f"  Moving existing {os.path.basename(existing_file)} -> {os.path.basename(backup)}")
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(existing_file, backup)

    try:
        evaluate(
            split=split,
            subset=subset,
            samples=samples_path,
            execution=execution,
            pass_k=pass_k,
            calibrated=calibrated,
            parallel=parallel,
            min_time_limit=min_time_limit,
            no_gt=no_gt,
            save_pass_rate=True,
        )

        if os.path.exists(eval_results_path):
            with open(eval_results_path, "r") as f:
                return json.load(f)
        print(f"  Warning: eval_results.json not found at {eval_results_path}")
        return None
    except Exception as e:
        print(f"  Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return None


def build_evaluation_cache(
    eval_results: Dict,
    result_dir: str,
    num_trajectories: int = 32,
) -> Dict:
    """Build evaluation_cache.json from BigCodeBench eval_results."""
    _ = num_trajectories  # Backward-compatible signature.
    print("\nBuilding evaluation_cache.json...")

    trajectories = {}
    total_correct = 0
    total_count = 0

    eval_data = eval_results.get("eval", {})
    for task_id, task_results in eval_data.items():
        for traj_idx, result in enumerate(task_results):
            traj_id = f"{task_id}_traj_{traj_idx}"
            status = result.get("status", "fail")
            is_correct = status == "pass"
            trajectories[traj_id] = {
                "is_correct": is_correct,
                "original_id": task_id,
                "status": status,
            }
            if is_correct:
                total_correct += 1
            total_count += 1

    cache = {
        "version": 2,
        "task_type": "bigcodebench",
        "trajectories": trajectories,
    }

    cache_path = canonical_cache_path(result_dir)
    temp_path = cache_path + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(temp_path, cache_path)
        print(f"  ✓ Saved evaluation cache: {cache_path}")
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
