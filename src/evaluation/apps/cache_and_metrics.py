"""Cache preparation and pass@k metrics for answer evaluation CLI."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from evaluation.answer_comparator import AnswerComparator
from evaluation.evaluation_cache import (
    evaluate_all_trajectories,
    load_evaluation_cache,
)


def check_evaluation_completed(result_dir: str) -> bool:
    """Check if evaluation summary already exists."""
    summary_file = os.path.join(result_dir, "answer_evaluation_summary.json")
    return os.path.exists(summary_file)


def ensure_evaluation_cache(result_dir: str, comparator, args) -> bool:
    """
    Ensure evaluation cache exists. Generate it if not present.

    Behavior is unchanged from the previous monolithic script.
    """
    from centroid.evaluator import detect_task_type
    from evaluation.answer_extractor import get_task_answer_extractor
    from evaluation.evaluation_cache import create_lightweight_answer_cache

    cache = load_evaluation_cache(result_dir)
    if cache and not args.force:
        trajs = cache.get("trajectories", {})
        if trajs:
            print(f"  Using existing evaluation cache ({len(trajs)} trajectories)")
            return True

    entropy_file = os.path.join(result_dir, "entropy_results.json")
    if not os.path.exists(entropy_file):
        print("  Warning: entropy_results.json not found, cannot generate cache")
        return False

    file_size_gb = os.path.getsize(entropy_file) / (1024**3)

    def _amo_api_configured() -> bool:
        return any(
            os.environ.get(key)
            for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "AMO_BENCH_API_KEY")
        )

    def _build_unified_evaluator(task_type: str):
        if task_type != "amo_bench":
            return None
        try:
            from centroid.evaluator import UnifiedEvaluator

            return UnifiedEvaluator(
                timeout=args.timeout,
                skip_amo_description=not _amo_api_configured(),
            )
        except Exception:
            return None

    if file_size_gb > 1.0:
        print(f"  entropy_results.json is {file_size_gb:.1f}GB, using streaming approach...")
        try:
            import ijson

            with open(entropy_file, "rb") as f:
                parser = ijson.items(f, "item")
                first_sample = next(parser)
                task_type = detect_task_type(first_sample)
            evaluator = _build_unified_evaluator(task_type)
            extract_answer_fn = get_task_answer_extractor(task_type)

            trajectories = create_lightweight_answer_cache(
                result_dir=result_dir,
                comparator=comparator,
                extract_answer_fn=extract_answer_fn,
                detect_task_type_fn=detect_task_type,
                evaluator=evaluator,
            )

            if trajectories:
                return True
        except ImportError:
            print("  Warning: ijson not installed, will load full file (slow)")
            print("  Install with: pip install ijson")
        except Exception as e:
            print(f"  Warning: Streaming failed ({e}), falling back to full load")

    print("  Generating evaluation cache (loading full file)...")
    with open(entropy_file, "r") as f:
        entropy_results = json.load(f)

    if not entropy_results:
        print("  Warning: No trajectories in entropy_results.json")
        return False

    task_type = detect_task_type(entropy_results[0])
    evaluator = _build_unified_evaluator(task_type)
    extract_answer_fn = get_task_answer_extractor(task_type)

    trajectories = evaluate_all_trajectories(
        entropy_results=entropy_results,
        comparator=comparator,
        extract_answer_fn=extract_answer_fn,
        detect_task_type_fn=detect_task_type,
        result_dir=result_dir,
        use_cache=False,
        evaluator=evaluator,
    )
    return len(trajectories) > 0


def compute_pass_at_k_from_cache(result_dir: str, k: int) -> dict:
    """Compute pass@k from cache using unbiased estimator."""
    if k < 1:
        return {}

    cache = load_evaluation_cache(result_dir)
    if not cache:
        return {}

    trajectories = cache.get("trajectories", {})
    if not trajectories:
        return {}

    grouped = defaultdict(list)
    for traj_id, info in trajectories.items():
        if not isinstance(info, dict):
            continue
        original_id = info.get("original_id")
        if original_id is None:
            original_id = traj_id.split("_traj_")[0] if "_traj_" in traj_id else traj_id
        grouped[str(original_id)].append(bool(info.get("is_correct", False)))

    if not grouped:
        return {}

    per_problem = []
    ns_used = []
    for values in grouped.values():
        n = len(values)
        if n <= k:
            continue
        c = sum(1 for v in values if v)
        if n - c < k:
            p = 1.0
        else:
            p = 1.0 - (math.comb(n - c, k) / math.comb(n, k))
        per_problem.append(p)
        ns_used.append(n)

    if not per_problem:
        return {}

    mean_n = sum(ns_used) / len(ns_used)
    return {
        "pass_k": k,
        "pass_at_k_accuracy": sum(per_problem) / len(per_problem),
        "pass_at_k_num_problems": len(per_problem),
        "pass_at_k_mean_trajectories": mean_n,
    }


def get_available_pass_k_values(result_dir: str) -> List[int]:
    """Return available k values based on trajectory count distribution."""
    cache = load_evaluation_cache(result_dir)
    if not cache:
        return []
    trajectories = cache.get("trajectories", {})
    if not trajectories:
        return []

    grouped = defaultdict(int)
    for traj_id, info in trajectories.items():
        if not isinstance(info, dict):
            continue
        original_id = info.get("original_id")
        if original_id is None:
            original_id = traj_id.split("_traj_")[0] if "_traj_" in traj_id else traj_id
        grouped[str(original_id)] += 1

    if not grouped:
        return []
    max_n = max(grouped.values())
    if max_n <= 1:
        return []
    return list(range(1, max_n))


def parse_pass_k_values(pass_k_arg: str, result_dir: str) -> List[int]:
    """Parse --pass_k argument values."""
    value = pass_k_arg.strip().lower()
    if value == "all":
        return get_available_pass_k_values(result_dir)

    values = []
    for part in pass_k_arg.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k >= 1:
            values.append(k)
    return sorted(set(values))


def save_pass_at_k_results(result_dir: str, output_dir: str, pass_k_stats: dict):
    """Save pass@k outputs with existing naming convention."""
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "selection": "pass_at_k",
        "pass_k": pass_k_stats.get("pass_k"),
        "overall_accuracy": pass_k_stats.get("pass_at_k_accuracy", 0.0),
        "pass_at_k_accuracy": pass_k_stats.get("pass_at_k_accuracy", 0.0),
        "pass_at_k_num_problems": pass_k_stats.get("pass_at_k_num_problems", 0),
        "pass_at_k_mean_trajectories": pass_k_stats.get("pass_at_k_mean_trajectories", 0.0),
    }

    json_path = os.path.join(output_dir, "answer_evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved aggregate statistics: {json_path}")

    details_path = os.path.join(output_dir, "trajectory_selections.json")
    with open(details_path, "w") as f:
        json.dump(
            {
                "selection": "pass_at_k",
                "result_dir": result_dir,
                "pass_k": pass_k_stats.get("pass_k"),
                "notes": "pass@k computed from evaluation cache trajectories using unbiased estimator",
            },
            f,
            indent=2,
        )
    print(f"  ✓ Saved trajectory selections: {details_path}")

    report_path = os.path.join(output_dir, "answer_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PASS@K EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Selection: pass_at_k\n")
        f.write(f"k: {pass_k_stats.get('pass_k')}\n")
        f.write(f"Accuracy (pass@k): {pass_k_stats.get('pass_at_k_accuracy', 0.0) * 100:.2f}%\n")
        f.write(f"Problems Used: {pass_k_stats.get('pass_at_k_num_problems', 0)}\n")
        f.write(f"Mean n (used problems): {pass_k_stats.get('pass_at_k_mean_trajectories', 0.0):.2f}\n")
        f.write("=" * 80 + "\n")
    print(f"  ✓ Saved evaluation report: {report_path}")


def build_default_comparator(timeout: int) -> AnswerComparator:
    """Factory kept for runner compatibility."""
    return AnswerComparator(timeout=timeout)

