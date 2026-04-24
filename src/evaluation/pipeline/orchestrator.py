"""
Unified evaluation orchestrator.

Pipeline:
1) Ensure `evaluation_cache.json` exists
2) Ensure trajectory centroid cache exists
3) Run lowest-centroid final evaluation
"""

from __future__ import annotations

import json
import os
from typing import Dict

from .cache_paths import find_existing_cache
from .lowest_centroid import run_lowest_centroid_evaluation

def ensure_evaluation_cache_for_math_logic(
    result_dir: str,
    timeout: int = 5,
    force: bool = False,
) -> bool:
    """
    Ensure evaluation cache exists for math/logic-style datasets.

    This reuses existing evaluation logic from `evaluation.evaluation_cache`
    without changing answer comparison behavior.
    """
    from evaluation.answer_comparator import AnswerComparator
    from evaluation.answer_extractor import get_task_answer_extractor
    from evaluation.evaluation_cache import evaluate_all_trajectories, load_evaluation_cache

    try:
        from centroid.evaluator import UnifiedEvaluator, detect_task_type
    except Exception:
        UnifiedEvaluator = None

        def detect_task_type(sample):  # type: ignore
            _ = sample
            return "math"

    existing = load_evaluation_cache(result_dir)
    if existing and not force:
        return True

    entropy_file = os.path.join(result_dir, "entropy_results.json")
    if not os.path.exists(entropy_file):
        return False

    with open(entropy_file, "r") as f:
        entropy_results = json.load(f)
    if not entropy_results:
        return False

    task_type = detect_task_type(entropy_results[0])
    evaluator = None
    if UnifiedEvaluator is not None and task_type in ("korbench", "synlogic", "amo_bench"):
        evaluator = UnifiedEvaluator(timeout=timeout)

    comparator = AnswerComparator(timeout=timeout)
    extract_answer_fn = get_task_answer_extractor(task_type)

    trajectories = evaluate_all_trajectories(
        entropy_results=entropy_results,
        comparator=comparator,
        extract_answer_fn=extract_answer_fn,
        detect_task_type_fn=detect_task_type,
        result_dir=result_dir,
        use_cache=not force,
        evaluator=evaluator,
    )
    return len(trajectories) > 0


def ensure_centroid_cache(
    result_dir: str,
    top_percent: float = 5.0,
    bottom_percent: float = 50.0,
    consecutive_low_threshold: int = 3,
    centroid_method: str = "hep",
) -> bool:
    """Ensure trajectory centroid cache exists; compute if absent."""
    from centroid.io import (
        get_trajectory_centroid_cache_path,
        load_trajectory_centroid_cache,
        stream_compute_centroid_cache,
    )

    cache_path = get_trajectory_centroid_cache_path(
        result_dir=result_dir,
        top_percent=top_percent,
        bottom_percent=bottom_percent,
        consecutive_low_threshold=consecutive_low_threshold,
        centroid_method=centroid_method,
    )
    if os.path.exists(cache_path):
        existing = load_trajectory_centroid_cache(
            result_dir=result_dir,
            top_percent=top_percent,
            bottom_percent=bottom_percent,
            consecutive_low_threshold=consecutive_low_threshold,
            centroid_method=centroid_method,
        )
        if existing:
            return True

    computed = stream_compute_centroid_cache(
        result_dir=result_dir,
        top_percent=top_percent,
        bottom_percent=bottom_percent,
        consecutive_low_threshold=consecutive_low_threshold,
        centroid_method=centroid_method,
    )
    return bool(computed)


def run_unified_lowest_centroid_pipeline(
    result_dir: str,
    ensure_eval_cache: bool = False,
    timeout: int = 5,
    force_eval_cache: bool = False,
    top_percent: float = 5.0,
    bottom_percent: float = 50.0,
    consecutive_low_threshold: int = 3,
    centroid_method: str = "hep",
    output_subdir: str = "eval_lowest_centroid",
) -> Dict:
    """
    End-to-end unified evaluation pipeline for one result directory.
    """
    if ensure_eval_cache:
        ok = ensure_evaluation_cache_for_math_logic(
            result_dir=result_dir,
            timeout=timeout,
            force=force_eval_cache,
        )
        if not ok:
            raise RuntimeError("Failed to create evaluation cache for result directory.")

    if not find_existing_cache(result_dir):
        raise FileNotFoundError("Missing evaluation cache for unified evaluation.")

    ok_centroid = ensure_centroid_cache(
        result_dir=result_dir,
        top_percent=top_percent,
        bottom_percent=bottom_percent,
        consecutive_low_threshold=consecutive_low_threshold,
        centroid_method=centroid_method,
    )
    if not ok_centroid:
        raise RuntimeError("Failed to prepare centroid cache.")

    return run_lowest_centroid_evaluation(
        result_dir=result_dir,
        top_percent=top_percent,
        bottom_percent=bottom_percent,
        consecutive_low_threshold=consecutive_low_threshold,
        centroid_method=centroid_method,
        output_subdir=output_subdir,
    )

