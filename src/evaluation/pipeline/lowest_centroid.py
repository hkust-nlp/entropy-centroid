"""
Unified lowest-centroid selection and accuracy evaluation.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cache_paths import find_existing_cache


def _load_evaluation_map(result_dir: str) -> Dict[str, Dict]:
    """Load trajectory correctness map from evaluation cache."""
    from evaluation.evaluation_cache import load_evaluation_cache

    cache = load_evaluation_cache(result_dir)
    if not cache:
        return {}

    if "trajectories" in cache and isinstance(cache["trajectories"], dict):
        return cache["trajectories"]
    return {}


def _find_centroid_cache(
    result_dir: str,
    top_percent: float,
    bottom_percent: float,
    consecutive_low_threshold: int,
    centroid_method: str,
) -> Optional[str]:
    """Resolve centroid cache file by exact naming first, then fallback."""
    from centroid.io import get_trajectory_centroid_cache_path

    exact = get_trajectory_centroid_cache_path(
        result_dir,
        top_percent=top_percent,
        bottom_percent=bottom_percent,
        consecutive_low_threshold=consecutive_low_threshold,
        centroid_method=centroid_method,
    )
    if os.path.exists(exact):
        return exact

    pattern = os.path.join(result_dir, "trajectory_centroid_cache_*.json")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def _load_centroid_map(cache_path: str) -> Dict[str, Dict]:
    """Load trajectory centroid payload from cache file."""
    with open(cache_path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "trajectories" in payload:
        return payload.get("trajectories", {})
    return payload if isinstance(payload, dict) else {}


def _group_by_problem(
    centroid_map: Dict[str, Dict],
    evaluation_map: Dict[str, Dict],
) -> Dict[str, List[Dict]]:
    """Group trajectories by original problem id."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)

    for traj_id, info in centroid_map.items():
        centroid = info.get("centroid")
        if centroid is None:
            continue

        original_id = info.get("original_id")
        if not original_id:
            original_id = traj_id.split("_traj_")[0] if "_traj_" in traj_id else traj_id

        eval_info = evaluation_map.get(traj_id, {})
        is_correct = info.get("is_correct")
        if is_correct is None:
            is_correct = bool(eval_info.get("is_correct", False))

        grouped[str(original_id)].append(
            {
                "id": traj_id,
                "centroid": float(centroid),
                "is_correct": bool(is_correct),
            }
        )

    return dict(grouped)


def _select_lowest_centroid_with_filter(
    trajectories: List[Dict],
    outlier_threshold: float,
) -> Tuple[Dict, Dict]:
    """Select final trajectory."""
    ordered = sorted(trajectories, key=lambda x: x["centroid"])
    if len(ordered) == 1:
        return ordered[0], {"mean_centroid": ordered[0]["centroid"], "used_filter": False}

    centroids = [t["centroid"] for t in ordered]
    mean_centroid = float(np.mean(centroids))

    filtered = [
        t for t in ordered if (mean_centroid - t["centroid"]) <= outlier_threshold
    ]
    if not filtered:
        filtered = ordered

    return filtered[0], {"mean_centroid": mean_centroid, "used_filter": len(filtered) != len(ordered)}


def _candidate_outlier_thresholds() -> List[float]:
    """Candidate thresholds for built-in filtering search."""
    return [round(i * 0.01, 3) for i in range(1, 31)]


def _select_with_threshold_search(
    grouped: Dict[str, List[Dict]],
) -> Tuple[Dict[str, Dict], Dict]:
    """
    Search candidate outlier thresholds and return best selections.

    Selection score: highest accuracy. Tie-breaker: smaller threshold.
    """
    thresholds = _candidate_outlier_thresholds()
    if not grouped:
        return {}, {"best_threshold": None, "best_accuracy": 0.0, "threshold_results": []}

    threshold_results = []
    best_threshold = None
    best_accuracy = -1.0
    best_selections: Dict[str, Dict] = {}

    for threshold in thresholds:
        selections: Dict[str, Dict] = {}
        correct = 0
        total = 0
        filtered_count = 0

        for original_id, trajectories in grouped.items():
            if not trajectories:
                continue
            selected, meta = _select_lowest_centroid_with_filter(trajectories, threshold)
            total += 1
            is_correct = bool(selected["is_correct"])
            if is_correct:
                correct += 1
            if meta.get("used_filter"):
                filtered_count += 1
            selections[original_id] = {
                "selected_id": selected["id"],
                "selected_centroid": selected["centroid"],
                "is_correct": is_correct,
                "num_trajectories": len(trajectories),
            }

        accuracy = (correct / total) if total else 0.0
        threshold_results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "correct_count": correct,
                "total_samples": total,
                "filtered_problems": filtered_count,
            }
        )

        if accuracy > best_accuracy or (
            accuracy == best_accuracy and (best_threshold is None or threshold < best_threshold)
        ):
            best_accuracy = accuracy
            best_threshold = threshold
            best_selections = selections

    return best_selections, {
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy if best_accuracy >= 0 else 0.0,
        "threshold_results": threshold_results,
    }


def run_lowest_centroid_evaluation(
    result_dir: str,
    top_percent: float = 5.0,
    bottom_percent: float = 50.0,
    consecutive_low_threshold: int = 3,
    centroid_method: str = "hep",
    output_subdir: str = "eval_lowest_centroid",
) -> Dict:
    """
    Evaluate accuracy using lowest-centroid selection.

    Returns a concise summary and writes outputs to `<result_dir>/<output_subdir>`.
    """
    eval_cache_path = find_existing_cache(result_dir)
    if not eval_cache_path:
        raise FileNotFoundError("No evaluation cache found. Expected evaluation_cache.json.")

    centroid_cache_path = _find_centroid_cache(
        result_dir=result_dir,
        top_percent=top_percent,
        bottom_percent=bottom_percent,
        consecutive_low_threshold=consecutive_low_threshold,
        centroid_method=centroid_method,
    )
    if not centroid_cache_path:
        raise FileNotFoundError("No trajectory_centroid_cache_*.json found.")

    evaluation_map = _load_evaluation_map(result_dir)
    centroid_map = _load_centroid_map(centroid_cache_path)
    grouped = _group_by_problem(centroid_map, evaluation_map)

    selections, threshold_search = _select_with_threshold_search(grouped)
    total = len(selections)
    correct = sum(1 for item in selections.values() if item["is_correct"])

    summary = {
        "selection": "lowest_centroid",
        "overall_accuracy": (correct / total) if total else 0.0,
        "correct_count": correct,
        "total_samples": total,
        "incorrect_count": total - correct,
        "parameters": {
            "top_percent": top_percent,
            "bottom_percent": bottom_percent,
            "consecutive_low_threshold": consecutive_low_threshold,
            "centroid_method": centroid_method,
        },
        "filtering": {
            "mode": "auto_threshold_search",
            "best_accuracy": threshold_search["best_accuracy"],
        },
        "inputs": {
            "evaluation_cache": os.path.basename(eval_cache_path),
            "centroid_cache": os.path.basename(centroid_cache_path),
        },
    }

    output_dir = os.path.join(result_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "answer_evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "trajectory_selections.json"), "w") as f:
        json.dump(selections, f, indent=2)

    return summary

