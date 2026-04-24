#!/usr/bin/env python3
"""Tau2 dataset preparation helpers for unified runner."""

import json
import os
from collections import defaultdict
from typing import Dict, List

from evaluation.pipeline.cache_paths import canonical_cache_path


def load_jsonl_deduplicated(jsonl_path: str) -> List[Dict]:
    """
    Load JSONL file with deduplication (keep last occurrence per id).

    Checkpoint/resume can produce duplicate entries; we keep the last one
    for each trajectory id.
    """
    by_id = {}
    total_lines = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            record = json.loads(line)
            traj_id = record.get("id")
            if traj_id is None:
                continue
            by_id[traj_id] = record

    n_dedup = total_lines - len(by_id)
    if n_dedup > 0:
        print(
            f"  Dedup: {total_lines} lines -> {len(by_id)} unique trajectories "
            f"({n_dedup} duplicates removed)"
        )
    else:
        print(f"  Loaded {len(by_id)} trajectories")
    return list(by_id.values())


def step_cache(
    result_dir: str,
    trajectories: List[Dict],
    force: bool = False,
    filter_llm_error: bool = False,
) -> Dict:
    """
    Generate evaluation_cache.json from trajectory data.

    Determines correctness from reward > 0, stores per-trajectory metadata.
    """
    cache_path = canonical_cache_path(result_dir)

    if os.path.exists(cache_path) and not force:
        print(f"  Loading existing cache: {cache_path}")
        with open(cache_path, "r") as f:
            v2_cache = json.load(f)
        if v2_cache.get("version") == 2 and "trajectories" in v2_cache:
            cache_trajs = v2_cache["trajectories"]
            loaded_ids = {t["id"] for t in trajectories}
            cached_ids = set(cache_trajs.keys())
            missing = loaded_ids - cached_ids
            if not missing:
                print(f"  Cache covers all {len(loaded_ids)} trajectories")
                _print_cache_statistics(v2_cache, filter_llm_error)
                return {tid: info.get("is_correct", False) for tid, info in cache_trajs.items()}
            print(f"  Cache missing {len(missing)} trajectories, rebuilding...")

    print("  Building evaluation cache...")
    cache_trajectories = {}
    domain_stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})
    termination_stats = defaultdict(int)
    skipped_llm_error = 0

    for traj in trajectories:
        traj_id = traj["id"]
        original_id = traj.get("original_id", traj_id)
        reward = traj.get("reward", 0.0)
        termination_reason = traj.get("termination_reason", "unknown")
        source = traj.get("source", "")

        domain = source.replace("tau2-bench-", "") if source.startswith("tau2-bench-") else source
        if reward is None:
            reward = 0.0
        is_correct = float(reward) > 0
        termination_stats[termination_reason] += 1

        if filter_llm_error and termination_reason == "llm_error":
            skipped_llm_error += 1
            continue

        cache_trajectories[traj_id] = {
            "is_correct": is_correct,
            "original_id": original_id,
            "reward": reward,
            "termination_reason": termination_reason,
        }
        domain_stats[domain]["total"] += 1
        if is_correct:
            domain_stats[domain]["correct"] += 1
        else:
            domain_stats[domain]["incorrect"] += 1

    v2_cache = {
        "version": 2,
        "task_type": "tau2_bench",
        "trajectories": cache_trajectories,
    }

    temp_path = cache_path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(v2_cache, f, indent=2, default=str)
    os.replace(temp_path, cache_path)
    print(f"  Saved: {cache_path} ({len(cache_trajectories)} trajectories)")
    _print_domain_stats(domain_stats)
    _print_termination_stats(termination_stats, skipped_llm_error)
    return {tid: info["is_correct"] for tid, info in cache_trajectories.items()}


def _print_cache_statistics(v2_cache: Dict, filter_llm_error: bool = False):
    """Print summary statistics from an existing cache."""
    _ = filter_llm_error
    trajs = v2_cache.get("trajectories", {})
    domain_stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})
    termination_stats = defaultdict(int)
    for traj_id, info in trajs.items():
        is_correct = info.get("is_correct", False)
        termination_reason = info.get("termination_reason", "unknown")
        _ = info.get("original_id", traj_id)
        termination_stats[termination_reason] += 1
        domain_stats["all"]["total"] += 1
        if is_correct:
            domain_stats["all"]["correct"] += 1
        else:
            domain_stats["all"]["incorrect"] += 1
    total = domain_stats["all"]["total"]
    correct = domain_stats["all"]["correct"]
    rate = correct / total * 100 if total > 0 else 0
    print(f"  Totals: {correct}/{total} correct ({rate:.1f}%)")
    _print_termination_stats(termination_stats)


def _print_domain_stats(domain_stats: Dict):
    """Print per-domain accuracy statistics."""
    print("\n  Domain statistics:")
    for domain in sorted(domain_stats.keys()):
        s = domain_stats[domain]
        rate = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"    {domain:20s}: {s['correct']:4d}/{s['total']:4d} correct ({rate:5.1f}%)")


def _print_termination_stats(termination_stats: Dict, skipped: int = 0):
    """Print termination reason breakdown."""
    print("\n  Termination reasons:")
    for reason in sorted(termination_stats.keys()):
        count = termination_stats[reason]
        print(f"    {reason:20s}: {count:4d}")
    if skipped > 0:
        print(f"    {'(filtered out)':20s}: {skipped:4d} llm_error trajectories")

