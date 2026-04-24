"""Unified dataset runner: prepare -> centroid -> lowest-centroid."""

from __future__ import annotations

import os
from typing import Dict, List

from .lowest_centroid import run_lowest_centroid_evaluation
from .orchestrator import ensure_centroid_cache


def _safe_cleanup(paths: List[str]):
    for path in paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass


def run_pipeline_with_adapter(adapter, args) -> Dict[str, Dict]:
    """
    Run unified pipeline for all discovered dirs.

    Unified skeleton:
    1) Discover result dirs
    2) Dataset-specific prepare_inputs()
    3) Ensure centroid cache
    4) Run lowest-centroid evaluation
    """
    result_dirs = adapter.discover_result_dirs(args)
    if not result_dirs:
        raise RuntimeError("No valid result directories found.")

    summaries: Dict[str, Dict] = {}
    for idx, result_dir in enumerate(sorted(result_dirs), 1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(result_dirs)}] {adapter.name}: {result_dir}")
        print("=" * 80)

        prepared = adapter.prepare_inputs(result_dir, args)

        ensure_centroid_cache(
            result_dir=result_dir,
            top_percent=args.centroid_top_percent,
            bottom_percent=args.centroid_bottom_percent,
            consecutive_low_threshold=args.centroid_consecutive_low,
            centroid_method=args.centroid_method,
        )

        summary = run_lowest_centroid_evaluation(
            result_dir=result_dir,
            top_percent=args.centroid_top_percent,
            bottom_percent=args.centroid_bottom_percent,
            consecutive_low_threshold=args.centroid_consecutive_low,
            centroid_method=args.centroid_method,
            output_subdir="eval_lowest_centroid",
        )
        summaries[result_dir] = summary

        if not getattr(args, "keep_intermediate", False):
            _safe_cleanup(prepared.intermediate_files)

    return summaries

