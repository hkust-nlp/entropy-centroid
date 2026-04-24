#!/usr/bin/env python3
"""Unified BigCodeBench CLI (thin entrypoint)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluation.datasets.adapters import BigCodeBenchAdapter
from evaluation.pipeline.unified_runner import run_pipeline_with_adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Unified BigCodeBench evaluation")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--result_dir", type=str, help="Single result directory")
    target.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument(
        "--pattern",
        type=str,
        default="outputs/results/bigcodebench_*",
        help="Batch glob pattern",
    )

    # Dataset-specific prepare options (preserved behavior)
    parser.add_argument("--split", type=str, default="instruct", choices=["instruct", "complete"])
    parser.add_argument("--subset", type=str, default="full", choices=["full", "hard"])
    parser.add_argument("--execution", type=str, default="local", choices=["local", "gradio", "e2b"])
    parser.add_argument("--pass_k", type=str, default="1")
    parser.add_argument("--calibrated", action="store_true", default=True)
    parser.add_argument("--no_calibrated", action="store_true")
    parser.add_argument("--parallel", type=int, default=-1)
    parser.add_argument("--min_time_limit", type=float, default=1.0)
    parser.add_argument("--no_gt", action="store_true")
    parser.add_argument("--select_strategy", type=str, default="all", choices=["all", "first", "index"])
    parser.add_argument("--trajectory_index", type=int, default=0)

    # Unified centroid/eval options
    parser.add_argument("--centroid_top_percent", type=float, default=5.0)
    parser.add_argument("--centroid_bottom_percent", type=float, default=50.0)
    parser.add_argument("--centroid_consecutive_low", type=int, default=3)
    parser.add_argument("--centroid_method", type=str, default="hep", choices=["hep", "raw_entropy"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep_intermediate", action="store_true")

    args = parser.parse_args()
    if args.no_calibrated:
        args.calibrated = False
    return args


def main():
    args = parse_args()
    adapter = BigCodeBenchAdapter()
    summaries = run_pipeline_with_adapter(adapter, args)
    print(f"\nCompleted {len(summaries)} directory(ies).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
