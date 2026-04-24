#!/usr/bin/env python3
"""
Unified evaluation entrypoint for all datasets.

This script unifies the evaluation phase:
1) prepare/ensure evaluation cache
2) prepare/ensure centroid cache
3) run lowest-centroid final evaluation
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluation.pipeline.orchestrator import run_unified_lowest_centroid_pipeline


def _find_result_dirs(pattern: str):
    matched = sorted(glob.glob(pattern))
    result_dirs = []
    for item in matched:
        if not os.path.isdir(item):
            continue
        if os.path.exists(os.path.join(item, "entropy_results.json")) or os.path.exists(
            os.path.join(item, "entropy_results.jsonl")
        ):
            result_dirs.append(item)
            continue
        for sub in sorted(os.listdir(item)):
            sub_path = os.path.join(item, sub)
            if not os.path.isdir(sub_path):
                continue
            if os.path.exists(os.path.join(sub_path, "entropy_results.json")) or os.path.exists(
                os.path.join(sub_path, "entropy_results.jsonl")
            ):
                result_dirs.append(sub_path)
    return sorted(set(result_dirs))


def _run_one(result_dir: str, args):
    summary = run_unified_lowest_centroid_pipeline(
        result_dir=result_dir,
        ensure_eval_cache=args.ensure_eval_cache,
        timeout=args.timeout,
        force_eval_cache=args.force_eval_cache,
        top_percent=args.centroid_top_percent,
        bottom_percent=args.centroid_bottom_percent,
        consecutive_low_threshold=args.centroid_consecutive_low,
        centroid_method=args.centroid_method,
        output_subdir=args.output_subdir,
    )
    print(
        f"[OK] {os.path.basename(result_dir)} | "
        f"acc={summary['overall_accuracy']*100:.2f}% "
        f"({summary['correct_count']}/{summary['total_samples']})"
    )


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for all datasets")
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--pattern", type=str, default="outputs/results/*")

    parser.add_argument("--ensure_eval_cache", action="store_true")
    parser.add_argument("--force_eval_cache", action="store_true")
    parser.add_argument("--timeout", type=int, default=5)

    parser.add_argument("--centroid_top_percent", type=float, default=5.0)
    parser.add_argument("--centroid_bottom_percent", type=float, default=50.0)
    parser.add_argument("--centroid_consecutive_low", type=int, default=3)
    parser.add_argument("--centroid_method", type=str, default="hep", choices=["hep", "raw_entropy"])
    parser.add_argument("--output_subdir", type=str, default="eval_lowest_centroid")

    args = parser.parse_args()

    if args.batch:
        result_dirs = _find_result_dirs(args.pattern)
        if not result_dirs:
            print("No valid result directories found.")
            return 1
        print(f"Found {len(result_dirs)} directories.")
        failed = 0
        for rd in result_dirs:
            try:
                _run_one(rd, args)
            except Exception as e:
                failed += 1
                print(f"[FAIL] {rd}: {e}")
        return 1 if failed else 0

    if not args.result_dir:
        parser.print_help()
        print("\nError: use --result_dir or --batch")
        return 1

    _run_one(args.result_dir, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

