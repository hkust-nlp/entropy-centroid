#!/usr/bin/env python3
"""Unified tau2-bench CLI (thin entrypoint)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluation.datasets.adapters import Tau2Adapter
from evaluation.pipeline.unified_runner import run_pipeline_with_adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Unified tau2-bench evaluation")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--result_dir", type=str, help="Single result directory")
    target.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--batch_base_dir", type=str, default="outputs/tau2_bench")

    parser.add_argument("--filter_llm_error", action="store_true")

    # Unified centroid/eval options
    parser.add_argument("--centroid_top_percent", type=float, default=5.0)
    parser.add_argument("--centroid_bottom_percent", type=float, default=50.0)
    parser.add_argument("--centroid_consecutive_low", type=int, default=3)
    parser.add_argument("--centroid_method", type=str, default="hep", choices=["hep", "raw_entropy"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep_intermediate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter = Tau2Adapter()
    summaries = run_pipeline_with_adapter(adapter, args)
    print(f"\nCompleted {len(summaries)} directory(ies).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
