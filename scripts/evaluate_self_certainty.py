#!/usr/bin/env python3
"""
Evaluate Self-Certainty selection using existing evaluation cache.

This script takes the Self-Certainty confidence scores and uses the pre-computed
evaluation_cache_v2.json (per-trajectory correctness) to measure pass@1 when
selecting the trajectory with the highest Self-Certainty score.

This avoids re-running code execution — it reuses the correctness labels from
the evaluation cache and only applies the Self-Certainty selection strategy.

Pipeline:
    1. convert_to_self_certainty.py   → self_certainty_input.json
    2. Self-Certainty/confidence_list.py → self_certainty_input-confidence-list.json
    3. THIS SCRIPT                    → selection results + pass@1

Usage:
    # Single directory
    python scripts/evaluate_self_certainty.py \
        --confidence_file outputs/results/.../self_certainty_input-confidence-list.json \
        --eval_cache outputs/results/.../evaluation_cache_v2.json \
        --best_N 1,2,4,8,16,32

    # Batch: auto-discover all directories with confidence-list output
    python scripts/evaluate_self_certainty.py \
        --batch --benchmark livecodebench

    # Batch: tau2-bench
    python scripts/evaluate_self_certainty.py \
        --batch --benchmark tau2_bench --results_root outputs/tau2_bench
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# Import directory discovery utilities from convert script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_self_certainty import find_result_dirs, detect_model_path_from_dirname


def load_eval_cache(cache_path: str) -> dict:
    """Load evaluation_cache_v2.json and return per-trajectory correctness."""
    with open(cache_path, 'r') as f:
        cache = json.load(f)

    trajectories = cache.get('trajectories', {})
    print(f"  Loaded evaluation cache: {len(trajectories)} trajectories")
    return trajectories


def load_confidence_data(confidence_path: str) -> list:
    """Load the Self-Certainty confidence list output."""
    ext = os.path.splitext(confidence_path)[1].lower()
    if ext == '.parquet':
        import pandas as pd
        df = pd.read_parquet(confidence_path)
        if df['output'].dtype == object and isinstance(df.iloc[0]['output'], str):
            df['output'] = df['output'].apply(json.loads)
        if df['confidence_list'].dtype == object and isinstance(df.iloc[0]['confidence_list'], str):
            df['confidence_list'] = df['confidence_list'].apply(json.loads)
        return df.to_dict(orient='records')
    else:
        with open(confidence_path, 'r') as f:
            return json.load(f)


def evaluate_self_certainty(
    confidence_path: str,
    eval_cache_path: str,
    best_n_values: list,
    output_file: str = None,
):
    """
    Evaluate Self-Certainty selection using pre-computed correctness labels.

    For each problem:
    1. Get the confidence scores for all N trajectories
    2. Select the trajectory with the highest confidence
    3. Look up whether that trajectory is correct in the eval cache
    4. Report pass@1 for Self-Certainty selection
    """
    # Load data
    print("Loading confidence scores...")
    data = load_confidence_data(confidence_path)
    print(f"  Loaded {len(data)} problems")

    print("Loading evaluation cache...")
    eval_trajectories = load_eval_cache(eval_cache_path)

    # Build correctness lookup: {original_id: {traj_index: is_correct}}
    correctness_by_problem = defaultdict(dict)
    for traj_id, traj_info in eval_trajectories.items():
        original_id = traj_info.get('original_id', '')
        is_correct = traj_info.get('is_correct', False)
        # Extract trajectory index from traj_id:
        #   "lcb_1873_A_traj_5" → 5
        #   "29_traj_44"        → 44
        #   "29"                → 0  (tau2-bench bare key = first trajectory)
        parts = traj_id.rsplit('_traj_', 1)
        if len(parts) == 2:
            try:
                traj_idx = int(parts[1])
                correctness_by_problem[original_id][traj_idx] = is_correct
            except ValueError:
                pass
        elif traj_id == original_id:
            # Bare key without _traj_ suffix: this is traj_idx=0
            correctness_by_problem[original_id][0] = is_correct

    print(f"  Correctness data for {len(correctness_by_problem)} problems")

    # Also compute random baseline (pass@1 with random selection = avg correctness)
    all_correct_counts = []
    all_total_counts = []
    for problem_id, traj_correctness in correctness_by_problem.items():
        n_correct = sum(1 for v in traj_correctness.values() if v)
        n_total = len(traj_correctness)
        all_correct_counts.append(n_correct)
        all_total_counts.append(n_total)

    if all_total_counts:
        random_pass_at_1 = sum(
            c / t for c, t in zip(all_correct_counts, all_total_counts)
        ) / len(all_total_counts)
    else:
        random_pass_at_1 = 0.0

    # Evaluate for each best_N value
    results = {}
    max_n = max(best_n_values)

    print(f"\n{'='*70}")
    print(f"{'Method':<30} {'Pass@1':>10} {'Problems':>10} {'Correct':>10}")
    print(f"{'='*70}")
    print(f"{'Random (baseline)':<30} {random_pass_at_1:>10.4f} "
          f"{len(correctness_by_problem):>10} {'':>10}")

    for best_n in sorted(best_n_values):
        correct_count = 0
        total_count = 0
        skipped = 0

        for item in data:
            question_id = item.get('question_id', '')
            confidence_list = item.get('confidence_list', [])
            outputs = item.get('output', [])

            if question_id not in correctness_by_problem:
                skipped += 1
                continue

            # Take top best_n
            n = min(best_n, len(confidence_list), len(outputs))
            if n == 0:
                skipped += 1
                continue

            confidences = confidence_list[:n]

            # Select trajectory with highest confidence
            best_idx = max(range(len(confidences)), key=lambda i: confidences[i])

            # Look up correctness
            traj_correctness = correctness_by_problem[question_id]
            is_correct = traj_correctness.get(best_idx, False)

            if is_correct:
                correct_count += 1
            total_count += 1

        pass_at_1 = correct_count / total_count if total_count > 0 else 0.0

        results[best_n] = {
            'pass_at_1': pass_at_1,
            'correct': correct_count,
            'total': total_count,
            'skipped': skipped,
        }

        print(f"{'Self-Certainty (N=' + str(best_n) + ')':<30} "
              f"{pass_at_1:>10.4f} {total_count:>10} {correct_count:>10}")

    print(f"{'='*70}")

    # Also compute oracle (best possible selection = any correct trajectory exists)
    oracle_correct = sum(1 for c in all_correct_counts if c > 0)
    oracle_total = len(all_correct_counts)
    oracle_pass_at_1 = oracle_correct / oracle_total if oracle_total > 0 else 0.0
    print(f"{'Oracle (any correct exists)':<30} {oracle_pass_at_1:>10.4f} "
          f"{oracle_total:>10} {oracle_correct:>10}")
    print()

    # Save results
    summary = {
        'confidence_file': confidence_path,
        'eval_cache_file': eval_cache_path,
        'random_pass_at_1': random_pass_at_1,
        'oracle_pass_at_1': oracle_pass_at_1,
        'self_certainty_results': {
            str(k): v for k, v in results.items()
        },
    }

    if output_file is None:
        output_file = os.path.splitext(confidence_path)[0] + '-eval-results.json'

    try:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to: {output_file}")
    except PermissionError:
        print(f"  WARNING: Cannot write {output_file} (permission denied)")

    return summary


def run_batch(args):
    """Auto-discover directories and evaluate all with confidence-list output."""
    results_root = args.results_root
    benchmark = args.benchmark
    best_n_values = [int(x.strip()) for x in args.best_N.split(',')]

    print(f"Batch mode: searching for '{benchmark}' result dirs "
          f"under {results_root}")
    print(f"  (recursive search up to depth {args.max_depth})\n")

    dirs = find_result_dirs(results_root, benchmark, max_depth=args.max_depth)

    if not dirs:
        print(f"No result directories found matching benchmark '{benchmark}'")
        return 1

    # Filter: only dirs that have both confidence-list AND evaluation_cache
    eligible = []
    for result_dir, rel_path in dirs:
        # Find confidence-list file
        conf_json = os.path.join(result_dir, "self_certainty_input-confidence-list.json")
        conf_parquet = os.path.join(result_dir, "self_certainty_input-confidence-list.parquet")
        if os.path.exists(conf_json):
            conf_file = conf_json
        elif os.path.exists(conf_parquet):
            conf_file = conf_parquet
        else:
            continue

        # Find evaluation cache
        eval_cache = os.path.join(result_dir, "evaluation_cache.json")
        if not os.path.exists(eval_cache):
            print(f"  SKIP {rel_path}: no evaluation_cache.json")
            continue

        model_path = detect_model_path_from_dirname(rel_path, benchmark)
        eligible.append((result_dir, rel_path, conf_file, eval_cache, model_path))

    if not eligible:
        print(f"No directories with both confidence-list and evaluation_cache found.")
        print(f"Run Steps 1-2 first.")
        return 1

    print(f"Found {len(eligible)} directories to evaluate:")
    for _, rel_path, _, _, model_path in eligible:
        print(f"  - {rel_path}  (model: {model_path})")
    print()

    # Run evaluation for each directory
    all_summaries = []
    success_count = 0
    fail_count = 0

    for idx, (result_dir, rel_path, conf_file, eval_cache, model_path) in enumerate(eligible, 1):
        print(f"\n{'#' * 80}")
        print(f"# [{idx}/{len(eligible)}] {rel_path}")
        print(f"{'#' * 80}")

        try:
            summary = evaluate_self_certainty(
                confidence_path=conf_file,
                eval_cache_path=eval_cache,
                best_n_values=best_n_values,
                output_file=None,  # auto-generate in result_dir
            )
            summary['rel_path'] = rel_path
            summary['model'] = model_path
            all_summaries.append(summary)
            success_count += 1
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    # Print aggregate summary table
    if all_summaries:
        print(f"\n\n{'=' * 100}")
        print(f"BATCH SUMMARY  ({success_count} evaluated, {fail_count} failed)")
        print(f"{'=' * 100}")

        # Header
        n_cols = sorted(best_n_values)
        header = f"{'Directory':<45} {'Random':>8} {'Oracle':>8}"
        for n in n_cols:
            header += f" {'N=' + str(n):>8}"
        print(header)
        print('-' * 100)

        # Rows
        for s in all_summaries:
            row = f"{s['rel_path']:<45} {s['random_pass_at_1']:>8.4f} {s['oracle_pass_at_1']:>8.4f}"
            for n in n_cols:
                sc_result = s['self_certainty_results'].get(str(n), {})
                p1 = sc_result.get('pass_at_1', 0.0)
                row += f" {p1:>8.4f}"
            print(row)

        print(f"{'=' * 100}")

        # Save batch summary
        batch_output = os.path.join(
            results_root,
            f"self_certainty_eval_summary_{benchmark}.json"
        )
        with open(batch_output, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nBatch summary saved to: {batch_output}")

    return 0 if fail_count == 0 else 1


def run_single(args):
    """Evaluate a single directory."""
    best_n_values = [int(x.strip()) for x in args.best_N.split(',')]
    evaluate_self_certainty(
        confidence_path=args.confidence_file,
        eval_cache_path=args.eval_cache,
        best_n_values=best_n_values,
        output_file=args.output_file,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Self-Certainty selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single directory
  python scripts/evaluate_self_certainty.py \\
      --confidence_file .../self_certainty_input-confidence-list.json \\
      --eval_cache .../evaluation_cache_v2.json \\
      --best_N 1,2,4,8,16,32

  # Batch: auto-discover all livecodebench dirs
  python scripts/evaluate_self_certainty.py \\
      --batch --benchmark livecodebench

  # Batch: tau2-bench
  python scripts/evaluate_self_certainty.py \\
      --batch --benchmark tau2_bench --results_root outputs/tau2_bench

  # Batch: bigcodebench
  python scripts/evaluate_self_certainty.py \\
      --batch --benchmark bigcodebench
        """,
    )

    # --- Mode selection ---
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--batch", action="store_true",
        help="Batch mode: auto-discover result dirs matching --benchmark"
    )
    mode_group.add_argument(
        "--confidence_file", type=str, default=None,
        help="Single-dir mode: path to confidence-list output"
    )

    # --- Batch options ---
    batch_group = parser.add_argument_group("Batch options")
    batch_group.add_argument(
        "--benchmark", type=str, default=None,
        help="Benchmark keyword (e.g., livecodebench, tau2_bench, bigcodebench)"
    )
    batch_group.add_argument(
        "--results_root", type=str, default="outputs/results",
        help="Root directory to search (default: outputs/results)"
    )
    batch_group.add_argument(
        "--max_depth", type=int, default=3,
        help="Max directory depth for recursive search (default: 3)"
    )

    # --- Common options ---
    common_group = parser.add_argument_group("Common options")
    common_group.add_argument(
        "--eval_cache", type=str, default=None,
        help="Path to evaluation_cache_v2.json (single-dir mode only)"
    )
    common_group.add_argument(
        "--best_N", type=str, default="1,2,4,8,16,32,64",
        help="Comma-separated list of N values for Best-of-N selection "
             "(default: 1,2,4,8,16,32,64)"
    )
    common_group.add_argument(
        "--output_file", type=str, default=None,
        help="Output file for evaluation results (single-dir mode only)"
    )
    args = parser.parse_args()

    if args.batch:
        if not args.benchmark:
            parser.error("--benchmark is required in batch mode")
        return run_batch(args)
    elif args.confidence_file:
        if not args.eval_cache:
            parser.error("--eval_cache is required in single-dir mode")
        return run_single(args)
    else:
        parser.error("Must specify either --batch or --confidence_file")


if __name__ == "__main__":
    sys.exit(main())
