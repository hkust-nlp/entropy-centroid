#!/usr/bin/env python3
"""Offline answer evaluation entrypoint (thin CLI wrapper)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluation.apps.runner import batch_evaluate, evaluate_single_directory


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate answer accuracy from inference results"
    )

    # Input options
    parser.add_argument(
        '--result_dir',
        type=str,
        help='Path to result directory with entropy_results.json'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process multiple result directories'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='outputs/results/config_*',
        help='Glob pattern for batch processing (default: outputs/results/config_*)'
    )

    # Evaluation options
    parser.add_argument(
        '--strategy',
        type=str,
        default='entropy_mean',
        choices=['entropy_mean', 'early_high_entropy_centroid', 'entropy_min'],
        help='Trajectory scoring strategy for best_of_n selection (default: entropy_mean)'
    )
    parser.add_argument(
        '--selection',
        type=str,
        default='best_of_n',
        choices=['best_of_n', 'majority_voting', 'llm_majority_voting', 'entropy_centroid', 'centroid_voting', 'centroid_weighted_voting', 'lowest_centroid', 'random', 'pass_at_k'],
        help='Answer selection method: best_of_n (trajectory scoring), majority_voting (equal votes), llm_majority_voting (LLM-filtered votes for AMO description), entropy_centroid (earliest centroid), centroid_voting (vote among bottom X%% centroid trajectories), centroid_weighted_voting (weighted votes by centroid position), lowest_centroid (unified cache-based pipeline with built-in filtering search), random (random baseline), pass_at_k (compute pass@k from full trajectory cache; use --pass_k) (default: best_of_n)'
    )
    parser.add_argument(
        '--pass_k',
        type=str,
        default='1',
        help='For selection=pass_at_k: k values. Supports single "1", comma-separated "1,2,4", or "all" (all k with k < n from cache)'
    )
    parser.add_argument(
        '--entropy_percentile',
        type=float,
        default=30.0,
        help='Entropy percentile for early_high_entropy_centroid strategy (default: 30.0)'
    )
    # Entropy centroid parameters (support comma-separated values for batch evaluation)
    parser.add_argument(
        '--centroid_top_percent',
        type=str,
        default='1.0',
        help='Top percent of entropy tokens to start HEP. Supports comma-separated values for batch: "1,3,5" (default: 5.0)'
    )
    parser.add_argument(
        '--centroid_bottom_percent',
        type=str,
        default='80.0',
        help='Bottom percent of entropy tokens to end HEP. Supports comma-separated values for batch: "30,50,80" (default: 50.0)'
    )
    parser.add_argument(
        '--centroid_consecutive_low',
        type=str,
        default='2',
        help='Consecutive low-entropy tokens to end HEP. Supports comma-separated values for batch: "2,3,5" (default: 3)'
    )
    parser.add_argument(
        '--centroid_outlier_threshold',
        type=str,
        default='0.1',
        help='Outlier threshold for entropy_centroid. Supports comma-separated values for batch: "0.05,0.1,0.2" (default: 0.1)'
    )
    parser.add_argument(
        '--centroid_select_percent',
        type=str,
        default='30.0',
        help='For centroid_voting: select trajectories with centroid in bottom X%%. Supports comma-separated values for batch: "10,20,30,50,70" (default: 30.0)'
    )
    parser.add_argument(
        '--centroid_method',
        type=str,
        default='hep',
        choices=['hep', 'raw_entropy'],
        help='Centroid calculation method: hep (High-Entropy Phase based, requires top/bottom/cons params), raw_entropy (direct entropy weighted, simpler) (default: hep)'
    )
    # Centroid weighted voting parameters
    parser.add_argument(
        '--weight_method',
        type=str,
        default='linear',
        choices=['linear', 'inverse', 'exponential'],
        help='For centroid_weighted_voting: weight calculation method. linear (1-normalized_centroid), inverse (1/centroid), exponential (exp(-centroid*temp)) (default: linear)'
    )
    parser.add_argument(
        '--outlier_method',
        type=str,
        default='std',
        choices=['none', 'threshold', 'std'],
        help='For centroid_weighted_voting: outlier detection method. none (no filtering), threshold (|centroid-mean|>threshold), std (outside mean±N*std) (default: std)'
    )
    parser.add_argument(
        '--outlier_num_std',
        type=str,
        default='1.0',
        help='For centroid_weighted_voting with outlier_method=std: number of standard deviations. Supports comma-separated values for batch: "0.5,1.0,1.5,2.0" (default: 1.0)'
    )
    parser.add_argument(
        '--outlier_penalty',
        type=str,
        default='-0.5',
        help='For centroid_weighted_voting: weight penalty for outliers (negative=penalty). Supports comma-separated values for batch: "-1.0,-0.5,0" (default: -0.5)'
    )
    parser.add_argument(
        '--exp_temperature',
        type=float,
        default=1.0,
        help='For centroid_weighted_voting with weight_method=exponential: temperature parameter T in exp(-centroid*T). Higher T = steeper decay, more emphasis on low centroids (default: 1.0)'
    )

    # Random selection parameters
    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help='Random seed for random selection method (default: None, uses random seed)'
    )
    parser.add_argument(
        '--random_num_samples',
        type=int,
        default=1,
        help='Number of random samples for random selection (default: 1). Higher values provide metadata for accuracy estimation.'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=5,
        help='Timeout in seconds for symbolic comparison (default: 5)'
    )
    parser.add_argument(
        '--ensure_eval_cache',
        dest='ensure_eval_cache',
        action='store_true',
        default=True,
        help='For lowest_centroid unified pipeline: ensure evaluation_cache.json exists before evaluation (default: enabled)'
    )
    parser.add_argument(
        '--no_ensure_eval_cache',
        dest='ensure_eval_cache',
        action='store_false',
        help='For lowest_centroid unified pipeline: disable automatic evaluation cache generation'
    )

    # Task type options
    parser.add_argument(
        '--task_type',
        type=str,
        default='auto',
        choices=['auto', 'math', 'korbench', 'synlogic'],
        help='Task type for evaluation: auto (detect from ID), math, korbench, synlogic (default: auto)'
    )

    # Output options
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recompute even if results exist'
    )
    parser.add_argument(
        '--no_viz',
        action='store_true',
        help='Skip visualization creation'
    )

    args = parser.parse_args()

    # Validate input
    if args.batch:
        batch_evaluate(args.pattern, args)
    elif args.result_dir:
        if not os.path.exists(args.result_dir):
            print(f"Error: Result directory not found: {args.result_dir}")
            return 1
        evaluate_single_directory(args.result_dir, args)
    else:
        parser.print_help()
        print("\nError: Must specify either --result_dir or --batch")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
