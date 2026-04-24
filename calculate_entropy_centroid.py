#!/usr/bin/env python3
"""
Calculate Entropy Centroid

Computes the centroid (center of mass) of high-entropy phases (HEP) in reasoning trajectories.
Supports both math and logic (KOR-Bench, SynLogic) tasks.

Each HEP is treated as a mass point where:
- Position: Start token position of the HEP
- Weight (Mass): Duration of the HEP (number of tokens)

Supported calculation methods:
1. moment: Centroid = Σ(weight × position) / trajectory_length
2. weighted_average: Centroid = [Σ(weight × position) / Σ(weight)] / trajectory_length
3. weighted_average_center: Same as #2 but uses center position of HEP
4. raw_entropy_weighted: Uses raw entropy values instead of HEP phases

Usage:
    python calculate_entropy_centroid.py --result_dir <path>
    python calculate_entropy_centroid.py --result_dir <path> --method weighted_average_center
    python calculate_entropy_centroid.py --result_dir <path> --top_percent 5.0 --bottom_percent 50.0
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from centroid import (
    load_data,
    load_or_create_evaluation_cache,
    save_centroid_results,
    analyze_trajectories,
    create_evaluator,
    CentroidVisualizer,
    generate_summary_statistics,
    generate_variance_analysis_summary,
)
from centroid.io import check_output_exists, get_output_dir_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate entropy centroid for high-entropy phases (supports math and logic tasks)"
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='Path to result directory with entropy_results.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: auto-generated based on parameters)'
    )
    parser.add_argument(
        '--top_percent',
        type=float,
        default=5.0,
        help='Percentage of highest entropy tokens for start threshold (default: 5.0)'
    )
    parser.add_argument(
        '--bottom_percent',
        type=float,
        default=50.0,
        help='Percentage for low entropy threshold to end HEP (default: 50.0)'
    )
    parser.add_argument(
        '--consecutive_low_threshold',
        type=int,
        default=3,
        help='Number of consecutive low-entropy tokens to end a HEP (default: 3)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='moment',
        choices=['moment', 'weighted_average', 'weighted_average_center', 'raw_entropy_weighted'],
        help='Centroid calculation method (default: moment)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=5,
        help='Timeout for symbolic comparison in math tasks (default: 5)'
    )
    parser.add_argument(
        '--filter_llm_error',
        action='store_true',
        default=False,
        help='Filter out tau2-bench trajectories with termination_reason=llm_error'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        default=False,
        help='Force re-run even if output already exists'
    )
    parser.add_argument(
        '--skip_per_problem',
        action='store_true',
        default=False,
        help='Skip per-problem visualization (faster execution)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate input
    if not os.path.exists(args.result_dir):
        print(f"Error: Result directory not found: {args.result_dir}")
        return 1

    # Set output directory
    if args.output_dir is None:
        args.output_dir = get_output_dir_name(
            args.result_dir,
            args.method,
            args.top_percent,
            args.bottom_percent,
            args.consecutive_low_threshold
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if results already exist
    if check_output_exists(args.output_dir) and not args.force:
        print("=" * 80)
        print("⊙ Output already exists, skipping calculation")
        print("=" * 80)
        print(f"Output directory: {args.output_dir}")
        print("To re-run, use --force flag or delete the output directory.")
        return 0

    # Print configuration
    print("=" * 80)
    print("Entropy Centroid Calculation")
    print("=" * 80)
    print(f"Result directory: {args.result_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Calculation method: {args.method}")
    if args.method != 'raw_entropy_weighted':
        print(f"High-entropy threshold: Top {args.top_percent}%")
        print(f"Low-entropy threshold: Bottom {args.bottom_percent}%")
        print(f"Consecutive low tokens to end HEP: {args.consecutive_low_threshold}")
    print("=" * 80)

    # Load data
    try:
        entropy_results = load_data(args.result_dir)
        print(f"\n✓ Loaded {len(entropy_results)} trajectories")
    except Exception as e:
        print(f"\n✗ Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    # Create evaluator (unified for math and logic)
    evaluator = create_evaluator(timeout=args.timeout)

    # Load or create evaluation cache
    evaluation_cache = load_or_create_evaluation_cache(
        args.result_dir,
        entropy_results,
        evaluator=evaluator
    )

    # Analyze trajectories
    try:
        correct_data, incorrect_data = analyze_trajectories(
            entropy_results,
            evaluator=evaluator,
            top_percent=args.top_percent,
            bottom_percent=args.bottom_percent,
            consecutive_low_threshold=args.consecutive_low_threshold,
            method=args.method,
            evaluation_cache=evaluation_cache,
            filter_llm_error=args.filter_llm_error,
        )

        n_correct = len(correct_data['trajectories'])
        n_incorrect = len(incorrect_data['trajectories'])
        n_problems = len(set(correct_data['by_problem'].keys()) | 
                        set(incorrect_data['by_problem'].keys()))

        print(f"\n✓ Analysis complete:")
        print(f"  - Correct trajectories: {n_correct}")
        print(f"  - Incorrect trajectories: {n_incorrect}")
        print(f"  - Total problems: {n_problems}")

    except Exception as e:
        print(f"\n✗ Error analyzing trajectories: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    # Generate results
    try:
        print("\n[1/4] Saving centroid results...")
        save_centroid_results(correct_data, incorrect_data, args.output_dir)

        print("[2/4] Generating summary statistics...")
        generate_summary_statistics(
            correct_data, incorrect_data, args.output_dir,
            args.top_percent, args.bottom_percent,
            args.consecutive_low_threshold, args.method
        )

        print("[3/4] Generating variance analysis...")
        generate_variance_analysis_summary(
            correct_data, incorrect_data, args.output_dir,
            args.top_percent, args.bottom_percent, args.method
        )

        print("[4/4] Generating visualizations...")
        visualizer = CentroidVisualizer(
            args.output_dir,
            top_percent=args.top_percent,
            method=args.method
        )
        visualizer.create_all_visualizations(
            correct_data, incorrect_data,
            skip_per_problem=args.skip_per_problem
        )

        print("\n" + "=" * 80)
        print("✓ All results generated!")
        print("=" * 80)
        print(f"\nOutput saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("  - entropy_centroid_results.json")
        print("  - entropy_centroid_summary.txt")
        print("  - variance_analysis_summary.txt")
        print("  - entropy_centroid_histogram.png")
        print("  - entropy_centroid_boxplot.png")
        print("  - entropy_centroid_vs_num_heps.png")
        print("  - hep_variance_histogram.png")
        print("  - centroid_vs_hep_variance.png")
        print("  - combined_centroid_variance_analysis.png")
        if not args.skip_per_problem:
            print("  - entropy_centroid_per_problem/")
            print("  - hep_variance_per_problem/")
            print("  - problem_categorization_summary.json")

    except Exception as e:
        print(f"\n✗ Error generating results: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
