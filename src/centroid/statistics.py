"""
Statistics generation for entropy centroid analysis.

Contains functions for generating summary statistics and variance analysis.
"""

import os
from typing import Dict, List

import numpy as np


def generate_summary_statistics(
    correct_data: Dict,
    incorrect_data: Dict,
    output_dir: str,
    top_percent: float = 5.0,
    bottom_percent: float = 50.0,
    consecutive_low_threshold: int = 3,
    method: str = 'moment'
) -> str:
    """
    Generate summary statistics for entropy centroids.

    Args:
        correct_data: Data for correct trajectories
        incorrect_data: Data for incorrect trajectories
        output_dir: Directory to save summary
        top_percent: Top percentage used
        bottom_percent: Bottom percentage used
        consecutive_low_threshold: Consecutive low threshold used
        method: Calculation method used

    Returns:
        Path to saved summary file
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, 'entropy_centroid_summary.txt')

    correct_centroids = [traj['centroid'] for traj in correct_data['trajectories']]
    incorrect_centroids = [traj['centroid'] for traj in incorrect_data['trajectories']]
    correct_num_heps = [traj['num_heps'] for traj in correct_data['trajectories']]
    incorrect_num_heps = [traj['num_heps'] for traj in incorrect_data['trajectories']]
    correct_weights = [traj['total_hep_weight'] for traj in correct_data['trajectories']]
    incorrect_weights = [traj['total_hep_weight'] for traj in incorrect_data['trajectories']]

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ENTROPY CENTROID SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        # Configuration
        f.write("Analysis Configuration:\n")
        f.write(f"  High-entropy threshold: Top {top_percent}%\n")
        f.write(f"  Low-entropy threshold: Bottom {bottom_percent}%\n")
        f.write(f"  Consecutive low tokens to end HEP: {consecutive_low_threshold}\n")
        f.write(f"  Calculation method: {method}\n\n")

        # Method description
        f.write("Centroid Formula:\n")
        if method == 'moment':
            f.write("  Method 1 (moment):\n")
            f.write("    Centroid = Σ(weight_i × position_i) / Total_Trajectory_Length\n")
        elif method == 'weighted_average':
            f.write("  Method 2 (weighted_average):\n")
            f.write("    Centroid = [Σ(weight_i × position_i) / Σ(weight_i)] / Total_Trajectory_Length\n")
        elif method == 'weighted_average_center':
            f.write("  Method 3 (weighted_average_center):\n")
            f.write("    Centroid = [Σ(weight_i × center_position_i) / Σ(weight_i)] / Total_Trajectory_Length\n")
        else:
            f.write("  Method 4 (raw_entropy_weighted):\n")
            f.write("    Centroid = [Σ(entropy_i × position_i) / Σ(entropy_i)] / Total_Trajectory_Length\n")
        f.write("\n")

        # Counts
        f.write("Trajectory Counts:\n")
        f.write(f"  Correct trajectories: {len(correct_data['trajectories'])}\n")
        f.write(f"  Incorrect trajectories: {len(incorrect_data['trajectories'])}\n")
        f.write(f"  Total problems: {len(set(correct_data['by_problem'].keys()) | set(incorrect_data['by_problem'].keys()))}\n\n")

        # Correct statistics
        if correct_centroids:
            f.write("Correct Trajectories:\n")
            f.write("  Centroid Statistics:\n")
            f.write(f"    Mean: {np.mean(correct_centroids):.4f}\n")
            f.write(f"    Median: {np.median(correct_centroids):.4f}\n")
            f.write(f"    Std: {np.std(correct_centroids):.4f}\n")
            f.write(f"    Min: {np.min(correct_centroids):.4f}\n")
            f.write(f"    Max: {np.max(correct_centroids):.4f}\n")
            f.write(f"    25th percentile: {np.percentile(correct_centroids, 25):.4f}\n")
            f.write(f"    75th percentile: {np.percentile(correct_centroids, 75):.4f}\n")
            f.write("  Number of HEPs:\n")
            f.write(f"    Mean: {np.mean(correct_num_heps):.2f}\n")
            f.write(f"    Median: {np.median(correct_num_heps):.1f}\n")
            f.write("  Total HEP Weight:\n")
            f.write(f"    Mean: {np.mean(correct_weights):.2f} tokens\n")
            f.write(f"    Median: {np.median(correct_weights):.1f} tokens\n\n")
        else:
            f.write("Correct Trajectories: No data\n\n")

        # Incorrect statistics
        if incorrect_centroids:
            f.write("Incorrect Trajectories:\n")
            f.write("  Centroid Statistics:\n")
            f.write(f"    Mean: {np.mean(incorrect_centroids):.4f}\n")
            f.write(f"    Median: {np.median(incorrect_centroids):.4f}\n")
            f.write(f"    Std: {np.std(incorrect_centroids):.4f}\n")
            f.write(f"    Min: {np.min(incorrect_centroids):.4f}\n")
            f.write(f"    Max: {np.max(incorrect_centroids):.4f}\n")
            f.write(f"    25th percentile: {np.percentile(incorrect_centroids, 25):.4f}\n")
            f.write(f"    75th percentile: {np.percentile(incorrect_centroids, 75):.4f}\n")
            f.write("  Number of HEPs:\n")
            f.write(f"    Mean: {np.mean(incorrect_num_heps):.2f}\n")
            f.write(f"    Median: {np.median(incorrect_num_heps):.1f}\n")
            f.write("  Total HEP Weight:\n")
            f.write(f"    Mean: {np.mean(incorrect_weights):.2f} tokens\n")
            f.write(f"    Median: {np.median(incorrect_weights):.1f} tokens\n\n")
        else:
            f.write("Incorrect Trajectories: No data\n\n")

        # Comparison
        if correct_centroids and incorrect_centroids:
            f.write("Comparison (Incorrect - Correct):\n")
            f.write(f"  Mean centroid diff: {np.mean(incorrect_centroids) - np.mean(correct_centroids):.4f}\n")
            f.write(f"  Median centroid diff: {np.median(incorrect_centroids) - np.median(correct_centroids):.4f}\n")
            f.write(f"  Mean num_heps diff: {np.mean(incorrect_num_heps) - np.mean(correct_num_heps):.2f}\n")
            f.write(f"  Mean weight diff: {np.mean(incorrect_weights) - np.mean(correct_weights):.2f} tokens\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✓ Saved summary statistics: {summary_file}")
    return summary_file


def generate_variance_analysis_summary(
    correct_data: Dict,
    incorrect_data: Dict,
    output_dir: str,
    top_percent: float = 5.0,
    bottom_percent: float = 50.0,
    method: str = 'moment'
) -> str:
    """
    Generate detailed variance analysis summary.

    Args:
        correct_data: Data for correct trajectories
        incorrect_data: Data for incorrect trajectories
        output_dir: Directory to save summary
        top_percent: Top percentage used
        bottom_percent: Bottom percentage used
        method: Calculation method used

    Returns:
        Path to saved summary file
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, 'variance_analysis_summary.txt')

    correct_variances = [traj['hep_duration_variance'] for traj in correct_data['trajectories']]
    incorrect_variances = [traj['hep_duration_variance'] for traj in incorrect_data['trajectories']]
    correct_pos_variances = [traj['hep_position_variance'] for traj in correct_data['trajectories']]
    incorrect_pos_variances = [traj['hep_position_variance'] for traj in incorrect_data['trajectories']]
    correct_spreads = [traj['hep_spread'] for traj in correct_data['trajectories']]
    incorrect_spreads = [traj['hep_spread'] for traj in incorrect_data['trajectories']]

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HEP VARIANCE ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Analysis Parameters:\n")
        f.write(f"  Top percent: {top_percent}%\n")
        f.write(f"  Bottom percent: {bottom_percent}%\n")
        f.write(f"  Method: {method}\n\n")

        # Duration variance
        f.write("HEP Duration Variance (variance of HEP lengths in tokens²):\n")
        if correct_variances:
            f.write(f"  Correct:\n")
            f.write(f"    Mean: {np.mean(correct_variances):.6f}\n")
            f.write(f"    Median: {np.median(correct_variances):.6f}\n")
            f.write(f"    Std: {np.std(correct_variances):.6f}\n")
        if incorrect_variances:
            f.write(f"  Incorrect:\n")
            f.write(f"    Mean: {np.mean(incorrect_variances):.6f}\n")
            f.write(f"    Median: {np.median(incorrect_variances):.6f}\n")
            f.write(f"    Std: {np.std(incorrect_variances):.6f}\n")
        f.write("\n")

        # Position variance
        f.write("HEP Position Variance (variance of normalized HEP positions):\n")
        if correct_pos_variances:
            f.write(f"  Correct:\n")
            f.write(f"    Mean: {np.mean(correct_pos_variances):.6f}\n")
            f.write(f"    Median: {np.median(correct_pos_variances):.6f}\n")
        if incorrect_pos_variances:
            f.write(f"  Incorrect:\n")
            f.write(f"    Mean: {np.mean(incorrect_pos_variances):.6f}\n")
            f.write(f"    Median: {np.median(incorrect_pos_variances):.6f}\n")
        f.write("\n")

        # HEP Spread
        f.write("HEP Spread (range of normalized HEP positions):\n")
        if correct_spreads:
            f.write(f"  Correct:\n")
            f.write(f"    Mean: {np.mean(correct_spreads):.4f}\n")
            f.write(f"    Median: {np.median(correct_spreads):.4f}\n")
        if incorrect_spreads:
            f.write(f"  Incorrect:\n")
            f.write(f"    Mean: {np.mean(incorrect_spreads):.4f}\n")
            f.write(f"    Median: {np.median(incorrect_spreads):.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✓ Saved variance analysis: {summary_file}")
    return summary_file


def compute_aggregate_statistics(correct_data: Dict, incorrect_data: Dict) -> Dict:
    """
    Compute aggregate statistics for both correct and incorrect trajectories.

    Args:
        correct_data: Data for correct trajectories
        incorrect_data: Data for incorrect trajectories

    Returns:
        Dictionary with aggregate statistics
    """
    def compute_stats(data_list: List) -> Dict:
        if not data_list:
            return {'count': 0, 'mean': None, 'median': None, 'std': None, 'min': None, 'max': None}
        return {
            'count': len(data_list),
            'mean': float(np.mean(data_list)),
            'median': float(np.median(data_list)),
            'std': float(np.std(data_list)),
            'min': float(np.min(data_list)),
            'max': float(np.max(data_list)),
        }

    correct_centroids = [t['centroid'] for t in correct_data['trajectories']]
    incorrect_centroids = [t['centroid'] for t in incorrect_data['trajectories']]

    return {
        'correct': {
            'centroid': compute_stats(correct_centroids),
            'num_heps': compute_stats([t['num_heps'] for t in correct_data['trajectories']]),
            'total_weight': compute_stats([t['total_hep_weight'] for t in correct_data['trajectories']]),
        },
        'incorrect': {
            'centroid': compute_stats(incorrect_centroids),
            'num_heps': compute_stats([t['num_heps'] for t in incorrect_data['trajectories']]),
            'total_weight': compute_stats([t['total_hep_weight'] for t in incorrect_data['trajectories']]),
        },
        'summary': {
            'n_correct': len(correct_data['trajectories']),
            'n_incorrect': len(incorrect_data['trajectories']),
            'n_problems': len(set(correct_data['by_problem'].keys()) | 
                            set(incorrect_data['by_problem'].keys())),
        }
    }
