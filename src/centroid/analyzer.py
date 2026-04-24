"""
Trajectory analysis for entropy centroid calculation.

This module contains the main analysis logic for computing centroids
across all trajectories and categorizing them by correctness.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .calculator import (
    compute_high_entropy_phases,
    compute_entropy_centroid,
    compute_raw_entropy_centroid,
    compute_hep_statistics,
)
from .evaluator import evaluate_trajectory, detect_task_type


def analyze_trajectories(
    entropy_results: List[Dict],
    comparator=None,
    evaluator=None,
    top_percent: float = 5.0,
    bottom_percent: float = 50.0,
    consecutive_low_threshold: int = 3,
    method: str = 'moment',
    evaluation_cache: Dict[str, bool] = None,
    filter_llm_error: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Analyze all trajectories and compute entropy centroids.

    Args:
        entropy_results: List of trajectory entropy data
        comparator: Legacy AnswerComparator (for math tasks backward compatibility)
        evaluator: UnifiedEvaluator instance (preferred for mixed task types)
        top_percent: Percentage for high entropy threshold
        bottom_percent: Percentage for low entropy threshold
        consecutive_low_threshold: Number of consecutive low-entropy tokens to end HEP
        method: Calculation method: 'moment', 'weighted_average',
                'weighted_average_center', or 'raw_entropy_weighted'
        evaluation_cache: Pre-computed evaluation results (trajectory_id -> is_correct)
        filter_llm_error: If True, skip trajectories with termination_reason='llm_error'

    Returns:
        Tuple of (correct_data, incorrect_data) where each contains:
        - 'trajectories': List of trajectory data with centroid info
        - 'by_problem': Dict mapping problem_id to list of trajectory data
    """
    correct_data = {
        'trajectories': [],
        'by_problem': defaultdict(list)
    }
    incorrect_data = {
        'trajectories': [],
        'by_problem': defaultdict(list)
    }

    # Display parameters
    if method == 'raw_entropy_weighted':
        print(f"\nAnalyzing entropy centroids (method={method}, using raw entropy values)...")
    else:
        print(f"\nAnalyzing entropy centroids (top {top_percent}%, bottom {bottom_percent}%, "
              f"consecutive_low={consecutive_low_threshold}, method={method})...")

    # Detect task type from first trajectory for logging
    if entropy_results:
        first_task_type = detect_task_type(entropy_results[0])
        print(f"Detected task type: {first_task_type}")

    skipped_error = 0
    for trajectory in tqdm(entropy_results, desc="Analyzing trajectories"):
        traj_id = str(trajectory['id'])
        original_id = str(trajectory.get('original_id', traj_id.split('_traj_')[0]))

        # Skip trajectories from LLM errors
        if filter_llm_error and trajectory.get('termination_reason') == 'llm_error':
            skipped_error += 1
            continue

        # Check if trajectory is correct
        if evaluation_cache is not None and traj_id in evaluation_cache:
            is_correct = evaluation_cache[traj_id]
        else:
            is_correct = evaluate_trajectory(trajectory, comparator, evaluator)

        # Get entropy sequence
        entropy_sequence = trajectory.get('entropy_sequence', [])

        if not entropy_sequence:
            continue

        trajectory_length = len(entropy_sequence)

        # Compute centroid based on method
        traj_data = _compute_trajectory_data(
            traj_id=traj_id,
            original_id=original_id,
            entropy_sequence=entropy_sequence,
            trajectory_length=trajectory_length,
            method=method,
            top_percent=top_percent,
            bottom_percent=bottom_percent,
            consecutive_low_threshold=consecutive_low_threshold,
        )

        if traj_data is None:
            continue

        # Add to appropriate category
        if is_correct:
            correct_data['trajectories'].append(traj_data)
            correct_data['by_problem'][original_id].append(traj_data)
        else:
            incorrect_data['trajectories'].append(traj_data)
            incorrect_data['by_problem'][original_id].append(traj_data)

    if skipped_error > 0:
        print(f"  Skipped {skipped_error} trajectories with LLM errors (partial/anomalous data)")

    return correct_data, incorrect_data


def _compute_trajectory_data(
    traj_id: str,
    original_id: str,
    entropy_sequence: List[Dict],
    trajectory_length: int,
    method: str,
    top_percent: float,
    bottom_percent: float,
    consecutive_low_threshold: int,
) -> Optional[Dict]:
    """
    Compute trajectory data including centroid and statistics.

    Args:
        traj_id: Trajectory identifier
        original_id: Original problem identifier
        entropy_sequence: List of token entropy data
        trajectory_length: Number of tokens
        method: Centroid calculation method
        top_percent: High entropy threshold percentage
        bottom_percent: Low entropy threshold percentage
        consecutive_low_threshold: Consecutive low tokens to end HEP

    Returns:
        Dictionary with trajectory data, or None if computation fails
    """
    if method == 'raw_entropy_weighted':
        # Raw entropy weighted method
        centroid, raw_stats = compute_raw_entropy_centroid(
            entropy_sequence, trajectory_length
        )
        
        if centroid is None:
            return None
        
        return {
            'id': traj_id,
            'original_id': original_id,
            'trajectory_length': trajectory_length,
            'centroid': centroid,
            'centroid_absolute': centroid * trajectory_length,
            # Raw entropy statistics
            'total_entropy': raw_stats.get('total_entropy', 0),
            'mean_entropy': raw_stats.get('mean_entropy', 0),
            'median_entropy': raw_stats.get('median_entropy', 0),
            'std_entropy': raw_stats.get('std_entropy', 0),
            'max_entropy': raw_stats.get('max_entropy', 0),
            'min_entropy': raw_stats.get('min_entropy', 0),
            'num_valid_tokens': raw_stats.get('num_valid_tokens', 0),
            # Placeholder HEP values for compatibility
            'hep_events': [],
            'num_heps': 0,
            'total_hep_weight': 0,
            'avg_hep_duration': 0,
            'first_hep_position': None,
            'last_hep_position': None,
            'hep_duration_variance': 0.0,
            'hep_duration_std': 0.0,
            'hep_position_variance': 0.0,
            'hep_position_std': 0.0,
            'hep_spread': 0.0,
        }
    else:
        # HEP-based methods
        hep_events = compute_high_entropy_phases(
            entropy_sequence, top_percent, bottom_percent, consecutive_low_threshold
        )

        if not hep_events:
            return None

        centroid = compute_entropy_centroid(hep_events, trajectory_length, method)

        if centroid is None:
            return None

        # Compute HEP statistics
        hep_stats = compute_hep_statistics(hep_events, trajectory_length)

        return {
            'id': traj_id,
            'original_id': original_id,
            'trajectory_length': trajectory_length,
            'hep_events': hep_events,
            'centroid': centroid,
            'centroid_absolute': centroid * trajectory_length,
            **hep_stats,
        }


def get_analysis_summary(correct_data: Dict, incorrect_data: Dict) -> Dict:
    """
    Generate summary of analysis results.

    Args:
        correct_data: Data for correct trajectories
        incorrect_data: Data for incorrect trajectories

    Returns:
        Summary dictionary
    """
    n_correct = len(correct_data['trajectories'])
    n_incorrect = len(incorrect_data['trajectories'])
    n_problems = len(set(correct_data['by_problem'].keys()) | 
                     set(incorrect_data['by_problem'].keys()))

    return {
        'n_correct_trajectories': n_correct,
        'n_incorrect_trajectories': n_incorrect,
        'n_total_trajectories': n_correct + n_incorrect,
        'n_problems': n_problems,
        'n_problems_with_correct': len(correct_data['by_problem']),
        'n_problems_with_incorrect': len(incorrect_data['by_problem']),
    }
