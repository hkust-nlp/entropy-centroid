"""
Entropy Centroid Analysis Module

Computes the centroid (center of mass) of high-entropy phases (HEP) in reasoning trajectories.
Supports both math and logic task evaluation.
"""

from .calculator import (
    compute_entropy_thresholds,
    compute_high_entropy_phases,
    compute_entropy_centroid,
    compute_raw_entropy_centroid,
)

from .evaluator import (
    detect_task_type,
    evaluate_trajectory,
    create_evaluator,
    UnifiedEvaluator,
)

from .analyzer import analyze_trajectories

from .visualizer import CentroidVisualizer

from .statistics import (
    generate_summary_statistics,
    generate_variance_analysis_summary,
)

from .io import (
    load_data,
    save_centroid_results,
    load_or_create_evaluation_cache,
    get_trajectory_centroid_cache_path,
    load_trajectory_centroid_cache,
    load_trajectory_centroid_cache_with_fallback,
    load_from_analysis_results,
    save_trajectory_centroid_cache,
)

__all__ = [
    # Calculator
    'compute_entropy_thresholds',
    'compute_high_entropy_phases',
    'compute_entropy_centroid',
    'compute_raw_entropy_centroid',
    # Evaluator
    'detect_task_type',
    'evaluate_trajectory',
    'create_evaluator',
    'UnifiedEvaluator',
    # Analyzer
    'analyze_trajectories',
    # Visualizer
    'CentroidVisualizer',
    # Statistics
    'generate_summary_statistics',
    'generate_variance_analysis_summary',
    # I/O
    'load_data',
    'save_centroid_results',
    'load_or_create_evaluation_cache',
    'get_trajectory_centroid_cache_path',
    'load_trajectory_centroid_cache',
    'load_trajectory_centroid_cache_with_fallback',
    'load_from_analysis_results',
    'save_trajectory_centroid_cache',
]
