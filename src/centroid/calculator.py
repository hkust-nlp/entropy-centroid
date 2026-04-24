"""
Core entropy centroid calculation functions.

This module contains the mathematical core for computing entropy centroids
from high-entropy phases (HEPs) in reasoning trajectories.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def compute_entropy_thresholds(
    entropy_sequence: List[Dict],
    top_percent: float = 5.0,
    bottom_percent: float = 50.0
) -> Tuple[float, float]:
    """
    Compute entropy thresholds for top N% and bottom M%.

    Args:
        entropy_sequence: List of token entropy data
        top_percent: Percentage for high entropy threshold
        bottom_percent: Percentage for low entropy threshold

    Returns:
        Tuple of (high_threshold, low_threshold)
    """
    # Extract all entropy values (convert to float to handle decimal.Decimal)
    entropies = [float(item['entropy']) for item in entropy_sequence if item.get('entropy') is not None]

    if not entropies:
        return float('inf'), float('-inf')

    # High threshold: top N% (e.g., 95th percentile for top 5%)
    high_percentile = 100.0 - top_percent
    high_threshold = np.percentile(entropies, high_percentile)

    # Low threshold: bottom M% (e.g., 50th percentile for bottom 50%)
    low_threshold = np.percentile(entropies, bottom_percent)

    return high_threshold, low_threshold


def compute_high_entropy_phases(
    entropy_sequence: List[Dict],
    top_percent: float = 5.0,
    bottom_percent: float = 50.0,
    consecutive_low_threshold: int = 3
) -> List[Tuple[int, int]]:
    """
    Compute high-entropy phases (HEPs).

    A high-entropy period (HEP) starts when a top N% high-entropy token appears
    and ends when M consecutive bottom M% tokens appear.

    Args:
        entropy_sequence: List of token entropy data
        top_percent: Percentage for high entropy threshold
        bottom_percent: Percentage for low entropy threshold
        consecutive_low_threshold: Number of consecutive low-entropy tokens to end a period

    Returns:
        List of (start_position, duration) tuples for each completed high-entropy period
    """
    high_threshold, low_threshold = compute_entropy_thresholds(
        entropy_sequence, top_percent, bottom_percent
    )

    hep_events = []

    in_high_entropy_period = False
    current_period_start = 0
    current_duration = 0
    consecutive_low_count = 0

    for i, item in enumerate(entropy_sequence):
        entropy = item.get('entropy')

        if entropy is None:
            # Skip tokens without entropy
            continue
        
        entropy = float(entropy)  # Convert to handle decimal.Decimal

        if not in_high_entropy_period:
            # Check if we should start a high-entropy period
            if entropy >= high_threshold:
                in_high_entropy_period = True
                current_period_start = i
                current_duration = 1
                consecutive_low_count = 0
        else:
            # We are in a high-entropy period
            current_duration += 1

            # Check if token is low-entropy
            if entropy <= low_threshold:
                consecutive_low_count += 1
            else:
                consecutive_low_count = 0

            # Check if we should end the period
            if consecutive_low_count >= consecutive_low_threshold:
                # End the period (subtract the consecutive_low_threshold tokens from duration)
                final_duration = current_duration - consecutive_low_threshold
                if final_duration > 0:
                    hep_events.append((current_period_start, final_duration))

                # Reset for next period
                in_high_entropy_period = False
                current_duration = 0
                consecutive_low_count = 0

    # If we end while still in a high-entropy period, record it
    if in_high_entropy_period and current_duration > 0:
        hep_events.append((current_period_start, current_duration))

    return hep_events


def compute_entropy_centroid(
    hep_events: List[Tuple[int, int]],
    trajectory_length: int,
    method: str = 'moment'
) -> Optional[float]:
    """
    Compute the centroid (center of mass) of high-entropy phases.

    Three methods are available:

    Method 1 - 'moment':
        Centroid = Σ(weight_i × position_i) / Total_Trajectory_Length

    Method 2 - 'weighted_average':
        Centroid = [Σ(weight_i × position_i) / Σ(weight_i)] / Total_Trajectory_Length
        Position is the start of each HEP.

    Method 3 - 'weighted_average_center':
        Same as Method 2, but position is the center of each HEP (start + duration/2).

    Args:
        hep_events: List of (start_position, duration) tuples
        trajectory_length: Total length of the trajectory in tokens
        method: Calculation method

    Returns:
        Normalized centroid position (0.0 to 1.0)
        Returns None if no HEP events or invalid trajectory length
    """
    if not hep_events or trajectory_length <= 0:
        return None

    # Calculate moment: Σ(weight_i × position_i)
    total_moment = 0.0
    total_weight = 0.0

    for start_pos, duration in hep_events:
        weight = duration

        # Determine position based on method
        if method == 'weighted_average_center':
            # Use center position of HEP
            position = start_pos + duration / 2.0
        else:
            # Use start position of HEP
            position = start_pos

        total_moment += weight * position
        total_weight += weight

    if method == 'moment':
        # Method 1: Normalize moment by trajectory length directly
        centroid = total_moment / trajectory_length
    elif method in ['weighted_average', 'weighted_average_center']:
        # Method 2 & 3: Calculate weighted average position first, then normalize
        if total_weight == 0:
            return None
        weighted_avg_position = total_moment / total_weight
        centroid = weighted_avg_position / trajectory_length
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'moment', 'weighted_average', or 'weighted_average_center'")

    return centroid


def compute_raw_entropy_centroid(
    entropy_sequence: List[Dict],
    trajectory_length: int
) -> Tuple[Optional[float], Dict]:
    """
    Compute the centroid using raw entropy values directly (without HEP phases).
    
    Formula:
        Centroid = [Σ(entropy_i × position_i) / Σ(entropy_i)] / Total_Trajectory_Length
    
    Each token's entropy value is treated as its "mass".

    Args:
        entropy_sequence: List of token entropy data with 'entropy' field
        trajectory_length: Total length of the trajectory in tokens
    
    Returns:
        Tuple of (centroid, stats) where:
        - centroid: Normalized centroid position (0.0 to 1.0)
        - stats: Dict with additional statistics
        Returns (None, {}) if no valid entropy data
    """
    if not entropy_sequence or trajectory_length <= 0:
        return None, {}
    
    # Calculate moment: Σ(entropy_i × position_i)
    total_moment = 0.0
    total_entropy = 0.0
    entropy_values = []
    
    for i, item in enumerate(entropy_sequence):
        entropy = item.get('entropy')
        
        if entropy is None or entropy < 0:
            continue
        
        entropy = float(entropy)  # Convert to handle decimal.Decimal
        position = i
        total_moment += entropy * position
        total_entropy += entropy
        entropy_values.append(entropy)
    
    if total_entropy == 0 or not entropy_values:
        return None, {}
    
    # Calculate weighted average position, then normalize
    weighted_avg_position = total_moment / total_entropy
    centroid = weighted_avg_position / trajectory_length
    
    # Calculate additional statistics
    stats = {
        'total_entropy': total_entropy,
        'mean_entropy': float(np.mean(entropy_values)),
        'median_entropy': float(np.median(entropy_values)),
        'std_entropy': float(np.std(entropy_values)),
        'max_entropy': float(np.max(entropy_values)),
        'min_entropy': float(np.min(entropy_values)),
        'num_valid_tokens': len(entropy_values),
        'weighted_avg_position_absolute': weighted_avg_position,
    }
    
    return centroid, stats


def compute_hep_statistics(hep_events: List[Tuple[int, int]], trajectory_length: int) -> Dict:
    """
    Compute detailed statistics about HEP events.

    Args:
        hep_events: List of (start_position, duration) tuples
        trajectory_length: Total length of the trajectory

    Returns:
        Dictionary with HEP statistics
    """
    if not hep_events:
        return {
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

    total_weight = sum(duration for _, duration in hep_events)
    avg_hep_duration = total_weight / len(hep_events)
    hep_positions = [start_pos for start_pos, _ in hep_events]
    first_hep_position = hep_positions[0] if hep_positions else None
    last_hep_position = hep_positions[-1] if hep_positions else None

    # HEP duration variance
    hep_durations = [duration for _, duration in hep_events]
    if len(hep_durations) > 1:
        hep_duration_variance = float(np.var(hep_durations))
        hep_duration_std = float(np.std(hep_durations))
    else:
        hep_duration_variance = 0.0
        hep_duration_std = 0.0

    # HEP position variance (normalized)
    hep_center_positions = [(start_pos + duration / 2.0) / trajectory_length
                            for start_pos, duration in hep_events]
    if len(hep_center_positions) > 1:
        hep_position_variance = float(np.var(hep_center_positions))
        hep_position_std = float(np.std(hep_center_positions))
    else:
        hep_position_variance = 0.0
        hep_position_std = 0.0

    # HEP spread
    hep_spread = (hep_center_positions[-1] - hep_center_positions[0]) if len(hep_center_positions) > 1 else 0.0

    return {
        'num_heps': len(hep_events),
        'total_hep_weight': total_weight,
        'avg_hep_duration': avg_hep_duration,
        'first_hep_position': first_hep_position,
        'last_hep_position': last_hep_position,
        'hep_duration_variance': hep_duration_variance,
        'hep_duration_std': hep_duration_std,
        'hep_position_variance': hep_position_variance,
        'hep_position_std': hep_position_std,
        'hep_spread': hep_spread,
    }
