"""
Answer selection strategies for multi-trajectory inference.

Implements different strategies for selecting the final answer when multiple
trajectories are generated per problem (Best-of-N sampling).

Key concepts:
1. **Outlier Detection**: Identifies anomalous trajectories based on centroid position
   - 'none': No filtering
   - 'threshold': Filter if |centroid - mean| > threshold
   - 'std': Filter if centroid outside mean ± N*std
   
2. **Selection Methods**: Different strategies to select/vote for final answer
   - entropy_centroid: Select trajectory with earliest centroid
   - centroid_voting: Majority vote among bottom X% centroid trajectories
   - centroid_weighted_voting: Weighted vote where weight depends on centroid position
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Tuple
from collections import Counter
import numpy as np

# Import evaluation cache utilities (lazy import to avoid circular dependency)
def _get_evaluation_cache_module():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from evaluation.evaluation_cache import (
        load_evaluation_cache,
        get_answers_from_cache,
        get_correctness_from_cache
    )
    return load_evaluation_cache, get_answers_from_cache, get_correctness_from_cache


# ============================================================================
# Centroid Computation Utilities
# ============================================================================

def compute_raw_entropy_centroid(entropy_sequence: List[Dict], trajectory_length: int) -> Optional[float]:
    """
    Compute centroid using raw entropy values directly (no HEP phases).
    
    Formula: Centroid = [Σ(entropy_i × position_i) / Σ(entropy_i)] / trajectory_length
    
    Each token's entropy value is treated as its "mass".
    
    Args:
        entropy_sequence: List of token entropy data with 'entropy' field
        trajectory_length: Total length of trajectory in tokens
        
    Returns:
        Normalized centroid position (0.0 to 1.0), or None if invalid
    """
    if not entropy_sequence or trajectory_length <= 0:
        return None
    
    total_moment = 0.0
    total_entropy = 0.0
    
    for i, item in enumerate(entropy_sequence):
        entropy = item.get('entropy')
        if entropy is None or entropy < 0:
            continue
        entropy = float(entropy)  # Convert to handle decimal.Decimal from ijson
        total_moment += entropy * i
        total_entropy += entropy
    
    if total_entropy == 0:
        return None
    
    weighted_avg_position = total_moment / total_entropy
    return weighted_avg_position / trajectory_length


# ============================================================================
# Outlier Detection Utilities
# ============================================================================

def detect_outliers(
    centroids: List[float],
    method: str = 'none',
    threshold: float = 0.1,
    num_std: float = 1.0
) -> Tuple[List[bool], Dict]:
    """
    Detect outlier trajectories based on centroid values.
    
    Args:
        centroids: List of centroid values
        method: Detection method:
            - 'none': No outlier detection
            - 'threshold': Outlier if |centroid - mean| > threshold
            - 'std': Outlier if centroid outside mean ± num_std * std
            - 'below_mean_threshold': Outlier if centroid < mean - threshold
        threshold: Threshold value for 'threshold' method
        num_std: Number of standard deviations for 'std' method
        
    Returns:
        Tuple of (is_outlier list, detection_info dict)
    """
    if not centroids:
        return [], {}
    
    n = len(centroids)
    mean_centroid = np.mean(centroids)
    std_centroid = np.std(centroids) if n > 1 else 0.0
    
    is_outlier = [False] * n
    detection_info = {
        'method': method,
        'mean_centroid': float(mean_centroid),
        'std_centroid': float(std_centroid),
        'num_outliers': 0
    }
    
    if method == 'none':
        pass
    
    elif method == 'threshold':
        # Outlier if |centroid - mean| > threshold
        for i, c in enumerate(centroids):
            if abs(c - mean_centroid) > threshold:
                is_outlier[i] = True
        detection_info['threshold'] = threshold
        detection_info['lower_bound'] = float(mean_centroid - threshold)
        detection_info['upper_bound'] = float(mean_centroid + threshold)
    
    elif method == 'std':
        # Outlier if outside mean ± num_std * std
        if std_centroid > 0:
            lower_bound = mean_centroid - num_std * std_centroid
            upper_bound = mean_centroid + num_std * std_centroid
            for i, c in enumerate(centroids):
                if c < lower_bound or c > upper_bound:
                    is_outlier[i] = True
            detection_info['num_std'] = num_std
            detection_info['lower_bound'] = float(lower_bound)
            detection_info['upper_bound'] = float(upper_bound)
    
    elif method == 'below_mean_threshold':
        # Outlier if centroid < mean - threshold (for early outliers only)
        lower_bound = mean_centroid - threshold
        for i, c in enumerate(centroids):
            if c < lower_bound:
                is_outlier[i] = True
        detection_info['threshold'] = threshold
        detection_info['lower_bound'] = float(lower_bound)
    
    detection_info['num_outliers'] = sum(is_outlier)
    detection_info['outlier_indices'] = [i for i, v in enumerate(is_outlier) if v]
    
    return is_outlier, detection_info


def compute_centroid_weights(
    centroids: List[float],
    is_outlier: List[bool],
    weight_method: str = 'linear',
    outlier_penalty: float = -1.0,
    temperature: float = 1.0
) -> Tuple[List[float], Dict]:
    """
    Compute voting weights based on centroid position.
    
    Lower centroid = higher weight (hypothesis: earlier HEPs are better)
    Outliers can receive negative weights (penalty)
    
    Args:
        centroids: List of centroid values
        is_outlier: List indicating which trajectories are outliers
        weight_method: Weight calculation method:
            - 'linear': weight = 1 - normalized_centroid (linear scaling)
            - 'inverse': weight = 1 / centroid (inverse proportional)
            - 'exponential': weight = exp(-centroid * temperature)
        outlier_penalty: Weight assigned to outliers (negative = penalty)
        temperature: Temperature parameter for exponential method
        
    Returns:
        Tuple of (weights list, weight_info dict)
    """
    if not centroids:
        return [], {}
    
    n = len(centroids)
    weights = [0.0] * n
    
    # Get non-outlier centroids for normalization
    valid_centroids = [c for c, out in zip(centroids, is_outlier) if not out]
    
    if not valid_centroids:
        # All are outliers, use uniform weights
        return [1.0 / n] * n, {'method': weight_method, 'all_outliers': True}
    
    min_c = min(valid_centroids)
    max_c = max(valid_centroids)
    range_c = max_c - min_c if max_c > min_c else 1.0
    
    weight_info = {
        'method': weight_method,
        'min_centroid': float(min_c),
        'max_centroid': float(max_c),
        'range': float(range_c),
        'outlier_penalty': outlier_penalty
    }
    
    for i, (c, out) in enumerate(zip(centroids, is_outlier)):
        if out:
            # Outliers get penalty weight
            weights[i] = outlier_penalty
        else:
            if weight_method == 'linear':
                # Linear: lower centroid = higher weight
                # Normalize to [0, 1], then invert
                normalized = (c - min_c) / range_c if range_c > 0 else 0.5
                weights[i] = 1.0 - normalized
            
            elif weight_method == 'inverse':
                # Inverse: weight proportional to 1/centroid
                weights[i] = 1.0 / (c + 0.01)  # Add small epsilon to avoid division by zero
            
            elif weight_method == 'exponential':
                # Exponential decay based on centroid
                weights[i] = np.exp(-c * temperature)
            
            else:
                weights[i] = 1.0
    
    # Normalize non-negative weights to sum to 1 (excluding negative outlier penalties)
    positive_sum = sum(w for w in weights if w > 0)
    if positive_sum > 0:
        weights = [w / positive_sum if w > 0 else w for w in weights]
    
    weight_info['weights_before_norm'] = [float(w) for w in weights]
    
    return weights, weight_info


class AnswerSelector(ABC):
    """
    Abstract base class for answer selection strategies.
    """

    @abstractmethod
    def select_answer(
        self,
        trajectories: List[Dict],
        trajectory_selector,
        entropy_results: List[Dict],
        step_divisions: List[Dict],
        auxiliary_scores: List[Dict],
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict:
        """
        Select the final answer from multiple trajectories.

        Args:
            trajectories: List of trajectory dictionaries
            trajectory_selector: Trajectory selector for scoring trajectories
            entropy_results: Full entropy results
            step_divisions: Full step divisions
            auxiliary_scores: Optional auxiliary trajectory scores
            extract_answer_fn: Function to extract answer from generated text

        Returns:
            Dictionary with:
            {
                'selected_trajectory': Dict,  # The trajectory to use for evaluation
                'selection_method': str,
                'selection_metadata': Dict,  # Method-specific metadata
            }
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this selection strategy."""
        pass


class BestOfNAnswerSelector(AnswerSelector):
    """
    Traditional Best-of-N selection: choose trajectory with highest score.

    Uses the trajectory selector's scoring function to rank trajectories
    and selects the one with the highest score.
    """

    def select_answer(
        self,
        trajectories: List[Dict],
        trajectory_selector,
        entropy_results: List[Dict],
        step_divisions: List[Dict],
        auxiliary_scores: List[Dict],
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict:
        """
        Select trajectory with highest score.

        Args:
            trajectories: List of trajectory dictionaries
            trajectory_selector: Trajectory selector for scoring
            entropy_results: Full entropy results
            step_divisions: Full step divisions
            auxiliary_scores: Optional auxiliary trajectory scores
            extract_answer_fn: Function to extract answer (not used in Best-of-N)

        Returns:
            Selection result with best trajectory
        """
        if not trajectories:
            return {
                'selected_trajectory': None,
                'selection_method': 'best_of_n',
                'selection_metadata': {
                    'num_trajectories': 0,
                    'selection_score': 0.0
                }
            }

        # Use trajectory selector to select best trajectory
        selection = trajectory_selector.select_best_trajectory(
            trajectories,
            entropy_results,
            step_divisions,
            auxiliary_scores
        )

        return {
            'selected_trajectory': selection['selected_trajectory'],
            'selection_method': 'best_of_n',
            'selection_metadata': {
                'num_trajectories': len(trajectories),
                'selection_score': selection['selection_score'],
                'all_scores': selection['all_scores'],
                'trajectory_strategy': selection['strategy']
            }
        }

    def get_name(self) -> str:
        return "best_of_n"


class EntropyCentroidAnswerSelector(AnswerSelector):
    """
    Entropy centroid-based answer selection (no auxiliary scores needed).

    Selects the trajectory with the smallest (earliest) entropy centroid,
    based on the hypothesis that correct answers tend to have high-entropy
    phases concentrated earlier in the reasoning process.

    The centroid is computed using the weighted_average_center method:
    1. Identify High-Entropy Phases (HEPs) in each trajectory
    2. For each HEP: weight = duration, position = center (start + duration/2)
    3. Centroid = [Σ(weight × position) / Σ(weight)] / trajectory_length

    Outlier filtering can use different methods:
    - 'below_mean_threshold': Exclude if centroid < mean - threshold
    - 'threshold': Exclude if |centroid - mean| > threshold
    - 'std': Exclude if centroid outside mean ± num_std * std
    - 'none': No outlier filtering
    """

    def __init__(
        self,
        top_percent: float = 5.0,
        bottom_percent: float = 50.0,
        consecutive_low_threshold: int = 3,
        outlier_threshold: float = 0.1,
        outlier_method: str = 'below_mean_threshold',
        outlier_num_std: float = 1.0,
        normalize_for_grouping: bool = True
    ):
        """
        Initialize entropy centroid selector.

        Args:
            top_percent: Percentage for high entropy threshold (default: 5.0)
            bottom_percent: Percentage for low entropy threshold (default: 50.0)
            consecutive_low_threshold: Number of consecutive low-entropy tokens to end HEP (default: 3)
            outlier_threshold: Threshold for outlier detection (default: 0.1)
            outlier_method: Outlier detection method:
                - 'below_mean_threshold': Exclude if centroid < mean - threshold
                - 'threshold': Exclude if |centroid - mean| > threshold
                - 'std': Exclude if centroid outside mean ± num_std * std
                - 'none': No outlier filtering
            outlier_num_std: Number of std deviations for 'std' method (default: 1.0)
            normalize_for_grouping: Whether to normalize answers before grouping
        """
        self.top_percent = top_percent
        self.bottom_percent = bottom_percent
        self.consecutive_low_threshold = consecutive_low_threshold
        self.outlier_threshold = outlier_threshold
        self.outlier_method = outlier_method
        self.outlier_num_std = outlier_num_std
        self.normalize_for_grouping = normalize_for_grouping

    def _compute_entropy_thresholds(
        self,
        entropy_sequence: List[Dict]
    ) -> Tuple[float, float]:
        """
        Compute entropy thresholds for top N% and bottom M%.

        Args:
            entropy_sequence: List of token entropy data

        Returns:
            Tuple of (high_threshold, low_threshold)
        """
        import numpy as np

        # Convert to float to handle decimal.Decimal values
        entropies = [float(item['entropy']) for item in entropy_sequence if item.get('entropy') is not None]

        if not entropies:
            return float('inf'), float('-inf')

        high_percentile = 100.0 - self.top_percent
        high_threshold = np.percentile(entropies, high_percentile)
        low_threshold = np.percentile(entropies, self.bottom_percent)

        return high_threshold, low_threshold

    def _compute_high_entropy_phases(
        self,
        entropy_sequence: List[Dict]
    ) -> List[Tuple[int, int]]:
        """
        Compute high-entropy phases (HEPs).

        Args:
            entropy_sequence: List of token entropy data

        Returns:
            List of (start_position, duration) tuples for each HEP
        """
        high_threshold, low_threshold = self._compute_entropy_thresholds(entropy_sequence)

        hep_events = []
        in_high_entropy_period = False
        current_period_start = 0
        current_duration = 0
        consecutive_low_count = 0

        for i, item in enumerate(entropy_sequence):
            entropy = item.get('entropy')

            if entropy is None:
                continue

            if not in_high_entropy_period:
                if entropy >= high_threshold:
                    in_high_entropy_period = True
                    current_period_start = i
                    current_duration = 1
                    consecutive_low_count = 0
            else:
                current_duration += 1

                if entropy <= low_threshold:
                    consecutive_low_count += 1
                else:
                    consecutive_low_count = 0

                if consecutive_low_count >= self.consecutive_low_threshold:
                    final_duration = current_duration - self.consecutive_low_threshold
                    if final_duration > 0:
                        hep_events.append((current_period_start, final_duration))

                    in_high_entropy_period = False
                    current_duration = 0
                    consecutive_low_count = 0

        if in_high_entropy_period and current_duration > 0:
            hep_events.append((current_period_start, current_duration))

        return hep_events

    def _compute_centroid(
        self,
        hep_events: List[Tuple[int, int]],
        trajectory_length: int
    ) -> Optional[float]:
        """
        Compute the centroid using weighted_average_center method.

        Centroid = [Σ(weight × position) / Σ(weight)] / trajectory_length
        where position = center of each HEP (start + duration/2)

        Args:
            hep_events: List of (start_position, duration) tuples
            trajectory_length: Total length of trajectory in tokens

        Returns:
            Normalized centroid position (0.0 to 1.0), or None if invalid
        """
        if not hep_events or trajectory_length <= 0:
            return None

        total_moment = 0.0
        total_weight = 0.0

        for start_pos, duration in hep_events:
            weight = duration
            # Use center position of HEP
            position = start_pos + duration / 2.0
            total_moment += weight * position
            total_weight += weight

        if total_weight == 0:
            return None

        weighted_avg_position = total_moment / total_weight
        centroid = weighted_avg_position / trajectory_length

        return centroid

    def select_answer(
        self,
        trajectories: List[Dict],
        trajectory_selector,  # Not used
        entropy_results: List[Dict],
        step_divisions: List[Dict],  # Not used
        auxiliary_scores: List[Dict],  # Not used
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict:
        """
        Select answer based on entropy centroid (smallest/earliest wins).

        Args:
            trajectories: List of trajectory dictionaries
            trajectory_selector: Not used (kept for interface compatibility)
            entropy_results: Full entropy results (needed for entropy_sequence)
            step_divisions: Not used
            auxiliary_scores: Not used
            extract_answer_fn: Function to extract answer from text

        Returns:
            Selection result with trajectory that has smallest valid centroid
        """
        import numpy as np

        if not trajectories:
            return {
                'selected_trajectory': None,
                'selection_method': 'entropy_centroid',
                'selection_metadata': {
                    'num_trajectories': 0,
                    'selection_result': 'no_trajectories'
                }
            }

        if len(trajectories) == 1:
            return {
                'selected_trajectory': trajectories[0],
                'selection_method': 'entropy_centroid',
                'selection_metadata': {
                    'num_trajectories': 1,
                    'selection_result': 'single_trajectory'
                }
            }

        # Create lookup for entropy data
        entropy_by_id = {item['id']: item for item in entropy_results}

        # Compute centroid for each trajectory
        trajectory_centroids = []

        for traj in trajectories:
            traj_id = traj.get('id')
            entropy_data = entropy_by_id.get(traj_id, {})
            entropy_sequence = entropy_data.get('entropy_sequence', [])

            if not entropy_sequence:
                continue

            # Compute HEPs and centroid
            hep_events = self._compute_high_entropy_phases(entropy_sequence)
            trajectory_length = len(entropy_sequence)
            centroid = self._compute_centroid(hep_events, trajectory_length)

            if centroid is not None:
                trajectory_centroids.append({
                    'trajectory': traj,
                    'centroid': centroid,
                    'num_heps': len(hep_events),
                    'trajectory_id': traj_id
                })

        if not trajectory_centroids:
            # No valid centroids computed, fallback to first trajectory
            return {
                'selected_trajectory': trajectories[0],
                'selection_method': 'entropy_centroid',
                'selection_metadata': {
                    'num_trajectories': len(trajectories),
                    'selection_result': 'no_valid_centroids',
                    'fallback': 'first_trajectory'
                }
            }

        # Apply outlier detection using the unified function
        all_centroids = [tc['centroid'] for tc in trajectory_centroids]
        mean_centroid = np.mean(all_centroids)

        is_outlier, outlier_info = detect_outliers(
            all_centroids,
            method=self.outlier_method,
            threshold=self.outlier_threshold,
            num_std=self.outlier_num_std
        )
        
        valid_trajectories = []
        excluded_trajectories = []
        
        for tc, is_out in zip(trajectory_centroids, is_outlier):
            if not is_out:
                valid_trajectories.append(tc)
            else:
                excluded_trajectories.append(tc)

        # If all were filtered out, use the unfiltered list
        if not valid_trajectories:
            valid_trajectories = trajectory_centroids
            excluded_trajectories = []

        # Select trajectory with smallest (earliest) centroid
        valid_trajectories.sort(key=lambda x: x['centroid'])
        selected = valid_trajectories[0]

        # Prepare metadata
        centroid_distribution = {
            'all_centroids': [(tc['trajectory_id'], tc['centroid']) for tc in trajectory_centroids],
            'mean_centroid': float(mean_centroid),
            'std_centroid': outlier_info.get('std_centroid', 0.0),
            'outlier_detection': outlier_info,
            'num_excluded': len(excluded_trajectories),
            'excluded_ids': [tc['trajectory_id'] for tc in excluded_trajectories]
        }

        return {
            'selected_trajectory': selected['trajectory'],
            'selection_method': 'entropy_centroid',
            'selection_metadata': {
                'num_trajectories': len(trajectories),
                'num_valid_centroids': len(trajectory_centroids),
                'selected_centroid': selected['centroid'],
                'selected_num_heps': selected['num_heps'],
                'centroid_distribution': centroid_distribution,
                'parameters': {
                    'top_percent': self.top_percent,
                    'bottom_percent': self.bottom_percent,
                    'consecutive_low_threshold': self.consecutive_low_threshold,
                    'outlier_threshold': self.outlier_threshold,
                    'outlier_method': self.outlier_method,
                    'outlier_num_std': self.outlier_num_std
                }
            }
        }

    def get_name(self) -> str:
        return "entropy_centroid"


class MajorityVotingAnswerSelector(AnswerSelector):
    """
    Pure majority voting answer selection (no auxiliary scores needed).

    Each trajectory gets equal voting weight (1 vote per trajectory).
    The answer that appears most frequently wins.

    Strategy:
    1. Extracts answer from each trajectory (or uses cached answers)
    2. Groups trajectories by their extracted answer
    3. Counts votes for each unique answer (1 vote per trajectory)
    4. Selects the answer with the most votes
    5. In case of ties, selects the answer from the trajectory that appears first

    This is useful when you want a simple ensemble method without relying on
    auxiliary quality metrics.
    
    Caching:
    - Can use evaluation_cache.json for pre-extracted answers
    - Falls back to on-the-fly extraction if cache not available
    """

    def __init__(self, normalize_for_grouping: bool = True, cache_dir: str = None):
        """
        Initialize majority voting selector.

        Args:
            normalize_for_grouping: Whether to normalize answers before grouping
                                   (case-insensitive, whitespace-normalized)
            cache_dir: Directory to load/save evaluation cache (optional)
        """
        self.normalize_for_grouping = normalize_for_grouping
        self.cache_dir = cache_dir
        self._answer_cache = {}  # trajectory_id -> extracted_answer
        self._cache_loaded = False
    
    def _load_answer_cache(self):
        """Load cached answers from evaluation cache."""
        if self._cache_loaded or not self.cache_dir:
            return
        
        try:
            load_evaluation_cache, get_answers_from_cache, _ = _get_evaluation_cache_module()
            cache = load_evaluation_cache(self.cache_dir)
            if cache:
                self._answer_cache = get_answers_from_cache(cache)
                if self._answer_cache:
                    print(f"  Loaded {len(self._answer_cache)} cached answers")
        except Exception as e:
            print(f"  Warning: Failed to load answer cache: {e}")
        
        self._cache_loaded = True

    def _normalize_answer_key(self, answer: Optional[str]) -> str:
        """
        Normalize answer for grouping purposes.

        Args:
            answer: Raw extracted answer

        Returns:
            Normalized answer key for grouping
        """
        if answer is None:
            return '__none__'

        if self.normalize_for_grouping:
            # Normalize: lowercase, strip whitespace, collapse multiple spaces
            key = str(answer).strip().lower()
            key = ' '.join(key.split())  # Collapse whitespace
            return key if key else '__empty__'
        else:
            return str(answer)

    def _build_missing_answer_fallback(
        self,
        trajectories: List[Dict],
        trajectory_answers: List[Dict],
    ) -> Dict:
        """Fallback when no trajectory produced an extractable answer."""
        selected_trajectory = trajectories[0]
        return {
            'selected_trajectory': selected_trajectory,
            'selection_method': 'majority_voting',
            'selection_metadata': {
                'num_trajectories': len(trajectories),
                'num_unique_answers': 0,
                'winning_answer': None,
                'winning_answer_key': None,
                'winning_votes': 0,
                'winning_percentage': 0.0,
                'num_trajectories_with_winning_answer': 0,
                'vote_distribution': {},
                'had_tie': False,
                'tied_answers': None,
                'voting_result': 'all_answers_missing',
                'missing_answer_count': len(trajectory_answers),
            }
        }

    def select_answer(
        self,
        trajectories: List[Dict],
        trajectory_selector,  # Not used in majority voting
        entropy_results: List[Dict],  # Not used in majority voting
        step_divisions: List[Dict],  # Not used in majority voting
        auxiliary_scores: List[Dict],  # Not used in majority voting
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict:
        """
        Select answer using pure majority voting.

        Args:
            trajectories: List of trajectory dictionaries
            trajectory_selector: Not used (kept for interface compatibility)
            entropy_results: Not used (kept for interface compatibility)
            step_divisions: Not used (kept for interface compatibility)
            auxiliary_scores: Not used (kept for interface compatibility)
            extract_answer_fn: Function to extract answer from text

        Returns:
            Selection result with trajectory that produced winning answer
        """
        if not trajectories:
            return {
                'selected_trajectory': None,
                'selection_method': 'majority_voting',
                'selection_metadata': {
                    'num_trajectories': 0,
                    'voting_result': 'no_trajectories'
                }
            }

        # Single trajectory - return it directly
        if len(trajectories) == 1:
            return {
                'selected_trajectory': trajectories[0],
                'selection_method': 'majority_voting',
                'selection_metadata': {
                    'num_trajectories': 1,
                    'voting_result': 'single_trajectory'
                }
            }

        # Load cached answers if available
        self._load_answer_cache()
        
        # Extract answers from all trajectories (using cache when available)
        trajectory_answers = []
        answer_to_trajectories = {}  # Map normalized answer key to list of trajectories

        for traj in trajectories:
            traj_id = traj.get('id')
            
            # Try to use cached answer first
            if traj_id in self._answer_cache:
                extracted_answer = self._answer_cache[traj_id]
            else:
                generated_text = traj['sample'].get('generated_text', '')
                extracted_answer = extract_answer_fn(generated_text)
            
            answer_key = self._normalize_answer_key(extracted_answer)

            trajectory_answers.append({
                'trajectory': traj,
                'answer': extracted_answer,
                'answer_key': answer_key,
                'trajectory_id': traj_id
            })

            if extracted_answer is None:
                continue

            if answer_key not in answer_to_trajectories:
                answer_to_trajectories[answer_key] = []
            answer_to_trajectories[answer_key].append({
                'trajectory': traj,
                'answer': extracted_answer
            })

        if not answer_to_trajectories:
            return self._build_missing_answer_fallback(trajectories, trajectory_answers)

        # Count votes for each answer
        answer_votes = {key: len(trajs) for key, trajs in answer_to_trajectories.items()}

        # Find the answer with maximum votes
        max_votes = max(answer_votes.values())
        winning_answers = [key for key, votes in answer_votes.items() if votes == max_votes]

        # If there's a tie, select the first occurring answer
        if len(winning_answers) > 1:
            # Find which tied answer appears first in the trajectory list
            for ta in trajectory_answers:
                if ta['answer_key'] in winning_answers:
                    winning_answer_key = ta['answer_key']
                    break
        else:
            winning_answer_key = winning_answers[0]

        # Get the winning trajectories and select the first one
        winning_trajectories = answer_to_trajectories[winning_answer_key]
        selected_trajectory = winning_trajectories[0]['trajectory']
        selected_answer = winning_trajectories[0]['answer']

        # Prepare voting statistics
        vote_distribution = {
            key: {
                'votes': votes,
                'percentage': votes / len(trajectories) * 100
            }
            for key, votes in answer_votes.items()
        }

        return {
            'selected_trajectory': selected_trajectory,
            'selection_method': 'majority_voting',
            'selection_metadata': {
                'num_trajectories': len(trajectories),
                'num_unique_answers': len(answer_votes),
                'winning_answer': selected_answer,
                'winning_answer_key': winning_answer_key,
                'winning_votes': max_votes,
                'winning_percentage': max_votes / len(trajectories) * 100,
                'num_trajectories_with_winning_answer': len(winning_trajectories),
                'vote_distribution': vote_distribution,
                'had_tie': len(winning_answers) > 1,
                'tied_answers': winning_answers if len(winning_answers) > 1 else None
            }
        }

    def get_name(self) -> str:
        return "majority_voting"


class LLMMajorityVotingAnswerSelector(AnswerSelector):
    """
    LLM-based majority voting for AMO-Bench description questions.

    Behavior:
    - For non-AMO tasks, or non-description AMO questions: fall back to
      pure majority voting on extracted answers.
    - For AMO-Bench description questions: use LLM-based grading to filter
      candidate answers, then pick the most-voted correct answer.

    This selector is opt-in and will fall back to standard majority voting
    if the AMO evaluator or API key is not available.
    """

    def __init__(self, normalize_for_grouping: bool = True, cache_dir: str = None):
        self.normalize_for_grouping = normalize_for_grouping
        self.cache_dir = cache_dir
        self._answer_cache = {}
        self._cache_loaded = False
        self._amo_evaluator = None
        self._llm_judge_cache = {}  # (qid, answer_key) -> bool

    def _load_answer_cache(self):
        if self._cache_loaded or not self.cache_dir:
            return

        try:
            load_evaluation_cache, get_answers_from_cache, _ = _get_evaluation_cache_module()
            cache = load_evaluation_cache(self.cache_dir)
            if cache:
                self._answer_cache = get_answers_from_cache(cache)
                if self._answer_cache:
                    print(f"  Loaded {len(self._answer_cache)} cached answers")
        except Exception as e:
            print(f"  Warning: Failed to load answer cache: {e}")

        self._cache_loaded = True

    def _normalize_answer_key(self, answer: Optional[str]) -> str:
        if answer is None:
            return '__none__'

        if self.normalize_for_grouping:
            key = str(answer).strip().lower()
            key = ' '.join(key.split())
            return key if key else '__empty__'
        return str(answer)

    def _amo_api_configured(self) -> bool:
        import os
        return any(
            os.environ.get(key)
            for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "AMO_BENCH_API_KEY")
        )

    def _get_amo_evaluator(self):
        if self._amo_evaluator is not None:
            return self._amo_evaluator
        try:
            from evaluation.amo_bench_evaluator import AMOBenchEvaluator
            self._amo_evaluator = AMOBenchEvaluator(
                skip_description=not self._amo_api_configured()
            )
        except Exception:
            self._amo_evaluator = None
        return self._amo_evaluator

    def _get_question_id(self, sample: Dict) -> Optional[int]:
        original_id = sample.get('original_id', '')
        try:
            return int(original_id)
        except (ValueError, TypeError):
            traj_id = str(sample.get('id', ''))
            try:
                return int(traj_id.split('_')[0])
            except (ValueError, IndexError):
                return None

    def _is_amo_description(self, sample: Dict) -> bool:
        evaluator = self._get_amo_evaluator()
        if evaluator is None:
            return False
        qid = self._get_question_id(sample)
        if qid is None:
            return False
        info = evaluator.get_question_info(qid)
        if not info:
            return False
        return info.get('answer_type') == 'description'

    def _judge_description_answer(self, sample: Dict, answer_key: str, answer_text: str) -> Optional[bool]:
        evaluator = self._get_amo_evaluator()
        if evaluator is None or not self._amo_api_configured():
            return None

        qid = self._get_question_id(sample)
        if qid is None:
            return None

        cache_key = (qid, answer_key)
        if cache_key in self._llm_judge_cache:
            return self._llm_judge_cache[cache_key]

        # Construct a minimal generated_text so AMO extractor can find the answer
        generated_text = f"### The final answer is: {answer_text}"
        trajectory = {
            'id': f'{qid}_traj_0',
            'original_id': qid,
            'generated_text': generated_text,
            'solution': sample.get('solution', '')
        }

        try:
            is_correct = evaluator.evaluate(trajectory)
        except Exception:
            is_correct = None

        if is_correct is not None:
            self._llm_judge_cache[cache_key] = bool(is_correct)
        return is_correct

    def select_answer(
        self,
        trajectories: List[Dict],
        trajectory_selector,
        entropy_results: List[Dict],
        step_divisions: List[Dict],
        auxiliary_scores: List[Dict],
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict:
        if not trajectories:
            return {
                'selected_trajectory': None,
                'selection_method': 'llm_majority_voting',
                'selection_metadata': {
                    'num_trajectories': 0,
                    'voting_result': 'no_trajectories'
                }
            }

        # Load cached answers if available
        self._load_answer_cache()

        # Extract answers from all trajectories (using cache when available)
        trajectory_answers = []
        answer_to_trajectories = {}

        for traj in trajectories:
            traj_id = traj.get('id')
            if traj_id in self._answer_cache:
                extracted_answer = self._answer_cache[traj_id]
            else:
                generated_text = traj['sample'].get('generated_text', '')
                extracted_answer = extract_answer_fn(generated_text)

            answer_key = self._normalize_answer_key(extracted_answer)
            trajectory_answers.append({
                'trajectory': traj,
                'answer': extracted_answer,
                'answer_key': answer_key,
                'trajectory_id': traj_id
            })

            if answer_key not in answer_to_trajectories:
                answer_to_trajectories[answer_key] = []
            answer_to_trajectories[answer_key].append({
                'trajectory': traj,
                'answer': extracted_answer
            })

        # Majority vote counts
        answer_votes = {key: len(trajs) for key, trajs in answer_to_trajectories.items()}

        # Check if this is AMO-Bench description
        sample = trajectories[0]['sample']
        is_amo_description = self._is_amo_description(sample)

        # If not AMO description, fall back to pure majority voting
        if not is_amo_description:
            max_votes = max(answer_votes.values())
            winning_answers = [key for key, votes in answer_votes.items() if votes == max_votes]
            if len(winning_answers) > 1:
                for ta in trajectory_answers:
                    if ta['answer_key'] in winning_answers:
                        winning_answer_key = ta['answer_key']
                        break
            else:
                winning_answer_key = winning_answers[0]

            winning_trajectories = answer_to_trajectories[winning_answer_key]
            selected_trajectory = winning_trajectories[0]['trajectory']
            selected_answer = winning_trajectories[0]['answer']

            return {
                'selected_trajectory': selected_trajectory,
                'selection_method': 'llm_majority_voting',
                'selection_metadata': {
                    'num_trajectories': len(trajectories),
                    'num_unique_answers': len(answer_votes),
                    'winning_answer': selected_answer,
                    'winning_answer_key': winning_answer_key,
                    'winning_votes': max_votes,
                    'winning_percentage': max_votes / len(trajectories) * 100,
                    'num_trajectories_with_winning_answer': len(winning_trajectories),
                    'vote_distribution': {
                        key: {
                            'votes': votes,
                            'percentage': votes / len(trajectories) * 100
                        }
                        for key, votes in answer_votes.items()
                    },
                    'llm_used': False
                }
            }

        # LLM-based filtering for AMO description
        judged = {}
        for answer_key, trajs in answer_to_trajectories.items():
            answer_text = trajs[0]['answer']
            judged[answer_key] = self._judge_description_answer(sample, answer_key, answer_text)

        correct_keys = [k for k, v in judged.items() if v is True]

        # If LLM unavailable or no correct answers, fall back to majority
        if not correct_keys:
            max_votes = max(answer_votes.values())
            winning_answers = [key for key, votes in answer_votes.items() if votes == max_votes]
            if len(winning_answers) > 1:
                for ta in trajectory_answers:
                    if ta['answer_key'] in winning_answers:
                        winning_answer_key = ta['answer_key']
                        break
            else:
                winning_answer_key = winning_answers[0]

            winning_trajectories = answer_to_trajectories[winning_answer_key]
            selected_trajectory = winning_trajectories[0]['trajectory']
            selected_answer = winning_trajectories[0]['answer']

            return {
                'selected_trajectory': selected_trajectory,
                'selection_method': 'llm_majority_voting',
                'selection_metadata': {
                    'num_trajectories': len(trajectories),
                    'num_unique_answers': len(answer_votes),
                    'winning_answer': selected_answer,
                    'winning_answer_key': winning_answer_key,
                    'winning_votes': max_votes,
                    'winning_percentage': max_votes / len(trajectories) * 100,
                    'num_trajectories_with_winning_answer': len(winning_trajectories),
                    'vote_distribution': {
                        key: {
                            'votes': votes,
                            'percentage': votes / len(trajectories) * 100
                        }
                        for key, votes in answer_votes.items()
                    },
                    'llm_used': True,
                    'llm_fallback': True,
                    'llm_judgments': judged
                }
            }

        # Among correct answers, pick the most voted
        max_correct_votes = max(answer_votes[k] for k in correct_keys)
        winning_answers = [k for k in correct_keys if answer_votes[k] == max_correct_votes]
        if len(winning_answers) > 1:
            for ta in trajectory_answers:
                if ta['answer_key'] in winning_answers:
                    winning_answer_key = ta['answer_key']
                    break
        else:
            winning_answer_key = winning_answers[0]

        winning_trajectories = answer_to_trajectories[winning_answer_key]
        selected_trajectory = winning_trajectories[0]['trajectory']
        selected_answer = winning_trajectories[0]['answer']

        return {
            'selected_trajectory': selected_trajectory,
            'selection_method': 'llm_majority_voting',
            'selection_metadata': {
                'num_trajectories': len(trajectories),
                'num_unique_answers': len(answer_votes),
                'winning_answer': selected_answer,
                'winning_answer_key': winning_answer_key,
                'winning_votes': max_correct_votes,
                'winning_percentage': max_correct_votes / len(trajectories) * 100,
                'num_trajectories_with_winning_answer': len(winning_trajectories),
                'vote_distribution': {
                    key: {
                        'votes': votes,
                        'percentage': votes / len(trajectories) * 100
                    }
                    for key, votes in answer_votes.items()
                },
                'llm_used': True,
                'llm_judgments': judged,
                'llm_filtered_keys': correct_keys
            }
        }

    def get_name(self) -> str:
        return "llm_majority_voting"


class RandomAnswerSelector(AnswerSelector):
    """
    Random answer selection (no auxiliary scores needed).

    Randomly selects one trajectory from the available trajectories.
    This is useful as a baseline to measure the expected accuracy when
    randomly selecting one trajectory from a set of N generated trajectories.

    The selection can be made reproducible by setting a seed.
    """

    def __init__(self, seed: Optional[int] = None, num_samples: int = 1):
        """
        Initialize random selector.

        Args:
            seed: Random seed for reproducibility. If None, uses a random seed.
            num_samples: Number of random samples to run (for averaging accuracy estimation).
                        When > 1, returns metadata about all samples but selects from first sample.
        """
        self.seed = seed
        self.num_samples = num_samples
        self._rng = np.random.RandomState(seed)

    def select_answer(
        self,
        trajectories: List[Dict],
        trajectory_selector,  # Not used
        entropy_results: List[Dict],  # Not used
        step_divisions: List[Dict],  # Not used
        auxiliary_scores: List[Dict],  # Not used
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict:
        """
        Randomly select a trajectory.

        Args:
            trajectories: List of trajectory dictionaries
            trajectory_selector: Not used (kept for interface compatibility)
            entropy_results: Not used (kept for interface compatibility)
            step_divisions: Not used (kept for interface compatibility)
            auxiliary_scores: Not used (kept for interface compatibility)
            extract_answer_fn: Function to extract answer from text

        Returns:
            Selection result with randomly selected trajectory
        """
        if not trajectories:
            return {
                'selected_trajectory': None,
                'selection_method': 'random',
                'selection_metadata': {
                    'num_trajectories': 0,
                    'selection_result': 'no_trajectories'
                }
            }

        # Single trajectory - return it directly
        if len(trajectories) == 1:
            return {
                'selected_trajectory': trajectories[0],
                'selection_method': 'random',
                'selection_metadata': {
                    'num_trajectories': 1,
                    'selection_result': 'single_trajectory',
                    'seed': self.seed
                }
            }

        # Randomly select one trajectory
        selected_index = self._rng.randint(0, len(trajectories))
        selected_trajectory = trajectories[selected_index]

        # For multiple samples, record all selected indices (useful for accuracy estimation)
        sample_indices = [selected_index]
        if self.num_samples > 1:
            for _ in range(self.num_samples - 1):
                sample_indices.append(self._rng.randint(0, len(trajectories)))

        # Extract the answer from selected trajectory (for metadata)
        generated_text = selected_trajectory['sample'].get('generated_text', '')
        selected_answer = extract_answer_fn(generated_text)

        return {
            'selected_trajectory': selected_trajectory,
            'selection_method': 'random',
            'selection_metadata': {
                'num_trajectories': len(trajectories),
                'selected_index': selected_index,
                'selected_id': selected_trajectory.get('id'),
                'selected_answer': selected_answer,
                'seed': self.seed,
                'num_samples': self.num_samples,
                'all_sample_indices': sample_indices if self.num_samples > 1 else None
            }
        }

    def get_name(self) -> str:
        seed_str = f"_seed{self.seed}" if self.seed is not None else ""
        return f"random{seed_str}"


class CentroidVotingAnswerSelector(AnswerSelector):
    """
    Centroid-based voting answer selection (no auxiliary scores needed).

    Selects trajectories with the lowest (earliest) entropy centroids to participate
    in voting. This implements the hypothesis that correct answers tend to have
    high-entropy phases concentrated earlier in the reasoning process.

    Strategy:
    1. Compute entropy centroid for each trajectory (HEP-based or raw entropy)
    2. Select trajectories with centroid in bottom X% (e.g., bottom 10%, 20%, 30%)
    3. Use majority voting among selected trajectories
    4. Return the answer with most votes (from selected trajectories)

    This is efficient because it computes centroids once and allows different
    percentile thresholds to be tested without recomputing.
    
    Centroid computation methods:
    - 'hep': Use High-Entropy Phases (HEPs) to compute centroid (default)
    - 'raw_entropy': Use raw entropy values directly (no HEP computation needed)
    """

    def __init__(
        self,
        top_percent: float = 5.0,
        bottom_percent: float = 50.0,
        consecutive_low_threshold: int = 3,
        centroid_select_percent: float = 30.0,
        centroid_method: str = 'hep',
        normalize_for_grouping: bool = True,
        cache_dir: str = None
    ):
        """
        Initialize centroid voting selector.

        Args:
            top_percent: Percentage for high entropy threshold (for HEP detection)
            bottom_percent: Percentage for low entropy threshold (for HEP detection)
            consecutive_low_threshold: Number of consecutive low-entropy tokens to end HEP
            centroid_select_percent: Select trajectories with centroid in bottom X% (default: 30.0)
            centroid_method: Centroid calculation method:
                - 'hep': Use HEP-based centroid (requires top/bottom/consecutive params)
                - 'raw_entropy': Use raw entropy weighted centroid (simpler, no HEP params needed)
            normalize_for_grouping: Whether to normalize answers before grouping
            cache_dir: Directory to load/save centroid cache (enables efficiency)
        """
        self.top_percent = top_percent
        self.bottom_percent = bottom_percent
        self.consecutive_low_threshold = consecutive_low_threshold
        self.centroid_select_percent = centroid_select_percent
        self.centroid_method = centroid_method
        self.normalize_for_grouping = normalize_for_grouping
        self.cache_dir = cache_dir
        
        # Cache for computed centroids (trajectory_id -> centroid data)
        self._centroid_cache = {}
        self._cache_loaded = False

    def _compute_entropy_thresholds(
        self,
        entropy_sequence: List[Dict]
    ) -> Tuple[float, float]:
        """Compute entropy thresholds for top N% and bottom M%."""
        # Convert to float to handle decimal.Decimal values
        entropies = [float(item['entropy']) for item in entropy_sequence if item.get('entropy') is not None]
        if not entropies:
            return float('inf'), float('-inf')
        high_percentile = 100.0 - self.top_percent
        high_threshold = np.percentile(entropies, high_percentile)
        low_threshold = np.percentile(entropies, self.bottom_percent)
        return high_threshold, low_threshold

    def _compute_high_entropy_phases(
        self,
        entropy_sequence: List[Dict]
    ) -> List[Tuple[int, int]]:
        """Compute high-entropy phases (HEPs)."""
        high_threshold, low_threshold = self._compute_entropy_thresholds(entropy_sequence)
        hep_events = []
        in_high_entropy_period = False
        current_period_start = 0
        current_duration = 0
        consecutive_low_count = 0

        for i, item in enumerate(entropy_sequence):
            entropy = item.get('entropy')
            if entropy is None:
                continue
            if not in_high_entropy_period:
                if entropy >= high_threshold:
                    in_high_entropy_period = True
                    current_period_start = i
                    current_duration = 1
                    consecutive_low_count = 0
            else:
                current_duration += 1
                if entropy <= low_threshold:
                    consecutive_low_count += 1
                else:
                    consecutive_low_count = 0
                if consecutive_low_count >= self.consecutive_low_threshold:
                    final_duration = current_duration - self.consecutive_low_threshold
                    if final_duration > 0:
                        hep_events.append((current_period_start, final_duration))
                    in_high_entropy_period = False
                    current_duration = 0
                    consecutive_low_count = 0

        if in_high_entropy_period and current_duration > 0:
            hep_events.append((current_period_start, current_duration))
        return hep_events

    def _compute_centroid(
        self,
        hep_events: List[Tuple[int, int]],
        trajectory_length: int
    ) -> Optional[float]:
        """Compute centroid using weighted_average_center method."""
        if not hep_events or trajectory_length <= 0:
            return None
        total_moment = 0.0
        total_weight = 0.0
        for start_pos, duration in hep_events:
            weight = duration
            position = start_pos + duration / 2.0
            total_moment += weight * position
            total_weight += weight
        if total_weight == 0:
            return None
        weighted_avg_position = total_moment / total_weight
        return weighted_avg_position / trajectory_length

    def _normalize_answer_key(self, answer: Optional[str]) -> str:
        """Normalize answer for grouping purposes."""
        if answer is None:
            return '__none__'
        if self.normalize_for_grouping:
            key = str(answer).strip().lower()
            key = ' '.join(key.split())
            return key if key else '__empty__'
        return str(answer)

    def _load_cache(self):
        """Load centroid cache if available (with fallback to analysis results)."""
        if self._cache_loaded or not self.cache_dir:
            return
        
        try:
            # Import here to avoid circular dependency
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from centroid.io import load_trajectory_centroid_cache_with_fallback
            
            cache = load_trajectory_centroid_cache_with_fallback(
                self.cache_dir,
                self.top_percent,
                self.bottom_percent,
                self.consecutive_low_threshold,
                self.centroid_method
            )
            
            if cache and 'trajectories' in cache:
                self._centroid_cache = cache['trajectories']
                source = cache.get('parameters', {}).get('source', 'cache')
                print(f"  Loaded {len(self._centroid_cache)} centroids from {source} (method={self.centroid_method})")
        except Exception as e:
            print(f"  Warning: Failed to load centroid cache: {e}")
        
        self._cache_loaded = True

    def _save_cache(self, trajectory_data: Dict[str, Dict]):
        """Save computed centroids to cache."""
        if not self.cache_dir:
            return
        
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from centroid.io import save_trajectory_centroid_cache
            
            save_trajectory_centroid_cache(
                self.cache_dir,
                self.top_percent,
                self.bottom_percent,
                self.consecutive_low_threshold,
                trajectory_data,
                self.centroid_method
            )
        except Exception as e:
            print(f"  Warning: Failed to save centroid cache: {e}")

    def compute_all_centroids(
        self,
        trajectories: List[Dict],
        entropy_results: List[Dict],
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict[str, Dict]:
        """
        Compute centroids for all trajectories (with caching).
        
        This method computes centroids once and caches them, allowing different
        centroid_select_percent values to be tested efficiently.
        
        Supports two centroid computation methods:
        - 'hep': High-Entropy Phase based (requires top/bottom/consecutive params)
        - 'raw_entropy': Raw entropy weighted (simpler, no HEP params needed)
        
        Args:
            trajectories: List of trajectory dictionaries
            entropy_results: Full entropy results
            extract_answer_fn: Function to extract answer from text
            
        Returns:
            Dict mapping trajectory_id to centroid data
        """
        # Try to load from cache first
        self._load_cache()
        
        # Check if all trajectories are already cached
        traj_ids = [traj.get('id') for traj in trajectories]
        uncached_ids = [tid for tid in traj_ids if tid not in self._centroid_cache]
        
        if not uncached_ids:
            # All trajectories already cached, no need to compute anything
            return self._centroid_cache
        
        # Create entropy lookup only for uncached trajectories
        entropy_by_id = {item['id']: item for item in entropy_results if item['id'] in uncached_ids}
        
        # Track if we need to update cache
        cache_updated = False
        
        # Try to load evaluation cache for extracted answers (fallback)
        # Only load if we have uncached trajectories
        eval_cache = {}
        if self.cache_dir and uncached_ids:
            try:
                import os
                eval_cache_path = os.path.join(self.cache_dir, 'evaluation_cache.json')
                if os.path.exists(eval_cache_path):
                    import json
                    with open(eval_cache_path, 'r') as f:
                        eval_data = json.load(f)
                    eval_cache = eval_data.get('trajectories', {})
            except Exception:
                pass  # Ignore errors, will extract from text
        
        for traj in trajectories:
            traj_id = traj.get('id')
            traj_id_str = str(traj_id) if traj_id is not None else ''

            # Skip if already cached
            if traj_id in self._centroid_cache:
                continue

            cache_updated = True
            entropy_data = entropy_by_id.get(traj_id, {})
            entropy_sequence = entropy_data.get('entropy_sequence', [])

            # Get extracted answer: try eval cache first, then extract from text
            extracted_answer = None
            if traj_id in eval_cache:
                extracted_answer = eval_cache[traj_id].get('extracted_answer')
            if extracted_answer is None:
                generated_text = traj['sample'].get('generated_text', '')
                if generated_text:  # Only try extraction if text exists
                    extracted_answer = extract_answer_fn(generated_text)

            if not entropy_sequence:
                self._centroid_cache[traj_id] = {
                    'centroid': None,
                    'num_heps': 0,
                    'trajectory_length': 0,
                    'centroid_method': self.centroid_method,
                    'original_id': str(traj.get('original_id', traj_id_str.split('_traj_')[0] if '_traj_' in traj_id_str else traj_id_str)),
                    'extracted_answer': str(extracted_answer) if extracted_answer is not None else None
                }
                continue

            trajectory_length = len(entropy_sequence)

            # Compute centroid based on method
            if self.centroid_method == 'raw_entropy':
                # Use raw entropy weighted centroid
                centroid = compute_raw_entropy_centroid(entropy_sequence, trajectory_length)
                num_heps = 0  # No HEPs in raw entropy method
            else:
                # Default: Use HEP-based centroid
                hep_events = self._compute_high_entropy_phases(entropy_sequence)
                centroid = self._compute_centroid(hep_events, trajectory_length)
                num_heps = len(hep_events)

            self._centroid_cache[traj_id] = {
                'centroid': centroid,
                'num_heps': num_heps,
                'trajectory_length': trajectory_length,
                'centroid_method': self.centroid_method,
                'original_id': str(traj.get('original_id', traj_id_str.split('_traj_')[0] if '_traj_' in traj_id_str else traj_id_str)),
                'extracted_answer': str(extracted_answer) if extracted_answer is not None else None
            }
        
        # Save updated cache
        if cache_updated and self.cache_dir:
            self._save_cache(self._centroid_cache)
        
        return self._centroid_cache

    def select_answer(
        self,
        trajectories: List[Dict],
        trajectory_selector,  # Not used
        entropy_results: List[Dict],
        step_divisions: List[Dict],  # Not used
        auxiliary_scores: List[Dict],  # Not used
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict:
        """
        Select answer using centroid-based voting.

        Args:
            trajectories: List of trajectory dictionaries
            trajectory_selector: Not used (kept for interface compatibility)
            entropy_results: Full entropy results (needed for centroid computation)
            step_divisions: Not used
            auxiliary_scores: Not used
            extract_answer_fn: Function to extract answer from text

        Returns:
            Selection result with trajectory that produced winning answer
        """
        if not trajectories:
            return {
                'selected_trajectory': None,
                'selection_method': 'centroid_voting',
                'selection_metadata': {
                    'num_trajectories': 0,
                    'centroid_select_percent': self.centroid_select_percent,
                    'selection_result': 'no_trajectories'
                }
            }

        # Single trajectory - return it directly
        if len(trajectories) == 1:
            return {
                'selected_trajectory': trajectories[0],
                'selection_method': 'centroid_voting',
                'selection_metadata': {
                    'num_trajectories': 1,
                    'centroid_select_percent': self.centroid_select_percent,
                    'selection_result': 'single_trajectory'
                }
            }

        # Compute all centroids (with caching)
        centroid_data = self.compute_all_centroids(trajectories, entropy_results, extract_answer_fn)
        
        # Collect trajectories with valid centroids
        trajectory_centroids = []
        for traj in trajectories:
            traj_id = traj.get('id')
            if traj_id in centroid_data:
                data = centroid_data[traj_id]
                if data['centroid'] is not None:
                    trajectory_centroids.append({
                        'trajectory': traj,
                        'centroid': data['centroid'],
                        'num_heps': data['num_heps'],
                        'trajectory_id': traj_id,
                        'extracted_answer': data['extracted_answer']
                    })

        if not trajectory_centroids:
            # No valid centroids, fallback to majority voting on all
            return self._fallback_majority_voting(trajectories, extract_answer_fn)

        # Sort by centroid (ascending - lower centroid = earlier)
        trajectory_centroids.sort(key=lambda x: x['centroid'])
        
        # Select bottom X% by centroid (trajectories with earliest centroids)
        num_to_select = max(1, int(len(trajectory_centroids) * self.centroid_select_percent / 100.0))
        selected_for_voting = trajectory_centroids[:num_to_select]
        
        # Perform majority voting among selected trajectories
        answer_votes = {}
        answer_to_trajectories = {}
        
        for tc in selected_for_voting:
            answer = tc['extracted_answer']
            answer_key = self._normalize_answer_key(answer)
            
            if answer_key not in answer_votes:
                answer_votes[answer_key] = 0
                answer_to_trajectories[answer_key] = []
            
            answer_votes[answer_key] += 1
            answer_to_trajectories[answer_key].append(tc)
        
        # Find winning answer
        max_votes = max(answer_votes.values())
        winning_answers = [key for key, votes in answer_votes.items() if votes == max_votes]
        
        # If tie, use the one with smallest centroid among winners
        if len(winning_answers) > 1:
            best_winner = None
            best_centroid = float('inf')
            for key in winning_answers:
                for tc in answer_to_trajectories[key]:
                    if tc['centroid'] < best_centroid:
                        best_centroid = tc['centroid']
                        best_winner = key
            winning_answer_key = best_winner
        else:
            winning_answer_key = winning_answers[0]
        
        # Select trajectory with smallest centroid among winners
        winning_trajectories = answer_to_trajectories[winning_answer_key]
        best_winning_traj = min(winning_trajectories, key=lambda x: x['centroid'])
        
        # Prepare vote distribution
        vote_distribution = {
            key: {
                'votes': votes,
                'percentage': votes / len(selected_for_voting) * 100
            }
            for key, votes in answer_votes.items()
        }
        
        # Prepare centroid statistics
        all_centroids = [tc['centroid'] for tc in trajectory_centroids]
        selected_centroids = [tc['centroid'] for tc in selected_for_voting]

        return {
            'selected_trajectory': best_winning_traj['trajectory'],
            'selection_method': 'centroid_voting',
            'selection_metadata': {
                'num_trajectories': len(trajectories),
                'num_valid_centroids': len(trajectory_centroids),
                'num_selected_for_voting': len(selected_for_voting),
                'centroid_select_percent': self.centroid_select_percent,
                'winning_answer': best_winning_traj['extracted_answer'],
                'winning_answer_key': winning_answer_key,
                'winning_votes': max_votes,
                'winning_centroid': best_winning_traj['centroid'],
                'vote_distribution': vote_distribution,
                'centroid_statistics': {
                    'all_mean': float(np.mean(all_centroids)),
                    'all_std': float(np.std(all_centroids)),
                    'all_min': float(np.min(all_centroids)),
                    'all_max': float(np.max(all_centroids)),
                    'selected_mean': float(np.mean(selected_centroids)),
                    'selected_max': float(np.max(selected_centroids)),
                    'selection_threshold': float(selected_centroids[-1]) if selected_centroids else None
                },
                'parameters': {
                    'top_percent': self.top_percent,
                    'bottom_percent': self.bottom_percent,
                    'consecutive_low_threshold': self.consecutive_low_threshold,
                    'centroid_select_percent': self.centroid_select_percent
                },
                'had_tie': len(winning_answers) > 1,
                'tied_answers': winning_answers if len(winning_answers) > 1 else None
            }
        }

    def _fallback_majority_voting(
        self,
        trajectories: List[Dict],
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict:
        """Fallback to pure majority voting when no valid centroids."""
        answer_votes = {}
        answer_to_trajectories = {}
        
        for traj in trajectories:
            generated_text = traj['sample'].get('generated_text', '')
            answer = extract_answer_fn(generated_text)
            answer_key = self._normalize_answer_key(answer)
            
            if answer_key not in answer_votes:
                answer_votes[answer_key] = 0
                answer_to_trajectories[answer_key] = []
            
            answer_votes[answer_key] += 1
            answer_to_trajectories[answer_key].append(traj)
        
        if not answer_votes:
            return {
                'selected_trajectory': trajectories[0],
                'selection_method': 'centroid_voting',
                'selection_metadata': {
                    'num_trajectories': len(trajectories),
                    'centroid_select_percent': self.centroid_select_percent,
                    'selection_result': 'fallback_first_trajectory'
                }
            }
        
        winning_answer_key = max(answer_votes.keys(), key=lambda k: answer_votes[k])
        selected_trajectory = answer_to_trajectories[winning_answer_key][0]
        
        return {
            'selected_trajectory': selected_trajectory,
            'selection_method': 'centroid_voting',
            'selection_metadata': {
                'num_trajectories': len(trajectories),
                'centroid_select_percent': self.centroid_select_percent,
                'selection_result': 'fallback_majority_voting',
                'winning_votes': answer_votes[winning_answer_key]
            }
        }

    def get_name(self) -> str:
        method_str = "_raw" if self.centroid_method == 'raw_entropy' else ""
        return f"centroid_voting{method_str}_{int(self.centroid_select_percent)}pct"


class CentroidWeightedVotingAnswerSelector(AnswerSelector):
    """
    Centroid-weighted voting answer selection (no auxiliary scores needed).

    Each trajectory votes with a weight determined by its centroid position:
    - Lower centroid = higher positive weight (earlier HEPs are better)
    - Higher centroid = lower weight or negative weight (penalty for late HEPs)
    
    Centroid computation methods:
    - 'hep': Use High-Entropy Phases (HEPs) to compute centroid (default)
    - 'raw_entropy': Use raw entropy values directly (no HEP computation needed)
    
    Outlier detection can filter or penalize anomalous trajectories:
    - 'none': No outlier detection
    - 'threshold': Outlier if |centroid - mean| > threshold
    - 'std': Outlier if centroid outside mean ± N*std

    Strategy:
    1. Compute entropy centroid for each trajectory (HEP-based or raw entropy)
    2. Detect outliers based on configured method
    3. Compute voting weights based on centroid (lower = higher weight)
    4. Outliers receive penalty weight (configurable, can be negative)
    5. Aggregate weighted votes for each unique answer
    6. Return the answer with highest total weight
    """

    def __init__(
        self,
        top_percent: float = 5.0,
        bottom_percent: float = 50.0,
        consecutive_low_threshold: int = 3,
        centroid_method: str = 'hep',
        weight_method: str = 'linear',
        outlier_method: str = 'std',
        outlier_threshold: float = 0.1,
        outlier_num_std: float = 1.0,
        outlier_penalty: float = -0.5,
        temperature: float = 1.0,
        normalize_for_grouping: bool = True,
        cache_dir: str = None
    ):
        """
        Initialize centroid weighted voting selector.

        Args:
            top_percent: Percentage for high entropy threshold (for HEP detection)
            bottom_percent: Percentage for low entropy threshold (for HEP detection)
            consecutive_low_threshold: Number of consecutive low-entropy tokens to end HEP
            centroid_method: Centroid calculation method:
                - 'hep': Use HEP-based centroid (requires top/bottom/consecutive params)
                - 'raw_entropy': Use raw entropy weighted centroid (simpler, no HEP params needed)
            weight_method: Weight calculation method:
                - 'linear': weight = 1 - normalized_centroid
                - 'inverse': weight = 1 / centroid
                - 'exponential': weight = exp(-centroid * temperature)
            outlier_method: Outlier detection method:
                - 'none': No outlier detection
                - 'threshold': Outlier if |centroid - mean| > outlier_threshold
                - 'std': Outlier if centroid outside mean ± outlier_num_std * std
            outlier_threshold: Threshold value for 'threshold' method
            outlier_num_std: Number of standard deviations for 'std' method
            outlier_penalty: Weight assigned to outliers (negative = penalty)
            temperature: Temperature for exponential weight method
            normalize_for_grouping: Whether to normalize answers before grouping
            cache_dir: Directory to load/save centroid cache
        """
        self.top_percent = top_percent
        self.bottom_percent = bottom_percent
        self.consecutive_low_threshold = consecutive_low_threshold
        self.centroid_method = centroid_method
        self.weight_method = weight_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.outlier_num_std = outlier_num_std
        self.outlier_penalty = outlier_penalty
        self.temperature = temperature
        self.normalize_for_grouping = normalize_for_grouping
        self.cache_dir = cache_dir
        
        self._centroid_cache = {}
        self._cache_loaded = False

    def _compute_entropy_thresholds(self, entropy_sequence: List[Dict]) -> Tuple[float, float]:
        """Compute entropy thresholds for top N% and bottom M%."""
        # Convert to float to handle decimal.Decimal values
        entropies = [float(item['entropy']) for item in entropy_sequence if item.get('entropy') is not None]
        if not entropies:
            return float('inf'), float('-inf')
        high_percentile = 100.0 - self.top_percent
        high_threshold = np.percentile(entropies, high_percentile)
        low_threshold = np.percentile(entropies, self.bottom_percent)
        return high_threshold, low_threshold

    def _compute_high_entropy_phases(self, entropy_sequence: List[Dict]) -> List[Tuple[int, int]]:
        """Compute high-entropy phases (HEPs)."""
        high_threshold, low_threshold = self._compute_entropy_thresholds(entropy_sequence)
        hep_events = []
        in_hep = False
        start = 0
        duration = 0
        low_count = 0

        for i, item in enumerate(entropy_sequence):
            entropy = item.get('entropy')
            if entropy is None:
                continue
            if not in_hep:
                if entropy >= high_threshold:
                    in_hep = True
                    start = i
                    duration = 1
                    low_count = 0
            else:
                duration += 1
                if entropy <= low_threshold:
                    low_count += 1
                else:
                    low_count = 0
                if low_count >= self.consecutive_low_threshold:
                    final_duration = duration - self.consecutive_low_threshold
                    if final_duration > 0:
                        hep_events.append((start, final_duration))
                    in_hep = False
                    duration = 0
                    low_count = 0

        if in_hep and duration > 0:
            hep_events.append((start, duration))
        return hep_events

    def _compute_centroid(self, hep_events: List[Tuple[int, int]], trajectory_length: int) -> Optional[float]:
        """Compute centroid using weighted_average_center method."""
        if not hep_events or trajectory_length <= 0:
            return None
        total_moment = 0.0
        total_weight = 0.0
        for start_pos, duration in hep_events:
            weight = duration
            position = start_pos + duration / 2.0
            total_moment += weight * position
            total_weight += weight
        if total_weight == 0:
            return None
        return (total_moment / total_weight) / trajectory_length

    def _normalize_answer_key(self, answer: Optional[str]) -> str:
        """Normalize answer for grouping purposes."""
        if answer is None:
            return '__none__'
        if self.normalize_for_grouping:
            key = str(answer).strip().lower()
            key = ' '.join(key.split())
            return key if key else '__empty__'
        return str(answer)

    def _load_cache(self):
        """Load centroid cache if available (with fallback to analysis results)."""
        if self._cache_loaded or not self.cache_dir:
            return
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from centroid.io import load_trajectory_centroid_cache_with_fallback
            cache = load_trajectory_centroid_cache_with_fallback(
                self.cache_dir, self.top_percent, self.bottom_percent,
                self.consecutive_low_threshold, self.centroid_method
            )
            if cache and 'trajectories' in cache:
                self._centroid_cache = cache['trajectories']
                source = cache.get('parameters', {}).get('source', 'cache')
                print(f"  Loaded {len(self._centroid_cache)} centroids from {source} (method={self.centroid_method})")
        except Exception as e:
            print(f"  Warning: Failed to load centroid cache: {e}")
        self._cache_loaded = True

    def _save_cache(self, trajectory_data: Dict[str, Dict]):
        """Save computed centroids to cache."""
        if not self.cache_dir:
            return
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from centroid.io import save_trajectory_centroid_cache
            save_trajectory_centroid_cache(
                self.cache_dir, self.top_percent, self.bottom_percent,
                self.consecutive_low_threshold, trajectory_data, self.centroid_method
            )
        except Exception as e:
            print(f"  Warning: Failed to save centroid cache: {e}")

    def compute_all_centroids(
        self,
        trajectories: List[Dict],
        entropy_results: List[Dict],
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict[str, Dict]:
        """
        Compute centroids for all trajectories (with caching).
        
        Supports two centroid computation methods:
        - 'hep': High-Entropy Phase based (requires top/bottom/consecutive params)
        - 'raw_entropy': Raw entropy weighted (simpler, no HEP params needed)
        """
        self._load_cache()
        
        # Check if all trajectories are already cached
        traj_ids = [traj.get('id') for traj in trajectories]
        uncached_ids = [tid for tid in traj_ids if tid not in self._centroid_cache]
        
        if not uncached_ids:
            # All trajectories already cached, no need to compute anything
            return self._centroid_cache
        
        # Create entropy lookup only for uncached trajectories
        entropy_by_id = {item['id']: item for item in entropy_results if item['id'] in uncached_ids}
        cache_updated = False
        
        # Try to load evaluation cache for extracted answers (fallback)
        # Only load if we have uncached trajectories
        eval_cache = {}
        if self.cache_dir and uncached_ids:
            try:
                import os
                eval_cache_path = os.path.join(self.cache_dir, 'evaluation_cache.json')
                if os.path.exists(eval_cache_path):
                    import json
                    with open(eval_cache_path, 'r') as f:
                        eval_data = json.load(f)
                    eval_cache = eval_data.get('trajectories', {})
            except Exception:
                pass  # Ignore errors, will extract from text

        for traj in trajectories:
            traj_id = traj.get('id')
            traj_id_str = str(traj_id) if traj_id is not None else ''
            if traj_id in self._centroid_cache:
                continue
            cache_updated = True
            entropy_data = entropy_by_id.get(traj_id, {})
            entropy_sequence = entropy_data.get('entropy_sequence', [])

            # Get extracted answer: try eval cache first, then extract from text
            extracted_answer = None
            if traj_id in eval_cache:
                extracted_answer = eval_cache[traj_id].get('extracted_answer')
            if extracted_answer is None:
                generated_text = traj['sample'].get('generated_text', '')
                if generated_text:  # Only try extraction if text exists
                    extracted_answer = extract_answer_fn(generated_text)

            if not entropy_sequence:
                self._centroid_cache[traj_id] = {
                    'centroid': None, 'num_heps': 0, 'trajectory_length': 0,
                    'centroid_method': self.centroid_method,
                    'original_id': str(traj.get('original_id', traj_id_str.split('_traj_')[0] if '_traj_' in traj_id_str else traj_id_str)),
                    'extracted_answer': str(extracted_answer) if extracted_answer is not None else None
                }
                continue

            trajectory_length = len(entropy_sequence)

            # Compute centroid based on method
            if self.centroid_method == 'raw_entropy':
                # Use raw entropy weighted centroid
                centroid = compute_raw_entropy_centroid(entropy_sequence, trajectory_length)
                num_heps = 0  # No HEPs in raw entropy method
            else:
                # Default: Use HEP-based centroid
                hep_events = self._compute_high_entropy_phases(entropy_sequence)
                centroid = self._compute_centroid(hep_events, trajectory_length)
                num_heps = len(hep_events)

            self._centroid_cache[traj_id] = {
                'centroid': centroid,
                'num_heps': num_heps,
                'trajectory_length': trajectory_length,
                'centroid_method': self.centroid_method,
                'original_id': str(traj.get('original_id', traj_id_str.split('_traj_')[0] if '_traj_' in traj_id_str else traj_id_str)),
                'extracted_answer': str(extracted_answer) if extracted_answer is not None else None
            }

        if cache_updated and self.cache_dir:
            self._save_cache(self._centroid_cache)
        return self._centroid_cache

    def select_answer(
        self,
        trajectories: List[Dict],
        trajectory_selector,  # Not used
        entropy_results: List[Dict],
        step_divisions: List[Dict],  # Not used
        auxiliary_scores: List[Dict],  # Not used
        extract_answer_fn: Callable[[str], Optional[str]]
    ) -> Dict:
        """
        Select answer using centroid-weighted voting.

        Args:
            trajectories: List of trajectory dictionaries
            trajectory_selector: Not used (kept for interface compatibility)
            entropy_results: Full entropy results (needed for centroid computation)
            step_divisions: Not used
            auxiliary_scores: Not used
            extract_answer_fn: Function to extract answer from text

        Returns:
            Selection result with trajectory that produced winning answer
        """
        if not trajectories:
            return {
                'selected_trajectory': None,
                'selection_method': 'centroid_weighted_voting',
                'selection_metadata': {
                    'num_trajectories': 0,
                    'selection_result': 'no_trajectories'
                }
            }

        if len(trajectories) == 1:
            return {
                'selected_trajectory': trajectories[0],
                'selection_method': 'centroid_weighted_voting',
                'selection_metadata': {
                    'num_trajectories': 1,
                    'selection_result': 'single_trajectory'
                }
            }

        # Compute all centroids
        centroid_data = self.compute_all_centroids(trajectories, entropy_results, extract_answer_fn)

        # Collect trajectories with valid centroids
        trajectory_info = []
        for traj in trajectories:
            traj_id = traj.get('id')
            if traj_id in centroid_data:
                data = centroid_data[traj_id]
                if data['centroid'] is not None:
                    trajectory_info.append({
                        'trajectory': traj,
                        'centroid': data['centroid'],
                        'num_heps': data['num_heps'],
                        'trajectory_id': traj_id,
                        'extracted_answer': data['extracted_answer']
                    })

        if not trajectory_info:
            # Fallback to first trajectory
            return {
                'selected_trajectory': trajectories[0],
                'selection_method': 'centroid_weighted_voting',
                'selection_metadata': {
                    'num_trajectories': len(trajectories),
                    'selection_result': 'no_valid_centroids',
                    'fallback': 'first_trajectory'
                }
            }

        # Extract centroids for outlier detection and weighting
        centroids = [ti['centroid'] for ti in trajectory_info]

        # Detect outliers
        is_outlier, outlier_info = detect_outliers(
            centroids,
            method=self.outlier_method,
            threshold=self.outlier_threshold,
            num_std=self.outlier_num_std
        )

        # Compute weights
        weights, weight_info = compute_centroid_weights(
            centroids,
            is_outlier,
            weight_method=self.weight_method,
            outlier_penalty=self.outlier_penalty,
            temperature=self.temperature
        )

        # Aggregate weighted votes for each answer
        answer_weights = {}
        answer_to_trajectories = {}

        for ti, weight, outlier in zip(trajectory_info, weights, is_outlier):
            answer = ti['extracted_answer']
            answer_key = self._normalize_answer_key(answer)

            if answer_key not in answer_weights:
                answer_weights[answer_key] = 0.0
                answer_to_trajectories[answer_key] = []

            answer_weights[answer_key] += weight
            answer_to_trajectories[answer_key].append({
                **ti,
                'weight': weight,
                'is_outlier': outlier
            })

        # Find winning answer (highest total weight)
        if not answer_weights:
            return {
                'selected_trajectory': trajectories[0],
                'selection_method': 'centroid_weighted_voting',
                'selection_metadata': {
                    'num_trajectories': len(trajectories),
                    'selection_result': 'no_valid_answers'
                }
            }

        winning_answer_key = max(answer_weights.keys(), key=lambda k: answer_weights[k])
        winning_weight = answer_weights[winning_answer_key]

        # Among winning trajectories, select the one with smallest centroid
        winning_trajectories = answer_to_trajectories[winning_answer_key]
        best_winning_traj = min(winning_trajectories, key=lambda x: x['centroid'])

        # Prepare statistics
        centroid_stats = {
            'mean': float(np.mean(centroids)),
            'std': float(np.std(centroids)),
            'min': float(np.min(centroids)),
            'max': float(np.max(centroids))
        }

        vote_distribution = {
            key: {
                'total_weight': float(weight),
                'num_trajectories': len(answer_to_trajectories[key]),
                'num_outliers': sum(1 for t in answer_to_trajectories[key] if t['is_outlier'])
            }
            for key, weight in answer_weights.items()
        }

        return {
            'selected_trajectory': best_winning_traj['trajectory'],
            'selection_method': 'centroid_weighted_voting',
            'selection_metadata': {
                'num_trajectories': len(trajectories),
                'num_valid_centroids': len(trajectory_info),
                'num_outliers': outlier_info.get('num_outliers', 0),
                'winning_answer': best_winning_traj['extracted_answer'],
                'winning_answer_key': winning_answer_key,
                'winning_weight': float(winning_weight),
                'winning_centroid': best_winning_traj['centroid'],
                'vote_distribution': vote_distribution,
                'centroid_statistics': centroid_stats,
                'outlier_detection': outlier_info,
                'weight_calculation': weight_info,
                'parameters': {
                    'top_percent': self.top_percent,
                    'bottom_percent': self.bottom_percent,
                    'consecutive_low_threshold': self.consecutive_low_threshold,
                    'weight_method': self.weight_method,
                    'outlier_method': self.outlier_method,
                    'outlier_threshold': self.outlier_threshold,
                    'outlier_num_std': self.outlier_num_std,
                    'outlier_penalty': self.outlier_penalty
                }
            }
        }

    def get_name(self) -> str:
        method_str = "_raw" if self.centroid_method == 'raw_entropy' else ""
        outlier_str = f"_{self.outlier_method}" if self.outlier_method != 'none' else ""
        return f"centroid_weighted{method_str}_{self.weight_method}{outlier_str}"


def create_answer_selector(
    selection_method: str = "best_of_n",
    **kwargs
) -> AnswerSelector:
    """
    Factory function to create answer selector.

    Args:
        selection_method: Selection method:
            - "best_of_n": Traditional Best-of-N (select trajectory with highest score)
            - "majority_voting": Pure majority voting (equal weight per trajectory)
            - "llm_majority_voting": LLM-filtered majority voting for AMO description
            - "entropy_centroid": Select trajectory with earliest entropy centroid
            - "centroid_voting": Vote among trajectories with lowest centroids
            - "centroid_weighted_voting": Weighted voting based on centroid position
            - "random": Random selection (useful as baseline)
        **kwargs: Additional arguments:
            - normalize_for_grouping (bool): For majority/LLM majority, normalize answers
            - top_percent (float): For entropy_centroid/centroid_voting, high entropy threshold (default: 5.0)
            - bottom_percent (float): For entropy_centroid/centroid_voting, low entropy threshold (default: 50.0)
            - consecutive_low_threshold (int): For entropy_centroid/centroid_voting, consecutive low tokens to end HEP (default: 3)
            - outlier_threshold (float): For entropy_centroid, exclude centroids below mean - threshold (default: 0.1)
            - centroid_select_percent (float): For centroid_voting, select bottom X% by centroid (default: 30.0)
            - cache_dir (str): For centroid_voting/centroid_weighted_voting, directory to cache centroid computations
            - seed (int): For random selection, random seed for reproducibility
            - num_samples (int): For random selection, number of samples for averaging (default: 1)

            For centroid_weighted_voting:
            - weight_method (str): 'linear', 'inverse', or 'exponential' (default: 'linear')
            - outlier_method (str): 'none', 'threshold', or 'std' (default: 'std')
            - outlier_threshold (float): Threshold for 'threshold' outlier method (default: 0.1)
            - outlier_num_std (float): Number of std deviations for 'std' outlier method (default: 1.0)
            - outlier_penalty (float): Weight penalty for outliers, can be negative (default: -0.5)

    Returns:
        AnswerSelector instance
    """
    if selection_method == "best_of_n":
        return BestOfNAnswerSelector()

    elif selection_method == "majority_voting":
        # Pure majority voting - no auxiliary scores needed
        normalize_for_grouping = kwargs.get('normalize_for_grouping', True)
        cache_dir = kwargs.get('cache_dir', None)
        return MajorityVotingAnswerSelector(
            normalize_for_grouping=normalize_for_grouping,
            cache_dir=cache_dir
        )

    elif selection_method == "llm_majority_voting":
        normalize_for_grouping = kwargs.get('normalize_for_grouping', True)
        cache_dir = kwargs.get('cache_dir', None)
        return LLMMajorityVotingAnswerSelector(
            normalize_for_grouping=normalize_for_grouping,
            cache_dir=cache_dir
        )

    elif selection_method == "entropy_centroid":
        # Entropy centroid based selection - no auxiliary scores needed
        return EntropyCentroidAnswerSelector(
            top_percent=kwargs.get('top_percent', 5.0),
            bottom_percent=kwargs.get('bottom_percent', 50.0),
            consecutive_low_threshold=kwargs.get('consecutive_low_threshold', 3),
            outlier_threshold=kwargs.get('outlier_threshold', 0.1),
            outlier_method=kwargs.get('outlier_method', 'below_mean_threshold'),
            outlier_num_std=kwargs.get('outlier_num_std', 1.0),
            normalize_for_grouping=kwargs.get('normalize_for_grouping', True)
        )

    elif selection_method == "centroid_voting":
        # Centroid-based voting - select bottom X% centroids for voting
        return CentroidVotingAnswerSelector(
            top_percent=kwargs.get('top_percent', 5.0),
            bottom_percent=kwargs.get('bottom_percent', 50.0),
            consecutive_low_threshold=kwargs.get('consecutive_low_threshold', 3),
            centroid_select_percent=kwargs.get('centroid_select_percent', 30.0),
            centroid_method=kwargs.get('centroid_method', 'hep'),
            normalize_for_grouping=kwargs.get('normalize_for_grouping', True),
            cache_dir=kwargs.get('cache_dir', None)
        )

    elif selection_method == "centroid_weighted_voting":
        # Centroid-weighted voting - weight by centroid position with outlier detection
        return CentroidWeightedVotingAnswerSelector(
            top_percent=kwargs.get('top_percent', 5.0),
            bottom_percent=kwargs.get('bottom_percent', 50.0),
            consecutive_low_threshold=kwargs.get('consecutive_low_threshold', 3),
            centroid_method=kwargs.get('centroid_method', 'hep'),
            weight_method=kwargs.get('weight_method', 'linear'),
            outlier_method=kwargs.get('outlier_method', 'std'),
            outlier_threshold=kwargs.get('outlier_threshold', 0.1),
            outlier_num_std=kwargs.get('outlier_num_std', 1.0),
            outlier_penalty=kwargs.get('outlier_penalty', -0.5),
            temperature=kwargs.get('temperature', 1.0),
            normalize_for_grouping=kwargs.get('normalize_for_grouping', True),
            cache_dir=kwargs.get('cache_dir', None)
        )

    elif selection_method == "random":
        # Random selection - useful as baseline
        seed = kwargs.get('seed', None)
        num_samples = kwargs.get('num_samples', 1)
        return RandomAnswerSelector(seed=seed, num_samples=num_samples)

    else:
        raise ValueError(
            f"Unknown answer selection method: {selection_method}. "
            f"Supported methods: 'best_of_n', 'majority_voting', 'llm_majority_voting', 'entropy_centroid', 'centroid_voting', 'centroid_weighted_voting', 'random'"
        )
