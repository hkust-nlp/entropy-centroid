"""
Trajectory selection strategies for Best-of-N sampling.

Implements extensible strategy pattern for selecting the best trajectory
from multiple candidate trajectories per problem.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class TrajectorySelectionStrategy(ABC):
    """Abstract base class for trajectory selection strategies."""

    @abstractmethod
    def score_trajectory(self, trajectory: Dict) -> float:
        """
        Calculate score for ranking a trajectory.

        Args:
            trajectory: Trajectory dictionary with:
                - entropy_data: Entropy sequence
                - step_data: Step divisions

        Returns:
            Score (higher is better)
        """

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name for reporting."""


class EntropyMeanStrategy(TrajectorySelectionStrategy):
    """
    Select trajectory with the lowest average token entropy.

    Lower entropy means higher model confidence on average.
    """

    def score_trajectory(self, trajectory: Dict) -> float:
        entropy_data = trajectory.get("entropy_data", {})
        entropy_sequence = entropy_data.get("entropy_sequence", [])
        values = [item.get("entropy") for item in entropy_sequence if item.get("entropy") is not None]
        if not values:
            return 0.0
        return float(-np.mean(values))

    def get_strategy_name(self) -> str:
        return "entropy_mean"


class EarlyHighEntropyCentroidStrategy(TrajectorySelectionStrategy):
    """
    Select trajectory whose high-entropy mass appears earlier.

    This approximates "uncertainty concentrated earlier, then stabilized".
    """

    def __init__(self, entropy_percentile: float = 30.0):
        self.entropy_percentile = entropy_percentile

    def score_trajectory(self, trajectory: Dict) -> float:
        entropy_data = trajectory.get("entropy_data", {})
        entropy_sequence = entropy_data.get("entropy_sequence", [])
        values = [item.get("entropy") for item in entropy_sequence if item.get("entropy") is not None]
        if not values:
            return 0.0

        threshold = float(np.percentile(values, 100.0 - self.entropy_percentile))
        weighted_sum = 0.0
        total_weight = 0.0

        for idx, item in enumerate(entropy_sequence):
            entropy = item.get("entropy")
            if entropy is None or entropy < threshold:
                continue
            entropy = float(entropy)
            weighted_sum += entropy * idx
            total_weight += entropy

        if total_weight == 0.0:
            return 0.0

        centroid = weighted_sum / total_weight
        normalized_centroid = centroid / max(1, len(entropy_sequence))
        return float(1.0 - normalized_centroid)

    def get_strategy_name(self) -> str:
        return f"early_high_entropy_centroid_p{int(self.entropy_percentile)}"


class EntropyMinStrategy(TrajectorySelectionStrategy):
    """
    Select trajectory with the lowest minimum token entropy.

    Useful as a conservative proxy for strong local confidence.
    """

    def score_trajectory(self, trajectory: Dict) -> float:
        entropy_data = trajectory.get("entropy_data", {})
        entropy_sequence = entropy_data.get("entropy_sequence", [])
        values = [item.get("entropy") for item in entropy_sequence if item.get("entropy") is not None]
        if not values:
            return 0.0
        return float(-np.min(values))

    def get_strategy_name(self) -> str:
        return "entropy_min"


class TrajectorySelector:
    """Selects best trajectory from multiple candidates using a strategy."""

    def __init__(self, strategy: Optional[TrajectorySelectionStrategy] = None):
        """
        Initialize trajectory selector.

        Args:
            strategy: Selection strategy (default: EntropyMeanStrategy)
        """
        self.strategy = strategy or EntropyMeanStrategy()

    def select_best_trajectory(
        self,
        trajectories: List[Dict],
        entropy_results: List[Dict],
        step_divisions: List[Dict],
        auxiliary_scores: List[Dict],
    ) -> Dict:
        """
        Select best trajectory from candidates.

        Args:
            trajectories: List of trajectory dictionaries with 'id' field
            entropy_results: Full entropy results data
            step_divisions: Full step division data
            auxiliary_scores: Reserved for future strategy extensions

        Returns:
            Dictionary with:
            {
                'selected_trajectory': Dict,
                'selection_score': float,
                'strategy': str,
                'all_scores': List[float]
            }
        """
        if not trajectories:
            return {
                "selected_trajectory": None,
                "selection_score": 0.0,
                "strategy": self.strategy.get_strategy_name(),
                "all_scores": [],
            }

        entropy_by_id = {item["id"]: item for item in entropy_results}
        step_by_id = {item["id"]: item for item in step_divisions}

        scored_trajectories = []
        for traj in trajectories:
            traj_id = traj.get("id")
            trajectory_data = {
                "id": traj_id,
                "entropy_data": entropy_by_id.get(traj_id, {}),
                "step_data": step_by_id.get(traj_id, {}),
                "auxiliary_data": {},
            }
            score = self.strategy.score_trajectory(trajectory_data)
            scored_trajectories.append(
                {
                    "trajectory": traj,
                    "score": score,
                    "trajectory_data": trajectory_data,
                }
            )

        scored_trajectories.sort(key=lambda x: x["score"], reverse=True)
        best = scored_trajectories[0]

        return {
            "selected_trajectory": best["trajectory"],
            "selection_score": best["score"],
            "strategy": self.strategy.get_strategy_name(),
            "all_scores": [st["score"] for st in scored_trajectories],
        }


def create_trajectory_selector(strategy_name: str = "entropy_mean", **kwargs) -> TrajectorySelector:
    """
    Factory function to create trajectory selector with specified strategy.

    Args:
        strategy_name: Name of strategy:
            - "entropy_mean": Select trajectory with lowest mean entropy
            - "early_high_entropy_centroid": Select trajectory with earliest high-entropy centroid
            - "entropy_min": Select trajectory with lowest minimum token entropy
        **kwargs: Additional arguments for strategy:
            - entropy_percentile (float): Percentile for high-entropy centroid strategy (default: 30.0)

    Returns:
        TrajectorySelector instance
    """
    if strategy_name == "entropy_mean":
        strategy = EntropyMeanStrategy()
    elif strategy_name == "early_high_entropy_centroid":
        strategy = EarlyHighEntropyCentroidStrategy(
            entropy_percentile=kwargs.get("entropy_percentile", 30.0)
        )
    elif strategy_name == "entropy_min":
        strategy = EntropyMinStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return TrajectorySelector(strategy)
