"""
Statistics computation for entropy analysis.
"""

from typing import Dict, List
from collections import Counter
import pandas as pd


class StatisticsAnalyzer:
    """
    Analyze and compute statistics from entropy results.
    Uses percentile-based high-entropy token identification instead of fixed threshold.
    """

    def __init__(self, high_entropy_percentile: float = 5.0):
        """
        Initialize statistics analyzer.

        Args:
            high_entropy_percentile: Percentile threshold to classify high-entropy tokens
                                     (default: 5.0 means top 5% are considered high-entropy)
        """
        self.high_entropy_percentile = high_entropy_percentile

    def analyze_results(self, results: List[Dict]) -> Dict:
        """
        Analyze all results and compute aggregate statistics.
        Uses percentile-based definition for high-entropy tokens (top 5% by default).

        Args:
            results: List of generation results with entropy information

        Returns:
            Dictionary of aggregate statistics
        """
        # Collect all entropy values and tokens
        all_entropies = []
        all_tokens = []
        high_entropy_tokens = []  # Top percentile tokens across all samples
        problem_stats = []

        # Track per-sample high-entropy tokens based on color coding
        colored_high_entropy_tokens = []  # Top 5% within each sample

        for result in results:
            entropy_seq = result.get("entropy_sequence", [])
            sample_id = result.get("id", "unknown")

            # Extract entropy values and tokens
            for item in entropy_seq:
                if item.get("entropy") is not None:
                    entropy = item["entropy"]
                    token = item["token"]
                    color = item.get("color", "black")

                    all_entropies.append(entropy)
                    all_tokens.append(token)

                    # Track tokens marked as high-entropy by color (top 5% within sample)
                    if color in ["red", "pink", "purple"]:
                        colored_high_entropy_tokens.append({
                            "sample_id": sample_id,
                            "token": token,
                            "entropy": entropy,
                            "position": item["position"],
                            "color": color,
                            "percentile": item.get("percentile", 0),
                        })

            # Per-problem statistics
            stats = result.get("statistics", {})
            if stats:
                # Count high-entropy tokens in this sample (based on color)
                sample_high_entropy_count = sum(
                    1 for item in entropy_seq
                    if item.get("color") in ["red", "pink", "purple"]
                )

                problem_stats.append({
                    "id": result.get("id"),
                    "total_tokens": stats.get("total_tokens", 0),
                    "valid_tokens": stats.get("valid_tokens", 0),
                    "avg_entropy": stats.get("avg_entropy"),
                    "max_entropy": stats.get("max_entropy"),
                    "min_entropy": stats.get("min_entropy"),
                    "std_entropy": stats.get("std_entropy"),
                    "high_entropy_count": sample_high_entropy_count,
                    "high_entropy_ratio": sample_high_entropy_count / stats.get("valid_tokens", 1) if stats.get("valid_tokens", 0) > 0 else 0,
                })

        # Also compute global high-entropy tokens (top percentile across all samples)
        if all_entropies:
            all_entropies_sorted = sorted(all_entropies, reverse=True)
            n_total = len(all_entropies_sorted)
            threshold_idx = max(1, int(n_total * self.high_entropy_percentile / 100))
            global_high_entropy_threshold = all_entropies_sorted[threshold_idx - 1]

            # Find tokens above global threshold
            for result in results:
                for item in result.get("entropy_sequence", []):
                    if item.get("entropy") is not None and item["entropy"] >= global_high_entropy_threshold:
                        high_entropy_tokens.append({
                            "token": item["token"],
                            "entropy": item["entropy"],
                            "position": item["position"],
                        })

        # Compute aggregate statistics
        aggregate_stats = self._compute_aggregate_stats(
            all_entropies, all_tokens, high_entropy_tokens, colored_high_entropy_tokens
        )

        return {
            "aggregate_statistics": aggregate_stats,
            "per_problem_statistics": problem_stats,
            "high_entropy_tokens": high_entropy_tokens,  # Global top percentile
            "colored_high_entropy_tokens": colored_high_entropy_tokens,  # Per-sample top 5%
        }

    def _compute_aggregate_stats(
        self,
        all_entropies: List[float],
        all_tokens: List[str],
        high_entropy_tokens: List[Dict],
        colored_high_entropy_tokens: List[Dict],
    ) -> Dict:
        """
        Compute aggregate statistics.

        Args:
            all_entropies: List of all entropy values
            all_tokens: List of all tokens
            high_entropy_tokens: List of globally high-entropy token information
            colored_high_entropy_tokens: List of per-sample high-entropy tokens (top 5%)

        Returns:
            Dictionary of aggregate statistics
        """
        if not all_entropies:
            return {}

        # Basic statistics
        avg_entropy = sum(all_entropies) / len(all_entropies)
        max_entropy = max(all_entropies)
        min_entropy = min(all_entropies)

        variance = sum((e - avg_entropy) ** 2 for e in all_entropies) / len(all_entropies)
        std_entropy = variance ** 0.5

        # High-entropy token statistics (global)
        high_entropy_count = len(high_entropy_tokens)
        high_entropy_ratio = high_entropy_count / len(all_entropies)

        # Token frequency analysis
        token_freq = Counter(all_tokens)
        most_common_tokens = token_freq.most_common(20)

        # High-entropy token frequency (global)
        high_entropy_token_list = [item["token"] for item in high_entropy_tokens]
        high_entropy_freq = Counter(high_entropy_token_list)
        most_common_high_entropy = high_entropy_freq.most_common(20)

        # Colored (per-sample) high-entropy token frequency
        colored_token_list = [item["token"] for item in colored_high_entropy_tokens]
        colored_freq = Counter(colored_token_list)
        most_common_colored = colored_freq.most_common(20)

        # Color distribution
        color_counts = Counter(item["color"] for item in colored_high_entropy_tokens)

        return {
            "total_tokens": len(all_entropies),
            "avg_entropy": avg_entropy,
            "max_entropy": max_entropy,
            "min_entropy": min_entropy,
            "std_entropy": std_entropy,
            "high_entropy_count": high_entropy_count,
            "high_entropy_ratio": high_entropy_ratio,
            "most_common_tokens": most_common_tokens,
            "most_common_high_entropy_tokens": most_common_high_entropy,
            "most_common_colored_high_entropy_tokens": most_common_colored,
            "colored_high_entropy_count": len(colored_high_entropy_tokens),
            "colored_high_entropy_ratio": len(colored_high_entropy_tokens) / len(all_entropies),
            "color_distribution": dict(color_counts),
        }

    def create_summary_dataframe(self, per_problem_stats: List[Dict]) -> pd.DataFrame:
        """
        Create a pandas DataFrame from per-problem statistics.

        Args:
            per_problem_stats: List of per-problem statistics

        Returns:
            Pandas DataFrame
        """
        return pd.DataFrame(per_problem_stats)

    def create_high_entropy_dataframe(self, high_entropy_tokens: List[Dict]) -> pd.DataFrame:
        """
        Create a pandas DataFrame from high-entropy tokens.

        Args:
            high_entropy_tokens: List of high-entropy token information

        Returns:
            Pandas DataFrame
        """
        return pd.DataFrame(high_entropy_tokens)


def create_statistics_analyzer(config: Dict) -> StatisticsAnalyzer:
    """
    Create a statistics analyzer from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        StatisticsAnalyzer instance
    """
    stats_config = config.get("statistics", {})
    # Use percentile instead of threshold
    percentile = stats_config.get("high_entropy_percentile", 5.0)

    return StatisticsAnalyzer(high_entropy_percentile=percentile)
