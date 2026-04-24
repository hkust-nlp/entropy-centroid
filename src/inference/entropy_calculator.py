"""
Token entropy calculator for vLLM outputs.
"""

import math
from typing import Dict, List, Optional, Tuple


class EntropyCalculator:
    """
    Calculate token-level entropy from vLLM logprobs output.
    """

    def __init__(self, top_k: int = 20):
        """
        Initialize entropy calculator.

        Args:
            top_k: Number of top candidates to use for entropy calculation
        """
        self.top_k = top_k

    def calculate_entropy(self, logprobs_dict: Dict[int, float]) -> float:
        """
        Calculate entropy from logprobs dictionary.

        Args:
            logprobs_dict: Dictionary mapping token_id to logprob

        Returns:
            Entropy value (in nats if using natural log, or bits if using log2)
        """
        if not logprobs_dict:
            return 0.0

        # Get top-k items by logprob
        top_k_items = sorted(
            logprobs_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[: self.top_k]

        # Convert logprobs to probabilities
        logprobs = [logprob for _, logprob in top_k_items]
        probs = [math.exp(logprob) for logprob in logprobs]

        # Normalize probabilities
        prob_sum = sum(probs)
        if prob_sum == 0:
            return 0.0

        normalized_probs = [p / prob_sum for p in probs]

        # Calculate entropy: H = -sum(p * log(p))
        entropy = 0.0
        for p in normalized_probs:
            if p > 0:
                entropy -= p * math.log(p)

        return entropy

    def process_sequence_logprobs(
        self,
        sequence_logprobs: List[Optional[Dict[int, float]]],
        tokens: List[str],
        token_ids: List[int],
    ) -> List[Dict]:
        """
        Process logprobs for an entire sequence.

        Args:
            sequence_logprobs: List of logprobs dicts for each token position
            tokens: List of generated tokens (strings)
            token_ids: List of generated token IDs

        Returns:
            List of entropy information for each token
        """
        entropy_sequence = []

        for position, (logprobs_dict, token, token_id) in enumerate(
            zip(sequence_logprobs, tokens, token_ids)
        ):
            if logprobs_dict is None:
                # No logprobs available for this position (e.g., first token)
                entropy_info = {
                    "token": token,
                    "token_id": token_id,
                    "position": position,
                    "entropy": None,
                    "top_k_tokens": [],
                    "top_k_probs": [],
                }
            else:
                # Calculate entropy
                entropy = self.calculate_entropy(logprobs_dict)

                # Get top-k tokens and their probabilities
                top_k_items = sorted(
                    logprobs_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[: self.top_k]

                top_k_token_ids = [token_id for token_id, _ in top_k_items]
                top_k_logprobs = [logprob for _, logprob in top_k_items]
                top_k_probs = [math.exp(lp) for lp in top_k_logprobs]

                # Normalize probabilities
                prob_sum = sum(top_k_probs)
                if prob_sum > 0:
                    top_k_probs = [p / prob_sum for p in top_k_probs]

                entropy_info = {
                    "token": token,
                    "token_id": token_id,
                    "position": position,
                    "entropy": entropy,
                    "top_k_token_ids": top_k_token_ids,
                    "top_k_probs": top_k_probs,
                }

            entropy_sequence.append(entropy_info)

        return entropy_sequence

    def classify_tokens_by_percentile(
        self, entropy_sequence: List[Dict]
    ) -> List[Dict]:
        """
        Classify tokens by entropy percentile within the sequence.

        Colors:
        - Top 1%: red (highest uncertainty)
        - 1-3%: pink
        - 3-5%: purple
        - Rest: black (normal)

        Args:
            entropy_sequence: List of entropy information dicts

        Returns:
            List of entropy information with added 'percentile' and 'color' fields
        """
        # Extract valid entropies with their indices
        valid_items = [
            (i, item["entropy"])
            for i, item in enumerate(entropy_sequence)
            if item["entropy"] is not None
        ]

        if not valid_items:
            # No valid entropies, return original with default values
            for item in entropy_sequence:
                item["percentile"] = None
                item["color"] = "black"
            return entropy_sequence

        # Sort by entropy (descending) to get percentiles
        sorted_items = sorted(valid_items, key=lambda x: x[1], reverse=True)
        n_valid = len(sorted_items)

        # Calculate percentile thresholds
        top_1_percent_idx = max(1, int(n_valid * 0.01))
        top_3_percent_idx = max(1, int(n_valid * 0.03))
        top_5_percent_idx = max(1, int(n_valid * 0.05))

        # Create percentile ranking
        percentile_map = {}
        for rank, (idx, entropy_val) in enumerate(sorted_items):
            percentile = (rank / n_valid) * 100  # 0-100 scale
            percentile_map[idx] = percentile

        # Assign colors based on percentile
        for i, item in enumerate(entropy_sequence):
            if item["entropy"] is None:
                item["percentile"] = None
                item["color"] = "black"
            else:
                percentile = percentile_map[i]
                item["percentile"] = percentile

                # Determine color based on percentile
                if percentile < 1.0:  # Top 1%
                    item["color"] = "red"
                elif percentile < 3.0:  # 1-3%
                    item["color"] = "pink"
                elif percentile < 5.0:  # 3-5%
                    item["color"] = "purple"
                else:
                    item["color"] = "black"

        return entropy_sequence

    def calculate_statistics(self, entropy_sequence: List[Dict]) -> Dict:
        """
        Calculate statistics from entropy sequence.

        Args:
            entropy_sequence: List of entropy information dicts

        Returns:
            Dictionary of statistics
        """
        # Filter out None entropy values
        entropies = [
            item["entropy"]
            for item in entropy_sequence
            if item["entropy"] is not None
        ]

        if not entropies:
            return {
                "total_tokens": len(entropy_sequence),
                "valid_tokens": 0,
                "avg_entropy": None,
                "max_entropy": None,
                "min_entropy": None,
                "std_entropy": None,
            }

        # Calculate statistics
        avg_entropy = sum(entropies) / len(entropies)
        max_entropy = max(entropies)
        min_entropy = min(entropies)

        # Calculate standard deviation
        variance = sum((e - avg_entropy) ** 2 for e in entropies) / len(entropies)
        std_entropy = math.sqrt(variance)

        return {
            "total_tokens": len(entropy_sequence),
            "valid_tokens": len(entropies),
            "avg_entropy": avg_entropy,
            "max_entropy": max_entropy,
            "min_entropy": min_entropy,
            "std_entropy": std_entropy,
        }


def create_entropy_calculator(config: Dict) -> EntropyCalculator:
    """
    Create an entropy calculator from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        EntropyCalculator instance
    """
    entropy_config = config.get("entropy", {})
    top_k = entropy_config.get("top_k", 20)

    return EntropyCalculator(top_k=top_k)
