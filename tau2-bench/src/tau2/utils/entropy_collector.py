"""
Entropy data collector for tau2-bench agent simulations.

Aggregates per-token entropy data across multiple agent turns in a simulation,
then exports in centroid-compatible format for downstream HEP/centroid analysis.
"""

import math
from typing import Any, Dict, List, Optional

from loguru import logger


class EntropyCalculator:
    """Lightweight entropy calculator for OpenAI logprobs format.

    Adapted from src/inference/entropy_calculator.py for use with
    the OpenAI-compatible logprobs returned by vLLM.
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def calculate_entropy(self, logprobs_dict: Dict[int, float]) -> float:
        """Calculate Shannon entropy from a {token_id: logprob} mapping."""
        if not logprobs_dict:
            return 0.0

        top_k_items = sorted(
            logprobs_dict.items(),
            key=lambda x: x[1],
            reverse=True,
        )[: self.top_k]

        probs = [math.exp(lp) for _, lp in top_k_items]
        prob_sum = sum(probs)
        if prob_sum == 0:
            return 0.0

        normalized = [p / prob_sum for p in probs]
        entropy = 0.0
        for p in normalized:
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    def calculate_statistics(self, entropy_sequence: List[Dict]) -> Dict:
        """Calculate summary statistics from an entropy sequence."""
        entropies = [
            item["entropy"]
            for item in entropy_sequence
            if item.get("entropy") is not None
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

        avg = sum(entropies) / len(entropies)
        variance = sum((e - avg) ** 2 for e in entropies) / len(entropies)
        return {
            "total_tokens": len(entropy_sequence),
            "valid_tokens": len(entropies),
            "avg_entropy": avg,
            "max_entropy": max(entropies),
            "min_entropy": min(entropies),
            "std_entropy": math.sqrt(variance),
        }

    def classify_tokens_by_percentile(self, entropy_sequence: List[Dict]) -> List[Dict]:
        """Add percentile and color classification to each token entry."""
        valid_items = [
            (i, item["entropy"])
            for i, item in enumerate(entropy_sequence)
            if item.get("entropy") is not None
        ]

        if not valid_items:
            for item in entropy_sequence:
                item["percentile"] = None
                item["color"] = "black"
            return entropy_sequence

        sorted_items = sorted(valid_items, key=lambda x: x[1], reverse=True)
        n_valid = len(sorted_items)

        percentile_map = {}
        for rank, (idx, _) in enumerate(sorted_items):
            percentile_map[idx] = (rank / n_valid) * 100

        for i, item in enumerate(entropy_sequence):
            if item.get("entropy") is None:
                item["percentile"] = None
                item["color"] = "black"
            else:
                pct = percentile_map[i]
                item["percentile"] = pct
                if pct < 1.0:
                    item["color"] = "red"
                elif pct < 3.0:
                    item["color"] = "pink"
                elif pct < 5.0:
                    item["color"] = "purple"
                else:
                    item["color"] = "black"

        return entropy_sequence


def extract_entropy_from_logprobs(
    logprobs_content: list,
    entropy_calculator: EntropyCalculator,
) -> List[Dict]:
    """Convert OpenAI-format logprobs to centroid entropy sequence entries.

    Args:
        logprobs_content: The `response.choices[0].logprobs.content` list
            from a litellm/OpenAI response (each entry has token, logprob,
            top_logprobs).
        entropy_calculator: EntropyCalculator instance.

    Returns:
        List of per-token entropy dicts in centroid format.
    """
    entropy_sequence = []
    for position, token_logprob in enumerate(logprobs_content):
        token = token_logprob.get("token", "")
        # Try to get token_id; vLLM may include it but it's not standard in OpenAI format
        token_id = token_logprob.get("token_id", 0)

        top_logprobs = token_logprob.get("top_logprobs", [])
        if top_logprobs:
            # Build {index: logprob} dict for entropy calculation
            # Use position-based indices since OpenAI format uses token strings, not IDs
            logprobs_dict = {}
            top_k_token_ids = []
            top_k_probs = []
            for idx, entry in enumerate(top_logprobs):
                lp = entry.get("logprob", 0.0)
                tid = entry.get("token_id", idx)
                logprobs_dict[tid] = lp
                top_k_token_ids.append(tid)
                top_k_probs.append(math.exp(lp))

            # Normalize probs
            prob_sum = sum(top_k_probs)
            if prob_sum > 0:
                top_k_probs = [p / prob_sum for p in top_k_probs]

            entropy = entropy_calculator.calculate_entropy(logprobs_dict)
        else:
            entropy = None
            top_k_token_ids = []
            top_k_probs = []

        entropy_sequence.append({
            "token": token,
            "token_id": token_id,
            "position": position,
            "entropy": entropy,
            "top_k_token_ids": top_k_token_ids,
            "top_k_probs": top_k_probs,
        })

    return entropy_sequence


class EntropyCollector:
    """Collects entropy data across agent turns in a simulation.

    Usage:
        collector = EntropyCollector(entropy_calculator)
        # After each agent turn:
        collector.add_turn(entropy_data)
        # After simulation completes:
        result = collector.to_centroid_format(task_id, problem, ...)
    """

    def __init__(self, entropy_calculator: EntropyCalculator):
        self.turns: List[List[Dict]] = []
        self.entropy_calculator = entropy_calculator

    def add_turn(self, entropy_data: List[Dict]) -> None:
        """Append one agent turn's entropy sequence."""
        if entropy_data:
            self.turns.append(entropy_data)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def total_tokens(self) -> int:
        return sum(len(turn) for turn in self.turns)

    def get_concatenated_sequence(self) -> List[Dict]:
        """Merge all turns into a continuous sequence, re-indexing positions.

        Also runs percentile classification on the full merged sequence.
        """
        result = []
        offset = 0
        for turn in self.turns:
            for entry in turn:
                new_entry = dict(entry)
                new_entry["position"] = offset + entry["position"]
                result.append(new_entry)
            offset += len(turn)

        # Classify across the full concatenated sequence
        if result:
            self.entropy_calculator.classify_tokens_by_percentile(result)

        return result

    def to_centroid_format(
        self,
        task_id: str,
        problem: str,
        solution: Optional[str],
        reward: float,
        generated_text: str,
        domain: str,
        trial: int = 0,
    ) -> Dict[str, Any]:
        """Export entropy data in centroid entropy_results.jsonl format.

        Args:
            task_id: The tau2-bench task ID.
            problem: Task description / user scenario text.
            solution: Ground truth solution (if available).
            reward: Simulation reward (0 or 1).
            generated_text: Concatenated agent response text.
            domain: tau2-bench domain (airline, retail, telecom).
            trial: Trial index (0-based) for multi-trial runs.

        Returns:
            Dict matching centroid's entropy_results.jsonl schema.
        """
        sequence = self.get_concatenated_sequence()
        statistics = self.entropy_calculator.calculate_statistics(sequence)

        # Use task_id_traj_N format for multi-trial, consistent with centroid framework
        entry_id = f"{task_id}_traj_{trial}" if trial > 0 else str(task_id)

        return {
            "id": entry_id,
            "original_id": str(task_id),
            "trajectory_index": trial,
            "problem": problem,
            "solution": solution,
            "source": f"tau2-bench-{domain}",
            "generated_text": generated_text,
            "tokens": [e["token"] for e in sequence],
            "token_ids": [e["token_id"] for e in sequence],
            "entropy_sequence": sequence,
            "statistics": statistics,
            "reward": reward,
        }
