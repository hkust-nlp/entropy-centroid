"""
Step division based on high-entropy tokens for reasoning trajectory analysis.
"""

import re
from typing import Dict, List, Optional, Tuple


class StepDivider:
    """
    Divide reasoning trajectories into steps based on high-entropy tokens.
    """

    # Special tokens to filter from the end of trajectories
    TRAILING_SPECIAL_TOKENS = [
        "<|im_end|>",
        "<|return|>",
        "<|endoftext|>",
        "</s>",
    ]

    def __init__(
        self,
        high_entropy_percentile: float = 5.0,
        min_token_distance: int = 100,
    ):
        """
        Initialize step divider.

        Args:
            high_entropy_percentile: Percentile threshold for high-entropy (default: 5%)
            min_token_distance: Minimum token distance between step boundaries (default: 100)
        """
        self.high_entropy_percentile = high_entropy_percentile
        self.min_token_distance = min_token_distance

    def is_punctuation_or_digit(self, token: str) -> bool:
        """
        Check if token is punctuation or digit.

        Args:
            token: Token string

        Returns:
            True if token is punctuation or digit
        """
        # Remove whitespace
        token_stripped = token.strip()

        if not token_stripped:
            return True

        # Check if all characters are punctuation or digits
        return all(
            char in ".,;:!?()[]{}\"'`-–—…/\\|@#$%^&*+=<>~" or char.isdigit() or char.isspace()
            for char in token_stripped
        )

    def is_sentence_ending(self, token: str) -> bool:
        """
        Check if token contains sentence ending punctuation.

        Sentence endings include: . ! ? 。！？

        Args:
            token: Token string

        Returns:
            True if token contains sentence ending punctuation
        """
        # Common sentence ending marks in both English and Chinese
        sentence_endings = {'.', '!', '?', '。', '！', '？'}

        # Check if token contains any sentence ending punctuation
        return any(char in sentence_endings for char in token)

    def filter_trailing_special_tokens(
        self,
        tokens: List[str],
        entropy_sequence: List[Dict],
    ) -> Tuple[List[str], List[Dict]]:
        """
        Remove trailing special tokens from the end of token sequence.

        This ensures that special tokens like <|im_end|>, <|return|> don't appear
        in the final step or downstream analysis outputs.

        Args:
            tokens: List of token strings
            entropy_sequence: List of entropy information dicts

        Returns:
            Tuple of (filtered_tokens, filtered_entropy_sequence)
        """
        if not tokens:
            return tokens, entropy_sequence

        # Find the last non-special token position
        last_valid_idx = len(tokens) - 1

        # Scan backwards from the end to find trailing special tokens
        while last_valid_idx >= 0:
            token = tokens[last_valid_idx]

            # Check if this token is a special token (exact match or contains)
            is_special = False
            for special_token in self.TRAILING_SPECIAL_TOKENS:
                if special_token in token or token.strip() == special_token.strip():
                    is_special = True
                    break

            if is_special:
                last_valid_idx -= 1
            else:
                # Found a non-special token, stop scanning
                break

        # If all tokens are special (shouldn't happen in practice)
        if last_valid_idx < 0:
            return [], []

        # Return sliced lists (from 0 to last_valid_idx+1, inclusive)
        filtered_tokens = tokens[:last_valid_idx + 1]
        filtered_entropy_sequence = entropy_sequence[:last_valid_idx + 1]

        return filtered_tokens, filtered_entropy_sequence

    def find_step_boundaries(self, entropy_sequence: List[Dict]) -> List[Dict]:
        """
        Find step boundaries based on high-entropy tokens, extended to sentence endings.

        NEW Strategy:
        1. Find high-entropy tokens as boundary candidates (trigger tokens)
        2. From each candidate, search forward for the next sentence ending (. ! ? 。！？)
        3. Use the sentence ending position as the actual boundary
        4. Calculate distance from the previous boundary's sentence ending position
        5. Return both the boundary position and trigger token info for display

        This ensures steps end at natural sentence boundaries, avoiding mid-sentence splits,
        while preserving information about which high-entropy token triggered the split.

        Criteria:
        1. Token is in top 5% entropy (has color red/pink/purple)
        2. Token is not punctuation or digit
        3. Distance from previous boundary's ending >= min_token_distance
        4. Extend to next sentence ending after the high-entropy token

        Args:
            entropy_sequence: List of entropy information dicts with tokens

        Returns:
            List of boundary info dicts, each containing:
            - boundary_position: where the step actually ends (sentence ending)
            - trigger_position: position of the high-entropy token that triggered this boundary
            - trigger_token: the high-entropy token text
            - trigger_entropy: entropy value of the trigger token
        """
        boundaries = []
        last_boundary_end = -self.min_token_distance  # Position of last boundary's sentence ending

        for item in entropy_sequence:
            position = item.get("position", 0)
            token = item.get("token", "")
            color = item.get("color", "black")
            entropy = item.get("entropy")

            # Skip if no entropy data
            if entropy is None:
                continue

            # Criterion 1: Must be high-entropy (top 5%)
            if color not in ["red", "pink", "purple"]:
                continue

            # Criterion 2: Must not be punctuation or digit
            if self.is_punctuation_or_digit(token):
                continue

            # Criterion 3: Must be far enough from previous boundary's ending
            if position - last_boundary_end < self.min_token_distance:
                continue

            # Found a candidate boundary token (trigger)
            trigger_position = position
            trigger_token = token
            trigger_entropy = entropy

            # Now search forward for the next sentence ending
            sentence_end_pos = self._find_next_sentence_ending(
                entropy_sequence, start_pos=position
            )

            if sentence_end_pos is not None:
                # Found a sentence ending, use it as the boundary
                boundaries.append({
                    "boundary_position": sentence_end_pos,
                    "trigger_position": trigger_position,
                    "trigger_token": trigger_token,
                    "trigger_entropy": trigger_entropy,
                })
                last_boundary_end = sentence_end_pos
            else:
                # No sentence ending found after this token (rare case)
                # Use the candidate position itself as boundary
                boundaries.append({
                    "boundary_position": trigger_position,
                    "trigger_position": trigger_position,
                    "trigger_token": trigger_token,
                    "trigger_entropy": trigger_entropy,
                })
                last_boundary_end = trigger_position

        return boundaries

    def _find_next_sentence_ending(
        self,
        entropy_sequence: List[Dict],
        start_pos: int,
        max_search_distance: int = 1000 # 我认为这里意义不大，不太需要额外的限制
    ) -> Optional[int]:
        """
        Find the next sentence ending punctuation after a given position.

        Args:
            entropy_sequence: List of entropy information dicts
            start_pos: Starting position to search from
            max_search_distance: Maximum number of tokens to search forward

        Returns:
            Position of the next sentence ending, or None if not found
        """
        search_end = min(start_pos + max_search_distance, len(entropy_sequence))

        for i in range(start_pos, search_end):
            item = entropy_sequence[i]
            token = item.get("token", "")

            if self.is_sentence_ending(token):
                return i

        # No sentence ending found within search distance
        return None

    def divide_into_steps(
        self,
        tokens: List[str],
        boundaries: List[Dict],
    ) -> List[Dict]:
        """
        Divide token sequence into steps based on boundaries.

        Args:
            tokens: List of token strings
            boundaries: List of boundary info dicts (from find_step_boundaries)

        Returns:
            List of step dictionaries
        """
        if not boundaries:
            # No boundaries found, return entire sequence as one step
            return [{
                "step_number": 1,
                "start_position": 0,
                "end_position": len(tokens) - 1,
                "tokens": tokens,
                "text": "".join(tokens),
                "boundary_token": None,
                "boundary_entropy": None,
            }]

        steps = []
        start_pos = 0

        for i, boundary_info in enumerate(boundaries):
            # Extract boundary position from the dict
            boundary_pos = boundary_info["boundary_position"]

            # Get tokens for this step (from start to boundary)
            step_tokens = tokens[start_pos:boundary_pos + 1]
            step_text = "".join(step_tokens)

            # Use the TRIGGER token information for display (not the sentence ending)
            trigger_token = boundary_info["trigger_token"]
            trigger_entropy = boundary_info["trigger_entropy"]

            steps.append({
                "step_number": i + 1,
                "start_position": start_pos,
                "end_position": boundary_pos,
                "tokens": step_tokens,
                "text": step_text,
                "boundary_token": trigger_token,  # Show trigger token, not sentence ending
                "boundary_entropy": trigger_entropy,
                "trigger_position": boundary_info["trigger_position"],  # Additional info
            })

            start_pos = boundary_pos + 1

        # Add final step (from last boundary to end)
        if start_pos < len(tokens):
            final_tokens = tokens[start_pos:]
            final_text = "".join(final_tokens)

            steps.append({
                "step_number": len(boundaries) + 1,
                "start_position": start_pos,
                "end_position": len(tokens) - 1,
                "tokens": final_tokens,
                "text": final_text,
                "boundary_token": None,
                "boundary_entropy": None,
            })

        return steps

    def process_sample(self, result: Dict) -> Dict:
        """
        Process a single sample and divide it into steps.

        Args:
            result: Result dictionary with entropy_sequence and tokens

        Returns:
            Dictionary with step division information
        """
        entropy_sequence = result.get("entropy_sequence", [])
        tokens = result.get("tokens", [])

        # Filter out trailing special tokens BEFORE step division
        # This ensures special tokens don't appear in the final step output
        filtered_tokens, filtered_entropy_sequence = self.filter_trailing_special_tokens(
            tokens, entropy_sequence
        )

        # Find step boundaries (returns list of dicts with trigger info)
        boundaries = self.find_step_boundaries(filtered_entropy_sequence)

        # Divide into steps (using boundary positions and trigger info)
        steps = self.divide_into_steps(filtered_tokens, boundaries)

        return {
            "id": result.get("id", "unknown"),
            "problem": result.get("problem", ""),
            "num_steps": len(steps),
            "num_boundaries": len(boundaries),
            "boundaries": boundaries,
            "steps": steps,
            "full_text": result.get("generated_text", ""),
        }

    def process_all_results(self, results: List[Dict]) -> List[Dict]:
        """
        Process all results and divide them into steps.

        Args:
            results: List of result dictionaries

        Returns:
            List of step division results
        """
        step_divisions = []

        for result in results:
            step_division = self.process_sample(result)
            step_divisions.append(step_division)

        return step_divisions


def create_step_divider(config: Dict) -> StepDivider:
    """
    Create a step divider from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        StepDivider instance
    """
    step_config = config.get("step_division", {})

    return StepDivider(
        high_entropy_percentile=step_config.get("high_entropy_percentile", 5.0),
        min_token_distance=step_config.get("min_token_distance", 100),
    )
