"""
Semantic entropy calculator using token embedding similarity.

This module computes semantic entropy by grouping semantically similar tokens
based on their embeddings and calculating entropy over semantic groups rather
than individual tokens.
"""

import math
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class UnionFind:
    """Union-Find data structure for clustering tokens into semantic groups."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_groups(self) -> Dict[int, List[int]]:
        """Get all connected components as groups."""
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            groups[root].append(i)
        return dict(groups)


def is_math_symbol(token: str) -> bool:
    """
    Check if a token is a mathematical symbol (number or operator).

    Args:
        token: Token string to check

    Returns:
        True if token is a math symbol, False otherwise
    """
    # Remove leading/trailing whitespace for checking
    token_stripped = token.strip()

    if not token_stripped:
        return False

    # Check if token contains digits
    if re.search(r'\d', token_stripped):
        return True

    # Common mathematical operators and symbols
    math_operators = {
        '+', '-', '*', '/', '=', '<', '>', '≤', '≥', '≠',
        '×', '÷', '±', '∓', '^', '²', '³', '√', '∛',
        '(', ')', '[', ']', '{', '}',
        '.', ',',  # Decimal point and thousand separator
        '%', '°',  # Percent and degree
        '∑', '∏', '∫', '∂',  # Sum, product, integral, partial derivative
        '∞', 'π', 'e',  # Common constants
        '\\frac', '\\sqrt', '\\sum', '\\int', '\\pi',  # LaTeX math
    }

    # Check if token is a math operator
    if token_stripped in math_operators:
        return True

    # Check for LaTeX math expressions
    if '\\' in token_stripped and any(op in token_stripped for op in ['frac', 'sqrt', 'sum', 'int', 'pi']):
        return True

    return False


class SemanticEntropyCalculator:
    """
    Calculate semantic entropy using token embeddings.

    This calculator groups candidate tokens by semantic similarity and computes
    entropy over these groups rather than individual tokens.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.9,
        device: str = "cuda",
        batch_size: int = 32,
        enable_cache: bool = True
    ):
        """
        Initialize semantic entropy calculator.

        Args:
            model: The language model with embedding layer
            threshold: Cosine similarity threshold for grouping (default: 0.9)
            device: Device to use for computation
            batch_size: Batch size for embedding extraction (default: 32)
            enable_cache: Enable embedding caching to avoid redundant computation
        """
        self.model = model
        self.threshold = threshold
        self.device = device
        self.batch_size = batch_size
        self.enable_cache = enable_cache

        # Embedding cache: {token_id: embedding_tensor}
        self.embedding_cache = {} if enable_cache else None

        # Get embedding layer
        try:
            self.embedding_layer = model.get_input_embeddings()
        except AttributeError:
            # For some models, might need different access method
            if hasattr(model, 'model'):
                self.embedding_layer = model.model.get_input_embeddings()
            else:
                raise ValueError("Cannot access embedding layer from model")

        self.embedding_layer.eval()

    def batch_extract_embeddings(self, token_ids: List[int]) -> torch.Tensor:
        """
        Extract embeddings for a batch of token IDs with caching.

        Args:
            token_ids: List of token IDs (may contain duplicates)

        Returns:
            Tensor of embeddings [len(token_ids), hidden_dim]
        """
        # Deduplicate token IDs to minimize computation
        unique_token_ids = list(set(token_ids))
        embeddings_dict = {}

        # Check cache for existing embeddings
        uncached_ids = []
        if self.enable_cache:
            for token_id in unique_token_ids:
                if token_id in self.embedding_cache:
                    embeddings_dict[token_id] = self.embedding_cache[token_id]
                else:
                    uncached_ids.append(token_id)
        else:
            uncached_ids = unique_token_ids

        # Extract embeddings for uncached tokens in batches
        if uncached_ids:
            with torch.no_grad():
                for i in range(0, len(uncached_ids), self.batch_size):
                    batch_ids = uncached_ids[i:i + self.batch_size]
                    token_tensor = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
                    batch_embeddings = self.embedding_layer(token_tensor)  # [batch, hidden_dim]

                    # Store in dict and cache
                    for j, token_id in enumerate(batch_ids):
                        embedding = batch_embeddings[j]
                        embeddings_dict[token_id] = embedding
                        if self.enable_cache:
                            self.embedding_cache[token_id] = embedding

        # Reconstruct embeddings in original order
        embeddings_list = [embeddings_dict[token_id] for token_id in token_ids]
        embeddings = torch.stack(embeddings_list, dim=0)

        return embeddings

    def compute_cosine_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            embeddings: Tensor of shape [k, hidden_dim]

        Returns:
            Similarity matrix of shape [k, k]
        """
        # Normalize embeddings
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Compute similarity: S[i,j] = cos(E[i], E[j])
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())

        return similarity_matrix

    def group_tokens_by_similarity(
        self,
        token_ids: List[int],
        probs: List[float]
    ) -> Tuple[List[Dict], int]:
        """
        Group tokens by semantic similarity.

        Args:
            token_ids: List of candidate token IDs
            probs: List of corresponding probabilities

        Returns:
            Tuple of (semantic_groups, num_groups) where semantic_groups is a list
            of dicts with keys 'token_ids', 'probs', 'combined_prob'
        """
        if len(token_ids) == 0:
            return [], 0

        if len(token_ids) == 1:
            return [{"token_ids": token_ids, "probs": probs, "combined_prob": probs[0]}], 1

        # Get embeddings using batched extraction (with caching)
        embeddings = self.batch_extract_embeddings(token_ids)  # [k, hidden_dim]

        # Compute similarity matrix
        with torch.no_grad():
            similarity_matrix = self.compute_cosine_similarity(embeddings)  # [k, k]

        # Build graph using Union-Find
        uf = UnionFind(len(token_ids))

        for i in range(len(token_ids)):
            for j in range(i + 1, len(token_ids)):
                if similarity_matrix[i, j].item() > self.threshold:
                    uf.union(i, j)

        # Get groups
        groups_dict = uf.get_groups()

        # Format groups with probabilities
        semantic_groups = []
        for group_indices in groups_dict.values():
            group_token_ids = [token_ids[i] for i in group_indices]
            group_probs = [probs[i] for i in group_indices]
            combined_prob = sum(group_probs)

            semantic_groups.append({
                "token_ids": group_token_ids,
                "probs": group_probs,
                "combined_prob": combined_prob
            })

        # Sort groups by combined probability (descending)
        semantic_groups.sort(key=lambda x: x["combined_prob"], reverse=True)

        return semantic_groups, len(semantic_groups)

    def calculate_semantic_entropy(self, semantic_groups: List[Dict]) -> float:
        """
        Calculate entropy over semantic groups.

        Args:
            semantic_groups: List of semantic groups with combined probabilities

        Returns:
            Semantic entropy value
        """
        if not semantic_groups:
            return 0.0

        entropy = 0.0
        for group in semantic_groups:
            p = group["combined_prob"]
            if p > 0:
                entropy -= p * math.log(p)

        return entropy

    def process_token_position(
        self,
        top_k_token_ids: List[int],
        top_k_probs: List[float],
        original_entropy: float,
        token: str,
        position: int
    ) -> Dict:
        """
        Process a single token position.

        Args:
            top_k_token_ids: Top-k candidate token IDs
            top_k_probs: Corresponding probabilities
            original_entropy: Original token-level entropy
            token: The selected token string
            position: Position in sequence

        Returns:
            Dictionary with semantic entropy information
        """
        if not top_k_token_ids or original_entropy is None:
            return {
                "position": position,
                "token": token,
                "original_entropy": None,
                "semantic_entropy": None,
                "num_semantic_groups": 0,
                "semantic_groups": [],
                "entropy_reduction": None,
                "is_math_symbol": False
            }

        # Check if token is a math symbol - if so, skip semantic grouping
        is_math = is_math_symbol(token)

        if is_math:
            # Math symbols: use original entropy directly without semantic grouping
            return {
                "position": position,
                "token": token,
                "original_entropy": original_entropy,
                "semantic_entropy": original_entropy,  # Use original entropy
                "num_semantic_groups": len(top_k_token_ids),  # Each token is its own group
                "semantic_groups": [],  # No grouping performed
                "entropy_reduction": 0.0,  # No reduction for math symbols
                "is_math_symbol": True
            }

        # Non-math tokens: perform semantic grouping
        # Group tokens by similarity
        semantic_groups, num_groups = self.group_tokens_by_similarity(
            top_k_token_ids, top_k_probs
        )

        # Calculate semantic entropy
        semantic_entropy = self.calculate_semantic_entropy(semantic_groups)

        # Calculate reduction
        entropy_reduction = original_entropy - semantic_entropy if original_entropy is not None else None

        return {
            "position": position,
            "token": token,
            "original_entropy": original_entropy,
            "semantic_entropy": semantic_entropy,
            "num_semantic_groups": num_groups,
            "semantic_groups": semantic_groups,
            "entropy_reduction": entropy_reduction,
            "is_math_symbol": False
        }

    def prefetch_sequence_embeddings(self, entropy_sequence: List[Dict]):
        """
        Prefetch all embeddings for a sequence to maximize batching efficiency.

        Optimized to skip embeddings for math symbols since they don't participate
        in semantic grouping.

        Args:
            entropy_sequence: List of entropy info dicts from EntropyCalculator
        """
        if not self.enable_cache:
            return

        # Collect all unique token IDs from the sequence, skipping math symbols
        all_token_ids = set()
        math_symbol_count = 0

        for item in entropy_sequence:
            token = item.get("token", "")

            # Skip math symbols - they don't need embedding extraction
            if is_math_symbol(token):
                math_symbol_count += 1
                continue

            top_k_token_ids = item.get("top_k_token_ids", [])
            all_token_ids.update(top_k_token_ids)

        # Remove already cached IDs
        uncached_ids = [tid for tid in all_token_ids if tid not in self.embedding_cache]

        if not uncached_ids:
            if math_symbol_count > 0:
                print(f"  Skipped {math_symbol_count} math symbols, all other embeddings cached")
            return

        # Batch extract all embeddings at once
        print(f"  Prefetching embeddings for {len(uncached_ids)} unique tokens (skipped {math_symbol_count} math symbols)...")
        self.batch_extract_embeddings(uncached_ids)

    def process_sequence(
        self,
        entropy_sequence: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """
        Process an entire entropy sequence.

        Args:
            entropy_sequence: List of entropy info dicts from EntropyCalculator

        Returns:
            Tuple of (semantic_entropy_sequence, statistics)
        """
        # Prefetch all embeddings for this sequence to maximize batch efficiency
        self.prefetch_sequence_embeddings(entropy_sequence)

        semantic_entropy_sequence = []

        total_reduction = 0.0
        num_positions_with_reduction = 0
        max_groups = 0
        valid_positions = 0
        math_symbol_positions = 0
        non_math_positions = 0

        for item in entropy_sequence:
            token = item.get("token", "")
            position = item.get("position", 0)
            original_entropy = item.get("entropy")
            top_k_token_ids = item.get("top_k_token_ids", [])
            top_k_probs = item.get("top_k_probs", [])

            result = self.process_token_position(
                top_k_token_ids=top_k_token_ids,
                top_k_probs=top_k_probs,
                original_entropy=original_entropy,
                token=token,
                position=position
            )

            semantic_entropy_sequence.append(result)

            # Update statistics
            if result["semantic_entropy"] is not None:
                valid_positions += 1

                # Track math symbols separately
                if result.get("is_math_symbol", False):
                    math_symbol_positions += 1
                else:
                    non_math_positions += 1
                    max_groups = max(max_groups, result["num_semantic_groups"])

                    if result["entropy_reduction"] is not None and result["entropy_reduction"] > 0:
                        total_reduction += result["entropy_reduction"]
                        num_positions_with_reduction += 1

        # Compute statistics
        statistics = {
            "total_positions": len(entropy_sequence),
            "valid_positions": valid_positions,
            "math_symbol_positions": math_symbol_positions,
            "non_math_positions": non_math_positions,
            "avg_entropy_reduction": total_reduction / non_math_positions if non_math_positions > 0 else 0.0,
            "positions_with_reduction": num_positions_with_reduction,
            "reduction_rate": num_positions_with_reduction / non_math_positions if non_math_positions > 0 else 0.0,
            "max_semantic_groups": max_groups,
            "threshold_used": self.threshold
        }

        return semantic_entropy_sequence, statistics


def load_model_for_embeddings(model_path: str, device: str = "cuda", gpu_ids: List[int] = None, load_in_8bit: bool = True):
    """
    Load model for embedding extraction with memory optimization.

    Supports tensor parallelism across multiple GPUs when gpu_ids is specified.

    Note: Respects pre-existing model quantization. If the model is already quantized
    (e.g., Mxfp4Config), the quantization parameters are ignored to avoid conflicts.

    Args:
        model_path: Path to the model
        device: Device to load model on (used if gpu_ids not specified)
        gpu_ids: List of GPU IDs for tensor parallelism (overrides device)
        load_in_8bit: Whether to use 8-bit quantization (ignored if model is pre-quantized)

    Returns:
        Loaded model
    """
    from transformers import AutoModelForCausalLM
    import torch
    import os

    print(f"Loading model from {model_path}...")

    # Handle gpu_ids for tensor parallelism
    if gpu_ids and len(gpu_ids) > 1:
        print(f"  Tensor Parallelism: Using GPUs {gpu_ids}")
        # Set environment variables for multi-GPU support
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # Load without quantization parameters to respect pre-quantized models
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # Automatically distribute across available GPUs
            trust_remote_code=True
        )
    else:
        # Single GPU loading
        if gpu_ids:
            device = f"cuda:{gpu_ids[0]}"
        print(f"  Device: {device}")

        # Load without quantization parameters to respect pre-quantized models
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True
        )

    model.eval()
    print("✓ Model loaded successfully!")

    return model
