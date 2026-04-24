"""
Visualization utilities for semantic entropy analysis.

This module provides functions for visualizing:
1. Original vs semantic entropy comparison
2. Correct/incorrect solution comparison
3. Entropy reduction heatmaps
4. Semantic grouping examples
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple
from collections import defaultdict
import seaborn as sns
from matplotlib.lines import Line2D

# Use non-interactive backend
plt.switch_backend('Agg')


class SemanticEntropyVisualizer:
    """Visualizer for semantic entropy analysis."""

    def __init__(self, output_dir: str):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        self.vis_dir = os.path.join(output_dir, "semantic_visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

    def plot_entropy_comparison(
        self,
        semantic_results: List[Dict],
        threshold: float,
        max_samples: int = 10
    ):
        """
        Plot original vs semantic entropy comparison for each trajectory.

        Args:
            semantic_results: List of semantic entropy result dicts
            threshold: Similarity threshold used
            max_samples: Maximum number of samples to plot
        """
        print(f"Creating entropy comparison plots (threshold={threshold})...")

        for idx, result in enumerate(semantic_results[:max_samples]):
            sample_id = result.get("id", f"sample_{idx}")
            sequence = result.get("semantic_entropy_sequence", [])

            if not sequence:
                continue

            # Extract data
            positions = [item["position"] for item in sequence if item["original_entropy"] is not None]
            original_entropies = [item["original_entropy"] for item in sequence if item["original_entropy"] is not None]
            semantic_entropies = [item["semantic_entropy"] for item in sequence if item["semantic_entropy"] is not None]

            if not positions:
                continue

            # Create plot
            fig, ax = plt.subplots(figsize=(14, 6))

            ax.plot(positions, original_entropies, 'o-', linewidth=2, markersize=4,
                   color='#3498db', label='Token-level Entropy', alpha=0.8)
            ax.plot(positions, semantic_entropies, 's-', linewidth=2, markersize=4,
                   color='#e74c3c', label='Semantic Entropy', alpha=0.8)

            ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
            ax.set_ylabel('Entropy', fontsize=12, fontweight='bold')
            # Simple title using only sample ID
            ax.set_title(f'Entropy Comparison: Sample {sample_id} (τ={threshold:.2f})',
                        fontsize=13, fontweight='bold', pad=15)

            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=11, loc='best', framealpha=0.95)

            plt.tight_layout()
            output_path = os.path.join(self.vis_dir, f"comparison_{sample_id}_tau{threshold:.2f}.png")
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()

            if idx == 0:  # Print only for first sample
                print(f"  ✓ Saved: comparison_{sample_id}_tau{threshold:.2f}.png")

        print(f"  ✓ Created {min(max_samples, len(semantic_results))} comparison plots")

    def plot_correct_vs_incorrect_comparison(
        self,
        semantic_results: List[Dict],
        step_divisions: List[Dict],
        classification: Dict[str, bool],
        threshold: float
    ):
        """
        Plot step-wise semantic entropy for correct vs incorrect solutions.

        Args:
            semantic_results: List of semantic entropy result dicts
            step_divisions: List of step division dicts
            classification: Dict mapping sample_id to correctness
            threshold: Similarity threshold used
        """
        print(f"Creating correct vs incorrect comparison (threshold={threshold})...")

        # Compute step-wise semantic entropies
        step_semantic_entropies = self._compute_step_semantic_entropies(
            semantic_results, step_divisions
        )

        if not step_semantic_entropies:
            print("  ⚠ No step entropy data available")
            return

        # Find max number of steps
        max_steps = max((len(entropies) for entropies in step_semantic_entropies.values()), default=0)

        if max_steps == 0:
            print("  ⚠ No steps found")
            return

        # Aggregate by correctness
        correct_accum = [[] for _ in range(max_steps)]
        incorrect_accum = [[] for _ in range(max_steps)]

        for sample_id, entropies in step_semantic_entropies.items():
            is_correct = classification.get(sample_id, False)

            for step_idx, entropy_val in enumerate(entropies):
                if step_idx < max_steps:
                    if is_correct:
                        correct_accum[step_idx].append(entropy_val)
                    else:
                        incorrect_accum[step_idx].append(entropy_val)

        # Compute means
        correct_entropies = [np.mean(vals) if vals else 0.0 for vals in correct_accum]
        incorrect_entropies = [np.mean(vals) if vals else 0.0 for vals in incorrect_accum]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        steps = list(range(1, len(correct_entropies) + 1))

        ax.plot(steps, correct_entropies, 'o-', linewidth=2.5, markersize=6,
               color='#2ecc71', label='Correct Solutions', alpha=0.8)
        ax.plot(steps, incorrect_entropies, 's-', linewidth=2.5, markersize=6,
               color='#e74c3c', label='Incorrect Solutions', alpha=0.8)

        ax.set_xlabel('Step Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Semantic Entropy', fontsize=12, fontweight='bold')
        ax.set_title(f'Semantic Entropy by Step: Correct vs Incorrect\nThreshold={threshold}',
                    fontsize=14, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best', framealpha=0.95)

        if len(steps) <= 20:
            ax.set_xticks(steps)

        plt.tight_layout()
        output_path = os.path.join(self.vis_dir, f"correct_vs_incorrect_tau{threshold:.2f}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: correct_vs_incorrect_tau{threshold:.2f}.png")

    def plot_entropy_reduction_heatmap(
        self,
        semantic_results: List[Dict],
        threshold: float,
        max_samples: int = 20,
        max_positions: int = 200
    ):
        """
        Plot heatmap showing entropy reduction across samples and positions.

        Args:
            semantic_results: List of semantic entropy result dicts
            threshold: Similarity threshold used
            max_samples: Maximum number of samples to include
            max_positions: Maximum sequence length to plot
        """
        print(f"Creating entropy reduction heatmap (threshold={threshold})...")

        # Collect reduction data
        reduction_matrix = []
        sample_labels = []

        for idx, result in enumerate(semantic_results[:max_samples]):
            sample_id = result.get("id", f"sample_{idx}")
            sequence = result.get("semantic_entropy_sequence", [])

            if not sequence:
                continue

            # Extract reduction values
            reductions = []
            for item in sequence[:max_positions]:
                reduction = item.get("entropy_reduction")
                if reduction is not None:
                    reductions.append(max(0, reduction))  # Only positive reductions
                else:
                    reductions.append(0)

            if reductions:
                reduction_matrix.append(reductions)
                sample_labels.append(sample_id.split('_')[0][:15])  # Truncate label

        if not reduction_matrix:
            print("  ⚠ No reduction data available")
            return

        # Pad sequences to same length
        max_len = max(len(row) for row in reduction_matrix)
        padded_matrix = [row + [0] * (max_len - len(row)) for row in reduction_matrix]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, max(8, len(sample_labels) * 0.4)))

        im = ax.imshow(padded_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

        ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample ID', fontsize=12, fontweight='bold')
        ax.set_title(f'Entropy Reduction Heatmap\nThreshold={threshold}',
                    fontsize=14, fontweight='bold', pad=20)

        ax.set_yticks(range(len(sample_labels)))
        ax.set_yticklabels(sample_labels, fontsize=9)

        # Set x-axis ticks
        if max_len <= 50:
            ax.set_xticks(range(0, max_len, 5))
        else:
            ax.set_xticks(range(0, max_len, 20))

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Entropy Reduction', fontsize=11, fontweight='bold')

        plt.tight_layout()
        output_path = os.path.join(self.vis_dir, f"entropy_reduction_heatmap_tau{threshold:.2f}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: entropy_reduction_heatmap_tau{threshold:.2f}.png")

    def plot_per_problem_entropy_comparison(
        self,
        semantic_results: List[Dict],
        step_divisions: List[Dict],
        classification: Dict[str, bool],
        threshold: float
    ):
        """
        Create per-problem plots showing original vs semantic entropy by step.
        Separate plots for correct and incorrect solutions.

        Each problem gets two plots:
        1. All correct trajectories (with individual lines and mean)
        2. All incorrect trajectories (with individual lines and mean)

        Args:
            semantic_results: List of semantic entropy result dicts
            step_divisions: List of step division dicts
            classification: Dict mapping sample_id to correctness
            threshold: Similarity threshold used
        """
        print(f"Creating per-problem step-wise entropy comparison plots (threshold={threshold})...")

        # Create subdirectory for per-problem entropy comparisons
        comparison_dir = os.path.join(self.vis_dir, "per_problem_entropy_comparison")
        os.makedirs(comparison_dir, exist_ok=True)

        # Compute step-wise entropies for both original and semantic
        step_original_entropies = self._compute_step_entropies(semantic_results, step_divisions, entropy_type='original')
        step_semantic_entropies = self._compute_step_entropies(semantic_results, step_divisions, entropy_type='semantic')

        # Create lookup tables for faster access
        semantic_results_dict = {sr.get('id'): sr for sr in semantic_results}
        step_divisions_dict = {sd.get('id'): sd for sd in step_divisions}

        # Group trajectories by problem
        problem_trajectories = defaultdict(lambda: {'correct': [], 'incorrect': []})

        for sample_id, original_steps in step_original_entropies.items():
            is_correct = classification.get(sample_id, False)
            semantic_steps = step_semantic_entropies.get(sample_id, [])

            # Find problem ID using original_id field (numeric ID, not problem text)
            # This follows the same logic as visualize_entropy_by_step.py
            problem_id = None

            # Try to get original_id from semantic_results
            sem_result = semantic_results_dict.get(sample_id)
            if sem_result:
                problem_id = sem_result.get('original_id')

            # Fallback to step_divisions
            if problem_id is None:
                step_div = step_divisions_dict.get(sample_id)
                if step_div:
                    problem_id = step_div.get('original_id')

            # Final fallback: extract from sample_id (e.g., "0_traj_0" -> "0")
            if problem_id is None:
                try:
                    problem_id = sample_id.split('_traj_')[0]
                except:
                    problem_id = sample_id

            traj_data = {
                'id': sample_id,
                'original_steps': original_steps,
                'semantic_steps': semantic_steps,
                'problem_id': problem_id
            }

            if is_correct:
                problem_trajectories[problem_id]['correct'].append(traj_data)
            else:
                problem_trajectories[problem_id]['incorrect'].append(traj_data)

        # Sort problems with proper numeric handling
        # If problem_id is numeric, sort numerically; otherwise sort lexicographically
        def sort_key(item):
            problem_id = item[0]
            try:
                return (0, int(problem_id))  # (0, numeric) for numeric IDs
            except (ValueError, TypeError):
                return (1, str(problem_id))  # (1, string) for non-numeric IDs

        sorted_problems = sorted(problem_trajectories.items(), key=sort_key)

        # Create plots for each problem
        plot_count = 0
        for prob_idx, (problem_id, trajectories) in enumerate(sorted_problems):
            # Plot correct solutions
            if trajectories['correct']:
                plot_count += self._plot_step_entropy_by_correctness(
                    trajectories=trajectories['correct'],
                    problem_id=problem_id,
                    prob_idx=prob_idx,
                    is_correct=True,
                    threshold=threshold,
                    output_dir=comparison_dir
                )

            # Plot incorrect solutions
            if trajectories['incorrect']:
                plot_count += self._plot_step_entropy_by_correctness(
                    trajectories=trajectories['incorrect'],
                    problem_id=problem_id,
                    prob_idx=prob_idx,
                    is_correct=False,
                    threshold=threshold,
                    output_dir=comparison_dir
                )

        print(f"  ✓ Created {plot_count} per-problem step-wise entropy plots")

    def _compute_step_entropies(
        self,
        semantic_results: List[Dict],
        step_divisions: List[Dict],
        entropy_type: str = 'semantic'
    ) -> Dict[str, List[float]]:
        """
        Compute average entropy per step for each trajectory.

        Args:
            semantic_results: List of semantic entropy result dicts
            step_divisions: List of step division dicts
            entropy_type: 'original' or 'semantic'

        Returns:
            Dict mapping sample_id to list of average entropies per step
        """
        step_entropies = {}

        for sem_result in semantic_results:
            sample_id = sem_result.get('id', '')
            sem_sequence = sem_result.get('semantic_entropy_sequence', [])

            # Find corresponding step divisions
            step_div = None
            for sd in step_divisions:
                if sd.get('id') == sample_id:
                    step_div = sd
                    break

            if step_div is None:
                continue

            steps = step_div.get('steps', [])
            step_entropy_list = []

            for step in steps:
                start_pos = step.get('start_position', 0)
                end_pos = step.get('end_position', 0)

                # Collect entropy values for tokens in this step
                step_entropies_vals = []
                for pos in range(start_pos, min(end_pos, len(sem_sequence))):
                    if pos < len(sem_sequence):
                        if entropy_type == 'original':
                            entropy_val = sem_sequence[pos].get('original_entropy')
                        else:  # semantic
                            entropy_val = sem_sequence[pos].get('semantic_entropy')

                        if entropy_val is not None:
                            step_entropies_vals.append(entropy_val)

                # Compute average entropy for this step
                if step_entropies_vals:
                    avg_entropy = np.mean(step_entropies_vals)
                else:
                    avg_entropy = 0.0

                step_entropy_list.append(avg_entropy)

            step_entropies[sample_id] = step_entropy_list

        return step_entropies

    def _plot_step_entropy_by_correctness(
        self,
        trajectories: List[Dict],
        problem_id: str,
        prob_idx: int,
        is_correct: bool,
        threshold: float,
        output_dir: str
    ) -> int:
        """
        Plot step-wise entropy comparison (original vs semantic) for multiple trajectories.

        Shows individual trajectories as thin lines, mean as thick line with markers,
        and uses standard deviation for uncertainty bands.

        Args:
            trajectories: List of trajectory dicts with original_steps and semantic_steps
            problem_id: Original problem ID from dataset
            prob_idx: Index in sorted problem list (used for display position)
            is_correct: Whether these are correct (True) or incorrect (False) solutions
            threshold: Similarity threshold used
            output_dir: Directory to save output

        Returns:
            1 if plot created successfully, 0 otherwise
        """
        if not trajectories:
            return 0

        # Find max number of steps
        max_steps = max((len(t['original_steps']) for t in trajectories), default=0)

        if max_steps == 0:
            return 0

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Store values for computing mean and std
        original_accum = [[] for _ in range(max_steps)]
        semantic_accum = [[] for _ in range(max_steps)]

        # Plot individual trajectories as thin lines
        for traj in trajectories:
            original_steps = traj['original_steps']
            semantic_steps = traj['semantic_steps']

            steps = list(range(1, len(original_steps) + 1))

            # Original entropy - thin blue line
            ax.plot(steps, original_steps, 'o-', linewidth=1.0, markersize=2,
                   color='#3498db', alpha=0.3)

            # Semantic entropy - thin red line
            ax.plot(steps, semantic_steps, 's-', linewidth=1.0, markersize=2,
                   color='#e74c3c', alpha=0.3)

            # Accumulate for mean/std calculation
            for step_idx, (orig_val, sem_val) in enumerate(zip(original_steps, semantic_steps)):
                if step_idx < max_steps:
                    original_accum[step_idx].append(orig_val)
                    semantic_accum[step_idx].append(sem_val)

        # Compute mean and std for both entropy types
        original_means = []
        original_stds = []
        semantic_means = []
        semantic_stds = []

        for step_idx in range(max_steps):
            if original_accum[step_idx]:
                original_means.append(np.mean(original_accum[step_idx]))
                original_stds.append(np.std(original_accum[step_idx]))
            else:
                original_means.append(0.0)
                original_stds.append(0.0)

            if semantic_accum[step_idx]:
                semantic_means.append(np.mean(semantic_accum[step_idx]))
                semantic_stds.append(np.std(semantic_accum[step_idx]))
            else:
                semantic_means.append(0.0)
                semantic_stds.append(0.0)

        steps = list(range(1, len(original_means) + 1))

        # Plot mean lines with markers and std bands
        # Original entropy
        ax.plot(steps, original_means, 'o-', linewidth=2.5, markersize=6,
               color='#3498db', label='Token-level Entropy (Mean)', alpha=0.9, zorder=10)
        ax.fill_between(steps,
                        np.array(original_means) - np.array(original_stds),
                        np.array(original_means) + np.array(original_stds),
                        color='#3498db', alpha=0.15)

        # Semantic entropy
        ax.plot(steps, semantic_means, 's-', linewidth=2.5, markersize=6,
               color='#e74c3c', label='Semantic Entropy (Mean)', alpha=0.9, zorder=10)
        ax.fill_between(steps,
                        np.array(semantic_means) - np.array(semantic_stds),
                        np.array(semantic_means) + np.array(semantic_stds),
                        color='#e74c3c', alpha=0.15)

        # Formatting
        ax.set_xlabel('Step Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Entropy', fontsize=12, fontweight='bold')

        # Title with status - simple format using only problem ID
        status = "Correct" if is_correct else "Incorrect"
        status_color = '#2ecc71' if is_correct else '#e74c3c'

        # Display title with only problem ID number to avoid LaTeX parsing issues
        ax.set_title(f'Problem {problem_id}: {status} Solutions (τ={threshold:.2f})',
                    fontsize=13, fontweight='bold', pad=15, color=status_color)

        ax.grid(True, alpha=0.3, linestyle='--')

        # Create custom legend
        handles = []
        labels = []

        # Thin lines for individual trajectories
        handles.append(Line2D([0], [0], color='#3498db', linewidth=1.0, alpha=0.3))
        labels.append(f'Individual token-level (n={len(trajectories)})')

        handles.append(Line2D([0], [0], color='#e74c3c', linewidth=1.0, alpha=0.3))
        labels.append(f'Individual semantic (n={len(trajectories)})')

        # Thick lines for means
        handles.append(Line2D([0], [0], color='#3498db', linewidth=2.5, marker='o'))
        labels.append('Token-level Mean ± Std')

        handles.append(Line2D([0], [0], color='#e74c3c', linewidth=2.5, marker='s'))
        labels.append('Semantic Mean ± Std')

        ax.legend(handles=handles, labels=labels, fontsize=10, loc='best', framealpha=0.95)

        if len(steps) <= 20:
            ax.set_xticks(steps)

        plt.tight_layout()

        # Save with simple filename using only problem ID
        status_prefix = "correct" if is_correct else "incorrect"
        filename = f"problem_{problem_id}_{status_prefix}_tau{threshold:.2f}.png"
        output_path = os.path.join(output_dir, filename)

        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        return 1

    def plot_semantic_grouping_examples(
        self,
        semantic_results: List[Dict],
        entropy_results: List[Dict],
        threshold: float,
        top_percentile: float = 5.0,
        max_examples: int = 10
    ):
        """
        Visualize semantic grouping for high-entropy positions.

        Args:
            semantic_results: List of semantic entropy result dicts
            entropy_results: List of original entropy result dicts
            threshold: Similarity threshold used
            top_percentile: Percentile threshold for high-entropy positions
            max_examples: Maximum number of examples to visualize
        """
        print(f"Creating semantic grouping visualizations (top {top_percentile}% positions)...")

        grouping_dir = os.path.join(self.vis_dir, "grouping_examples")
        os.makedirs(grouping_dir, exist_ok=True)

        examples_created = 0

        for sem_result, orig_result in zip(semantic_results, entropy_results):
            if examples_created >= max_examples:
                break

            sample_id = sem_result.get("id", "unknown")
            sem_sequence = sem_result.get("semantic_entropy_sequence", [])
            orig_sequence = orig_result.get("entropy_sequence", [])

            # Find high-entropy positions
            valid_entropies = [(i, item["original_entropy"]) for i, item in enumerate(sem_sequence)
                              if item["original_entropy"] is not None]

            if not valid_entropies:
                continue

            # Calculate percentile threshold
            entropies = [e for _, e in valid_entropies]
            threshold_value = np.percentile(entropies, 100 - top_percentile)

            # Get high-entropy positions with grouping
            high_entropy_positions = [
                (i, item) for i, item in enumerate(sem_sequence)
                if item["original_entropy"] is not None
                and item["original_entropy"] >= threshold_value
                and item["num_semantic_groups"] > 1  # Only show positions with actual grouping
            ]

            # Visualize up to 3 positions per sample
            for pos_idx, (seq_idx, item) in enumerate(high_entropy_positions[:3]):
                if examples_created >= max_examples:
                    break

                self._plot_single_grouping(
                    item=item,
                    sample_id=sample_id,
                    threshold=threshold,
                    output_dir=grouping_dir
                )

                examples_created += 1

        print(f"  ✓ Created {examples_created} grouping visualization examples")

    def _plot_single_grouping(
        self,
        item: Dict,
        sample_id: str,
        threshold: float,
        output_dir: str
    ):
        """Plot semantic grouping for a single position."""
        position = item["position"]
        token = item["token"]
        groups = item["semantic_groups"]
        num_groups = item["num_semantic_groups"]

        if not groups:
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, num_groups * 1.5)))

        # Prepare data
        colors = plt.cm.Set3(np.linspace(0, 1, num_groups))

        y_pos = 0
        labels = []
        values = []
        group_colors = []

        for group_idx, group in enumerate(groups):
            combined_prob = group["combined_prob"]
            token_ids = group["token_ids"]
            probs = group["probs"]

            # Add group label
            labels.append(f"Group {group_idx + 1}")
            values.append(combined_prob)
            group_colors.append(colors[group_idx])

            y_pos += 1

        # Create horizontal bar chart
        y_positions = np.arange(len(labels))
        ax.barh(y_positions, values, color=group_colors, alpha=0.7, edgecolor='black')

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Combined Probability', fontsize=11, fontweight='bold')
        # Simple title using only sample ID and position number
        ax.set_title(f'Semantic Grouping: Sample {sample_id} at Position {position}',
                    fontsize=11, fontweight='bold', pad=15)

        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"sample_{sample_id}_pos{position}_tau{threshold:.2f}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _compute_step_semantic_entropies(
        self,
        semantic_results: List[Dict],
        step_divisions: List[Dict]
    ) -> Dict[str, List[float]]:
        """Compute average semantic entropy per step."""
        step_entropies = {}

        for sem_result in semantic_results:
            sample_id = sem_result.get('id', '')
            sem_sequence = sem_result.get('semantic_entropy_sequence', [])

            # Find corresponding step divisions
            step_div = None
            for sd in step_divisions:
                if sd.get('id') == sample_id:
                    step_div = sd
                    break

            if step_div is None:
                continue

            steps = step_div.get('steps', [])
            step_entropy_list = []

            for step in steps:
                start_pos = step.get('start_position', 0)
                end_pos = step.get('end_position', 0)

                # Collect semantic entropy values for tokens in this step
                step_entropies_vals = []
                for pos in range(start_pos, min(end_pos, len(sem_sequence))):
                    if pos < len(sem_sequence):
                        entropy_val = sem_sequence[pos].get('semantic_entropy')
                        if entropy_val is not None:
                            step_entropies_vals.append(entropy_val)

                # Compute average
                if step_entropies_vals:
                    avg_entropy = np.mean(step_entropies_vals)
                else:
                    avg_entropy = 0.0

                step_entropy_list.append(avg_entropy)

            step_entropies[sample_id] = step_entropy_list

        return step_entropies


def create_all_visualizations(
    semantic_results: List[Dict],
    entropy_results: List[Dict],
    step_divisions: List[Dict],
    classification: Dict[str, bool],
    output_dir: str,
    threshold: float,
    top_percentile: float = 5.0
):
    """
    Create all visualizations for semantic entropy analysis.

    Args:
        semantic_results: Semantic entropy results
        entropy_results: Original entropy results
        step_divisions: Step division data
        classification: Correctness classification
        output_dir: Output directory
        threshold: Similarity threshold
        top_percentile: Percentile for high-entropy positions
    """
    visualizer = SemanticEntropyVisualizer(output_dir)

    print(f"\n{'='*80}")
    print(f"Creating visualizations for threshold={threshold}")
    print(f"{'='*80}")

    # 1. Entropy comparison plots
    visualizer.plot_entropy_comparison(semantic_results, threshold, max_samples=10)

    # 2. Correct vs incorrect comparison
    visualizer.plot_correct_vs_incorrect_comparison(
        semantic_results, step_divisions, classification, threshold
    )

    # 3. Entropy reduction heatmap
    visualizer.plot_entropy_reduction_heatmap(semantic_results, threshold)

    # 4. Per-problem entropy comparison (NEW: separate correct and incorrect trajectories)
    visualizer.plot_per_problem_entropy_comparison(
        semantic_results, step_divisions, classification, threshold
    )

    # 5. Semantic grouping examples
    visualizer.plot_semantic_grouping_examples(
        semantic_results, entropy_results, threshold,
        top_percentile=top_percentile, max_examples=15
    )

    print(f"{'='*80}\n")
