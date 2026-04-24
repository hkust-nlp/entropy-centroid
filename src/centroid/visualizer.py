"""
Visualization module for entropy centroid analysis.

Contains all plotting functions for centroid distributions,
HEP variance analysis, and per-problem visualizations.
"""

import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class CentroidVisualizer:
    """
    Visualizer for entropy centroid analysis results.
    
    Generates various plots comparing correct vs incorrect trajectories.
    """

    def __init__(self, output_dir: str, top_percent: float = 5.0, method: str = 'moment'):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
            top_percent: Top percentage used for high entropy threshold
            method: Centroid calculation method used
        """
        self.output_dir = output_dir
        self.top_percent = top_percent
        self.method = method
        os.makedirs(output_dir, exist_ok=True)

    def create_all_visualizations(
        self,
        correct_data: Dict,
        incorrect_data: Dict,
        skip_per_problem: bool = False
    ) -> Dict:
        """
        Generate all visualizations.

        Args:
            correct_data: Data for correct trajectories
            incorrect_data: Data for incorrect trajectories
            skip_per_problem: Skip per-problem plots (for faster execution)

        Returns:
            Dictionary with categorization results
        """
        print("\n" + "=" * 80)
        print("Generating visualizations...")
        print("=" * 80)

        print("\n[1/11] Generating centroid histogram...")
        self.plot_centroid_histogram(correct_data, incorrect_data)

        print("[2/11] Generating centroid boxplot...")
        self.plot_centroid_boxplot(correct_data, incorrect_data)

        print("[3/11] Generating centroid vs num_heps scatter plot...")
        self.plot_centroid_vs_num_heps(correct_data, incorrect_data)

        print("[4/11] Generating HEP variance histogram...")
        self.plot_hep_variance_histogram(correct_data, incorrect_data)

        print("[5/11] Generating centroid vs variance scatter plot...")
        self.plot_centroid_vs_variance(correct_data, incorrect_data)

        print("[6/11] Generating combined analysis plot...")
        self.plot_combined_analysis(correct_data, incorrect_data)

        categorization = None
        if not skip_per_problem:
            print("[7/11] Generating per-problem centroid plots...")
            categorization = self.plot_centroid_per_problem(correct_data, incorrect_data)

            print("[8/11] Generating per-problem variance plots...")
            self.plot_variance_per_problem(correct_data, incorrect_data)

            print("[9/11] Generating per-problem centroid vs num_heps plots...")
            self.plot_centroid_vs_num_heps_per_problem(correct_data, incorrect_data)
        else:
            print("[7-9/11] Skipping per-problem plots...")

        print("\n✓ All visualizations generated!")
        return categorization

    def plot_centroid_histogram(self, correct_data: Dict, incorrect_data: Dict):
        """Plot histogram comparing correct vs incorrect centroid distributions."""
        correct_centroids = [traj['centroid'] for traj in correct_data['trajectories']]
        incorrect_centroids = [traj['centroid'] for traj in incorrect_data['trajectories']]

        if not correct_centroids and not incorrect_centroids:
            print("  Warning: No centroid data for histogram")
            return

        fig, ax = plt.subplots(figsize=(14, 8))
        bins = np.linspace(0, 1, 30)

        if correct_centroids:
            ax.hist(correct_centroids, bins=bins, alpha=0.6, color='#2ecc71',
                   label=f'Correct (n={len(correct_centroids)} trajectories)',
                   edgecolor='black', density=True)

        if incorrect_centroids:
            ax.hist(incorrect_centroids, bins=bins, alpha=0.6, color='#e74c3c',
                   label=f'Incorrect (n={len(incorrect_centroids)} trajectories)',
                   edgecolor='black', density=True)

        ax.set_xlabel('Entropy Centroid (Normalized Position)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax.set_title(f'Entropy Centroid Distribution (Top {self.top_percent}%): Correct vs Incorrect',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        # Statistics text box
        stats_text = self._format_stats_text(correct_centroids, incorrect_centroids)
        if stats_text:
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'entropy_centroid_histogram.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def plot_centroid_boxplot(self, correct_data: Dict, incorrect_data: Dict):
        """Plot boxplot comparing correct vs incorrect centroid distributions."""
        correct_centroids = [traj['centroid'] for traj in correct_data['trajectories']]
        incorrect_centroids = [traj['centroid'] for traj in incorrect_data['trajectories']]

        if not correct_centroids and not incorrect_centroids:
            print("  Warning: No centroid data for boxplot")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        data_to_plot = []
        labels = []
        colors = []

        if correct_centroids:
            data_to_plot.append(correct_centroids)
            labels.append(f'Correct\n(n={len(correct_centroids)})')
            colors.append('#2ecc71')

        if incorrect_centroids:
            data_to_plot.append(incorrect_centroids)
            labels.append(f'Incorrect\n(n={len(incorrect_centroids)})')
            colors.append('#e74c3c')

        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Entropy Centroid (Normalized Position)', fontsize=14, fontweight='bold')
        ax.set_title(f'Entropy Centroid Comparison (Top {self.top_percent}%)',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'entropy_centroid_boxplot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def plot_centroid_vs_num_heps(self, correct_data: Dict, incorrect_data: Dict):
        """Plot scatter plot of centroid vs number of HEPs."""
        fig, ax = plt.subplots(figsize=(12, 8))

        if correct_data['trajectories']:
            correct_num_heps = [traj['num_heps'] for traj in correct_data['trajectories']]
            correct_centroids = [traj['centroid'] for traj in correct_data['trajectories']]
            ax.scatter(correct_num_heps, correct_centroids, color='#2ecc71', alpha=0.6,
                      s=50, label=f'Correct (n={len(correct_centroids)})',
                      edgecolors='black', linewidths=0.5)

        if incorrect_data['trajectories']:
            incorrect_num_heps = [traj['num_heps'] for traj in incorrect_data['trajectories']]
            incorrect_centroids = [traj['centroid'] for traj in incorrect_data['trajectories']]
            ax.scatter(incorrect_num_heps, incorrect_centroids, color='#e74c3c', alpha=0.4,
                      s=50, label=f'Incorrect (n={len(incorrect_centroids)})',
                      edgecolors='black', linewidths=0.5)

        ax.set_xlabel('Number of HEPs', fontsize=14, fontweight='bold')
        ax.set_ylabel('Entropy Centroid (Normalized Position)', fontsize=14, fontweight='bold')
        ax.set_title(f'Entropy Centroid vs Number of HEPs (Top {self.top_percent}%)',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'entropy_centroid_vs_num_heps.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def plot_hep_variance_histogram(self, correct_data: Dict, incorrect_data: Dict):
        """Plot histogram comparing HEP duration variance distributions."""
        correct_variances = [traj['hep_duration_variance'] for traj in correct_data['trajectories']]
        incorrect_variances = [traj['hep_duration_variance'] for traj in incorrect_data['trajectories']]

        if not correct_variances and not incorrect_variances:
            print("  Warning: No variance data for histogram")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        all_variances = correct_variances + incorrect_variances
        max_var = max(all_variances) if all_variances else 0.1
        bins = np.linspace(0, min(max_var * 1.1, 0.15), 30)

        if correct_variances:
            ax.hist(correct_variances, bins=bins, alpha=0.6, color='#2ecc71',
                   label=f'Correct (n={len(correct_variances)} trajectories)',
                   edgecolor='black', density=True)

        if incorrect_variances:
            ax.hist(incorrect_variances, bins=bins, alpha=0.6, color='#e74c3c',
                   label=f'Incorrect (n={len(incorrect_variances)} trajectories)',
                   edgecolor='black', density=True)

        ax.set_xlabel('HEP Duration Variance (tokens²)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax.set_title(f'HEP Duration Variance Distribution (Top {self.top_percent}%)',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'hep_variance_histogram.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def plot_centroid_vs_variance(self, correct_data: Dict, incorrect_data: Dict):
        """Plot scatter plot of centroid vs HEP duration variance."""
        fig, ax = plt.subplots(figsize=(12, 8))

        if correct_data['trajectories']:
            correct_centroids = [traj['centroid'] for traj in correct_data['trajectories']]
            correct_variances = [traj['hep_duration_variance'] for traj in correct_data['trajectories']]
            ax.scatter(correct_variances, correct_centroids, color='#2ecc71', alpha=0.6,
                      s=50, label=f'Correct (n={len(correct_centroids)})',
                      edgecolors='black', linewidths=0.5)

        if incorrect_data['trajectories']:
            incorrect_centroids = [traj['centroid'] for traj in incorrect_data['trajectories']]
            incorrect_variances = [traj['hep_duration_variance'] for traj in incorrect_data['trajectories']]
            ax.scatter(incorrect_variances, incorrect_centroids, color='#e74c3c', alpha=0.4,
                      s=50, label=f'Incorrect (n={len(incorrect_centroids)})',
                      edgecolors='black', linewidths=0.5)

        ax.set_xlabel('HEP Duration Variance (tokens²)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Entropy Centroid (Normalized Position)', fontsize=14, fontweight='bold')
        ax.set_title(f'Entropy Centroid vs HEP Duration Variance (Top {self.top_percent}%)',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'centroid_vs_hep_variance.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def plot_combined_analysis(self, correct_data: Dict, incorrect_data: Dict):
        """Plot combined analysis with multiple subplots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        correct_centroids = [traj['centroid'] for traj in correct_data['trajectories']]
        incorrect_centroids = [traj['centroid'] for traj in incorrect_data['trajectories']]
        correct_num_heps = [traj['num_heps'] for traj in correct_data['trajectories']]
        incorrect_num_heps = [traj['num_heps'] for traj in incorrect_data['trajectories']]
        correct_variances = [traj['hep_duration_variance'] for traj in correct_data['trajectories']]
        incorrect_variances = [traj['hep_duration_variance'] for traj in incorrect_data['trajectories']]

        # Subplot 1: Centroid histogram
        ax1 = axes[0, 0]
        bins = np.linspace(0, 1, 25)
        if correct_centroids:
            ax1.hist(correct_centroids, bins=bins, alpha=0.6, color='#2ecc71',
                    label='Correct', edgecolor='black', density=True)
        if incorrect_centroids:
            ax1.hist(incorrect_centroids, bins=bins, alpha=0.6, color='#e74c3c',
                    label='Incorrect', edgecolor='black', density=True)
        ax1.set_xlabel('Entropy Centroid')
        ax1.set_ylabel('Density')
        ax1.set_title('Centroid Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Centroid vs Num HEPs
        ax2 = axes[0, 1]
        if correct_centroids:
            ax2.scatter(correct_num_heps, correct_centroids, color='#2ecc71',
                       alpha=0.5, s=30, label='Correct')
        if incorrect_centroids:
            ax2.scatter(incorrect_num_heps, incorrect_centroids, color='#e74c3c',
                       alpha=0.3, s=30, label='Incorrect')
        ax2.set_xlabel('Number of HEPs')
        ax2.set_ylabel('Entropy Centroid')
        ax2.set_title('Centroid vs Num HEPs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Variance histogram
        ax3 = axes[1, 0]
        all_variances = correct_variances + incorrect_variances
        if all_variances:
            max_var = max(all_variances)
            bins = np.linspace(0, min(max_var * 1.1, 0.15), 25)
            if correct_variances:
                ax3.hist(correct_variances, bins=bins, alpha=0.6, color='#2ecc71',
                        label='Correct', edgecolor='black', density=True)
            if incorrect_variances:
                ax3.hist(incorrect_variances, bins=bins, alpha=0.6, color='#e74c3c',
                        label='Incorrect', edgecolor='black', density=True)
        ax3.set_xlabel('HEP Duration Variance')
        ax3.set_ylabel('Density')
        ax3.set_title('HEP Variance Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Subplot 4: Centroid vs Variance
        ax4 = axes[1, 1]
        if correct_centroids:
            ax4.scatter(correct_variances, correct_centroids, color='#2ecc71',
                       alpha=0.5, s=30, label='Correct')
        if incorrect_centroids:
            ax4.scatter(incorrect_variances, incorrect_centroids, color='#e74c3c',
                       alpha=0.3, s=30, label='Incorrect')
        ax4.set_xlabel('HEP Duration Variance')
        ax4.set_ylabel('Entropy Centroid')
        ax4.set_title('Centroid vs Variance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.suptitle(f'Combined Centroid Analysis (Top {self.top_percent}%, Method: {self.method})',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = os.path.join(self.output_dir, 'combined_centroid_variance_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def plot_centroid_per_problem(
        self,
        correct_data: Dict,
        incorrect_data: Dict
    ) -> Dict:
        """
        Plot per-problem centroid distributions with categorization.

        Returns:
            Dictionary with problem categorization results
        """
        per_problem_dir = os.path.join(self.output_dir, 'entropy_centroid_per_problem')
        os.makedirs(per_problem_dir, exist_ok=True)

        # Categorization directories
        cat_dirs = {
            'cat1': os.path.join(self.output_dir, 'category_1_incorrect_lower_mean'),
            'cat2': os.path.join(self.output_dir, 'category_2_top3_more_incorrect'),
            'cat3': os.path.join(self.output_dir, 'category_3_top1_incorrect'),
            'cat_all': os.path.join(self.output_dir, 'category_all_three_conditions'),
        }
        for d in cat_dirs.values():
            os.makedirs(d, exist_ok=True)

        all_problem_ids = set(correct_data['by_problem'].keys()) | set(incorrect_data['by_problem'].keys())

        # Categorization tracking
        category_1_problems = []
        category_2_problems = []
        category_3_problems = []
        category_all_problems = []

        for problem_id in tqdm(sorted(all_problem_ids), desc="Per-problem plots"):
            correct_trajs = correct_data['by_problem'].get(problem_id, [])
            incorrect_trajs = incorrect_data['by_problem'].get(problem_id, [])

            if not correct_trajs and not incorrect_trajs:
                continue

            correct_centroids = [t['centroid'] for t in correct_trajs]
            incorrect_centroids = [t['centroid'] for t in incorrect_trajs]

            # Categorization (only for problems with both)
            conditions = self._categorize_problem(correct_centroids, incorrect_centroids,
                                                   correct_trajs, incorrect_trajs)

            if conditions['cat1']:
                category_1_problems.append(problem_id)
            if conditions['cat2']:
                category_2_problems.append(problem_id)
            if conditions['cat3']:
                category_3_problems.append(problem_id)
            if conditions['all']:
                category_all_problems.append(problem_id)

            # Create plot
            output_file = self._plot_single_problem(
                problem_id, correct_centroids, incorrect_centroids, per_problem_dir
            )

            # Copy to category directories
            if conditions['cat1']:
                shutil.copy2(output_file, os.path.join(cat_dirs['cat1'], os.path.basename(output_file)))
            if conditions['cat2']:
                shutil.copy2(output_file, os.path.join(cat_dirs['cat2'], os.path.basename(output_file)))
            if conditions['cat3']:
                shutil.copy2(output_file, os.path.join(cat_dirs['cat3'], os.path.basename(output_file)))
            if conditions['all']:
                shutil.copy2(output_file, os.path.join(cat_dirs['cat_all'], os.path.basename(output_file)))

        # Save categorization summary
        categorization = self._save_categorization_summary(
            category_1_problems, category_2_problems,
            category_3_problems, category_all_problems
        )

        print(f"  ✓ Saved {len(all_problem_ids)} per-problem plots to: {per_problem_dir}")
        print(f"  Category 1: {len(category_1_problems)}, Category 2: {len(category_2_problems)}, "
              f"Category 3: {len(category_3_problems)}, All: {len(category_all_problems)}")

        return categorization

    def plot_variance_per_problem(self, correct_data: Dict, incorrect_data: Dict):
        """Plot per-problem HEP variance distributions."""
        per_problem_dir = os.path.join(self.output_dir, 'hep_variance_per_problem')
        os.makedirs(per_problem_dir, exist_ok=True)

        all_problem_ids = set(correct_data['by_problem'].keys()) | set(incorrect_data['by_problem'].keys())

        for problem_id in tqdm(sorted(all_problem_ids), desc="Variance per-problem"):
            correct_trajs = correct_data['by_problem'].get(problem_id, [])
            incorrect_trajs = incorrect_data['by_problem'].get(problem_id, [])

            if not correct_trajs and not incorrect_trajs:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            correct_variances = [t['hep_duration_variance'] for t in correct_trajs]
            incorrect_variances = [t['hep_duration_variance'] for t in incorrect_trajs]
            correct_centroids = [t['centroid'] for t in correct_trajs]
            incorrect_centroids = [t['centroid'] for t in incorrect_trajs]

            # Variance histogram
            all_variances = correct_variances + incorrect_variances
            if all_variances:
                bins = np.linspace(0, max(all_variances) * 1.1, 15)
                if correct_variances:
                    ax1.hist(correct_variances, bins=bins, alpha=0.6, color='#2ecc71',
                            label=f'Correct (n={len(correct_variances)})', edgecolor='black', density=True)
                if incorrect_variances:
                    ax1.hist(incorrect_variances, bins=bins, alpha=0.6, color='#e74c3c',
                            label=f'Incorrect (n={len(incorrect_variances)})', edgecolor='black', density=True)

            ax1.set_xlabel('HEP Duration Variance')
            ax1.set_ylabel('Density')
            ax1.set_title(f'Variance Distribution - {problem_id}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Centroid vs Variance
            if correct_variances:
                ax2.scatter(correct_variances, correct_centroids, color='#2ecc71', alpha=0.7,
                           s=80, label='Correct', edgecolors='black', linewidths=0.5)
            if incorrect_variances:
                ax2.scatter(incorrect_variances, incorrect_centroids, color='#e74c3c', alpha=0.5,
                           s=80, label='Incorrect', edgecolors='black', linewidths=0.5)

            ax2.set_xlabel('HEP Duration Variance')
            ax2.set_ylabel('Entropy Centroid')
            ax2.set_title(f'Centroid vs Variance - {problem_id}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = os.path.join(per_problem_dir, f'problem_{problem_id}_variance.png')
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close()

        print(f"  ✓ Saved variance plots to: {per_problem_dir}")

    def plot_centroid_vs_num_heps_per_problem(self, correct_data: Dict, incorrect_data: Dict):
        """Plot per-problem centroid vs num_heps scatter plots."""
        per_problem_dir = os.path.join(self.output_dir, 'centroid_vs_num_heps_per_problem')
        os.makedirs(per_problem_dir, exist_ok=True)

        all_problem_ids = set(correct_data['by_problem'].keys()) | set(incorrect_data['by_problem'].keys())

        for problem_id in tqdm(sorted(all_problem_ids), desc="Centroid vs HEPs per-problem"):
            correct_trajs = correct_data['by_problem'].get(problem_id, [])
            incorrect_trajs = incorrect_data['by_problem'].get(problem_id, [])

            if not correct_trajs and not incorrect_trajs:
                continue

            fig, ax = plt.subplots(figsize=(10, 8))

            if correct_trajs:
                ax.scatter([t['num_heps'] for t in correct_trajs],
                          [t['centroid'] for t in correct_trajs],
                          color='#2ecc71', alpha=0.7, s=80,
                          label=f'Correct (n={len(correct_trajs)})',
                          edgecolors='black', linewidths=0.5)

            if incorrect_trajs:
                ax.scatter([t['num_heps'] for t in incorrect_trajs],
                          [t['centroid'] for t in incorrect_trajs],
                          color='#e74c3c', alpha=0.5, s=80,
                          label=f'Incorrect (n={len(incorrect_trajs)})',
                          edgecolors='black', linewidths=0.5)

            ax.set_xlabel('Number of HEPs')
            ax.set_ylabel('Entropy Centroid')
            ax.set_title(f'Centroid vs Num HEPs - {problem_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = os.path.join(per_problem_dir, f'problem_{problem_id}_centroid_vs_heps.png')
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close()

        print(f"  ✓ Saved centroid vs HEPs plots to: {per_problem_dir}")

    def _format_stats_text(self, correct_data: List, incorrect_data: List) -> str:
        """Format statistics text for plot annotation."""
        stats_text = ""
        if correct_data:
            stats_text += f"Correct:\n  Mean: {np.mean(correct_data):.4f}\n  Median: {np.median(correct_data):.4f}\n  Std: {np.std(correct_data):.4f}\n\n"
        if incorrect_data:
            stats_text += f"Incorrect:\n  Mean: {np.mean(incorrect_data):.4f}\n  Median: {np.median(incorrect_data):.4f}\n  Std: {np.std(incorrect_data):.4f}"
        return stats_text

    def _categorize_problem(
        self,
        correct_centroids: List,
        incorrect_centroids: List,
        correct_trajs: List,
        incorrect_trajs: List
    ) -> Dict:
        """Categorize problem based on centroid analysis."""
        conditions = {'cat1': False, 'cat2': False, 'cat3': False, 'all': False}

        if not correct_centroids or not incorrect_centroids:
            return conditions

        # Category 1: Incorrect mean < Correct mean
        if np.mean(incorrect_centroids) < np.mean(correct_centroids):
            conditions['cat1'] = True

        # Category 2 & 3: Analyze top trajectories
        all_trajs = [{'centroid': t['centroid'], 'is_correct': True} for t in correct_trajs]
        all_trajs += [{'centroid': t['centroid'], 'is_correct': False} for t in incorrect_trajs]
        sorted_trajs = sorted(all_trajs, key=lambda x: x['centroid'])

        # Category 3: Top 1 is incorrect
        if not sorted_trajs[0]['is_correct']:
            conditions['cat3'] = True

        # Category 2: Top 3 has more incorrect than correct
        top_3 = sorted_trajs[:min(3, len(sorted_trajs))]
        incorrect_count = sum(1 for t in top_3 if not t['is_correct'])
        if incorrect_count > len(top_3) - incorrect_count:
            conditions['cat2'] = True

        # Category ALL
        if conditions['cat1'] and conditions['cat2'] and conditions['cat3']:
            conditions['all'] = True

        return conditions

    def _plot_single_problem(
        self,
        problem_id: str,
        correct_centroids: List,
        incorrect_centroids: List,
        output_dir: str
    ) -> str:
        """Plot single problem centroid distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        bins = np.linspace(0, 1, 20)

        # Histogram
        if correct_centroids:
            ax1.hist(correct_centroids, bins=bins, alpha=0.6, color='#2ecc71',
                    label=f'Correct (n={len(correct_centroids)})', edgecolor='black', density=True)
        if incorrect_centroids:
            ax1.hist(incorrect_centroids, bins=bins, alpha=0.6, color='#e74c3c',
                    label=f'Incorrect (n={len(incorrect_centroids)})', edgecolor='black', density=True)

        ax1.set_xlabel('Entropy Centroid')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Centroid Distribution - {problem_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Boxplot
        data_to_plot = []
        labels = []
        colors = []

        if correct_centroids:
            data_to_plot.append(correct_centroids)
            labels.append(f'Correct\n(n={len(correct_centroids)})')
            colors.append('#2ecc71')
        if incorrect_centroids:
            data_to_plot.append(incorrect_centroids)
            labels.append(f'Incorrect\n(n={len(incorrect_centroids)})')
            colors.append('#e74c3c')

        if data_to_plot:
            bp = ax2.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax2.set_ylabel('Entropy Centroid')
        ax2.set_title(f'Centroid Comparison - {problem_id}')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Problem {problem_id} (Top {self.top_percent}%, Method: {self.method})',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'problem_{problem_id}_centroid.png')
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        return output_file

    def _save_categorization_summary(
        self,
        cat1: List,
        cat2: List,
        cat3: List,
        cat_all: List
    ) -> Dict:
        """Save categorization summary to files."""
        summary = {
            'category_1_incorrect_lower_mean': {
                'description': 'Problems where incorrect trajectories have lower mean centroid',
                'count': len(cat1),
                'problem_ids': sorted(cat1)
            },
            'category_2_top3_more_incorrect': {
                'description': 'Problems where top 3 smallest centroids have more incorrect',
                'count': len(cat2),
                'problem_ids': sorted(cat2)
            },
            'category_3_top1_incorrect': {
                'description': 'Problems where smallest centroid is incorrect',
                'count': len(cat3),
                'problem_ids': sorted(cat3)
            },
            'category_all_three_conditions': {
                'description': 'Problems satisfying all three conditions',
                'count': len(cat_all),
                'problem_ids': sorted(cat_all)
            }
        }

        # Save JSON
        json_file = os.path.join(self.output_dir, 'problem_categorization_summary.json')
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save text
        txt_file = os.path.join(self.output_dir, 'problem_categorization_summary.txt')
        with open(txt_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PROBLEM CATEGORIZATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            for cat_name, cat_data in summary.items():
                f.write(f"{cat_name}:\n")
                f.write(f"  Description: {cat_data['description']}\n")
                f.write(f"  Count: {cat_data['count']}\n")
                f.write(f"  Problem IDs: {', '.join(map(str, cat_data['problem_ids']))}\n\n")

        return summary
