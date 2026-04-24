#!/usr/bin/env python3
"""
Visualize high-entropy duration curves on a normalized token-position axis.

Compared with `visualize_high_entropy_duration.py`, this script:
1. Uses token position percentage (0-100%) on the x-axis.
2. Overlays all trajectories as thin lines.
3. Adds bold mean curves for correct and incorrect trajectories.
4. Produces both global and per-problem normalized line plots.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm

# Add repo src to path regardless of the current working directory.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluation.answer_comparator import AnswerComparator
from evaluation.answer_extractor import extract_answer

PLOT_DURATION_CAP = 250.0
CORRECT_COLOR = "#0f8f3d"
INCORRECT_COLOR = "#c0392b"
DIFF_COLOR = "#34495e"

# ---- AI-conference figure defaults ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,      # TrueType fonts in PDF (editable in Illustrator)
    "ps.fonttype": 42,
    "font.size": 28,
})


def _style_axes_conference(ax_left, ax_right=None):
    """Apply clean AI-conference-quality axis styling (NeurIPS / ICML / ICLR)."""
    # Left axis: keep bottom + left spines only
    ax_left.spines["top"].set_visible(False)
    if ax_right is None:
        ax_left.spines["right"].set_visible(False)
    for sp in ax_left.spines.values():
        if sp.get_visible():
            sp.set_linewidth(1.2)
    ax_left.tick_params(
        axis="both", which="major", direction="in",
        length=8, width=1.2, labelsize=26,
    )
    ax_left.tick_params(
        axis="both", which="minor", direction="in",
        length=4, width=0.8,
    )
    ax_left.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_left.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Right (twin) axis
    if ax_right is not None:
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_linewidth(1.2)
        ax_right.tick_params(
            axis="y", which="major", direction="in",
            length=8, width=1.2, labelsize=26,
        )
        ax_right.tick_params(
            axis="y", which="minor", direction="in",
            length=4, width=0.8,
        )
        ax_right.yaxis.set_minor_locator(AutoMinorLocator(2))


def load_data(result_dir: str) -> List[Dict]:
    """Load entropy results from `entropy_results.json`."""
    entropy_file = os.path.join(result_dir, "entropy_results.json")

    print("Loading entropy results...")
    with open(entropy_file, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_trajectory(trajectory: Dict, comparator: AnswerComparator) -> bool:
    """Evaluate whether a trajectory's extracted answer is correct."""
    ground_truth = trajectory.get("solution", "")
    generated_text = trajectory.get("generated_text", "")

    extracted_answer = extract_answer(generated_text)
    if extracted_answer is None:
        return False

    result = comparator.compare(ground_truth, extracted_answer)
    return result["is_correct"]


def compute_entropy_thresholds(
    entropy_sequence: List[Dict],
    top_percent: float = 5.0,
    bottom_percent: float = 50.0,
) -> Tuple[float, float]:
    """Compute entropy thresholds for starting and ending HEPs."""
    entropies = [float(item["entropy"]) for item in entropy_sequence if item.get("entropy") is not None]

    if not entropies:
        return float("inf"), float("-inf")

    high_threshold = np.percentile(entropies, 100.0 - top_percent)
    low_threshold = np.percentile(entropies, bottom_percent)
    return float(high_threshold), float(low_threshold)


def compute_high_entropy_durations(
    entropy_sequence: List[Dict],
    top_percent: float = 5.0,
    bottom_percent: float = 50.0,
    consecutive_low_threshold: int = 3,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Compute high-entropy durations for each token position.

    Returns:
    - duration_at_position: current HEP duration at each token position
    - duration_events: (start_position, duration) for each completed HEP
    """
    high_threshold, low_threshold = compute_entropy_thresholds(
        entropy_sequence, top_percent, bottom_percent
    )

    duration_at_position: List[int] = []
    duration_events: List[Tuple[int, int]] = []

    in_high_entropy_period = False
    current_period_start = 0
    current_duration = 0
    consecutive_low_count = 0

    for i, item in enumerate(entropy_sequence):
        entropy = item.get("entropy")

        if entropy is None:
            duration_at_position.append(0)
            continue

        entropy = float(entropy)

        if not in_high_entropy_period:
            if entropy >= high_threshold:
                in_high_entropy_period = True
                current_period_start = i
                current_duration = 1
                consecutive_low_count = 0
                duration_at_position.append(current_duration)
            else:
                duration_at_position.append(0)
        else:
            current_duration += 1

            if entropy <= low_threshold:
                consecutive_low_count += 1
            else:
                consecutive_low_count = 0

            if consecutive_low_count >= consecutive_low_threshold:
                final_duration = current_duration - consecutive_low_threshold
                if final_duration > 0:
                    duration_events.append((current_period_start, final_duration))

                in_high_entropy_period = False
                current_duration = 0
                consecutive_low_count = 0
                duration_at_position.append(0)
            else:
                duration_at_position.append(current_duration)

    if in_high_entropy_period and current_duration > 0:
        duration_events.append((current_period_start, current_duration))

    return duration_at_position, duration_events


def analyze_trajectories(
    entropy_results: List[Dict],
    comparator: AnswerComparator,
    top_percent: float = 5.0,
    bottom_percent: float = 50.0,
    consecutive_low_threshold: int = 3,
) -> Tuple[Dict, Dict]:
    """Analyze trajectories and split normalized duration data by correctness."""
    correct_data = {"trajectories": [], "by_problem": defaultdict(list)}
    incorrect_data = {"trajectories": [], "by_problem": defaultdict(list)}

    print(
        f"\nAnalyzing high-entropy durations (top {top_percent}%, bottom {bottom_percent}%)..."
    )

    for trajectory in tqdm(entropy_results):
        traj_id = trajectory["id"]
        original_id = trajectory.get("original_id", traj_id.split("_traj_")[0])
        is_correct = evaluate_trajectory(trajectory, comparator)

        entropy_sequence = trajectory.get("entropy_sequence", [])
        duration_at_position, duration_events = compute_high_entropy_durations(
            entropy_sequence,
            top_percent=top_percent,
            bottom_percent=bottom_percent,
            consecutive_low_threshold=consecutive_low_threshold,
        )

        if not duration_events:
            continue

        traj_data = {
            "id": traj_id,
            "original_id": original_id,
            "duration_at_position": duration_at_position,
            "duration_events": duration_events,
            "num_events": len(duration_events),
            "trajectory_length": len(duration_at_position),
        }

        if is_correct:
            correct_data["trajectories"].append(traj_data)
            correct_data["by_problem"][original_id].append(traj_data)
        else:
            incorrect_data["trajectories"].append(traj_data)
            incorrect_data["by_problem"][original_id].append(traj_data)

    return correct_data, incorrect_data


def normalize_curve_to_percent(
    duration_at_position: List[int],
    num_points: int = 201,
) -> Optional[np.ndarray]:
    """
    Resample a trajectory onto a shared 0-100% x-axis.

    Each trajectory is interpolated onto `num_points` evenly spaced percentage
    positions so curves with different lengths can be directly averaged.
    """
    if not duration_at_position:
        return None

    values = np.asarray(duration_at_position, dtype=float)
    if len(values) == 1:
        return np.full(num_points, values[0], dtype=float)

    original_x = np.linspace(0.0, 100.0, num=len(values))
    target_x = np.linspace(0.0, 100.0, num=num_points)
    return np.interp(target_x, original_x, values)


def gaussian_smooth(y: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Apply Gaussian smoothing to a 1-D array via numpy convolution."""
    if y.size == 0:
        return y
    radius = int(3 * sigma)
    window = 2 * radius + 1
    x = np.arange(window) - radius
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(y, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def compute_mean_curve(
    trajectories: List[Dict],
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return shared x-axis and mean curve for a list of trajectories."""
    target_x = np.linspace(0.0, 100.0, num=num_points)
    normalized_curves = []

    for traj in trajectories:
        curve = normalize_curve_to_percent(traj["duration_at_position"], num_points=num_points)
        if curve is not None:
            normalized_curves.append(curve)

    if not normalized_curves:
        return target_x, np.array([])

    stacked = np.vstack(normalized_curves)
    return target_x, np.mean(stacked, axis=0)


def add_group_lines_to_axis(
    ax: plt.Axes,
    trajectories: List[Dict],
    color: str,
    thin_alpha: float,
    thin_linewidth: float,
    mean_linewidth: float,
    num_points: int,
    label_prefix: str,
    plot_duration_cap: float,
):
    """Plot thin individual trajectories and a bold group mean line."""
    if not trajectories:
        return

    target_x = np.linspace(0.0, 100.0, num=num_points)
    normalized_curves = []

    for traj in trajectories:
        curve = normalize_curve_to_percent(traj["duration_at_position"], num_points=num_points)
        if curve is None:
            continue
        normalized_curves.append(curve)
        display_curve = np.minimum(curve, plot_duration_cap)
        ax.plot(
            target_x,
            display_curve,
            color=color,
            linewidth=thin_linewidth,
            alpha=thin_alpha,
            label="_nolegend_",
        )

    if normalized_curves:
        mean_curve = np.mean(np.vstack(normalized_curves), axis=0)
        display_mean_curve = np.minimum(mean_curve, plot_duration_cap)
        ax.plot(
            target_x,
            display_mean_curve,
            color=color,
            linewidth=mean_linewidth,
            alpha=1.0,
            label=f"{label_prefix} Mean (n={len(normalized_curves)})",
        )


def plot_normalized_duration_line_global(
    correct_data: Dict,
    incorrect_data: Dict,
    output_dir: str,
    top_percent: float = 5.0,
    num_points: int = 201,
    plot_duration_cap: float = PLOT_DURATION_CAP,
):
    """Plot global normalized line chart with dual axes: means + difference curve."""
    correct_trajs = correct_data["trajectories"]
    incorrect_trajs = incorrect_data["trajectories"]

    if not correct_trajs and not incorrect_trajs:
        print("Warning: No trajectory data for normalized global line plot")
        return

    target_x = np.linspace(0.0, 100.0, num=num_points)

    # ---- Normalize individual curves ----
    correct_curves = []
    for t in correct_trajs:
        c = normalize_curve_to_percent(t["duration_at_position"], num_points)
        if c is not None:
            correct_curves.append(c)

    incorrect_curves = []
    for t in incorrect_trajs:
        c = normalize_curve_to_percent(t["duration_at_position"], num_points)
        if c is not None:
            incorrect_curves.append(c)

    # ---- Compute group means ----
    correct_mean = np.mean(np.vstack(correct_curves), axis=0) if correct_curves else None
    incorrect_mean = np.mean(np.vstack(incorrect_curves), axis=0) if incorrect_curves else None
    has_diff = correct_mean is not None and incorrect_mean is not None

    # ---- Figure ----
    fig, ax1 = plt.subplots(figsize=(22, 10))

    # Individual trajectory clouds
    for curve in correct_curves:
        ax1.plot(
            target_x, np.minimum(curve, plot_duration_cap),
            color=CORRECT_COLOR, lw=0.5, alpha=0.06,
        )
    for curve in incorrect_curves:
        ax1.plot(
            target_x, np.minimum(curve, plot_duration_cap),
            color=INCORRECT_COLOR, lw=0.5, alpha=0.06,
        )

    # Left axis
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, plot_duration_cap)
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.set_xlabel("Token Position in Trajectory (%)", fontsize=30)
    ax1.set_ylabel("High-Entropy Duration (tokens)", fontsize=30)

    # ---- Right axis: difference curve ----
    ax2 = None
    if has_diff:
        diff_raw = correct_mean - incorrect_mean
        diff_smooth = gaussian_smooth(diff_raw, sigma=5.0)

        # SEM of the difference (assuming independence)
        correct_sem = np.std(np.vstack(correct_curves), axis=0) / np.sqrt(len(correct_curves))
        incorrect_sem = np.std(np.vstack(incorrect_curves), axis=0) / np.sqrt(len(incorrect_curves))
        diff_sem = np.sqrt(correct_sem**2 + incorrect_sem**2)
        diff_sem_smooth = gaussian_smooth(diff_sem, sigma=5.0)

        ax2 = ax1.twinx()

        # Positive / negative fills
        ax2.fill_between(
            target_x, 0, diff_smooth,
            where=diff_smooth >= 0,
            color=CORRECT_COLOR, alpha=0.10, interpolate=True,
        )
        ax2.fill_between(
            target_x, 0, diff_smooth,
            where=diff_smooth < 0,
            color=INCORRECT_COLOR, alpha=0.10, interpolate=True,
        )

        # SEM confidence band
        ax2.fill_between(
            target_x,
            diff_smooth - diff_sem_smooth,
            diff_smooth + diff_sem_smooth,
            color=DIFF_COLOR, alpha=0.10,
        )

        # Difference line
        ax2.plot(target_x, diff_smooth, color=DIFF_COLOR, lw=2.2, alpha=0.85)

        # Zero reference
        ax2.axhline(y=0, color="#7f8c8d", lw=0.8, ls="--", alpha=0.5)

        # Right axis styling
        ax2.set_ylabel(
            r"$\Delta$ Duration (Correct $-$ Incorrect)",
            fontsize=30, color=DIFF_COLOR,
        )
        ax2.tick_params(axis="y", labelcolor=DIFF_COLOR)

        # Symmetric y-limits so zero is centered
        y_abs_max = max(abs(diff_smooth.min()), abs(diff_smooth.max())) * 1.3
        ax2.set_ylim(-y_abs_max, y_abs_max)

        # Mark crossover points
        sign_changes = np.where(np.diff(np.sign(diff_smooth)))[0]
        for idx in sign_changes:
            cross_pct = target_x[idx]
            ax2.axvline(x=cross_pct, color="#95a5a6", lw=0.7, ls=":", alpha=0.5)

    # ---- Legend ----
    legend_handles = []
    if correct_curves:
        legend_handles.append(
            Line2D([0], [0], color=CORRECT_COLOR, lw=1.0, alpha=0.4,
                   label=f"Correct (n={len(correct_curves)})"),
        )
    if incorrect_curves:
        legend_handles.append(
            Line2D([0], [0], color=INCORRECT_COLOR, lw=1.0, alpha=0.4,
                   label=f"Incorrect (n={len(incorrect_curves)})"),
        )
    if has_diff:
        legend_handles.append(
            Line2D([0], [0], color=DIFF_COLOR, lw=2.2,
                   label=r"$\Delta$ (Correct $-$ Incorrect)"),
        )

    ax1.legend(
        handles=legend_handles, fontsize=26, loc="upper left",
        framealpha=0.8, edgecolor="none", fancybox=False,
    )

    _style_axes_conference(ax1, ax2)
    plt.tight_layout()
    output_file = os.path.join(output_dir, "high_entropy_duration_percent_global.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    output_pdf = os.path.join(output_dir, "high_entropy_duration_percent_global.pdf")
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved global plot: {output_file}")
    print(f"Saved global PDF:  {output_pdf}")


def plot_normalized_duration_line_per_problem(
    correct_data: Dict,
    incorrect_data: Dict,
    output_dir: str,
    top_percent: float = 5.0,
    num_points: int = 201,
    plot_duration_cap: float = PLOT_DURATION_CAP,
):
    """Plot per-problem normalized line charts with dual axes when both groups exist."""
    per_problem_dir = os.path.join(output_dir, "high_entropy_duration_percent_per_problem")
    os.makedirs(per_problem_dir, exist_ok=True)

    all_problem_ids = set(correct_data["by_problem"].keys()) | set(incorrect_data["by_problem"].keys())
    print(f"\nGenerating normalized per-problem line plots for {len(all_problem_ids)} problems...")
    min_for_diff = 3  # minimum trajectories per group to show diff curve

    for problem_id in tqdm(sorted(all_problem_ids)):
        correct_trajs = correct_data["by_problem"].get(problem_id, [])
        incorrect_trajs = incorrect_data["by_problem"].get(problem_id, [])

        if not correct_trajs and not incorrect_trajs:
            continue

        target_x = np.linspace(0.0, 100.0, num=num_points)

        correct_curves = []
        for t in correct_trajs:
            c = normalize_curve_to_percent(t["duration_at_position"], num_points)
            if c is not None:
                correct_curves.append(c)

        incorrect_curves = []
        for t in incorrect_trajs:
            c = normalize_curve_to_percent(t["duration_at_position"], num_points)
            if c is not None:
                incorrect_curves.append(c)

        correct_mean = np.mean(np.vstack(correct_curves), axis=0) if correct_curves else None
        incorrect_mean = np.mean(np.vstack(incorrect_curves), axis=0) if incorrect_curves else None
        has_diff = (
            correct_mean is not None
            and incorrect_mean is not None
            and len(correct_curves) >= min_for_diff
            and len(incorrect_curves) >= min_for_diff
        )

        fig, ax1 = plt.subplots(figsize=(22, 10))

        # Individual trajectories
        for curve in correct_curves:
            ax1.plot(
                target_x, np.minimum(curve, plot_duration_cap),
                color=CORRECT_COLOR, lw=0.6, alpha=0.18,
            )
        for curve in incorrect_curves:
            ax1.plot(
                target_x, np.minimum(curve, plot_duration_cap),
                color=INCORRECT_COLOR, lw=0.6, alpha=0.16,
            )

        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, plot_duration_cap)
        ax1.set_xticks(np.arange(0, 101, 10))
        ax1.set_xlabel("Token Position in Trajectory (%)", fontsize=30)
        ax1.set_ylabel("High-Entropy Duration (tokens)", fontsize=30)

        # Right axis: difference curve (only with enough data)
        ax2 = None
        if has_diff:
            diff_raw = correct_mean - incorrect_mean
            diff_smooth = gaussian_smooth(diff_raw, sigma=5.0)

            ax2 = ax1.twinx()
            ax2.fill_between(
                target_x, 0, diff_smooth,
                where=diff_smooth >= 0,
                color=CORRECT_COLOR, alpha=0.10, interpolate=True,
            )
            ax2.fill_between(
                target_x, 0, diff_smooth,
                where=diff_smooth < 0,
                color=INCORRECT_COLOR, alpha=0.10, interpolate=True,
            )
            ax2.plot(target_x, diff_smooth, color=DIFF_COLOR, lw=2.0, alpha=0.85)
            ax2.axhline(y=0, color="#7f8c8d", lw=0.8, ls="--", alpha=0.5)
            ax2.set_ylabel(
                r"$\Delta$ Duration (Correct $-$ Incorrect)",
                fontsize=30, color=DIFF_COLOR,
            )
            ax2.tick_params(axis="y", labelcolor=DIFF_COLOR)
            y_abs_max = max(abs(diff_smooth.min()), abs(diff_smooth.max())) * 1.3
            if y_abs_max > 0:
                ax2.set_ylim(-y_abs_max, y_abs_max)

            sign_changes = np.where(np.diff(np.sign(diff_smooth)))[0]
            for idx in sign_changes:
                ax2.axvline(x=target_x[idx], color="#95a5a6", lw=0.7, ls=":", alpha=0.5)

        # Legend
        legend_handles = []
        if correct_curves:
            legend_handles.append(
                Line2D([0], [0], color=CORRECT_COLOR, lw=1.0, alpha=0.45,
                       label=f"Correct (n={len(correct_curves)})"),
            )
        if incorrect_curves:
            legend_handles.append(
                Line2D([0], [0], color=INCORRECT_COLOR, lw=1.0, alpha=0.45,
                       label=f"Incorrect (n={len(incorrect_curves)})"),
            )
        if has_diff:
            legend_handles.append(
                Line2D([0], [0], color=DIFF_COLOR, lw=2.0,
                       label=r"$\Delta$ (Correct $-$ Incorrect)"),
            )

        ax1.legend(
            handles=legend_handles, fontsize=26, loc="upper left",
            framealpha=0.8, edgecolor="none", fancybox=False,
        )

        _style_axes_conference(ax1, ax2)
        plt.tight_layout()
        output_file = os.path.join(per_problem_dir, f"problem_{problem_id}_percent_line.png")
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        output_pdf = os.path.join(per_problem_dir, f"problem_{problem_id}_percent_line.pdf")
        plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
        plt.close()

    print(f"Saved normalized per-problem line plots to: {per_problem_dir}")


def save_mean_curve_summary(
    correct_data: Dict,
    incorrect_data: Dict,
    output_dir: str,
    num_points: int,
):
    """Save mean curves for later inspection or downstream use."""
    x_percent, correct_mean = compute_mean_curve(correct_data["trajectories"], num_points)
    _, incorrect_mean = compute_mean_curve(incorrect_data["trajectories"], num_points)

    diff_curve_smoothed = []
    if correct_mean.size and incorrect_mean.size:
        diff_raw = correct_mean - incorrect_mean
        diff_curve_smoothed = gaussian_smooth(diff_raw, sigma=5.0).tolist()

    summary = {
        "x_percent": x_percent.tolist(),
        "correct_mean_curve": correct_mean.tolist() if correct_mean.size else [],
        "incorrect_mean_curve": incorrect_mean.tolist() if incorrect_mean.size else [],
        "diff_curve_smoothed": diff_curve_smoothed,
        "num_correct_trajectories": len(correct_data["trajectories"]),
        "num_incorrect_trajectories": len(incorrect_data["trajectories"]),
    }

    output_file = os.path.join(output_dir, "high_entropy_duration_percent_mean_curves.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved mean curve summary: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize normalized high-entropy duration patterns"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to result directory with entropy_results.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for plots "
            "(default: result_dir/high_entropy_duration_percent_visualizations)"
        ),
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=5.0,
        help="Percentage of highest entropy tokens for start threshold (default: 5.0)",
    )
    parser.add_argument(
        "--bottom_percent",
        type=float,
        default=50.0,
        help="Percentage for low entropy threshold to end period (default: 50.0)",
    )
    parser.add_argument(
        "--consecutive_low_threshold",
        type=int,
        default=3,
        help="Number of consecutive low-entropy tokens to end period (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout for symbolic comparison (default: 5)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=201,
        help="Number of percentage positions used when resampling curves (default: 201)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        print(f"Error: Result directory not found: {args.result_dir}")
        return 1

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.result_dir, "high_entropy_duration_percent_visualizations"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Normalized High-Entropy Duration Visualization")
    print("=" * 80)
    print(f"Result directory: {args.result_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"High-entropy threshold: Top {args.top_percent}%")
    print(f"Low-entropy threshold: Bottom {args.bottom_percent}%")
    print(f"Consecutive low tokens to end period: {args.consecutive_low_threshold}")
    print(f"Resampled points on 0-100% axis: {args.num_points}")
    print("=" * 80)

    try:
        entropy_results = load_data(args.result_dir)
        print(f"\n✓ Loaded {len(entropy_results)} trajectories")
    except Exception as e:
        print(f"\n✗ Error loading data: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    comparator = AnswerComparator(timeout=args.timeout)

    try:
        correct_data, incorrect_data = analyze_trajectories(
            entropy_results,
            comparator,
            top_percent=args.top_percent,
            bottom_percent=args.bottom_percent,
            consecutive_low_threshold=args.consecutive_low_threshold,
        )

        n_correct = len(correct_data["trajectories"])
        n_incorrect = len(incorrect_data["trajectories"])
        n_problems = len(
            set(correct_data["by_problem"].keys()) | set(incorrect_data["by_problem"].keys())
        )

        print("\n✓ Analysis complete:")
        print(f"  - Correct trajectories: {n_correct}")
        print(f"  - Incorrect trajectories: {n_incorrect}")
        print(f"  - Total problems: {n_problems}")
    except Exception as e:
        print(f"\n✗ Error analyzing trajectories: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    try:
        print("\n" + "=" * 80)
        print("Generating visualizations...")
        print("=" * 80)

        print("\n[1/3] Generating normalized global line plot...")
        plot_normalized_duration_line_global(
            correct_data,
            incorrect_data,
            args.output_dir,
            top_percent=args.top_percent,
            num_points=args.num_points,
            plot_duration_cap=PLOT_DURATION_CAP,
        )

        print("[2/3] Generating normalized per-problem line plots...")
        plot_normalized_duration_line_per_problem(
            correct_data,
            incorrect_data,
            args.output_dir,
            top_percent=args.top_percent,
            num_points=args.num_points,
            plot_duration_cap=PLOT_DURATION_CAP,
        )

        print("[3/3] Saving mean curve summary...")
        save_mean_curve_summary(
            correct_data,
            incorrect_data,
            args.output_dir,
            num_points=args.num_points,
        )

        print("\n" + "=" * 80)
        print("✓ All visualizations complete!")
        print("=" * 80)
        print(f"\nOutput saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("  - high_entropy_duration_percent_global.png / .pdf")
        print("  - high_entropy_duration_percent_per_problem/ (png + pdf)")
        print("  - high_entropy_duration_percent_mean_curves.json")
    except Exception as e:
        print(f"\n✗ Error generating visualizations: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
