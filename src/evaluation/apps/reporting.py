"""Reporting, visualization, and parameter combination helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    pd = None
    plt = None
    sns = None


def save_evaluation_results(evaluation_results: dict, output_dir: str):
    """Save evaluation outputs (csv/json/report)."""
    os.makedirs(output_dir, exist_ok=True)

    per_sample = evaluation_results["per_sample_results"]
    if not VISUALIZATION_AVAILABLE:
        print("Warning: pandas not available, skipping DataFrame creation")
        return None
    df = pd.DataFrame(per_sample)
    csv_path = os.path.join(output_dir, "answer_evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved per-sample results: {csv_path}")

    aggregate_stats = evaluation_results["aggregate_statistics"]
    json_path = os.path.join(output_dir, "answer_evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"  ✓ Saved aggregate statistics: {json_path}")

    trajectory_selections = evaluation_results["trajectory_selections"]
    traj_path = os.path.join(output_dir, "trajectory_selections.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory_selections, f, indent=2)
    print(f"  ✓ Saved trajectory selections: {traj_path}")

    report_path = os.path.join(output_dir, "answer_evaluation_report.txt")
    generate_report(evaluation_results, report_path)
    print(f"  ✓ Saved evaluation report: {report_path}")


def generate_report(evaluation_results: dict, output_path: str):
    """Generate human-readable report from evaluation results."""
    stats = evaluation_results["aggregate_statistics"]
    per_sample = evaluation_results["per_sample_results"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ANSWER EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Samples: {stats['total_samples']}\n")
        f.write(f"Correct: {stats['correct_count']} ({stats['overall_accuracy']*100:.2f}%)\n")
        f.write(f"Incorrect: {stats['incorrect_count']}\n")
        f.write(f"Failed Extraction: {stats['failed_extraction_count']}\n")
        f.write(f"Low Confidence: {stats['low_confidence_count']}\n\n")

        f.write("TRAJECTORY SELECTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Single Trajectory Problems: {stats['single_trajectory_count']}\n")
        f.write(f"Multi-Trajectory Problems: {stats['multi_trajectory_count']}\n\n")

        f.write("ACCURACY BY COMPARISON METHOD\n")
        f.write("-" * 80 + "\n")
        for method, accuracy in stats["accuracy_by_comparison_method"].items():
            count = stats["comparison_method_counts"][method]
            f.write(f"  {method:15s}: {accuracy*100:6.2f}% ({count} samples)\n")
        f.write("\n")

        f.write("EXTRACTION METHOD DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for method, count in stats["extraction_method_counts"].items():
            pct = count / stats["total_samples"] * 100
            f.write(f"  {method:15s}: {count:4d} samples ({pct:.1f}%)\n")
        f.write("\n")

        if "task_type_breakdown" in stats and stats["task_type_breakdown"]:
            f.write("TASK TYPE BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            for task_type, task_stats in stats["task_type_breakdown"].items():
                f.write(
                    f"  {task_type:15s}: {task_stats['accuracy']*100:6.2f}% "
                    f"({task_stats['correct']}/{task_stats['total']} correct)\n"
                )
            f.write("\n")

        if "subtask_breakdown" in stats and stats["subtask_breakdown"]:
            f.write("SUB-TASK BREAKDOWN (Logic Tasks)\n")
            f.write("-" * 80 + "\n")
            korbench_subtasks = {}
            synlogic_subtasks = {}
            other_subtasks = {}

            for subtask_key, subtask_stats in stats["subtask_breakdown"].items():
                if subtask_key.startswith("korbench/"):
                    korbench_subtasks[subtask_key.replace("korbench/", "")] = subtask_stats
                elif subtask_key.startswith("synlogic/"):
                    synlogic_subtasks[subtask_key.replace("synlogic/", "")] = subtask_stats
                else:
                    other_subtasks[subtask_key] = subtask_stats

            if korbench_subtasks:
                f.write("\n  KOR-Bench Categories:\n")
                for subtask, sstats in sorted(korbench_subtasks.items()):
                    f.write(
                        f"    {subtask:20s}: {sstats['accuracy']*100:6.2f}% "
                        f"({sstats['correct']}/{sstats['total']} correct)\n"
                    )
            if synlogic_subtasks:
                f.write("\n  SynLogic Tasks:\n")
                for subtask, sstats in sorted(synlogic_subtasks.items()):
                    f.write(
                        f"    {subtask:20s}: {sstats['accuracy']*100:6.2f}% "
                        f"({sstats['correct']}/{sstats['total']} correct)\n"
                    )
            if other_subtasks:
                f.write("\n  Other:\n")
                for subtask, sstats in sorted(other_subtasks.items()):
                    f.write(
                        f"    {subtask:20s}: {sstats['accuracy']*100:6.2f}% "
                        f"({sstats['correct']}/{sstats['total']} correct)\n"
                    )
            f.write("\n")

        f.write("EXAMPLES OF CORRECT PREDICTIONS (first 5)\n")
        f.write("-" * 80 + "\n")
        correct_samples = [s for s in per_sample if s["is_correct"]][:5]
        for i, sample in enumerate(correct_samples, 1):
            f.write(f"\n[{i}] ID: {sample['id']}\n")
            f.write(f"    Ground Truth: {sample['ground_truth']}\n")
            f.write(f"    Generated: {sample['generated_answer']}\n")
            f.write(
                f"    Method: {sample['comparison_method']} "
                f"(confidence: {sample.get('comparison_confidence', 0):.2f})\n"
            )

        f.write("\n\nEXAMPLES OF INCORRECT PREDICTIONS (first 5)\n")
        f.write("-" * 80 + "\n")
        incorrect_samples = [s for s in per_sample if not s["is_correct"] and not s.get("error")][:5]
        for i, sample in enumerate(incorrect_samples, 1):
            f.write(f"\n[{i}] ID: {sample['id']}\n")
            f.write(f"    Ground Truth: {sample['ground_truth']}\n")
            f.write(f"    Generated: {sample.get('generated_answer', 'N/A')}\n")
            f.write(f"    Method: {sample['comparison_method']}\n")

        if stats["failed_extraction_count"] > 0:
            f.write("\n\nFAILED EXTRACTIONS (first 5)\n")
            f.write("-" * 80 + "\n")
            failed_samples = [s for s in per_sample if s.get("error")][:5]
            for i, sample in enumerate(failed_samples, 1):
                f.write(f"\n[{i}] ID: {sample['id']}\n")
                f.write(f"    Error: {sample['error']}\n")
                if "ground_truth" in sample:
                    f.write(f"    Ground Truth: {sample['ground_truth']}\n")

        if stats["low_confidence_count"] > 0:
            f.write("\n\nLOW CONFIDENCE SAMPLES (for manual review)\n")
            f.write("-" * 80 + "\n")
            f.write("Sample IDs:\n")
            for sample_id in stats["low_confidence_samples"][:20]:
                f.write(f"  - {sample_id}\n")
            if stats["low_confidence_count"] > 20:
                f.write(f"  ... and {stats['low_confidence_count'] - 20} more\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def create_visualizations(evaluation_results: dict, output_dir: str):
    """Create plots for evaluation results."""
    if not VISUALIZATION_AVAILABLE:
        print("  ⚠ Visualization skipped (matplotlib/seaborn not available)")
        return

    vis_dir = os.path.join(output_dir, "answer_evaluation_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    stats = evaluation_results["aggregate_statistics"]

    plt.figure(figsize=(8, 6))
    labels = ["Correct", "Incorrect", "Failed Extraction"]
    values = [stats["correct_count"], stats["incorrect_count"], stats["failed_extraction_count"]]
    plt.bar(labels, values, color=["#2ecc71", "#e74c3c", "#95a5a6"])
    plt.ylabel("Count")
    plt.title(f"Answer Evaluation Results\nOverall Accuracy: {stats['overall_accuracy']*100:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "accuracy_distribution.png"), dpi=300)
    plt.close()

    if stats["comparison_method_counts"]:
        plt.figure(figsize=(10, 6))
        methods = list(stats["comparison_method_counts"].keys())
        counts = list(stats["comparison_method_counts"].values())
        plt.pie(counts, labels=methods, autopct="%1.1f%%", startangle=90)
        plt.title("Comparison Method Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "comparison_method_breakdown.png"), dpi=300)
        plt.close()

    if stats["extraction_method_counts"]:
        plt.figure(figsize=(10, 6))
        methods = list(stats["extraction_method_counts"].keys())
        counts = list(stats["extraction_method_counts"].values())
        plt.bar(methods, counts, color="steelblue")
        plt.xlabel("Extraction Method")
        plt.ylabel("Count")
        plt.title("Answer Extraction Method Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "extraction_method_distribution.png"), dpi=300)
        plt.close()

    print(f"  ✓ Saved visualizations to: {vis_dir}")


def parse_comma_separated(value: str, value_type=float):
    """Parse comma-separated values."""
    if "," in value:
        return [value_type(v.strip()) for v in value.split(",")]
    return [value_type(value.strip())]


def get_centroid_param_combinations(args):
    """Generate entropy_centroid parameter combinations."""
    top_percents = parse_comma_separated(args.centroid_top_percent, float)
    bottom_percents = parse_comma_separated(args.centroid_bottom_percent, float)
    consecutive_lows = parse_comma_separated(args.centroid_consecutive_low, int)
    outlier_thresholds = parse_comma_separated(args.centroid_outlier_threshold, float)
    return [
        (top, bottom, cons, outlier)
        for top in top_percents
        for bottom in bottom_percents
        for cons in consecutive_lows
        for outlier in outlier_thresholds
    ]


def get_centroid_voting_param_combinations(args):
    """Generate centroid_voting parameter combinations."""
    top_percents = parse_comma_separated(args.centroid_top_percent, float)
    bottom_percents = parse_comma_separated(args.centroid_bottom_percent, float)
    consecutive_lows = parse_comma_separated(args.centroid_consecutive_low, int)
    select_percents = parse_comma_separated(args.centroid_select_percent, float)
    return [
        (top, bottom, cons, select)
        for top in top_percents
        for bottom in bottom_percents
        for cons in consecutive_lows
        for select in select_percents
    ]


def get_centroid_weighted_voting_param_combinations(args):
    """Generate centroid_weighted_voting parameter combinations."""
    top_percents = parse_comma_separated(args.centroid_top_percent, float)
    bottom_percents = parse_comma_separated(args.centroid_bottom_percent, float)
    consecutive_lows = parse_comma_separated(args.centroid_consecutive_low, int)
    outlier_num_stds = parse_comma_separated(args.outlier_num_std, float)
    outlier_penalties = parse_comma_separated(args.outlier_penalty, float)
    return [
        (top, bottom, cons, num_std, penalty)
        for top in top_percents
        for bottom in bottom_percents
        for cons in consecutive_lows
        for num_std in outlier_num_stds
        for penalty in outlier_penalties
    ]

