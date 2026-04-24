"""Runner logic for evaluate_answers CLI."""

from __future__ import annotations

import copy
import glob
import json
import os
from typing import Any, List

from evaluation.answer_comparator import AnswerComparator
from evaluation.answer_selector import create_answer_selector
from evaluation.evaluator import AnswerEvaluator
from evaluation.pipeline.orchestrator import run_unified_lowest_centroid_pipeline
from evaluation.trajectory_selector import create_trajectory_selector

from .cache_and_metrics import (
    compute_pass_at_k_from_cache,
    ensure_evaluation_cache,
    parse_pass_k_values,
    save_pass_at_k_results,
)
from .reporting import (
    create_visualizations,
    get_centroid_param_combinations,
    get_centroid_voting_param_combinations,
    get_centroid_weighted_voting_param_combinations,
    parse_comma_separated,
    save_evaluation_results,
)


def _display_name(path: str) -> str:
    """Pretty name for logs, resilient to trailing slash."""
    normalized = os.path.normpath(path)
    return os.path.basename(normalized) or normalized


def _has_entropy_input(path: str) -> bool:
    return os.path.exists(os.path.join(path, "entropy_results.json")) or os.path.exists(
        os.path.join(path, "entropy_results.jsonl")
    )


def _has_eval_or_entropy_input(path: str) -> bool:
    if _has_entropy_input(path):
        return True
    return os.path.exists(os.path.join(path, "evaluation_cache.json"))


def _resolve_lowest_centroid_dirs(input_dir: str) -> List[str]:
    """
    Resolve lowest-centroid target directories from a possibly nested input path.

    If input_dir itself is a leaf result dir, return [input_dir].
    Otherwise, scan one level of subdirectories and return all valid leaves.
    """
    root = os.path.normpath(input_dir)
    if _has_eval_or_entropy_input(root):
        return [root]

    if not os.path.isdir(root):
        return []

    resolved: List[str] = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if os.path.isdir(sub) and _has_eval_or_entropy_input(sub):
            resolved.append(sub)
    return resolved


def evaluate_single_directory(result_dir: str, args):
    """Evaluate one result directory with selection-specific flow."""
    if args.selection == "lowest_centroid":
        target_dirs = _resolve_lowest_centroid_dirs(result_dir)
        if not target_dirs:
            raise FileNotFoundError(
                f"No valid result directory found under: {result_dir} "
                "(expected entropy_results.json/jsonl or evaluation_cache.json)"
            )

        if len(target_dirs) > 1:
            print(f"Found {len(target_dirs)} nested result directories under: {os.path.normpath(result_dir)}")

        for idx, target_dir in enumerate(target_dirs, 1):
            print(f"\n{'='*80}")
            if len(target_dirs) > 1:
                print(f"[{idx}/{len(target_dirs)}] Evaluating: {_display_name(target_dir)}")
            else:
                print(f"Evaluating: {_display_name(target_dir)}")
            print("Selection: lowest_centroid (unified pipeline)")
            print(f"{'='*80}")
            summary = run_unified_lowest_centroid_pipeline(
                result_dir=target_dir,
                ensure_eval_cache=bool(getattr(args, "ensure_eval_cache", True)),
                timeout=args.timeout,
                force_eval_cache=args.force,
                top_percent=float(parse_comma_separated(args.centroid_top_percent, float)[0]),
                bottom_percent=float(parse_comma_separated(args.centroid_bottom_percent, float)[0]),
                consecutive_low_threshold=int(parse_comma_separated(args.centroid_consecutive_low, int)[0]),
                centroid_method=args.centroid_method,
                output_subdir="eval_lowest_centroid",
            )
            print(
                f"  ✓ lowest_centroid accuracy: {summary['overall_accuracy']*100:.2f}% "
                f"({summary['correct_count']}/{summary['total_samples']})"
            )
        return

    if args.selection == "pass_at_k":
        _run_pass_at_k(result_dir, args)
        return

    if args.selection == "entropy_centroid":
        combos = get_centroid_param_combinations(args)
        if len(combos) > 1:
            print(f"\n{'='*80}")
            print(f"Evaluating: {os.path.basename(result_dir)}")
            print(f"Running {len(combos)} entropy_centroid parameter combinations")
            print(f"{'='*80}")
        for i, (top, bottom, cons, outlier) in enumerate(combos, 1):
            if len(combos) > 1:
                print(
                    f"\n[{i}/{len(combos)}] Parameters: top={top}, bottom={bottom}, "
                    f"cons={cons}, outlier={outlier}"
                )
            modified = copy.copy(args)
            modified.centroid_top_percent = top
            modified.centroid_bottom_percent = bottom
            modified.centroid_consecutive_low = cons
            modified.centroid_outlier_threshold = outlier
            _evaluate_single_directory_core(result_dir, modified)
        return

    if args.selection == "centroid_voting":
        combos = get_centroid_voting_param_combinations(args)
        if len(combos) > 1:
            print(f"\n{'='*80}")
            print(f"Evaluating: {os.path.basename(result_dir)}")
            print(f"Running {len(combos)} centroid_voting parameter combinations")
            print(f"{'='*80}")
        for i, (top, bottom, cons, sel) in enumerate(combos, 1):
            if len(combos) > 1:
                print(
                    f"\n[{i}/{len(combos)}] Parameters: top={top}, bottom={bottom}, "
                    f"cons={cons}, select={sel}%"
                )
            modified = copy.copy(args)
            modified.centroid_top_percent = top
            modified.centroid_bottom_percent = bottom
            modified.centroid_consecutive_low = cons
            modified.centroid_select_percent = sel
            _evaluate_single_directory_core(result_dir, modified)
        return

    if args.selection == "centroid_weighted_voting":
        combos = get_centroid_weighted_voting_param_combinations(args)
        if len(combos) > 1:
            print(f"\n{'='*80}")
            print(f"Evaluating: {os.path.basename(result_dir)}")
            print(f"Running {len(combos)} centroid_weighted_voting parameter combinations")
            print(f"  Weight method: {args.weight_method}")
            print(f"  Outlier method: {args.outlier_method}")
            print(f"{'='*80}")
        for i, (top, bottom, cons, num_std, penalty) in enumerate(combos, 1):
            if len(combos) > 1:
                print(
                    f"\n[{i}/{len(combos)}] Parameters: top={top}, bottom={bottom}, cons={cons}, "
                    f"outlier_num_std={num_std}, penalty={penalty}"
                )
            modified = copy.copy(args)
            modified.centroid_top_percent = top
            modified.centroid_bottom_percent = bottom
            modified.centroid_consecutive_low = cons
            modified.outlier_num_std = num_std
            modified.outlier_penalty = penalty
            _evaluate_single_directory_core(result_dir, modified)
        return

    _evaluate_single_directory_core(result_dir, args)


def _run_pass_at_k(result_dir: str, args):
    print(f"\n{'='*80}")
    print(f"Evaluating: {os.path.basename(result_dir)}")
    print("Selection: pass_at_k")
    print(f"{'='*80}")

    comparator = AnswerComparator(timeout=args.timeout)
    print("Checking/generating evaluation cache...")
    ensure_evaluation_cache(result_dir, comparator, args)

    k_values = parse_pass_k_values(args.pass_k, result_dir)
    if not k_values:
        print("  ✗ No valid k values found for pass@k. Use --pass_k 1 or --pass_k all")
        return

    print(f"Running pass@k for k values: {k_values}")
    for k in k_values:
        output_dir = os.path.join(result_dir, f"eval_pass_at_k_k{k}")
        summary_file = os.path.join(output_dir, "answer_evaluation_summary.json")
        if os.path.exists(summary_file) and not args.force:
            print(f"  ℹ pass@{k} already complete. Skipping... (use --force to recompute)")
            continue

        stats = compute_pass_at_k_from_cache(result_dir, k)
        if not stats:
            print(f"  ⚠ pass@{k} skipped (no problems satisfy k < n)")
            continue

        print(
            f"  pass@{k}: {stats['pass_at_k_accuracy']*100:.2f}% "
            f"(problems={stats.get('pass_at_k_num_problems', 0)}, "
            f"mean_n={stats.get('pass_at_k_mean_trajectories', 0):.2f})"
        )
        save_pass_at_k_results(result_dir, output_dir, stats)


def _strategy_output_dir(result_dir: str, args) -> str:
    if args.selection == "majority_voting":
        return os.path.join(result_dir, "eval_majority_voting")
    if args.selection == "llm_majority_voting":
        return os.path.join(result_dir, "eval_llm_majority_voting")
    if args.selection == "entropy_centroid":
        params = (
            f"top{int(args.centroid_top_percent)}_bot{int(args.centroid_bottom_percent)}_"
            f"cons{args.centroid_consecutive_low}_out{args.centroid_outlier_threshold}"
        )
        return os.path.join(result_dir, f"eval_entropy_centroid_{params}")
    if args.selection == "centroid_voting":
        method = (
            "raw"
            if args.centroid_method == "raw_entropy"
            else f"top{int(args.centroid_top_percent)}_bot{int(args.centroid_bottom_percent)}_cons{args.centroid_consecutive_low}"
        )
        return os.path.join(result_dir, f"eval_centroid_voting_{method}_sel{int(args.centroid_select_percent)}")
    if args.selection == "centroid_weighted_voting":
        method = (
            "raw"
            if args.centroid_method == "raw_entropy"
            else f"top{int(args.centroid_top_percent)}_bot{int(args.centroid_bottom_percent)}_cons{args.centroid_consecutive_low}"
        )
        outlier = f"{args.outlier_method}"
        if args.outlier_method == "threshold":
            outlier += f"_{args.centroid_outlier_threshold}"
        elif args.outlier_method == "std":
            outlier += f"_{args.outlier_num_std}std"
        if float(args.outlier_penalty) != 0:
            outlier += f"_pen{args.outlier_penalty}"
        return os.path.join(result_dir, f"eval_centroid_weighted_{method}_{args.weight_method}_{outlier}")
    if args.selection == "random":
        seed = f"_seed{args.random_seed}" if args.random_seed is not None else ""
        return os.path.join(result_dir, f"eval_random{seed}")

    strategy_name = args.strategy
    if args.strategy == "early_high_entropy_centroid":
        strategy_name = f"{strategy_name}_p{int(args.entropy_percentile)}"
    return os.path.join(result_dir, f"eval_{strategy_name}_{args.selection}")


def _evaluate_single_directory_core(result_dir: str, args):
    print(f"\n{'='*80}")
    print(f"Evaluating: {os.path.basename(result_dir)}")
    print(f"{'='*80}")

    strategy_output_dir = _strategy_output_dir(result_dir, args)
    summary_file = os.path.join(strategy_output_dir, "answer_evaluation_summary.json")
    if os.path.exists(summary_file) and not args.force:
        if args.selection in ["majority_voting", "llm_majority_voting", "entropy_centroid", "centroid_voting", "random"]:
            print(f"  ℹ Evaluation with {args.selection} already complete. Skipping...")
        else:
            print(f"  ℹ Evaluation with {args.strategy} + {args.selection} already complete. Skipping...")
        print("  (Use --force to recompute)")
        return

    comparator = AnswerComparator(timeout=args.timeout)
    print("Checking/generating evaluation cache...")
    ensure_evaluation_cache(result_dir, comparator, args)

    try:
        trajectory_selector, answer_selector = _build_selectors(result_dir, args)

        comparator = AnswerComparator(timeout=args.timeout)
        evaluator = AnswerEvaluator(
            trajectory_selector=trajectory_selector,
            answer_selector=answer_selector,
            comparator=comparator,
            task_type=args.task_type,
        )

        evaluation_results = evaluator.evaluate_result_file(result_dir, selection_strategy=args.strategy)
        stats = evaluation_results["aggregate_statistics"]
        print("\n  ✓ Evaluation complete!")
        print(f"    Total samples: {stats['total_samples']}")
        print(f"    Accuracy: {stats['overall_accuracy']*100:.2f}%")
        print(f"    Correct: {stats['correct_count']}")
        print(f"    Incorrect: {stats['incorrect_count']}")
        print(f"    Failed extraction: {stats['failed_extraction_count']}")

        print(f"\nSaving results to: {strategy_output_dir}")
        save_evaluation_results(evaluation_results, strategy_output_dir)

        if not args.no_viz:
            print("Creating visualizations...")
            create_visualizations(evaluation_results, strategy_output_dir)

        if args.selection in [
            "majority_voting",
            "llm_majority_voting",
            "entropy_centroid",
            "centroid_voting",
            "random",
        ]:
            print(f"\n✓ Successfully evaluated with selection {args.selection} (no trajectory strategy needed)")
        else:
            print(f"\n✓ Successfully evaluated with strategy {args.strategy} + selection {args.selection}")
    except Exception as e:
        print(f"\n✗ Error evaluating {result_dir}: {str(e)}")
        import traceback

        traceback.print_exc()


def _build_selectors(result_dir: str, args) -> tuple[Any, Any]:
    if args.selection == "majority_voting":
        print("Running evaluation with pure majority voting")
        return None, create_answer_selector(selection_method="majority_voting", cache_dir=result_dir)
    if args.selection == "llm_majority_voting":
        print("Running evaluation with LLM majority voting")
        return None, create_answer_selector(selection_method="llm_majority_voting", cache_dir=result_dir)
    if args.selection == "entropy_centroid":
        print("Running evaluation with entropy centroid selection")
        print(
            f"  Parameters: top_percent={args.centroid_top_percent}, "
            f"bottom_percent={args.centroid_bottom_percent}, consecutive_low={args.centroid_consecutive_low}, "
            f"outlier_threshold={args.centroid_outlier_threshold}"
        )
        return None, create_answer_selector(
            selection_method="entropy_centroid",
            top_percent=args.centroid_top_percent,
            bottom_percent=args.centroid_bottom_percent,
            consecutive_low_threshold=args.centroid_consecutive_low,
            outlier_threshold=args.centroid_outlier_threshold,
        )
    if args.selection == "centroid_voting":
        print("Running evaluation with centroid voting")
        print(f"  Centroid Method: {args.centroid_method}")
        if args.centroid_method == "hep":
            print(
                f"  HEP Parameters: top_percent={args.centroid_top_percent}, "
                f"bottom_percent={args.centroid_bottom_percent}, consecutive_low={args.centroid_consecutive_low}"
            )
        print(f"  Voting: select bottom {args.centroid_select_percent}% of trajectories by centroid")
        return None, create_answer_selector(
            selection_method="centroid_voting",
            top_percent=args.centroid_top_percent,
            bottom_percent=args.centroid_bottom_percent,
            consecutive_low_threshold=args.centroid_consecutive_low,
            centroid_select_percent=args.centroid_select_percent,
            centroid_method=args.centroid_method,
            cache_dir=result_dir,
        )
    if args.selection == "random":
        print("Running evaluation with random selection")
        seed_str = f"seed={args.random_seed}" if args.random_seed is not None else "seed=random"
        print(f"  Parameters: {seed_str}, num_samples={args.random_num_samples}")
        return None, create_answer_selector(
            selection_method="random",
            seed=args.random_seed,
            num_samples=args.random_num_samples,
        )
    if args.selection == "centroid_weighted_voting":
        print("Running evaluation with centroid weighted voting")
        print(f"  Centroid Method: {args.centroid_method}")
        if args.centroid_method == "hep":
            print(
                f"  HEP Parameters: top_percent={args.centroid_top_percent}, "
                f"bottom_percent={args.centroid_bottom_percent}, consecutive_low={args.centroid_consecutive_low}"
            )
        weight_method_str = args.weight_method
        if args.weight_method == "exponential":
            weight_method_str += f" (temperature={args.exp_temperature})"
        print(f"  Weight Method: {weight_method_str}")
        print(
            f"  Outlier Detection: method={args.outlier_method}, "
            f"threshold={args.centroid_outlier_threshold}, num_std={args.outlier_num_std}, "
            f"penalty={args.outlier_penalty}"
        )
        return None, create_answer_selector(
            selection_method="centroid_weighted_voting",
            top_percent=args.centroid_top_percent,
            bottom_percent=args.centroid_bottom_percent,
            consecutive_low_threshold=args.centroid_consecutive_low,
            centroid_method=args.centroid_method,
            weight_method=args.weight_method,
            outlier_method=args.outlier_method,
            outlier_threshold=args.centroid_outlier_threshold,
            outlier_num_std=args.outlier_num_std,
            outlier_penalty=args.outlier_penalty,
            temperature=args.exp_temperature,
            cache_dir=result_dir,
        )

    selector_kwargs = {}
    if args.strategy == "early_high_entropy_centroid":
        selector_kwargs["entropy_percentile"] = args.entropy_percentile
    trajectory_selector = create_trajectory_selector(strategy_name=args.strategy, **selector_kwargs)
    answer_selector = create_answer_selector(selection_method=args.selection)
    print(f"Running evaluation with strategy: {args.strategy}, selection: {args.selection}")
    if args.strategy == "early_high_entropy_centroid":
        print(f"  Strategy parameters: entropy_percentile={args.entropy_percentile}")
    return trajectory_selector, answer_selector


def batch_evaluate(pattern: str, args):
    """Batch evaluate directories containing entropy results."""
    matched_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not matched_dirs:
        print(f"No result directories found matching pattern: {pattern}")
        return

    print(f"Found {len(matched_dirs)} result directories\n")
    result_dirs = []
    for matched_dir in matched_dirs:
        entropy_file = os.path.join(matched_dir, "entropy_results.json")
        if os.path.exists(entropy_file):
            result_dirs.append(matched_dir)
        else:
            for subdir in os.listdir(matched_dir):
                subdir_path = os.path.join(matched_dir, subdir)
                if os.path.isdir(subdir_path) and os.path.exists(os.path.join(subdir_path, "entropy_results.json")):
                    result_dirs.append(subdir_path)

    if not result_dirs:
        print("No directories with entropy_results.json found")
        return

    print(f"Found {len(result_dirs)} directories with entropy_results.json\n")
    processed = 0
    errors = 0
    for result_dir in sorted(result_dirs):
        try:
            evaluate_single_directory(result_dir, args)
            processed += 1
        except Exception as e:
            print(f"\n✗ Failed to evaluate {result_dir}: {str(e)}")
            errors += 1

    print(f"\n{'='*80}")
    print("Batch evaluation complete!")
    print(f"  Total directories: {len(result_dirs)}")
    print(f"  ✓ Successfully processed: {processed}")
    print(f"  ✗ Failed: {errors}")
    print(f"{'='*80}")

