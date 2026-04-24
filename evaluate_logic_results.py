#!/usr/bin/env python3
"""
Evaluate Logic Task Results

This script evaluates inference results from KOR-Bench and SynLogic tasks
using their respective verifiers.

Usage:
    python evaluate_logic_results.py --result_dir <path>
    python evaluate_logic_results.py --result_dir <path> --task_type korbench
"""

import argparse
import json
import os
import sys
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluation.logic_evaluator import (
    LogicEvaluator,
    KorBenchEvaluator,
    SynLogicEvaluator,
)


def load_results(result_dir: str) -> List[Dict]:
    """Load inference results from directory."""
    entropy_file = os.path.join(result_dir, 'entropy_results.json')
    
    if not os.path.exists(entropy_file):
        raise FileNotFoundError(f"entropy_results.json not found in {result_dir}")
    
    print(f"Loading results from: {entropy_file}")
    with open(entropy_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} results")
    return results


def save_evaluation_results(
    eval_results: Dict,
    output_dir: str,
    prefix: str = "logic_evaluation"
):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f'{prefix}_detailed.json')
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results['per_sample_results'], f, indent=2, ensure_ascii=False)
    print(f"Saved detailed results to: {detailed_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, f'{prefix}_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results['aggregate_statistics'], f, indent=2, ensure_ascii=False)
    print(f"Saved summary to: {summary_file}")
    
    # Save human-readable report
    report_file = os.path.join(output_dir, f'{prefix}_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        stats = eval_results['aggregate_statistics']
        
        f.write("="*60 + "\n")
        f.write("LOGIC TASK EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Samples: {stats['total_samples']}\n")
        f.write(f"Correct: {stats['correct_count']}\n")
        f.write(f"Accuracy: {stats['accuracy']*100:.2f}%\n\n")
        
        # Task breakdown if available
        if 'task_breakdown' in stats:
            f.write("-"*60 + "\n")
            f.write("Task Breakdown:\n")
            f.write("-"*60 + "\n")
            for task, task_stats in stats['task_breakdown'].items():
                f.write(f"  {task}:\n")
                f.write(f"    Accuracy: {task_stats['accuracy']*100:.2f}%\n")
                f.write(f"    Correct: {task_stats['correct']}/{task_stats['total']}\n")
            f.write("\n")
        
        # Sample results
        f.write("-"*60 + "\n")
        f.write("Sample Results (first 20):\n")
        f.write("-"*60 + "\n")
        for i, result in enumerate(eval_results['per_sample_results'][:20]):
            status = "✓" if result['is_correct'] else "✗"
            f.write(f"{status} [{result.get('id', i)}] ")
            if 'task_name' in result:
                f.write(f"({result['task_name']}) ")
            f.write("\n")
        
        f.write("="*60 + "\n")
    
    print(f"Saved report to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate logic task inference results"
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='Path to result directory with entropy_results.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for evaluation results (default: result_dir)'
    )
    parser.add_argument(
        '--task_type',
        type=str,
        default='auto',
        choices=['auto', 'korbench', 'synlogic', 'mixed'],
        help='Task type for evaluation (default: auto-detect from results)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.result_dir
    
    print("="*60)
    print("Logic Task Evaluation")
    print("="*60)
    print(f"Result directory: {args.result_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Task type: {args.task_type}")
    print("="*60)
    
    # Load results
    try:
        results = load_results(args.result_dir)
    except Exception as e:
        print(f"Error loading results: {e}")
        return 1
    
    # Determine task type if auto
    if args.task_type == 'auto':
        sources = set(r.get('source', 'unknown') for r in results)
        if 'korbench' in sources and 'synlogic' in sources:
            args.task_type = 'mixed'
        elif 'korbench' in sources:
            args.task_type = 'korbench'
        elif 'synlogic' in sources:
            args.task_type = 'synlogic'
        else:
            args.task_type = 'mixed'
        print(f"Auto-detected task type: {args.task_type}")
    
    # Create evaluator
    if args.task_type == 'korbench':
        evaluator = KorBenchEvaluator()
    elif args.task_type == 'synlogic':
        evaluator = SynLogicEvaluator()
    else:
        evaluator = LogicEvaluator()
    
    # Evaluate
    print("\nEvaluating results...")
    eval_results = evaluator.evaluate_batch(results)
    
    # Print summary
    stats = eval_results['aggregate_statistics']
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Correct: {stats['correct_count']}")
    print(f"Accuracy: {stats['accuracy']*100:.2f}%")
    
    if 'task_breakdown' in stats:
        print("\nTask Breakdown:")
        for task, task_stats in stats['task_breakdown'].items():
            print(f"  {task}: {task_stats['accuracy']*100:.2f}% ({task_stats['correct']}/{task_stats['total']})")
    
    print("="*60)
    
    # Save results
    save_evaluation_results(eval_results, args.output_dir, f'logic_evaluation_{args.task_type}')
    
    print("\nEvaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
