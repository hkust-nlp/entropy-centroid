"""
Apply DeepConf confidence-based trajectory selection to existing entropy_results.json.

Reads the entropy_results.json output files (which contain top_k_probs for each token),
computes DeepConf-style confidence metrics (tail_confidence, bottom_window_confidence),
and selects the best single trajectory per problem.

Uses evaluation_cache_v2.json (in the same directory) for correctness evaluation,
so it works across all datasets (math, code, etc.) without custom answer extraction.

Usage:
    python scripts/deepconf_select.py \
        --input outputs/results/config_aime_2025_allenai_Olmo-3.1-32B-Instruct_20260317_080957/entropy_results.json \
        --method tail \
        --tail_tokens 2048 \
        --window_size 2048
"""
import argparse
import ijson
import json
import numpy as np
import os
from collections import defaultdict
from typing import List, Dict


# ============= DeepConf Confidence Computation =============

def compute_deepconf_confidence_from_top_k_probs(top_k_probs_sequence: List[List[float]]) -> List[float]:
    """
    Compute DeepConf-style confidence from stored top_k_probs.

    DeepConf confidence per token: -mean(log(p_i)) for top-k tokens
    This is equivalent to deepconf/utils.py:compute_confidence but works on probabilities
    instead of vLLM Logprob objects.
    """
    confs = []
    for probs in top_k_probs_sequence:
        if probs:
            log_probs = np.log(np.array(probs, dtype=np.float64) + 1e-30)  # avoid log(0)
            conf = -np.mean(log_probs)
            confs.append(round(float(conf), 3))
    return confs


def calculate_tail_confidence(confs: List[float], tail_tokens: int = 2048) -> float:
    """Mean confidence of the last N tokens (DeepConf tail_confidence)."""
    if not confs:
        return 0.0
    tail = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
    return float(np.mean(tail))


def calculate_bottom_window_confidence(confs: List[float], window_size: int = 2048,
                                        bottom_percent: float = 0.1) -> float:
    """
    Sliding window mean confidence, return average of bottom percentile windows.
    Faithful to deepconf/utils.py:calculate_bottom_window_confidence.
    """
    if not confs:
        return 0.0
    if len(confs) < window_size:
        return float(np.mean(confs))

    # Efficient sliding window
    window_means = []
    current_sum = sum(confs[:window_size])
    window_means.append(current_sum / window_size)
    for i in range(1, len(confs) - window_size + 1):
        current_sum = current_sum - confs[i - 1] + confs[i + window_size - 1]
        window_means.append(current_sum / window_size)

    if not window_means:
        return 0.0

    if bottom_percent == -1:  # min_window mode
        return float(min(window_means))

    num_bottom = max(1, int(len(window_means) * bottom_percent))
    if num_bottom == 1:
        return float(min(window_means))
    else:
        bottom_means = np.partition(window_means, num_bottom - 1)[:num_bottom]
        return float(np.mean(bottom_means))


def calculate_mean_confidence(confs: List[float]) -> float:
    """Mean confidence across all tokens."""
    if not confs:
        return 0.0
    return float(np.mean(confs))


def calculate_min_window_confidence(confs: List[float], window_size: int = 2048) -> float:
    """Minimum of sliding window means (DeepConf min_conf)."""
    return calculate_bottom_window_confidence(confs, window_size, bottom_percent=-1)


# ============= Trajectory Processing =============

def process_trajectory(item: Dict, tail_tokens: int, window_size: int) -> Dict:
    """Process a single trajectory from entropy_results.json."""
    entropy_seq = item.get('entropy_sequence', [])

    # Extract top_k_probs from each token
    top_k_probs_seq = []
    for token_data in entropy_seq:
        probs = token_data.get('top_k_probs', [])
        top_k_probs_seq.append([float(p) for p in probs])

    # Compute DeepConf confidence sequence
    confs = compute_deepconf_confidence_from_top_k_probs(top_k_probs_seq)

    # Compute all confidence metrics
    tail_conf = calculate_tail_confidence(confs, tail_tokens)
    bottom_window_conf = calculate_bottom_window_confidence(confs, window_size)
    min_window_conf = calculate_min_window_confidence(confs, window_size)
    mean_conf = calculate_mean_confidence(confs)

    return {
        'id': item.get('id'),
        'original_id': item.get('original_id'),
        'trajectory_index': item.get('trajectory_index'),
        'solution': item.get('solution'),
        'num_tokens': len(entropy_seq),
        # DeepConf confidence metrics
        'tail_confidence': tail_conf,
        'bottom_window_confidence': bottom_window_conf,
        'min_window_confidence': min_window_conf,
        'mean_confidence': mean_conf,
    }


def load_eval_cache(input_path: str) -> Dict:
    """
    Load evaluation_cache_v2.json from the same directory as input entropy_results.
    Returns a dict mapping trajectory id to its evaluation info.
    Handles both formats:
      - results/: keys like '0_traj_0', has is_correct/extracted_answer/ground_truth
      - tau2_bench/: keys like '48' or '22_traj_1', has is_correct or reward
    """
    eval_cache_path = os.path.join(os.path.dirname(input_path), 'evaluation_cache.json')
    if not os.path.exists(eval_cache_path):
        print(f"WARNING: evaluation_cache.json not found at {eval_cache_path}")
        return {}

    with open(eval_cache_path, 'r') as f:
        cache = json.load(f)

    raw_trajectories = cache.get('trajectories', {})

    # Normalize: ensure all keys are strings, and unify is_correct field
    trajectories = {}
    for key, val in raw_trajectories.items():
        str_key = str(key)
        # Some datasets use 'reward' instead of 'is_correct'
        if 'is_correct' not in val and 'reward' in val:
            val['is_correct'] = bool(val['reward'] and val['reward'] > 0)
        trajectories[str_key] = val

    print(f"Loaded evaluation cache: {len(trajectories)} trajectories, task_type={cache.get('task_type', 'unknown')}")
    return trajectories


def iter_entropy_items(input_path: str):
    """
    Iterate over items from entropy results file.
    Supports both .json (JSON array, streamed via ijson) and .jsonl (one JSON per line).
    """
    if input_path.endswith('.jsonl'):
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(input_path, 'rb') as f:
            for item in ijson.items(f, 'item'):
                yield item


# ============= Main =============

def main():
    parser = argparse.ArgumentParser(description='DeepConf trajectory selection from entropy_results.json')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to entropy_results.json or entropy_results.jsonl')
    parser.add_argument('--method', type=str, default='tail',
                        choices=['tail', 'bottom_window', 'min_window', 'mean'],
                        help='Confidence metric for selection (default: tail)')
    parser.add_argument('--tail_tokens', type=int, default=2048,
                        help='Number of tail tokens for tail_confidence (default: 2048)')
    parser.add_argument('--window_size', type=int, default=2048,
                        help='Sliding window size (default: 2048)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: auto-generated)')
    parser.add_argument('--top_n', type=int, default=1,
                        help='Select top N trajectories per problem (default: 1)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output file')
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        args.output = os.path.join(input_dir, f'deepconf_selection_{args.method}_top{args.top_n}.json')

    # Skip if output already exists
    if os.path.exists(args.output) and not args.force:
        print(f"SKIP: {args.output} already exists (use --force to overwrite)")
        return

    # Metric key mapping
    metric_key = {
        'tail': 'tail_confidence',
        'bottom_window': 'bottom_window_confidence',
        'min_window': 'min_window_confidence',
        'mean': 'mean_confidence',
    }[args.method]

    print(f"Input: {args.input}")
    print(f"Method: {args.method} (key: {metric_key})")
    print(f"Tail tokens: {args.tail_tokens}, Window size: {args.window_size}")
    print(f"Selecting top {args.top_n} trajectory(s) per problem")
    print()

    # Load evaluation cache
    eval_cache = load_eval_cache(args.input)
    if not eval_cache:
        print("ERROR: Cannot proceed without evaluation_cache.json")
        return

    # Process all trajectories
    problems = defaultdict(list)  # original_id -> list of processed trajectories
    total_count = 0

    print("Processing trajectories...")
    for item in iter_entropy_items(args.input):
        result = process_trajectory(item, args.tail_tokens, args.window_size)
        traj_id = str(result['id'])
        # Attach evaluation info from cache
        if traj_id in eval_cache:
            eval_info = eval_cache[traj_id]
            result['is_correct'] = eval_info.get('is_correct', False)
            result['extracted_answer'] = eval_info.get('extracted_answer')
            result['ground_truth'] = eval_info.get('ground_truth')
        else:
            result['is_correct'] = None
            result['extracted_answer'] = None
            result['ground_truth'] = None

        # Normalize original_id to comparable type
        orig_id = result['original_id']
        problems[orig_id].append(result)
        total_count += 1
        if total_count % 100 == 0:
            print(f"  Processed {total_count} trajectories...")

    print(f"\nTotal: {total_count} trajectories across {len(problems)} problems")

    # Select best trajectory per problem
    selections = []
    correct_count = 0
    total_problems = 0

    for original_id in sorted(problems.keys()):
        trajs = problems[original_id]
        # Use all trajectories for confidence ranking (not just those with answers)
        ranked_trajs = sorted(trajs, key=lambda t: t[metric_key], reverse=True)

        selected = ranked_trajs[:args.top_n]
        ground_truth = selected[0].get('ground_truth') or str(trajs[0].get('solution', '')).strip()

        # Check correctness using evaluation cache
        is_correct = selected[0].get('is_correct', False)
        if is_correct:
            correct_count += 1
        total_problems += 1

        # Confidence stats for all trajectories of this problem
        all_confs = [t[metric_key] for t in trajs]

        # Count how many trajectories are correct in this problem
        num_correct_trajs = sum(1 for t in trajs if t.get('is_correct', False))

        selection_info = {
            'original_id': original_id,
            'ground_truth': ground_truth,
            'num_trajectories': len(trajs),
            'num_correct_trajs': num_correct_trajs,
            'selected_trajectory_index': selected[0]['trajectory_index'],
            'selected_answer': selected[0].get('extracted_answer'),
            'selected_confidence': selected[0][metric_key],
            'is_correct': is_correct,
            'confidence_stats': {
                'mean': float(np.mean(all_confs)),
                'std': float(np.std(all_confs)),
                'min': float(np.min(all_confs)),
                'max': float(np.max(all_confs)),
            },
        }

        if args.top_n > 1:
            selection_info['selected_answers'] = [
                {
                    'trajectory_index': s['trajectory_index'],
                    'answer': s.get('extracted_answer'),
                    'confidence': s[metric_key],
                    'is_correct': s.get('is_correct', False),
                }
                for s in selected
            ]

        selections.append(selection_info)

        status = "OK" if is_correct else "WRONG"
        print(f"  Problem {original_id}: {status} | "
              f"answer={selected[0].get('extracted_answer')} (gt={ground_truth}) | "
              f"conf={selected[0][metric_key]:.4f} "
              f"(range [{np.min(all_confs):.4f}, {np.max(all_confs):.4f}]) | "
              f"correct_trajs={num_correct_trajs}/{len(trajs)}")

    # Summary
    accuracy = correct_count / total_problems if total_problems > 0 else 0
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.method} confidence, top-{args.top_n} selection")
    print(f"{'='*60}")
    print(f"Accuracy: {correct_count}/{total_problems} = {accuracy:.1%}")
    print(f"Output: {args.output}")

    # Save results
    output_data = {
        'config': {
            'input': args.input,
            'method': args.method,
            'metric_key': metric_key,
            'tail_tokens': args.tail_tokens,
            'window_size': args.window_size,
            'top_n': args.top_n,
            'select_highest': True,
        },
        'summary': {
            'total_problems': total_problems,
            'correct': correct_count,
            'accuracy': accuracy,
        },
        'selections': selections,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
