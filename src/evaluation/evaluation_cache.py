"""
Evaluation cache management for trajectory correctness and extracted answers.

This module provides a unified caching system for:
1. Trajectory correctness (is_correct)
2. Extracted answers
3. Centroid information (when available)

Cache file format (v2):
{
    "version": 2,
    "task_type": "math",  # or "korbench", "synlogic"
    "trajectories": {
        "trajectory_id": {
            "is_correct": true/false,
            "extracted_answer": "42",
            "ground_truth": "42",
            "original_id": "problem_001",
            "comparison_method": "grader_exact"
        },
        ...
    }
}
"""

import json
import os
from decimal import Decimal
from typing import Dict, Optional, List, Callable
from tqdm import tqdm

from .answer_extractor import extract_boxed_answer
from .pipeline.cache_paths import (
    CANONICAL_EVALUATION_CACHE,
    canonical_cache_path,
)


def _parse_synlogic_task_from_id(traj_id: str) -> str:
    """Parse SynLogic task name from trajectory ID string.

    ID format: synlogic_{task_name}_{sample_id}[_traj_{n}]
    sample_id is a UUID or numeric index (never contains underscores).
    Task names may contain underscores (e.g., 'boolean_expressions').
    """
    if not traj_id.startswith('synlogic_'):
        return 'unknown'
    rest = traj_id[len('synlogic_'):]
    traj_suffix_idx = rest.find('_traj_')
    if traj_suffix_idx >= 0:
        rest = rest[:traj_suffix_idx]
    last_underscore = rest.rfind('_')
    if last_underscore > 0:
        return rest[:last_underscore]
    return rest or 'unknown'


def _sanitize_decimals(obj):
    """Recursively convert Decimal to float in nested dicts/lists.

    ijson returns Decimal instead of float, which breaks SynLogic verifiers
    (Data.from_json_dict) and JSON serialization.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _sanitize_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_decimals(v) for v in obj]
    return obj


def _should_use_unified_task_evaluator(
    task_type: Optional[str],
    unified_evaluator,
    extracted_answer,
    ground_truth,
) -> bool:
    """
    Decide whether to use the task-native unified evaluator.

    SynLogic verifier-based evaluation should run whenever the unified evaluator
    is available, even if no textual ground truth was stored in the dataset.
    Other task types still require the extracted answer and ground truth pair
    before running their task-specific comparison path.
    """
    if unified_evaluator is None:
        return False
    if task_type == 'synlogic':
        return True
    if task_type in ('korbench', 'amo_bench'):
        return extracted_answer is not None and ground_truth is not None
    return False


CACHE_VERSION = 2
CACHE_FILENAME = CANONICAL_EVALUATION_CACHE


def _amo_api_configured() -> bool:
    """Return True if AMO-Bench description evaluation API key is configured."""
    return any(
        os.environ.get(key)
        for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "AMO_BENCH_API_KEY")
    )


def _get_amo_bench_answer_type(evaluator, sample: Dict) -> str:
    """
    Resolve AMO-Bench answer_type from dataset metadata when available.
    Defaults to 'number' if not found.
    """
    answer_type = "number"
    if evaluator is None:
        return answer_type

    try:
        amo_wrapper = evaluator.get_evaluator("amo_bench")
        amo_core = getattr(amo_wrapper, "evaluator", None)
        if amo_core is None:
            return answer_type

        original_id = str(sample.get("original_id", ""))
        try:
            qid = int(original_id)
        except (ValueError, TypeError):
            traj_id = str(sample.get("id", ""))
            try:
                qid = int(traj_id.split("_")[0])
            except (ValueError, IndexError):
                qid = None

        if qid is not None:
            info = amo_core.get_question_info(qid)
            if info:
                answer_type = info.get("answer_type", "number")
    except Exception:
        pass

    return answer_type


def get_cache_path(result_dir: str, version: int = 2) -> str:
    """
    Get path to evaluation cache file.

    Always returns canonical `evaluation_cache.json`.
    """
    _ = version
    return canonical_cache_path(result_dir)


def load_evaluation_cache(result_dir: str) -> Optional[Dict]:
    """
    Load canonical evaluation cache.
    
    Args:
        result_dir: Directory containing the cache file
        
    Returns:
        Cache dictionary, or None if not found or invalid
    """
    path = canonical_cache_path(result_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            cache = json.load(f)
        if isinstance(cache, dict) and "trajectories" in cache:
            return cache
    except Exception:
        return None
    return None


def save_evaluation_cache(
    result_dir: str,
    trajectories: Dict[str, Dict],
    task_type: str = 'math'
):
    """
    Save evaluation cache in v2 format with atomic write.
    
    Args:
        result_dir: Directory to save cache
        trajectories: Dict mapping trajectory_id to evaluation info
        task_type: Type of task (math, korbench, synlogic)
    """
    cache_path = canonical_cache_path(result_dir)
    temp_path = cache_path + '.tmp'
    
    try:
        # Write to temporary file first
        # Use custom encoder to handle Decimal from ijson streaming
        from decimal import Decimal

        class _CacheEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, Decimal):
                    return float(o)
                return super().default(o)

        # Full cache payload persisted to disk.
        cache = {
            "version": CACHE_VERSION,
            "task_type": task_type,
            "trajectories": trajectories,
        }

        with open(temp_path, 'w') as f:
            json.dump(cache, f, indent=2, cls=_CacheEncoder)
        
        # Atomic rename (prevents partial writes)
        os.replace(temp_path, cache_path)
        print(f"  ✓ Saved evaluation cache: {len(trajectories)} trajectories")
    except Exception as e:
        print(f"  Warning: Failed to save evaluation cache: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def evaluate_all_trajectories(
    entropy_results: List[Dict],
    comparator,
    extract_answer_fn: Callable,
    detect_task_type_fn: Callable,
    result_dir: str = None,
    use_cache: bool = True,
    evaluator=None,
) -> Dict[str, Dict]:
    """
    Evaluate all trajectories and cache results.
    
    This is the core function that evaluates every trajectory's answer
    against the ground truth. Results are cached for reuse.
    
    Args:
        entropy_results: List of trajectory data from entropy_results.json
        comparator: AnswerComparator instance for comparing answers
        extract_answer_fn: Function to extract answer from generated text
        detect_task_type_fn: Function to detect task type from sample
        result_dir: Directory for caching (if None, no caching)
        use_cache: Whether to use cached results
        
    Returns:
        Dict mapping trajectory_id to evaluation info:
        {
            "trajectory_id": {
                "is_correct": bool,
                "extracted_answer": str,
                "ground_truth": str,
                "original_id": str,
                "comparison_method": str
            }
        }
    """
    # Try to load from cache first
    if use_cache and result_dir:
        cache = load_evaluation_cache(result_dir)
        if cache and 'trajectories' in cache:
            cached_trajs = cache['trajectories']
            return cached_trajs
    
    # Evaluate all trajectories
    print(f"  Evaluating {len(entropy_results)} trajectories...")
    
    trajectories = {}
    task_type = None
    unified_evaluator = None  # For korbench/synlogic/amo_bench — same as centroid/evaluator.py
    amo_api_configured = _amo_api_configured()
    
    for sample in tqdm(entropy_results, desc="  Evaluating trajectories"):
        traj_id = str(sample.get('id', ''))
        original_id = sample.get('original_id', traj_id.split('_traj_')[0] if '_traj_' in traj_id else traj_id)
        original_id = str(original_id)
        
        # Detect task type from first sample
        if task_type is None:
            task_type = detect_task_type_fn(sample)
            # Initialize UnifiedEvaluator for ALL non-math tasks
            # This ensures the same evaluation logic as centroid/evaluator.py
            if task_type in ('korbench', 'synlogic', 'amo_bench'):
                try:
                    from centroid.evaluator import UnifiedEvaluator
                    unified_evaluator = UnifiedEvaluator(
                        timeout=5,
                        skip_amo_description=not amo_api_configured
                    )
                except ImportError:
                    pass
            elif task_type == 'amo_bench' and evaluator is None:
                # Try to initialize unified evaluator for AMO-Bench
                try:
                    from centroid.evaluator import UnifiedEvaluator
                    evaluator = UnifiedEvaluator(
                        timeout=5,
                        skip_amo_description=not amo_api_configured
                    )
                except Exception:
                    evaluator = None
        
        # Extract answer from generated text
        generated_text = sample.get('generated_text', '')
        answer_type = None
        if task_type == 'amo_bench':
            answer_type = _get_amo_bench_answer_type(evaluator, sample)
            amo_wrapper = evaluator.get_evaluator('amo_bench') if evaluator is not None else None
            amo_core = getattr(amo_wrapper, 'evaluator', None) if amo_wrapper else None
            if amo_core is not None:
                extracted_answer = amo_core.extract_answer(generated_text, answer_type)
            else:
                extracted_answer = extract_answer_fn(generated_text)
        else:
            extracted_answer = extract_answer_fn(generated_text)
        
        # Get ground truth
        ground_truth = sample.get('solution') or sample.get('answer') or sample.get('final_answer')
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0] if ground_truth else None

        # Extract the final answer from ground truth if it contains \boxed{}
        # (indicating a full solution text, not a clean answer).
        # This check is purely based on \boxed{} presence — no dependency on
        # task_type, which may be unreliable (detect_task_type can return None,
        # v1 caches lack task_type, etc.). \boxed{} is an unambiguous signal.
        gt_for_comparison = ground_truth
        if ground_truth is not None:
            gt_str = str(ground_truth)
            if '\\boxed' in gt_str:
                gt_extracted, _, _ = extract_boxed_answer(gt_str)
                if gt_extracted is not None:
                    gt_for_comparison = gt_extracted

        # Compare answers - use appropriate evaluator for task type
        is_correct = False
        comparison_method = 'none'

        can_use_unified_evaluator = _should_use_unified_task_evaluator(
            task_type,
            unified_evaluator,
            extracted_answer,
            ground_truth,
        )
        can_use_text_comparator = extracted_answer is not None and ground_truth is not None

        if can_use_unified_evaluator or can_use_text_comparator:
            try:
                # For korbench/synlogic/amo_bench: use UnifiedEvaluator (same as centroid/evaluator.py)
                # This ensures consistent evaluation with full access to generated_text,
                # original_data (rule_id, idx), and game_data.
                if can_use_unified_evaluator:
                    # Sanitize Decimal from ijson before passing to evaluator
                    clean_sample = _sanitize_decimals(sample)
                    is_correct = unified_evaluator.evaluate(clean_sample)
                    if task_type == 'korbench':
                        category = sample.get('category', 'logic')
                        if not category:
                            parts = traj_id.split('_')
                            if len(parts) >= 2:
                                category = parts[1]
                        comparison_method = f'korbench_{category}'
                    elif task_type == 'synlogic':
                        task_name = sample.get('task_name', 'unknown')
                        if not task_name or task_name == 'unknown':
                            task_name = _parse_synlogic_task_from_id(traj_id)
                        comparison_method = f'synlogic_{task_name}'
                    else:
                        answer_type = answer_type or 'number'
                        comparison_method = f'amo_bench_{answer_type}'
                else:
                    # Default: use AnswerComparator for math tasks
                    comparison_result = comparator.compare(
                        gt_for_comparison,
                        extracted_answer
                    )
                    is_correct = comparison_result.get('is_correct', False)
                    comparison_method = comparison_result.get('comparison_method', 'unknown')
            except Exception as e:
                comparison_method = f'error: {str(e)[:50]}'
        
        entry = {
            'is_correct': is_correct,
            'extracted_answer': str(extracted_answer) if extracted_answer is not None else None,
            'ground_truth': str(gt_for_comparison) if gt_for_comparison is not None else None,
            'original_id': original_id,
            'comparison_method': comparison_method
        }
        # Store game_data for synlogic tasks (needed for verifier-based re-evaluation)
        if task_type == 'synlogic':
            game_data = sample.get('game_data')
            if game_data:
                entry['game_data'] = _sanitize_decimals(game_data)
        
        trajectories[traj_id] = entry
    
    # Save cache
    if result_dir:
        save_evaluation_cache(result_dir, trajectories, task_type or 'math')
    
    return trajectories


def merge_centroid_and_evaluation_cache(
    centroid_cache: Dict[str, Dict],
    evaluation_cache: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Merge centroid cache with evaluation cache.
    
    Used when entropy_centroid_results.json exists (has centroid but no answers)
    and evaluation_cache.json exists (has correctness but maybe no answers).
    
    Args:
        centroid_cache: From load_from_analysis_results() - has centroid, is_correct
        evaluation_cache: From load_evaluation_cache() - has is_correct, extracted_answer
        
    Returns:
        Merged cache with both centroid and answer info
    """
    merged = {}
    
    for traj_id, centroid_info in centroid_cache.items():
        merged[traj_id] = {
            'centroid': centroid_info.get('centroid'),
            'num_heps': centroid_info.get('num_heps', 0),
            'trajectory_length': centroid_info.get('trajectory_length', 0),
            'centroid_method': centroid_info.get('centroid_method', 'hep'),
            'original_id': centroid_info.get('original_id'),
            'is_correct': centroid_info.get('is_correct'),
            'extracted_answer': None  # Default
        }
        
        # Add answer info from evaluation cache
        if traj_id in evaluation_cache:
            eval_info = evaluation_cache[traj_id]
            merged[traj_id]['extracted_answer'] = eval_info.get('extracted_answer')
            # Use evaluation cache correctness if available (more reliable)
            if eval_info.get('is_correct') is not None:
                merged[traj_id]['is_correct'] = eval_info.get('is_correct')
    
    return merged


def get_correctness_from_cache(cache: Dict[str, Dict]) -> Dict[str, bool]:
    """
    Extract trajectory_id -> is_correct mapping from any cache format.
    
    Works with both evaluation_cache and centroid_cache formats.
    """
    result = {}
    
    if 'trajectories' in cache:
        trajs = cache['trajectories']
    else:
        trajs = cache
    
    for traj_id, info in trajs.items():
        if isinstance(info, bool):
            # Old v1 format
            result[traj_id] = info
        elif isinstance(info, dict):
            result[traj_id] = info.get('is_correct', False)
    
    return result


def get_answers_from_cache(cache: Dict[str, Dict]) -> Dict[str, str]:
    """
    Extract trajectory_id -> extracted_answer mapping from cache.
    """
    result = {}
    
    if 'trajectories' in cache:
        trajs = cache['trajectories']
    else:
        trajs = cache
    
    for traj_id, info in trajs.items():
        if isinstance(info, dict):
            answer = info.get('extracted_answer')
            if answer is not None:
                result[traj_id] = answer
    
    return result


# ============================================================================
# Lightweight Answer Extraction (without loading full entropy_results.json)
# ============================================================================

def create_lightweight_answer_cache(
    result_dir: str,
    comparator,
    extract_answer_fn,
    detect_task_type_fn,
    chunk_size: int = 1000,
    evaluator=None,
) -> Dict[str, Dict]:
    """
    Create evaluation cache by streaming through entropy_results.json.
    
    This avoids loading the entire file into memory by processing in chunks.
    Only extracts the fields needed for answer evaluation.
    
    Args:
        result_dir: Directory containing entropy_results.json
        comparator: AnswerComparator for comparing answers
        extract_answer_fn: Function to extract answer from generated text
        detect_task_type_fn: Function to detect task type
        chunk_size: Number of samples to process at a time
        
    Returns:
        Dict of trajectory evaluations
    """
    import ijson  # For streaming JSON parsing
    
    entropy_file = os.path.join(result_dir, 'entropy_results.json')
    if not os.path.exists(entropy_file):
        return {}
    
    print(f"  Creating lightweight answer cache (streaming)...")

    trajectories = {}
    task_type = None
    unified_evaluator = None  # Same as centroid/evaluator.py for consistent evaluation
    amo_api_configured = _amo_api_configured()
    count = 0

    try:
        with open(entropy_file, 'rb') as f:
            # Stream through the JSON array
            parser = ijson.items(f, 'item')

            for sample in parser:
                traj_id = str(sample.get('id', ''))
                original_id = sample.get('original_id', traj_id.split('_traj_')[0] if '_traj_' in traj_id else traj_id)
                original_id = str(original_id)
                
                # Detect task type once and init unified evaluator
                if task_type is None:
                    task_type = detect_task_type_fn(sample)
                    if task_type in ('korbench', 'synlogic', 'amo_bench'):
                        try:
                            from centroid.evaluator import UnifiedEvaluator
                            unified_evaluator = UnifiedEvaluator(
                                timeout=5,
                                skip_amo_description=not amo_api_configured
                            )
                        except ImportError:
                            pass
                
                # Extract answer from generated text
                generated_text = sample.get('generated_text', '')
                answer_type = None
                if task_type == 'amo_bench':
                    answer_type = _get_amo_bench_answer_type(evaluator, sample)
                    amo_wrapper = evaluator.get_evaluator('amo_bench') if evaluator is not None else None
                    amo_core = getattr(amo_wrapper, 'evaluator', None) if amo_wrapper else None
                    if amo_core is not None:
                        extracted_answer = amo_core.extract_answer(generated_text, answer_type)
                    else:
                        extracted_answer = extract_answer_fn(generated_text)
                else:
                    extracted_answer = extract_answer_fn(generated_text)
                
                # Get ground truth
                ground_truth = sample.get('solution') or sample.get('answer') or sample.get('final_answer')
                if isinstance(ground_truth, list):
                    ground_truth = ground_truth[0] if ground_truth else None

                # Extract final answer from ground truth if it contains \boxed{}
                # Purely based on \boxed{} presence — no task_type dependency
                gt_for_comparison = ground_truth
                if ground_truth is not None:
                    gt_str = str(ground_truth)
                    if '\\boxed' in gt_str:
                        gt_extracted, _, _ = extract_boxed_answer(gt_str)
                        if gt_extracted is not None:
                            gt_for_comparison = gt_extracted

                # Compare answers - use appropriate evaluator for task type
                
                # Compare answers - use appropriate evaluator for task type
                is_correct = False
                comparison_method = 'none'

                can_use_unified_evaluator = _should_use_unified_task_evaluator(
                    task_type,
                    unified_evaluator,
                    extracted_answer,
                    ground_truth,
                )
                can_use_text_comparator = extracted_answer is not None and ground_truth is not None

                if can_use_unified_evaluator or can_use_text_comparator:
                    try:
                        if can_use_unified_evaluator:
                            # Sanitize Decimal from ijson before passing to evaluator
                            clean_sample = _sanitize_decimals(sample)
                            is_correct = unified_evaluator.evaluate(clean_sample)
                            if task_type == 'korbench':
                                category = sample.get('category', 'logic')
                                if not category:
                                    parts = traj_id.split('_')
                                    if len(parts) >= 2:
                                        category = parts[1]
                                comparison_method = f'korbench_{category}'
                            elif task_type == 'synlogic':
                                task_name = sample.get('task_name', 'unknown')
                                if not task_name or task_name == 'unknown':
                                    task_name = _parse_synlogic_task_from_id(traj_id)
                                comparison_method = f'synlogic_{task_name}'
                            else:
                                comparison_method = f'amo_bench_{answer_type or "number"}'
                        else:
                            comparison_result = comparator.compare(
                                ground_truth,
                                extracted_answer
                            )
                            is_correct = comparison_result.get('is_correct', False)
                            comparison_method = comparison_result.get('comparison_method', 'unknown')
                    except Exception as e:
                        comparison_method = f'error: {str(e)[:50]}'
                
                entry = {
                    'is_correct': is_correct,
                    'extracted_answer': str(extracted_answer) if extracted_answer is not None else None,
                    'ground_truth': str(gt_for_comparison) if gt_for_comparison is not None else None,
                    'original_id': original_id,
                    'comparison_method': comparison_method
                }
                # Store game_data for synlogic tasks
                if task_type == 'synlogic':
                    gd = sample.get('game_data')
                    if gd:
                        entry['game_data'] = _sanitize_decimals(gd)
                
                trajectories[traj_id] = entry
                
                count += 1
                if count % 1000 == 0:
                    print(f"    Processed {count} trajectories...")
        
        print(f"  ✓ Created cache for {len(trajectories)} trajectories")
        
    except ImportError:
        print("  Warning: ijson not installed, falling back to full load")
        return {}
    except Exception as e:
        print(f"  Warning: Streaming failed ({e}), falling back to full load")
        return {}
    
    # Save cache
    if trajectories:
        save_evaluation_cache(result_dir, trajectories, task_type or 'math')
    
    return trajectories


def extract_answers_only(result_dir: str, output_file: str = None) -> str:
    """
    Extract only trajectory IDs, generated text, and ground truth from entropy_results.json.
    
    Creates a lightweight file (~1-5% of original size) that can be used for
    answer extraction without loading the full entropy data.
    
    Args:
        result_dir: Directory containing entropy_results.json
        output_file: Output file path (default: answers_only.json in result_dir)
        
    Returns:
        Path to the created file
    """
    import ijson
    
    entropy_file = os.path.join(result_dir, 'entropy_results.json')
    if output_file is None:
        output_file = os.path.join(result_dir, 'answers_only.json')
    
    if os.path.exists(output_file):
        print(f"  Lightweight answers file already exists: {output_file}")
        return output_file
    
    print(f"  Extracting answers from entropy_results.json...")
    
    count = 0
    temp_path = output_file + '.tmp'
    
    try:
        with open(entropy_file, 'rb') as f, open(temp_path, 'w') as out_f:
            parser = ijson.items(f, 'item')
            out_f.write('[')
            first = True

            for sample in parser:
                # Only keep essential fields
                lightweight_sample = {
                    'id': sample.get('id'),
                    'original_id': sample.get('original_id'),
                    'generated_text': sample.get('generated_text', ''),
                    'solution': sample.get('solution'),
                    'answer': sample.get('answer'),
                    'final_answer': sample.get('final_answer'),
                    'data_source': sample.get('data_source'),
                }

                if not first:
                    out_f.write(',\n')
                json.dump(lightweight_sample, out_f)
                first = False

                count += 1
                if count % 1000 == 0:
                    print(f"    Processed {count} samples...")

            out_f.write(']')

        os.replace(temp_path, output_file)
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ Created lightweight file: {output_file} ({file_size_mb:.1f} MB)")
        
        return output_file
        
    except ImportError:
        print("  Warning: ijson not installed. Install with: pip install ijson")
        return None
    except Exception as e:
        print(f"  Warning: Failed to create lightweight file: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return None
