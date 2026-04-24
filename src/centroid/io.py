"""
I/O utilities for entropy centroid analysis.

Handles loading entropy results, saving centroid results,
and managing evaluation caches.
"""

import json
import os
from typing import Dict, List, Optional
from tqdm import tqdm
from evaluation.pipeline.cache_paths import canonical_cache_path


def load_data(result_dir: str) -> List[Dict]:
    """
    Load entropy results with memory optimization for large files.

    Args:
        result_dir: Path to result directory

    Returns:
        List of entropy results
    """
    entropy_file = os.path.join(result_dir, "entropy_results.json")

    if not os.path.exists(entropy_file):
        raise FileNotFoundError(f"entropy_results.json not found in {result_dir}")

    print("Loading entropy results...")

    # Try to load the file directly first
    try:
        with open(entropy_file, 'r') as f:
            entropy_results = json.load(f)
        return entropy_results
    except MemoryError:
        # If memory error, try loading line by line (for JSONL format)
        print("Memory error detected, attempting line-by-line loading...")
        entropy_results = []
        with open(entropy_file, 'r') as f:
            for line in f:
                if line.strip():
                    entropy_results.append(json.loads(line))
        return entropy_results


def load_or_create_evaluation_cache(
    result_dir: str,
    entropy_results: List[Dict],
    evaluator=None,
    comparator=None
) -> Dict[str, bool]:
    """
    Load cached evaluation results or create them if not exists.

    This ensures consistent is_correct classification across runs by caching
    the evaluation results.

    Args:
        result_dir: Path to result directory
        entropy_results: List of trajectory data
        evaluator: UnifiedEvaluator instance (preferred)
        comparator: Legacy AnswerComparator (for backward compatibility)

    Returns:
        Dict mapping trajectory_id to is_correct boolean
    """
    from .evaluator import evaluate_trajectory

    cache_file = canonical_cache_path(result_dir)

    # Try to load existing cache
    if os.path.exists(cache_file):
        print(f"Loading evaluation cache from {cache_file}...")
        try:
            with open(cache_file, 'r') as f:
                payload = json.load(f)
            trajectories = payload.get("trajectories", {}) if isinstance(payload, dict) else {}
            cache = {
                traj_id: info.get("is_correct", False)
                for traj_id, info in trajectories.items()
                if isinstance(info, dict)
            }
            print(f"  Loaded {len(cache)} cached evaluations")
            return cache
        except Exception as e:
            print(f"  Warning: Failed to load cache: {e}")

    # Create new cache
    print("Creating evaluation cache (this ensures consistent results across runs)...")
    cache = {}

    for trajectory in tqdm(entropy_results, desc="Evaluating trajectories"):
        traj_id = trajectory['id']
        cache[traj_id] = evaluate_trajectory(trajectory, comparator, evaluator)

    # Save cache with atomic write
    try:
        temp_path = cache_file + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(cache, f, indent=2)
        os.replace(temp_path, cache_file)
        print(f"  Saved evaluation cache to {cache_file}")
    except Exception as e:
        print(f"  Warning: Failed to save cache: {e}")
        # Clean up temp file if it exists
        temp_path = cache_file + '.tmp'
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return cache


def save_centroid_results(
    correct_data: Dict,
    incorrect_data: Dict,
    output_dir: str
) -> str:
    """
    Save centroid analysis results to JSON.

    Args:
        correct_data: Data for correct trajectories
        incorrect_data: Data for incorrect trajectories
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare results
    results = {
        'correct': {
            'count': len(correct_data['trajectories']),
            'trajectories': correct_data['trajectories'],
            'by_problem': {k: v for k, v in correct_data['by_problem'].items()},
        },
        'incorrect': {
            'count': len(incorrect_data['trajectories']),
            'trajectories': incorrect_data['trajectories'],
            'by_problem': {k: v for k, v in incorrect_data['by_problem'].items()},
        }
    }

    output_file = os.path.join(output_dir, 'entropy_centroid_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✓ Saved centroid results: {output_file}")
    return output_file


def check_output_exists(output_dir: str) -> bool:
    """
    Check if output files already exist.

    Args:
        output_dir: Output directory to check

    Returns:
        True if main output files exist
    """
    results_file = os.path.join(output_dir, 'entropy_centroid_results.json')
    categorization_file = os.path.join(output_dir, 'problem_categorization_summary.json')
    return os.path.exists(results_file) and os.path.exists(categorization_file)


def get_output_dir_name(
    result_dir: str,
    method: str,
    top_percent: float,
    bottom_percent: float,
    consecutive_low_threshold: int
) -> str:
    """
    Generate parameterized output directory name.

    Args:
        result_dir: Base result directory
        method: Centroid calculation method
        top_percent: High entropy threshold percentage
        bottom_percent: Low entropy threshold percentage
        consecutive_low_threshold: Consecutive low tokens threshold

    Returns:
        Full path to output directory
    """
    if method == 'raw_entropy_weighted':
        method_short = 'raw_ew'
        dir_suffix = method_short
    else:
        method_short = (method
                       .replace('weighted_average_center', 'wa_center')
                       .replace('weighted_average', 'wa')
                       .replace('moment', 'm'))
        top_int = int(top_percent)
        bottom_int = int(bottom_percent)
        dir_suffix = f"{method_short}_{top_int}_{bottom_int}_{consecutive_low_threshold}"

    return os.path.join(result_dir, f'entropy_centroid_results_{dir_suffix}')


def load_centroid_results(output_dir: str) -> Dict:
    """
    Load previously saved centroid results.

    Args:
        output_dir: Directory containing results

    Returns:
        Loaded results dictionary
    """
    results_file = os.path.join(output_dir, 'entropy_centroid_results.json')
    with open(results_file, 'r') as f:
        return json.load(f)


def get_trajectory_centroid_cache_path(
    result_dir: str,
    top_percent: float,
    bottom_percent: float,
    consecutive_low_threshold: int,
    centroid_method: str = 'hep'
) -> str:
    """
    Get the path to trajectory centroid cache file.
    
    Args:
        result_dir: Base result directory
        top_percent: High entropy threshold percentage
        bottom_percent: Low entropy threshold percentage
        consecutive_low_threshold: Consecutive low tokens threshold
        centroid_method: 'hep' or 'raw_entropy'
        
    Returns:
        Path to cache file
    """
    if centroid_method == 'raw_entropy':
        cache_name = "trajectory_centroid_cache_raw_entropy.json"
    else:
        cache_name = f"trajectory_centroid_cache_top{int(top_percent)}_bot{int(bottom_percent)}_cons{consecutive_low_threshold}.json"
    return os.path.join(result_dir, cache_name)


def _is_truncated_json(filepath: str) -> bool:
    """
    Check if a JSON file appears to be truncated (incomplete).
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        True if file appears truncated, False otherwise
    """
    try:
        with open(filepath, 'rb') as f:
            # Check last few bytes
            f.seek(0, 2)  # Go to end
            file_size = f.tell()
            if file_size < 10:
                return True  # Too small to be valid
            
            # Read last 100 bytes to check for proper ending
            check_size = min(100, file_size)
            f.seek(-check_size, 2)
            tail = f.read().decode('utf-8', errors='ignore').strip()
            
            # Valid JSON should end with } or ]
            if not tail.endswith('}') and not tail.endswith(']'):
                return True
            
        return False
    except Exception:
        return True


def _backup_corrupted_file(filepath: str):
    """
    Backup a corrupted file before regenerating.
    
    Args:
        filepath: Path to the corrupted file
    """
    if os.path.exists(filepath):
        backup_path = filepath + '.corrupted'
        try:
            os.rename(filepath, backup_path)
            print(f"  Backed up corrupted file to: {os.path.basename(backup_path)}")
        except Exception as e:
            print(f"  Warning: Could not backup corrupted file: {e}")
            # Try to delete instead
            try:
                os.remove(filepath)
                print(f"  Removed corrupted file: {os.path.basename(filepath)}")
            except Exception:
                pass


def load_trajectory_centroid_cache(
    result_dir: str,
    top_percent: float,
    bottom_percent: float,
    consecutive_low_threshold: int,
    centroid_method: str = 'hep'
) -> Dict:
    """
    Load cached trajectory centroid data with error recovery.
    
    If the cache file is corrupted (truncated due to OOM), it will be
    backed up and regenerated.
    
    Cache structure:
    {
        "parameters": {
            "top_percent": 5.0,
            "bottom_percent": 50.0,
            "consecutive_low_threshold": 3,
            "centroid_method": "hep"
        },
        "trajectories": {
            "trajectory_id": {
                "centroid": 0.45,
                "num_heps": 12,
                "trajectory_length": 1500,
                "original_id": "problem_123",
                "is_correct": true,
                "extracted_answer": "42"
            },
            ...
        }
    }
    
    Args:
        result_dir: Base result directory
        top_percent: High entropy threshold percentage
        bottom_percent: Low entropy threshold percentage
        consecutive_low_threshold: Consecutive low tokens threshold
        
    Returns:
        Cache dictionary or None if not exists or corrupted
    """
    cache_path = get_trajectory_centroid_cache_path(
        result_dir, top_percent, bottom_percent, consecutive_low_threshold, centroid_method
    )
    
    if not os.path.exists(cache_path):
        return None
    
    # Check for truncation first (quick check without full parse)
    if _is_truncated_json(cache_path):
        print(f"  Warning: {os.path.basename(cache_path)} appears truncated (possibly OOM during write)")
        _backup_corrupted_file(cache_path)
        return None
        
    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        
        # Validate cache parameters match
        params = cache.get('parameters', {})
        cache_method = params.get('centroid_method', 'hep')  # Default to hep for backward compatibility
        if centroid_method == 'raw_entropy':
            # For raw_entropy, just check method matches
            if cache_method == 'raw_entropy':
                return cache
            else:
                print(f"  Cache method mismatch (expected raw_entropy), will recompute")
                return None
        else:
            # For HEP method, check all parameters
            if (params.get('top_percent') == top_percent and
                params.get('bottom_percent') == bottom_percent and
                params.get('consecutive_low_threshold') == consecutive_low_threshold and
                cache_method in ['hep', None]):  # None for backward compatibility
                return cache
            else:
                print(f"  Cache parameters mismatch, will recompute")
                return None
    except json.JSONDecodeError as e:
        print(f"  Warning: Corrupted JSON in {os.path.basename(cache_path)}: {e}")
        _backup_corrupted_file(cache_path)
        return None
    except Exception as e:
        print(f"  Warning: Failed to load centroid cache: {e}")
        return None


def save_trajectory_centroid_cache(
    result_dir: str,
    top_percent: float,
    bottom_percent: float,
    consecutive_low_threshold: int,
    trajectory_data: Dict[str, Dict],
    centroid_method: str = 'hep'
):
    """
    Save trajectory centroid data to cache file with atomic write.
    
    Uses a temporary file and rename to prevent corruption from OOM/interruption.
    
    Args:
        result_dir: Base result directory
        top_percent: High entropy threshold percentage
        bottom_percent: Low entropy threshold percentage
        consecutive_low_threshold: Consecutive low tokens threshold
        trajectory_data: Dict mapping trajectory_id to centroid data
        centroid_method: 'hep' or 'raw_entropy'
    """
    cache_path = get_trajectory_centroid_cache_path(
        result_dir, top_percent, bottom_percent, consecutive_low_threshold, centroid_method
    )
    temp_path = cache_path + '.tmp'
    
    cache = {
        "parameters": {
            "top_percent": top_percent,
            "bottom_percent": bottom_percent,
            "consecutive_low_threshold": consecutive_low_threshold,
            "centroid_method": centroid_method
        },
        "trajectories": trajectory_data
    }
    
    try:
        # Write to temporary file first
        with open(temp_path, 'w') as f:
            json.dump(cache, f, indent=2)
        
        # Atomic rename (prevents corruption if interrupted)
        os.replace(temp_path, cache_path)
        print(f"  ✓ Saved centroid cache: {cache_path}")
    except Exception as e:
        print(f"  Warning: Failed to save centroid cache: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def get_existing_analysis_path(
    result_dir: str,
    top_percent: float,
    bottom_percent: float,
    consecutive_low_threshold: int
) -> Optional[str]:
    """
    Get path to existing entropy_centroid_results.json from analysis output.
    
    Looks for directories matching pattern:
    entropy_centroid_results_wa_center_{top}_{bottom}_{cons}/entropy_centroid_results.json
    
    Args:
        result_dir: Base result directory
        top_percent: High entropy threshold percentage
        bottom_percent: Low entropy threshold percentage
        consecutive_low_threshold: Consecutive low tokens threshold
        
    Returns:
        Path to analysis JSON file, or None if not found
    """
    # Try different possible directory naming patterns
    patterns = [
        f"entropy_centroid_results_wa_center_{int(top_percent)}_{int(bottom_percent)}_{consecutive_low_threshold}",
        f"entropy_centroid_results_weighted_average_center_{int(top_percent)}_{int(bottom_percent)}_{consecutive_low_threshold}",
    ]
    
    for pattern in patterns:
        analysis_dir = os.path.join(result_dir, pattern)
        analysis_file = os.path.join(analysis_dir, "entropy_centroid_results.json")
        if os.path.exists(analysis_file):
            return analysis_file
    
    return None


def load_from_analysis_results(
    result_dir: str,
    top_percent: float,
    bottom_percent: float,
    consecutive_low_threshold: int
) -> Optional[Dict]:
    """
    Load centroid data from existing analysis results (entropy_centroid_results.json).
    
    This provides backward compatibility with the analysis pipeline output,
    allowing the evaluation code to use pre-computed centroids from analysis.
    Also loads extracted answers from evaluation_cache_v2.json if available.
    
    Args:
        result_dir: Base result directory
        top_percent: High entropy threshold percentage
        bottom_percent: Low entropy threshold percentage
        consecutive_low_threshold: Consecutive low tokens threshold
        
    Returns:
        Cache-format dict or None if not found
    """
    analysis_path = get_existing_analysis_path(
        result_dir, top_percent, bottom_percent, consecutive_low_threshold
    )
    
    if not analysis_path:
        return None
    
    try:
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Try to load evaluation cache for extracted answers
        eval_cache = {}
        eval_cache_path = canonical_cache_path(result_dir)
        if os.path.exists(eval_cache_path):
            try:
                with open(eval_cache_path, "r") as f:
                    eval_data = json.load(f)
                eval_cache = eval_data.get("trajectories", {})
            except Exception:
                eval_cache = {}
        
        # Convert analysis format to cache format
        trajectories = {}
        
        # Process correct trajectories
        if 'correct' in analysis_data:
            correct_data = analysis_data['correct']
            if isinstance(correct_data, dict) and 'trajectories' in correct_data:
                for traj in correct_data['trajectories']:
                    traj_id = traj.get('id')
                    if traj_id:
                        # Get extracted_answer from evaluation cache if available
                        extracted_answer = eval_cache.get(traj_id, {}).get('extracted_answer')
                        trajectories[traj_id] = {
                            'centroid': traj.get('centroid'),
                            'num_heps': traj.get('num_heps', 0),
                            'trajectory_length': traj.get('trajectory_length', 0),
                            'centroid_method': 'hep',
                            'original_id': str(traj.get('original_id', str(traj_id).split('_traj_')[0] if '_traj_' in str(traj_id) else traj_id)),
                            'is_correct': True,
                            'extracted_answer': extracted_answer
                        }
        
        # Process incorrect trajectories
        if 'incorrect' in analysis_data:
            incorrect_data = analysis_data['incorrect']
            if isinstance(incorrect_data, dict) and 'trajectories' in incorrect_data:
                for traj in incorrect_data['trajectories']:
                    traj_id = traj.get('id')
                    if traj_id:
                        # Get extracted_answer from evaluation cache if available
                        extracted_answer = eval_cache.get(traj_id, {}).get('extracted_answer')
                        trajectories[traj_id] = {
                            'centroid': traj.get('centroid'),
                            'num_heps': traj.get('num_heps', 0),
                            'trajectory_length': traj.get('trajectory_length', 0),
                            'centroid_method': 'hep',
                            'original_id': str(traj.get('original_id', str(traj_id).split('_traj_')[0] if '_traj_' in str(traj_id) else traj_id)),
                            'is_correct': False,
                            'extracted_answer': extracted_answer
                        }
        
        if trajectories:
            answer_count = sum(1 for t in trajectories.values() if t.get('extracted_answer') is not None)
            print(f"  Loaded {len(trajectories)} centroids from analysis: {os.path.basename(os.path.dirname(analysis_path))}")
            if eval_cache:
                print(f"  Loaded {answer_count} extracted answers from evaluation cache")
            return {
                'parameters': {
                    'top_percent': top_percent,
                    'bottom_percent': bottom_percent,
                    'consecutive_low_threshold': consecutive_low_threshold,
                    'centroid_method': 'hep',
                    'source': 'analysis'  # Mark source
                },
                'trajectories': trajectories
            }
        
        return None
        
    except Exception as e:
        print(f"  Warning: Failed to load from analysis results: {e}")
        return None


def load_trajectory_centroid_cache_with_fallback(
    result_dir: str,
    top_percent: float,
    bottom_percent: float,
    consecutive_low_threshold: int,
    centroid_method: str = 'hep'
) -> Dict:
    """
    Load trajectory centroid data with fallback to analysis results.
    
    Priority:
    1. New cache file (trajectory_centroid_cache_*.json)
    2. Existing analysis results (entropy_centroid_results_*/entropy_centroid_results.json)
    
    Args:
        result_dir: Base result directory
        top_percent: High entropy threshold percentage
        bottom_percent: Low entropy threshold percentage
        consecutive_low_threshold: Consecutive low tokens threshold
        centroid_method: 'hep' or 'raw_entropy'
        
    Returns:
        Cache dictionary or None if nothing found
    """
    # First try the new cache
    cache = load_trajectory_centroid_cache(
        result_dir, top_percent, bottom_percent, 
        consecutive_low_threshold, centroid_method
    )
    
    if cache:
        return cache
    
    # For HEP method, try loading from existing analysis results
    if centroid_method == 'hep':
        analysis_cache = load_from_analysis_results(
            result_dir, top_percent, bottom_percent, consecutive_low_threshold
        )
        if analysis_cache:
            return analysis_cache
    
    return None


def stream_compute_centroid_cache(
    result_dir: str,
    top_percent: float,
    bottom_percent: float,
    consecutive_low_threshold: int,
    centroid_method: str = 'hep',
    extract_answer_fn=None,
    eval_cache: Dict = None
) -> Dict[str, Dict]:
    """
    Compute centroid cache by streaming through entropy_results.json.
    
    This avoids loading the entire file into memory by processing one trajectory
    at a time and only keeping the computed results.
    
    Args:
        result_dir: Directory containing entropy_results.json
        top_percent: High entropy threshold percentage (for HEP method)
        bottom_percent: Low entropy threshold percentage (for HEP method)
        consecutive_low_threshold: Consecutive low tokens threshold (for HEP method)
        centroid_method: 'hep' or 'raw_entropy'
        extract_answer_fn: Optional function to extract answer from generated text
        eval_cache: Optional evaluation cache with extracted answers
        
    Returns:
        Dict of trajectory centroid data
    """
    import numpy as np
    
    entropy_file = os.path.join(result_dir, 'entropy_results.json')
    if not os.path.exists(entropy_file):
        print(f"  Warning: entropy_results.json not found")
        return {}
    
    file_size_gb = os.path.getsize(entropy_file) / (1024**3)
    print(f"  Computing centroid cache via streaming ({file_size_gb:.1f}GB file)...")
    
    trajectory_data = {}
    count = 0
    
    def compute_entropy_thresholds(entropy_sequence):
        """Compute high and low entropy thresholds."""
        # Convert to float to handle decimal.Decimal values from some datasets
        entropies = [float(item['entropy']) for item in entropy_sequence if item.get('entropy') is not None]
        if not entropies:
            return float('inf'), float('-inf')
        high_percentile = 100.0 - top_percent
        high_threshold = np.percentile(entropies, high_percentile)
        low_threshold = np.percentile(entropies, bottom_percent)
        return high_threshold, low_threshold
    
    def compute_hep_centroid(entropy_sequence):
        """Compute HEP-based centroid."""
        if not entropy_sequence:
            return None, 0
        
        high_threshold, low_threshold = compute_entropy_thresholds(entropy_sequence)
        hep_events = []
        in_hep = False
        period_start = 0
        duration = 0
        consecutive_low = 0
        
        for i, item in enumerate(entropy_sequence):
            entropy = item.get('entropy')
            if entropy is None:
                continue
            entropy = float(entropy)  # Convert to handle decimal.Decimal
            if not in_hep:
                if entropy >= high_threshold:
                    in_hep = True
                    period_start = i
                    duration = 1
                    consecutive_low = 0
            else:
                duration += 1
                if entropy <= low_threshold:
                    consecutive_low += 1
                else:
                    consecutive_low = 0
                if consecutive_low >= consecutive_low_threshold:
                    final_duration = duration - consecutive_low_threshold
                    if final_duration > 0:
                        hep_events.append((period_start, final_duration))
                    in_hep = False
                    duration = 0
                    consecutive_low = 0
        
        if in_hep and duration > 0:
            hep_events.append((period_start, duration))
        
        # Compute centroid
        if not hep_events:
            return None, 0
        
        trajectory_length = len(entropy_sequence)
        total_moment = 0.0
        total_weight = 0.0
        for start_pos, dur in hep_events:
            weight = dur
            position = start_pos + dur / 2.0
            total_moment += weight * position
            total_weight += weight
        
        if total_weight == 0:
            return None, len(hep_events)
        
        weighted_avg = total_moment / total_weight
        centroid = weighted_avg / trajectory_length
        return centroid, len(hep_events)
    
    def compute_raw_centroid(entropy_sequence):
        """Compute raw entropy weighted centroid."""
        if not entropy_sequence:
            return None
        
        total_weighted_pos = 0.0
        total_weight = 0.0
        trajectory_length = len(entropy_sequence)
        
        for i, item in enumerate(entropy_sequence):
            entropy = item.get('entropy', 0)
            if entropy is not None and entropy > 0:
                entropy = float(entropy)  # Convert to handle decimal.Decimal from ijson
                position = i / trajectory_length
                total_weighted_pos += entropy * position
                total_weight += entropy
        
        if total_weight == 0:
            return None
        
        return total_weighted_pos / total_weight
    
    try:
        import ijson
        
        with open(entropy_file, 'rb') as f:
            parser = ijson.items(f, 'item')
            
            for sample in tqdm(parser, desc="Computing centroids (streaming)"):
                traj_id = str(sample.get('id', ''))
                original_id = str(sample.get('original_id', traj_id.split('_traj_')[0] if '_traj_' in traj_id else traj_id))
                entropy_sequence = sample.get('entropy_sequence', [])
                
                # Get extracted answer
                extracted_answer = None
                if eval_cache and traj_id in eval_cache:
                    extracted_answer = eval_cache[traj_id].get('extracted_answer')
                if extracted_answer is None and extract_answer_fn:
                    generated_text = sample.get('generated_text', '')
                    if generated_text:
                        extracted_answer = extract_answer_fn(generated_text)
                
                # Compute centroid
                if not entropy_sequence:
                    trajectory_data[traj_id] = {
                        'centroid': None,
                        'num_heps': 0,
                        'trajectory_length': 0,
                        'centroid_method': centroid_method,
                        'original_id': original_id,
                        'extracted_answer': str(extracted_answer) if extracted_answer else None
                    }
                else:
                    trajectory_length = len(entropy_sequence)
                    
                    if centroid_method == 'raw_entropy':
                        centroid = compute_raw_centroid(entropy_sequence)
                        num_heps = 0
                    else:
                        centroid, num_heps = compute_hep_centroid(entropy_sequence)
                    
                    trajectory_data[traj_id] = {
                        'centroid': centroid,
                        'num_heps': num_heps,
                        'trajectory_length': trajectory_length,
                        'centroid_method': centroid_method,
                        'original_id': original_id,
                        'extracted_answer': str(extracted_answer) if extracted_answer else None
                    }
                
                count += 1
                if count % 5000 == 0:
                    print(f"    Processed {count} trajectories...")
        
        print(f"  ✓ Computed centroids for {len(trajectory_data)} trajectories")
        
        # Save cache
        save_trajectory_centroid_cache(
            result_dir, top_percent, bottom_percent,
            consecutive_low_threshold, trajectory_data, centroid_method
        )
        
        return trajectory_data
        
    except ImportError:
        print("  Warning: ijson not installed for streaming")
        print("  Install with: pip install ijson")
        return {}
    except Exception as e:
        print(f"  Warning: Streaming centroid computation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}
