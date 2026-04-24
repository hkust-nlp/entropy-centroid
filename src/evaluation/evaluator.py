"""
Main answer evaluator orchestrator.

Coordinates extraction, normalization, comparison, and trajectory selection
to produce comprehensive evaluation results.

Supports both math and logic tasks:
- Math tasks: Symbolic comparison with multi-answer support
- KOR-Bench: [[answer]] format, string comparison
- SynLogic: <answer> format, verifier-based comparison
"""

import json
import os
import re
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm

from .answer_extractor import extract_answer, extract_answer_with_metadata
from .answer_normalizer import normalize_answer
from .answer_comparator import AnswerComparator
from .trajectory_selector import TrajectorySelector, create_trajectory_selector
from .answer_selector import AnswerSelector, create_answer_selector
from .pipeline.cache_paths import canonical_cache_path

# Import unified evaluator from centroid module
# Use direct file import to avoid matplotlib dependency from centroid/__init__.py
UNIFIED_EVALUATOR_AVAILABLE = False
UnifiedEvaluator = None


def _import_centroid_evaluator():
    """Dynamically import centroid evaluator module."""
    global UNIFIED_EVALUATOR_AVAILABLE, UnifiedEvaluator
    
    import importlib.util
    
    # Find centroid/evaluator.py relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    centroid_evaluator_path = os.path.join(current_dir, '..', 'centroid', 'evaluator.py')
    
    if not os.path.exists(centroid_evaluator_path):
        return None
    
    try:
        spec = importlib.util.spec_from_file_location('centroid_evaluator', centroid_evaluator_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        UNIFIED_EVALUATOR_AVAILABLE = True
        UnifiedEvaluator = module.UnifiedEvaluator
        return module
    except Exception:
        return None


# Try to import the centroid evaluator module
_centroid_module = _import_centroid_evaluator()


def _amo_api_configured() -> bool:
    """Return True if AMO-Bench description evaluation API key is configured."""
    return any(
        os.environ.get(key)
        for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "AMO_BENCH_API_KEY")
    )


def detect_task_type(trajectory: Dict) -> str:
    """
    Detect task type from trajectory ID.
    
    Args:
        trajectory: Trajectory data with 'id' field
        
    Returns:
        Task type: 'korbench', 'synlogic', 'amo_bench', or 'math'
    """
    if _centroid_module is not None:
        return _centroid_module.detect_task_type(trajectory)
    
    # Fallback implementation
    traj_id = str(trajectory.get('id', ''))
    original_id = str(trajectory.get('original_id', ''))

    if traj_id.startswith('korbench_'):
        return 'korbench'
    elif traj_id.startswith('synlogic_'):
        return 'synlogic'

    # AMO-Bench detection
    if original_id:
        try:
            int(original_id)
            if traj_id and '_traj_' in traj_id:
                try:
                    prefix = traj_id.split('_traj_')[0]
                    int(prefix)
                    prompt = trajectory.get('prompt', '') or trajectory.get('problem', '')
                    if prompt and ('### The final answer is:' in prompt or 
                                   'final answer should be given as precisely' in prompt):
                        return 'amo_bench'
                except (ValueError, IndexError):
                    pass
        except (ValueError, TypeError):
            pass
    
    return 'math'


def extract_korbench_category(trajectory: Dict) -> str:
    """Extract KOR-Bench category from trajectory ID."""
    if _centroid_module is not None:
        return _centroid_module.extract_korbench_category(trajectory)
    
    # Fallback implementation
    traj_id = str(trajectory.get('id', ''))
    parts = traj_id.split('_')
    if len(parts) >= 2:
        return parts[1]
    return 'unknown'


def extract_synlogic_task(trajectory: Dict) -> str:
    """Extract SynLogic task name from trajectory ID.

    Supports task names with underscores (e.g., 'boolean_expressions').
    """
    if _centroid_module is not None:
        return _centroid_module.extract_synlogic_task(trajectory)

    # Fallback implementation: prefer task_name field, then parse ID
    task_name = trajectory.get('task_name', '')
    if task_name and task_name != 'unknown':
        return task_name

    traj_id = str(trajectory.get('id', ''))
    return _parse_synlogic_task_from_id(traj_id)


def _parse_synlogic_task_from_id(traj_id: str) -> str:
    """Parse SynLogic task name from trajectory ID string.

    ID format: synlogic_{task_name}_{sample_id}[_traj_{n}]
    sample_id is a UUID or numeric index (never contains underscores).
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


class AnswerEvaluator:
    """
    Main evaluator class for both math and logic tasks.
    
    Automatically detects task type (math, korbench, synlogic) from trajectory ID
    and uses the appropriate evaluation method.
    """

    def __init__(
        self,
        trajectory_selector: Optional[TrajectorySelector] = None,
        answer_selector: Optional[AnswerSelector] = None,
        comparator: Optional[AnswerComparator] = None,
        task_type: str = 'auto',
    ):
        """
        Initialize evaluator.

        Args:
            trajectory_selector: Trajectory selection strategy (for scoring trajectories)
            answer_selector: Answer selection strategy (Best-of-N or Weighted Voting)
            comparator: Answer comparator (used for math tasks)
            task_type: Task type hint ('auto', 'math', 'korbench', 'synlogic')
                       'auto' will detect task type from trajectory ID
        """
        self.trajectory_selector = trajectory_selector or create_trajectory_selector()
        self.answer_selector = answer_selector or create_answer_selector()
        self.comparator = comparator or AnswerComparator()
        self.task_type = task_type
        
        # Initialize unified evaluator if available
        if UNIFIED_EVALUATOR_AVAILABLE:
            self.unified_evaluator = UnifiedEvaluator(
                timeout=self.comparator.timeout,
                skip_amo_description=not _amo_api_configured()
            )
        else:
            self.unified_evaluator = None

    def _detect_sample_task_type(self, sample: Dict) -> str:
        """
        Detect task type for a sample.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Task type: 'math', 'korbench', or 'synlogic'
        """
        if self.task_type != 'auto':
            return self.task_type
        return detect_task_type(sample)

    def _get_answer_extractor(self, task_type: str):
        """
        Get the appropriate answer extraction function for a task type.
        
        Args:
            task_type: 'math', 'korbench', 'synlogic', or 'amo_bench'
            
        Returns:
            Callable that extracts answer from generated text
        """
        if task_type == 'korbench':
            return self._extract_korbench_answer
        elif task_type == 'synlogic':
            return self._extract_synlogic_answer
        elif task_type == 'amo_bench':
            return self._extract_amo_bench_answer
        else:
            return extract_answer

    def _extract_korbench_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from KOR-Bench format [[answer]].
        
        Args:
            text: Generated text
            
        Returns:
            Extracted answer or None
        """
        if self.unified_evaluator is not None:
            evaluator = self.unified_evaluator.get_evaluator('korbench')
            return evaluator.extract_answer(text)
        
        # Fallback extraction
        matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'\[\s*(.*?)\s*\]', text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _extract_synlogic_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from SynLogic format <answer>...</answer>.
        
        Args:
            text: Generated text
            
        Returns:
            Extracted answer or None
        """
        if self.unified_evaluator is not None:
            evaluator = self.unified_evaluator.get_evaluator('synlogic')
            return evaluator.extract_answer(text)
        
        # Fallback extraction
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_amo_bench_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from AMO-Bench format.
        
        Args:
            text: Generated text
            
        Returns:
            Extracted answer or None
        """
        if self.unified_evaluator is not None:
            evaluator = self.unified_evaluator.get_evaluator('amo_bench')
            return evaluator.extract_answer(text)
        
        # Fallback extraction using boxed notation
        from .answer_extractor import extract_amo_bench_answer
        return extract_amo_bench_answer(text)

    def evaluate_single_sample(
        self,
        sample: Dict,
        extract_metadata: bool = True,
        cached_ground_truth: str = None
    ) -> Dict:
        """
        Evaluate a single sample with automatic task type detection.

        Args:
            sample: Sample dictionary with 'problem', 'solution', 'generated_text'
            extract_metadata: Whether to include extraction metadata
            cached_ground_truth: Ground truth from evaluation cache (preferred over sample['solution'])

        Returns:
            Evaluation result dictionary
        """
        sample_id = sample.get('id', 'unknown')
        task_type = self._detect_sample_task_type(sample)
        
        # Use unified evaluator for logic tasks if available
        if task_type in ['korbench', 'synlogic'] and self.unified_evaluator is not None:
            return self._evaluate_logic_sample(sample, task_type, cached_ground_truth)
        
        # Use unified evaluator for AMO-Bench if available
        if task_type == 'amo_bench' and self.unified_evaluator is not None:
            return self._evaluate_amo_bench_sample(sample, cached_ground_truth)
        
        # Math task or fallback: use original evaluation logic
        return self._evaluate_math_sample(sample, extract_metadata, cached_ground_truth)

    def _extract_subtask_info(self, sample: Dict, task_type: str) -> str:
        """
        Extract sub-task/category information from a sample.
        
        For KOR-Bench: Extract category (cipher, logic, operation, puzzle, counterfactual)
        For SynLogic: Extract task name (campsite, sudoku, etc.)
        
        Args:
            sample: Sample dictionary
            task_type: 'korbench' or 'synlogic'
            
        Returns:
            Sub-task name/category
        """
        sample_id = sample.get('id', '')
        
        if task_type == 'korbench':
            # Try to get from sample's category field first
            category = sample.get('category', '')
            if category:
                return category
            
            # Extract from ID: korbench_{category}_{idx}_traj_{n}
            category = extract_korbench_category(sample)
            return category if category != 'unknown' else 'other'
            
        elif task_type == 'synlogic':
            # Try to get from sample's task_name field first
            task_name = sample.get('task_name', '')
            if task_name:
                return task_name
            
            # Try to extract from data_source field (e.g., 'val/campsite')
            original_data = sample.get('original_data', {})
            data_source = original_data.get('data_source', '')
            if data_source:
                parts = data_source.split('/')
                if len(parts) >= 2:
                    return parts[-1].lower()
            
            # Extract from ID: synlogic_{task}_{idx}_traj_{n}
            task_name = extract_synlogic_task(sample)
            return task_name if task_name != 'unknown' else 'other'
        
        return 'unknown'

    def _evaluate_logic_sample(self, sample: Dict, task_type: str, cached_ground_truth: str = None) -> Dict:
        """
        Evaluate a logic task sample (KOR-Bench or SynLogic).
        
        Args:
            sample: Sample dictionary
            task_type: 'korbench' or 'synlogic'
            cached_ground_truth: Ground truth from evaluation cache (preferred over sample['solution'])
            
        Returns:
            Evaluation result dictionary
        """
        sample_id = sample.get('id', 'unknown')
        ground_truth = cached_ground_truth if cached_ground_truth is not None else sample.get('solution', '')
        generated_text = sample.get('generated_text', '')
        
        # Use unified evaluator
        is_correct = self.unified_evaluator.evaluate(sample)
        
        # Get evaluator for extraction
        evaluator = self.unified_evaluator.get_evaluator(task_type)
        extracted_answer = evaluator.extract_answer(generated_text)
        
        # Extract sub-task information
        subtask = self._extract_subtask_info(sample, task_type)
        
        # Determine comparison method based on task type
        comparison_method = f'{task_type}_{subtask}'
        
        return {
            'id': sample_id,
            'is_correct': is_correct,
            'ground_truth': str(ground_truth),
            'generated_answer': str(extracted_answer) if extracted_answer else '',
            'comparison_method': comparison_method,
            'comparison_confidence': 1.0 if is_correct else 0.0,
            'extraction_method': f'{task_type}_native',
            'extraction_confidence': 1.0 if extracted_answer else 0.0,
            'task_type': task_type,
            'subtask': subtask,  # Add subtask info
            'error': None,
        }

    def _evaluate_amo_bench_sample(self, sample: Dict, cached_ground_truth: str = None) -> Dict:
        """
        Evaluate an AMO-Bench sample.
        
        Args:
            sample: Sample dictionary
            cached_ground_truth: Ground truth from evaluation cache (preferred over sample['solution'])
            
        Returns:
            Evaluation result dictionary
        """
        sample_id = sample.get('id', 'unknown')
        ground_truth = cached_ground_truth if cached_ground_truth is not None else sample.get('solution', '')
        generated_text = sample.get('generated_text', '')
        
        # Use unified evaluator
        is_correct = self.unified_evaluator.evaluate(sample)
        
        # Get evaluator for extraction
        evaluator = self.unified_evaluator.get_evaluator('amo_bench')
        extracted_answer = evaluator.extract_answer(generated_text)
        
        # Get answer_type if available from evaluator
        answer_type = 'number'  # Default
        if hasattr(evaluator, 'evaluator') and evaluator.evaluator is not None:
            original_id = sample.get('original_id', '')
            try:
                qid = int(original_id)
                info = evaluator.evaluator.get_question_info(qid)
                if info:
                    answer_type = info.get('answer_type', 'number')
            except (ValueError, TypeError):
                pass
        
        return {
            'id': sample_id,
            'is_correct': is_correct,
            'ground_truth': str(ground_truth),
            'generated_answer': str(extracted_answer) if extracted_answer else '',
            'comparison_method': f'amo_bench_{answer_type}',
            'comparison_confidence': 1.0 if is_correct else 0.0,
            'extraction_method': 'amo_bench_native',
            'extraction_confidence': 1.0 if extracted_answer else 0.0,
            'task_type': 'amo_bench',
            'subtask': answer_type,  # Use answer_type as subtask
            'error': None,
        }

    def _answers_match_for_cache(self, answer1: str, answer2: str) -> bool:
        """
        Check if two answers are equivalent for cache lookup purposes.
        
        Uses simple string normalization to determine if the voting winner
        matches the cached answer, allowing us to reuse the cached is_correct.
        
        Args:
            answer1: First answer (e.g., winning_answer from voting)
            answer2: Second answer (e.g., extracted_answer from cache)
            
        Returns:
            True if answers match sufficiently for cache reuse
        """
        if answer1 is None or answer2 is None:
            return False
        
        # Normalize both answers for comparison
        def normalize(s):
            if s is None:
                return ''
            s = str(s).strip().lower()
            # Remove common formatting variations
            s = ' '.join(s.split())  # Collapse whitespace
            return s
        
        norm1 = normalize(answer1)
        norm2 = normalize(answer2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return True
        
        # If one is empty and the other is not, they don't match
        if not norm1 or not norm2:
            return False
        
        # For numeric answers, try numeric comparison
        try:
            # Handle potential LaTeX/math formatting
            val1 = float(norm1.replace('$', '').replace('\\', '').strip())
            val2 = float(norm2.replace('$', '').replace('\\', '').strip())
            if abs(val1 - val2) < 1e-9:
                return True
        except (ValueError, TypeError):
            pass
        
        return False

    def _evaluate_with_pre_extracted_answer(
        self, 
        sample: Dict, 
        pre_extracted_answer: str,
        task_type: str,
        cached_ground_truth: str = None
    ) -> Dict:
        """
        Evaluate a sample when the answer has already been extracted (e.g., from cache).
        
        This is used in lightweight loading mode where generated_text may be empty
        but we already have the extracted answer from the selection process.
        
        Args:
            sample: Sample dictionary with 'solution' field
            pre_extracted_answer: Pre-extracted answer from cache/selection
            task_type: Task type for comparison method
            cached_ground_truth: Ground truth from evaluation cache (preferred over sample['solution'])
            
        Returns:
            Evaluation result dictionary
        """
        sample_id = sample.get('id', 'unknown')
        
        # Use cached ground truth if available (more reliable, already processed)
        # This avoids re-processing ground truth which may have different escaping
        if cached_ground_truth is not None:
            gt_answer = cached_ground_truth
        else:
            ground_truth = sample.get('solution', '')
            # Extract answer from ground truth if needed
            gt_answer = extract_answer(ground_truth) if ground_truth else ground_truth
            if gt_answer is None:
                gt_answer = ground_truth
        
        # Compare pre-extracted answer with ground truth
        try:
            comparison = self.comparator.compare(gt_answer, pre_extracted_answer)
            
            return {
                'id': sample_id,
                'is_correct': comparison['is_correct'],
                'ground_truth': comparison['ground_truth_normalized'],
                'generated_answer': comparison['generated_normalized'],
                'comparison_method': comparison['comparison_method'],
                'comparison_confidence': comparison['confidence'],
                'extraction_method': 'cached',
                'extraction_confidence': 1.0,
                'task_type': task_type,
                'error': comparison.get('error_message'),
            }
        except Exception as e:
            return {
                'id': sample_id,
                'is_correct': False,
                'ground_truth': str(gt_answer),
                'generated_answer': str(pre_extracted_answer) if pre_extracted_answer else '',
                'comparison_method': 'error',
                'comparison_confidence': 0.0,
                'extraction_method': 'cached',
                'extraction_confidence': 1.0,
                'task_type': task_type,
                'error': f'Comparison error: {str(e)}'
            }

    def _evaluate_math_sample(self, sample: Dict, extract_metadata: bool = True, cached_ground_truth: str = None) -> Dict:
        """
        Evaluate a math task sample using math grader.
        
        Args:
            sample: Sample dictionary
            extract_metadata: Whether to include extraction metadata
            cached_ground_truth: Ground truth from evaluation cache (preferred over sample['solution'])

        Returns:
            Evaluation result dictionary
        """
        sample_id = sample.get('id', 'unknown')

        # Use cached ground truth if available (already processed, avoids escaping issues)
        if cached_ground_truth is not None:
            gt_answer = cached_ground_truth
        else:
            # Extract ground truth answer
            gt_text = sample.get('solution') or ''
            # Handle list format (e.g., ['answer']) - take first element
            if isinstance(gt_text, list):
                gt_text = gt_text[0] if gt_text else ''
            # Ensure it's a string (but don't use str() on list to avoid escaping issues)
            if not isinstance(gt_text, str):
                gt_text = str(gt_text)
            gt_answer = extract_answer(gt_text)

        # Extract generated answer (ensure it's a string)
        gen_text = sample.get('generated_text') or ''
        if not isinstance(gen_text, str):
            gen_text = str(gen_text)

        if extract_metadata:
            gen_extraction = extract_answer_with_metadata(gen_text)
            gen_answer = gen_extraction['answer']
            extraction_method = gen_extraction['method']
            extraction_confidence = gen_extraction['confidence']
        else:
            gen_answer = extract_answer(gen_text)
            extraction_method = 'unknown'
            extraction_confidence = 1.0

        # Handle extraction failures
        if gt_answer is None:
            return {
                'id': sample_id,
                'is_correct': False,
                'error': 'ground_truth_extraction_failed',
                'extraction_method': extraction_method,
                'extraction_confidence': extraction_confidence,
                'task_type': 'math',
            }

        if gen_answer is None:
            return {
                'id': sample_id,
                'is_correct': False,
                'error': 'generated_extraction_failed',
                'ground_truth': gt_answer,
                'extraction_method': extraction_method,
                'extraction_confidence': extraction_confidence,
                'task_type': 'math',
            }

        # Compare answers
        comparison = self.comparator.compare(gt_answer, gen_answer)

        # Combine results
        result = {
            'id': sample_id,
            'is_correct': comparison['is_correct'],
            'ground_truth': comparison['ground_truth_normalized'],
            'generated_answer': comparison['generated_normalized'],
            'comparison_method': comparison['comparison_method'],
            'comparison_confidence': comparison['confidence'],
            'extraction_method': extraction_method,
            'extraction_confidence': extraction_confidence,
            'task_type': 'math',
            'error': comparison.get('error_message'),
        }

        return result

    def evaluate_result_file(
        self,
        result_dir: str,
        selection_strategy: str = "entropy_mean"
    ) -> Dict:
        """
        Evaluate all samples in a result directory.

        Args:
            result_dir: Path to result directory with entropy_results.json
            selection_strategy: Strategy for trajectory selection

        Returns:
            Complete evaluation results with aggregate statistics
        """
        # Load data files
        entropy_file = os.path.join(result_dir, 'entropy_results.json')
        step_file = os.path.join(result_dir, 'step_divisions.json')

        if not os.path.exists(entropy_file):
            raise FileNotFoundError(f"entropy_results.json not found in {result_dir}")

        # Check if we can use lightweight loading (for methods with caches)
        selector_name = (
            self.answer_selector.get_name() 
            if hasattr(self.answer_selector, 'get_name') 
            else ''
        )
        
        # Methods that only need evaluation cache (no entropy data needed)
        eval_cache_only_methods = ['majority_voting', 'llm_majority_voting', 'random']
        # Methods that need centroid cache (computed from entropy data)
        centroid_methods = ['centroid_voting', 'centroid_weighted']
        
        is_eval_cache_method = any(m in selector_name for m in eval_cache_only_methods)
        is_centroid_method = any(m in selector_name for m in centroid_methods)
        
        # Check if evaluation cache exists (required for lightweight mode)
        eval_cache_file = canonical_cache_path(result_dir)
        has_eval_cache = eval_cache_file is not None
        if has_eval_cache and not os.path.exists(eval_cache_file):
            has_eval_cache = False
            eval_cache_file = None
        
        # For centroid methods, check if centroid cache is available
        has_valid_centroid_cache = False
        if is_centroid_method and hasattr(self.answer_selector, '_load_cache'):
            self.answer_selector._load_cache()
            if self.answer_selector._centroid_cache:
                has_valid_centroid_cache = True
                print(f"  Found centroid cache with {len(self.answer_selector._centroid_cache)} trajectories")
        
        # Use lightweight loading if possible (avoids loading full entropy sequences)
        entropy_results = []
        eval_cache = None  # Will be loaded if available
        
        # Determine if we can use lightweight loading:
        # - eval_cache_only_methods: just need evaluation cache
        # - centroid_methods: need valid centroid cache (which includes extracted answers)
        can_use_lightweight = (
            (is_eval_cache_method and has_eval_cache) or
            (is_centroid_method and has_valid_centroid_cache and has_eval_cache)
        )
        use_lightweight = can_use_lightweight
        
        if use_lightweight:
            print("  Using lightweight loading mode (caches available)")
            # Load only trajectory metadata from evaluation cache
            with open(eval_cache_file, 'r') as f:
                eval_cache = json.load(f)
            
            trajectories_data = eval_cache.get('trajectories', {})
            for traj_id, traj_info in trajectories_data.items():
                original_id = traj_info.get('original_id', traj_id.split('_traj_')[0] if '_traj_' in traj_id else traj_id)
                # Create minimal sample structure
                entropy_results.append({
                    'id': traj_id,
                    'original_id': original_id,
                    'trajectory_index': int(traj_id.split('_traj_')[-1]) if '_traj_' in traj_id else 0,
                    'generated_text': '',  # Not needed when using cache
                    'solution': traj_info.get('ground_truth', ''),
                    # Don't include entropy_sequence - saves memory!
                })
        else:
            # Full loading mode (needed for methods without caches)
            file_size_gb = os.path.getsize(entropy_file) / (1024**3)
            print(f"  Loading entropy_results.json ({file_size_gb:.1f}GB)...")
            
            # For centroid methods with large files: use streaming centroid computation
            # This computes and saves centroid cache without loading everything into memory
            if is_centroid_method and file_size_gb > 50 and not has_valid_centroid_cache:
                print(f"  Using streaming centroid computation for large file...")
                try:
                    from centroid.io import stream_compute_centroid_cache
                    
                    # Load eval cache for extracted answers
                    eval_cache_for_stream = {}
                    if has_eval_cache and eval_cache_file:
                        with open(eval_cache_file, 'r') as f:
                            eval_data = json.load(f)
                        eval_cache_for_stream = eval_data.get('trajectories', {})
                    
                    # Compute centroid cache via streaming (saves to file)
                    centroid_cache = stream_compute_centroid_cache(
                        result_dir=result_dir,
                        top_percent=self.answer_selector.top_percent if hasattr(self.answer_selector, 'top_percent') else 5.0,
                        bottom_percent=self.answer_selector.bottom_percent if hasattr(self.answer_selector, 'bottom_percent') else 50.0,
                        consecutive_low_threshold=self.answer_selector.consecutive_low_threshold if hasattr(self.answer_selector, 'consecutive_low_threshold') else 3,
                        centroid_method=self.answer_selector.centroid_method if hasattr(self.answer_selector, 'centroid_method') else 'hep',
                        eval_cache=eval_cache_for_stream
                    )
                    
                    if centroid_cache:
                        # Update the selector's cache
                        self.answer_selector._centroid_cache = centroid_cache
                        self.answer_selector._cache_loaded = True
                        has_valid_centroid_cache = True
                        
                        # Now we can use lightweight loading
                        print("  Centroid cache computed, switching to lightweight mode...")
                        with open(eval_cache_file, 'r') as f:
                            eval_cache = json.load(f)
                        
                        trajectories_data = eval_cache.get('trajectories', {})
                        for traj_id, traj_info in trajectories_data.items():
                            original_id = traj_info.get('original_id', traj_id.split('_traj_')[0] if '_traj_' in traj_id else traj_id)
                            entropy_results.append({
                                'id': traj_id,
                                'original_id': original_id,
                                'trajectory_index': int(traj_id.split('_traj_')[-1]) if '_traj_' in traj_id else 0,
                                'generated_text': '',
                                'solution': traj_info.get('ground_truth', ''),
                            })
                    else:
                        raise Exception("Streaming centroid computation returned empty")
                        
                except Exception as e:
                    print(f"  Warning: Streaming centroid failed ({e}), falling back to standard streaming")
                    # Fall through to regular streaming below
                    entropy_results = []
            
            # Regular streaming for large files (or fallback)
            if not entropy_results and file_size_gb > 50:
                print(f"  Using streaming field extraction...")
                try:
                    import ijson
                    
                    # Determine which fields we actually need
                    need_entropy_seq = is_centroid_method and not has_valid_centroid_cache
                    
                    with open(entropy_file, 'rb') as f:
                        parser = ijson.items(f, 'item')
                        for item in tqdm(parser, desc="Streaming load"):
                            filtered_item = {
                                'id': item.get('id'),
                                'original_id': item.get('original_id', item.get('id')),
                                'trajectory_index': item.get('trajectory_index', 0),
                                'generated_text': item.get('generated_text', ''),
                                'solution': item.get('solution', ''),
                            }
                            if need_entropy_seq:
                                filtered_item['entropy_sequence'] = item.get('entropy_sequence', [])
                            entropy_results.append(filtered_item)
                    
                    print(f"  Loaded {len(entropy_results)} trajectories via streaming")
                    
                except ImportError:
                    print("  Warning: ijson not installed, falling back to full load")
                    print("  Install with: pip install ijson")
                    with open(entropy_file, 'r') as f:
                        entropy_results = json.load(f)
                except Exception as e:
                    print(f"  Warning: Streaming failed ({e}), falling back to full load")
                    with open(entropy_file, 'r') as f:
                        entropy_results = json.load(f)
            
            # Standard full load for smaller files
            if not entropy_results:
                if file_size_gb > 10:
                    print(f"  Warning: Large file, this may take a while...")
                with open(entropy_file, 'r') as f:
                    entropy_results = json.load(f)

        # Load step divisions if available
        step_divisions = []

        if os.path.exists(step_file) and not use_lightweight:
            with open(step_file, 'r') as f:
                step_divisions = json.load(f)

        # Load evaluation cache if not already loaded (needed for answer lookup in lightweight scenarios)
        if eval_cache is None and has_eval_cache and eval_cache_file:
            try:
                with open(eval_cache_file, 'r') as f:
                    eval_cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                eval_cache = None

        # Group trajectories by original_id (for Best-of-N)
        trajectories_by_problem = defaultdict(list)

        for sample in entropy_results:
            sample_id = sample.get('id')
            original_id = sample.get('original_id', sample_id)
            trajectory_idx = sample.get('trajectory_index', 0)

            trajectories_by_problem[original_id].append({
                'id': sample_id,
                'original_id': original_id,
                'trajectory_index': trajectory_idx,
                'sample': sample
            })

        # Evaluate each problem
        per_sample_results = []
        selected_trajectories = {}

        print(f"Evaluating {len(trajectories_by_problem)} problems...")

        # Detect overall task type for logging
        if entropy_results:
            first_task_type = self._detect_sample_task_type(entropy_results[0])
            print(f"Detected task type: {first_task_type}")

        for original_id, trajectories in tqdm(trajectories_by_problem.items()):
            # Select best trajectory if multiple
            should_use_selector = len(trajectories) > 1
            
            if should_use_selector:
                # Get task-appropriate answer extraction function
                sample_task_type = self._detect_sample_task_type(trajectories[0]['sample'])
                answer_extractor = self._get_answer_extractor(sample_task_type)
                
                # Use answer selector to determine final answer
                selection = self.answer_selector.select_answer(
                    trajectories,
                    self.trajectory_selector,
                    entropy_results,
                    step_divisions,
                    [],
                    answer_extractor  # Pass task-appropriate answer extraction function
                )
                selected_traj = selection['selected_trajectory']
                selection_metadata = selection['selection_metadata']
                selected_trajectories[original_id] = {
                    'selected_id': selected_traj['id'],
                    'selection_method': selection['selection_method'],
                    'selection_metadata': selection_metadata,
                    'num_trajectories': len(trajectories)
                }
                
                # If the selector already extracted the answer (e.g., from cache),
                # inject it into the sample to avoid re-extraction from empty generated_text
                pre_extracted_answer = selection_metadata.get('winning_answer')
            else:
                # Single trajectory - select it by default
                selected_traj = trajectories[0]
                selected_trajectories[original_id] = {
                    'selected_id': selected_traj['id'],
                    'selection_method': 'single_trajectory',
                    'selection_metadata': {},
                    'num_trajectories': 1
                }
                pre_extracted_answer = None
                sample_task_type = self._detect_sample_task_type(selected_traj['sample'])

            # Evaluate selected trajectory
            sample = selected_traj['sample']
            traj_id = selected_traj['id']
            
            # Get cached ground truth if available (preferred over sample['solution'])
            cached_ground_truth = None
            cached_info = None
            cached_is_correct = None
            cached_answer = None
            comparison_method = ''
            
            if eval_cache:
                cached_info = eval_cache.get('trajectories', {}).get(traj_id, {})
                cached_ground_truth = cached_info.get('ground_truth')
                cached_is_correct = cached_info.get('is_correct')
                cached_answer = cached_info.get('extracted_answer')
                comparison_method = cached_info.get('comparison_method', '')
            
            # Use cached is_correct directly for the selected trajectory whenever the cache is healthy.
            # The selector has already chosen the final trajectory, so its cached correctness is the
            # authoritative evaluation result for lightweight selection methods such as majority voting.
            use_cached_result = False
            is_comparison_error = str(comparison_method).lower().startswith('error:')

            if cached_is_correct is not None and not is_comparison_error:
                use_cached_result = True
            
            if use_cached_result:
                result = {
                    'id': sample.get('id', 'unknown'),
                    'is_correct': cached_is_correct,
                    'ground_truth': cached_ground_truth or '',
                    'generated_answer': cached_answer or '',
                    'comparison_method': comparison_method + '_cached',
                    'comparison_confidence': 1.0 if cached_is_correct else 0.0,
                    'extraction_method': 'cached',
                    'extraction_confidence': 1.0 if cached_answer else 0.0,
                    'task_type': sample_task_type,
                    'error': None,
                }
            elif pre_extracted_answer is not None and not sample.get('generated_text', ''):
                # Has pre-extracted answer (e.g., voting winner) but doesn't match cache
                # → re-compare this answer against ground truth
                result = self._evaluate_with_pre_extracted_answer(
                    sample, pre_extracted_answer, sample_task_type, cached_ground_truth
                )
            elif not sample.get('generated_text', '') and eval_cache:
                # Lightweight mode, no pre-extracted answer, cache has error or no match
                if cached_answer is not None:
                    result = self._evaluate_with_pre_extracted_answer(
                        sample, cached_answer, sample_task_type, cached_ground_truth
                    )
                else:
                    result = self.evaluate_single_sample(sample, cached_ground_truth=cached_ground_truth)
            else:
                result = self.evaluate_single_sample(sample, cached_ground_truth=cached_ground_truth)
            result['original_id'] = original_id
            result['trajectory_index'] = selected_traj['trajectory_index']
            result['num_trajectories'] = len(trajectories)

            per_sample_results.append(result)

        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_statistics(
            per_sample_results,
            selected_trajectories
        )

        return {
            'per_sample_results': per_sample_results,
            'aggregate_statistics': aggregate_stats,
            'trajectory_selections': selected_trajectories
        }

    def _calculate_aggregate_statistics(
        self,
        results: List[Dict],
        trajectory_selections: Dict
    ) -> Dict:
        """
        Calculate aggregate statistics from evaluation results.

        Args:
            results: List of per-sample evaluation results
            trajectory_selections: Dictionary of trajectory selection info

        Returns:
            Aggregate statistics dictionary
        """
        total_samples = len(results)
        correct_count = sum(1 for r in results if r['is_correct'])
        incorrect_count = sum(1 for r in results if not r['is_correct'] and not r.get('error'))
        failed_extraction_count = sum(1 for r in results if r.get('error'))

        # Count by comparison method
        comparison_methods = defaultdict(int)
        correct_by_method = defaultdict(int)

        for r in results:
            method = r.get('comparison_method', 'unknown')
            comparison_methods[method] += 1
            if r['is_correct']:
                correct_by_method[method] += 1

        # Calculate accuracy by method
        accuracy_by_method = {
            method: (correct_by_method[method] / count if count > 0 else 0.0)
            for method, count in comparison_methods.items()
        }

        # Identify low confidence samples
        low_confidence_samples = [
            r['id'] for r in results
            if r.get('comparison_confidence', 1.0) < 0.8 or r.get('extraction_confidence', 1.0) < 0.8
        ]

        # Count by extraction method
        extraction_methods = defaultdict(int)
        for r in results:
            method = r.get('extraction_method', 'unknown')
            extraction_methods[method] += 1

        # Overall accuracy
        overall_accuracy = correct_count / total_samples if total_samples > 0 else 0.0

        # Trajectory selection statistics
        multi_trajectory_count = sum(1 for sel in trajectory_selections.values() if sel['num_trajectories'] > 1)

        # Task type breakdown (for mixed datasets)
        task_types = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            task_type = r.get('task_type', 'math')
            task_types[task_type]['total'] += 1
            if r['is_correct']:
                task_types[task_type]['correct'] += 1

        task_type_accuracy = {
            task: {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                'correct': stats['correct'],
                'total': stats['total'],
            }
            for task, stats in task_types.items()
        }

        # Sub-task breakdown (for logic tasks: KOR-Bench categories, SynLogic task types)
        subtask_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            subtask = r.get('subtask')
            if subtask:
                task_type = r.get('task_type', '')
                subtask_key = f"{task_type}/{subtask}" if task_type else subtask
                subtask_stats[subtask_key]['total'] += 1
                if r['is_correct']:
                    subtask_stats[subtask_key]['correct'] += 1

        subtask_accuracy = {
            subtask: {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                'correct': stats['correct'],
                'total': stats['total'],
            }
            for subtask, stats in sorted(subtask_stats.items())
        }

        return {
            'total_samples': total_samples,
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
            'failed_extraction_count': failed_extraction_count,
            'overall_accuracy': overall_accuracy,
            'accuracy_by_comparison_method': dict(accuracy_by_method),
            'comparison_method_counts': dict(comparison_methods),
            'extraction_method_counts': dict(extraction_methods),
            'low_confidence_samples': low_confidence_samples,
            'low_confidence_count': len(low_confidence_samples),
            'multi_trajectory_count': multi_trajectory_count,
            'single_trajectory_count': len(trajectory_selections) - multi_trajectory_count,
            'task_type_breakdown': task_type_accuracy,
            'subtask_breakdown': subtask_accuracy,
        }
