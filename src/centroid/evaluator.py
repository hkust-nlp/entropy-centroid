"""
Unified evaluator for math and logic tasks.

Supports:
- Math tasks: Extract boxed answers, symbolic comparison
- KOR-Bench: Extract [[answer]] format, string comparison
- SynLogic: Extract <answer> format, verifier-based comparison
"""

import contextlib
import io
import re
import os
import sys
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

# Add paths for evaluation modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def detect_task_type(trajectory: Dict) -> str:
    """
    Detect task type from trajectory ID or original_id.

    Args:
        trajectory: Trajectory data with 'id' field

    Returns:
        Task type: 'korbench', 'synlogic', 'amo_bench', or 'math'
    """
    traj_id = str(trajectory.get('id', ''))
    original_id = str(trajectory.get('original_id', ''))

    if traj_id.startswith('korbench_'):
        return 'korbench'
    elif traj_id.startswith('synlogic_'):
        return 'synlogic'
    
    # AMO-Bench detection: numeric original_id (like "1", "2", etc.) 
    # and trajectory ID pattern like "1_traj_0"
    # Also check for AMO-Bench specific fields
    if original_id:
        try:
            # AMO-Bench uses simple numeric question_id
            qid = int(original_id)
            # Check if ID pattern matches AMO-Bench (numeric_traj_N)
            if traj_id and '_traj_' in traj_id:
                try:
                    prefix = traj_id.split('_traj_')[0]
                    int(prefix)  # Should be numeric for AMO-Bench
                    # Additional check: look for AMO-Bench specific prompt pattern
                    prompt = trajectory.get('prompt', '') or trajectory.get('problem', '')
                    if prompt and ('### The final answer is:' in prompt or 
                                   'final answer should be given as precisely' in prompt or
                                   '$\\boxed{<your answer>}$' in prompt):
                        return 'amo_bench'
                except (ValueError, IndexError):
                    pass
        except (ValueError, TypeError):
            pass

    return 'math'


def _parse_synlogic_task_from_id(traj_id: str) -> str:
    """
    Parse SynLogic task name from a trajectory ID string.

    ID format: synlogic_{task_name}_{sample_id}[_traj_{n}]
    where sample_id is a UUID (hex+hyphens, no underscores) or a numeric index.
    Task names may contain underscores (e.g., 'boolean_expressions').

    Strategy: strip prefix and _traj_N suffix, then split on the last '_'
    to separate {task_name} from {sample_id}.
    """
    if not traj_id.startswith('synlogic_'):
        return 'unknown'

    # Strip 'synlogic_' prefix
    rest = traj_id[len('synlogic_'):]

    # Strip '_traj_N' suffix if present
    traj_suffix_idx = rest.find('_traj_')
    if traj_suffix_idx >= 0:
        rest = rest[:traj_suffix_idx]

    # rest = '{task_name}_{sample_id}'
    # sample_id (UUID or numeric) never contains underscores,
    # so split on the last '_' to get the task name.
    last_underscore = rest.rfind('_')
    if last_underscore > 0:
        return rest[:last_underscore]

    return rest or 'unknown'


def extract_korbench_category(trajectory: Dict) -> str:
    """
    Extract KOR-Bench category from trajectory ID.

    Args:
        trajectory: Trajectory data

    Returns:
        Category: 'cipher', 'logic', 'operation', 'puzzle', 'counterfactual'
    """
    traj_id = str(trajectory.get('id', ''))
    # Format: korbench_cipher_1_traj_0
    parts = traj_id.split('_')
    if len(parts) >= 2:
        return parts[1]
    return 'unknown'


def extract_synlogic_task(trajectory: Dict) -> str:
    """
    Extract SynLogic task name from trajectory ID.

    Supports task names with underscores (e.g., 'boolean_expressions',
    'arrow_maze', 'dyck_language_errors').

    ID format: synlogic_{task_name}_{uuid}[_traj_{n}]
    Since UUIDs (and numeric IDs) never contain underscores, we split on the
    last '_' after stripping the prefix and trajectory suffix.

    Args:
        trajectory: Trajectory data

    Returns:
        Task name: 'campsite', 'boolean_expressions', etc.
    """
    # Prefer the explicit task_name field set by the data loader
    task_name = trajectory.get('task_name', '')
    if task_name and task_name != 'unknown':
        return task_name

    traj_id = str(trajectory.get('id', ''))
    return _parse_synlogic_task_from_id(traj_id)


class BaseEvaluator(ABC):
    """Base class for trajectory evaluators."""

    @abstractmethod
    def evaluate(self, trajectory: Dict) -> bool:
        """
        Evaluate if trajectory answer is correct.

        Args:
            trajectory: Trajectory data

        Returns:
            True if correct, False otherwise
        """
        pass

    @abstractmethod
    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from generated text.

        Args:
            text: Generated text

        Returns:
            Extracted answer or None
        """
        pass


class MathEvaluator(BaseEvaluator):
    """Evaluator for math tasks using symbolic comparison."""

    def __init__(self, timeout: int = 5):
        """
        Initialize math evaluator.

        Args:
            timeout: Timeout for symbolic comparison
        """
        self.timeout = timeout
        self._comparator = None
        self._extractor = None
        self._grader_check = None

    @property
    def comparator(self):
        """Lazy load AnswerComparator."""
        if self._comparator is None:
            from evaluation.answer_comparator import AnswerComparator
            self._comparator = AnswerComparator(timeout=self.timeout, use_math_grader=True)
        return self._comparator

    @property
    def grader_check(self):
        """Lazy load grader check_is_correct function."""
        if self._grader_check is None:
            try:
                from evaluation.math_grader import check_is_correct
                self._grader_check = check_is_correct
            except ImportError:
                self._grader_check = None
        return self._grader_check

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract boxed answer from math solution."""
        if self._extractor is None:
            from evaluation.answer_extractor import extract_answer
            self._extractor = extract_answer
        return self._extractor(text)

    def evaluate(self, trajectory: Dict) -> bool:
        """
        Evaluate math trajectory using symbolic comparison.
        
        Supports:
        - Multi-answer (comma-separated, unordered)
        - Tuple/list comparison (ordered)
        - Equation forms (f(x)=1, x=5)
        - Symbolic equivalence
        """
        ground_truth_text = trajectory.get('solution', '')
        generated_text = trajectory.get('generated_text', '')

        # Extract answer from generated text
        extracted_answer = self.extract_answer(generated_text)
        if extracted_answer is None:
            return False

        # Extract answer from ground truth (if it contains boxed notation)
        # This ensures symmetric comparison
        gt_extracted = self.extract_answer(ground_truth_text)
        ground_truth = gt_extracted if gt_extracted is not None else ground_truth_text

        # Use grader check_is_correct if available (more robust for multi-answer)
        if self.grader_check is not None:
            try:
                return self.grader_check(extracted_answer, ground_truth, timeout=True)
            except Exception:
                pass

        # Fallback to AnswerComparator
        result = self.comparator.compare(ground_truth, extracted_answer)
        return result['is_correct']


class KorBenchEvaluator(BaseEvaluator):
    """Evaluator for KOR-Bench logic tasks."""

    def __init__(self):
        """Initialize KOR-Bench evaluator."""
        self._eval_utils_loaded = False
        self._load_eval_utils()

    def _load_eval_utils(self):
        """Try to load KOR-Bench evaluation utilities."""
        try:
            korbench_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'KOR-Bench'
            )
            if os.path.exists(korbench_path):
                sys.path.insert(0, korbench_path)

            from eval.eval_utils import (
                evaluate_response_vs_answer,
                extract_text_from_brackets,
            )
            self._evaluate_response_vs_answer = evaluate_response_vs_answer
            self._extract_text_from_brackets = extract_text_from_brackets
            self._eval_utils_loaded = True
        except ImportError:
            self._eval_utils_loaded = False

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from [[...]] brackets."""
        if self._eval_utils_loaded:
            try:
                return self._extract_text_from_brackets(text)
            except Exception:
                pass

        # Fallback extraction
        matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'\[\s*(.*?)\s*\]', text, re.DOTALL)
        if matches:
            return matches[-1].strip()  # Use last match
        return None

    def evaluate(self, trajectory: Dict) -> bool:
        """Evaluate KOR-Bench trajectory."""
        ground_truth = trajectory.get('solution', '')
        generated_text = trajectory.get('generated_text', '')
        category = extract_korbench_category(trajectory)

        if self._eval_utils_loaded:
            try:
                # Use official evaluation
                original_data = trajectory.get('original_data', {})
                rule_id = original_data.get('rule_id', '')
                idx = original_data.get('idx', '')
                return self._evaluate_response_vs_answer(
                    generated_text, ground_truth, category, rule_id, idx
                )
            except Exception:
                pass

        # Fallback: simple string comparison
        extracted = self.extract_answer(generated_text)
        if extracted is None:
            return False

        # Normalize and compare
        gt_clean = re.sub(r'[\[\]<>/ \n\t]', '', ground_truth).lower().strip()
        ext_clean = re.sub(r'[\[\]<>/ \n\t]', '', extracted).lower().strip()
        return gt_clean == ext_clean


class SynLogicEvaluator(BaseEvaluator):
    """Evaluator for SynLogic tasks."""

    def __init__(self, verbose: bool = False):
        """Initialize SynLogic evaluator."""
        self._verifiers = {}
        self._verifier_classes = {}
        self._verifiers_loaded = False
        self._verbose = verbose
        self._load_verifiers()

    def _load_verifiers(self):
        """Try to load SynLogic verifiers."""
        try:
            synlogic_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'SynLogic'
            )
            if os.path.exists(synlogic_path):
                sys.path.insert(0, synlogic_path)

            from task2verifier import verifier_classes
            self._verifier_classes = verifier_classes
            self._verifiers_loaded = True
            if self._verbose:
                print(f"Loaded {len(verifier_classes)} SynLogic verifiers")
        except ImportError as e:
            if self._verbose:
                print(f"Warning: Could not import SynLogic verifiers: {e}")
            self._verifiers_loaded = False

    def _get_verifier(self, task_name: str):
        """Get or create verifier for task."""
        if task_name not in self._verifiers:
            verifier_class = self._verifier_classes.get(task_name)
            if verifier_class:
                try:
                    self._verifiers[task_name] = verifier_class()
                    if self._verbose:
                        print(f"Created verifier for task: {task_name}")
                except Exception as e:
                    if self._verbose:
                        print(f"Warning: Failed to create verifier for {task_name}: {e}")
                    self._verifiers[task_name] = None
            else:
                if self._verbose and task_name != 'unknown':
                    print(f"Warning: No verifier found for task: {task_name}")
                self._verifiers[task_name] = None
        return self._verifiers[task_name]

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from <answer>...</answer> tags."""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _call_verifier_safely(self, verifier, data_obj, generated_text):
        """Run verifier while suppressing noisy stdout/stderr unless verbose."""
        if self._verbose:
            return verifier.verify(data_obj, generated_text)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return verifier.verify(data_obj, generated_text)

    def evaluate(self, trajectory: Dict) -> bool:
        """Evaluate SynLogic trajectory."""
        ground_truth = trajectory.get('solution', '')
        generated_text = trajectory.get('generated_text', '')
        task_name = extract_synlogic_task(trajectory)
        game_data = trajectory.get('game_data', {})

        # Check if game_data is available
        if not game_data:
            if self._verbose:
                traj_id = trajectory.get('id', 'unknown')
                print(f"Warning: No game_data for trajectory {traj_id}, task={task_name}")

        # Try verifier-based evaluation
        verifier = self._get_verifier(task_name)
        if verifier is not None and game_data:
            try:
                # Create data object for verifier
                from base.data import Data
                data_obj = Data.from_json_dict(game_data)
                score = self._call_verifier_safely(verifier, data_obj, generated_text)
                return score > 0 if isinstance(score, (int, float)) else bool(score)
            except Exception as e:
                if self._verbose:
                    traj_id = trajectory.get('id', 'unknown')
                    print(f"Verifier error for {traj_id} (task={task_name}): {e}")

        # Fallback: simple string comparison
        extracted = self.extract_answer(generated_text)
        if extracted is None:
            return False

        return extracted.lower().strip() == str(ground_truth).lower().strip()


class AMOBenchEvaluatorWrapper(BaseEvaluator):
    """Wrapper for AMO-Bench evaluator to match BaseEvaluator interface."""
    
    def __init__(self, skip_description: bool = True):
        """
        Initialize AMO-Bench evaluator wrapper.
        
        Args:
            skip_description: If True, skip description-type answers
        """
        self._evaluator = None
        self._skip_description = skip_description
    
    @property
    def evaluator(self):
        """Lazy load AMO-Bench evaluator."""
        if self._evaluator is None:
            try:
                from evaluation.amo_bench_evaluator import AMOBenchEvaluator
                self._evaluator = AMOBenchEvaluator(skip_description=self._skip_description)
            except ImportError:
                self._evaluator = None
        return self._evaluator
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from generated text."""
        if self.evaluator is None:
            return None
        return self.evaluator.extract_answer(text)
    
    def evaluate(self, trajectory: Dict) -> bool:
        """Evaluate trajectory."""
        if self.evaluator is None:
            # Fallback to math evaluation
            return MathEvaluator().evaluate(trajectory)
        return self.evaluator.evaluate(trajectory)


class UnifiedEvaluator:
    """
    Unified evaluator that automatically selects the appropriate evaluator
    based on task type.
    """

    def __init__(self, timeout: int = 5, skip_amo_description: bool = True):
        """
        Initialize unified evaluator.

        Args:
            timeout: Timeout for math symbolic comparison
            skip_amo_description: If True, skip AMO-Bench description-type answers
        """
        self.math_evaluator = MathEvaluator(timeout=timeout)
        self.korbench_evaluator = KorBenchEvaluator()
        self.synlogic_evaluator = SynLogicEvaluator()
        self.amo_bench_evaluator = AMOBenchEvaluatorWrapper(skip_description=skip_amo_description)

    def evaluate(self, trajectory: Dict) -> bool:
        """
        Evaluate trajectory using appropriate evaluator.

        Args:
            trajectory: Trajectory data

        Returns:
            True if correct, False otherwise
        """
        task_type = detect_task_type(trajectory)

        if task_type == 'korbench':
            return self.korbench_evaluator.evaluate(trajectory)
        elif task_type == 'synlogic':
            return self.synlogic_evaluator.evaluate(trajectory)
        elif task_type == 'amo_bench':
            return self.amo_bench_evaluator.evaluate(trajectory)
        else:
            return self.math_evaluator.evaluate(trajectory)

    def get_evaluator(self, task_type: str) -> BaseEvaluator:
        """Get specific evaluator by task type."""
        if task_type == 'korbench':
            return self.korbench_evaluator
        elif task_type == 'synlogic':
            return self.synlogic_evaluator
        elif task_type == 'amo_bench':
            return self.amo_bench_evaluator
        else:
            return self.math_evaluator


def evaluate_trajectory(trajectory: Dict, comparator=None, evaluator: UnifiedEvaluator = None) -> bool:
    """
    Evaluate if trajectory answer is correct.

    This is the main evaluation function that handles both math and logic tasks.
    Uses math grader for math tasks with support for:
    - Multi-answer (comma-separated, unordered)
    - Tuple/list comparison (ordered)
    - Equation forms (f(x)=1, x=5)
    - Symbolic equivalence

    Args:
        trajectory: Trajectory data
        comparator: Legacy AnswerComparator (for backward compatibility with math tasks)
        evaluator: UnifiedEvaluator instance (preferred)

    Returns:
        True if correct, False otherwise
    """
    # If unified evaluator is provided, use it
    if evaluator is not None:
        return evaluator.evaluate(trajectory)

    # Auto-detect task type
    task_type = detect_task_type(trajectory)

    if task_type in ['korbench', 'synlogic']:
        # Use logic evaluators
        if task_type == 'korbench':
            eval_instance = KorBenchEvaluator()
        else:
            eval_instance = SynLogicEvaluator()
        return eval_instance.evaluate(trajectory)
    else:
        # Math task - use MathEvaluator
        return MathEvaluator().evaluate(trajectory)


def create_evaluator(task_type: str = 'auto', timeout: int = 5) -> UnifiedEvaluator:
    """
    Create an evaluator instance.

    Args:
        task_type: Task type hint ('auto', 'math', 'korbench', 'synlogic')
        timeout: Timeout for math symbolic comparison

    Returns:
        UnifiedEvaluator instance
    """
    return UnifiedEvaluator(timeout=timeout)
