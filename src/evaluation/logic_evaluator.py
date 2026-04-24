"""
Logic task evaluators for KOR-Bench and SynLogic.

Integrates evaluation logic from:
- KOR-Bench: eval/eval_utils.py
- SynLogic: task2verifier.py and individual verifiers
"""

import contextlib
import io
import json
import os
import sys
import re
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Add paths for importing from KOR-Bench and SynLogic
# Support environment variables for custom paths
# Use abspath to handle relative __file__ paths correctly
_THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE))))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE)))

# Try multiple possible locations for KOR-Bench
KORBENCH_SEARCH_PATHS = [
    os.environ.get('KORBENCH_PATH', ''),  # Environment variable (highest priority)
    os.path.join(WORKSPACE_ROOT, 'KOR-Bench'),  # workspace/KOR-Bench
    os.path.join(PROJECT_ROOT, 'KOR-Bench'),  # prmagent/KOR-Bench
    os.path.join(os.path.expanduser('~'), 'KOR-Bench'),  # ~/KOR-Bench
]

SYNLOGIC_SEARCH_PATHS = [
    os.environ.get('SYNLOGIC_PATH', ''),  # Environment variable (highest priority)
    os.path.join(WORKSPACE_ROOT, 'SynLogic'),  # workspace/SynLogic
    os.path.join(PROJECT_ROOT, 'SynLogic'),  # prmagent/SynLogic
    os.path.join(os.path.expanduser('~'), 'SynLogic'),  # ~/SynLogic
]

KORBENCH_PATH = None
for path in KORBENCH_SEARCH_PATHS:
    if path and os.path.exists(path):
        KORBENCH_PATH = path
        break

SYNLOGIC_PATH = None
for path in SYNLOGIC_SEARCH_PATHS:
    if path and os.path.exists(path):
        SYNLOGIC_PATH = path
        break

if KORBENCH_PATH:
    sys.path.insert(0, KORBENCH_PATH)
if SYNLOGIC_PATH:
    sys.path.insert(0, SYNLOGIC_PATH)


class KorBenchEvaluator:
    """
    Evaluator for KOR-Bench tasks.
    
    Integrates evaluation logic from KOR-Bench/eval/eval_utils.py
    """
    
    def __init__(self):
        """Initialize KOR-Bench evaluator."""
        self._load_eval_utils()
    
    def _load_eval_utils(self):
        """Try to load KOR-Bench eval utilities."""
        try:
            from eval.eval_utils import (
                evaluate_response_vs_answer,
                extract_text_from_brackets,
            )
            self.evaluate_response_vs_answer = evaluate_response_vs_answer
            self.extract_text_from_brackets = extract_text_from_brackets
            self._utils_loaded = True
        except ImportError as e:
            self._utils_loaded = False
            # Only print warning once (use class variable to track)
            if not hasattr(KorBenchEvaluator, '_import_warning_shown'):
                KorBenchEvaluator._import_warning_shown = True
                print(f"Warning: Could not import KOR-Bench eval_utils: {e}")
                if KORBENCH_PATH:
                    print(f"  KOR-Bench path found: {KORBENCH_PATH}")
                else:
                    print(f"  KOR-Bench not found. Searched paths:")
                    for p in KORBENCH_SEARCH_PATHS:
                        if p:
                            print(f"    - {p} {'(exists)' if os.path.exists(p) else '(not found)'}")
                print("  Using fallback evaluation. Set KORBENCH_PATH environment variable to specify location.")
    
    def _fallback_extract_brackets(self, text: str) -> str:
        """Fallback bracket extraction."""
        matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'\[\s*(.*?)\s*\]', text, re.DOTALL)
        if matches:
            return f'[[{matches[0].strip()}]]'
        return "NULL"
    
    def _fallback_evaluate(self, response: str, answer: str, question_type: str) -> bool:
        """Fallback evaluation when eval_utils not available."""
        response_text = self._fallback_extract_brackets(response)
        answer_text = self._fallback_extract_brackets(answer)
        
        # Clean and compare
        response_clean = re.sub(r'[^A-Za-z0-9]', '', response_text).lower()
        answer_clean = re.sub(r'[^A-Za-z0-9]', '', answer_text).lower()
        
        return response_clean == answer_clean
    
    def evaluate_single(
        self,
        generated_text: str,
        ground_truth: str,
        question_type: str = 'logic',
        rule_id: str = '',
        idx: str = '',
    ) -> Dict:
        """
        Evaluate a single KOR-Bench sample.
        
        Args:
            generated_text: Model generated response
            ground_truth: Ground truth answer
            question_type: Category (cipher, logic, operation, puzzle, counterfactual)
            rule_id: Rule ID for the question
            idx: Index of the question (must be a valid integer string for 'operation' type)
            
        Returns:
            Evaluation result dictionary
        """
        # Validate idx - KOR-Bench's is_in_idx_ranges() requires a valid integer
        # If idx is empty or invalid, use a safe default that won't match any idx_ranges
        safe_idx = idx
        if not idx or not idx.strip():
            safe_idx = '-1'  # Use -1 as a safe default that won't match any idx_ranges
        else:
            try:
                int(idx)  # Validate it's a valid integer
            except ValueError:
                safe_idx = '-1'
        
        if self._utils_loaded:
            try:
                is_correct = self.evaluate_response_vs_answer(
                    generated_text, ground_truth, question_type, rule_id, safe_idx
                )
                extracted = self.extract_text_from_brackets(generated_text)
            except Exception as e:
                print(f"Evaluation error: {e}")
                is_correct = self._fallback_evaluate(generated_text, ground_truth, question_type)
                extracted = self._fallback_extract_brackets(generated_text)
        else:
            is_correct = self._fallback_evaluate(generated_text, ground_truth, question_type)
            extracted = self._fallback_extract_brackets(generated_text)
        
        return {
            'is_correct': is_correct,
            'extracted_answer': extracted,
            'ground_truth': ground_truth,
        }
    
    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """
        Evaluate a batch of KOR-Bench results.
        
        Args:
            results: List of result dictionaries with generated_text and original_data
            
        Returns:
            Evaluation summary with per-sample and aggregate statistics
        """
        per_sample_results = []
        correct_count = 0
        
        for result in results:
            original_data = result.get('original_data', {})
            sample_id = result.get('id', 'unknown')
            generated_text = result.get('generated_text', '')
            ground_truth = original_data.get('answer', result.get('solution', ''))
            category = result.get('category', 'logic')
            rule_id = original_data.get('rule_id', '')
            idx = original_data.get('idx', '')
            
            eval_result = self.evaluate_single(
                generated_text, ground_truth, category, rule_id, idx
            )
            
            eval_result['id'] = sample_id
            eval_result['category'] = category
            eval_result['rule_id'] = rule_id
            per_sample_results.append(eval_result)
            
            if eval_result['is_correct']:
                correct_count += 1
        
        total = len(per_sample_results)
        accuracy = correct_count / total if total > 0 else 0.0
        
        return {
            'per_sample_results': per_sample_results,
            'aggregate_statistics': {
                'total_samples': total,
                'correct_count': correct_count,
                'accuracy': accuracy,
            }
        }


class SynLogicEvaluator:
    """
    Evaluator for SynLogic tasks.
    
    Uses verifiers from SynLogic/task2verifier.py
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize SynLogic evaluator."""
        self.verifiers = {}
        self.verifier_classes = {}
        self._verifiers_loaded = False
        self._verbose = verbose
        self._load_verifiers()
    
    def _load_verifiers(self):
        """Load verifier classes from SynLogic."""
        try:
            from task2verifier import verifier_classes
            self.verifier_classes = verifier_classes
            self._verifiers_loaded = True
            if self._verbose:
                print(f"Loaded {len(verifier_classes)} SynLogic verifiers")
        except ImportError as e:
            if self._verbose:
                print(f"Warning: Could not import SynLogic verifiers: {e}")
            self.verifier_classes = {}
            self._verifiers_loaded = False
    
    def _get_verifier(self, task_name: str):
        """Get or create verifier instance for task."""
        if task_name not in self.verifiers:
            verifier_class = self.verifier_classes.get(task_name)
            if verifier_class:
                try:
                    self.verifiers[task_name] = verifier_class()
                except Exception as e:
                    if self._verbose:
                        print(f"Warning: Failed to create verifier for {task_name}: {e}")
                    self.verifiers[task_name] = None
            else:
                if self._verbose and task_name != 'unknown':
                    print(f"Warning: No verifier class found for task: {task_name}")
                self.verifiers[task_name] = None
        return self.verifiers[task_name]
    
    def _create_data_object(self, game_data: Dict):
        """Create Data object for verifier.
        
        Ensures required fields (question, answer) are present with defaults.
        """
        # Ensure required fields have defaults
        safe_game_data = {
            'question': game_data.get('question', ''),
            'answer': game_data.get('answer', ''),
            'difficulty': game_data.get('difficulty', 1),
            'metadata': game_data.get('metadata', {}),
        }
        # Copy any additional fields from original game_data
        for key, value in game_data.items():
            if key not in safe_game_data:
                safe_game_data[key] = value
        
        try:
            from base.data import Data
            return Data.from_json_dict(safe_game_data)
        except ImportError:
            # Fallback: create simple object with required attributes
            class SimpleData:
                def __init__(self, data):
                    self.question = data.get('question', '')
                    self.answer = data.get('answer', '')
                    self.difficulty = data.get('difficulty', 1)
                    self.metadata = data.get('metadata', {})
            return SimpleData(safe_game_data)
    
    def _fallback_evaluate(self, generated_text: str, ground_truth: str) -> bool:
        """
        Fallback evaluation using robust string matching.
        
        Borrowed from fix_logic_evaluation.py for better comparison.
        """
        # Try to extract answer from <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            extracted = answer_match.group(1).strip()
        else:
            # Try **answer** or Answer: format
            answer_match = re.search(r'(?:\*\*answer\*\*|answer\s*:)\s*(.+?)(?:\n|$)', generated_text, re.IGNORECASE)
            if answer_match:
                extracted = answer_match.group(1).strip()
            else:
                extracted = generated_text.strip()
        
        # Normalize answers for comparison
        def normalize_synlogic_answer(answer: str) -> str:
            if answer is None:
                return None
            answer = str(answer).strip()
            # Try to parse as JSON (handle list format)
            try:
                parsed = json.loads(answer.replace("'", '"'))
                def normalize_item(item):
                    if isinstance(item, list):
                        return [normalize_item(i) for i in item]
                    elif isinstance(item, str):
                        return item.strip().lower()
                    else:
                        return item
                return json.dumps(normalize_item(parsed), sort_keys=True)
            except:
                pass
            # Simple normalization
            return re.sub(r'\s+', '', answer).lower()
        
        gen_norm = normalize_synlogic_answer(extracted)
        gt_norm = normalize_synlogic_answer(ground_truth)
        
        if gen_norm is None or gt_norm is None:
            return False
        
        return gen_norm == gt_norm

    def _call_verifier_safely(self, verifier, data_obj, generated_text):
        """Run verifier while suppressing noisy stdout/stderr unless verbose."""
        if self._verbose:
            return verifier.verify(data_obj, generated_text)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return verifier.verify(data_obj, generated_text)
    
    def evaluate_single(
        self,
        generated_text: str,
        task_name: str,
        game_data: Dict,
    ) -> Dict:
        """
        Evaluate a single SynLogic sample.
        
        Args:
            generated_text: Model generated response
            task_name: Task name (e.g., 'campsite', 'sudoku')
            game_data: Original game data with question, answer, metadata
            
        Returns:
            Evaluation result dictionary
        """
        verifier = self._get_verifier(task_name)
        
        if verifier is not None:
            try:
                data_obj = self._create_data_object(game_data)
                score = self._call_verifier_safely(verifier, data_obj, generated_text)
                is_correct = score > 0 if isinstance(score, (int, float)) else bool(score)
                
                # Try to extract answer
                try:
                    extracted = verifier.extract_answer(generated_text)
                except:
                    extracted = None
                
                return {
                    'is_correct': is_correct,
                    'score': score,
                    'extracted_answer': extracted,
                    'ground_truth': game_data.get('answer', ''),
                    'verifier_used': task_name,
                }
            except Exception as e:
                print(f"Verifier error for {task_name}: {e}")
        
        # Fallback evaluation
        ground_truth = game_data.get('answer', '')
        is_correct = self._fallback_evaluate(generated_text, ground_truth)
        
        return {
            'is_correct': is_correct,
            'score': 1.0 if is_correct else 0.0,
            'extracted_answer': None,
            'ground_truth': ground_truth,
            'verifier_used': 'fallback',
        }
    
    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """
        Evaluate a batch of SynLogic results.
        
        Args:
            results: List of result dictionaries with generated_text and game_data
            
        Returns:
            Evaluation summary with per-sample and aggregate statistics
        """
        per_sample_results = []
        correct_count = 0
        task_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for result in results:
            sample_id = result.get('id', 'unknown')
            generated_text = result.get('generated_text', '')
            task_name = result.get('task_name', 'unknown')
            game_data = result.get('game_data', {})
            
            eval_result = self.evaluate_single(generated_text, task_name, game_data)
            eval_result['id'] = sample_id
            eval_result['task_name'] = task_name
            per_sample_results.append(eval_result)
            
            task_stats[task_name]['total'] += 1
            if eval_result['is_correct']:
                correct_count += 1
                task_stats[task_name]['correct'] += 1
        
        total = len(per_sample_results)
        accuracy = correct_count / total if total > 0 else 0.0
        
        # Calculate per-task accuracy
        task_accuracy = {}
        for task, stats in task_stats.items():
            task_accuracy[task] = {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                'correct': stats['correct'],
                'total': stats['total'],
            }
        
        return {
            'per_sample_results': per_sample_results,
            'aggregate_statistics': {
                'total_samples': total,
                'correct_count': correct_count,
                'accuracy': accuracy,
                'task_breakdown': task_accuracy,
            }
        }


class LogicEvaluator:
    """
    Unified evaluator for logic tasks.
    
    Automatically routes to appropriate evaluator based on source.
    """
    
    def __init__(self):
        """Initialize unified logic evaluator."""
        self.korbench_evaluator = KorBenchEvaluator()
        self.synlogic_evaluator = SynLogicEvaluator()
    
    def evaluate_single(self, result: Dict) -> Dict:
        """
        Evaluate a single result, routing to appropriate evaluator.
        
        Args:
            result: Result dictionary with source information
            
        Returns:
            Evaluation result dictionary
        """
        source = result.get('source', 'unknown')
        
        if source == 'korbench':
            original_data = result.get('original_data', {})
            return self.korbench_evaluator.evaluate_single(
                generated_text=result.get('generated_text', ''),
                ground_truth=original_data.get('answer', result.get('solution', '')),
                question_type=result.get('category', 'logic'),
                rule_id=original_data.get('rule_id', ''),
                idx=original_data.get('idx', ''),
            )
        elif source == 'synlogic':
            return self.synlogic_evaluator.evaluate_single(
                generated_text=result.get('generated_text', ''),
                task_name=result.get('task_name', 'unknown'),
                game_data=result.get('game_data', {}),
            )
        else:
            # Unknown source - use fallback
            return {
                'is_correct': False,
                'error': f'Unknown source: {source}',
            }
    
    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """
        Evaluate a batch of results.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Evaluation summary
        """
        # Group by source
        korbench_results = [r for r in results if r.get('source') == 'korbench']
        synlogic_results = [r for r in results if r.get('source') == 'synlogic']
        other_results = [r for r in results if r.get('source') not in ['korbench', 'synlogic']]
        
        combined_per_sample = []
        total_correct = 0
        total_samples = 0
        
        # Evaluate KOR-Bench
        if korbench_results:
            kb_eval = self.korbench_evaluator.evaluate_batch(korbench_results)
            combined_per_sample.extend(kb_eval['per_sample_results'])
            total_correct += kb_eval['aggregate_statistics']['correct_count']
            total_samples += kb_eval['aggregate_statistics']['total_samples']
        
        # Evaluate SynLogic
        if synlogic_results:
            sl_eval = self.synlogic_evaluator.evaluate_batch(synlogic_results)
            combined_per_sample.extend(sl_eval['per_sample_results'])
            total_correct += sl_eval['aggregate_statistics']['correct_count']
            total_samples += sl_eval['aggregate_statistics']['total_samples']
        
        # Handle other results
        for result in other_results:
            eval_result = self.evaluate_single(result)
            eval_result['id'] = result.get('id', 'unknown')
            combined_per_sample.append(eval_result)
            total_samples += 1
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'per_sample_results': combined_per_sample,
            'aggregate_statistics': {
                'total_samples': total_samples,
                'correct_count': total_correct,
                'accuracy': accuracy,
                'korbench_samples': len(korbench_results),
                'synlogic_samples': len(synlogic_results),
            }
        }


def create_logic_evaluator(config: Dict = None) -> LogicEvaluator:
    """
    Create a logic evaluator.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LogicEvaluator instance
    """
    return LogicEvaluator()
