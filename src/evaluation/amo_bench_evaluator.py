"""
AMO-Bench evaluator wrapper.

This module provides a clean interface to the AMO-Bench grading code.
It directly uses the official grading functions from amo_grading.py and amo_utils.py.

Supports three answer types:
- description: Requires LLM-based scoring (needs API key via env or set_api_config)
- number/set: Uses math_verify for parsing and comparison
- variable: Uses symbolic computation with SymPy
"""

import os
from typing import Dict, Optional, Tuple

# Import the official AMO-Bench grading functions
from .amo_grading import grading, verify_result
from .amo_utils import (
    pred_extractor,
    pred_cut,
    append_try_list,
    set_api_config,
    API_KEY,
)


class AMOBenchEvaluator:
    """
    Evaluator for AMO-Bench mathematical olympiad problems.
    
    This is a wrapper around the official AMO-Bench grading code.
    
    Supports:
    - number: Numeric answers (uses math_verify)
    - set: Set answers (uses math_verify)
    - variable: Variable/expression answers (uses symbolic computation)
    - description: Descriptive answers (requires LLM API)
    
    For description type answers, set one of:
    - Environment variable: OPENAI_API_KEY, OPENROUTER_API_KEY, or AMO_BENCH_API_KEY
    - Call set_api_config(api_key="your-key") before evaluation
    - Pass api_key to constructor
    
    OpenRouter Support:
    - Set USE_OPENROUTER=true environment variable, or
    - Use base_url="https://openrouter.ai/api/v1", or
    - Pass use_openrouter=True to constructor
    - Models are automatically mapped to use official provider (e.g., openai/gpt-4o-mini)
    """
    
    def __init__(
        self,
        skip_description: bool = False,
        dataset_cache: Optional[Dict] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_openrouter: bool = None,
    ):
        """
        Initialize AMO-Bench evaluator.
        
        Args:
            skip_description: If True, skip description-type answers (return False)
            dataset_cache: Optional pre-loaded dataset info {question_id: info}
            api_key: OpenAI/OpenRouter API key for description type evaluation
            base_url: Custom base URL for API (e.g., "https://openrouter.ai/api/v1")
            use_openrouter: If True, use OpenRouter with official provider routing
        """
        self.skip_description = skip_description
        self._dataset_cache = dataset_cache
        self._dataset_loaded = False
        
        # Set API configuration if provided
        if api_key or base_url or use_openrouter is not None:
            set_api_config(api_key=api_key, base_url=base_url, use_openrouter=use_openrouter)
        
        # Check if API is configured for description type
        from .amo_utils import API_KEY as current_api_key
        self._api_configured = (current_api_key and current_api_key != "API_KEY_HERE")
        
        if not self.skip_description and not self._api_configured:
            print("Warning: AMO-Bench evaluator - no API key configured. "
                  "Description type answers will return False.")
    
    def _load_dataset(self):
        """Load AMO-Bench dataset for answer_type lookup."""
        if self._dataset_loaded:
            return
        
        if self._dataset_cache is not None:
            self._dataset_loaded = True
            return
        
        try:
            import datasets
            dataset = datasets.load_dataset("meituan-longcat/AMO-Bench")
            self._dataset_cache = {}
            for item in dataset["test"]:
                qid = item["question_id"]
                self._dataset_cache[qid] = append_try_list(item)
            self._dataset_loaded = True
        except Exception as e:
            print(f"Warning: Could not load AMO-Bench dataset: {e}")
            self._dataset_cache = {}
            self._dataset_loaded = True
    
    def get_question_info(self, question_id: int) -> Optional[Dict]:
        """
        Get question info from dataset cache.
        
        Args:
            question_id: Question ID
            
        Returns:
            Question info dict or None
        """
        self._load_dataset()
        return self._dataset_cache.get(question_id)
    
    def extract_answer(self, text: str, answer_type: str = "number") -> Optional[str]:
        """
        Extract answer from generated text using AMO-Bench extractor.
        
        Args:
            text: Generated text
            answer_type: Type of answer for extraction
            
        Returns:
            Extracted answer or None
        """
        if not isinstance(text, str):
            return None
        
        return pred_extractor(text, answer_type)
    
    def evaluate(self, trajectory: Dict) -> bool:
        """
        Evaluate if trajectory answer is correct.
        
        Uses the official AMO-Bench grading function.
        
        Args:
            trajectory: Trajectory data with 'generated_text', 'solution', 
                       and optionally 'original_id' for question_id lookup
                       
        Returns:
            True if correct, False otherwise
        """
        generated_text = trajectory.get('generated_text', '')
        
        # Get question ID to look up answer_type and info
        original_id = trajectory.get('original_id', '')
        try:
            question_id = int(original_id)
        except (ValueError, TypeError):
            # Try to extract from id field
            traj_id = trajectory.get('id', '')
            try:
                question_id = int(traj_id.split('_')[0])
            except (ValueError, IndexError):
                question_id = None
        
        # Get info from dataset
        info = None
        if question_id is not None:
            info = self.get_question_info(question_id)
        
        # If no info from dataset, build minimal info
        if info is None:
            ground_truth = trajectory.get('solution', '')
            info = {
                "answer": ground_truth,
                "answer_type": "number",  # Default to number type
                "question_id": question_id or 0,
                "prompt": trajectory.get('problem', ''),
            }
        
        # Skip description type if requested or no API configured
        answer_type = info.get("answer_type", "number")
        if answer_type == "description" and (self.skip_description or not self._api_configured):
            return False
        
        # Use official grading function
        acc = grading(generated_text, info)
        return acc == 1.0
    
    def evaluate_with_info(self, trajectory: Dict) -> Dict:
        """
        Evaluate trajectory and return detailed info.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Dict with is_correct, answer_type, extracted_answer, etc.
        """
        generated_text = trajectory.get('generated_text', '')
        ground_truth = trajectory.get('solution', '')
        
        # Get question ID
        original_id = trajectory.get('original_id', '')
        try:
            question_id = int(original_id)
        except (ValueError, TypeError):
            traj_id = trajectory.get('id', '')
            try:
                question_id = int(traj_id.split('_')[0])
            except (ValueError, IndexError):
                question_id = None
        
        # Get info from dataset
        info = None
        answer_type = "number"
        
        if question_id is not None:
            info = self.get_question_info(question_id)
            if info:
                answer_type = info.get("answer_type", "number")
        
        # Build info if not from dataset
        if info is None:
            info = {
                "answer": ground_truth,
                "answer_type": answer_type,
                "question_id": question_id or 0,
                "prompt": trajectory.get('problem', ''),
            }
        
        # Extract answer
        extracted_answer = self.extract_answer(generated_text, answer_type)
        
        # Check if should skip
        should_skip = answer_type == "description" and (self.skip_description or not self._api_configured)
        
        if should_skip:
            is_correct = False
        else:
            acc = grading(generated_text, info)
            is_correct = acc == 1.0
        
        return {
            "is_correct": is_correct,
            "answer_type": answer_type,
            "extracted_answer": extracted_answer,
            "ground_truth": ground_truth,
            "question_id": question_id,
            "skipped": should_skip,
            "api_used": answer_type == "description" and not should_skip
        }


def grade_amo_bench_answer(
    pred: str, 
    info: Dict,
    skip_description: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[bool, float]:
    """
    Grade an AMO-Bench answer using official grading function.
    
    Args:
        pred: Model prediction (full generated text)
        info: Question info dict with 'answer', 'answer_type', 'question_id', 'prompt'
        skip_description: If True, skip description-type answers (return False)
        api_key: OpenAI API key for description type evaluation
        base_url: Custom base URL for API
        
    Returns:
        Tuple of (is_correct, accuracy_score)
    """
    # Set API config if provided
    if api_key or base_url:
        set_api_config(api_key=api_key, base_url=base_url)
    
    answer_type = info.get("answer_type", "number")
    
    # Skip description if requested
    if skip_description and answer_type == "description":
        return False, 0.0
    
    # Check API for description type
    from .amo_utils import API_KEY as current_api_key
    if answer_type == "description" and (not current_api_key or current_api_key == "API_KEY_HERE"):
        print("Warning: Skipping description type - no API key configured")
        return False, 0.0
    
    # Use official grading function
    acc = grading(pred, info)
    is_correct = acc == 1.0
    
    return is_correct, acc
