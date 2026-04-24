"""
Dataset loader for BigCodeBench benchmark.

BigCodeBench is a benchmark for code generation with execution-based evaluation.
Dataset: bigcode/bigcodebench

Supports two modes:
- 'instruct': Instruction-following code generation (recommended)
- 'complete': Code completion tasks

Note on dataset structure (as of 2026):
- The dataset uses version names as splits: 'v0.1.0_hf', 'v0.1.1', 'v0.1.2', 'v0.1.3', 'v0.1.4'
- Fields: task_id, complete_prompt, instruct_prompt, canonical_solution, 
          code_prompt, test, entry_point, doc_struct, libs
- The 'libs' field is a string representation of a list, e.g., "['random', 'itertools']"
"""

import ast
import json
from typing import Dict, List, Optional, Union
from datasets import load_dataset
from tqdm import tqdm


class BigCodeBenchLoader:
    """
    Loader for BigCodeBench dataset from HuggingFace.
    
    Dataset: bigcode/bigcodebench
    
    The dataset contains code generation tasks with:
    - task_id: Unique identifier for each task (e.g., 'BigCodeBench/0')
    - instruct_prompt: Natural language instruction for the task
    - complete_prompt: Code completion prompt (partial code with docstring)
    - code_prompt: Minimal code prompt (function signature only)
    - canonical_solution: Reference solution
    - test: Test cases for evaluation (unittest code)
    - entry_point: Function name to test
    - doc_struct: JSON string with structured documentation
    - libs: String representation of required libraries list
    
    Available splits/versions: 'v0.1.0_hf', 'v0.1.1', 'v0.1.2', 'v0.1.3', 'v0.1.4'
    """
    
    # Valid task modes
    VALID_MODES = ['instruct', 'complete']
    
    # Valid dataset versions (used as splits)
    VALID_VERSIONS = ['v0.1.0_hf', 'v0.1.1', 'v0.1.2', 'v0.1.3', 'v0.1.4']
    
    def __init__(
        self,
        mode: str = 'instruct',
        subset: str = 'v0.1.2',
        split: str = 'default',  # Kept for backward compatibility, but ignored
        max_samples: Optional[int] = None,
        prompt_type: str = 'default',
    ):
        """
        Initialize BigCodeBench dataset loader.
        
        Args:
            mode: Task mode - 'instruct' for instruction-following, 'complete' for code completion
            subset: Dataset version to use as split (e.g., 'v0.1.2', 'v0.1.0_hf')
                   Note: This is used as the split name when loading from HuggingFace
            split: Deprecated parameter, kept for backward compatibility. Use 'subset' instead.
            max_samples: Maximum number of samples to load (None for all)
            prompt_type: Prompt template type (default, detailed, concise)
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {self.VALID_MODES}")
        
        if subset not in self.VALID_VERSIONS:
            print(f"Warning: subset '{subset}' not in known versions {self.VALID_VERSIONS}. Proceeding anyway.")
        
        self.mode = mode
        self.subset = subset  # This is the actual split name for HuggingFace
        self.split = split    # Deprecated, kept for backward compatibility
        self.max_samples = max_samples
        self.prompt_type = prompt_type
        self.dataset = None
    
    def _format_instruct_prompt(self, sample: Dict) -> str:
        """
        Format instruction prompt for code generation.
        
        Args:
            sample: Raw sample from dataset
            
        Returns:
            Formatted prompt string
        """
        instruct_prompt = sample.get('instruct_prompt', '')
        
        if self.prompt_type == 'detailed':
            return f"Please write Python code to solve the following task:\n\n{instruct_prompt}\n\nProvide a complete, working solution."
        elif self.prompt_type == 'concise':
            return instruct_prompt
        else:  # default
            return f"Task:\n{instruct_prompt}"
    
    def _format_complete_prompt(self, sample: Dict) -> str:
        """
        Format completion prompt for code generation.
        
        Args:
            sample: Raw sample from dataset
            
        Returns:
            Formatted prompt string (partial code to complete)
        """
        complete_prompt = sample.get('complete_prompt', '')
        
        if self.prompt_type == 'detailed':
            return f"Complete the following Python code:\n\n{complete_prompt}"
        elif self.prompt_type == 'concise':
            return complete_prompt
        else:  # default
            return f"Complete the code:\n{complete_prompt}"
    
    def _parse_libs(self, libs_raw: Union[str, List]) -> List[str]:
        """
        Parse the libs field which may be a string representation of a list.
        
        In the HuggingFace dataset, libs is stored as a string like "['random', 'itertools']"
        
        Args:
            libs_raw: Raw libs value from dataset (string or list)
            
        Returns:
            List of library names
        """
        if isinstance(libs_raw, list):
            return libs_raw
        
        if not libs_raw or libs_raw == '[]':
            return []
        
        try:
            # Safely evaluate the string representation of a list
            parsed = ast.literal_eval(libs_raw)
            if isinstance(parsed, list):
                return parsed
            return [str(parsed)]
        except (ValueError, SyntaxError):
            # If parsing fails, return empty list
            return []
    
    def _format_sample(self, sample: Dict, index: int) -> Dict:
        """
        Format a single sample with prompt.
        
        Args:
            sample: Raw sample from dataset
            index: Sample index
            
        Returns:
            Formatted sample dictionary
        """
        # Get task ID
        task_id = sample.get('task_id', f'bigcodebench_{index:05d}')
        
        # Format prompt based on mode
        if self.mode == 'instruct':
            problem = sample.get('instruct_prompt', '')
            prompt = self._format_instruct_prompt(sample)
        else:  # complete mode
            problem = sample.get('complete_prompt', '')
            prompt = self._format_complete_prompt(sample)
        
        # Get canonical solution
        solution = sample.get('canonical_solution', '')
        
        # Get test code for potential evaluation
        test_code = sample.get('test', '')
        
        # Get entry point (function name)
        entry_point = sample.get('entry_point', '')
        
        # Get additional metadata
        # Note: 'libs' is stored as a string representation of a list in the dataset
        libs_raw = sample.get('libs', '[]')
        libs = self._parse_libs(libs_raw)
        
        # Get code_prompt (minimal function signature)
        code_prompt = sample.get('code_prompt', '')
        
        # Get doc_struct (JSON string with structured documentation)
        doc_struct = sample.get('doc_struct', '{}')
        
        return {
            'id': task_id,
            'problem': problem,
            'solution': solution,
            'prompt': prompt,
            # BigCodeBench specific fields
            'task_id': task_id,
            'entry_point': entry_point,
            'test_code': test_code,
            'libs': libs,
            'code_prompt': code_prompt,
            'doc_struct': doc_struct,
            'mode': self.mode,
            'source': 'bigcodebench',
            # Store original data for reference
            'original_data': {
                'instruct_prompt': sample.get('instruct_prompt', ''),
                'complete_prompt': sample.get('complete_prompt', ''),
                'code_prompt': code_prompt,
                'canonical_solution': solution,
            },
        }
    
    def load(self) -> List[Dict]:
        """
        Load the BigCodeBench dataset and format with prompts.
        
        Returns:
            List of formatted samples with prompts
        """
        print(f"Loading BigCodeBench dataset (mode: {self.mode}, version: {self.subset})")
        
        try:
            # Load dataset from HuggingFace
            # BigCodeBench uses version names as splits (e.g., 'v0.1.2')
            # Note: trust_remote_code is no longer supported in newer datasets versions
            self.dataset = load_dataset(
                "bigcode/bigcodebench",
                split=self.subset,  # Use subset (version) as the split name
            )
        except Exception as e:
            print(f"Error loading with version '{self.subset}': {e}")
            print("Trying to load with fallback version 'v0.1.2'...")
            try:
                self.dataset = load_dataset(
                    "bigcode/bigcodebench",
                    split='v0.1.2',  # Fallback to a known stable version
                )
            except Exception as e2:
                raise ValueError(f"Failed to load BigCodeBench dataset: {e2}")
        
        print(f"Loaded {len(self.dataset)} samples")
        print(f"Dataset columns: {self.dataset.column_names}")
        
        # Limit samples if specified
        if self.max_samples is not None:
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))
            print(f"Limited to {len(self.dataset)} samples")
        
        # Format samples with prompts
        formatted_samples = []
        for idx, sample in enumerate(tqdm(self.dataset, desc="Formatting BigCodeBench samples")):
            formatted_sample = self._format_sample(sample, idx)
            formatted_samples.append(formatted_sample)
        
        print(f"Formatted {len(formatted_samples)} samples for {self.mode} mode")
        
        return formatted_samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset) if self.dataset is not None else 0


def create_bigcodebench_loader(config: Dict) -> BigCodeBenchLoader:
    """
    Create a BigCodeBench loader from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        BigCodeBenchLoader instance
    """
    dataset_config = config.get('dataset', {})
    bigcodebench_config = dataset_config.get('bigcodebench', {})
    
    return BigCodeBenchLoader(
        mode=bigcodebench_config.get('mode', 'instruct'),
        subset=bigcodebench_config.get('subset', 'v0.1.2'),
        split=dataset_config.get('split', 'default'),
        max_samples=dataset_config.get('max_samples'),
        prompt_type=bigcodebench_config.get('prompt_type', 'default'),
    )
