"""
Dataset loader for HuggingFace datasets.
"""

from typing import Dict, List, Optional
from datasets import load_dataset
from tqdm import tqdm

from .prompts import format_prompt


class DatasetLoader:
    """
    Loader for HuggingFace datasets with prompt formatting.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        prompt_type: str = "default",
    ):
        """
        Initialize the dataset loader.

        Args:
            dataset_name: Name of the HuggingFace dataset
            split: Dataset split to load (train, test, validation)
            max_samples: Maximum number of samples to load (None for all)
            field_mapping: Mapping of field names (e.g., {"id": "id", "problem": "problem"})
            prompt_type: Type of prompt template to use
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.field_mapping = field_mapping or {
            "id": "id",
            "problem": "problem",
            "solution": "solution",
        }
        self.prompt_type = prompt_type
        self.dataset = None

    def load(self) -> List[Dict]:
        """
        Load the dataset and format with prompts.

        Returns:
            List of formatted samples with prompts
        """
        print(f"Loading dataset: {self.dataset_name} (split: {self.split})")

        # Load dataset from HuggingFace
        self.dataset = load_dataset(self.dataset_name, split=self.split)

        # Limit samples if specified
        if self.max_samples is not None:
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))

        print(f"Loaded {len(self.dataset)} samples")

        # Format samples with prompts (with auto-incremented IDs)
        formatted_samples = []
        for idx, sample in enumerate(tqdm(self.dataset, desc="Formatting samples")):
            formatted_sample = self._format_sample(sample, idx)
            formatted_samples.append(formatted_sample)

        return formatted_samples

    def _format_sample(self, sample: Dict, index: int) -> Dict:
        """
        Format a single sample with prompt.

        Args:
            sample: Raw sample from dataset
            index: Auto-incremented index for this sample

        Returns:
            Formatted sample with prompt
        """
        # Try to get ID from dataset, otherwise use auto-incremented index
        if "id" in self.field_mapping and self.field_mapping["id"] in sample:
            sample_id = sample.get(self.field_mapping["id"])
        else:
            # Auto-generate ID based on index
            sample_id = f"sample_{index:05d}"

        # Extract problem and solution fields
        problem = sample.get(self.field_mapping.get("problem", "problem"), "")
        solution = sample.get(self.field_mapping.get("solution", "solution"), "")

        # Process solution: handle list format (e.g., OlympiadBench uses ["2"] format)
        solution = self._process_solution(solution)

        # Format with prompt
        prompt = format_prompt(problem, self.prompt_type)

        return {
            "id": sample_id,
            "problem": problem,
            "solution": solution,
            "prompt": prompt,
        }

    def _process_solution(self, solution) -> str:
        """
        Process solution field to handle various formats.
        
        Handles:
        - List format: ["2"] -> "2", ["a", "b"] -> "a, b"
        - Already string: returns as-is
        - Other types: converts to string
        
        Args:
            solution: Raw solution from dataset
            
        Returns:
            Processed solution as string
        """
        if solution is None:
            return ""
        
        # Handle list format (e.g., OlympiadBench uses final_answer as ["2"])
        if isinstance(solution, list):
            if len(solution) == 0:
                return ""
            elif len(solution) == 1:
                # Single element list: extract the element
                return str(solution[0])
            else:
                # Multiple elements: join with comma
                # This handles multi-answer cases like [(2,4), (3,5)]
                return ", ".join(str(s) for s in solution)
        
        # Already a string
        if isinstance(solution, str):
            return solution
        
        # Other types: convert to string
        return str(solution)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset) if self.dataset is not None else 0


def create_dataset_loader(config: Dict):
    """
    Create a dataset loader from configuration.

    Supports:
    - Math datasets (default): HuggingFace datasets
    - Logic datasets: KOR-Bench and SynLogic
    - Code datasets: BigCodeBench

    Args:
        config: Configuration dictionary

    Returns:
        DatasetLoader instance (math, logic, or code)
    """
    dataset_config = config.get("dataset", {})
    dataset_type = dataset_config.get("type", "math")
    
    # Route to appropriate loader based on type
    if dataset_type in ["korbench", "synlogic", "logic"]:
        # Use logic dataset loader
        from .logic_dataset_loader import create_logic_dataset_loader
        return create_logic_dataset_loader(config)
    
    if dataset_type == "bigcodebench":
        # Use BigCodeBench dataset loader
        from .bigcodebench_loader import create_bigcodebench_loader
        return create_bigcodebench_loader(config)

    if dataset_type == "livecodebench":
        # Use LiveCodeBench dataset loader
        from .livecodebench_loader import create_livecodebench_loader
        return create_livecodebench_loader(config)

    # Default: math dataset loader
    # Get field mapping, with defaults
    field_mapping = dataset_config.get("fields", {})

    # Ensure we have at least problem and solution mappings
    if "problem" not in field_mapping:
        field_mapping["problem"] = "problem"
    if "solution" not in field_mapping:
        field_mapping["solution"] = "solution"
    # ID is optional - will be auto-generated if not present in dataset

    return DatasetLoader(
        dataset_name=dataset_config.get("name", "yentinglin/aime_2025"),
        split=dataset_config.get("split", "train"),
        max_samples=dataset_config.get("max_samples"),
        field_mapping=field_mapping,
        prompt_type="default",
    )
