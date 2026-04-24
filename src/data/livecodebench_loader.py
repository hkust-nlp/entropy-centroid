"""
Dataset loader for LiveCodeBench benchmark.

LiveCodeBench evaluates code generation using LeetCode/Codeforces/AtCoder style
problems with actual test case execution (stdin/stdout or function-call based).

Dataset: livecodebench/code_generation_lite (HuggingFace)
"""

import json
import sys
import os
from typing import Dict, List, Optional

from tqdm import tqdm


class LiveCodeBenchLoader:
    """
    Loader for LiveCodeBench dataset using LCB's own loading infrastructure.

    Supports release versions (release_v1 through release_v6) and date filtering.
    """

    def __init__(
        self,
        release_version: str = "release_v5",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize LiveCodeBench dataset loader.

        Args:
            release_version: Dataset release version (e.g., 'release_v5')
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            max_samples: Maximum number of samples to load (None for all)
        """
        self.release_version = release_version
        self.start_date = start_date
        self.end_date = end_date
        self.max_samples = max_samples
        self.dataset = None

    def _format_prompt(self, problem) -> str:
        """
        Construct a prompt for the given CodeGenerationProblem.

        - With starter_code (LeetCode): complete the given function
        - Without starter_code (Codeforces/AtCoder): read from stdin, write to stdout

        Args:
            problem: CodeGenerationProblem instance

        Returns:
            Formatted prompt string
        """
        question_content = problem.question_content

        if problem.starter_code and problem.starter_code.strip():
            prompt = (
                f"### Question:\n{question_content}\n\n"
                f"### Format: You will use the following starter code to write the solution "
                f"to the problem and enclose your final solution in ```python ``` code blocks.\n\n"
                f"```python\n{problem.starter_code}\n```\n\n"
                f"### Answer: (use the provided format with backticks)\n"
            )
        else:
            prompt = (
                f"### Question:\n{question_content}\n\n"
                f"### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
                f"(do not directly test on the sample inputs). Enclose your final solution "
                f"in ```python ``` code blocks.\n\n"
                f"```python\n# YOUR CODE HERE\n```\n\n"
                f"### Answer: (use the provided format with backticks)\n"
            )

        return prompt

    def load(self) -> List[Dict]:
        """
        Load the LiveCodeBench dataset and format with prompts.

        Returns:
            List of formatted samples with prompts
        """
        # Add LiveCodeBench directory to path for imports
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        lcb_path = os.path.join(project_root, "LiveCodeBench")
        if lcb_path not in sys.path:
            sys.path.insert(0, lcb_path)

        from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

        print(f"Loading LiveCodeBench dataset (version: {self.release_version})")
        if self.start_date:
            print(f"  Start date filter: {self.start_date}")
        if self.end_date:
            print(f"  End date filter: {self.end_date}")

        problems = load_code_generation_dataset(
            release_version=self.release_version,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        print(f"Loaded {len(problems)} problems")

        if self.max_samples is not None:
            problems = problems[:self.max_samples]
            print(f"Limited to {len(problems)} samples")

        formatted_samples = []
        for problem in tqdm(problems, desc="Formatting LiveCodeBench samples"):
            formatted_sample = self._format_sample(problem)
            formatted_samples.append(formatted_sample)

        print(f"Formatted {len(formatted_samples)} samples")
        return formatted_samples

    def _format_sample(self, problem) -> Dict:
        """
        Format a single CodeGenerationProblem into pipeline format.

        Args:
            problem: CodeGenerationProblem instance

        Returns:
            Formatted sample dictionary
        """
        question_id = problem.question_id
        prompt = self._format_prompt(problem)

        # Serialize test cases for storage
        public_tests = [
            {"input": t.input, "output": t.output, "testtype": t.testtype.value}
            for t in problem.public_test_cases
        ] if problem.public_test_cases else []

        private_tests = [
            {"input": t.input, "output": t.output, "testtype": t.testtype.value}
            for t in problem.private_test_cases
        ] if problem.private_test_cases else []

        return {
            "id": f"lcb_{question_id}",
            "problem": problem.question_content,
            "solution": None,
            "prompt": prompt,
            "question_id": question_id,
            "starter_code": problem.starter_code or "",
            "platform": problem.platform.value if hasattr(problem.platform, 'value') else str(problem.platform),
            "difficulty": problem.difficulty.value if hasattr(problem.difficulty, 'value') else str(problem.difficulty),
            "public_test_cases": public_tests,
            "private_test_cases": private_tests,
            "metadata": problem.metadata if isinstance(problem.metadata, dict) else {},
            "source": "livecodebench",
        }

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset) if self.dataset is not None else 0


def create_livecodebench_loader(config: Dict) -> LiveCodeBenchLoader:
    """
    Create a LiveCodeBench loader from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        LiveCodeBenchLoader instance
    """
    dataset_config = config.get("dataset", {})
    lcb_config = dataset_config.get("livecodebench", {})

    return LiveCodeBenchLoader(
        release_version=lcb_config.get("release_version", "release_v5"),
        start_date=lcb_config.get("start_date"),
        end_date=lcb_config.get("end_date"),
        max_samples=dataset_config.get("max_samples"),
    )
