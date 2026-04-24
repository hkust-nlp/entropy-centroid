"""
Prompt templates for mathematical reasoning tasks.
"""

# NOTE: System messages are now handled by chat template in vllm_engine.py
# These prompts are for the user message content only

# Default math prompt template (simple format)
DEFAULT_MATH_PROMPT = "Problem: {problem}"

# Alternative prompt templates
MATH_PROMPTS = {
    "default": "Problem: {problem}",
    "cot": "Please solve the following problem:\n\n{problem}",
    "detailed": "Problem to solve:\n\n{problem}",
    "concise": "{problem}",
}


def get_math_prompt(prompt_type: str = "default") -> str:
    """
    Get a math prompt template by type.

    Args:
        prompt_type: Type of prompt to use (default, cot, detailed, concise)

    Returns:
        The prompt template string
    """
    return MATH_PROMPTS.get(prompt_type, DEFAULT_MATH_PROMPT)


def format_prompt(problem: str, prompt_type: str = "default") -> str:
    """
    Format a problem with the prompt template.

    Note: System-level instructions are now handled by chat template in vllm_engine.py.
    This function only formats the user message content.

    Args:
        problem: The math problem to solve
        prompt_type: Type of prompt template to use

    Returns:
        Formatted prompt string (user message only)
    """
    template = get_math_prompt(prompt_type)
    return template.format(problem=problem)

