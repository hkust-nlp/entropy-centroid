"""
Prompt templates for code generation tasks.

Used by BigCodeBench and other code-related benchmarks.
"""

# NOTE: System messages are handled by chat template in vllm_engine.py
# These prompts are for the user message content only

# Default code prompt templates
CODE_PROMPTS = {
    # For instruction-following code generation
    "instruct_default": "Task:\n{problem}",
    "instruct_detailed": "Please write Python code to solve the following task:\n\n{problem}\n\nProvide a complete, working solution.",
    "instruct_concise": "{problem}",
    
    # For code completion
    "complete_default": "Complete the code:\n{problem}",
    "complete_detailed": "Complete the following Python code:\n\n{problem}",
    "complete_concise": "{problem}",
}


def get_code_prompt(prompt_type: str = "instruct_default") -> str:
    """
    Get a code prompt template by type.
    
    Args:
        prompt_type: Type of prompt to use (instruct_default, complete_default, etc.)
        
    Returns:
        The prompt template string
    """
    return CODE_PROMPTS.get(prompt_type, CODE_PROMPTS["instruct_default"])


def format_code_prompt(problem: str, prompt_type: str = "instruct_default") -> str:
    """
    Format a code problem with the prompt template.
    
    Args:
        problem: The code task description or partial code
        prompt_type: Type of prompt template to use
        
    Returns:
        Formatted prompt string (user message only)
    """
    template = get_code_prompt(prompt_type)
    return template.format(problem=problem)
