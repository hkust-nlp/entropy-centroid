"""
Prompt templates for logic reasoning tasks.

Includes system prompts and templates for:
- KOR-Bench tasks (cipher, logic, operation, puzzle, counterfactual)
- SynLogic tasks (various puzzle and logic games)
"""

# System prompts for different task types
LOGIC_SYSTEM_PROMPTS = {
    "default": "You are a logic reasoning expert. Please solve the following problem step by step and provide your final answer.",
    
    "korbench": "You are a logic reasoning expert. Follow the given rules carefully and solve the problem. Format your answer within double brackets like [[answer]].",
    
    "korbench_cipher": "You are a cipher and code expert. Follow the given encoding/decoding rules carefully and solve the problem. Format your answer within double brackets like [[answer]].",
    
    "korbench_logic": "You are a formal logic expert. Follow the given logical rules and axioms carefully and solve the problem. Format your answer within double brackets like [[answer]].",
    
    "korbench_operation": "You are a mathematical operation expert. Follow the given operation rules carefully and solve the problem. Format your answer within double brackets like [[answer]].",
    
    "korbench_puzzle": "You are a puzzle solving expert. Follow the given puzzle rules carefully and solve the problem. Format your answer within double brackets like [[answer]].",
    
    "korbench_counterfactual": "You are a counterfactual reasoning expert. Reason based on the given hypothetical rules (not real-world rules) and solve the problem. Format your answer within double brackets like [[answer]].",
    
    "synlogic": "You are a logic puzzle solving expert. Analyze the problem carefully, think step by step, and provide your final answer within <answer> and </answer> tags.",
    
    "synlogic_puzzle": "You are a puzzle solving expert. Analyze the grid/puzzle constraints carefully, think step by step, and provide your solution. Put your final answer within <answer> and </answer> tags.",
    
    "synlogic_reasoning": "You are a logical reasoning expert. Analyze the given information carefully, apply logical deductions, and provide your final answer within <answer> and </answer> tags.",
}


def get_logic_system_prompt(task_type: str = "default", category: str = None) -> str:
    """
    Get the appropriate system prompt for a logic task.
    
    Args:
        task_type: Type of task (korbench, synlogic, default)
        category: Optional category for more specific prompts
        
    Returns:
        System prompt string
    """
    # Try to get category-specific prompt
    if task_type == "korbench" and category:
        key = f"korbench_{category}"
        if key in LOGIC_SYSTEM_PROMPTS:
            return LOGIC_SYSTEM_PROMPTS[key]
        return LOGIC_SYSTEM_PROMPTS["korbench"]
    
    # Try task type
    if task_type in LOGIC_SYSTEM_PROMPTS:
        return LOGIC_SYSTEM_PROMPTS[task_type]
    
    return LOGIC_SYSTEM_PROMPTS["default"]


# User prompt templates
LOGIC_USER_PROMPTS = {
    "default": "{problem}",
    "with_instruction": "Please solve the following problem:\n\n{problem}",
}


def format_logic_prompt(problem: str, prompt_type: str = "default") -> str:
    """
    Format a logic problem with the prompt template.
    
    Args:
        problem: The logic problem to solve
        prompt_type: Type of prompt template to use
        
    Returns:
        Formatted prompt string
    """
    template = LOGIC_USER_PROMPTS.get(prompt_type, LOGIC_USER_PROMPTS["default"])
    return template.format(problem=problem)
