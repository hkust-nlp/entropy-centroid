"""
Answer extraction from model outputs.

Implements multi-priority extraction strategy:
1. LaTeX boxed notation (\boxed{...})
2. Keyword-based fallback (final answer:, the answer is, etc.)
3. Last line extraction
"""

import re
from typing import Optional, Tuple


def extract_boxed_answer(text: str) -> Tuple[Optional[str], int, int]:
    """
    Extract answer from \boxed{...} notation.

    Handles nested braces and multiple boxed answers (takes last one).

    Args:
        text: Text containing boxed answer

    Returns:
        Tuple of (content, start_pos, end_pos) or (None, -1, -1) if not found
    """
    # Ensure text is a string
    if not isinstance(text, str):
        return None, -1, -1

    # Pattern to find \boxed{
    boxed_pattern = r'\\boxed\s*\{'

    matches = list(re.finditer(boxed_pattern, text))
    if not matches:
        return None, -1, -1

    # Take the LAST \boxed{...} as the final answer
    last_match = matches[-1]
    start = last_match.end() - 1  # Position of opening brace

    # Balance braces to find matching closing brace
    brace_count = 1
    i = start + 1

    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        # Found matching closing brace
        content = text[start + 1:i - 1]
        return content, start, i
    else:
        # Incomplete boxed answer
        return None, -1, -1


def extract_with_keywords(text: str) -> Optional[str]:
    """
    Extract answer using keyword patterns.

    Searches for keywords like "final answer:", "the answer is", etc.
    and extracts the content after them.

    Args:
        text: Text containing answer

    Returns:
        Extracted answer or None
    """
    # Keywords that typically precede the final answer
    keywords = [
        r'final answer\s*:?\s*',
        r'the answer is\s*:?\s*',
        r'therefore\s*,?\s*',
        r'thus\s*,?\s*',
        r'so\s+the\s+answer\s+is\s*:?\s*',
        r'answer\s*:?\s*',
    ]

    # Try each keyword pattern
    for keyword in keywords:
        # Case-insensitive search for keyword followed by content
        pattern = keyword + r'(.+?)(?:\n\n|$)'
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))

        if matches:
            # Take the LAST occurrence
            last_match = matches[-1]
            content = last_match.group(1).strip()

            # Remove trailing punctuation
            content = content.rstrip('.!?,;')

            if content:
                return content

    return None


def extract_last_line(text: str) -> Optional[str]:
    """
    Extract answer from the last non-empty line.

    Args:
        text: Text to extract from

    Returns:
        Last non-empty line or None
    """
    lines = text.strip().split('\n')

    # Find last non-empty line
    for line in reversed(lines):
        line = line.strip()
        if line:
            # Remove common prefixes
            line = re.sub(r'^(answer|result|solution)\s*:?\s*', '', line, flags=re.IGNORECASE)
            line = line.rstrip('.!?,;')
            return line

    return None


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer from model output using multi-priority strategy.

    Priority order:
    1. \boxed{...} notation (highest priority)
    2. Keyword-based extraction (final answer:, the answer is, etc.)
    3. Last line extraction (fallback)

    Args:
        text: Model output text

    Returns:
        Extracted answer string (raw, before normalization) or None
    """
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text) if text is not None else ''

    if not text:
        return None

    # Priority 1: Boxed answer
    content, _, _ = extract_boxed_answer(text)
    if content is not None:
        return content

    # Priority 2: Keyword-based extraction
    content = extract_with_keywords(text)
    if content is not None:
        return content

    # Priority 3: Last line extraction
    content = extract_last_line(text)
    if content is not None:
        return content

    # Failed to extract
    return None


def extract_answer_with_metadata(text: str) -> dict:
    """
    Extract answer with metadata about extraction method.

    Args:
        text: Model output text

    Returns:
        Dictionary with answer and extraction metadata:
        {
            'answer': str or None,
            'method': str ('boxed', 'keyword', 'last_line', 'failed'),
            'confidence': float (0.0-1.0)
        }
    """
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text) if text is not None else ''

    if not text:
        return {
            'answer': None,
            'method': 'failed',
            'confidence': 0.0
        }

    # Try boxed extraction
    content, _, _ = extract_boxed_answer(text)
    if content is not None:
        return {
            'answer': content,
            'method': 'boxed',
            'confidence': 1.0
        }

    # Try keyword extraction
    content = extract_with_keywords(text)
    if content is not None:
        return {
            'answer': content,
            'method': 'keyword',
            'confidence': 0.8
        }

    # Try last line extraction
    content = extract_last_line(text)
    if content is not None:
        return {
            'answer': content,
            'method': 'last_line',
            'confidence': 0.5
        }

    # Failed to extract
    return {
        'answer': None,
        'method': 'failed',
        'confidence': 0.0
    }


def extract_amo_bench_answer(text: str) -> Optional[str]:
    """
    Extract answer from AMO-Bench format.
    
    Tries:
    1. \\boxed{} notation (most common)
    2. "### The final answer is:" pattern
    3. Keyword-based extraction
    
    Args:
        text: Generated text
        
    Returns:
        Extracted answer or None
    """
    if not isinstance(text, str):
        return None
    
    # First try boxed notation
    boxed, _, _ = extract_boxed_answer(text)
    if boxed:
        return boxed
    
    # Try AMO-Bench specific patterns
    amo_patterns = [
        r'###\s*[Tt]he\s+final\s+answer\s+is[:\s]*(.+?)(?:\n|$)',
        r'###\s*[Ff]inal\s+answer[:\s]*(.+?)(?:\n|$)',
        r'[Tt]he\s+final\s+answer\s+is[:\s]*(.+?)(?:\n|$)',
    ]
    
    for pattern in amo_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Clean up the answer
            answer = answer.rstrip('.!?,;')
            # If the answer contains boxed, extract it
            boxed_in_answer, _, _ = extract_boxed_answer(answer)
            if boxed_in_answer:
                return boxed_in_answer
            return answer if answer else None
    
    # Fallback to keyword extraction
    return extract_with_keywords(text)


def get_task_answer_extractor(task_type: str):
    """
    Get the appropriate answer extraction function for a task type.
    
    Args:
        task_type: Type of task ('math', 'korbench', 'synlogic', 'amo_bench')
        
    Returns:
        Function that takes text and returns extracted answer
    """
    if task_type == 'math':
        return extract_answer
    elif task_type == 'amo_bench':
        return extract_amo_bench_answer
    elif task_type in ('korbench', 'synlogic'):
        # Logic tasks use [[answer]] format
        def extract_logic_answer(text: str) -> Optional[str]:
            if not isinstance(text, str):
                return None
            # Try [[answer]] format first
            match = re.search(r'\[\[([^\]]+)\]\]', text)
            if match:
                return match.group(1).strip()
            # Try <answer>...</answer> format
            match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
            # Fallback to boxed notation
            boxed, _, _ = extract_boxed_answer(text)
            if boxed:
                return boxed
            return None
        return extract_logic_answer
    else:
        # Default to math extraction
        return extract_answer
