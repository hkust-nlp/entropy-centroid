"""
Answer normalization for mathematical equivalence checking.

Implements normalization following DeepSeek/lm-eval standards:
- LaTeX command unification
- Whitespace and punctuation removal
- Unit removal
- Number format standardization
"""

import re
from typing import Optional


def normalize_latex_commands(text: str) -> str:
    """
    Normalize LaTeX commands to standard forms.

    Args:
        text: Text with LaTeX commands

    Returns:
        Normalized text
    """
    # Unify fraction commands
    text = re.sub(r'\\dfrac', r'\\frac', text)
    text = re.sub(r'\\tfrac', r'\\frac', text)

    # Remove \left and \right decorators
    text = re.sub(r'\\left\(', '(', text)
    text = re.sub(r'\\right\)', ')', text)
    text = re.sub(r'\\left\[', '[', text)
    text = re.sub(r'\\right\]', ']', text)
    text = re.sub(r'\\left\\{', '{', text)
    text = re.sub(r'\\right\\}', '}', text)
    text = re.sub(r'\\left\|', '|', text)
    text = re.sub(r'\\right\|', '|', text)

    # Remove display style modifiers
    text = re.sub(r'\\displaystyle\s*', '', text)
    text = re.sub(r'\\scriptstyle\s*', '', text)
    text = re.sub(r'\\textstyle\s*', '', text)

    # Remove text formatting
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathit\{([^}]*)\}', r'\1', text)

    # Standardize multiplication symbols
    text = re.sub(r'\\cdot', '*', text)
    text = re.sub(r'\\times', '*', text)

    # Standardize common symbols
    text = re.sub(r'\\infty', 'oo', text)  # SymPy uses 'oo' for infinity

    return text


def remove_units(text: str) -> str:
    """
    Remove common units from answers.

    Args:
        text: Text potentially containing units

    Returns:
        Text with units removed
    """
    # Common units to remove
    units = [
        r'\s*kg', r'\s*g', r'\s*mg', r'\s*lb',  # Mass
        r'\s*m', r'\s*km', r'\s*cm', r'\s*mm', r'\s*ft', r'\s*in',  # Length
        r'\s*s', r'\s*min', r'\s*hr', r'\s*h',  # Time
        r'\s*m/s', r'\s*km/h', r'\s*mph',  # Velocity
        r'\s*m\^2', r'\s*km\^2', r'\s*cm\^2',  # Area
        r'\s*m\^3', r'\s*L', r'\s*mL',  # Volume
        r'\s*°C', r'\s*°F', r'\s*K',  # Temperature
        r'\s*degrees?',  # Angles
    ]

    for unit in units:
        text = re.sub(unit + r'(?=\s|$)', '', text, flags=re.IGNORECASE)

    return text


def normalize_number_format(text: str) -> str:
    """
    Normalize number formats.

    Args:
        text: Text with numbers

    Returns:
        Text with normalized numbers
    """
    # Replace .5 with 0.5
    text = re.sub(r'(?<![0-9])(\.)\s*([0-9]+)', r'0.\2', text)

    # Remove trailing .0 from integers
    text = re.sub(r'(\d+)\.0+(?![0-9])', r'\1', text)

    # Remove commas from large numbers: 1,000 -> 1000
    text = re.sub(r'(\d+),(\d{3})', r'\1\2', text)

    return text


def remove_unnecessary_spaces(text: str) -> str:
    """
    Remove unnecessary spaces from mathematical expressions.

    Args:
        text: Text with potential extra spaces

    Returns:
        Text with cleaned spaces
    """
    # Remove spaces around operators
    text = re.sub(r'\s*([+\-*/=<>])\s*', r'\1', text)

    # Remove spaces inside parentheses
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)

    # Remove spaces inside brackets
    text = re.sub(r'\[\s+', '[', text)
    text = re.sub(r'\s+\]', ']', text)

    # Remove spaces inside braces
    text = re.sub(r'\{\s+', '{', text)
    text = re.sub(r'\s+\}', '}', text)

    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    return text


def remove_trailing_punctuation(text: str) -> str:
    """
    Remove trailing punctuation from answers.

    Args:
        text: Text with potential trailing punctuation

    Returns:
        Text without trailing punctuation
    """
    # Remove common trailing punctuation
    text = text.rstrip('.!?,;:')

    return text


def normalize_answer(answer) -> str:
    """
    Normalize answer for comparison.

    Applies comprehensive normalization including:
    - Answer format conversion (str, list, etc.)
    - LaTeX command standardization
    - Unit removal
    - Whitespace cleanup
    - Number format normalization
    - Punctuation removal

    Args:
        answer: Raw extracted answer (can be str, list, or other types)

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    # Handle different answer formats
    # Convert list to string (e.g., ["2"] -> "2")
    if isinstance(answer, list):
        if len(answer) == 0:
            return ""
        # Take first element if list, recursively normalize it
        answer = normalize_answer(answer[0])
        return answer

    # Convert non-string types to string
    if not isinstance(answer, str):
        answer = str(answer)

    # Strip leading/trailing whitespace
    answer = answer.strip()

    # Normalize LaTeX commands
    answer = normalize_latex_commands(answer)

    # Remove units
    answer = remove_units(answer)

    # Normalize number formats
    answer = normalize_number_format(answer)

    # Remove unnecessary spaces
    answer = remove_unnecessary_spaces(answer)

    # Remove trailing punctuation
    answer = remove_trailing_punctuation(answer)

    # Final whitespace cleanup
    answer = answer.strip()

    return answer


def normalize_for_text_comparison(answer) -> str:
    """
    Normalize answer for text-based comparison (non-mathematical).

    Args:
        answer: Raw answer (can be str, list, or other types)

    Returns:
        Normalized text for fuzzy matching
    """
    if not answer:
        return ""

    # Handle different answer formats
    if isinstance(answer, list):
        if len(answer) == 0:
            return ""
        # Take first element if list
        answer = answer[0]

    # Convert non-string types to string
    if not isinstance(answer, str):
        answer = str(answer)

    # Lowercase
    answer = answer.lower()

    # Remove articles
    answer = re.sub(r'\b(the|a|an)\b', '', answer)

    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)

    # Normalize whitespace
    answer = re.sub(r'\s+', ' ', answer)

    return answer.strip()
