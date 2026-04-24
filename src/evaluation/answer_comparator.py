"""
Answer comparison with symbolic equivalence checking.

Implements multi-strategy comparison:
1. Exact string match (after normalization)
2. Numeric comparison with tolerance
3. Symbolic comparison via SymPy (with timeout)
4. Fuzzy string match (fallback)

Default math grader supports:
- Multi-answer support (comma-separated, unordered)
- Tuple/list comparison (ordered)
- Equation form answers (f(x)=1, x=5)
- Choice answers (A, B, C, D, E)
- Matrix comparison
"""

import re
import signal
from typing import Dict, Optional, Tuple
from difflib import SequenceMatcher
import numpy as np

# Try to import math grader
try:
    from .math_grader import (
        check_is_correct as grader_check_correct,
        strip_string as grader_strip_string,
        math_equal as grader_math_equal,
        SYMPY_AVAILABLE as GRADER_SYMPY_AVAILABLE,
        LATEX2SYMPY_AVAILABLE as GRADER_LATEX2SYMPY_AVAILABLE,
    )
    MATH_GRADER_AVAILABLE = True
except ImportError:
    MATH_GRADER_AVAILABLE = False
    grader_check_correct = None
    grader_strip_string = None
    grader_math_equal = None

# Try to import symbolic math libraries
try:
    import sympy
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: sympy not available. Symbolic comparison disabled.")

try:
    from latex2sympy2_extended import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    LATEX2SYMPY_AVAILABLE = False
    print("Warning: latex2sympy2 not available. LaTeX parsing will be limited.")


class TimeoutError(Exception):
    """Raised when symbolic comparison times out."""
    pass


class AnswerComparator:
    """
    Compares mathematical answers using multiple strategies.
    
    Supports two modes:
    1. Default mode: Uses math grader for robust comparison
       - Multi-answer support (comma-separated, unordered)
       - Tuple/list comparison (ordered)
       - Equation form answers
       - Choice answers
       - Matrix comparison
    
    2. Legacy mode: Uses original multi-strategy approach
       - Exact string match
       - Numeric comparison
       - Symbolic comparison
       - Fuzzy string match
    """

    def __init__(
        self,
        numeric_rtol: float = 1e-6,
        numeric_atol: float = 1e-9,
        fuzzy_threshold: float = 0.9,
        timeout: int = 5,
        use_limo_grader: bool = True,  # Kept for backward compatibility
        use_math_grader: bool = True,
    ):
        """
        Initialize answer comparator.

        Args:
            numeric_rtol: Relative tolerance for numeric comparison
            numeric_atol: Absolute tolerance for numeric comparison
            fuzzy_threshold: Similarity threshold for fuzzy string matching
            timeout: Maximum seconds for symbolic comparison
            use_limo_grader: Deprecated, use use_math_grader instead
            use_math_grader: Whether to use default math grader (default: True)
        """
        self.numeric_rtol = numeric_rtol
        self.numeric_atol = numeric_atol
        self.fuzzy_threshold = fuzzy_threshold
        self.timeout = timeout
        # Support both old and new parameter names
        self.use_math_grader = (use_limo_grader or use_math_grader) and MATH_GRADER_AVAILABLE

    def _run_with_timeout(self, func, args, timeout: Optional[int] = None):
        """
        Run function with timeout using Unix signals.

        Args:
            func: Function to run
            args: Arguments tuple for function
            timeout: Timeout in seconds (uses self.timeout if None)

        Returns:
            Function result

        Raises:
            TimeoutError: If function exceeds timeout
        """
        if timeout is None:
            timeout = self.timeout

        def handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

        # Set signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)

        try:
            result = func(*args)
        finally:
            # Disable alarm and restore old handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        return result

    def _exact_match(self, ans1: str, ans2: str) -> bool:
        """
        Check exact string match.

        Args:
            ans1: First answer (normalized)
            ans2: Second answer (normalized)

        Returns:
            True if exactly equal
        """
        return ans1 == ans2

    def _numeric_match(self, ans1: str, ans2: str) -> Tuple[bool, Optional[str]]:
        """
        Check numeric equivalence with tolerance.

        Args:
            ans1: First answer
            ans2: Second answer

        Returns:
            Tuple of (is_match, error_message)
        """
        try:
            # Try to parse as float
            val1 = self._parse_number(ans1)
            val2 = self._parse_number(ans2)

            if val1 is None or val2 is None:
                return False, "Not numeric"

            # Use numpy's close comparison
            is_close = np.isclose(val1, val2, rtol=self.numeric_rtol, atol=self.numeric_atol)
            return bool(is_close), None

        except Exception as e:
            return False, f"Numeric comparison error: {str(e)}"

    def _parse_number(self, text: str) -> Optional[float]:
        """
        Parse number from text, handling various formats.

        Args:
            text: Text to parse

        Returns:
            Parsed number or None
        """
        # Remove LaTeX formatting for simple cases
        text = text.replace('\\', '')

        # Handle fractions: a/b
        frac_match = re.match(r'^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$', text.strip())
        if frac_match:
            try:
                numerator = float(frac_match.group(1))
                denominator = float(frac_match.group(2))
                return numerator / denominator if denominator != 0 else None
            except:
                pass

        # Handle LaTeX frac: \frac{a}{b}
        frac_match = re.match(r'^frac\{([^}]+)\}\{([^}]+)\}$', text.strip())
        if frac_match:
            try:
                numerator = float(frac_match.group(1))
                denominator = float(frac_match.group(2))
                return numerator / denominator if denominator != 0 else None
            except:
                pass

        # Try direct float conversion
        try:
            return float(text)
        except:
            return None

    def _symbolic_match(self, ans1: str, ans2: str) -> Tuple[bool, Optional[str]]:
        """
        Check symbolic equivalence using SymPy.

        Args:
            ans1: First answer (LaTeX or expression)
            ans2: Second answer (LaTeX or expression)

        Returns:
            Tuple of (is_match, error_message)
        """
        if not SYMPY_AVAILABLE:
            return False, "SymPy not available"

        try:
            # Convert LaTeX to SymPy expressions
            expr1 = self._latex_to_sympy(ans1)
            expr2 = self._latex_to_sympy(ans2)

            if expr1 is None or expr2 is None:
                return False, "Failed to parse expressions"

            # Check equivalence with timeout
            def check_equivalence(e1, e2):
                try:
                    # Method 1: Check if difference simplifies to zero
                    diff = sympy.simplify(e1 - e2)
                    if diff == 0 or diff == sympy.S.Zero:
                        return True

                    # Method 2: Check if ratio simplifies to 1 (for non-zero expressions)
                    if e2 != 0:
                        ratio = sympy.simplify(e1 / e2)
                        if ratio == 1 or ratio == sympy.S.One:
                            return True

                    # Method 3: Expand and compare
                    if sympy.expand(e1 - e2) == 0:
                        return True

                    return False
                except Exception as e:
                    raise e

            is_equiv = self._run_with_timeout(check_equivalence, (expr1, expr2))
            return is_equiv, None

        except TimeoutError as e:
            return False, f"Symbolic comparison timeout: {str(e)}"
        except Exception as e:
            return False, f"Symbolic comparison error: {str(e)}"

    def _latex_to_sympy(self, latex_str: str) -> Optional["sympy.Expr"]:
        """
        Convert LaTeX string to SymPy expression.

        Args:
            latex_str: LaTeX mathematical expression

        Returns:
            SymPy expression or None
        """
        import warnings
        
        try:
            # Suppress SyntaxWarnings from SymPy when parsing unusual expressions
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=SyntaxWarning)
                
                # Try latex2sympy2 first (more robust)
                if LATEX2SYMPY_AVAILABLE:
                    try:
                        expr = latex2sympy(latex_str)
                        return expr
                    except:
                        pass

                # Fallback to SymPy's built-in parser
                if SYMPY_AVAILABLE:
                    try:
                        # Check if it looks like LaTeX
                        if '\\' in latex_str:
                            expr = parse_latex(latex_str)
                        else:
                            # Try parsing as SymPy expression directly
                            expr = sympy.sympify(latex_str, evaluate=False)
                        return expr
                    except:
                        pass

            return None

        except Exception:
            return None

    def _fuzzy_match(self, ans1: str, ans2: str) -> Tuple[bool, float]:
        """
        Check fuzzy string similarity.

        Args:
            ans1: First answer
            ans2: Second answer

        Returns:
            Tuple of (is_match, similarity_score)
        """
        # Use SequenceMatcher for fuzzy comparison
        similarity = SequenceMatcher(None, ans1.lower(), ans2.lower()).ratio()
        is_match = similarity >= self.fuzzy_threshold

        return is_match, similarity

    @staticmethod
    def _clean_latex(s: str) -> str:
        """
        Strip LaTeX formatting, units, and decorations from an answer.

        Handles patterns that strip_string misses:
        - \\text{cm}^{-1} (unit with exponent)
        - \\ (LaTeX forced space)
        - \\approx, \\displaystyle, etc.
        """
        if not s:
            return s
        import re as _re
        # Remove \text{...} and optional trailing ^{...} (unit exponents)
        s = _re.sub(r'\\text\{[^}]*\}(\s*\^\{[^}]*\})?', '', s)
        # Remove LaTeX spacing: \, \; \: \! and '\ '
        s = _re.sub(r'\\[,;:!]', '', s)
        s = _re.sub(r'\\ ', ' ', s)
        # Remove decorators
        s = _re.sub(r'\\(approx|sim|simeq|displaystyle|mathrm|mathbf|mathit|'
                     r'left|right|bigl|bigr|Bigl|Bigr|quad|qquad)', '', s)
        s = _re.sub(r'\s+', ' ', s).strip().strip(' ,;.')
        return s

    def compare(self, ground_truth: str, generated: str) -> Dict:
        """
        Compare two answers using multi-strategy approach.

        First attempts comparison with the raw answer. If that fails,
        retries with LaTeX-cleaned answer (strips units, spacing commands,
        and decorators that strip_string may miss).

        Args:
            ground_truth: Ground truth answer (normalized)
            generated: Generated answer (normalized)

        Returns:
            Comparison result dictionary with:
            {
                'is_correct': bool,
                'ground_truth_raw': str,
                'generated_raw': str,
                'ground_truth_normalized': str,
                'generated_normalized': str,
                'comparison_method': str,
                'confidence': float,
                'error_message': Optional[str]
            }
        """
        # Store raw answers for reference
        gt_raw = ground_truth
        gen_raw = generated

        # First pass: compare with raw answer
        if self.use_math_grader and grader_check_correct is not None:
            result = self._compare_with_grader(gt_raw, gen_raw)
        else:
            result = self._compare_legacy(gt_raw, gen_raw)

        if result['is_correct']:
            return result

        # Second pass: retry with LaTeX-cleaned answer
        gen_cleaned = self._clean_latex(gen_raw)
        if gen_cleaned != gen_raw and gen_cleaned:
            if self.use_math_grader and grader_check_correct is not None:
                result2 = self._compare_with_grader(gt_raw, gen_cleaned)
            else:
                result2 = self._compare_legacy(gt_raw, gen_cleaned)

            if result2['is_correct']:
                result2['comparison_method'] += '_cleaned'
                result2['generated_raw'] = gen_raw  # Preserve original
                return result2

        return result

    def _preprocess_answer(self, answer) -> str:
        """
        Preprocess answer to handle various formats.
        
        Handles:
        - List format: ["2"] -> "2", [2] -> "2"
        - String representation of list: "[2]" -> "2", "['a']" -> "a"
        - Already string: returns as-is
        
        Args:
            answer: Raw answer (string, list, or other)
            
        Returns:
            Preprocessed answer as string
        """
        if answer is None:
            return ""
        
        # Handle actual list objects
        if isinstance(answer, list):
            if len(answer) == 0:
                return ""
            elif len(answer) == 1:
                return str(answer[0])
            else:
                return ", ".join(str(a) for a in answer)
        
        # Convert to string
        answer_str = str(answer)
        
        # Handle string representation of single-element list: "[2]" or "['2']"
        # Match patterns like [2], ['2'], ["2"], [value]
        import re
        
        # Pattern for single-element list representation
        single_list_pattern = r'^\s*\[\s*[\'"]?([^\[\]\'\"]+)[\'"]?\s*\]\s*$'
        match = re.match(single_list_pattern, answer_str)
        if match:
            return match.group(1).strip()
        
        return answer_str

    def _compare_with_grader(self, ground_truth: str, generated: str) -> Dict:
        """
        Compare using math grader.
        
        This method supports:
        - Multi-answer comparison (comma-separated, unordered)
        - Tuple/list comparison (ordered)
        - Equation form answers (f(x)=1, x=5)
        - Choice answers (A, B, C, D, E)
        - Matrix comparison
        - Percentage tolerance
        """
        # Preprocess to handle list formats (e.g., OlympiadBench uses ["2"])
        ground_truth = self._preprocess_answer(ground_truth)
        generated = self._preprocess_answer(generated)
        
        gt_raw = ground_truth
        gen_raw = generated
        
        # Normalize using grader's strip_string
        gt_norm = grader_strip_string(str(ground_truth)) if ground_truth else ""
        gen_norm = grader_strip_string(str(generated)) if generated else ""
        
        result = {
            'ground_truth_raw': gt_raw,
            'generated_raw': gen_raw,
            'ground_truth_normalized': gt_norm,
            'generated_normalized': gen_norm,
            'is_correct': False,
            'comparison_method': 'math_grader',
            'confidence': 0.0,
            'error_message': None,
        }
        
        try:
            # Use grader's check_is_correct (which uses math_equal internally)
            is_correct = grader_check_correct(str(generated), str(ground_truth), timeout=True)
            
            if is_correct:
                result['is_correct'] = True
                result['confidence'] = 0.95
                
                # Try to determine the specific comparison method used
                if gt_norm == gen_norm:
                    result['comparison_method'] = 'exact'
                    result['confidence'] = 1.0
                elif "," in gt_norm and "," in gen_norm:
                    result['comparison_method'] = 'multi_answer'
                elif gt_norm.startswith("(") or gt_norm.startswith("["):
                    result['comparison_method'] = 'tuple'
                elif "=" in gt_norm or "=" in gen_norm:
                    result['comparison_method'] = 'equation'
                else:
                    result['comparison_method'] = 'symbolic'
            else:
                result['comparison_method'] = 'no_match'
                
        except Exception as e:
            result['error_message'] = f"Math grader error: {str(e)}"
            result['comparison_method'] = 'grader_error'
            # Fall back to legacy comparison on error
            legacy_result = self._compare_legacy(ground_truth, generated)
            if legacy_result['is_correct']:
                return legacy_result
        
        return result

    def _compare_legacy(self, ground_truth: str, generated: str) -> Dict:
        """
        Legacy comparison using original multi-strategy approach.
        """
        # Preprocess to handle list formats
        ground_truth = self._preprocess_answer(ground_truth)
        generated = self._preprocess_answer(generated)
        
        gt_raw = ground_truth
        gen_raw = generated

        # Normalize answers
        from .answer_normalizer import normalize_answer
        gt_norm = normalize_answer(ground_truth)
        gen_norm = normalize_answer(generated)

        result = {
            'ground_truth_raw': gt_raw,
            'generated_raw': gen_raw,
            'ground_truth_normalized': gt_norm,
            'generated_normalized': gen_norm,
            'is_correct': False,
            'comparison_method': 'none',
            'confidence': 0.0,
            'error_message': None,
        }

        # Strategy 1: Exact match
        if self._exact_match(gt_norm, gen_norm):
            result['is_correct'] = True
            result['comparison_method'] = 'exact'
            result['confidence'] = 1.0
            return result

        # Strategy 2: Numeric comparison
        is_numeric, error = self._numeric_match(gt_norm, gen_norm)
        if is_numeric:
            result['is_correct'] = True
            result['comparison_method'] = 'numeric'
            result['confidence'] = 0.95
            return result

        # Strategy 3: Symbolic comparison (with timeout)
        is_symbolic, error = self._symbolic_match(gt_norm, gen_norm)
        if error and 'timeout' in error.lower():
            result['comparison_method'] = 'timeout'
            result['confidence'] = 0.0
            result['error_message'] = error
            return result
        elif is_symbolic:
            result['is_correct'] = True
            result['comparison_method'] = 'symbolic'
            result['confidence'] = 0.9
            return result

        # Strategy 4: Fuzzy match (fallback)
        # is_fuzzy, similarity = self._fuzzy_match(gt_norm, gen_norm)
        # if is_fuzzy:
        #     result['is_correct'] = True
        #     result['comparison_method'] = 'fuzzy'
        #     result['confidence'] = similarity
        #     return result

        # No match found
        result['comparison_method'] = 'no_match'
        result['confidence'] = 0.0
        return result


def compare_answers(ground_truth: str, generated: str, use_math_grader: bool = True, **kwargs) -> Dict:
    """
    Convenience function for comparing two answers.

    Args:
        ground_truth: Ground truth answer
        generated: Generated answer
        use_math_grader: Whether to use math grader (default: True)
        **kwargs: Additional arguments for AnswerComparator

    Returns:
        Comparison result dictionary
    """
    comparator = AnswerComparator(use_math_grader=use_math_grader, **kwargs)
    return comparator.compare(ground_truth, generated)


def check_math_equivalence(pred: str, gt: str, timeout: bool = True) -> bool:
    """
    Check if two mathematical answers are equivalent.
    
    This is the simplest way to check answer equivalence.
    Supports:
    - Multi-answer comparison (comma-separated)
    - Tuple/list comparison
    - Equation forms (f(x)=1)
    - Choice answers (A, B, C, D, E)
    - Matrix comparison
    - Percentage tolerance
    
    Args:
        pred: Predicted answer
        gt: Ground truth answer
        timeout: Whether to use timeout for symbolic comparison
    
    Returns:
        True if answers are equivalent, False otherwise
    """
    if MATH_GRADER_AVAILABLE and grader_check_correct is not None:
        try:
            return grader_check_correct(pred, gt, timeout=timeout)
        except Exception:
            pass
    
    # Fallback to simple comparison
    comparator = AnswerComparator(use_math_grader=False)
    result = comparator.compare(gt, pred)
    return result['is_correct']
