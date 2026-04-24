"""
LIMO-compatible math grader for robust answer comparison.

This module is adapted from LIMO/eval/utils/grader.py and provides:
- Comma-separated multiple answers (unordered comparison)
- Tuple/list answers (ordered comparison)
- Equation form answers (f(x)=1, x=5)
- Choice answers (A, B, C, D, E)
- Percentage tolerance
- Matrix comparison
- Symbolic equivalence via SymPy

Original source: https://github.com/GAIR-NLP/LIMO
License: Apache 2.0
"""

import re
import multiprocessing
from math import isclose
from typing import Union, Optional, List

# Try to import regex (more powerful than re)
try:
    import regex
    REGEX_AVAILABLE = True
except ImportError:
    import re as regex
    REGEX_AVAILABLE = False

# SymPy imports
try:
    from sympy import simplify, N
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    simplify = None
    N = None
    parse_expr = None
    parse_latex = None

# latex2sympy imports (prefer latex2sympy2_extended)
try:
    from latex2sympy2_extended import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    try:
        from latex2sympy2 import latex2sympy
        LATEX2SYMPY_AVAILABLE = True
    except ImportError:
        latex2sympy = None
        LATEX2SYMPY_AVAILABLE = False

# word2number for converting word numbers to digits
try:
    from word2number import w2n
    WORD2NUMBER_AVAILABLE = True
except ImportError:
    WORD2NUMBER_AVAILABLE = False


# =============================================================================
# Unit texts for removal (from LIMO parser.py)
# =============================================================================
UNIT_TEXTS = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m", "deg",
    "mile", "q .", "monkey", "prime", "ratio", "profit of rs", "rd", "o", "gm",
    "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab", "way", "west",
    "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .", "t", "a", "h",
    "no change", "men", "soldier", "pie", "bc", "excess", "st", "inches",
    "noon", "percent", "by", "gal", "kmh", "c", "acre", "rise", "a . m", "th",
    "π r 2", "sq", "mark", "l", "toy", "coin", "sq . m", "gallon", "° f",
    "profit", "minw", "yr", "women", "feet", "am", "pm", "hr", "cu cm",
    "square", "v â € ™", "are", "rupee", "rounds", "cubic", "cc", "mtr", "s",
    "ohm", "number", "kmph", "day", "hour", "minute", "min", "second", "man",
    "woman", "sec", "cube", "mt", "sq inch", "mp", "∏ cm ³", "hectare", "more",
    "sec", "unit", "cu . m", "cm 2", "rs .", "rs", "kg", "g", "month", "km",
    "m", "cm", "mm", "apple", "liter", "loss", "yard", "pure", "year",
    "increase", "decrease", "d", "less", "Surface", "litre", "pi sq m", "s .",
    "metre", "meter", "inch",
]
# Add plural forms
UNIT_TEXTS.extend([t + "s" for t in UNIT_TEXTS])


# =============================================================================
# Choice answer patterns (from LIMO grader.py)
# =============================================================================
SINGLE_CHOICE_PATTERNS = [
    r"^\(A\)", r"^\(B\)", r"^\(C\)", r"^\(D\)", r"^\(E\)",  # (A) (B) (C) (D) (E)
    r"^A\.", r"^B\.", r"^C\.", r"^D\.", r"^E\.",            # A. B. C. D. E.
    r"^A\)", r"^B\)", r"^C\)", r"^D\)", r"^E\)",            # A) B) C) D) E)
    r"^\*\*A\*\*", r"^\*\*B\*\*", r"^\*\*C\*\*", r"^\*\*D\*\*", r"^\*\*E\*\*",  # **A** **B**
    r"^A:", r"^B:", r"^C:", r"^D:", r"^E:",                 # A: B: C: D: E:
]


# =============================================================================
# Helper functions
# =============================================================================

def convert_word_number(text: str) -> str:
    """Convert word numbers to digits (e.g., 'three' -> '3')"""
    if not WORD2NUMBER_AVAILABLE:
        return text
    try:
        return str(w2n.word_to_num(text))
    except:
        return text


def _fix_fracs(string: str) -> str:
    """Fix LaTeX fraction formatting"""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    """Convert a/b to \\frac{a}{b}"""
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string: str) -> str:
    """Fix LaTeX sqrt formatting"""
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def strip_string(string: str) -> str:
    """
    Normalize string for mathematical comparison.
    
    This is the main normalization function from LIMO that handles:
    - LaTeX command normalization
    - Unit removal
    - Whitespace cleanup
    - Number format standardization
    """
    string = str(string).strip()
    
    # Linebreaks
    string = string.replace("\n", "")
    
    # Right "."
    string = string.rstrip(".")
    
    # Remove inverse spaces
    string = string.replace("\\!", "")
    
    # Matrix normalization
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    
    # Replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )
    
    # Remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")
    
    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string
    
    # Remove units
    for unit_text in UNIT_TEXTS:
        _string = re.sub(r"(^|\W)" + re.escape(unit_text) + r"($|\W)", r"\1\2", string, flags=re.IGNORECASE)
        if _string != "":
            string = _string
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    
    # Remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")
    
    # Convert word number to digit
    string = convert_word_number(string)
    
    # Replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")
    
    # Remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace("%", "")
    
    # Remove months
    months = r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b"
    string = re.sub(months, "", string, flags=re.IGNORECASE)
    
    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    
    # Handle brackets
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]
    
    # Infinity
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    
    # And
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    
    # Use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)
    
    # Quote
    string = string.replace("'", "")
    string = string.replace('"', "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")
    
    # Replace a.000b where b is not number or b is end, with ab
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)
    
    # If empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    
    # Get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    
    # Fix fractions
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    
    return string


def str_to_pmatrix(input_str: str) -> str:
    """Convert {a,b} format to pmatrix format"""
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []
    
    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)
    
    return ", ".join(pmatrix_list)


def parse_digits(num: str) -> Optional[float]:
    """Parse number from string, handling percentages"""
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num: str) -> bool:
    """Check if string represents a digit"""
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float) -> bool:
    """Check numeric equality with tolerance"""
    return isclose(reference, prediction, abs_tol=1e-4)


def choice_answer_clean(pred: str) -> str:
    """Clean choice answer format"""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    
    # Handle formats like (A), [A], A., A), **A**, A:
    # First try to extract from parentheses/brackets
    paren_match = re.match(r'^\s*[\(\[\{]?\s*([A-E])\s*[\)\]\}]?\s*$', pred.upper())
    if paren_match:
        return paren_match.group(1)
    
    # Try to find A-E in the string
    tmp = re.findall(r'\b([A-E])\b', pred.upper())
    if tmp:
        return tmp[-1]
    
    # Try to find A-E even without word boundary (for cases like "(A)")
    tmp = re.findall(r'([A-E])', pred.upper())
    if tmp:
        return tmp[-1]
    
    pred = pred.strip().strip(".")
    pred = pred.rstrip(".").rstrip("/")
    return pred


# =============================================================================
# Symbolic comparison functions
# =============================================================================

def symbolic_equal(a: str, b: str) -> bool:
    """
    Check symbolic equivalence using SymPy.
    
    Tries multiple parsing methods and comparison strategies.
    """
    import warnings
    
    if not SYMPY_AVAILABLE:
        return False
    
    def _parse(s):
        """Try multiple parsing methods"""
        # Suppress SyntaxWarnings from SymPy when parsing unusual expressions
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=SyntaxWarning)
            
            for f in [parse_latex, parse_expr]:
                if f is None:
                    continue
                try:
                    return f(s.replace("\\\\", "\\"))
                except:
                    try:
                        return f(s)
                    except:
                        pass
            
            if LATEX2SYMPY_AVAILABLE and latex2sympy is not None:
                try:
                    return latex2sympy(s.replace("\\\\", "\\"))
                except:
                    try:
                        return latex2sympy(s)
                    except:
                        pass
        return s

    a_expr = _parse(a)
    b_expr = _parse(b)

    # Direct equal
    try:
        if str(a_expr) == str(b_expr) or a_expr == b_expr:
            return True
    except:
        pass

    # Simplify equal
    try:
        if a_expr.equals(b_expr) or simplify(a_expr - b_expr) == 0:
            return True
    except:
        pass

    # Equation equal
    try:
        if (abs(a_expr.lhs - a_expr.rhs)).equals(abs(b_expr.lhs - b_expr.rhs)):
            return True
    except:
        pass

    # Numeric equal
    try:
        if numeric_equal(float(N(a_expr)), float(N(b_expr))):
            return True
    except:
        pass

    # Matrix equal
    try:
        if a_expr.shape == b_expr.shape:
            _a = a_expr.applyfunc(lambda x: round(x, 3))
            _b = b_expr.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a: str, b: str, output_queue: multiprocessing.Queue):
    """Wrapper for multiprocessing symbolic comparison"""
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout: int = 3, **kwargs):
    """Call function with timeout using multiprocessing"""
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    try:
        return output_queue.get_nowait()
    except:
        return False


# =============================================================================
# Main comparison function
# =============================================================================

def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = True,
    depth: int = 0,
    max_depth: int = 5
) -> bool:
    """
    Check mathematical equivalence between prediction and reference.
    
    This is the main comparison function from LIMO that supports:
    1. Exact string match
    2. Numerical equality with tolerance
    3. Symbolic equality via SymPy
    4. Multi-answer comparison (comma-separated, unordered)
    5. Tuple/list comparison (ordered)
    6. Equation form comparison (f(x)=1, x=5)
    7. Choice answer comparison (A, B, C, D, E)
    8. Matrix comparison
    9. Percentage tolerance (0.01 vs 1% vs 1)
    
    Args:
        prediction: Predicted answer
        reference: Ground truth answer
        include_percentage: Whether to check percentage equivalence
        is_close: Whether to use tolerance for numeric comparison
        timeout: Whether to use timeout for symbolic comparison
        depth: Current recursion depth
        max_depth: Maximum recursion depth
    
    Returns:
        True if answers are equivalent, False otherwise
    """
    if depth > max_depth:
        return False

    if prediction is None or reference is None:
        return False
    
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    
    # Exact string match (case-insensitive)
    if prediction.lower() == reference.lower():
        return True
    
    # Choice answer comparison
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True
    
    # Check for choice patterns at the beginning
    for pattern in SINGLE_CHOICE_PATTERNS:
        if regex.match(pattern, prediction):
            prediction_cleaned = regex.sub(pattern, "", prediction, count=1).strip()
            if math_equal(prediction_cleaned, reference, include_percentage, is_close, 
                         timeout=timeout, depth=depth+1, max_depth=max_depth):
                return True
    
    # Multi-answer comparison (comma-separated, unordered)
    if "," in prediction and "," in reference:
        pred_parts = [part.strip() for part in prediction.split(",")]
        ref_parts = [part.strip() for part in reference.split(",")]

        if len(pred_parts) == len(ref_parts):
            # Sort and compare
            pred_parts_sorted = sorted(pred_parts)
            ref_parts_sorted = sorted(ref_parts)
            
            if all(
                math_equal(pred_parts_sorted[i], ref_parts_sorted[i], 
                          include_percentage, is_close, timeout=timeout, 
                          depth=depth+1, max_depth=max_depth)
                for i in range(len(pred_parts_sorted))
            ):
                return True

    # Numerical equality
    try:
        if is_digit(prediction) and is_digit(reference):
            pred_val = parse_digits(prediction)
            ref_val = parse_digits(reference)
            
            if include_percentage:
                gt_result = [ref_val / 100, ref_val, ref_val * 100]
            else:
                gt_result = [ref_val]
            
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(pred_val, item):
                            return True
                    else:
                        if item == pred_val:
                            return True
                except:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # Symbolic comparison
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # pmatrix conversion
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    # Deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    
    if pred_str.lower() == ref_str.lower():
        return True

    # Tuple/list comparison (ordered): [a, b] vs [c, d]
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(
                    pred_parts[i].strip(), ref_parts[i].strip(), 
                    include_percentage, is_close, timeout=timeout, 
                    depth=depth+1, max_depth=max_depth
                )
                for i in range(len(pred_parts))
            ):
                return True

    # Matrix comparison (pmatrix/bmatrix)
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        math_equal(
                            pred_parts[i].strip(),
                            ref_parts[i].strip(),
                            include_percentage,
                            is_close,
                            timeout=timeout,
                            depth=depth+1, 
                            max_depth=max_depth
                        )
                        for i in range(len(pred_parts))
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    # Equation comparison
    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred_expr = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref_expr = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred_expr, ref_expr) or symbolic_equal(f"-({pred_expr})", ref_expr):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close, 
            timeout=timeout, depth=depth+1, max_depth=max_depth
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close, 
            timeout=timeout, depth=depth+1, max_depth=max_depth
        ):
            return True

    # Symbolic comparison with timeout
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


# =============================================================================
# Main entry points
# =============================================================================

def check_is_correct(pred: str, gt: str, timeout: bool = True) -> bool:
    """
    Main entry point for answer comparison.
    
    Normalizes both answers using strip_string and then compares using math_equal.
    
    Args:
        pred: Predicted answer (raw string)
        gt: Ground truth answer (raw string)
        timeout: Whether to use timeout for symbolic comparison
    
    Returns:
        True if answers are equivalent, False otherwise
    """
    pred = str(pred).strip() if pred else ""
    gt = str(gt).strip() if gt else ""
    
    # Special handling for choice answers - check before strip_string
    # because strip_string may incorrectly remove single letters like 'A'
    gt_upper = gt.upper()
    if gt_upper in ["A", "B", "C", "D", "E"]:
        cleaned_pred = choice_answer_clean(pred)
        if cleaned_pred.upper() == gt_upper:
            return True
    
    # Also check if prediction is a simple choice answer
    pred_upper = pred.upper()
    if pred_upper in ["A", "B", "C", "D", "E"]:
        cleaned_gt = choice_answer_clean(gt)
        if cleaned_gt.upper() == pred_upper:
            return True
    
    # Normal comparison with strip_string
    return math_equal(strip_string(pred), strip_string(gt), timeout=timeout)


def check_is_correct_simple(pred: str, gt: str, timeout: bool = True) -> bool:
    """
    Simplified comparison without full recursive support.
    
    Uses latex2sympy for quick comparison.
    """
    pred = strip_string(pred)
    gt = strip_string(gt)
    
    if not LATEX2SYMPY_AVAILABLE:
        return pred == gt
    
    flag = False
    
    try:
        pred_expr = latex2sympy(pred)
    except:
        pred_expr = pred
        flag = True
        
    try:  
        gt_expr = latex2sympy(gt)
    except:
        gt_expr = gt
        flag = True
        
    if flag:
        return pred == gt
    
    try:
        if abs(N(pred_expr) - N(gt_expr)) <= 1e-5:
            return True
    except:
        return False

    return False


# =============================================================================
# Answer extraction (from LIMO parser.py)
# =============================================================================

def find_box(pred_str: str) -> str:
    """Extract content from \\boxed{...}"""
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def extract_answer(pred_str: str, use_last_number: bool = True) -> str:
    """
    Extract answer from model output.
    
    Priority:
    1. \\boxed{...} notation
    2. Last number (fallback)
    """
    pred_str = pred_str.replace("\u043a\u0438", "")
    
    pred = ""
    
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a

    # Clean up
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    
    return pred


# =============================================================================
# Batch processing
# =============================================================================

def math_equal_process(param):
    """Wrapper for multiprocessing batch comparison"""
    return math_equal(param[-2], param[-1])


def batch_check_answers(
    predictions: List[str], 
    references: List[str], 
    timeout: bool = True
) -> List[bool]:
    """
    Check multiple answers in batch.
    
    Args:
        predictions: List of predicted answers
        references: List of ground truth answers
        timeout: Whether to use timeout for symbolic comparison
    
    Returns:
        List of boolean results
    """
    results = []
    for pred, ref in zip(predictions, references):
        try:
            result = check_is_correct(pred, ref, timeout=timeout)
        except Exception as e:
            result = False
        results.append(result)
    return results
