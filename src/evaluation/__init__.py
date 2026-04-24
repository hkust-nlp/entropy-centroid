"""
Mathematical answer evaluation module.

Implements robust extraction, normalization, and symbolic comparison
following best practices from lm-eval-harness, Qwen, DeepSeek, and LIMO.

Key features:
- LIMO-compatible math grader for robust answer comparison
- Multi-answer support (comma-separated, unordered comparison)
- Tuple/list comparison (ordered)
- Equation form answers (f(x)=1, x=5)
- Choice answers (A, B, C, D, E)
- Matrix comparison
- Percentage tolerance
"""

from .answer_extractor import extract_answer
from .answer_normalizer import normalize_answer
from .answer_comparator import compare_answers, AnswerComparator, check_math_equivalence
from .trajectory_selector import (
    TrajectorySelector,
    EntropyMeanStrategy,
    EarlyHighEntropyCentroidStrategy,
    create_trajectory_selector
)
from .answer_selector import (
    AnswerSelector,
    BestOfNAnswerSelector,
    MajorityVotingAnswerSelector,
    EntropyCentroidAnswerSelector,
    create_answer_selector
)
from .evaluator import AnswerEvaluator
from .evaluation_cache import (
    load_evaluation_cache,
    save_evaluation_cache,
    evaluate_all_trajectories,
    get_answers_from_cache,
    get_correctness_from_cache,
    CACHE_FILENAME,
)

# Import LIMO-compatible math grader functions
try:
    from .math_grader import (
        check_is_correct,
        math_equal,
        strip_string,
        extract_answer as limo_extract_answer,
        batch_check_answers,
        SYMPY_AVAILABLE,
        LATEX2SYMPY_AVAILABLE,
    )
    LIMO_GRADER_AVAILABLE = True
except ImportError:
    LIMO_GRADER_AVAILABLE = False
    check_is_correct = None
    math_equal = None
    strip_string = None
    limo_extract_answer = None
    batch_check_answers = None

__all__ = [
    # Answer extraction
    'extract_answer',
    
    # Answer normalization
    'normalize_answer',
    
    # Answer comparison
    'compare_answers',
    'AnswerComparator',
    'check_math_equivalence',
    
    # LIMO-compatible functions
    'check_is_correct',
    'math_equal',
    'strip_string',
    'limo_extract_answer',
    'batch_check_answers',
    'LIMO_GRADER_AVAILABLE',
    
    # Trajectory selection
    'TrajectorySelector',
    'EntropyMeanStrategy',
    'EarlyHighEntropyCentroidStrategy',
    'create_trajectory_selector',
    
    # Answer selection
    'AnswerSelector',
    'BestOfNAnswerSelector',
    'MajorityVotingAnswerSelector',
    'EntropyCentroidAnswerSelector',
    'create_answer_selector',
    
    # Main evaluator
    'AnswerEvaluator',
    
    # Evaluation cache
    'load_evaluation_cache',
    'save_evaluation_cache',
    'evaluate_all_trajectories',
    'get_answers_from_cache',
    'get_correctness_from_cache',
    'CACHE_FILENAME',
]
