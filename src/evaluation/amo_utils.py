import os
import copy
import signal

from math_verify import parse, verify
from sympy import solve

from openai import OpenAI

# API configuration - can be set via environment variables or directly
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("AMO_BENCH_API_KEY") or "API_KEY_HERE"
BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("AMO_BENCH_BASE_URL") or None
OPENROUTER_PROVIDER = os.environ.get("OPENROUTER_PROVIDER") or os.environ.get("AMO_BENCH_PROVIDER")
OPENROUTER_PROVIDER_ORDER = os.environ.get("OPENROUTER_PROVIDER_ORDER")
OPENROUTER_PROVIDER_ONLY = os.environ.get("OPENROUTER_PROVIDER_ONLY")
OPENROUTER_PROVIDER_IGNORE = os.environ.get("OPENROUTER_PROVIDER_IGNORE")
OPENROUTER_PROVIDER_SORT = os.environ.get("OPENROUTER_PROVIDER_SORT")
OPENROUTER_PROVIDER_REQUIRE_PARAMETERS = os.environ.get("OPENROUTER_PROVIDER_REQUIRE_PARAMETERS")
OPENROUTER_PROVIDER_ALLOW_FALLBACKS = os.environ.get("OPENROUTER_PROVIDER_ALLOW_FALLBACKS")

# OpenRouter specific settings
USE_OPENROUTER = os.environ.get("USE_OPENROUTER", "false").lower() == "true" or "openrouter" in (BASE_URL or "").lower()
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "https://github.com")
OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", "AMO-Bench-Evaluator")

# Model configuration for description type evaluation
LLM_MODEL = os.environ.get("AMO_BENCH_MODEL") or os.environ.get("LLM_MODEL") or "o4-mini"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


DEBUG_API = _env_flag("AMO_BENCH_DEBUG_API", default=False)
DEBUG_EVAL = _env_flag("AMO_BENCH_DEBUG_EVAL", default=False)
AMO_VARIABLE_SOLVE_TIMEOUT = _env_int("AMO_VARIABLE_SOLVE_TIMEOUT", 30)
AMO_VARIABLE_MAX_TRY = _env_int("AMO_VARIABLE_MAX_TRY", 0)  # 0 means use full try_list

# Default model mapping for OpenRouter (with official provider specified)
OPENROUTER_MODEL_MAP = {
    "o4-mini": "openai/o4-mini",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "gpt-4": "openai/gpt-4",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
}


def set_api_config(api_key: str = None, base_url: str = None, use_openrouter: bool = None):
    """Set API configuration for description type evaluation."""
    global API_KEY, BASE_URL, USE_OPENROUTER, OPENROUTER_PROVIDER
    if api_key:
        API_KEY = api_key
    if base_url:
        BASE_URL = base_url
    if use_openrouter is not None:
        USE_OPENROUTER = use_openrouter
    elif base_url and "openrouter" in base_url.lower():
        USE_OPENROUTER = True

SCORE_PROMPT = """
For the following math problem, we have the reference answer and the student's answer.
Determine whether the student's answer is equivalent to the reference answer.
If equivalent, output "Correct".
If not equivalent, output "Incorrect".

### Problem
QUESTION

### Reference Answer
GOLD

### Student Answer
PRED

Now, please provide your judgment.
Please strictly follow the format below to summarize your conclusion at the end of your judgment:
### Conclusion: Correct/Incorrect
If the answer involves a decimal approximation, it must be accurate to at least four decimal places.
""".strip()

answer_prefix_list = [
    "### the final answer is:", "### the final answer:", "### final answer is:", "### final answer:",
    "### the final answer is", "### the final answer", "### final answer is", "### final answer",
]
answer_prefix_list_wo_hashtag = [p[4:] for p in answer_prefix_list]

think_postfix_list = [
    "</think>",
    "</longcat_think>",
]

cut_list = [
    "\\medskip", "\n---"
]

remove_list = [
    "\\bigl", "\\bigr", 
    "\\Bigl", "\\Bigr",
    "\\biggl", "\\biggr",
    "\\Biggl", "\\Biggr",
    "\\bigg", "\\Bigg", "\\big", "\\Big",
    "\\left", "\\right",
]
replace_list = [
    ["‘", "'"],
    ["’", "'"],
    ["“", '"'],
    ["”", '"'],
    ["（", "("],
    ["）", ")"],
    ["，", ", "],
    ["：", ": "],
    ["；", "; "],
    ["。", ". "],
    ["！", "! "],
    ["？", "? "],
    ["…", "..."],
    ["–", "-"],
    ["−", "-"],
]


class AMOSolveTimeoutError(TimeoutError):
    """Raised when symbolic solve exceeds configured timeout."""


def _run_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """Run a function with Unix signal-based timeout."""
    if timeout_seconds is None or timeout_seconds <= 0:
        return func(*args, **kwargs)

    def _handler(signum, frame):
        raise AMOSolveTimeoutError(f"solve timed out after {timeout_seconds}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout_seconds))
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def solve_with_timeout(exp):
    return _run_with_timeout(solve, AMO_VARIABLE_SOLVE_TIMEOUT, exp)

def pred_cut(pred_extract):
    for pat in cut_list:
        pred_extract = pred_extract.split(pat)[0].strip()
    return pred_extract

def pred_extractor(pred, answer_type):

    pred_extract = pred.replace('：', ': ')

    for think_postfix in think_postfix_list:
        pred_extract = pred_extract.split(think_postfix)[-1].strip()

    for prefix in answer_prefix_list + answer_prefix_list_wo_hashtag:
        if prefix in pred_extract.lower():
            pred_extract_lower = pred_extract.lower().split(prefix)[-1]
            pred_extract = pred_extract[-len(pred_extract_lower):]
            pred_extract = pred_extract.strip()
            break
    
    if answer_type != "description":
        for pat in remove_list:
            pred_extract = pred_extract.replace(pat, "")
    
    for pat, new_pat in replace_list:
        pred_extract = pred_extract.replace(pat, new_pat)

    while " }" in pred_extract:
        pred_extract = pred_extract.replace(" }", "}")
    while ".}" in pred_extract:
        pred_extract = pred_extract.replace(".}", "}")
    
    if answer_type in ["number", "variable", "set"]:
        pred_extract = pred_extract.replace("\\,", "")
        pred_extract = pred_extract.replace("\\;", "")
        pred_extract = pred_extract.replace("\\,", "")
        pred_extract = pred_extract.replace("\\;", "")
        pred_extract = pred_extract.replace("\n", " ")
    
    if answer_type in ["number", "variable"]:
        pred_extract = pred_extract.replace(",", "")
        pred_extract = pred_extract.replace("\\{", "(").replace("\\}", ")").replace("\\[", "(").replace("\\]", ")")

    return pred_extract.strip()

def call_api(
    prompt,
    model="o4-mini",
    _retried_model: bool = False,
    _retried_provider: bool = False
):
    """
    Call OpenAI API or OpenRouter for LLM-based scoring.
    
    When using OpenRouter, the model is mapped to include the official provider
    (e.g., "o4-mini" -> "openai/o4-mini") to ensure consistent routing to the
    official provider instead of random third-party providers.
    """
    client_kwargs = {"api_key": API_KEY}
    
    # Determine if using OpenRouter
    use_openrouter = USE_OPENROUTER or (BASE_URL and "openrouter" in BASE_URL.lower())
    
    if use_openrouter:
        # Use OpenRouter base URL
        client_kwargs["base_url"] = BASE_URL or "https://openrouter.ai/api/v1"
        
        # Map model to OpenRouter format with official provider
        if model in OPENROUTER_MODEL_MAP:
            model = OPENROUTER_MODEL_MAP[model]
        elif "/" not in model:
            # If provider is pinned, keep model as-is (e.g., "kimi-k2.5")
            # Otherwise default to openai/ prefix for OpenAI models
            if not OPENROUTER_PROVIDER:
                model = f"openai/{model}"
    elif BASE_URL:
        client_kwargs["base_url"] = BASE_URL
    
    client = OpenAI(**client_kwargs)
    
    try:
        # Build request kwargs
        request_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 8192,
            "temperature": 1.0,
        }
        
        # Add OpenRouter specific headers
        if use_openrouter:
            request_kwargs["extra_headers"] = {
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": OPENROUTER_SITE_NAME,
            }
            provider_prefs = {}
            # Priority order: explicit order env var, then single provider env var
            if OPENROUTER_PROVIDER_ORDER:
                provider_prefs["order"] = [p.strip() for p in OPENROUTER_PROVIDER_ORDER.split(",") if p.strip()]
            elif OPENROUTER_PROVIDER:
                provider_prefs["order"] = [OPENROUTER_PROVIDER]

            if OPENROUTER_PROVIDER_ONLY:
                provider_prefs["only"] = [p.strip() for p in OPENROUTER_PROVIDER_ONLY.split(",") if p.strip()]
            if OPENROUTER_PROVIDER_IGNORE:
                provider_prefs["ignore"] = [p.strip() for p in OPENROUTER_PROVIDER_IGNORE.split(",") if p.strip()]
            if OPENROUTER_PROVIDER_SORT:
                provider_prefs["sort"] = OPENROUTER_PROVIDER_SORT.strip()
            if OPENROUTER_PROVIDER_REQUIRE_PARAMETERS is not None:
                provider_prefs["require_parameters"] = (
                    str(OPENROUTER_PROVIDER_REQUIRE_PARAMETERS).lower() == "true"
                )
            if OPENROUTER_PROVIDER_ALLOW_FALLBACKS is not None:
                provider_prefs["allow_fallbacks"] = (
                    str(OPENROUTER_PROVIDER_ALLOW_FALLBACKS).lower() == "true"
                )

            if provider_prefs:
                request_kwargs["extra_body"] = {"provider": provider_prefs}
            # For OpenAI models routed via OpenRouter, pin to official provider
            elif model.startswith("openai/"):
                request_kwargs["extra_body"] = {
                    "provider": {
                        "order": ["OpenAI"],
                        "allow_fallbacks": False,
                    }
                }
        
        if DEBUG_API:
            provider_payload = request_kwargs.get("extra_body", {}).get("provider")
            print(
                f"[AMO API] request model={request_kwargs['model']} "
                f"use_openrouter={use_openrouter} "
                f"prompt_chars={len(prompt)} "
                f"provider={provider_payload}"
            )

        # Try with reasoning_effort for o-series models (OpenAI specific)
        if not use_openrouter and model.startswith("o"):
            try:
                request_kwargs["reasoning_effort"] = "low"
                response = client.chat.completions.create(**request_kwargs)
                return response
            except (TypeError, Exception):
                del request_kwargs["reasoning_effort"]
        
        response = client.chat.completions.create(**request_kwargs)
        if DEBUG_API:
            try:
                content = response.choices[0].message.content
                preview = content[:180].replace("\n", " ") if content else ""
                print(f"[AMO API] response ok chars={len(content) if content else 0} preview={preview}")
            except Exception:
                print("[AMO API] response ok (unable to parse preview)")
        
    except Exception as e:
        error_msg = str(e)
        # If OpenRouter returns "No endpoints found", retry without provider pin first
        if (
            use_openrouter
            and (OPENROUTER_PROVIDER or OPENROUTER_PROVIDER_ORDER or OPENROUTER_PROVIDER_ONLY or OPENROUTER_PROVIDER_IGNORE)
            and not _retried_provider
            and "No endpoints found" in error_msg
        ):
            print("OpenRouter: no endpoints for provider preferences. Retrying without provider pin...")
            original_provider = OPENROUTER_PROVIDER
            try:
                # Temporarily clear provider and retry
                globals()["OPENROUTER_PROVIDER"] = None
                globals()["OPENROUTER_PROVIDER_ORDER"] = None
                globals()["OPENROUTER_PROVIDER_ONLY"] = None
                globals()["OPENROUTER_PROVIDER_IGNORE"] = None
                globals()["OPENROUTER_PROVIDER_SORT"] = None
                globals()["OPENROUTER_PROVIDER_REQUIRE_PARAMETERS"] = None
                globals()["OPENROUTER_PROVIDER_ALLOW_FALLBACKS"] = None
                return call_api(prompt, model=model, _retried_provider=True)
            finally:
                globals()["OPENROUTER_PROVIDER"] = original_provider

        # If OpenRouter returns "No endpoints found", try stripping provider prefix once
        if (
            use_openrouter
            and not _retried_model
            and "/" in model
            and "No endpoints found" in error_msg
        ):
            model_suffix = model.split("/", 1)[1]
            print(f"OpenRouter: no endpoints for {model}. Retrying with {model_suffix}...")
            return call_api(prompt, model=model_suffix, _retried_model=True)

        print(f"Error in API response: {e}")
        if DEBUG_API:
            print(f"[AMO API] response failed model={model} use_openrouter={use_openrouter}")
        response = None

    return response

def verify_description_answer(pred_extract, info, model=None):
    """Verify description type answers using LLM-based scoring with majority voting."""
    # Use global LLM_MODEL if not specified
    if model is None:
        model = LLM_MODEL
    
    # Check if API key is configured
    if API_KEY == "API_KEY_HERE" or not API_KEY:
        print("Warning: No API key configured for description type evaluation. Skipping.")
        return False
    
    llm_score_prompt = SCORE_PROMPT

    assert "GOLD" in llm_score_prompt
    llm_score_prompt = llm_score_prompt.replace("GOLD", info["answer"])

    assert "PRED" in llm_score_prompt
    llm_score_prompt = llm_score_prompt.replace("PRED", pred_extract)

    assert "QUESTION" in llm_score_prompt
    llm_score_prompt = llm_score_prompt.replace("QUESTION", info["prompt"])

    if len(llm_score_prompt) >= 40000:
        print("Prompt is too long, truncate it.")
        llm_score_prompt = llm_score_prompt[:10000] + "\n\n...\n\n" + llm_score_prompt[-10000:]

    llm_judge_vote_num = 2
    if DEBUG_EVAL:
        qid = info.get("question_id", "unknown")
        print(
            f"[AMO EVAL] description question_id={qid} "
            f"model={model} prompt_chars={len(llm_score_prompt)} votes={llm_judge_vote_num}"
        )
    llm_judge_vote_list = []
    for vote_round in range(llm_judge_vote_num):
        if DEBUG_EVAL:
            print(f"[AMO EVAL] description vote_round={vote_round + 1}/{llm_judge_vote_num}")
        response = call_api(llm_score_prompt, model=model)
        
        if response is None:
            llm_judge_vote_list.append(False)
            continue
            
        try:
            response_content = response.choices[0].message.content
            conclusion = response_content.lower().split("conclusion:")[-1]
            if "correct" in conclusion.split() and "not correct" not in conclusion and "n't correct" not in conclusion:
                llm_judge_vote = True
            else:
                llm_judge_vote = False
            if DEBUG_EVAL:
                print(f"[AMO EVAL] description vote={llm_judge_vote} conclusion_tail={conclusion[:80]}")
        except (AttributeError, IndexError) as e:
            print(f"Error parsing response: {e}")
            llm_judge_vote = False

        llm_judge_vote_list.append(llm_judge_vote)

    assert len(llm_judge_vote_list) == llm_judge_vote_num
    llm_judge = True if llm_judge_vote_list.count(True) > llm_judge_vote_list.count(False) else False

    return llm_judge

def verify_number_set_answer(pred_extract, info):
    pred_parse = parse(pred_extract)
    gold_parse = parse(info["answer"])
    verify_result = verify(gold_parse, pred_parse, float_rounding=4) or verify(pred_parse, gold_parse, float_rounding=4)

    if pred_parse and '=' in pred_parse[-1]:
        pred_last_str = pred_parse[-1].split('=')[-1]
        pred_last_str_parse = parse("\\boxed{" + pred_last_str + "}")
        verify_last_result = verify(gold_parse, pred_last_str_parse, float_rounding=4) or verify(pred_last_str_parse, gold_parse, float_rounding=4)
        verify_result = verify_result or verify_last_result

    return verify_result

def verify_variable_answer(pred_extract, info):
    assert "try_list" in info
            
    pred_parse_ori = parse(pred_extract)
    if not pred_parse_ori:
        print("Cannot parse the prediction: {}".format(pred_extract))
        info['verify_result'] = False
        return False
    
    pred_parse_str = pred_parse_ori[-1]
    pred_parse_str = pred_parse_str.split("\\qquad")[-2].strip() if "\\qquad" in pred_parse_str else pred_parse_str
    pred_parse_str = pred_parse_str.split("\\quad")[-2].strip() if "\\quad" in pred_parse_str else pred_parse_str
    pred_parse_str = pred_parse_str.split("=")[-1]

    gold_parse_ori = parse(info["answer"])
    gold_parse_str = gold_parse_ori[-1]
    gold_parse_str = gold_parse_str.split("=")[-1]

    assert len(info["try_list"]) >= 1
    try_list = info["try_list"]
    if AMO_VARIABLE_MAX_TRY > 0:
        try_list = try_list[:AMO_VARIABLE_MAX_TRY]
        if DEBUG_EVAL:
            print(
                f"[AMO EVAL] variable limited try_list: "
                f"{len(try_list)}/{len(info['try_list'])}"
            )

    verify_result = True
    for try_str in try_list:
        pred_parse_equ = parse("\\boxed{" + try_str + ", y=" + pred_parse_str + "}")
        gold_parse_equ = parse("\\boxed{" + try_str + ", y=" + gold_parse_str + "}")
        if not pred_parse_equ or not gold_parse_equ:
            verify_result = False
            break

        try:
            pred_parse_solve = solve_with_timeout(pred_parse_equ[0])
        except Exception as e:
            print("Error in solving the prediction: {}".format(pred_parse_equ[0]))
            print(e)
            verify_result = False
            break
        try:
            gold_parse_solve = solve_with_timeout(gold_parse_equ[0])
        except Exception as e:
            print("Error in solving the gold: {}".format(gold_parse_equ[0]))
            print(e)
            verify_result = False
            break

        if not gold_parse_solve:
            verify_result = False
            break

        if not pred_parse_solve:
            verify_result = False
            break

        if isinstance(pred_parse_solve, list):
            pred_parse_solve = pred_parse_solve[0]
        if isinstance(gold_parse_solve, list):
            gold_parse_solve = gold_parse_solve[0]

        pred_parse_solve_y = None
        gold_parse_solve_y = None

        try:
            for s in pred_parse_solve:
                if str(s) == 'y':
                    pred_parse_solve_y = pred_parse_solve[s]
        except Exception as e:
            print("Error in parsing the prediction solution: {}".format(pred_extract))
            print(e)
            verify_result = False
            break
            
        for s in gold_parse_solve:
            if str(s) == 'y':
                gold_parse_solve_y = gold_parse_solve[s]

        if gold_parse_solve_y is None or pred_parse_solve_y is None:
            verify_result = False
            break

        pred_parse_solve_y = pred_parse_solve_y.evalf()
        gold_parse_solve_y = gold_parse_solve_y.evalf()

        verify_result = verify(gold_parse_solve_y, pred_parse_solve_y, float_rounding=8) or verify(pred_parse_solve_y, gold_parse_solve_y, float_rounding=8)

        if not verify_result:
            break

    return verify_result

def append_try_list(ori_info):
    info = copy.deepcopy(ori_info)
    assert "question_id" in info
    question_id = info["question_id"]
    if question_id == 5:
        assert info["answer_type"] == "variable"
        try_list = ["n=1", "n=2", "n=3", "n=4", "n=5", "n=6", "n=7", "n=8", "n=9", "n=10",
                    "n=11", "n=12", "n=13", "n=14", "n=15", "n=16", "n=17", "n=18", "n=19", "n=20"]
        info["try_list"] = try_list
    elif question_id == 37:
        assert info["answer_type"] == "variable"
        try_list = ["a=2,b=3,c=4", "a=3,b=4,c=5", "a=4,b=5,c=6", "a=5,b=6,c=7", "a=6,b=7,c=8",
                    "a=7,b=8,c=9", "a=8,b=9,c=10", "a=9,b=10,c=11", "a=10,b=11,c=12",
                    "a=11,b=12,c=13", "a=12,b=13,c=14", "a=13,b=14,c=15", "a=14,b=15,c=16",
                    "a=15,b=16,c=17", "a=16,b=17,c=18", "a=17,b=18,c=19", "a=18,b=19,c=20"]
        info["try_list"] = try_list
    
    return info
