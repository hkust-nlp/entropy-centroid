import json
import re
from typing import Any, Optional

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.exceptions import ContextWindowExceededError
from litellm import BadRequestError
from litellm.main import ModelResponse, Usage
from loguru import logger

import tau2.config as _cfg

# Static config values: read once at import time (never mutated by CLI)
from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)

# Dynamic config values: CLI mutates these AFTER module import, so we must
# read them from the config module at call time via _cfg.XXX, NOT via
# "from tau2.config import XXX" which captures the default value at import.
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.entropy_collector import EntropyCalculator, extract_entropy_from_logprobs
from tau2.utils.prompt_tool_adapter import PromptToolAdapter

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")

# Module-level entropy calculator (lazy-initialized)
_entropy_calculator: Optional[EntropyCalculator] = None


def _get_entropy_calculator() -> EntropyCalculator:
    """Get or create the module-level EntropyCalculator."""
    global _entropy_calculator
    if _entropy_calculator is None:
        _entropy_calculator = EntropyCalculator(top_k=_cfg.LOCAL_VLLM_TOP_LOGPROBS)
    return _entropy_calculator


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        # Expected for local vLLM models not in litellm's price database.
        # Silently return 0.0 to avoid flooding logs.
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    collect_entropy: bool = False,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        collect_entropy: If True and LOCAL_VLLM_ENABLED, request logprobs
            and compute per-token entropy. Result is attached to the
            returned AssistantMessage.entropy_data.
        **kwargs: Additional arguments to pass to the model.

    Returns: An AssistantMessage (or UserMessage) with the response.
    """
    if kwargs.get("num_retries") is None:
        if _cfg.LOCAL_VLLM_ENABLED:
            # Local vLLM errors are deterministic (400 ContextWindow) or fatal
            # (server crash). litellm's retry with exponential backoff just wastes
            # minutes on errors that will never self-resolve.  We handle
            # ContextWindowExceededError ourselves below with smart max_tokens
            # reduction.
            kwargs["num_retries"] = 0
        else:
            kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages)
    openai_tools = [tool.openai_schema for tool in tools] if tools else None

    # Prompt-based tool calling: inject tools into system prompt
    use_prompt_tools = _cfg.PROMPT_TOOL_CALLING and openai_tools
    if use_prompt_tools:
        litellm_messages = PromptToolAdapter.inject_tools_into_messages(
            litellm_messages, openai_tools
        )
        openai_tools = None  # Don't pass tools to API
        tool_choice = None

    if openai_tools and tool_choice is None:
        tool_choice = "auto"

    # Request logprobs for entropy collection
    if collect_entropy and _cfg.LOCAL_VLLM_ENABLED:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = _cfg.LOCAL_VLLM_TOP_LOGPROBS

    # Route to local vLLM
    actual_model = model
    if _cfg.LOCAL_VLLM_ENABLED:
        actual_model = f"hosted_vllm/{model}"
        kwargs["api_base"] = _cfg.LOCAL_VLLM_API_BASE

    # Transient errors (e.g. tokenizer "Already borrowed" race condition under
    # high concurrency) deserve a few retries before giving up.  These are
    # distinct from context-window errors (deterministic, never self-resolve)
    # and from server crashes (fatal).
    _TRANSIENT_MAX_RETRIES = 3

    def _do_completion():
        return completion(
            model=actual_model,
            messages=litellm_messages,
            tools=openai_tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    try:
        response = _do_completion()
    except (ContextWindowExceededError, BadRequestError) as e:
        # litellm's hosted_vllm provider raises BadRequestError (not the more
        # specific ContextWindowExceededError) for context-window errors.
        # Check message content to confirm this is actually a context window
        # error before handling; re-raise immediately if it's something else.
        error_msg = str(e)
        if 'context length' not in error_msg and 'input tokens' not in error_msg:
            # "Already borrowed" is a transient tokenizer race — retry
            if 'Already borrowed' in error_msg:
                for attempt in range(1, _TRANSIENT_MAX_RETRIES + 1):
                    logger.warning(
                        f"Transient tokenizer error (attempt {attempt}/"
                        f"{_TRANSIENT_MAX_RETRIES}): {error_msg}"
                    )
                    try:
                        response = _do_completion()
                        break  # success
                    except (ContextWindowExceededError, BadRequestError) as retry_e:
                        if attempt == _TRANSIENT_MAX_RETRIES:
                            logger.error(
                                f"Transient error persisted after "
                                f"{_TRANSIENT_MAX_RETRIES} retries"
                            )
                            raise retry_e
                else:
                    # all retries exhausted (shouldn't reach here due to raise above)
                    raise
            else:
                logger.error(error_msg)
                raise

        # Context window exceeded — retrying with the same parameters will
        # always fail.  Parse the error to find out how many input tokens the
        # request has, then reduce max_tokens so that input + output fits.
        input_match = re.search(r'(?:has|passed)\s+(\d+)\s+input tokens', error_msg)
        context_match = re.search(r'(?:maximum\s+)?context length is (?:only\s+)?(\d+)', error_msg)

        if input_match and context_match:
            input_tokens = int(input_match.group(1))
            max_context = int(context_match.group(1))
            # Leave a small margin (64 tokens) for safety
            available = max_context - input_tokens - 64
            original_max = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")

            if available >= 256:
                logger.warning(
                    f"ContextWindow exceeded (input={input_tokens}, "
                    f"max_context={max_context}, requested_max_tokens={original_max}). "
                    f"Retrying with max_tokens={available}"
                )
                retry_kwargs = dict(kwargs)
                retry_kwargs["max_tokens"] = available
                retry_kwargs.pop("max_completion_tokens", None)
                # Disable litellm internal retries — this is our single retry
                retry_kwargs["num_retries"] = 0
                try:
                    response = completion(
                        model=actual_model,
                        messages=litellm_messages,
                        tools=openai_tools,
                        tool_choice=tool_choice,
                        **retry_kwargs,
                    )
                except Exception as retry_err:
                    logger.error(f"Retry with reduced max_tokens also failed: {retry_err}")
                    raise retry_err
            else:
                logger.error(
                    f"ContextWindow exceeded and no room left "
                    f"(input={input_tokens}, max_context={max_context}, "
                    f"available={available}). Cannot recover."
                )
                raise
        else:
            # Could not parse token counts from error — re-raise as-is
            logger.error(f"ContextWindowExceededError (unparseable): {e}")
            raise
    except Exception as e:
        logger.error(e)
        raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response_choice = response.choices[0]
    try:
        finish_reason = response_choice.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response_choice.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response_choice.message.content
    native_tool_calls = response_choice.message.tool_calls or []

    # Guard against empty responses.
    # This happens when a reasoning model (QwQ, Qwen3) hits the token limit
    # while still inside a <think>...</think> block. The reasoning parser
    # strips the incomplete thinking content, leaving content=None and no
    # tool_calls. Provide a fallback so the simulation can continue.
    if not content and not native_tool_calls:
        if finish_reason == "length":
            logger.warning(
                "Empty response after reasoning parser stripped incomplete "
                "<think> block (token limit reached). Using fallback content."
            )
            content = "I apologize, but I need a moment to process this. Could you please repeat your request?"
        else:
            logger.warning(
                f"Empty response (no content, no tool_calls, finish_reason={finish_reason}). "
                "Using fallback content."
            )
            content = "I'm sorry, I didn't generate a proper response. Could you please try again?"
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in native_tool_calls
    ]
    tool_calls = tool_calls or None

    # Parse tool calls from text (prompt-based mode)
    if use_prompt_tools and not tool_calls:
        parsed_content, parsed_tool_calls = PromptToolAdapter.parse_tool_calls_from_text(content)
        if parsed_tool_calls is not None:
            tool_calls = parsed_tool_calls
            content = None  # Tool call, no text content
        elif parsed_content is not None:
            content = parsed_content
            tool_calls = None

    # Extract logprobs and compute entropy
    entropy_data = None
    if collect_entropy and _cfg.LOCAL_VLLM_ENABLED:
        logprobs_obj = getattr(response_choice, "logprobs", None)
        if logprobs_obj is not None:
            logprobs_content = None
            if isinstance(logprobs_obj, dict):
                logprobs_content = logprobs_obj.get("content")
            elif hasattr(logprobs_obj, "content"):
                # litellm wraps logprobs in an object
                raw_content = logprobs_obj.content
                if raw_content:
                    logprobs_content = []
                    for item in raw_content:
                        if isinstance(item, dict):
                            logprobs_content.append(item)
                        else:
                            # Convert litellm LogprobsContent objects to dicts
                            entry = {
                                "token": getattr(item, "token", ""),
                                "logprob": getattr(item, "logprob", 0.0),
                                "top_logprobs": [],
                            }
                            top_lps = getattr(item, "top_logprobs", [])
                            if top_lps:
                                for tlp in top_lps:
                                    if isinstance(tlp, dict):
                                        entry["top_logprobs"].append(tlp)
                                    else:
                                        entry["top_logprobs"].append({
                                            "token": getattr(tlp, "token", ""),
                                            "logprob": getattr(tlp, "logprob", 0.0),
                                        })
                            logprobs_content.append(entry)

            if logprobs_content:
                entropy_data = extract_entropy_from_logprobs(
                    logprobs_content, _get_entropy_calculator()
                )

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response_choice.to_dict() if hasattr(response_choice, "to_dict") else None,
        entropy_data=entropy_data,
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
