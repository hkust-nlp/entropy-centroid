"""
vLLM inference engine with multi-GPU support.
"""

import re
import gc
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams
from tqdm import tqdm

from .entropy_calculator import EntropyCalculator


def detect_repetition(text: str, min_repeat_length: int = 50) -> int:
    """
    Detect repetitive patterns in generated text using suffix array approach.

    This function identifies where the text starts repeating itself, which is a
    common failure mode in LLM generation (e.g., repeating "\boxed{42}" hundreds of times).

    Strategy:
    1. Use sliding window to find repeated substrings
    2. Verify the repetition is consecutive and substantial
    3. Return the position where the first instance of the repeated pattern starts

    Args:
        text: Generated text to analyze
        min_repeat_length: Minimum length of pattern to consider as repetition

    Returns:
        Position where repetition begins, or -1 if no repetition detected
    """
    if len(text) < min_repeat_length * 2:
        return -1

    # Start checking from the end where repetitions typically occur
    # Work backwards to find the earliest repetition point
    text_len = len(text)

    # Try different pattern lengths, starting from longer patterns
    for pattern_len in range(min_repeat_length, min(500, text_len // 3), 10):
        # Check the last portion of text for repetition
        check_start = max(0, text_len - pattern_len * 10)

        for i in range(check_start, text_len - pattern_len * 2):
            pattern = text[i:i + pattern_len]

            # Count how many times this pattern repeats consecutively
            repeat_count = 0
            check_pos = i + pattern_len

            while check_pos + pattern_len <= text_len:
                next_chunk = text[check_pos:check_pos + pattern_len]

                # Check for exact or near-exact match (allowing minor tokenization differences)
                if next_chunk == pattern:
                    repeat_count += 1
                    check_pos += pattern_len
                elif len(next_chunk) == len(pattern):
                    # Calculate similarity (allow up to 5% difference)
                    matches = sum(c1 == c2 for c1, c2 in zip(pattern, next_chunk))
                    if matches / len(pattern) >= 0.9999:
                        repeat_count += 1
                        check_pos += pattern_len
                    else:
                        break
                else:
                    break

            # If pattern repeats at least twice (3 total occurrences), it's a repetition
            if repeat_count >= 2:
                return i

    return -1


def extract_boxed_answer(text: str) -> tuple[str, int, int]:
    """
    Extract content from \boxed{...} with proper brace matching.

    Note: This function is for ANALYSIS ONLY, not for truncation.
    The model's output should remain complete regardless of where \boxed{} appears.

    Returns:
        tuple: (boxed_content, start_pos, end_pos) or (None, -1, -1) if not found
    """
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))

    if not matches:
        return None, -1, -1

    # Find the last \boxed{
    last_match = matches[-1]
    start = last_match.end() - 1  # Position of the opening brace

    # Count braces to find matching closing brace
    brace_count = 1
    i = start + 1

    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        # Found complete \boxed{...}
        content = text[start+1:i-1]
        return content, last_match.start(), i
    else:
        # Incomplete \boxed{
        return None, -1, -1


def post_process_math_answer(text: str) -> str:
    """
    Post-process generated text with minimal intervention.

    NEW STRATEGY (Correct Approach):
    - vLLM handles stop_tokens automatically (defined in config)
    - Model generates until hitting stop_tokens or max_tokens
    - We ONLY remove detected repetitions, nothing else
    - DO NOT truncate based on answer markers like \boxed{}

    This preserves the model's complete reasoning, including multiple \boxed{}
    attempts, self-corrections, and verification steps.

    Args:
        text: Raw generated text from vLLM

    Returns:
        Processed text (only modified if repetition detected)
    """
    # Only intervention: detect and remove repetitive generation
    # repetition_start = detect_repetition(text)

    # if repetition_start > 0:
    #     # Truncate at the point where repetition begins
    #     text = text[:repetition_start]

    # Clean up trailing whitespace
    return text.rstrip()


class VLLMEngine:
    """
    vLLM inference engine with entropy calculation support.
    """

    # System prompts for different task types
    SYSTEM_PROMPTS = {
        "math": "You are a math expert. Please solve problems step by step and put your final answer within \\boxed{}.",
        "logic": "You are a logic reasoning expert. Please solve the problem step by step and provide your final answer.",
        "korbench": "You are a logic reasoning expert. Follow the given rules carefully and solve the problem. Format your answer within double brackets like [[answer]].",
        "synlogic": "You are a logic puzzle solving expert. Analyze the problem carefully, think step by step, and provide your final answer within <answer> and </answer> tags.",
        "bigcodebench": "You are an expert Python programmer. Write clean, efficient, and correct Python code to solve the given task. Include all necessary imports and ensure the code is complete and executable.",
        "livecodebench": "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. Wrap your final solution in ```python ``` code blocks.",
        "code": "You are an expert Python programmer. Write clean, efficient, and correct Python code to solve the given task. Include all necessary imports and ensure the code is complete and executable.",
    }

    # List of known multimodal models that support text-only mode
    # These models have vision encoders but can be used for text-only inference
    MULTIMODAL_TEXT_CAPABLE_MODELS = [
        "ministral",  # Mistral's Ministral-3 series
        "pixtral",    # Mistral's Pixtral series
        "llava",      # LLaVA models
        "internvl",   # InternVL models
        "qwen-vl",    # Qwen-VL models
        "cogvlm",     # CogVLM models
        "qwen3.5",    # Qwen3.5 is natively multimodal (VL)
    ]

    # Multimodal models that should use --language-model-only for text inference.
    # This skips loading the vision encoder entirely, freeing GPU memory for KV cache.
    # Ref: https://docs.vllm.ai/en/latest/models/supported_models.html
    LANGUAGE_MODEL_ONLY_MODELS = [
        "qwen3.5",    # Official vLLM docs recommend --language-model-only for text-only Qwen3.5
    ]
    
    # Models that require Mistral-specific loading parameters
    # Official vLLM command: --tokenizer_mode mistral --config_format mistral --load_format mistral
    MISTRAL_NATIVE_MODELS = [
        "ministral",
        "pixtral",
        "mistral-large",
    ]

    # Models that are incompatible with vLLM's prefix caching when using n>1.
    # GPT-OSS: prefix caching + n>1 causes degenerate output ("!!!..." + NaN logprobs).
    PREFIX_CACHE_INCOMPATIBLE_MODELS = [
        "gpt-oss",
    ]

    # Models susceptible to token_id=0 degeneration under KV cache pressure.
    # When many sequences are generated concurrently (batch_size * n), vLLM's
    # scheduler may preempt/swap sequences.  On these models the restored state
    # can be corrupted, causing the model to emit token_id=0 repeatedly with
    # entropy=0.  Empirical safe threshold: <=256 concurrent sequences.
    #
    # Qwen3.5: degeneration rate vs concurrent sequences (n=64, max_tokens=32768):
    #   64 seqs  → 3%,  320 → 11%,  640 → 71%,  1920 → 100%
    DEGENERATION_SENSITIVE_MODELS = [
        "qwen3.5",
    ]
    DEGENERATION_MAX_CONCURRENT_SEQS = 256
    DEGENERATION_CONSECUTIVE_THRESHOLD = 10   # consecutive token_id=0 to flag
    DEGENERATION_MAX_RETRIES = 3

    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        entropy_calculator: Optional[EntropyCalculator] = None,
        top_k_logprobs: int = 20,
        task_type: str = "math",
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        use_mistral_format: Optional[bool] = None,
        enable_prefix_caching: Optional[bool] = None,
    ):
        """
        Initialize vLLM engine.

        Args:
            model_name_or_path: Model path or HuggingFace model ID
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum model sequence length
            trust_remote_code: Whether to trust remote code
            dtype: Model data type (auto, float16, bfloat16)
            entropy_calculator: EntropyCalculator instance
            top_k_logprobs: Number of top-k logprobs to return
            task_type: Task type (math, logic, korbench, synlogic)
            limit_mm_per_prompt: Limit for multimodal inputs per prompt (e.g., {"image": 0} for text-only)
                                 Set to {"image": 0} for multimodal models used in text-only mode
            use_mistral_format: Whether to use Mistral-native loading format (auto-detected if None)
            enable_prefix_caching: Whether to enable prefix caching (None=auto-detect based on model)
        """
        self.model_name_or_path = model_name_or_path
        self.tensor_parallel_size = tensor_parallel_size
        self.entropy_calculator = entropy_calculator
        self.top_k_logprobs = top_k_logprobs
        self.task_type = task_type

        print(f"Initializing vLLM engine with model: {model_name_or_path}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        print(f"GPU memory utilization: {gpu_memory_utilization}")

        # Check if model is a known multimodal model
        is_multimodal = self._is_multimodal_model(model_name_or_path)
        
        # Check if model needs Mistral-native format
        needs_mistral_format = self._needs_mistral_format(model_name_or_path)
        if use_mistral_format is not None:
            needs_mistral_format = use_mistral_format
        
        # Auto-detect prefix caching compatibility
        if enable_prefix_caching is None:
            if self._is_prefix_cache_incompatible(model_name_or_path):
                enable_prefix_caching = False
                print(f"WARNING: Auto-disabling prefix_caching for {model_name_or_path} "
                      f"(known to cause degenerate output with n>1)")

        # Prepare vLLM initialization kwargs
        llm_kwargs = {
            "model": model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
        }
        if enable_prefix_caching is not None:
            llm_kwargs["enable_prefix_caching"] = enable_prefix_caching
            print(f"Prefix caching: {enable_prefix_caching}")
        
        # For Mistral-native models (Ministral-3, Pixtral, etc.), use official recommended settings
        # Reference: vllm serve mistralai/Ministral-3-14B-Instruct-2512 \
        #   --tokenizer_mode mistral --config_format mistral --load_format mistral
        if needs_mistral_format:
            print(f"Detected Mistral-native model: {model_name_or_path}")
            print("Using Mistral-specific loading format (tokenizer_mode=mistral, config_format=mistral, load_format=mistral)")
            llm_kwargs.update({
                "tokenizer_mode": "mistral",
                "config_format": "mistral",
                "load_format": "mistral",
            })
        
        # Handle multimodal models in text-only mode
        if is_multimodal:
            print(f"Detected multimodal model: {model_name_or_path}")

            # For models that support --language-model-only (e.g., Qwen3.5),
            # skip loading the vision encoder entirely for text-only inference.
            if self._needs_language_model_only(model_name_or_path):
                print("Using --language-model-only mode (skipping vision encoder)")
                llm_kwargs["language_model_only"] = True
            elif limit_mm_per_prompt is None:
                # Fallback: keep vision encoder but disable image inputs
                print("Auto-configuring for text-only mode (limit_mm_per_prompt={'image': 0})")
                limit_mm_per_prompt = {"image": 0}

            if limit_mm_per_prompt is not None:
                llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt
                print(f"Multimodal limit per prompt: {limit_mm_per_prompt}")
        elif limit_mm_per_prompt is not None:
            # User specified limit but model may not be multimodal
            print(f"Note: limit_mm_per_prompt specified but model may not be multimodal")
            llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

        # Initialize vLLM with appropriate configuration
        # Try with all kwargs first, then fall back if some are not supported
        self.llm = self._init_llm_with_fallback(llm_kwargs)

        print("vLLM engine initialized successfully")
    
    def _needs_mistral_format(self, model_name_or_path: str) -> bool:
        """
        Check if a model needs Mistral-native loading format.
        
        Args:
            model_name_or_path: Model path or HuggingFace model ID
            
        Returns:
            True if the model needs Mistral-native format
        """
        model_name_lower = model_name_or_path.lower()
        
        for mistral_model in self.MISTRAL_NATIVE_MODELS:
            if mistral_model in model_name_lower:
                return True
        
        return False

    def _is_prefix_cache_incompatible(self, model_name_or_path: str) -> bool:
        """
        Check if a model is known to be incompatible with prefix caching when n>1.

        Some models (e.g., GPT-OSS) produce degenerate output ("!!!..." + NaN logprobs)
        when prefix caching is enabled and multiple sequences are generated per prompt.
        """
        model_name_lower = model_name_or_path.lower()
        for model in self.PREFIX_CACHE_INCOMPATIBLE_MODELS:
            if model in model_name_lower:
                return True
        return False

    def _is_degeneration_sensitive(self) -> bool:
        """Check if the current model is susceptible to token_id=0 degeneration."""
        model_lower = self.model_name_or_path.lower()
        return any(m in model_lower for m in self.DEGENERATION_SENSITIVE_MODELS)

    @staticmethod
    def _detect_degeneration(token_ids: List[int],
                             threshold: int = 10) -> Optional[int]:
        """
        Detect token_id=0 degeneration in a token sequence.

        Returns the position where degeneration starts (first token of a run
        of >= *threshold* consecutive token_id=0), or None if healthy.
        """
        run_start = None
        run_len = 0
        for i, tid in enumerate(token_ids):
            if tid == 0:
                if run_len == 0:
                    run_start = i
                run_len += 1
                if run_len >= threshold:
                    return run_start
            else:
                run_len = 0
                run_start = None
        return None

    def _retry_degenerated_trajectories(
        self,
        sample_results: List[Dict],
        formatted_prompt: str,
        sampling_params,
    ) -> List[Dict]:
        """
        Detect degenerated trajectories in *sample_results* and retry them
        one-by-one (n=1) to minimise KV-cache concurrency pressure.

        Only called for models in DEGENERATION_SENSITIVE_MODELS.

        Returns the (possibly updated) sample_results list.
        """
        threshold = self.DEGENERATION_CONSECUTIVE_THRESHOLD
        max_retries = self.DEGENERATION_MAX_RETRIES

        degen_indices = [
            idx for idx, r in enumerate(sample_results)
            if self._detect_degeneration(r["token_ids"], threshold) is not None
        ]
        if not degen_indices:
            return sample_results

        print(f"  [degeneration] {len(degen_indices)}/{len(sample_results)} "
              f"trajectories degenerated, retrying (max {max_retries} attempts each)")

        # Build a single-sequence SamplingParams for retries
        retry_params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            max_tokens=sampling_params.max_tokens,
            stop=sampling_params.stop,
            logprobs=sampling_params.logprobs,
            n=1,
        )

        recovered = 0
        for idx in degen_indices:
            original_traj_index = sample_results[idx]["trajectory_index"]
            success = False

            for attempt in range(1, max_retries + 1):
                outputs = self.llm.generate([formatted_prompt], retry_params)
                traj_result = self._process_single_completion(
                    outputs[0].outputs[0],
                    trajectory_index=original_traj_index,
                )
                del outputs

                if self._detect_degeneration(traj_result["token_ids"], threshold) is None:
                    # Good — swap in the healthy result, preserve sample metadata
                    new_result = dict(sample_results[idx])
                    new_result.update(traj_result)
                    new_result["retried"] = True
                    new_result["retry_attempt"] = attempt
                    sample_results[idx] = new_result
                    recovered += 1
                    success = True
                    break

            if not success:
                sample_results[idx]["degenerated"] = True
                print(f"    traj {original_traj_index}: all {max_retries} retries "
                      f"failed, marked degenerated")

        print(f"  [degeneration] recovered {recovered}/{len(degen_indices)}")
        return sample_results

    def _init_llm_with_fallback(self, llm_kwargs: Dict) -> LLM:
        """
        Initialize LLM with fallback for unsupported parameters.
        
        Args:
            llm_kwargs: Dictionary of kwargs for LLM initialization
            
        Returns:
            Initialized LLM instance
        """
        # Parameters to try removing if initialization fails
        optional_params = [
            "language_model_only",
            "limit_mm_per_prompt",
            "tokenizer_mode",
            "config_format",
            "load_format",
        ]
        
        current_kwargs = llm_kwargs.copy()
        
        while True:
            try:
                return LLM(**current_kwargs)
            except TypeError as e:
                error_str = str(e)
                removed_param = None
                
                # Find which parameter caused the error
                for param in optional_params:
                    if param in error_str and param in current_kwargs:
                        print(f"Warning: '{param}' not supported by this vLLM version, removing...")
                        del current_kwargs[param]
                        removed_param = param
                        break
                
                if removed_param is None:
                    # No optional param found in error, re-raise
                    raise
            except Exception as e:
                # For other errors, try removing mistral-specific params if present
                if any(p in current_kwargs for p in ["tokenizer_mode", "config_format", "load_format"]):
                    print(f"Warning: Error with Mistral-specific params: {e}")
                    print("Retrying without Mistral-specific loading format...")
                    for param in ["tokenizer_mode", "config_format", "load_format"]:
                        current_kwargs.pop(param, None)
                else:
                    raise

    def _is_multimodal_model(self, model_name_or_path: str) -> bool:
        """
        Check if a model is a known multimodal model that supports text-only mode.

        Args:
            model_name_or_path: Model path or HuggingFace model ID

        Returns:
            True if the model is a known multimodal model
        """
        model_name_lower = model_name_or_path.lower()

        for mm_model in self.MULTIMODAL_TEXT_CAPABLE_MODELS:
            if mm_model in model_name_lower:
                return True

        return False

    def _needs_language_model_only(self, model_name_or_path: str) -> bool:
        """
        Check if a model should use --language-model-only mode for text inference.
        This skips loading the vision encoder, freeing GPU memory for KV cache.
        """
        model_name_lower = model_name_or_path.lower()
        for model in self.LANGUAGE_MODEL_ONLY_MODELS:
            if model in model_name_lower:
                return True
        return False

    def create_sampling_params(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stop_tokens: Optional[List[str]] = None,
        n: int = 1,
    ) -> SamplingParams:
        """
        Create sampling parameters for generation.

        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop_tokens: List of stop tokens
            n: Number of output sequences to generate per prompt (Best-of-N sampling)

        Returns:
            SamplingParams instance
        """
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
            logprobs=self.top_k_logprobs,  # Request top-k logprobs
            n=n,  # Generate N trajectories per prompt
        )

    def _normalize_prompt(self, prompt) -> str:
        """
        Normalize prompt to ensure it's a string format suitable for vLLM.
        
        Some tokenizers (especially for multimodal models like Ministral-3) may return
        different formats from apply_chat_template:
        - str: Direct string output (expected)
        - list: List of strings or message dicts (needs extraction)
        - dict: Dictionary with text content (needs extraction)
        
        Args:
            prompt: The output from apply_chat_template (may be str, list, or dict)
            
        Returns:
            A normalized string prompt suitable for vLLM generate()
        """
        if isinstance(prompt, str):
            # Already a string - ideal case
            return prompt
        
        if isinstance(prompt, list):
            # Handle list output (common with some multimodal tokenizers)
            if len(prompt) == 0:
                print("Warning: Empty list returned from apply_chat_template")
                return ""
            
            # DEBUG: Print the structure for diagnosis
            print(f"DEBUG: _normalize_prompt received list with {len(prompt)} items")
            print(f"DEBUG: First item type: {type(prompt[0])}")
            if len(prompt) > 0:
                first_item = prompt[0]
                if isinstance(first_item, dict):
                    print(f"DEBUG: First item keys: {list(first_item.keys())}")
                elif isinstance(first_item, str):
                    print(f"DEBUG: First item (first 100 chars): {first_item[:100]}")
            
            # If it's a list of strings, join them
            if all(isinstance(item, str) for item in prompt):
                return "".join(prompt)
            
            # If it's a list of dicts (message format), extract content properly
            if all(isinstance(item, dict) for item in prompt):
                # Try to extract text content from each message
                texts = []
                for item in prompt:
                    # Handle OpenAI-style content parts
                    if "content" in item:
                        content = item["content"]
                        if isinstance(content, str):
                            texts.append(content)
                        elif isinstance(content, list):
                            # Content might be a list of text/image parts
                            for part in content:
                                if isinstance(part, str):
                                    texts.append(part)
                                elif isinstance(part, dict):
                                    # Handle {"type": "text", "text": "..."} format
                                    if part.get("type") == "text" and "text" in part:
                                        texts.append(part["text"])
                                    elif "text" in part:
                                        texts.append(part["text"])
                                    elif "content" in part:
                                        texts.append(str(part["content"]))
                    
                    # Also check for 'text' key directly
                    elif "text" in item:
                        texts.append(item["text"])
                
                if texts:
                    return "\n".join(texts)
                else:
                    print(f"Warning: Could not extract text from list of dicts")
                    print(f"DEBUG: Dict keys in list: {[list(item.keys()) for item in prompt]}")
                    return str(prompt)
            
            # Mixed list - try to extract meaningful content
            texts = []
            for item in prompt:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    for key in ["text", "content", "value"]:
                        if key in item:
                            val = item[key]
                            if isinstance(val, str):
                                texts.append(val)
                                break
            
            if texts:
                return "".join(texts)
            
            return str(prompt[0]) if len(prompt) > 0 else ""
        
        if isinstance(prompt, dict):
            # Handle dict output
            print(f"DEBUG: _normalize_prompt received dict with keys: {list(prompt.keys())}")
            
            # Try common keys for text content
            for key in ["text", "content", "prompt", "input", "value"]:
                if key in prompt:
                    val = prompt[key]
                    if isinstance(val, str):
                        return val
                    elif isinstance(val, list):
                        # Recursively normalize
                        return self._normalize_prompt(val)
            
            # If dict has 'messages', extract from there
            if "messages" in prompt:
                return self._normalize_prompt(prompt["messages"])
            
            # Check for OpenAI-style parts
            if "parts" in prompt:
                return self._normalize_prompt(prompt["parts"])
            
            # Last resort: convert to string representation
            print(f"Warning: Unexpected dict format for prompt: {list(prompt.keys())}")
            return str(prompt)
        
        # Fallback for any other type
        print(f"Warning: Unexpected prompt type {type(prompt)}, converting to string")
        return str(prompt)

    def _should_use_manual_template(self, model_name: str) -> bool:
        """
        Check if we should bypass the model's chat template and use manual formatting.
        
        Some models (especially multimodal ones like Ministral-3) have complex chat templates
        that may cause issues with text-only inference. For these models, we use a simpler
        manual format that's known to work.
        
        Args:
            model_name: Model name or path
            
        Returns:
            True if manual template should be used
        """
        model_lower = model_name.lower()
        
        # Models known to have problematic chat templates for text-only inference
        problematic_models = [
            "ministral",  # Ministral-3 has complex Jinja2 template with tools
            "pixtral",    # Pixtral may have similar issues
        ]
        
        for model in problematic_models:
            if model in model_lower:
                return True
        
        return False

    def _apply_chat_template(self, prompts: List[str]) -> List[str]:
        """
        Apply chat template to prompts for instruction-tuned models.
        
        This method handles various tokenizer behaviors, including:
        - Standard text tokenizers that return strings
        - Multimodal tokenizers that may return lists or dicts
        - Tokenizers without chat templates
        - Models with problematic templates (uses manual formatting)

        Args:
            prompts: List of plain text prompts

        Returns:
            List of formatted prompts with chat template applied (always strings)
        """
        tokenizer = self.llm.get_tokenizer()
        formatted_prompts = []
        
        # Get system prompt based on task type
        system_prompt = self.SYSTEM_PROMPTS.get(
            self.task_type, 
            self.SYSTEM_PROMPTS["math"]
        )

        # Check if we should bypass the chat template entirely
        # This is necessary for models like Ministral-3 which:
        # 1. Have complex Jinja2 templates with tools/defaults that don't work for simple inference
        # 2. Use MistralTokenizer which returns token IDs even with tokenize=False
        if self._should_use_manual_template(self.model_name_or_path):
            print(f"INFO: Using manual template for {self.model_name_or_path} (bypassing problematic chat template)")
            for prompt in prompts:
                formatted_prompt = self._construct_fallback_prompt(system_prompt, prompt)
                formatted_prompts.append(formatted_prompt)
            
            # Debug: show first formatted prompt
            if formatted_prompts:
                print(f"DEBUG: First formatted prompt (first 300 chars): {formatted_prompts[0][:300]}...")
            
            return formatted_prompts
        
        # Check if tokenizer has chat_template attribute
        has_chat_template = (
            hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        ) or (
            hasattr(tokenizer, "apply_chat_template")
        )
        
        if not has_chat_template:
            print("Warning: Tokenizer does not have chat_template, using prompts as-is")
            return prompts

        # Check if this is a multimodal model that might need special handling
        is_multimodal = self._is_multimodal_model(self.model_name_or_path)
        
        for idx, prompt in enumerate(prompts):
            # Create messages in chat format
            # Some models don't support system role, try with and without
            messages_with_system = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            messages_without_system = [
                {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
            ]

            # Apply chat template with robust error handling
            formatted_prompt = None
            
            # Try different approaches for applying chat template
            for attempt, messages in enumerate([messages_with_system, messages_without_system]):
                try:
                    # For multimodal models, we need to be more careful
                    # Some may return special formats that need handling
                    result = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # CRITICAL: Check if MistralTokenizer returned token IDs instead of string
                    # MistralTokenizer ignores tokenize=False and always returns token IDs
                    if isinstance(result, list) and result and isinstance(result[0], int):
                        if idx == 0:
                            print(f"DEBUG: Tokenizer returned token IDs (list of {len(result)} ints) instead of string")
                            print("WARNING: This tokenizer ignores tokenize=False, falling back to manual template")
                        # Fall back to manual template for this model
                        formatted_prompt = self._construct_fallback_prompt(system_prompt, prompt)
                        break
                    
                    # Debug: Check what type was returned
                    if idx == 0 and attempt == 0:
                        print(f"DEBUG: apply_chat_template returned type: {type(result)}")
                        if isinstance(result, str):
                            print(f"DEBUG: First 200 chars of formatted prompt: {result[:200]}...")
                        elif isinstance(result, list):
                            print(f"DEBUG: List length: {len(result)}, first item type: {type(result[0]) if result else 'empty'}")
                    
                    # Normalize the output to ensure it's a string
                    normalized_prompt = self._normalize_prompt(result)
                    
                    # CRITICAL: Verify the original prompt content is preserved
                    # This catches cases where the template drops the user content
                    if not self._verify_prompt_content(normalized_prompt, prompt):
                        if idx == 0:
                            print(f"WARNING: Original prompt content not found in formatted result!")
                            print(f"DEBUG: Original prompt (first 100 chars): {prompt[:100]}...")
                            print(f"DEBUG: Formatted prompt (first 200 chars): {normalized_prompt[:200]}...")
                        
                        if attempt == 0:
                            # Try without system prompt
                            continue
                        else:
                            # Fall back to manual construction
                            raise ValueError("Prompt content lost during template application")
                    
                    # Validate the result is a non-empty string
                    if not isinstance(normalized_prompt, str) or not normalized_prompt.strip():
                        raise ValueError("Failed to get valid string from chat template")
                    
                    formatted_prompt = normalized_prompt
                    break
                    
                except Exception as e:
                    if attempt == 0:
                        # Try without system prompt
                        continue
                    else:
                        # All attempts failed
                        print(f"Warning: Failed to apply chat template: {e}")
                        print("Falling back to manual prompt construction")
                        formatted_prompt = self._construct_fallback_prompt(system_prompt, prompt)
            
            if formatted_prompt is None:
                formatted_prompt = self._construct_fallback_prompt(system_prompt, prompt)
            
            formatted_prompts.append(formatted_prompt)

        return formatted_prompts
    
    def _verify_prompt_content(self, formatted_prompt: str, original_prompt: str) -> bool:
        """
        Verify that the original prompt content is preserved in the formatted prompt.
        
        This catches cases where chat template incorrectly processes the input
        and loses the actual user content.
        
        Args:
            formatted_prompt: The result after applying chat template
            original_prompt: The original user prompt
            
        Returns:
            True if a significant portion of the original content is found
        """
        if not formatted_prompt or not original_prompt:
            return False
        
        # Check if key parts of the original prompt appear in the formatted version
        # We use a simple heuristic: check for first 50 chars (excluding whitespace variations)
        original_normalized = " ".join(original_prompt.split())[:50]
        formatted_normalized = " ".join(formatted_prompt.split())
        
        return original_normalized in formatted_normalized
    
    def _construct_fallback_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Construct a fallback prompt when chat template is unavailable or fails.
        
        Uses a format that's compatible with most instruction-tuned models.
        
        Args:
            system_prompt: The system instruction
            user_prompt: The user's input prompt
            
        Returns:
            A formatted prompt string
        """
        # Check if model is Mistral-based (including Ministral)
        model_lower = self.model_name_or_path.lower()
        
        if "mistral" in model_lower or "ministral" in model_lower:
            # Mistral/Ministral format
            # Use the standard Mistral instruction format
            return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        elif "qwen" in model_lower:
            # Qwen format
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "llama" in model_lower:
            # Llama format
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        else:
            # Generic format that works with most models
            return f"{system_prompt}\n\nProblem: {user_prompt}\n\nSolution:"

    def _validate_prompts(self, prompts: List) -> List[str]:
        """
        Validate and normalize prompts to ensure they are all strings.
        
        This is a final safety check before passing prompts to vLLM generate().
        It catches any edge cases where prompts might still be in incorrect format.
        
        Args:
            prompts: List of prompts (should be strings, but may have other types)
            
        Returns:
            List of validated string prompts
            
        Raises:
            ValueError: If prompts cannot be converted to valid strings
        """
        validated = []
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                validated.append(prompt)
            elif isinstance(prompt, (list, dict)):
                # This shouldn't happen if _apply_chat_template worked correctly,
                # but handle it as a safety net
                normalized = self._normalize_prompt(prompt)
                if normalized:
                    print(f"Warning: Prompt {i} was not a string (was {type(prompt).__name__}), normalized it")
                    validated.append(normalized)
                else:
                    raise ValueError(f"Prompt {i} could not be normalized to a valid string")
            else:
                # Try to convert to string as last resort
                try:
                    validated.append(str(prompt))
                    print(f"Warning: Prompt {i} converted from {type(prompt).__name__} to string")
                except Exception as e:
                    raise ValueError(f"Prompt {i} (type {type(prompt).__name__}) cannot be converted to string: {e}")
        
        return validated

    def generate_batch(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        apply_chat_template: bool = True,
    ) -> List[List[Dict]]:
        """
        Generate responses for a batch of prompts.

        When sampling_params.n > 1, generates multiple trajectories per prompt (Best-of-N).

        Args:
            prompts: List of input prompts (plain text)
            sampling_params: Sampling parameters (including n for Best-of-N)
            apply_chat_template: Whether to apply chat template (default: True)

        Returns:
            List of lists: For each prompt, returns N generation results
            Format: [[result_0_traj_0, result_0_traj_1, ...], [result_1_traj_0, ...], ...]
        """
        # Apply chat template if requested
        if apply_chat_template:
            formatted_prompts = self._apply_chat_template(prompts)
            # Debug: print first formatted prompt (optional)
            if len(formatted_prompts) > 0 and sampling_params.n > 1:
                print(f"\n=== Generating {sampling_params.n} trajectories per prompt ===")
        else:
            formatted_prompts = prompts

        # Final validation: ensure all prompts are strings before passing to vLLM
        # This handles edge cases where tokenizers may return unexpected formats
        formatted_prompts = self._validate_prompts(formatted_prompts)

        # Generate with vLLM
        outputs = self.llm.generate(formatted_prompts, sampling_params)

        # Process outputs and calculate entropy
        # When n>1, each output contains multiple CompletionOutputs in output.outputs
        results = []
        for output in outputs:
            # Process all N trajectories for this prompt
            prompt_results = []
            for i, completion_output in enumerate(output.outputs):
                result = self._process_single_completion(completion_output, trajectory_index=i)
                prompt_results.append(result)
            results.append(prompt_results)

        return results

    def _process_single_completion(self, completion_output, trajectory_index: int = 0) -> Dict:
        """
        Process a single completion output (one trajectory) and calculate entropy.

        Args:
            completion_output: vLLM CompletionOutput object (one of the N outputs)
            trajectory_index: Index of this trajectory (0 to N-1)

        Returns:
            Dictionary with generation and entropy information
        """
        # Extract generated text
        raw_generated_text = completion_output.text

        # Apply post-processing to stop after final answer
        generated_text = post_process_math_answer(raw_generated_text)

        # Extract tokens and token IDs
        # IMPORTANT: copy token_ids to break reference to vLLM output object,
        # allowing GC to free the (large) vLLM output once we're done.
        token_ids = list(completion_output.token_ids)

        # Get tokens from token IDs using the tokenizer
        # IMPORTANT: Use skip_special_tokens=False to preserve special tokens for analysis
        tokenizer = self.llm.get_tokenizer()
        try:
            tokens = [
                tokenizer.decode([token_id], skip_special_tokens=False)
                for token_id in token_ids
            ]
        except Exception:
            # Fallback if tokenizer doesn't support individual decoding
            tokens = [f"<token_{tid}>" for tid in token_ids]

        # Extract logprobs
        logprobs_sequence = completion_output.logprobs

        # Calculate entropy if calculator is available
        entropy_sequence = []
        statistics = {}

        if self.entropy_calculator and logprobs_sequence:
            # Convert logprobs format from vLLM to our format
            # vLLM returns list of dicts with Logprob objects
            processed_logprobs = []
            for logprobs_dict in logprobs_sequence:
                if logprobs_dict is not None:
                    # Convert Logprob objects to simple dict
                    simple_dict = {
                        token_id: logprob_obj.logprob
                        for token_id, logprob_obj in logprobs_dict.items()
                    }
                    processed_logprobs.append(simple_dict)
                else:
                    processed_logprobs.append(None)

            # Calculate entropy for each token
            entropy_sequence = self.entropy_calculator.process_sequence_logprobs(
                processed_logprobs, tokens, token_ids
            )

            # Classify tokens by percentile and add color information
            entropy_sequence = self.entropy_calculator.classify_tokens_by_percentile(
                entropy_sequence
            )

            # Calculate statistics
            statistics = self.entropy_calculator.calculate_statistics(entropy_sequence)

        return {
            "trajectory_index": trajectory_index,
            "generated_text": generated_text,
            "raw_generated_text": raw_generated_text,  # Keep original for debugging
            "tokens": tokens,
            "token_ids": token_ids,
            "entropy_sequence": entropy_sequence,
            "statistics": statistics,
            "cutoff_applied": len(generated_text) < len(raw_generated_text),
        }

    def generate_with_progress(
        self,
        samples: List[Dict],
        batch_size: int,
        sampling_params: SamplingParams,
        on_batch_complete: Optional[callable] = None,
    ) -> List[Dict]:
        """
        Generate responses for samples with progress bar and batching.

        When sampling_params.n > 1, generates N trajectories per sample (Best-of-N).

        Memory-optimized: when on_batch_complete is provided, vLLM outputs are
        processed and freed one sample at a time to avoid holding all outputs
        simultaneously. Each sample's trajectories are flushed to disk via the
        callback immediately after processing.

        Args:
            samples: List of sample dictionaries with 'prompt' key
            batch_size: Batch size for processing
            sampling_params: Sampling parameters (including n for Best-of-N)
            on_batch_complete: Optional callback(batch_results: List[Dict]) called
                after each sample's trajectories are processed. If provided,
                results are NOT accumulated in memory — the callback is responsible
                for persisting them. The method returns an empty list in this case.

        Returns:
            List of results with entropy information (empty list if on_batch_complete is used).
            When n > 1, each sample produces n results (trajectory_0, trajectory_1, ...)
        """
        all_results = [] if on_batch_complete is None else None
        streaming = on_batch_complete is not None

        # For degeneration-sensitive models, cap concurrent sequences to avoid
        # KV-cache pressure that triggers token_id=0 degeneration.
        n = getattr(sampling_params, "n", 1) or 1
        if self._is_degeneration_sensitive() and n > 1:
            max_seqs = self.DEGENERATION_MAX_CONCURRENT_SEQS
            safe_batch = max(1, max_seqs // n)
            if batch_size > safe_batch:
                print(f"INFO: [{self.model_name_or_path}] Adjusting batch_size "
                      f"{batch_size} → {safe_batch} to cap concurrent sequences "
                      f"at ~{safe_batch * n} (n={n})")
                batch_size = safe_batch

        # Process in batches
        num_batches = (len(samples) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating responses"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]

            # Extract and format prompts
            prompts = [sample["prompt"] for sample in batch_samples]
            formatted_prompts = self._apply_chat_template(prompts)
            formatted_prompts = self._validate_prompts(formatted_prompts)

            if len(formatted_prompts) > 0 and sampling_params.n > 1:
                print(f"\n=== Generating {sampling_params.n} trajectories per prompt ===")

            # Generate with vLLM — all outputs returned at once (vLLM constraint)
            outputs = self.llm.generate(formatted_prompts, sampling_params)

            # Process outputs incrementally: one sample at a time.
            # After processing each sample's completions, drop the reference to the
            # vLLM RequestOutput so GC can reclaim its logprobs memory (~several MB
            # per trajectory × n trajectories × batch_size can be hundreds of GB).
            for j in range(len(outputs)):
                output = outputs[j]
                sample = batch_samples[j]
                outputs[j] = None  # Drop reference — allow GC to free this output

                sample_results = []
                for k in range(len(output.outputs)):
                    completion_output = output.outputs[k]
                    output.outputs[k] = None  # Drop reference to completion

                    traj_result = self._process_single_completion(
                        completion_output, trajectory_index=k
                    )
                    del completion_output  # Explicitly free

                    # Create unique ID for this trajectory
                    if sampling_params.n > 1:
                        trajectory_id = f"{sample['id']}_traj_{k}"
                    else:
                        trajectory_id = sample["id"]

                    combined_result = {
                        "id": trajectory_id,
                        "original_id": sample["id"],
                        "trajectory_index": k,
                        "problem": sample["problem"],
                        "solution": sample.get("solution", ""),
                        "prompt": sample["prompt"],
                        "source": sample.get("source", ""),
                        "task_name": sample.get("task_name", ""),
                        "game_data": sample.get("game_data", {}),
                        "metadata": sample.get("metadata", {}),
                        "original_data": sample.get("original_data", {}),
                        "category": sample.get("category", ""),
                        **traj_result,
                    }
                    sample_results.append(combined_result)

                del output  # Free the RequestOutput

                # Detect & retry degenerated trajectories (Qwen3.5-specific)
                if self._is_degeneration_sensitive() and n > 1:
                    sample_results = self._retry_degenerated_trajectories(
                        sample_results, formatted_prompts[j], sampling_params,
                    )

                if streaming:
                    on_batch_complete(sample_results)
                    del sample_results
                else:
                    all_results.extend(sample_results)

            del outputs
            gc.collect()

        return all_results if all_results is not None else []


def create_vllm_engine(
    config: Dict,
    tensor_parallel_size: int,
    entropy_calculator: EntropyCalculator,
) -> VLLMEngine:
    """
    Create a vLLM engine from configuration.

    Args:
        config: Configuration dictionary
        tensor_parallel_size: Number of GPUs for tensor parallelism
        entropy_calculator: EntropyCalculator instance

    Returns:
        VLLMEngine instance
    """
    model_config = config.get("model", {})
    gpu_config = config.get("gpu", {})
    entropy_config = config.get("entropy", {})
    dataset_config = config.get("dataset", {})
    
    # Determine task type from dataset configuration
    dataset_type = dataset_config.get("type", "math")
    if dataset_type in ["korbench", "synlogic"]:
        task_type = dataset_type
    elif dataset_type == "logic":
        task_type = "logic"
    elif dataset_type == "bigcodebench":
        task_type = "bigcodebench"
    elif dataset_type == "livecodebench":
        task_type = "livecodebench"
    elif dataset_type == "code":
        task_type = "code"
    else:
        task_type = "math"

    # Get multimodal configuration
    # This allows users to specify limit_mm_per_prompt for multimodal models
    # Example: model.limit_mm_per_prompt: {"image": 0} for text-only mode
    limit_mm_per_prompt = model_config.get("limit_mm_per_prompt", None)
    
    # If text_only mode is explicitly set, configure for text-only
    if model_config.get("text_only", False):
        limit_mm_per_prompt = {"image": 0}
        print("Text-only mode enabled via config")
    
    # Get Mistral format configuration
    # For models like Ministral-3, Pixtral, etc., use Mistral-native format
    # This is equivalent to: --tokenizer_mode mistral --config_format mistral --load_format mistral
    use_mistral_format = model_config.get("use_mistral_format", None)

    # Get prefix caching configuration
    # None = auto-detect (disabled for models known to be incompatible like GPT-OSS)
    # Can be explicitly set in config: model.enable_prefix_caching: true/false
    enable_prefix_caching = model_config.get("enable_prefix_caching", None)

    return VLLMEngine(
        model_name_or_path=model_config.get("name_or_path", "Qwen/Qwen2.5-7B-Instruct"),
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_config.get("gpu_memory_utilization", 0.9),
        max_model_len=model_config.get("max_model_len"),
        trust_remote_code=model_config.get("trust_remote_code", True),
        dtype=model_config.get("dtype", "auto"),
        entropy_calculator=entropy_calculator,
        top_k_logprobs=entropy_config.get("top_k", 20),
        task_type=task_type,
        limit_mm_per_prompt=limit_mm_per_prompt,
        use_mistral_format=use_mistral_format,
        enable_prefix_caching=enable_prefix_caching,
    )
