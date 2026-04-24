"""
vLLM OpenAI-compatible server manager.

Auto-launches and manages a vLLM server as a subprocess for local inference
with logprobs support.
"""

import os
import signal
import subprocess
import sys
import time
from typing import Optional

import requests
from loguru import logger


# Tool call parser mapping for known models.
#
# Valid vLLM parsers (vllm >= 0.12):
#   deepseek_v3, deepseek_v31, ernie45, glm45, granite, granite-20b-fc,
#   hermes, hunyuan_a13b, internlm, jamba, kimi_k2, llama3_json,
#   llama4_json, llama4_pythonic, longcat, minimax, minimax_m2, mistral,
#   olmo3, openai, phi4_mini_json, pythonic, qwen3_coder, qwen3_xml,
#   seed_oss, step3, xlam
#
# Official documentation references:
#   Qwen2.5 / QwQ-32B      → hermes   (chat template has Hermes-style tool use)
#   Qwen3 (non-Coder)      → hermes
#   Qwen3.5                → qwen3_coder + reasoning_parser=qwen3 + language_model_only
#   Qwen3.5-397B FP8       → qwen3_coder + EP/DP + language_model_only + prefix caching
#   Qwen3-Coder-480B FP8   → qwen3_coder + EP/DP + DeepGEMM
#   Qwen3-Coder-Next       → qwen3_coder
#   Qwen3-Coder            → qwen3_xml
#   OLMo-3 (Instruct/Think) → olmo3
#   Mistral / Ministral     → mistral
#   openai/gpt-oss          → openai
#   MiniMax-M2.5            → minimax_m2 (EP + TP, trust-remote-code, SAFETENSORS_FAST_GPU)
TOOL_CALL_PARSER_MAP = {
    "qwen3-coder-480b": "qwen3_coder",  # Official recipe uses qwen3_coder
    "qwen3-coder-next": "qwen3_coder",  # Qwen3-Coder-Next uses the dedicated coder parser
    "qwen3-coder": "qwen3_xml",   # Qwen3-Coder specific (must match before "qwen3.5")
    "qwen3.5": "qwen3_coder",     # Qwen3.5 uses qwen3_coder (official vLLM docs)
    "qwen": "hermes",             # Qwen2.5, Qwen3 (non-Coder)
    "qwq": "hermes",              # QwQ-32B
    "minimax-m2": "minimax_m2",   # MiniMaxAI/MiniMax-M2.5
    "ministral": "mistral",
    "mistral": "mistral",
    "hermes": "hermes",
    "gpt-oss": "openai",          # openai/gpt-oss-20b, gpt-oss-120b
    "olmo": "olmo3",              # allenai/Olmo-3-*
}

# Models incompatible with vLLM's prefix caching.
# GPT-OSS (mxfp4 quantization): prefix caching causes KV cache pollution
# → NaN logprobs → json serialization failure → 500 Internal Server Error.
PREFIX_CACHE_INCOMPATIBLE_MODELS = [
    "gpt-oss",
]

# Models that require Mistral-native loading format.
# These models use the mistral_common tokenizer and need special flags to load correctly.
# Reference: vllm serve mistralai/Ministral-3-14B-Instruct-2512 \
#   --tokenizer_mode mistral --config_format mistral --load_format mistral
MISTRAL_NATIVE_MODELS = [
    "ministral",
    "pixtral",
    "mistral-large",
]

# Multimodal models that should use --language-model-only for text-only inference.
# These models have a vision encoder that must be skipped when doing text-only tasks;
# without this flag, vLLM will crash during multimodal profiling.
# Ref: https://docs.vllm.ai/en/latest/models/supported_models.html
LANGUAGE_MODEL_ONLY_MODELS = [
    "qwen3.5",
]


EXPERT_PARALLEL_MODELS = [
    "qwen3-coder-480b-a35b-instruct-fp8",
    "qwen3.5-397b-a17b-fp8",
]

# Models that use Expert Parallel + Tensor Parallel (EP+TP) instead of EP+DP.
# These models need --enable-expert-parallel combined with --tensor-parallel-size,
# unlike the DP-based EP models above.
EXPERT_PARALLEL_TP_MODELS = [
    "minimax-m2",
]

DEEP_GEMM_MODELS = [
    "qwen3-coder-480b-a35b-instruct-fp8",
]


FORCE_ENABLE_PREFIX_CACHING_MODELS = [
    "qwen3.5-397b-a17b-fp8",
]

# Models that require --trust-remote-code for custom model/tokenizer code.
TRUST_REMOTE_CODE_MODELS = [
    "minimax-m2",
]

# Models that benefit from SAFETENSORS_FAST_GPU=1 for faster weight loading.
SAFETENSORS_FAST_GPU_MODELS = [
    "minimax-m2",
]

# Large MoE models that need extended startup timeout (seconds).
# Default timeout is 1200s; these models require more time for weight loading
# and CUDA graph capture across many experts / GPUs.
EXTENDED_STARTUP_TIMEOUT_MODELS = {
    "minimax-m2": 2400,             # 256 experts, 125 safetensor files
    "qwen3-coder-480b": 2400,
    "qwen3.5-397b": 2400,
}


REASONING_PARSER_MAP = {
    "qwen3.5": "qwen3",
    "minimax-m2": "minimax_m2_append_think",
}

def infer_tool_call_parser(model_name: str) -> Optional[str]:
    """Infer the tool call parser from the model name.

    Returns None for models that should use prompt-based tool calling
    (e.g., DeepSeek-R1-Distill).
    """
    model_lower = model_name.lower()
    # DeepSeek-R1-Distill models don't support native tool calling
    if "deepseek" in model_lower and "distill" in model_lower:
        return None
    for key, parser in TOOL_CALL_PARSER_MAP.items():
        if key in model_lower:
            return parser
    return None


class VLLMServerManager:
    """Manages a vLLM OpenAI-compatible server subprocess."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_ids: str = "0",
        port: int = 8000,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        tool_call_parser: Optional[str] = None,
        auto_tool_call_parser: bool = False,
        log_dir: Optional[str] = None,
        enable_prefix_caching: Optional[bool] = None,
    ):
        """
        Initialize the vLLM server manager.

        Args:
            model_name: HuggingFace model name or path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_ids: Comma-separated GPU IDs (e.g., "4,5,6,7").
            port: Port for the server.
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_len: Maximum model context length. None for auto-detect.
            tool_call_parser: Tool call parser name (e.g., "hermes", "mistral").
                None means no native tool calling (use prompt-based).
            auto_tool_call_parser: If True, auto-detect parser from model name.
            log_dir: Directory to write vLLM server logs. None to discard.
            enable_prefix_caching: Explicit prefix caching control. None for
                auto-detect (disabled for known-incompatible models).
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_ids = gpu_ids
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.process: Optional[subprocess.Popen] = None
        self.log_dir = log_dir
        self._log_file = None  # File handle for vLLM server log

        if auto_tool_call_parser and tool_call_parser is None:
            self.tool_call_parser = infer_tool_call_parser(model_name)
        else:
            self.tool_call_parser = tool_call_parser

        # Auto-detect prefix caching incompatibility
        if enable_prefix_caching is None:
            model_lower = model_name.lower()
            for pattern in FORCE_ENABLE_PREFIX_CACHING_MODELS:
                if pattern in model_lower:
                    enable_prefix_caching = True
                    logger.info(
                        f"Auto-enabling prefix caching for {model_name} "
                        f"(official launch recipe)"
                    )
                    break
            for pattern in PREFIX_CACHE_INCOMPATIBLE_MODELS:
                if pattern in model_lower:
                    enable_prefix_caching = False
                    logger.warning(
                        f"Auto-disabling prefix caching for {model_name} "
                        f"(known to cause NaN logprobs via KV cache pollution)"
                    )
                    break
        self.enable_prefix_caching = enable_prefix_caching

    def _should_use_expert_parallel(self) -> bool:
        """Whether this model should follow a model-specific EP/DP launch recipe."""
        model_lower = self.model_name.lower()
        return any(pattern in model_lower for pattern in EXPERT_PARALLEL_MODELS)

    def _should_use_expert_parallel_tp(self) -> bool:
        """Whether this model should use EP + TP (instead of EP + DP)."""
        model_lower = self.model_name.lower()
        return any(pattern in model_lower for pattern in EXPERT_PARALLEL_TP_MODELS)

    def _should_use_deep_gemm(self) -> bool:
        """Whether this model should enable DeepGEMM explicitly."""
        model_lower = self.model_name.lower()
        return any(pattern in model_lower for pattern in DEEP_GEMM_MODELS)

    def _get_data_parallel_size(self) -> int:
        """Infer DP size from the visible GPU list for single-node launches."""
        return len([gpu for gpu in self.gpu_ids.split(",") if gpu.strip()])

    def _get_reasoning_parser(self) -> Optional[str]:
        """Infer the reasoning parser for models that need one."""
        model_lower = self.model_name.lower()
        for key, parser in REASONING_PARSER_MAP.items():
            if key in model_lower:
                return parser
        return None

    def _should_trust_remote_code(self) -> bool:
        """Whether this model requires --trust-remote-code."""
        model_lower = self.model_name.lower()
        return any(pattern in model_lower for pattern in TRUST_REMOTE_CODE_MODELS)

    def _should_safetensors_fast_gpu(self) -> bool:
        """Whether this model benefits from SAFETENSORS_FAST_GPU=1."""
        model_lower = self.model_name.lower()
        return any(pattern in model_lower for pattern in SAFETENSORS_FAST_GPU_MODELS)

    def _get_startup_timeout(self) -> int:
        """Get startup timeout, extended for large MoE models."""
        model_lower = self.model_name.lower()
        for pattern, timeout in EXTENDED_STARTUP_TIMEOUT_MODELS.items():
            if pattern in model_lower:
                return timeout
        return 1200  # default

    def start(
        self,
        timeout: Optional[int] = None,
        poll_interval: float = 5.0,
    ) -> None:
        """Launch the vLLM server as a subprocess and wait for it to be ready.

        Args:
            timeout: Maximum seconds to wait for server readiness.
                None to auto-detect based on model (extended for large MoE models).
            poll_interval: Seconds between health check polls.
        """
        if self.process is not None:
            logger.warning("vLLM server is already running")
            return

        if timeout is None:
            timeout = self._get_startup_timeout()

        use_expert_parallel = self._should_use_expert_parallel()
        use_expert_parallel_tp = self._should_use_expert_parallel_tp()
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]

        if use_expert_parallel:
            # EP + DP pattern (e.g., Qwen3-Coder-480B, Qwen3.5-397B)
            data_parallel_size = self._get_data_parallel_size()
            cmd.extend([
                "--enable-expert-parallel",
                "--data-parallel-size", str(data_parallel_size),
            ])
        elif use_expert_parallel_tp:
            # EP + TP pattern (e.g., MiniMax-M2.5)
            cmd.extend([
                "--enable-expert-parallel",
                "--tensor-parallel-size", str(self.tensor_parallel_size),
            ])
        else:
            cmd.extend([
                "--tensor-parallel-size", str(self.tensor_parallel_size),
            ])

        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        if self.enable_prefix_caching is not None:
            if self.enable_prefix_caching:
                cmd.append("--enable-prefix-caching")
            else:
                cmd.append("--no-enable-prefix-caching")

        # Trust remote code for models with custom model/tokenizer implementations.
        if self._should_trust_remote_code():
            cmd.append("--trust-remote-code")
            logger.info(f"Using --trust-remote-code for {self.model_name}")

        # For Mistral-native models (Ministral, Pixtral, etc.), use official
        # recommended loading format to ensure correct tokenization.
        model_lower = self.model_name.lower()
        for pattern in MISTRAL_NATIVE_MODELS:
            if pattern in model_lower:
                cmd.extend([
                    "--tokenizer-mode", "mistral",
                    "--config-format", "mistral",
                    "--load-format", "mistral",
                ])
                logger.info(f"Using Mistral-native loading format for {self.model_name}")
                break

        # For multimodal models used in text-only mode (e.g., Qwen3.5),
        # skip loading the vision encoder to avoid CUBLAS profiling errors.
        for pattern in LANGUAGE_MODEL_ONLY_MODELS:
            if pattern in model_lower:
                cmd.append("--language-model-only")
                logger.info(f"Using --language-model-only for {self.model_name}")
                break

        if self.tool_call_parser is not None:
            cmd.extend([
                "--enable-auto-tool-choice",
                "--tool-call-parser", self.tool_call_parser,
            ])

        reasoning_parser = self._get_reasoning_parser()
        if reasoning_parser is not None:
            cmd.extend([
                "--reasoning-parser", reasoning_parser,
            ])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        use_deep_gemm = self._should_use_deep_gemm()
        if use_expert_parallel and use_deep_gemm:
            # Official Qwen3-Coder-480B FP8 recipe enables DeepGEMM explicitly.
            env["VLLM_USE_DEEP_GEMM"] = "1"
        if self._should_safetensors_fast_gpu():
            env["SAFETENSORS_FAST_GPU"] = "1"
            logger.info(f"Setting SAFETENSORS_FAST_GPU=1 for {self.model_name}")

        # Determine vLLM log file path
        vllm_log_path = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            model_short = self.model_name.replace("/", "_")
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            vllm_log_path = os.path.join(self.log_dir, f"vllm_{model_short}_{ts}.log")

        if use_expert_parallel:
            deep_gemm_msg = ", VLLM_USE_DEEP_GEMM=1" if use_deep_gemm else ""
            logger.info(
                f"Starting vLLM server: model={self.model_name}, "
                f"gpu_ids={self.gpu_ids}, ep=True, dp={self._get_data_parallel_size()}, "
                f"port={self.port}, tool_parser={self.tool_call_parser}, "
                f"reasoning_parser={reasoning_parser}, "
                f"prefix_caching={self.enable_prefix_caching}"
                f"{deep_gemm_msg}"
            )
        elif use_expert_parallel_tp:
            safetensors_msg = ", SAFETENSORS_FAST_GPU=1" if self._should_safetensors_fast_gpu() else ""
            logger.info(
                f"Starting vLLM server: model={self.model_name}, "
                f"gpu_ids={self.gpu_ids}, ep=True, tp={self.tensor_parallel_size}, "
                f"port={self.port}, tool_parser={self.tool_call_parser}, "
                f"reasoning_parser={reasoning_parser}, "
                f"prefix_caching={self.enable_prefix_caching}"
                f"{safetensors_msg}"
            )
        else:
            logger.info(
                f"Starting vLLM server: model={self.model_name}, "
                f"gpu_ids={self.gpu_ids}, tp={self.tensor_parallel_size}, "
                f"port={self.port}, tool_parser={self.tool_call_parser}, "
                f"reasoning_parser={reasoning_parser}, "
                f"prefix_caching={self.enable_prefix_caching}"
            )
        if vllm_log_path:
            logger.info(f"vLLM server log: {vllm_log_path}")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Write vLLM output to log file (or PIPE if no log_dir)
        if vllm_log_path:
            self._log_file = open(vllm_log_path, "w")
            stdout_target = self._log_file
        else:
            stdout_target = subprocess.PIPE

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                # Process has exited
                stdout = self.process.stdout.read().decode() if self.process.stdout else ""
                raise RuntimeError(
                    f"vLLM server process exited with code {self.process.returncode}.\n"
                    f"Output:\n{stdout[-2000:]}"
                )
            if self.health_check():
                logger.info(
                    f"vLLM server is ready at {self.get_api_base()} "
                    f"(took {time.time() - start_time:.1f}s)"
                )
                return
            time.sleep(poll_interval)

        # Timeout reached
        self.stop()
        raise TimeoutError(
            f"vLLM server did not become ready within {timeout}s"
        )

    def stop(self) -> None:
        """Terminate the vLLM server subprocess."""
        if self.process is None:
            return

        logger.info("Stopping vLLM server...")
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=10)
            except Exception:
                pass
        finally:
            self.process = None
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            logger.info("vLLM server stopped")

    def health_check(self) -> bool:
        """Check if the vLLM server is healthy via GET /health."""
        try:
            resp = requests.get(
                f"http://localhost:{self.port}/health",
                timeout=5,
            )
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def get_api_base(self) -> str:
        """Return the base URL for the OpenAI-compatible API."""
        return f"http://localhost:{self.port}/v1"

    def get_model_name(self) -> str:
        """Return the model name used by the server."""
        return self.model_name

    def is_running(self) -> bool:
        """Check if the server process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
