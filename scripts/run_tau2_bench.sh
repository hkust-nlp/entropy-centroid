#!/bin/bash
# tau2-bench inference pipeline (inference only)
#
# Runs tau2-bench test split (airline=20, retail=40, telecom=40) per model,
# collecting per-token entropy. For evaluation use evaluate_tau2_bench.py.
#
# Usage:
#   bash scripts/run_tau2_bench.sh                    # all models × all domains
#   bash scripts/run_tau2_bench.sh --model-idx 0      # single model by index
#   bash scripts/run_tau2_bench.sh --domain airline   # single domain
#   bash scripts/run_tau2_bench.sh --skip-existing    # skip if entropy_results.jsonl exists
#   bash scripts/run_tau2_bench.sh --dry-run          # 2 tasks per domain only
#
# Env vars (all optional):
#   VLLM_PORT, GPU_MEM_UTIL, TOP_LOGPROBS, NUM_TRIALS, MAX_STEPS,
#   MAX_CONCURRENCY, AGENT_TEMP, USER_TEMP, SEED,
#   AGENT_MAX_TOKENS, USER_MAX_TOKENS

set -euo pipefail

if command -v python3 >/dev/null 2>&1; then PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then PYTHON_BIN="python"
else echo "ERROR: python not found" >&2; exit 127; fi

# ============================================================
# Paths & cache dirs
# ============================================================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_BASE="${PROJECT_DIR}/outputs/tau2_bench"
LOG_DIR="${PROJECT_DIR}/outputs/logs/tau2_bench"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/data/minimax-dialogue/users/xiaoxian/tau2-bench-cache}"

export HF_HOME="${MODEL_CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export XDG_CACHE_HOME="${MODEL_CACHE_ROOT}/xdg"
export TRITON_CACHE_DIR="${MODEL_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${MODEL_CACHE_ROOT}/torchinductor"
export TMPDIR="${MODEL_CACHE_ROOT}/tmp"

mkdir -p "${LOG_DIR}"

# ============================================================
# Inference parameters
# ============================================================
VLLM_PORT="${VLLM_PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
TOP_LOGPROBS="${TOP_LOGPROBS:-10}"
NUM_TRIALS="${NUM_TRIALS:-64}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
AGENT_TEMP="${AGENT_TEMP:-0.7}"
USER_TEMP="${USER_TEMP:-0.7}"
SEED="${SEED:-300}"
# Keep max_tokens small so late-turn inputs (30k+ tokens) stay within context.
# QwQ-32B context=40960; reserve 4096 for output → max input ~36864 tokens.
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-8192}"
USER_MAX_TOKENS="${USER_MAX_TOKENS:-2048}"
TASK_SPLIT="test"

# ============================================================
# Domains
# ============================================================
ALL_DOMAINS=("retail")
# ALL_DOMAINS=("airline" "retail" "telecom")

# ============================================================
# Model list
# Format: "model_name | tp | gpu_ids | max_model_len | prompt_tool_calling"
#
# prompt_tool_calling:
#   false  model supports native tool calling; vLLM auto-selects --tool-call-parser
#   true   no native tool calling; use --prompt-tool-calling (PromptToolAdapter
#          injects schema into system prompt, model replies in JSON)
#
# tool_call_parser mapping (auto_tool_call_parser=True):
#   Qwen2.5 / QwQ / Qwen3-14B          → hermes
#   Qwen3.5-27B                         → qwen3_coder + reasoning_parser=qwen3
#   Qwen3-Coder-* / Qwen3.5-397B-FP8   → qwen3_coder (+ EP/DP for large variants)
#   gpt-oss                             → openai
#   Ministral / Mistral                 → mistral
#   OLMo-3 (Instruct/Think)            → olmo3
#   MiniMax-M2.5                        → minimax_m2 (EP+TP, trust-remote-code)
#   DeepSeek-R1-Distill                 → prompt-based (no tool calling capability)
# ============================================================
MODELS=(
    # model_name                                  | tp | gpu_ids               | max_model_len | prompt_tool_calling
    "MiniMaxAI/MiniMax-M2.5                        | 8  | 0,1,2,3,4,5,6,7       | 196608        | false"
    # "Qwen/Qwen3-Coder-Next                        | 8  | 0,1,2,3,4,5,6,7       | null          | false"
    # "Qwen/Qwen3.5-397B-A17B-FP8                   | 1  | 0,1,2,3,4,5,6,7       | null          | false"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8      | 1  | 0,1,2,3,4,5,6,7       | 131072        | false"
    # "Qwen/Qwen3.5-27B                             | 8  | 0,1,2,3,4,5,6,7       | null          | false"
    # "Qwen/Qwen3-14B                               | 8  | 0,1,2,3,4,5,6,7       | null          | false"
    # "Qwen/QwQ-32B                                 | 8  | 0,1,2,3,4,5,6,7       | null          | false"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B     | 8  | 0,1,2,3,4,5,6,7       | null          | true"
    # "openai/gpt-oss-120b                           | 8  | 0,1,2,3,4,5,6,7       | null          | false"
    # "allenai/Olmo-3.1-32B-Think                    | 8  | 0,1,2,3,4,5,6,7       | null          | false"
    # "allenai/Olmo-3.1-32B-Instruct                 | 8  | 0,1,2,3,4,5,6,7       | null          | false"
    # "mistralai/Ministral-3-14B-Instruct-2512       | 8  | 0,1,2,3,4,5,6,7       | null          | false"
)

# ============================================================
# Helpers
# ============================================================
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_section() { echo ""; echo "============================================================"; echo " $*"; echo "============================================================"; }

get_short_name() { echo "$1" | sed 's|/|_|g'; }

parse_model_spec() {
    IFS='|' read -r m_name m_tp m_gpus m_maxlen m_prompt <<< "$1"
    MODEL_NAME="$(echo "${m_name}" | xargs)"
    MODEL_TP="$(echo "${m_tp}" | xargs)"
    MODEL_GPUS="$(echo "${m_gpus}" | xargs)"
    if [ -z "${m_prompt:-}" ]; then
        MODEL_MAX_MODEL_LEN=""
        MODEL_PROMPT_TOOLS="$(echo "${m_maxlen}" | xargs)"
    else
        MODEL_MAX_MODEL_LEN="$(echo "${m_maxlen}" | xargs)"
        MODEL_PROMPT_TOOLS="$(echo "${m_prompt}" | xargs)"
    fi
}

# ============================================================
# Argument parsing
# ============================================================
FILTER_MODEL_IDX=""
FILTER_DOMAIN=""
SKIP_EXISTING="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-idx|-m) FILTER_MODEL_IDX="$2"; shift 2 ;;
        --domain|-d)    FILTER_DOMAIN="$2";    shift 2 ;;
        --skip-existing) SKIP_EXISTING="true"; shift ;;
        --dry-run)       DRY_RUN="true";       shift ;;
        --help|-h)       head -20 "$0"; exit 0 ;;
        *) log "Unknown argument: $1"; exit 1 ;;
    esac
done

DOMAINS=("${ALL_DOMAINS[@]}")
[ -n "${FILTER_DOMAIN}" ] && DOMAINS=("${FILTER_DOMAIN}")

# ============================================================
# run_single: one model × one domain
# ============================================================
run_single() {
    local model_name="$1" tp_size="$2" gpu_ids="$3" max_model_len="$4"
    local prompt_tool_calling="$5" domain="$6"
    local short_name; short_name="$(get_short_name "${model_name}")"
    local output_dir="${OUTPUT_BASE}/${short_name}/${domain}"
    local entropy_jsonl="${output_dir}/entropy_results.jsonl"
    local run_log="${LOG_DIR}/${short_name}_${domain}_$(date +%Y%m%d_%H%M%S).log"

    if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${entropy_jsonl}" ]; then
        local count; count=$(wc -l < "${entropy_jsonl}" 2>/dev/null || echo "0")
        if [ "${count}" -gt 0 ]; then
            log "  SKIP: ${entropy_jsonl} (${count} entries)"
            return 0
        fi
    fi

    log "  Model: ${model_name} | Domain: ${domain} | TP: ${tp_size} | GPUs: ${gpu_ids}"
    [ -n "${max_model_len}" ] && [ "${max_model_len}" != "null" ] \
        && log "  max_model_len: ${max_model_len}" || log "  max_model_len: auto"
    log "  Output: ${output_dir}"

    mkdir -p "${output_dir}"

    # --save-to uses a fixed name (no timestamp) to support checkpoint resume.
    # --auto-resume skips the interactive "resume? (y/n)" prompt.
    local save_name="${short_name}_${domain}"
    local cmd=(
        "${PYTHON_BIN}" -m tau2.cli run
        --domain "${domain}"
        --task-split-name "${TASK_SPLIT}"
        --num-trials "${NUM_TRIALS}"
        --max-steps "${MAX_STEPS}"
        --max-concurrency "${MAX_CONCURRENCY}"
        --seed "${SEED}"
        --agent "llm_agent"
        --user "user_simulator"
        --agent-llm "${model_name}"
        --user-llm "${model_name}"
        --agent-llm-args "{\"temperature\": ${AGENT_TEMP}, \"max_tokens\": ${AGENT_MAX_TOKENS}}"
        --user-llm-args "{\"temperature\": ${USER_TEMP}, \"max_tokens\": ${USER_MAX_TOKENS}}"
        --local-vllm
        --vllm-port "${VLLM_PORT}"
        --gpu-ids "${gpu_ids}"
        --tensor-parallel-size "${tp_size}"
        --gpu-memory-utilization "${GPU_MEM_UTIL}"
        --top-logprobs "${TOP_LOGPROBS}"
        --entropy-output-dir "${output_dir}"
        --save-to "${save_name}"
        --auto-resume
        --log-level "INFO"
    )

    [ -n "${max_model_len}" ] && [ "${max_model_len}" != "null" ] \
        && cmd+=(--max-model-len "${max_model_len}")
    [ "${prompt_tool_calling}" = "true" ] && cmd+=(--prompt-tool-calling)
    [ "${DRY_RUN}" = "true" ] && { cmd+=(--num-tasks 2); log "  [DRY-RUN] 2 tasks only"; }

    cd "${PROJECT_DIR}/tau2-bench"
    if "${cmd[@]}" 2>&1 | tee "${run_log}"; then
        log "  DONE: ${model_name} / ${domain}"
        cd "${PROJECT_DIR}"; return 0
    else
        log "  FAILED: ${model_name} / ${domain} — see ${run_log}"
        cd "${PROJECT_DIR}"; return 1
    fi
}

# ============================================================
# run_model_all_domains: one model × all domains
# ============================================================
run_model_all_domains() {
    local model_name="$1" tp_size="$2" gpu_ids="$3" max_model_len="$4" prompt_tool_calling="$5"
    local ok=0 fail=0

    log_section "Model: ${model_name}"
    for domain in "${DOMAINS[@]}"; do
        log "--- ${model_name} / ${domain} ---"
        if run_single "${model_name}" "${tp_size}" "${gpu_ids}" "${max_model_len}" \
                      "${prompt_tool_calling}" "${domain}"; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
            log "WARNING: ${model_name} / ${domain} failed, continuing"
        fi
    done
    log "Model $(get_short_name "${model_name}"): ${ok} ok, ${fail} failed"
    return ${fail}
}

# ============================================================
# Main
# ============================================================
log_section "tau2-bench Inference Pipeline"
log "Domains: ${DOMAINS[*]} | Trials: ${NUM_TRIALS} | Steps: ${MAX_STEPS} | Concurrency: ${MAX_CONCURRENCY}"
log "Skip existing: ${SKIP_EXISTING} | Dry run: ${DRY_RUN}"
log "Output: ${OUTPUT_BASE}/"

total_success=0
total_failed=0
failed_list=""

for idx in "${!MODELS[@]}"; do
    [ -n "${FILTER_MODEL_IDX}" ] && [ "${idx}" != "${FILTER_MODEL_IDX}" ] && continue
    parse_model_spec "${MODELS[$idx]}"
    if run_model_all_domains \
        "${MODEL_NAME}" "${MODEL_TP}" "${MODEL_GPUS}" "${MODEL_MAX_MODEL_LEN:-}" \
        "${MODEL_PROMPT_TOOLS}"; then
        total_success=$((total_success + 1))
    else
        total_failed=$((total_failed + 1))
        failed_list="${failed_list}\n  - ${MODEL_NAME}"
    fi
done

log_section "Summary"
log "Succeeded: ${total_success} | Failed: ${total_failed}"
[ ${total_failed} -gt 0 ] && echo -e "Failed:${failed_list}"
log "Output layout: outputs/tau2_bench/<model>/{airline,retail,telecom}/entropy_results.jsonl"

exit ${total_failed}
