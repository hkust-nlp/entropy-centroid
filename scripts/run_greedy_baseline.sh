#!/bin/bash
# Greedy Decoding Baseline — math | logic (korbench/synlogic) | code (livecodebench/bigcodebench)
# Runs temperature=0, n=1 inference as a deterministic baseline across all benchmarks.
#
# Usage:
#   bash scripts/run_greedy_baseline.sh                                 # all benchmarks
#   BENCHMARKS="math" bash scripts/run_greedy_baseline.sh               # math only
#   BENCHMARKS="logic" bash scripts/run_greedy_baseline.sh              # KOR-Bench + SynLogic
#   BENCHMARKS="code" bash scripts/run_greedy_baseline.sh               # LiveCodeBench + BigCodeBench
#   BENCHMARKS="math code" bash scripts/run_greedy_baseline.sh          # math + code
#   MAX_SAMPLES=5 bash scripts/run_greedy_baseline.sh                   # quick smoke test
#   GPU_IDS="4,5,6,7" TENSOR_PARALLEL_SIZE=4 bash scripts/run_greedy_baseline.sh
#   RUN_EVALUATION=false bash scripts/run_greedy_baseline.sh            # inference only
#   EVAL_ONLY=true bash scripts/run_greedy_baseline.sh                  # evaluate existing results
#   EVAL_ONLY=true BENCHMARKS="math" bash scripts/run_greedy_baseline.sh
#   EVAL_ONLY=true EVAL_PATTERN="outputs/greedy/greedy_*QwQ*" bash scripts/run_greedy_baseline.sh

cd "$(dirname "$0")/.."

MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/data/minimax-dialogue/users/xiaoxian/tau2-bench-cache}"
export HF_HOME="${MODEL_CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export XDG_CACHE_HOME="${MODEL_CACHE_ROOT}/xdg"
export TRITON_CACHE_DIR="${MODEL_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${MODEL_CACHE_ROOT}/torchinductor"
export TMPDIR="${MODEL_CACHE_ROOT}/tmp"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"

# ============================================
# Mode & Benchmark Selection
# ============================================
EVAL_ONLY="${EVAL_ONLY:-false}"
EVAL_PATTERN="${EVAL_PATTERN:-outputs/greedy/greedy_*}"
BENCHMARKS="${BENCHMARKS:-math logic code}"

# ============================================
# Model & GPU Configuration
# ============================================
INFERENCE_MODELS=(
    "allenai/Olmo-3.1-32B-Instruct"
    "mistralai/Ministral-3-14B-Instruct-2512"
    # "Qwen/QwQ-32B"
    # "Qwen/Qwen3-14B"
    # "Qwen/Qwen3.5-27B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    # "openai/gpt-oss-120b"
    # "allenai/Olmo-3.1-32B-Think"
)
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

# ============================================
# Greedy Parameters (fixed for all benchmarks)
# ============================================
TEMPERATURE=0.0
TRAJECTORIES_PER_SAMPLE=1
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_SAMPLES="${MAX_SAMPLES:-null}"
ENTROPY_TOP_K=10
ENABLE_STEP_DIVISION=true

# ============================================
# Dataset-specific Configuration
# ============================================

# Math
MATH_CONFIGS=(
    "scripts/configs/config_aime_2025.yaml"
    # "scripts/configs/config_math_500.yaml"
    "scripts/configs/config_olympiadbench.yaml"
    # "scripts/configs/config_amc_2023.yaml"
    # "scripts/configs/config_amo_bench.yaml"
    # "scripts/configs/config_minerva.yaml"
    # "scripts/configs/config_omega.yaml"
)
MATH_MAX_TOKENS=32768

# KOR-Bench
KORBENCH_CATEGORIES=("all")   # cipher | logic | operation | puzzle | counterfactual | all
KORBENCH_MODE="zero-shot"     # zero-shot | three-shot | trick | self-correction
KORBENCH_MAX_TOKENS=32768

# SynLogic
SYNLOGIC_SUBSET="hard"        # easy | hard
SYNLOGIC_TASKS=("all")
SYNLOGIC_SPLIT="validation"
SYNLOGIC_MAX_TOKENS=32768

# LiveCodeBench
LCB_RELEASE_VERSIONS=("release_v6")
LCB_START_DATE="null"
LCB_END_DATE="null"
LCB_MAX_TOKENS=32768

# BigCodeBench
BCB_MAX_TOKENS=32768

# ============================================
# Evaluation Configuration
# ============================================
RUN_EVALUATION="${RUN_EVALUATION:-true}"

# ============================================
# Helper Functions
# ============================================
clean_model_name() { echo "$1" | sed 's/\//_/g'; }

# Unified config patcher — mirrors run_batch.sh.
# Caller must set MAX_TOKENS and DATASET_TYPE before each domain block.
# Usage: create_run_config <base_config> <model> <run_idx> [key=value ...]
create_run_config() {
    local base_config=$1 model_name=$2 run_idx=$3
    shift 3
    local run_config="${base_config%.yaml}_greedy_run_${run_idx}.yaml"

    python3 - "$base_config" "$run_config" "$model_name" \
        "$BATCH_SIZE" "$TRAJECTORIES_PER_SAMPLE" "$GPU_IDS" "$TENSOR_PARALLEL_SIZE" \
        "$ENABLE_STEP_DIVISION" "$MAX_SAMPLES" "$MAX_TOKENS" "$TEMPERATURE" "$ENTROPY_TOP_K" \
        "$DATASET_TYPE" "$@" << 'PYTHON_EOF'
import yaml, sys

base_config, run_config, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
batch_size, trajectories = int(sys.argv[4]), int(sys.argv[5])
gpu_ids_str, tensor_parallel = sys.argv[6], int(sys.argv[7])
enable_step_div = sys.argv[8].lower() == 'true'
max_samples_str = sys.argv[9]
max_tokens, temperature, entropy_top_k = int(sys.argv[10]), float(sys.argv[11]), int(sys.argv[12])
dataset_type = sys.argv[13]
extra = dict(a.split('=', 1) for a in sys.argv[14:] if '=' in a)

try:
    with open(base_config) as f:
        config = yaml.safe_load(f)

    config.setdefault('model', {})['name_or_path'] = model_name

    inf = config.setdefault('inference', {})
    inf.update(batch_size=batch_size, n=trajectories, max_tokens=max_tokens, temperature=temperature)

    config.setdefault('entropy', {})['top_k'] = entropy_top_k

    if gpu_ids_str.strip():
        gpu = config.setdefault('gpu', {})
        gpu['device_ids'] = [int(x.strip()) for x in gpu_ids_str.split(',')]
        gpu['tensor_parallel_size'] = tensor_parallel

    config.setdefault('step_division', {})['enabled'] = enable_step_div

    ds = config.setdefault('dataset', {})
    ds['max_samples'] = int(max_samples_str) if max_samples_str != 'null' else None

    if dataset_type == 'livecodebench':
        ds['type'] = 'livecodebench'
        lcb = ds.setdefault('livecodebench', {})
        lcb['release_version'] = extra.get('release_version', 'release_v6')
        for k in ('start_date', 'end_date'):
            lcb[k] = extra[k] if extra.get(k, 'null') != 'null' else None
    elif dataset_type == 'korbench':
        ds['type'] = 'korbench'
        kb = ds.setdefault('korbench', {})
        kb['category'] = extra.get('category', 'all')
        kb['mode'] = extra.get('mode', 'zero-shot')
    elif dataset_type == 'synlogic':
        ds['type'] = 'synlogic'
        ds['split'] = extra.get('split', 'validation')
        sl = ds.setdefault('synlogic', {})
        sl['subset'] = extra.get('subset', 'hard')
        task = extra.get('task_name', 'all')
        sl['task_name'] = None if task == 'all' else task
    # math / bigcodebench: dataset type already set in base YAML, no override needed

    with open(run_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(run_config)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
}

# Run inference for one (config, model) combination; updates global counters.
run_one() {
    local domain=$1 config_file=$2 model=$3 output_dir=$4
    shift 4   # remaining: extra key=value pairs forwarded to create_run_config

    current_run=$((current_run + 1))
    echo "--- Run $current_run/$total_runs | $domain | model=$model ---"
    echo "    output: $output_dir"

    local temp_config
    if ! temp_config=$(create_run_config "$config_file" "$model" "$current_run" "$@"); then
        echo "ERROR: Config creation failed"
        fail_count=$((fail_count + 1))
        failed_runs+=("$current_run: $domain + $model (config error)")
        return 1
    fi

    local cmd="python3 src/main.py --config \"$temp_config\" --output_dir \"$output_dir\""
    [ "$MAX_SAMPLES" != "null" ] && cmd="$cmd --max_samples $MAX_SAMPLES"
    eval "$cmd" 2>&1 | tee "${output_dir}_log.txt"
    local rc=$?
    rm -f "$temp_config"

    if [ $rc -eq 0 ]; then
        success_count=$((success_count + 1))
        return 0
    else
        echo "ERROR: inference failed (exit=$rc)"
        fail_count=$((fail_count + 1))
        failed_runs+=("$current_run: $domain + $model")
        return 1
    fi
}

# Run evaluation on a completed output directory.
# Usage: run_eval <domain> <output_dir>
run_eval() {
    local domain=$1 output_dir=$2
    echo "  [eval] $domain: $(basename "$output_dir")"
    case "$domain" in
        math|korbench|synlogic)
            python3 evaluate_answers.py --result_dir "$output_dir" \
                --selection lowest_centroid \
                2>&1 | tee "${output_dir}/evaluation_log.txt" || \
                echo "WARNING: evaluation failed for $(basename "$output_dir")"
            ;;
        livecodebench)
            python3 evaluate_livecodebench.py --result_dir "$output_dir" \
                2>&1 | tee "${output_dir}/evaluation_log.txt" || \
                echo "WARNING: evaluation failed for $(basename "$output_dir")"
            ;;
        bigcodebench)
            python3 evaluate_bigcodebench.py --result_dir "$output_dir" \
                2>&1 | tee "${output_dir}/evaluation_log.txt" || \
                echo "WARNING: evaluation failed for $(basename "$output_dir")"
            ;;
    esac
}

# ============================================
# EVAL_ONLY Mode
# Scans existing greedy_* dirs and re-runs evaluation.
# Domain is auto-detected from directory name prefix.
# ============================================
if [ "$EVAL_ONLY" = true ]; then
    echo "============================================"
    echo "EVAL_ONLY | pattern=$EVAL_PATTERN | benchmarks=$BENCHMARKS"
    echo "============================================"
    start_time=$(date +%s)
    eval_ok=0; eval_fail=0; eval_skip=0
    eval_failed_dirs=()

    result_dirs=()
    for d in $EVAL_PATTERN; do [ -d "$d" ] && result_dirs+=("$d"); done

    [ ${#result_dirs[@]} -eq 0 ] && { echo "No directories found matching: $EVAL_PATTERN"; exit 0; }
    echo "Found ${#result_dirs[@]} directories"
    echo ""

    for result_dir in "${result_dirs[@]}"; do
        dir_name=$(basename "$result_dir")

        # Must have entropy_results
        has_entropy=false
        for f in "$result_dir"/entropy_results.*; do [ -f "$f" ] && { has_entropy=true; break; }; done
        if [ "$has_entropy" = false ]; then
            echo "SKIP: $dir_name (no entropy_results)"; eval_skip=$((eval_skip+1)); continue
        fi

        # Auto-detect domain from prefix
        domain=""
        [[ "$dir_name" == greedy_livecodebench_* ]] && domain="livecodebench"
        [[ "$dir_name" == greedy_bigcodebench_* ]]  && domain="bigcodebench"
        [[ "$dir_name" == greedy_korbench_* ]]       && domain="korbench"
        [[ "$dir_name" == greedy_synlogic_* ]]       && domain="synlogic"
        [[ "$dir_name" == greedy_config_* ]]         && domain="math"
        if [ -z "$domain" ]; then
            echo "SKIP: $dir_name (unknown prefix)"; eval_skip=$((eval_skip+1)); continue
        fi

        # Filter by BENCHMARKS
        case "$domain" in
            math)
                echo "$BENCHMARKS" | grep -qw "math"  || { eval_skip=$((eval_skip+1)); continue; } ;;
            korbench|synlogic)
                echo "$BENCHMARKS" | grep -qw "logic" || { eval_skip=$((eval_skip+1)); continue; } ;;
            livecodebench|bigcodebench)
                echo "$BENCHMARKS" | grep -qw "code"  || { eval_skip=$((eval_skip+1)); continue; } ;;
        esac

        echo "--- Evaluating: $dir_name ($domain) ---"

        if run_eval "$domain" "$result_dir"; then
            eval_ok=$((eval_ok + 1))
        else
            eval_fail=$((eval_fail + 1))
            eval_failed_dirs+=("$dir_name")
        fi
        echo ""
    done

    end_time=$(date +%s); total_time=$((end_time-start_time))
    echo "============================================"
    echo "EVAL_ONLY done: $eval_ok ok, $eval_fail failed, $eval_skip skipped"
    [ $eval_fail -gt 0 ] && for d in "${eval_failed_dirs[@]}"; do echo "  FAILED: $d"; done
    echo "Time: $((total_time/3600))h $(((total_time%3600)/60))m $((total_time%60))s"
    echo "============================================"
    exit $eval_fail
fi

# ============================================
# Inference Mode: pre-compute total_runs
# ============================================
total_runs=0
for bm in $BENCHMARKS; do
    case "$bm" in
        math)  total_runs=$((total_runs + ${#MATH_CONFIGS[@]} * ${#INFERENCE_MODELS[@]})) ;;
        logic) total_runs=$((total_runs + (${#KORBENCH_CATEGORIES[@]} + ${#SYNLOGIC_TASKS[@]}) * ${#INFERENCE_MODELS[@]})) ;;
        code)  total_runs=$((total_runs + (${#LCB_RELEASE_VERSIONS[@]} + 1) * ${#INFERENCE_MODELS[@]})) ;;
    esac
done

echo "============================================"
echo "Greedy Baseline | benchmarks: $BENCHMARKS"
echo "Models:  ${INFERENCE_MODELS[*]}"
echo "GPU:     $GPU_IDS (TP=$TENSOR_PARALLEL_SIZE)"
echo "Greedy:  temp=0.0, n=1 | Batch=$BATCH_SIZE | MaxSamples=$MAX_SAMPLES"
echo "Evaluation after inference: $RUN_EVALUATION"
echo "Total planned runs: $total_runs"
echo "============================================"
echo ""

start_time=$(date +%s)
current_run=0; success_count=0; fail_count=0; failed_runs=()

# ============================================
# MATH
# ============================================
if echo "$BENCHMARKS" | grep -qw "math"; then
    echo "======== MATH ========"
    MAX_TOKENS=$MATH_MAX_TOKENS; DATASET_TYPE="math"

    for config in "${MATH_CONFIGS[@]}"; do
        if [ ! -f "$config" ]; then
            echo "WARNING: $config not found, skipping"; continue
        fi
        config_base=$(basename "${config%.yaml}")
        for model in "${INFERENCE_MODELS[@]}"; do
            clean_model=$(clean_model_name "$model")
            output_dir="./outputs/greedy/greedy_${config_base}_${clean_model}_$(date +%Y%m%d_%H%M%S)"
            if run_one "math" "$config" "$model" "$output_dir"; then
                [ "$RUN_EVALUATION" = true ] && run_eval "math" "$output_dir"
            fi
        done
    done
fi

# ============================================
# LOGIC: KOR-Bench + SynLogic
# ============================================
if echo "$BENCHMARKS" | grep -qw "logic"; then

    echo "======== LOGIC: KOR-Bench ========"
    if [ ! -f "scripts/configs/config_korbench.yaml" ]; then
        echo "WARNING: config_korbench.yaml not found, skipping"
    else
        MAX_TOKENS=$KORBENCH_MAX_TOKENS; DATASET_TYPE="korbench"
        for cat in "${KORBENCH_CATEGORIES[@]}"; do
            for model in "${INFERENCE_MODELS[@]}"; do
                clean_model=$(clean_model_name "$model")
                output_dir="./outputs/greedy/greedy_korbench_${cat}_${KORBENCH_MODE}_${clean_model}_$(date +%Y%m%d_%H%M%S)"
                if run_one "korbench" "scripts/configs/config_korbench.yaml" "$model" "$output_dir" \
                    "category=$cat" "mode=$KORBENCH_MODE"; then
                    [ "$RUN_EVALUATION" = true ] && run_eval "korbench" "$output_dir"
                fi
            done
        done
    fi

    echo "======== LOGIC: SynLogic ========"
    if [ ! -f "scripts/configs/config_synlogic.yaml" ]; then
        echo "WARNING: config_synlogic.yaml not found, skipping"
    else
        MAX_TOKENS=$SYNLOGIC_MAX_TOKENS; DATASET_TYPE="synlogic"
        for task in "${SYNLOGIC_TASKS[@]}"; do
            for model in "${INFERENCE_MODELS[@]}"; do
                clean_model=$(clean_model_name "$model")
                output_dir="./outputs/greedy/greedy_synlogic_${SYNLOGIC_SUBSET}_${task}_${clean_model}_$(date +%Y%m%d_%H%M%S)"
                if run_one "synlogic" "scripts/configs/config_synlogic.yaml" "$model" "$output_dir" \
                    "task_name=$task" "subset=$SYNLOGIC_SUBSET" "split=$SYNLOGIC_SPLIT"; then
                    [ "$RUN_EVALUATION" = true ] && run_eval "synlogic" "$output_dir"
                fi
            done
        done
    fi
fi

# ============================================
# CODE: LiveCodeBench + BigCodeBench
# ============================================
if echo "$BENCHMARKS" | grep -qw "code"; then

    echo "======== CODE: LiveCodeBench ========"
    if [ ! -f "scripts/configs/config_livecodebench.yaml" ]; then
        echo "WARNING: config_livecodebench.yaml not found, skipping"
    else
        MAX_TOKENS=$LCB_MAX_TOKENS; DATASET_TYPE="livecodebench"
        for release in "${LCB_RELEASE_VERSIONS[@]}"; do
            for model in "${INFERENCE_MODELS[@]}"; do
                clean_model=$(clean_model_name "$model")
                output_dir="./outputs/greedy/greedy_livecodebench_${release}_${clean_model}_$(date +%Y%m%d_%H%M%S)"
                if run_one "livecodebench" "scripts/configs/config_livecodebench.yaml" "$model" "$output_dir" \
                    "release_version=$release" "start_date=$LCB_START_DATE" "end_date=$LCB_END_DATE"; then
                    [ "$RUN_EVALUATION" = true ] && run_eval "livecodebench" "$output_dir"
                fi
            done
        done
    fi

    echo "======== CODE: BigCodeBench ========"
    if [ ! -f "scripts/configs/config_bigcodebench.yaml" ]; then
        echo "WARNING: config_bigcodebench.yaml not found, skipping"
    else
        MAX_TOKENS=$BCB_MAX_TOKENS; DATASET_TYPE="bigcodebench"
        for model in "${INFERENCE_MODELS[@]}"; do
            clean_model=$(clean_model_name "$model")
            output_dir="./outputs/greedy/greedy_bigcodebench_${clean_model}_$(date +%Y%m%d_%H%M%S)"
            if run_one "bigcodebench" "scripts/configs/config_bigcodebench.yaml" "$model" "$output_dir"; then
                [ "$RUN_EVALUATION" = true ] && run_eval "bigcodebench" "$output_dir"
            fi
        done
    fi
fi

# ============================================
# Summary
# ============================================
end_time=$(date +%s); total_time=$((end_time - start_time))
echo ""
echo "============================================"
echo "Done: $success_count/$total_runs succeeded, $fail_count failed"
if [ $fail_count -gt 0 ]; then
    echo "Failed runs:"
    for r in "${failed_runs[@]}"; do echo "  - $r"; done
fi
echo "Time: $((total_time/3600))h $(((total_time%3600)/60))m $((total_time%60))s"
echo "Output: outputs/greedy/greedy_*"
echo "============================================"
exit $fail_count
