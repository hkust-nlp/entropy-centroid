#!/bin/bash
# Unified Batch Inference Script
# Supports: math | livecodebench | korbench | synlogic | bigcodebench
# Note: tau2-bench uses run_tau2_bench.sh (different invocation via tau2.cli)

# ============================================
# DATASET TYPE: math | livecodebench | korbench | synlogic | bigcodebench
# ============================================
DATASET_TYPE="math"

# ============================================
# Dataset-specific Configuration
# ============================================

# --- math: list of YAML configs to iterate (each encodes a dataset) ---
MATH_CONFIGS=(
    "scripts/configs/config_aime_2025.yaml"
    # "scripts/configs/config_math_500.yaml"
    # "scripts/configs/config_olympiadbench.yaml"
    # "scripts/configs/config_amc_2023.yaml"
    # "scripts/configs/config_amo_bench.yaml"
)

# --- livecodebench ---
LCB_RELEASE_VERSIONS=("release_v6")
LCB_START_DATE="null"
LCB_END_DATE="null"

# --- korbench ---
KORBENCH_CATEGORIES=("all")   # cipher | logic | operation | puzzle | counterfactual | all
KORBENCH_MODE="zero-shot"     # zero-shot | three-shot | trick | self-correction

# --- synlogic ---
SYNLOGIC_SUBSET="hard"        # easy | hard
SYNLOGIC_TASKS=("all")
SYNLOGIC_SPLIT="validation"

# --- bigcodebench ---
BCB_MODE="instruct"           # instruct | complete
BCB_SUBSET="v0.1.4"          # v0.1.0_hf | v0.1.1 | v0.1.2 | v0.1.3 | v0.1.4

# ============================================
# Model Configuration
# ============================================
INFERENCE_MODELS=(
    "allenai/Olmo-3.1-32B-Instruct"
    # "Qwen/Qwen3.5-27B"
    # "Qwen/QwQ-32B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    # "openai/gpt-oss-120b"
    # "mistralai/Ministral-3-14B-Instruct-2512"
)

# ============================================
# GPU Configuration
# ============================================
GPU_IDS="0,1,2,3,4,5,6,7"
TENSOR_PARALLEL_SIZE=8

# ============================================
# Batch & Inference Parameters
# ============================================
BATCH_SIZE=128
MAX_SAMPLES=null
TRAJECTORIES_PER_SAMPLE=64
MAX_TOKENS=32768
TEMPERATURE=0.7
ENTROPY_TOP_K=10
ENABLE_STEP_DIVISION=true

# ============================================
# Helper Functions
# ============================================
clean_model_name() { echo "$1" | sed 's/\//_/g'; }

# Unified config creation: patches a base YAML with runtime parameters.
# Usage: create_run_config <base_config> <model> <run_idx> [key=value ...]
create_run_config() {
    local base_config=$1 model_name=$2 run_idx=$3
    shift 3
    local run_config="${base_config%.yaml}_run_${run_idx}.yaml"

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
    elif dataset_type == 'bigcodebench':
        ds['type'] = 'bigcodebench'
        bcb = ds.setdefault('bigcodebench', {})
        bcb['mode'] = extra.get('bcb_mode', 'instruct')
        bcb['subset'] = extra.get('bcb_subset', 'v0.1.4')
    # math: dataset type already set in base YAML, no override needed

    with open(run_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(run_config)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
}

# ============================================
# Setup: resolve BASE_CONFIG and TASKS
# ============================================
case "$DATASET_TYPE" in
    math)
        TASKS=("${MATH_CONFIGS[@]}")
        ;;
    livecodebench)
        BASE_CONFIG="scripts/configs/config_livecodebench.yaml"
        TASKS=("${LCB_RELEASE_VERSIONS[@]}")
        ;;
    korbench)
        BASE_CONFIG="scripts/configs/config_korbench.yaml"
        TASKS=("${KORBENCH_CATEGORIES[@]}")
        ;;
    synlogic)
        BASE_CONFIG="scripts/configs/config_synlogic.yaml"
        TASKS=("${SYNLOGIC_TASKS[@]}")
        ;;
    bigcodebench)
        BASE_CONFIG="scripts/configs/config_bigcodebench.yaml"
        TASKS=("${BCB_MODE}")   # one task per mode; extend array for multiple modes
        ;;
    *)
        echo "ERROR: Unknown DATASET_TYPE: $DATASET_TYPE"
        exit 1
        ;;
esac

# Validate config files exist
if [ "$DATASET_TYPE" = "math" ]; then
    for cfg in "${TASKS[@]}"; do
        [ -f "$cfg" ] || { echo "ERROR: Config not found: $cfg"; exit 1; }
    done
else
    [ -f "$BASE_CONFIG" ] || { echo "ERROR: Config not found: $BASE_CONFIG"; exit 1; }
fi

# ============================================
# Print run summary
# ============================================
echo "============================================"
echo "Batch Inference: $DATASET_TYPE"
echo "Models:  ${INFERENCE_MODELS[*]}"
echo "Tasks:   ${TASKS[*]}"
echo "GPUs:    $GPU_IDS  (TP=$TENSOR_PARALLEL_SIZE)"
echo "Batch=$BATCH_SIZE  Traj=$TRAJECTORIES_PER_SAMPLE  MaxTokens=$MAX_TOKENS  Temp=$TEMPERATURE"
echo "============================================"
echo ""

# ============================================
# Main Loop
# ============================================
total_runs=$((${#TASKS[@]} * ${#INFERENCE_MODELS[@]}))
current_run=0
start_time=$(date +%s)
success_count=0
fail_count=0
failed_runs=()

for task in "${TASKS[@]}"; do
    for model in "${INFERENCE_MODELS[@]}"; do
        current_run=$((current_run + 1))
        clean_model=$(clean_model_name "$model")
        timestamp=$(date +%Y%m%d_%H%M%S)

        # Resolve per-iteration config file, extra args, and output dir
        if [ "$DATASET_TYPE" = "math" ]; then
            config_file="$task"
            extra_args=()
            output_dir="./outputs/results/$(basename "${task%.yaml}")_${clean_model}_${timestamp}"
        else
            config_file="$BASE_CONFIG"
            case "$DATASET_TYPE" in
                livecodebench)
                    extra_args=("release_version=$task" "start_date=$LCB_START_DATE" "end_date=$LCB_END_DATE")
                    output_dir="./outputs/results/livecodebench_${task}_${clean_model}_${timestamp}"
                    ;;
                korbench)
                    extra_args=("category=$task" "mode=$KORBENCH_MODE")
                    output_dir="./outputs/results/korbench_${task}_${KORBENCH_MODE}_${clean_model}_${timestamp}"
                    ;;
                synlogic)
                    extra_args=("task_name=$task" "subset=$SYNLOGIC_SUBSET" "split=$SYNLOGIC_SPLIT")
                    output_dir="./outputs/results/synlogic_${SYNLOGIC_SUBSET}_${task}_${SYNLOGIC_SPLIT}_${clean_model}_${timestamp}"
                    ;;
                bigcodebench)
                    extra_args=("bcb_mode=$task" "bcb_subset=$BCB_SUBSET")
                    output_dir="./outputs/results/bigcodebench_${task}_${BCB_SUBSET}_${clean_model}_${timestamp}"
                    ;;
            esac
        fi

        echo "--- Run $current_run/$total_runs | task=$task | model=$model ---"
        echo "    output: $output_dir"

        temp_config=$(create_run_config "$config_file" "$model" "$current_run" "${extra_args[@]}")

        if [ $? -ne 0 ] || [ -z "$temp_config" ]; then
            echo "ERROR: Failed to create run config"
            fail_count=$((fail_count + 1))
            failed_runs+=("$current_run: $task + $model (config error)")
            continue
        fi

        run_cmd="python3 src/main.py --config \"$temp_config\" --output_dir \"$output_dir\""
        [ "$MAX_SAMPLES" != "null" ] && run_cmd="$run_cmd --max_samples $MAX_SAMPLES"

        eval "$run_cmd" 2>&1 | tee "${output_dir}_log.txt"
        exit_code=$?
        rm -f "$temp_config"

        if [ $exit_code -eq 0 ]; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
            failed_runs+=("$current_run: $task + $model")
            echo "ERROR: exit code $exit_code — continuing"
        fi
    done
done

# ============================================
# Summary
# ============================================
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo ""
echo "============================================"
echo "Done: $success_count/$total_runs succeeded, $fail_count failed"
if [ $fail_count -gt 0 ]; then
    echo "Failed runs:"
    for r in "${failed_runs[@]}"; do echo "  - $r"; done
fi
echo "Time: $((total_time/3600))h $(((total_time%3600)/60))m $((total_time%60))s"
echo "============================================"

exit $fail_count
