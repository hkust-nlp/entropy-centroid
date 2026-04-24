#!/bin/bash
# tau2-bench Centroid Parallel Computation Script
#
# Parallelizes centroid computation at the (directory × parameter config) level.
# Supports both domain-level and model-level directories:
#   --dir outputs/tau2_bench/Qwen_QwQ-32B/airline/   (single domain)
#   --dir outputs/tau2_bench/Qwen_QwQ-32B/           (auto-expand to all domains)
#
# Usage:
#   ./scripts/eval_tau2_centroid_parallel.sh                         # All dirs under outputs/tau2_bench
#   ./scripts/eval_tau2_centroid_parallel.sh --dir <path>            # Single dir (model or domain level)
#   ./scripts/eval_tau2_centroid_parallel.sh --dir <p1> --dir <p2>   # Multiple dirs
#   ./scripts/eval_tau2_centroid_parallel.sh --max_jobs 16           # Limit concurrent jobs
#   ./scripts/eval_tau2_centroid_parallel.sh --extended              # Extended parameter sweep
#   ./scripts/eval_tau2_centroid_parallel.sh --raw                   # raw_entropy method
#   ./scripts/eval_tau2_centroid_parallel.sh --plan                  # Use tau2_bench_plan outputs
#   ./scripts/eval_tau2_centroid_parallel.sh --step cache,centroid   # Run specific steps

# 默认不传 --force，已存在的 cache 会自动跳过。 并行脚本中 FORCE="" 是默认值，只有显式传 --force 才会设为 "--force"


set -e
cd "$(dirname "$0")/.."

# Python interpreter selection
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "ERROR: Neither python3 nor python is available in PATH." >&2
    exit 127
fi

# ============================================
# Configuration
# ============================================
BASE_DIR="outputs/tau2_bench"
MAX_JOBS=$(nproc 2>/dev/null || echo 8)
MODE="default"
SPECIFIED_DIRS=()
STEP="cache,centroid"
FORCE=""
FILTER_LLM_ERROR=""
METHOD="hep"

# Default parameter sweep: 3×3×3 = 27 combinations
TOP_PERCENT="1,3,5"
BOTTOM_PERCENT="30,50,80"
CONSECUTIVE_LOW="2,3,5"

# ============================================
# Parse arguments
# ============================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --extended)
            MODE="extended"
            TOP_PERCENT="1,3,5,10"
            BOTTOM_PERCENT="30,50,70,80"
            CONSECUTIVE_LOW="2,3,5"
            shift
            ;;
        --raw)
            MODE="raw"
            METHOD="raw_entropy"
            shift
            ;;
        --plan)
            BASE_DIR="outputs/tau2_bench_plan"
            shift
            ;;
        --max_jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --dir)
            SPECIFIED_DIRS+=("$2")
            shift 2
            ;;
        --top)
            TOP_PERCENT="$2"
            shift 2
            ;;
        --bottom)
            BOTTOM_PERCENT="$2"
            shift 2
            ;;
        --cons)
            CONSECUTIVE_LOW="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --force)
            FORCE="--force"
            shift
            ;;
        --filter_llm_error)
            FILTER_LLM_ERROR="--filter_llm_error"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dir <path>         Directory (repeatable). Accepts model-level or domain-level."
            echo "                       Model-level dirs auto-expand to domain subdirs."
            echo "  --plan               Use outputs/tau2_bench_plan instead of outputs/tau2_bench"
            echo "  --max_jobs N         Max concurrent jobs (default: nproc=$(nproc 2>/dev/null || echo 8))"
            echo "                       0 = unlimited (launch all at once)"
            echo "  --step <steps>       Pipeline steps: cache, centroid, all (default: cache,centroid)"
            echo "  --top X              Top percent, comma-separated (default: 1,3,5)"
            echo "  --bottom X           Bottom percent, comma-separated (default: 30,50,80)"
            echo "  --cons X             Consecutive low, comma-separated (default: 2,3,5)"
            echo "  --extended           Extended sweep: top=1,3,5,10 bottom=30,50,70,80 cons=2,3,5"
            echo "  --raw                Use raw_entropy method (no parameter sweep)"
            echo "  --force              Force recomputation"
            echo "  --filter_llm_error   Filter llm_error trajectories"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run with -h for help"
            exit 1
            ;;
    esac
done

# ============================================
# Resolve directories
# ============================================
# Given a path, expand to domain-level directories containing entropy_results.jsonl.
# - If the dir itself has entropy_results.jsonl → use it directly
# - Otherwise scan subdirs (model-level → domain-level expansion)
expand_dir() {
    local dir="${1%/}"  # strip trailing slash to avoid double-slash
    if [ -f "$dir/entropy_results.jsonl" ]; then
        echo "$dir"
    elif [ -d "$dir" ]; then
        for sub in "$dir"/*/; do
            [ -f "${sub}entropy_results.jsonl" ] && echo "${sub%/}"
        done
    fi
}

DIRS=()

if [ ${#SPECIFIED_DIRS[@]} -gt 0 ]; then
    for spec in "${SPECIFIED_DIRS[@]}"; do
        while IFS= read -r d; do
            [ -n "$d" ] && DIRS+=("$d")
        done < <(expand_dir "$spec")
    done
else
    # Auto-discover: outputs/tau2_bench/<model>/<domain>/
    if [ -d "$BASE_DIR" ]; then
        for model_dir in "$BASE_DIR"/*/; do
            [ -d "$model_dir" ] || continue
            for domain_dir in "${model_dir%/}"/*/; do
                [ -f "${domain_dir}entropy_results.jsonl" ] && DIRS+=("${domain_dir%/}")
            done
        done
    fi
fi

if [ ${#DIRS[@]} -eq 0 ]; then
    echo "No tau2-bench result directories found"
    echo "  Searched: $BASE_DIR/<model>/<domain>/entropy_results.jsonl"
    exit 1
fi

# ============================================
# Build all (dir, top, bottom, cons) jobs
# ============================================
IFS=',' read -ra TOP_ARR <<< "$TOP_PERCENT"
IFS=',' read -ra BOT_ARR <<< "$BOTTOM_PERCENT"
IFS=',' read -ra CONS_ARR <<< "$CONSECUTIVE_LOW"

JOB_DIRS=()
JOB_TOP=()
JOB_BOT=()
JOB_CONS=()

if [ "$METHOD" = "raw_entropy" ]; then
    for dir in "${DIRS[@]}"; do
        JOB_DIRS+=("$dir")
        JOB_TOP+=("0")
        JOB_BOT+=("0")
        JOB_CONS+=("0")
    done
else
    for dir in "${DIRS[@]}"; do
        for top in "${TOP_ARR[@]}"; do
            for bot in "${BOT_ARR[@]}"; do
                for cons in "${CONS_ARR[@]}"; do
                    JOB_DIRS+=("$dir")
                    JOB_TOP+=("$top")
                    JOB_BOT+=("$bot")
                    JOB_CONS+=("$cons")
                done
            done
        done
    done
fi

TOTAL_JOBS=${#JOB_DIRS[@]}

if [ "$METHOD" = "raw_entropy" ]; then
    N_COMBOS=1
else
    N_COMBOS=$(( ${#TOP_ARR[@]} * ${#BOT_ARR[@]} * ${#CONS_ARR[@]} ))
fi

echo "============================================"
echo "tau2-bench Centroid Parallel Computation"
echo "============================================"
echo "Base directory: $BASE_DIR"
echo "Directories: ${#DIRS[@]}"
for dir in "${DIRS[@]}"; do
    # Show model/domain relative path
    rel="${dir#$BASE_DIR/}"
    echo "  - $rel"
done
echo ""
echo "Method: $METHOD"
echo "Steps: $STEP"
if [ "$METHOD" = "hep" ]; then
    echo "Parameters: top=[${TOP_PERCENT}], bottom=[${BOTTOM_PERCENT}], cons=[${CONSECUTIVE_LOW}]"
    echo "Combinations per dir: $N_COMBOS"
fi
echo "Total jobs: $TOTAL_JOBS  (${#DIRS[@]} dirs × $N_COMBOS configs)"
if [ "$MAX_JOBS" -eq 0 ]; then
    echo "Max concurrent: unlimited"
else
    echo "Max concurrent: $MAX_JOBS"
fi
[ -n "$FORCE" ] && echo "Force: yes"
[ -n "$FILTER_LLM_ERROR" ] && echo "Filter llm_error: yes"
echo "============================================"
echo ""

# ============================================
# Launch parallel jobs
# ============================================
LOG_DIR="outputs/logs/tau2_centroid"
mkdir -p "$LOG_DIR"

PIDS=()
JOB_LABELS=()
start_time=$(date +%s)

running=0

for i in $(seq 0 $((TOTAL_JOBS - 1))); do
    dir="${JOB_DIRS[$i]}"
    top="${JOB_TOP[$i]}"
    bot="${JOB_BOT[$i]}"
    cons="${JOB_CONS[$i]}"
    # Label: model_domain__params
    rel="${dir#$BASE_DIR/}"
    name="${rel//\//_}"

    # Build common flags
    COMMON_FLAGS="--result_dir $dir --step $STEP $FORCE $FILTER_LLM_ERROR"

    if [ "$METHOD" = "raw_entropy" ]; then
        label="${name}__raw_entropy"
        log_file="$LOG_DIR/${name}_raw_entropy_$(date +%Y%m%d_%H%M%S).log"

        $PYTHON_BIN evaluate_tau2_bench.py \
            $COMMON_FLAGS \
            --centroid_method raw_entropy \
            > "$log_file" 2>&1 &
    else
        label="${name}__top${top}_bot${bot}_cons${cons}"
        log_file="$LOG_DIR/${name}_top${top}_bot${bot}_cons${cons}.log"

        $PYTHON_BIN evaluate_tau2_bench.py \
            $COMMON_FLAGS \
            --centroid_top_percent "$top" \
            --centroid_bottom_percent "$bot" \
            --centroid_consecutive_low "$cons" \
            --centroid_method "$METHOD" \
            > "$log_file" 2>&1 &
    fi

    PIDS+=($!)
    JOB_LABELS+=("$label")
    running=$((running + 1))

    # Progress
    launched=$((i + 1))
    if (( launched % 10 == 0 )) || (( launched == TOTAL_JOBS )); then
        echo "  Launched $launched/$TOTAL_JOBS jobs..."
    fi

    # Throttle
    if [ "$MAX_JOBS" -gt 0 ] && [ "$running" -ge "$MAX_JOBS" ]; then
        wait -n 2>/dev/null || true
        running=$((running - 1))
    fi
done

echo ""
echo "All $TOTAL_JOBS jobs launched. Waiting for completion..."
echo ""

# ============================================
# Wait and collect results
# ============================================
success=0
fail=0

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    label=${JOB_LABELS[$i]}

    if wait "$pid"; then
        success=$((success + 1))
    else
        echo "  ✗ FAILED: $label (pid $pid)"
        fail=$((fail + 1))
    fi

    done_count=$((success + fail))
    if (( done_count % 20 == 0 )) || (( done_count == TOTAL_JOBS )); then
        elapsed=$(( $(date +%s) - start_time ))
        echo "  Progress: $done_count/$TOTAL_JOBS done (${elapsed}s elapsed)"
    fi
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo ""
echo "============================================"
echo "tau2-bench Centroid Parallel Complete!"
echo "============================================"
echo "  Total jobs: $TOTAL_JOBS"
echo "  Succeeded:  $success"
echo "  Failed:     $fail"
echo "  Time:       ${minutes}m ${seconds}s"
echo ""

# List generated cache files per directory
echo "Generated cache files:"
for dir in "${DIRS[@]}"; do
    rel="${dir#$BASE_DIR/}"
    caches=$(ls "$dir"/trajectory_centroid_cache_*.json 2>/dev/null | wc -l)
    if [ "$caches" -gt 0 ]; then
        echo "  $rel: $caches cache files"
    else
        echo "  $rel: no cache files"
    fi
done

echo ""
echo "============================================"
if [ "$fail" -gt 0 ]; then
    echo "Check logs in: $LOG_DIR/"
fi
exit $fail
