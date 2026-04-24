#!/bin/bash
# Parallel Centroid Cache Builder
#
# Computes trajectory_centroid_cache_*.json for all result directories
# across all dataset types (math/logic/livecodebench/bigcodebench/tau2).
# Purely a cache-building step — does NOT run evaluation.
#
# Each (directory × parameter config) is an independent parallel job.
# Existing caches are skipped unless --force is passed.
#
# Usage:
#   bash scripts/build_centroid_cache.sh                          # all dirs, default sweep
#   bash scripts/build_centroid_cache.sh --dir <path>             # single directory
#   bash scripts/build_centroid_cache.sh --dir <p1> --dir <p2>   # multiple dirs
#   bash scripts/build_centroid_cache.sh --extended               # wider param sweep
#   bash scripts/build_centroid_cache.sh --raw                    # raw_entropy only
#   bash scripts/build_centroid_cache.sh --max_jobs 32            # concurrency limit
#   bash scripts/build_centroid_cache.sh --force                  # recompute existing
#   bash scripts/build_centroid_cache.sh --pattern "outputs/results/*aime*"
#   bash scripts/build_centroid_cache.sh --filter_llm_error       # tau2: skip llm_error trajectories

cd "$(dirname "$0")/.."

if command -v python3 >/dev/null 2>&1; then PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then PYTHON_BIN="python"
else echo "ERROR: python not found" >&2; exit 127; fi

# ============================================
# Configuration
# ============================================
MAX_JOBS=$(nproc 2>/dev/null || echo 8)
SPECIFIED_DIRS=()
FORCE=false
FILTER_LLM_ERROR=false
PATTERN=""

# Default HEP parameter sweep: 3×3×3 = 27 combinations per directory
TOP_PERCENT="1,3,5"
BOTTOM_PERCENT="30,50,80"
CONSECUTIVE_LOW="2,3,5"
METHOD="hep"   # hep | raw_entropy

# Base directories to scan when no --dir is given
DEFAULT_SCAN_DIRS=(
    "outputs/results"
    "outputs/greedy"
    "outputs/tau2_bench"
)

# ============================================
# Parse arguments
# ============================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --extended)
            TOP_PERCENT="1,3,5,10"
            BOTTOM_PERCENT="30,50,70,80"
            CONSECUTIVE_LOW="2,3,5"
            shift ;;
        --raw)
            METHOD="raw_entropy"
            shift ;;
        --max_jobs)
            MAX_JOBS="$2"; shift 2 ;;
        --dir)
            SPECIFIED_DIRS+=("$2"); shift 2 ;;
        --pattern)
            PATTERN="$2"; shift 2 ;;
        --top)
            TOP_PERCENT="$2"; shift 2 ;;
        --bottom)
            BOTTOM_PERCENT="$2"; shift 2 ;;
        --cons)
            CONSECUTIVE_LOW="$2"; shift 2 ;;
        --force)
            FORCE=true; shift ;;
        --filter_llm_error)
            FILTER_LLM_ERROR=true; shift ;;
        -h|--help)
            head -25 "$0"; exit 0 ;;
        *)
            echo "Unknown argument: $1  (run with -h for help)"; exit 1 ;;
    esac
done

# ============================================
# Discover directories
# Two lists are maintained so job dispatch can pick the right code path.
# TAU2_DIRS: entropy_results.jsonl — need prepare_inputs() via evaluate_tau2_bench.py
# REGULAR_DIRS: entropy_results.json — direct stream_compute_centroid_cache()
# ============================================
TAU2_DIRS=()
REGULAR_DIRS=()

add_dir() {
    local d="${1%/}"
    if [ -f "$d/entropy_results.jsonl" ]; then
        TAU2_DIRS+=("$d")
    elif [ -f "$d/entropy_results.json" ]; then
        REGULAR_DIRS+=("$d")
    fi
}

if [ ${#SPECIFIED_DIRS[@]} -gt 0 ]; then
    for spec in "${SPECIFIED_DIRS[@]}"; do
        spec="${spec%/}"
        # Support model-level tau2 dirs: auto-expand to domain subdirs
        if [ -f "$spec/entropy_results.jsonl" ] || [ -f "$spec/entropy_results.json" ]; then
            add_dir "$spec"
        elif [ -d "$spec" ]; then
            for sub in "$spec"/*/; do add_dir "$sub"; done
        fi
    done
elif [ -n "$PATTERN" ]; then
    for d in $PATTERN; do add_dir "$d"; done
else
    for base in "${DEFAULT_SCAN_DIRS[@]}"; do
        [ -d "$base" ] || continue
        if [[ "$base" == *tau2_bench* ]]; then
            # tau2 layout: <base>/<model>/<domain>/
            for d in "$base"/*/*; do add_dir "$d"; done
        else
            for d in "$base"/*/; do add_dir "$d"; done
        fi
    done
fi

DIRS=("${REGULAR_DIRS[@]}" "${TAU2_DIRS[@]}")

if [ ${#DIRS[@]} -eq 0 ]; then
    echo "No result directories found. Use --dir or --pattern to specify them."
    exit 1
fi

# ============================================
# Build job list: (dir × param config)
# ============================================
JOB_DIRS=()
JOB_TOP=()
JOB_BOT=()
JOB_CONS=()
JOB_TYPE=()   # "tau2" | "regular"

if [ "$METHOD" = "raw_entropy" ]; then
    for dir in "${REGULAR_DIRS[@]}"; do
        JOB_DIRS+=("$dir"); JOB_TOP+=("0"); JOB_BOT+=("0"); JOB_CONS+=("0"); JOB_TYPE+=("regular")
    done
    for dir in "${TAU2_DIRS[@]}"; do
        JOB_DIRS+=("$dir"); JOB_TOP+=("0"); JOB_BOT+=("0"); JOB_CONS+=("0"); JOB_TYPE+=("tau2")
    done
    N_COMBOS=1
else
    IFS=',' read -ra TOP_ARR  <<< "$TOP_PERCENT"
    IFS=',' read -ra BOT_ARR  <<< "$BOTTOM_PERCENT"
    IFS=',' read -ra CONS_ARR <<< "$CONSECUTIVE_LOW"
    N_COMBOS=$(( ${#TOP_ARR[@]} * ${#BOT_ARR[@]} * ${#CONS_ARR[@]} ))

    for dir in "${REGULAR_DIRS[@]}"; do
        for top in "${TOP_ARR[@]}"; do
            for bot in "${BOT_ARR[@]}"; do
                for cons in "${CONS_ARR[@]}"; do
                    JOB_DIRS+=("$dir"); JOB_TOP+=("$top"); JOB_BOT+=("$bot"); JOB_CONS+=("$cons"); JOB_TYPE+=("regular")
                done
            done
        done
    done
    for dir in "${TAU2_DIRS[@]}"; do
        for top in "${TOP_ARR[@]}"; do
            for bot in "${BOT_ARR[@]}"; do
                for cons in "${CONS_ARR[@]}"; do
                    JOB_DIRS+=("$dir"); JOB_TOP+=("$top"); JOB_BOT+=("$bot"); JOB_CONS+=("$cons"); JOB_TYPE+=("tau2")
                done
            done
        done
    done
fi

TOTAL_JOBS=${#JOB_DIRS[@]}

echo "============================================"
echo "Centroid Cache Builder"
echo "============================================"
echo "Regular dirs (json):  ${#REGULAR_DIRS[@]}"
echo "tau2-bench dirs (jsonl): ${#TAU2_DIRS[@]}"
for d in "${DIRS[@]}"; do echo "  - $(basename "$d")"; done
echo ""
echo "Method: $METHOD"
[ "$METHOD" = "hep" ] && echo "Params: top=[${TOP_PERCENT}], bottom=[${BOTTOM_PERCENT}], cons=[${CONSECUTIVE_LOW}]"
echo "Combinations per dir: $N_COMBOS"
echo "Total jobs: $TOTAL_JOBS"
[ "$MAX_JOBS" -eq 0 ] && echo "Concurrency: unlimited" || echo "Concurrency: $MAX_JOBS"
[ "$FORCE" = true ] && echo "Force: yes (recomputing existing caches)"
[ "$FILTER_LLM_ERROR" = true ] && echo "Filter llm_error: yes (tau2 only)"
echo "============================================"
echo ""

# ============================================
# Python helper: compute one cache
# Calls orchestrator.ensure_centroid_cache() directly.
# Exits 0 on success (including cache-already-exists),
# exits 1 on failure.
# ============================================
run_cache_job() {
    local dir="$1" top="$2" bot="$3" cons="$4" force="$5" method="$6"

    $PYTHON_BIN - "$dir" "$top" "$bot" "$cons" "$force" "$method" << 'PYEOF'
import sys, os

result_dir   = sys.argv[1]
top_percent  = float(sys.argv[2])
bottom_pct   = float(sys.argv[3])
cons_low     = int(sys.argv[4])
force        = sys.argv[5].lower() == "true"
method       = sys.argv[6]

# Locate project src
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from centroid.io import (
        get_trajectory_centroid_cache_path,
        load_trajectory_centroid_cache,
        stream_compute_centroid_cache,
    )
except ImportError as e:
    print(f"ERROR: cannot import centroid.io: {e}", file=sys.stderr)
    sys.exit(1)

cache_path = get_trajectory_centroid_cache_path(
    result_dir=result_dir,
    top_percent=top_percent,
    bottom_percent=bottom_pct,
    consecutive_low_threshold=cons_low,
    centroid_method=method,
)

if not force and os.path.exists(cache_path):
    existing = load_trajectory_centroid_cache(
        result_dir=result_dir,
        top_percent=top_percent,
        bottom_percent=bottom_pct,
        consecutive_low_threshold=cons_low,
        centroid_method=method,
    )
    if existing:
        print(f"SKIP (exists): {os.path.basename(cache_path)}")
        sys.exit(0)

print(f"Computing: {os.path.basename(cache_path)}")
result = stream_compute_centroid_cache(
    result_dir=result_dir,
    top_percent=top_percent,
    bottom_percent=bottom_pct,
    consecutive_low_threshold=cons_low,
    centroid_method=method,
)
if result:
    print(f"OK: {os.path.basename(cache_path)}")
    sys.exit(0)
else:
    print(f"FAILED: {os.path.basename(cache_path)}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

export -f run_cache_job
export PYTHON_BIN

# ============================================
# Launch parallel jobs with throttling
# ============================================
LOG_DIR="outputs/logs/centroid_cache"
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
    jtype="${JOB_TYPE[$i]}"
    name=$(basename "$dir")

    if [ "$METHOD" = "raw_entropy" ]; then
        label="${name}__raw_entropy"
    else
        label="${name}__top${top}_bot${bot}_cons${cons}"
    fi

    log_file="$LOG_DIR/${label}.log"

    if [ "$jtype" = "tau2" ]; then
        # tau2: must go through evaluate_tau2_bench.py so prepare_inputs() runs first
        tau2_flags="--result_dir $dir --step cache,centroid"
        [ "$FORCE" = true ]            && tau2_flags="$tau2_flags --force"
        [ "$FILTER_LLM_ERROR" = true ] && tau2_flags="$tau2_flags --filter_llm_error"
        if [ "$METHOD" = "raw_entropy" ]; then
            tau2_flags="$tau2_flags --centroid_method raw_entropy"
        else
            tau2_flags="$tau2_flags --centroid_top_percent $top --centroid_bottom_percent $bot --centroid_consecutive_low $cons --centroid_method $METHOD"
        fi
        $PYTHON_BIN evaluate_tau2_bench.py $tau2_flags > "$log_file" 2>&1 &
    else
        # regular (math/logic/code): call stream_compute_centroid_cache() directly
        run_cache_job "$dir" "$top" "$bot" "$cons" "$FORCE" "$METHOD" > "$log_file" 2>&1 &
    fi

    PIDS+=($!)
    JOB_LABELS+=("$label")
    running=$((running + 1))

    launched=$((i + 1))
    if (( launched % 20 == 0 )) || (( launched == TOTAL_JOBS )); then
        echo "  Launched $launched/$TOTAL_JOBS jobs..."
    fi

    if [ "$MAX_JOBS" -gt 0 ] && [ "$running" -ge "$MAX_JOBS" ]; then
        wait -n 2>/dev/null || true
        running=$((running - 1))
    fi
done

echo ""
echo "All $TOTAL_JOBS jobs launched. Waiting for completion..."
echo ""

# ============================================
# Collect results
# ============================================
success=0; fail=0

for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        success=$((success + 1))
    else
        echo "  FAILED: ${JOB_LABELS[$i]}"
        fail=$((fail + 1))
    fi

    done_count=$((success + fail))
    if (( done_count % 20 == 0 )) || (( done_count == TOTAL_JOBS )); then
        elapsed=$(( $(date +%s) - start_time ))
        echo "  Progress: $done_count/$TOTAL_JOBS done (${elapsed}s)"
    fi
done

end_time=$(date +%s); elapsed=$((end_time - start_time))

echo ""
echo "============================================"
echo "Done: $success/$TOTAL_JOBS succeeded, $fail failed"
echo "Time: $((elapsed/60))m $((elapsed%60))s"
echo ""
echo "Cache files per directory:"
for dir in "${DIRS[@]}"; do
    n=$(ls "$dir"/trajectory_centroid_cache_*.json 2>/dev/null | wc -l)
    echo "  $(basename "$dir"): $n cache file(s)"
done
echo "============================================"
[ "$fail" -gt 0 ] && echo "Check logs in: $LOG_DIR/"
exit $fail
