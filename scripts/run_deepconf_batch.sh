#!/bin/bash
# Batch run deepconf_select.py on all entropy_results.json files
#
# Usage:
#   bash scripts/run_deepconf_batch.sh
#   bash scripts/run_deepconf_batch.sh --method tail --top_n 1
#   bash scripts/run_deepconf_batch.sh --filter aime   # only process dirs containing "aime"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUTS_DIR="${PROJECT_DIR}/outputs"
SELECT_SCRIPT="${SCRIPT_DIR}/deepconf_select.py"

# Default parameters
METHOD="tail"
TOP_N=1
TAIL_TOKENS=2048
WINDOW_SIZE=2048
FILTER=""
METHODS_ALL=false
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; if [ "$METHOD" = "all" ]; then METHODS_ALL=true; fi; shift 2 ;;
        --top_n) TOP_N="$2"; shift 2 ;;
        --tail_tokens) TAIL_TOKENS="$2"; shift 2 ;;
        --window_size) WINDOW_SIZE="$2"; shift 2 ;;
        --filter) FILTER="$2"; shift 2 ;;
        --all_methods) METHODS_ALL=true; shift ;;
        --force) FORCE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "DeepConf Batch Trajectory Selection"
echo "============================================"
echo "Outputs dir: ${OUTPUTS_DIR}"
echo "Method: ${METHOD}"
echo "Top N: ${TOP_N}"
echo "Filter: ${FILTER:-<none>}"
echo "All methods: ${METHODS_ALL}"
echo ""

# Find all entropy_results.json and entropy_results.jsonl files
# Only search in outputs/results/ and outputs/tau2_bench/
ENTROPY_FILES=$(find "${OUTPUTS_DIR}/results" "${OUTPUTS_DIR}/tau2_bench" \( -name "entropy_results.json" -o -name "entropy_results.jsonl" \) -type f 2>/dev/null | sort)

if [ -z "$ENTROPY_FILES" ]; then
    echo "No entropy_results.json/jsonl files found in ${OUTPUTS_DIR}"
    exit 1
fi

# Filter out .checkpoint files
ENTROPY_FILES=$(echo "$ENTROPY_FILES" | grep -v '\.checkpoint' || true)

# Apply filter if specified
if [ -n "$FILTER" ]; then
    ENTROPY_FILES=$(echo "$ENTROPY_FILES" | grep -i "$FILTER" || true)
    if [ -z "$ENTROPY_FILES" ]; then
        echo "No files match filter: $FILTER"
        exit 1
    fi
fi

# Count files
NUM_FILES=$(echo "$ENTROPY_FILES" | wc -l)
echo "Found ${NUM_FILES} entropy_results.json files to process"
echo ""

# List files with sizes
echo "Files to process:"
echo "$ENTROPY_FILES" | while read f; do
    # Show relative path from outputs/ for clarity
    rel_path=$(echo "$f" | sed "s|${OUTPUTS_DIR}/||")
    dir_path=$(dirname "$rel_path")
    size=$(du -h "$f" | cut -f1)
    echo "  ${size}  ${dir_path}"
done
echo ""

# Process each file
run_one() {
    local input_file="$1"
    local method="$2"
    local dir_name=$(echo "$input_file" | sed "s|${OUTPUTS_DIR}/||" | sed 's|/entropy_results\.json.*||')

    echo "--------------------------------------------"
    echo "[${method}] Processing: ${dir_name}"
    echo "--------------------------------------------"

    local FORCE_FLAG=""
    if [ "$FORCE" = true ]; then
        FORCE_FLAG="--force"
    fi

    python3 "${SELECT_SCRIPT}" \
        --input "$input_file" \
        --method "$method" \
        --top_n "$TOP_N" \
        --tail_tokens "$TAIL_TOKENS" \
        --window_size "$WINDOW_SIZE" \
        $FORCE_FLAG || echo "  FAILED: ${dir_name} (${method})"

    echo ""
}

COUNTER=0
echo "$ENTROPY_FILES" | while read f; do
    COUNTER=$((COUNTER + 1))
    dir_name=$(basename "$(dirname "$f")")
    echo "============================================"
    echo "[${COUNTER}/${NUM_FILES}] ${dir_name}"
    echo "============================================"

    if [ "$METHODS_ALL" = true ]; then
        for m in tail bottom_window min_window mean; do
            run_one "$f" "$m"
        done
    else
        run_one "$f" "$METHOD"
    fi
done

echo ""
echo "============================================"
echo "Batch processing complete!"
echo "============================================"

# Summary: collect all results
echo ""
echo "Summary of all results:"
echo "--------------------------------------------"
find "${OUTPUTS_DIR}/results" "${OUTPUTS_DIR}/tau2_bench" -name "deepconf_selection_*.json" -newer "$0" -type f 2>/dev/null | sort | while read result; do
    rel_dir=$(dirname "$result" | sed "s|${OUTPUTS_DIR}/||")
    file_name=$(basename "$result")
    accuracy=$(python3 -c "import json; d=json.load(open('$result')); print(f\"{d['summary']['correct']}/{d['summary']['total_problems']} = {d['summary']['accuracy']:.1%}\")" 2>/dev/null || echo "N/A")
    echo "  ${rel_dir}/${file_name}: ${accuracy}"
done
