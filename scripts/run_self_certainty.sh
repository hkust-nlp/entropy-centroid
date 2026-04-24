#!/bin/bash
# ============================================================================
# Self-Certainty 计算流水线
# ============================================================================
#
# 三步流程:
#   Step 1: 数据格式转换 (convert_to_self_certainty.py)
#           entropy_results.json → self_certainty_input.json
#   Step 2: 计算 Self-Certainty 分数 (run_self_certainty_parallel.py)
#           多 GPU 数据并行 forward pass → self_certainty_input-confidence-list.json
#   Step 3: 评估 (evaluate_self_certainty.py) [不需要 GPU]
#           利用 evaluation_cache 查表得 pass@1
#
# 使用方式:
#   # 运行完整三步流水线
#   bash scripts/run_self_certainty.sh bigcodebench
#
#   # 运行 tau2_bench 的完整三步流水线
#   bash scripts/run_self_certainty.sh tau2_bench
#
#   # 默认运行 bigcodebench
#   bash scripts/run_self_certainty.sh
# ============================================================================

set -euo pipefail

# ============================================================================
# 缓存目录配置
# ============================================================================
# 统一设置 HuggingFace / Triton / TorchInductor 缓存目录，避免:
#   1. 默认路径 (~/.cache) 磁盘空间不足
#   2. 8 个 worker 并发下载同一模型导致冲突
#   3. Triton JIT 编译缓存写入系统盘
# MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/data/minimax-dialogue/users/xiaoxian/self-certainty-cache}"
# export HF_HOME="${MODEL_CACHE_ROOT}/huggingface"
# export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
# export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
# export XDG_CACHE_HOME="${MODEL_CACHE_ROOT}/xdg"
# export TRITON_CACHE_DIR="${MODEL_CACHE_ROOT}/triton"
# export TORCHINDUCTOR_CACHE_DIR="${MODEL_CACHE_ROOT}/torchinductor"
# export TMPDIR="${MODEL_CACHE_ROOT}/tmp"

# mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
#     "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"

# echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cache root: ${MODEL_CACHE_ROOT}"

# ============================================================================
# Benchmark 选择
# ============================================================================
BENCHMARK="${1:-bigcodebench}"

# 根据 benchmark 选择 results_root
case "${BENCHMARK}" in
    tau2_bench|tau2-bench)
        RESULTS_ROOT="outputs/tau2_bench"
        BENCHMARK="tau2_bench"
        ;;
    bigcodebench)
        RESULTS_ROOT="outputs/results"
        ;;
    livecodebench)
        RESULTS_ROOT="outputs/results"
        ;;
    *)
        RESULTS_ROOT="outputs/results"
        ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark: ${BENCHMARK}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Results root: ${RESULTS_ROOT}"

# ============================================================================
# Step 1: 数据格式转换
# ============================================================================
# 流式读取 entropy_results.json/jsonl，提取 (prompt/problem, generated_text)，
# 按 original_id 分组，应用 chat template，输出 self_certainty_input.json

python3 scripts/convert_to_self_certainty.py \
      --batch --benchmark "${BENCHMARK}" \
      --results_root "${RESULTS_ROOT}"

# ============================================================================
# Step 2: 计算 Self-Certainty 分数 (多 GPU 并行)
# ============================================================================
# 自动发现所有匹配目录，按模型大小选择:
#   ≤80B: 每张 GPU 加载一份模型，8 路数据并行
#   >80B: device_map="auto" 跨 8 卡部署单份模型

python3 scripts/run_self_certainty_parallel.py \
      --batch --benchmark "${BENCHMARK}" --batch_size 8 \
      --results_root "${RESULTS_ROOT}"

# ============================================================================
# Step 3: 评估 Self-Certainty 选择效果 (不需要 GPU)
# ============================================================================
# 自动发现所有已完成 confidence-list 的目录，
# 结合 evaluation_cache_v2.json 查表计算 pass@1

python3 scripts/evaluate_self_certainty.py \
      --batch --benchmark "${BENCHMARK}" \
      --results_root "${RESULTS_ROOT}" \
      --best_N 1,2,4,8,16,32,64
