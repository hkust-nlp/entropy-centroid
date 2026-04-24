<div align="center">

# Entropy Centroids as Intrinsic Rewards for Test-Time Scaling

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/TODO)

</div>

<div align="center">
<img src="figures/Entropy Centroid.pdf" width="700" alt="centroid-intro-figure_00">
</div>

## Table of Contents

- [Introduction](#introduction)
- [News](#news)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Centroid Cache](#centroid-cache)
- [Output Structure](#output-structure)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Introduction

Centroid is a training-free test-time compute method that selects the best trajectory from N sampled responses using token-level entropy dynamics. Instead of relying on reward models or majority voting, it identifies the trajectory whose entropy profile is closest to the "centroid" of the distribution — a proxy for the most confident and consistent reasoning path.

**Key features:**

- Runs on top of any vLLM-compatible model — no fine-tuning required
- Supports math, logic (KOR-Bench, SynLogic), code (LiveCodeBench, BigCodeBench), and agent (tau2-bench) benchmarks
- Parallel centroid cache computation across parameter sweeps
- Unified evaluation pipeline for all dataset types

## News

- **[2026/04]** Initial release of inference scripts, evaluation pipeline.

## How It Works

1. **Inference** — sample N trajectories per problem, recording per-token Top 10 log-probabilities
2. **Centroid cache** — compute a scalar centroid score per trajectory from High Entropy Phase (HEP)
3. **Evaluation** — pick the trajectory with the lowest centroid score as the final answer and score the selected answer against ground truth.

## Installation

```bash
pip install -r requirements.txt
```

For tau2-bench (agent inference), install the submodule separately:

```bash
cd tau2-bench && pip install -e .
```

## Inference

### Math / Logic / Code

Config files live in `scripts/configs/`. Set `DATASET_TYPE` at the top of `run_batch.sh`, then run:

```bash
bash scripts/run_batch.sh
```

Supported dataset types:


| `DATASET_TYPE`  | Datasets                                                     | Config                      |
| --------------- | ------------------------------------------------------------ | --------------------------- |
| `aime`          | AIME 2025                                                    | `config_aime_2025yaml`      |
| `minerva`       | Minerva                                                      | `config_minerva.yaml`       |
| `livecodebench` | LiveCodeBench                                                | `config_livecodebench.yaml` |
| `bigcodebench`  | BigCodeBench                                                 | `config_bigcodebench.yaml`  |
| `synlogic`      | SynLogic                                                     | `config_synlogic.yaml`      |


Key parameters in `run_batch.sh`:

```bash
DATASET_TYPE="math"
INFERENCE_MODELS=("Qwen/QwQ-32B")
GPU_IDS="0,1,2,3,4,5,6,7"
TENSOR_PARALLEL_SIZE=8
TRAJECTORIES_PER_SAMPLE=64   # N in Best-of-N
TEMPERATURE=0.7
MAX_TOKENS=32768
```


### tau2-bench (Agent Inference)

tau2-bench uses a separate entry point (`tau2.cli`) with a local vLLM server:

```bash
bash scripts/run_tau2_bench.sh

# Options:
bash scripts/run_tau2_bench.sh --model-idx 0      # single model by index
bash scripts/run_tau2_bench.sh --domain retail    # single domain
bash scripts/run_tau2_bench.sh --skip-existing    # resume interrupted runs
bash scripts/run_tau2_bench.sh --dry-run          # 2 tasks per domain only
```

Output: `outputs/tau2_bench/<model>/{airline,retail,telecom}/entropy_results.jsonl`

### Greedy Baseline

```bash
bash scripts/run_greedy_baseline.sh

# Scope to specific benchmark groups:
BENCHMARKS="math"  bash scripts/run_greedy_baseline.sh
BENCHMARKS="logic" bash scripts/run_greedy_baseline.sh
BENCHMARKS="code"  bash scripts/run_greedy_baseline.sh

# Evaluate existing results without re-running inference:
EVAL_ONLY=true bash scripts/run_greedy_baseline.sh
```

## Evaluation

### Math / Logic

```bash
# Lowest-centroid selection (main method)
python evaluate_answers.py --result_dir <RESULT_DIR> --selection lowest_centroid

# Batch over multiple directories
python evaluate_unified.py --batch --pattern "outputs/results/config_aime*"

# Other selection methods
python evaluate_answers.py --result_dir <RESULT_DIR> --selection majority_voting
python evaluate_answers.py --result_dir <RESULT_DIR> --selection pass_at_k --pass_k 1,5,10,32
```

### LiveCodeBench

```bash
python evaluate_livecodebench.py --result_dir <RESULT_DIR>
python evaluate_livecodebench.py --batch --pattern "outputs/results/*livecodebench*"
```

### BigCodeBench

```bash
python evaluate_bigcodebench.py --result_dir <RESULT_DIR>
python evaluate_bigcodebench.py --batch --pattern "outputs/results/bigcodebench_*"
```

### tau2-bench

```bash
python evaluate_tau2_bench.py --result_dir <RESULT_DIR>
python evaluate_tau2_bench.py --batch --batch_base_dir outputs/tau2_bench
```

### Centroid Parameters

All evaluators accept the following centroid parameters:

```bash
--centroid_top_percent 1.0       # fraction of top-entropy tokens used as step boundaries
--centroid_bottom_percent 80.0   # fraction of low-entropy tokens used as centroid anchors
--centroid_consecutive_low 2     # consecutive low-entropy token threshold
--centroid_method hep            # hep | raw_entropy
--force                          # recompute existing caches
```

## Centroid Cache

The centroid cache (`trajectory_centroid_cache_*.json`) is built automatically during evaluation. To pre-build it in parallel across a parameter sweep (useful before running large-scale evaluations):

```bash
# Default sweep: top=[1,3,5] × bottom=[30,50,80] × cons=[2,3,5] = 27 combos per dir
bash scripts/build_centroid_cache.sh

# Scope to specific directories
bash scripts/build_centroid_cache.sh --dir outputs/results/config_aime_2025_QwQ-32B_*/

# Extended sweep
bash scripts/build_centroid_cache.sh --extended

# Options
bash scripts/build_centroid_cache.sh --raw                # raw_entropy method only
bash scripts/build_centroid_cache.sh --force              # recompute existing caches
bash scripts/build_centroid_cache.sh --max_jobs 32        # concurrency limit
bash scripts/build_centroid_cache.sh --filter_llm_error   # tau2: skip llm_error trajectories
```

tau2-bench directories (containing `entropy_results.jsonl`) are detected automatically and routed through the appropriate preparation pipeline.

## Output Structure

```
outputs/
├── results/                          # math / logic / code inference
│   └── <dataset>_<model>_<ts>/
│       ├── entropy_results.json
│       ├── evaluation_cache.json
│       ├── trajectory_centroid_cache_*.json
│       └── eval_lowest_centroid/
│           ├── answer_evaluation_summary.json
│           └── trajectory_selections.json
├── greedy/                           # greedy baseline
│   └── greedy_<dataset>_<model>_<ts>/
├── tau2_bench/                       # tau2-bench agent inference
│   └── <model>/
│       └── {airline,retail,telecom}/
│           └── entropy_results.jsonl
└── logs/
```


## Citation

If you find this work useful, please cite:

```bibtex


```

## Acknowledgement

We use [vLLM](https://github.com/vllm-project/vllm) for inference, [tau2-bench](https://github.com/sierra-research/tau2-bench) for agent loop, [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), [BigCodeBench](https://github.com/bigcode-project/bigcodebench) and [SynLogic](https://github.com/MiniMaxAI/SynLogic) for benchmarking.