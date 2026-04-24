"""
Main entry point for the step-wise token entropy analysis framework.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Set
import yaml

from data.dataset_loader import create_dataset_loader
from inference.entropy_calculator import create_entropy_calculator
from inference.vllm_engine import create_vllm_engine
from utils.gpu_manager import create_gpu_manager
from utils.logger import setup_logger
from utils.statistics import create_statistics_analyzer
from utils.step_divider import create_step_divider


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def nullable_int(value):
    """
    Convert a string to int, or None if value is 'null'.

    Args:
        value: String value to convert

    Returns:
        Integer or None
    """
    if value is None or value.lower() == 'null':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Step-wise Token Entropy Analysis Framework"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path or HuggingFace model ID (overrides config)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (overrides config)",
    )
    parser.add_argument(
        "--max_samples",
        type=nullable_int,
        default=None,
        help="Maximum number of samples to process (overrides config). Use 'null' to process all samples.",
    )

    # GPU configuration
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,2,3') (overrides config)",
    )

    # Inference configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference (overrides config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (overrides config)",
    )

    # Entropy configuration
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k for entropy calculation (overrides config)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    return parser.parse_args()


def merge_args_with_config(args, config: Dict) -> Dict:
    """
    Merge command line arguments with configuration.

    Args:
        args: Parsed command line arguments
        config: Configuration dictionary

    Returns:
        Merged configuration
    """
    # Model configuration
    if args.model_path is not None:
        config["model"]["name_or_path"] = args.model_path

    # Dataset configuration
    if args.dataset is not None:
        # For math datasets, set name; for logic datasets, this is ignored
        dataset_type = config["dataset"].get("type", "math")
        if dataset_type in ["korbench", "synlogic"]:
            # For logic tasks, --dataset can be used to override task/category
            if dataset_type == "korbench":
                config["dataset"].setdefault("korbench", {})["category"] = args.dataset
            elif dataset_type == "synlogic":
                config["dataset"].setdefault("synlogic", {})["task_name"] = args.dataset
        else:
            config["dataset"]["name"] = args.dataset
    if args.max_samples is not None:
        config["dataset"]["max_samples"] = args.max_samples

    # GPU configuration
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        config["gpu"]["device_ids"] = gpu_ids
        config["gpu"]["tensor_parallel_size"] = len(gpu_ids)

    # Inference configuration
    if args.batch_size is not None:
        config["inference"]["batch_size"] = args.batch_size
    if args.temperature is not None:
        config["inference"]["temperature"] = args.temperature
    if args.max_tokens is not None:
        config["inference"]["max_tokens"] = args.max_tokens

    # Entropy configuration
    if args.top_k is not None:
        config["entropy"]["top_k"] = args.top_k

    # Output configuration
    # Store --output_dir separately so it doesn't break checkpoint/resume.
    # get_deterministic_output_dir always uses the config-level save_dir,
    # not the CLI-provided timestamped path.
    if args.output_dir is not None:
        config["output"]["cli_output_dir"] = args.output_dir

    return config


def create_output_directory(config: Dict, timestamp: str) -> str:
    """
    Create output directory with model name, dataset name, and timestamp.

    Supports both math datasets (HuggingFace) and logic datasets (KOR-Bench, SynLogic).

    Args:
        config: Configuration dictionary
        timestamp: Timestamp string

    Returns:
        Output directory path
    """
    # Extract model name (remove path separators and special chars)
    model_path = config.get("model", {}).get("name_or_path", "unknown_model")
    model_name = model_path.replace("/", "_").replace("\\", "_").replace(":", "_")

    # Extract dataset name based on dataset type
    dataset_config = config.get("dataset", {})
    dataset_type = dataset_config.get("type", "math")

    if dataset_type == "korbench":
        # KOR-Bench dataset: use category for naming
        kb_config = dataset_config.get("korbench", {})
        category = kb_config.get("category", "all")
        mode = kb_config.get("mode", "zero-shot")
        # Handle category being a list
        if isinstance(category, list):
            category = "_".join(category)
        dataset_name = f"korbench_{category}_{mode}".replace("-", "_")
    elif dataset_type == "synlogic":
        # SynLogic dataset: use task_name for naming
        sl_config = dataset_config.get("synlogic", {})
        task_name = sl_config.get("task_name", "all") or "all"
        split = dataset_config.get("split", "validation")
        dataset_name = f"synlogic_{task_name}_{split}"
    elif dataset_type == "bigcodebench":
        # BigCodeBench dataset: use mode for naming
        bcb_config = dataset_config.get("bigcodebench", {})
        mode = bcb_config.get("mode", "instruct")
        subset = bcb_config.get("subset", "v0.1.2")
        dataset_name = f"bigcodebench_{mode}_{subset}".replace(".", "_")
    elif dataset_type == "livecodebench":
        # LiveCodeBench dataset: use release version for naming
        lcb_config = dataset_config.get("livecodebench", {})
        release_version = lcb_config.get("release_version", "release_v5")
        dataset_name = f"livecodebench_{release_version}"
    else:
        # Math datasets (HuggingFace): use dataset name
        dataset_path = dataset_config.get("name", "unknown_dataset")
        dataset_name = dataset_path.replace("/", "_").replace("\\", "_").replace(":", "_")

    # Create directory name: model_dataset_timestamp
    dir_name = f"{model_name}_{dataset_name}_{timestamp}"

    # Full path
    base_dir = config.get("output", {}).get("save_dir", "./outputs/results")
    output_dir = os.path.join(base_dir, dir_name)

    return output_dir


def get_deterministic_output_dir(config: Dict) -> str:
    """
    Get a deterministic output directory path based on model + dataset + inference params.

    This is used for checkpoint/resume: the same model + dataset combination
    always maps to the same directory, so we can detect and resume interrupted runs.

    IMPORTANT: always uses config-level save_dir (from YAML), NOT the CLI
    --output_dir (which typically contains a timestamp and would produce a
    different path on every invocation, breaking checkpoint/resume).

    Args:
        config: Configuration dictionary

    Returns:
        Deterministic output directory path
    """
    model_path = config.get("model", {}).get("name_or_path", "unknown_model")
    model_name = model_path.replace("/", "_").replace("\\", "_").replace(":", "_")

    dataset_config = config.get("dataset", {})
    dataset_type = dataset_config.get("type", "math")

    if dataset_type == "korbench":
        kb_config = dataset_config.get("korbench", {})
        category = kb_config.get("category", "all")
        mode = kb_config.get("mode", "zero-shot")
        if isinstance(category, list):
            category = "_".join(category)
        dataset_name = f"korbench_{category}_{mode}".replace("-", "_")
    elif dataset_type == "synlogic":
        sl_config = dataset_config.get("synlogic", {})
        task_name = sl_config.get("task_name", "all") or "all"
        split = dataset_config.get("split", "validation")
        dataset_name = f"synlogic_{task_name}_{split}"
    elif dataset_type == "bigcodebench":
        bcb_config = dataset_config.get("bigcodebench", {})
        mode = bcb_config.get("mode", "instruct")
        subset = bcb_config.get("subset", "v0.1.2")
        dataset_name = f"bigcodebench_{mode}_{subset}".replace(".", "_")
    elif dataset_type == "livecodebench":
        lcb_config = dataset_config.get("livecodebench", {})
        release_version = lcb_config.get("release_version", "release_v5")
        dataset_name = f"livecodebench_{release_version}"
    else:
        dataset_path = dataset_config.get("name", "unknown_dataset")
        dataset_name = dataset_path.replace("/", "_").replace("\\", "_").replace(":", "_")

    # Include n_trajectories in directory name to differentiate runs
    inference_config = config.get("inference", {})
    n_trajectories = inference_config.get("n", 1)
    n_suffix = f"_n{n_trajectories}" if n_trajectories > 1 else ""

    dir_name = f"{model_name}_{dataset_name}{n_suffix}"

    # Always use the YAML-level save_dir, ignoring --output_dir from CLI
    base_dir = config.get("output", {}).get("save_dir", "./outputs/results")
    return os.path.join(base_dir, dir_name)


# ==================== Streaming JSONL Writer ====================

def _checkpoint_path_for(jsonl_path: str) -> str:
    """Return the path of the lightweight checkpoint file for a given JSONL path."""
    return jsonl_path + ".checkpoint"


class StreamingResultWriter:
    """
    Writes inference results to a JSONL file incrementally (one JSON object per line).

    Also maintains a lightweight checkpoint file (one ID per line) so that resume
    can determine which trajectories have been completed without parsing the
    (potentially 100+ GB) JSONL file.
    """

    def __init__(self, jsonl_path: str, append: bool = False):
        """
        Args:
            jsonl_path: Path to the output JSONL file
            append: If True, append to existing file (for resume). Otherwise truncate.
        """
        self.jsonl_path = jsonl_path
        self.checkpoint_path = _checkpoint_path_for(jsonl_path)
        self.count = 0
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        mode = "a" if append else "w"
        self._file = open(jsonl_path, mode, encoding="utf-8")
        self._ckpt_file = open(self.checkpoint_path, mode, encoding="utf-8")

    def write_batch(self, batch_results: List[Dict]):
        """Write a batch of results, one per line. Flush after each batch."""
        for result in batch_results:
            line = json.dumps(result, ensure_ascii=False)
            self._file.write(line + "\n")
            self.count += 1
        self._file.flush()
        # Update checkpoint: append IDs only (one per line, tiny file)
        for result in batch_results:
            self._ckpt_file.write(str(result["id"]) + "\n")
        self._ckpt_file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()
        if self._ckpt_file and not self._ckpt_file.closed:
            self._ckpt_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_completed_ids(jsonl_path: str) -> Set[str]:
    """
    Load the set of already-completed trajectory IDs from the checkpoint file.

    The checkpoint file stores one ID per line and is typically only a few KB,
    so loading is near-instant regardless of how large the JSONL file is.

    Falls back to scanning the JSONL file if the checkpoint file does not exist
    but the JSONL does — this handles data produced by the old code (before
    checkpoint files were introduced). The fallback uses binary chunk-based
    reading with regex so it never loads a full 10+ MB line into memory.

    Args:
        jsonl_path: Path to the JSONL file (checkpoint path is derived from it)

    Returns:
        Set of trajectory IDs that have been completed
    """
    checkpoint_path = _checkpoint_path_for(jsonl_path)

    # Fast path: read from lightweight checkpoint file
    if os.path.exists(checkpoint_path):
        completed_ids = set()
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_ids.add(line)
        return completed_ids

    # Fallback: no checkpoint file but JSONL exists (legacy / first migration)
    # Read in fixed-size binary chunks and regex-extract IDs, avoiding
    # Python readline() which would allocate a 10-15 MB string per line.
    if os.path.exists(jsonl_path):
        import re
        completed_ids = set()
        # Our JSON output always starts with {"id": "...", so we look for
        # that pattern. Using \n{ ensures we only match top-level objects
        # (not nested "id" inside original_data etc.).
        id_pattern = re.compile(rb'\n\{"id":\s*"([^"]+)"')

        CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB
        OVERLAP = 256  # handle pattern spanning chunk boundary
        file_size = os.path.getsize(jsonl_path)
        print(f"Checkpoint file not found; scanning JSONL ({file_size / 1e9:.1f} GB) "
              f"for completed IDs (one-time)...")

        with open(jsonl_path, "rb") as f:
            prev_tail = b""
            bytes_read = 0
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    # Process leftover
                    if prev_tail:
                        for m in id_pattern.finditer(b"\n" + prev_tail):
                            completed_ids.add(m.group(1).decode("utf-8"))
                    break
                bytes_read += len(chunk)
                data = prev_tail + chunk
                prev_tail = data[-OVERLAP:]
                for m in id_pattern.finditer(data):
                    completed_ids.add(m.group(1).decode("utf-8"))

        # Write checkpoint file so next time is instant
        if completed_ids:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for tid in completed_ids:
                    f.write(tid + "\n")
        print(f"Created checkpoint file with {len(completed_ids)} IDs")
        return completed_ids

    return set()


def convert_jsonl_to_json(jsonl_path: str, json_path: str):
    """
    Convert a JSONL file to a standard JSON array file (for backward compatibility).

    Streams the JSONL file line-by-line and writes a JSON array incrementally,
    so neither file needs to fit entirely in memory at once.

    Args:
        jsonl_path: Path to the input JSONL file
        json_path: Path to the output JSON file
    """
    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(json_path, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        first = True
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not first:
                fout.write(",\n")
            json.dump(obj, fout, ensure_ascii=False)
            first = False
        fout.write("\n]\n")


def load_results_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load all results from a JSONL file into a list (for downstream consumers that
    still require in-memory access like statistics and visualization).

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        List of result dictionaries
    """
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def load_results_from_jsonl_lightweight(jsonl_path: str) -> List[Dict]:
    """
    Load results from JSONL with only the fields needed for statistics,
    step division, and visualization. Strips heavy fields that are not needed:
    - tokens, token_ids (redundant with entropy_sequence)
    - prompt, raw_generated_text (large text, not needed for analysis)
    - original_data, game_data, metadata (dataset-specific, not needed)
    - top_k_token_ids, top_k_probs from entropy_sequence entries

    This reduces memory usage by ~70-80% compared to full loading.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        List of lightweight result dictionaries
    """
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Slim down entropy_sequence: drop top_k_token_ids and top_k_probs
            slim_entropy_seq = []
            for entry in obj.get("entropy_sequence", []):
                slim_entropy_seq.append({
                    "token": entry.get("token"),
                    "token_id": entry.get("token_id"),
                    "position": entry.get("position"),
                    "entropy": entry.get("entropy"),
                    "percentile": entry.get("percentile"),
                    "color": entry.get("color", "black"),
                })

            results.append({
                "id": obj.get("id"),
                "original_id": obj.get("original_id"),
                "trajectory_index": obj.get("trajectory_index"),
                "problem": obj.get("problem", ""),
                "solution": obj.get("solution", ""),
                "generated_text": obj.get("generated_text", ""),
                "entropy_sequence": slim_entropy_seq,
                "statistics": obj.get("statistics", {}),
                "source": obj.get("source", ""),
                "task_name": obj.get("task_name", ""),
                "category": obj.get("category", ""),
            })
    return results


def save_results(results, config: Dict, output_dir: str):
    """
    Save results in various formats.

    Args:
        results: Generation results with entropy information
        config: Configuration dictionary
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed JSON results
    output_config = config.get("output", {})
    if "json" in output_config.get("save_formats", ["json"]):
        json_path = os.path.join(output_dir, "entropy_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved detailed results to: {json_path}")


def save_statistics(statistics_data: Dict, config: Dict, output_dir: str):
    """
    Save statistics in various formats.

    Args:
        statistics_data: Statistics data including aggregate and per-problem stats
        config: Configuration dictionary
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save aggregate statistics as JSON
    aggregate_stats = statistics_data["aggregate_statistics"]
    agg_path = os.path.join(output_dir, "aggregate_statistics.json")
    with open(agg_path, "w") as f:
        # Convert tuples to lists for JSON serialization
        serializable_stats = dict(aggregate_stats)
        if "most_common_tokens" in serializable_stats:
            serializable_stats["most_common_tokens"] = [
                {"token": token, "count": count}
                for token, count in serializable_stats["most_common_tokens"]
            ]
        if "most_common_high_entropy_tokens" in serializable_stats:
            serializable_stats["most_common_high_entropy_tokens"] = [
                {"token": token, "count": count}
                for token, count in serializable_stats["most_common_high_entropy_tokens"]
            ]
        if "most_common_colored_high_entropy_tokens" in serializable_stats:
            serializable_stats["most_common_colored_high_entropy_tokens"] = [
                {"token": token, "count": count}
                for token, count in serializable_stats["most_common_colored_high_entropy_tokens"]
            ]
        json.dump(serializable_stats, f, indent=2)
    print(f"Saved aggregate statistics to: {agg_path}")

    # Save per-problem statistics as CSV
    output_config = config.get("output", {})
    if "csv" in output_config.get("save_formats", ["csv"]):
        per_problem_stats = statistics_data["per_problem_statistics"]
        if per_problem_stats:
            import pandas as pd

            df = pd.DataFrame(per_problem_stats)
            csv_path = os.path.join(output_dir, "per_problem_statistics.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved per-problem statistics to: {csv_path}")


def save_step_divisions(step_divisions: list, output_dir: str):
    """
    Save step division results.

    Args:
        step_divisions: List of step division results
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed step divisions as JSON
    json_path = os.path.join(output_dir, "step_divisions.json")
    with open(json_path, "w") as f:
        json.dump(step_divisions, f, indent=2)
    print(f"Saved step divisions to: {json_path}")

    # Save human-readable step trajectories
    txt_path = os.path.join(output_dir, "step_trajectories.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for division in step_divisions:
            sample_id = division.get("id", "unknown")
            num_steps = division.get("num_steps", 0)
            num_boundaries = division.get("num_boundaries", 0)

            f.write("=" * 80 + "\n")
            f.write(f"Sample ID: {sample_id}\n")
            f.write(f"Number of Steps: {num_steps}\n")
            f.write(f"Number of Boundaries: {num_boundaries}\n")
            f.write("=" * 80 + "\n\n")

            # Write problem
            problem = division.get("problem", "")
            if problem:
                f.write("Problem:\n")
                f.write(problem + "\n\n")

            # Write each step
            for step in division.get("steps", []):
                step_num = step.get("step_number", 0)
                step_text = step.get("text", "")
                boundary_token = step.get("boundary_token")
                boundary_entropy = step.get("boundary_entropy")

                f.write(f"Step {step_num}:\n")
                f.write(step_text)
                f.write("\n")

                if boundary_token:
                    f.write(f"  [Boundary token: '{boundary_token}', entropy: {boundary_entropy:.4f}]\n")

                f.write("\n")

            f.write("\n" + "=" * 80 + "\n\n")

    print(f"Saved step trajectories to: {txt_path}")


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Merge command line arguments with config
    config = merge_args_with_config(args, config)

    # Setup logger
    log_dir = config.get("output", {}).get("log_dir", "./outputs/logs")
    logger = setup_logger(log_dir=log_dir)

    logger.info("=" * 80)
    logger.info("Step-wise Token Entropy Analysis Framework")
    logger.info("=" * 80)

    # Print configuration
    logger.info(f"Model: {config['model']['name_or_path']}")
    # Handle different dataset types
    dataset_config = config['dataset']
    dataset_type = dataset_config.get('type', 'math')
    if dataset_type == 'korbench':
        kb_config = dataset_config.get('korbench', {})
        dataset_info = f"KOR-Bench (category: {kb_config.get('category', 'all')}, mode: {kb_config.get('mode', 'zero-shot')})"
    elif dataset_type == 'synlogic':
        sl_config = dataset_config.get('synlogic', {})
        task_name = sl_config.get('task_name', 'all')
        dataset_info = f"SynLogic (task: {task_name}, split: {dataset_config.get('split', 'validation')})"
    elif dataset_type == 'bigcodebench':
        bcb_config = dataset_config.get('bigcodebench', {})
        mode = bcb_config.get('mode', 'instruct')
        subset = bcb_config.get('subset', 'v0.1.2')
        dataset_info = f"BigCodeBench (mode: {mode}, subset: {subset})"
    elif dataset_type == 'livecodebench':
        lcb_config = dataset_config.get('livecodebench', {})
        release_version = lcb_config.get('release_version', 'release_v5')
        dataset_info = f"LiveCodeBench (version: {release_version})"
    else:
        dataset_info = dataset_config.get('name', 'unknown')
    logger.info(f"Dataset: {dataset_info}")
    logger.info(f"GPU IDs: {config['gpu']['device_ids']}")
    logger.info(f"Tensor Parallel Size: {config['gpu']['tensor_parallel_size']}")
    logger.info(f"Batch Size: {config['inference']['batch_size']}")
    logger.info(f"Top-k for entropy: {config['entropy']['top_k']}")

    # Initialize GPU manager
    logger.info("\n" + "=" * 80)
    logger.info("Configuring GPUs")
    logger.info("=" * 80)
    gpu_manager = create_gpu_manager(config)
    gpu_manager.print_gpu_info()
    gpu_manager.configure()
    tensor_parallel_size = gpu_manager.get_tensor_parallel_size(
        config["gpu"]["tensor_parallel_size"]
    )

    # Load dataset
    logger.info("\n" + "=" * 80)
    logger.info("Loading Dataset")
    logger.info("=" * 80)
    dataset_loader = create_dataset_loader(config)
    samples = dataset_loader.load()

    # Determine output directory:
    # 1. If --output_dir is explicitly provided via CLI, use it directly
    # 2. Otherwise, use deterministic path for checkpoint/resume
    cli_output_dir = config.get("output", {}).get("cli_output_dir")
    if cli_output_dir:
        output_dir = cli_output_dir
        logger.info(f"Using CLI-specified output directory: {output_dir}")
    else:
        output_dir = get_deterministic_output_dir(config)
        logger.info(f"Using deterministic output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    jsonl_path = os.path.join(output_dir, "entropy_results.jsonl")
    json_path = os.path.join(output_dir, "entropy_results.json")

    # ==================== Checkpoint / Resume Logic ====================
    inference_config = config.get("inference", {})
    n_trajectories = inference_config.get("n", 1)

    completed_ids = load_completed_ids(jsonl_path)
    resume_mode = len(completed_ids) > 0

    if resume_mode:
        logger.info(f"Found {len(completed_ids)} completed trajectories in {jsonl_path}")
        # Filter out already completed samples
        # Note: completed_ids are always strings (read from checkpoint file),
        # but sample["id"] may be int (e.g., AIME 2025). Convert to str for comparison.
        if n_trajectories > 1:
            # For Best-of-N, a sample is complete if ALL N trajectories exist
            completed_sample_ids = set()
            for sample in samples:
                sid = str(sample["id"])
                expected_traj_ids = {f"{sid}_traj_{j}" for j in range(n_trajectories)}
                if expected_traj_ids.issubset(completed_ids):
                    completed_sample_ids.add(sid)
            remaining_samples = [s for s in samples if str(s["id"]) not in completed_sample_ids]
        else:
            remaining_samples = [s for s in samples if str(s["id"]) not in completed_ids]

        logger.info(f"Skipping {len(samples) - len(remaining_samples)} already completed samples")
        logger.info(f"Remaining samples to process: {len(remaining_samples)}")
        samples_to_process = remaining_samples
    else:
        samples_to_process = samples

    # ==================== Inference ====================
    need_inference = len(samples_to_process) > 0

    if need_inference:
        # Initialize entropy calculator
        logger.info("\n" + "=" * 80)
        logger.info("Initializing Entropy Calculator")
        logger.info("=" * 80)
        entropy_calculator = create_entropy_calculator(config)
        logger.info(f"Top-k for entropy calculation: {entropy_calculator.top_k}")

        # Initialize vLLM engine
        logger.info("\n" + "=" * 80)
        logger.info("Initializing vLLM Engine")
        logger.info("=" * 80)
        vllm_engine = create_vllm_engine(config, tensor_parallel_size, entropy_calculator)

        # Create sampling parameters
        if n_trajectories > 1:
            logger.info(f"Best-of-N sampling enabled: generating {n_trajectories} trajectories per sample")

        sampling_params = vllm_engine.create_sampling_params(
            temperature=inference_config.get("temperature", 0.7),
            top_p=inference_config.get("top_p", 0.9),
            max_tokens=inference_config.get("max_tokens", 2048),
            stop_tokens=inference_config.get("stop_tokens"),
            n=n_trajectories,
        )

        # Generate responses with streaming writes
        logger.info("\n" + "=" * 80)
        logger.info("Generating Responses (streaming to JSONL)")
        logger.info("=" * 80)

        with StreamingResultWriter(jsonl_path, append=resume_mode) as writer:
            vllm_engine.generate_with_progress(
                samples=samples_to_process,
                batch_size=inference_config.get("batch_size", 8),
                sampling_params=sampling_params,
                on_batch_complete=writer.write_batch,
            )
            logger.info(f"Wrote {writer.count} trajectories to {jsonl_path}")
    else:
        logger.info("All samples already completed, skipping inference")

    # Convert JSONL -> JSON for backward compatibility
    logger.info("Converting JSONL to JSON for backward compatibility...")
    convert_jsonl_to_json(jsonl_path, json_path)
    logger.info(f"Saved JSON results to: {json_path}")

    # Load results for downstream analysis (lightweight: drops top_k data,
    # tokens list, prompt, raw_generated_text, original_data, etc.)
    logger.info("Loading results for downstream analysis (lightweight mode)...")
    results = load_results_from_jsonl_lightweight(jsonl_path)
    logger.info(f"Loaded {len(results)} trajectories (lightweight)")

    # Analyze statistics
    logger.info("\n" + "=" * 80)
    logger.info("Computing Statistics")
    logger.info("=" * 80)
    statistics_analyzer = create_statistics_analyzer(config)
    statistics_data = statistics_analyzer.analyze_results(results)

    # Print summary statistics
    agg_stats = statistics_data["aggregate_statistics"]
    logger.info(f"Total tokens: {agg_stats['total_tokens']}")
    logger.info(f"Average entropy: {agg_stats['avg_entropy']:.4f}")
    logger.info(f"Max entropy: {agg_stats['max_entropy']:.4f}")
    logger.info(f"Min entropy: {agg_stats['min_entropy']:.4f}")
    logger.info(f"Std entropy: {agg_stats['std_entropy']:.4f}")
    logger.info(f"High-entropy tokens: {agg_stats['high_entropy_count']} ({agg_stats['high_entropy_ratio']*100:.2f}%)")

    # Save statistics
    logger.info("\n" + "=" * 80)
    logger.info("Saving Statistics")
    logger.info("=" * 80)

    save_statistics(statistics_data, config, output_dir)

    # Divide trajectories into steps based on high-entropy tokens
    if config.get("step_division", {}).get("enabled", True):
        logger.info("\n" + "=" * 80)
        logger.info("Dividing Trajectories into Steps")
        logger.info("=" * 80)

        step_divider = create_step_divider(config)
        step_divisions = step_divider.process_all_results(results)

        # Log summary
        total_steps = sum(d["num_steps"] for d in step_divisions)
        total_boundaries = sum(d["num_boundaries"] for d in step_divisions)
        avg_steps = total_steps / len(step_divisions) if step_divisions else 0

        logger.info(f"Processed {len(step_divisions)} samples")
        logger.info(f"Total boundaries found: {total_boundaries}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Average steps per sample: {avg_steps:.2f}")

        save_step_divisions(step_divisions, output_dir)

    else:
        step_divisions = None

    # Free results from memory
    del results

    # Clean up intermediate JSONL if JSON is valid and non-empty
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        if isinstance(parsed, list) and len(parsed) > 0:
            os.remove(jsonl_path)
            checkpoint_path = _checkpoint_path_for(jsonl_path)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            logger.info(f"Cleaned up intermediate files: {jsonl_path}")
        else:
            logger.warning(f"JSON file is empty or invalid, keeping {jsonl_path}")
    except Exception as e:
        logger.warning(f"Could not validate JSON, keeping {jsonl_path}: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Completed Successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
