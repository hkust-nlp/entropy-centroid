#!/usr/bin/env python3
"""
Convert centroid entropy_results.json → Self-Certainty input format.

Streams the large entropy_results JSON/JSONL file, groups trajectories by
problem ID, applies the chat template, and outputs the format expected by
Self-Certainty's confidence_list.py.

Output format (one record per problem):
    {
        "model_input": "<full chat-template prompt with special tokens>",
        "output": ["response_0", "response_1", ..., "response_N-1"],
        "generator": "Qwen/Qwen3-14B",
        "question_id": "lcb_1873_A"
    }

Usage:
    # Single directory
    python scripts/convert_to_self_certainty.py \
        --result_dir outputs/results/Qwen_Qwen3-14B_livecodebench_release_v6_n32 \
        --model_path Qwen/Qwen3-14B \
        --task_type livecodebench

    # Batch: auto-discover all livecodebench result dirs
    python scripts/convert_to_self_certainty.py \
        --batch --benchmark livecodebench

    # Batch: custom root and benchmark
    python scripts/convert_to_self_certainty.py \
        --batch --benchmark math --results_root outputs/results

    # Batch with explicit model override (applies to ALL dirs)
    python scripts/convert_to_self_certainty.py \
        --batch --benchmark livecodebench --model_path Qwen/Qwen3-14B
"""

import argparse
import glob as glob_mod
import json
import os
import re
import sys
from collections import OrderedDict

# Models that use Mistral-native tokenizer (tekken.json format),
# incompatible with HuggingFace AutoTokenizer.
# Matches src/inference/vllm_engine.py:MISTRAL_NATIVE_MODELS
MISTRAL_NATIVE_MODELS = ["ministral", "pixtral", "mistral-large"]

# System prompts matching those in src/inference/vllm_engine.py
SYSTEM_PROMPTS = {
    "math": (
        "You are a math expert. Please solve problems step by step "
        "and put your final answer within \\boxed{}."
    ),
    "logic": (
        "You are a logic reasoning expert. Please solve the problem "
        "step by step and provide your final answer."
    ),
    "livecodebench": (
        "You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests. Wrap your final "
        "solution in ```python ``` code blocks."
    ),
    "bigcodebench": (
        "You are an expert Python programmer. Write clean, efficient, and "
        "correct Python code to solve the given task. Include all necessary "
        "imports and ensure the code is complete and executable."
    ),
    "code": (
        "You are an expert Python programmer. Write clean, efficient, and "
        "correct Python code to solve the given task. Include all necessary "
        "imports and ensure the code is complete and executable."
    ),
    "tau2_bench": (
        "You are a helpful customer service agent. "
        "Help the user with their request by using the available tools."
    ),
}

# Benchmark keyword → task_type mapping
BENCHMARK_TASK_TYPE = {
    "livecodebench": "livecodebench",
    "bigcodebench": "bigcodebench",
    "aime": "math",
    "math": "math",
    "olympiadbench": "math",
    "korbench": "logic",
    "tau2_bench": "tau2_bench",
    "tau2-bench": "tau2_bench",
    "airline": "tau2_bench",
    "retail": "tau2_bench",
    "telecom": "tau2_bench",
}


# ============================================================================
# Tokenizer loading with Mistral fallback
# ============================================================================

def _is_mistral_native(model_path: str) -> bool:
    """Check if a model needs Mistral-native tokenizer format."""
    model_lower = model_path.lower()
    return any(m in model_lower for m in MISTRAL_NATIVE_MODELS)


class MistralTokenizerAdapter:
    """
    Wraps mistral_common.MistralTokenizer to provide an apply_chat_template()
    interface compatible with HuggingFace tokenizers.
    """

    def __init__(self, model_path: str):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as _MT
        self._tok = _MT.from_hf_hub(model_path)
        self.vocab_size = self._tok.instruct_tokenizer.tokenizer.n_words

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        from mistral_common.protocol.instruct.messages import (
            UserMessage, SystemMessage, AssistantMessage,
        )
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        converted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "user":
                converted.append(UserMessage(content=content))
            elif role == "assistant":
                converted.append(AssistantMessage(content=content))

        request = ChatCompletionRequest(messages=converted)
        encoded = self._tok.encode_chat_completion(request)
        return encoded.text


def load_tokenizer(model_path: str):
    """
    Load a tokenizer with fallback chain:
    1. HuggingFace AutoTokenizer
    2. mistral_common MistralTokenizer (for Mistral-native models)
    3. None (will use hardcoded templates in build_model_input)
    """
    # Try AutoTokenizer first
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"    Loaded HuggingFace tokenizer (vocab_size={tok.vocab_size})")
        return tok
    except Exception as e:
        print(f"    AutoTokenizer failed: {e}")

    # Try MistralTokenizer for Mistral-native models
    if _is_mistral_native(model_path):
        try:
            adapter = MistralTokenizerAdapter(model_path)
            print(f"    Loaded Mistral-native tokenizer via mistral_common "
                  f"(vocab_size={adapter.vocab_size})")
            return adapter
        except Exception as e2:
            print(f"    MistralTokenizer also failed: {e2}")

    # Return None — build_model_input will use hardcoded fallback
    print(f"    WARNING: No tokenizer available, using hardcoded chat template")
    return None


# ============================================================================
# Directory discovery
# ============================================================================

def find_result_dirs(results_root: str, benchmark: str, max_depth: int = 3):
    """
    Recursively find all directories under results_root that:
    1. Contain 'benchmark' keyword in their path
    2. Have entropy_results.json or entropy_results.jsonl

    Supports both single-level dirs like:
        outputs/results/Qwen_Qwen3-14B_livecodebench_release_v6_n32/
    and nested dirs like:
        outputs/results/config_aime_2025_.../allenai_Olmo-3.1-32B-Think_.../

    Returns list of (result_dir, dir_basename) tuples.
    """
    found = []

    for depth in range(1, max_depth + 1):
        # Build glob pattern: results_root/*/**/.../entropy_results.*
        wildcard = "/".join(["*"] * depth)
        for ext in ["jsonl", "json"]:
            pattern = os.path.join(results_root, wildcard, f"entropy_results.{ext}")
            for filepath in sorted(glob_mod.glob(pattern)):
                result_dir = os.path.dirname(filepath)

                # Check if benchmark keyword appears anywhere in the path
                # relative to results_root
                rel_path = os.path.relpath(result_dir, results_root)

                # For tau2_bench: expand to match any domain subdirectory
                if benchmark.lower() in ("tau2_bench", "tau2-bench"):
                    # Match all tau2-bench domain dirs (airline/retail/telecom/...)
                    pass  # accept all dirs under the tau2_bench results_root
                elif benchmark.lower() not in rel_path.lower():
                    continue

                # Deduplicate (a dir may have both .json and .jsonl)
                if result_dir not in [d for d, _ in found]:
                    found.append((result_dir, rel_path))

    return found


def detect_model_path_from_dirname(dir_basename: str, benchmark: str) -> str:
    """
    Auto-detect HuggingFace model path from result directory name.

    Directory naming convention (model-first or benchmark-first):
        {org}_{model}_{dataset}_{extra}
        {benchmark}_{variant}_{org}_{model}_{extra}
    Maps to HuggingFace path:
        {org}/{model}

    Strategy: find the underscore-delimited segment containing the model
    size indicator (e.g., 14B, 32B, 120b). The preceding segment is the
    org. This correctly handles both model-first and benchmark-first
    directory names.

    For nested dirs, uses the innermost directory name.

    Examples:
        Qwen_Qwen3-14B_livecodebench_release_v6_n32           → Qwen/Qwen3-14B
        Qwen_Qwen3.5-27B_yentinglin_aime_2025_n64             → Qwen/Qwen3.5-27B
        deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_...          → deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
        allenai_Olmo-3.1-32B-Think_livecodebench_...          → allenai/Olmo-3.1-32B-Think
        mistralai_Ministral-3-14B-Instruct-2512_...           → mistralai/Ministral-3-14B-Instruct-2512
        openai_gpt-oss-120b_livecodebench_...                 → openai/gpt-oss-120b
        bigcodebench_instruct_Qwen_Qwen3-14B_20260316_184121  → Qwen/Qwen3-14B
        Qwen_Qwen3-14B/airline  (nested tau2-bench)            → Qwen/Qwen3-14B
    """
    # For nested paths, use the innermost (leaf) directory name
    leaf_name = os.path.basename(dir_basename)

    # If the leaf has no model size indicator (e.g., domain name like "airline"),
    # try the parent directory (handles tau2-bench: <model>/<domain>/ structure)
    if not re.search(r'\d+[bB]', leaf_name):
        parent_name = os.path.basename(os.path.dirname(dir_basename))
        if parent_name:
            leaf_name = parent_name

    # Strategy 1: Find model size indicator (e.g., 14B, 32B, 120b),
    # then identify the underscore-delimited segment containing it.
    # The segment immediately before it is the org (HuggingFace namespace).
    # This works regardless of whether benchmark/variant prefixes exist.
    size_match = re.search(r'\d+[bB]', leaf_name)
    if size_match:
        size_pos = size_match.start()
        segments = leaf_name.split('_')

        # Find which segment contains the size indicator
        cumulative = 0
        size_seg_idx = 0
        for i, seg in enumerate(segments):
            seg_end = cumulative + len(seg)
            if cumulative <= size_pos < seg_end:
                size_seg_idx = i
                break
            cumulative = seg_end + 1  # +1 for the underscore

        # org = preceding segment, model = segment with size indicator
        if size_seg_idx > 0:
            org = segments[size_seg_idx - 1]
            model = segments[size_seg_idx]
            return f"{org}/{model}"
        return segments[size_seg_idx]

    # Strategy 2: Fallback — look for benchmark keywords
    benchmark_keywords = [
        "livecodebench", "bigcodebench", "korbench", "olympiadbench",
        "aime", "math", "synlogic",
    ]
    if benchmark and benchmark not in benchmark_keywords:
        benchmark_keywords.insert(0, benchmark)

    best_pos = len(leaf_name)
    for kw in benchmark_keywords:
        idx = leaf_name.lower().find(f"_{kw.lower()}")
        if 0 <= idx < best_pos:
            best_pos = idx

    if best_pos == len(leaf_name):
        match = re.match(r'^(.+?)_(?:n\d+|release_|20\d{6})', leaf_name)
        prefix = match.group(1) if match else leaf_name
    else:
        prefix = leaf_name[:best_pos]

    parts = prefix.split('_', 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return prefix


def detect_task_type(benchmark: str) -> str:
    """Map benchmark name to task_type for system prompt selection."""
    benchmark_lower = benchmark.lower()
    for key, task_type in BENCHMARK_TASK_TYPE.items():
        if key in benchmark_lower:
            return task_type
    return "math"  # default


# ============================================================================
# Streaming entropy results
# ============================================================================

def iter_entropy_results(result_dir: str):
    """
    Iterate over entries in entropy_results, trying JSONL then JSON.
    Only yields the lightweight fields needed for conversion.
    """
    jsonl_path = os.path.join(result_dir, 'entropy_results.jsonl')
    json_path = os.path.join(result_dir, 'entropy_results.json')

    # Strategy 1: JSONL (line-by-line, most efficient)
    if os.path.exists(jsonl_path):
        print(f"  Using JSONL format: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    yield {
                        'original_id': item.get('original_id', ''),
                        'trajectory_index': item.get('trajectory_index', 0),
                        'prompt': item.get('prompt', '') or item.get('problem', ''),
                        'generated_text': item.get('generated_text', ''),
                    }
                except json.JSONDecodeError:
                    continue
        return

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Neither entropy_results.jsonl nor entropy_results.json "
            f"found in {result_dir}"
        )

    # Strategy 2: ijson streaming on JSON array
    try:
        import ijson
        file_size_gb = os.path.getsize(json_path) / (1024**3)
        print(f"  Using ijson streaming on JSON file ({file_size_gb:.1f}GB)...")
        print(f"  This may take a while for large files.")

        count = 0
        with open(json_path, 'rb') as f:
            for item in ijson.items(f, 'item'):
                yield {
                    'original_id': item.get('original_id', ''),
                    'trajectory_index': item.get('trajectory_index', 0),
                    'prompt': item.get('prompt', '') or item.get('problem', ''),
                    'generated_text': item.get('generated_text', ''),
                }
                count += 1
                if count % 1000 == 0:
                    print(f"    Streamed {count} trajectories...")
        return
    except ImportError:
        print("  ijson not available, falling back to line-by-line parsing...")
    except Exception as e:
        if 'NaN' in str(e) or 'lexical error' in str(e):
            print(f"  ijson failed (likely NaN in JSON): {e}")
        else:
            raise

    # Strategy 3: Brace-depth line-by-line parsing with NaN sanitization
    print(f"  Using NaN-sanitized line-by-line parsing...")
    _nan_pattern = re.compile(r'\bNaN\b')
    _inf_pattern = re.compile(r'\bInfinity\b')
    _ninf_pattern = re.compile(r'-Infinity\b')

    with open(json_path, 'r', encoding='utf-8') as f:
        depth = 0
        buf = []
        in_string = False
        escape_next = False
        count = 0

        for line in f:
            for ch in line:
                if escape_next:
                    escape_next = False
                    if depth > 0:
                        buf.append(ch)
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    if depth > 0:
                        buf.append(ch)
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                if not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                if depth > 0 or (depth == 0 and ch == '}'):
                    buf.append(ch)

                if depth == 0 and buf and buf[-1] == '}':
                    raw = ''.join(buf)
                    buf = []
                    raw = _nan_pattern.sub('null', raw)
                    raw = _ninf_pattern.sub('null', raw)
                    raw = _inf_pattern.sub('null', raw)
                    try:
                        item = json.loads(raw)
                        yield {
                            'original_id': item.get('original_id', ''),
                            'trajectory_index': item.get('trajectory_index', 0),
                            'prompt': item.get('prompt', '') or item.get('problem', ''),
                            'generated_text': item.get('generated_text', ''),
                        }
                        count += 1
                        if count % 1000 == 0:
                            print(f"    Parsed {count} trajectories...")
                    except json.JSONDecodeError:
                        continue


# ============================================================================
# Chat template
# ============================================================================

def build_model_input(prompt: str, tokenizer, system_prompt: str,
                      model_path: str = "") -> str:
    """
    Apply chat template to produce the full model_input string
    with special tokens (e.g., <|im_start|>, <|im_end|>).

    Fallback chain when tokenizer is None or apply_chat_template fails:
    model-aware hardcoded templates matching src/inference/vllm_engine.py
    """
    if tokenizer is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            model_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(model_input, str):
                return model_input
        except Exception as e:
            print(f"  Warning: apply_chat_template failed: {e}")

    # Hardcoded fallback templates (matching _construct_fallback_prompt
    # in src/inference/vllm_engine.py)
    model_lower = model_path.lower()
    if "mistral" in model_lower or "ministral" in model_lower:
        return (f"<s>[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT]"
                f"[INST]{prompt}[/INST]")
    elif "llama" in model_lower:
        return (f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
                f"{prompt} [/INST]")
    else:
        # Qwen-style (default)
        return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n")


# ============================================================================
# Core conversion
# ============================================================================

def convert_single(
    result_dir: str,
    model_path: str,
    task_type: str = "livecodebench",
    output_file: str = None,
    output_format: str = "json",
    n_trajectories: int = None,
    force: bool = False,
):
    """
    Convert one result directory: stream entropy_results → Self-Certainty format.
    Returns the output file path, or None on skip/error.
    """
    # Determine output path
    if output_file is None:
        output_file = os.path.join(
            result_dir,
            f"self_certainty_input.{output_format}"
        )

    # Skip if already exists (unless --force)
    if not force and os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  SKIP: {output_file} already exists ({size_mb:.1f} MB). "
              f"Use --force to overwrite.")
        return output_file

    # Load tokenizer (with Mistral-native fallback)
    print(f"  Loading tokenizer from {model_path}...")
    tokenizer = load_tokenizer(model_path)

    system_prompt = SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["math"])
    print(f"    Task type: {task_type}")
    print(f"    System prompt: {system_prompt[:80]}...")

    # Stream and group trajectories by original_id
    print(f"\n  Streaming entropy_results and grouping trajectories...")
    groups = OrderedDict()

    for item in iter_entropy_results(result_dir):
        original_id = item['original_id']
        traj_idx = item['trajectory_index']

        if original_id not in groups:
            groups[original_id] = {
                'prompt': item['prompt'],
                'outputs': [],
            }

        groups[original_id]['outputs'].append(
            (traj_idx, item['generated_text'])
        )

    total_traj = sum(len(g['outputs']) for g in groups.values())
    print(f"  Grouped {total_traj} trajectories into {len(groups)} problems.")

    if len(groups) == 0:
        print(f"  WARNING: No trajectories found. Skipping.")
        return None

    # Sort outputs within each group by trajectory_index
    for gid, group in groups.items():
        group['outputs'].sort(key=lambda x: x[0])
        group['outputs'] = [text for _, text in group['outputs']]

    # Validate trajectory counts
    traj_counts = {gid: len(g['outputs']) for gid, g in groups.items()}
    unique_counts = set(traj_counts.values())
    if len(unique_counts) > 1:
        from collections import Counter
        count_freq = Counter(traj_counts.values())
        expected_n = count_freq.most_common(1)[0][0]
        print(f"  WARNING: Inconsistent trajectory counts: {dict(count_freq)}")
        print(f"  Expected {expected_n} per problem. Truncating extras.")
        for gid in groups:
            groups[gid]['outputs'] = groups[gid]['outputs'][:expected_n]
    else:
        expected_n = unique_counts.pop()

    if n_trajectories is not None:
        actual_n = min(n_trajectories, expected_n)
        print(f"  Using {actual_n} trajectories per problem "
              f"(requested {n_trajectories}).")
        for gid in groups:
            groups[gid]['outputs'] = groups[gid]['outputs'][:actual_n]
    else:
        actual_n = expected_n
        print(f"  {actual_n} trajectories per problem.")

    # Build Self-Certainty format records
    print(f"\n  Applying chat template and building output...")
    records = []
    for i, (original_id, group) in enumerate(groups.items()):
        model_input = build_model_input(
            group['prompt'], tokenizer, system_prompt, model_path
        )
        record = {
            "model_input": model_input,
            "output": group['outputs'],
            "generator": model_path,
            "question_id": original_id,
        }
        records.append(record)
        if (i + 1) % 200 == 0:
            print(f"    Processed {i + 1}/{len(groups)} problems...")

    # Write output
    print(f"\n  Writing {len(records)} records to {output_file}...")

    if output_format == "parquet":
        import pandas as pd
        df = pd.DataFrame(records)
        df['output'] = df['output'].apply(json.dumps)
        df.to_parquet(output_file, index=False)
    else:
        tmp_path = output_file + '.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, output_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Done! Output: {output_file} ({file_size_mb:.1f} MB)")
    print(f"  Records: {len(records)}, Trajectories per record: {actual_n}")
    return output_file


# ============================================================================
# Batch processing
# ============================================================================

def run_batch(args):
    """Discover and convert all matching result directories."""
    results_root = args.results_root
    benchmark = args.benchmark

    print(f"Batch mode: searching for '{benchmark}' result dirs "
          f"under {results_root}")
    print(f"  (recursive search up to depth {args.max_depth})\n")

    dirs = find_result_dirs(results_root, benchmark, max_depth=args.max_depth)

    if not dirs:
        print(f"No result directories found matching benchmark '{benchmark}'")
        return 1

    print(f"Found {len(dirs)} result directories:")
    for result_dir, rel_path in dirs:
        # Show detected model path for reference
        detected_model = detect_model_path_from_dirname(rel_path, benchmark)
        print(f"  - {rel_path}")
        print(f"    model (auto): {detected_model}")
    print()

    # Determine task type from benchmark
    task_type = args.task_type or detect_task_type(benchmark)
    print(f"Task type: {task_type}")
    print(f"System prompt: {SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS['math'])[:80]}...")
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, (result_dir, rel_path) in enumerate(dirs, 1):
        print(f"\n{'#' * 80}")
        print(f"# [{idx}/{len(dirs)}] {rel_path}")
        print(f"{'#' * 80}")

        # Determine model path
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = detect_model_path_from_dirname(rel_path, benchmark)
        print(f"  Model path: {model_path}")

        try:
            result = convert_single(
                result_dir=result_dir,
                model_path=model_path,
                task_type=task_type,
                output_file=None,  # auto-generate in result_dir
                output_format=args.output_format,
                n_trajectories=args.n_trajectories,
                force=args.force,
            )
            if result:
                success_count += 1
            else:
                skip_count += 1
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print(f"\n{'=' * 80}")
    print(f"Batch complete: {success_count} converted, "
          f"{skip_count} skipped, {fail_count} failed "
          f"(total: {len(dirs)})")
    print(f"{'=' * 80}")
    return 0 if fail_count == 0 else 1


def run_single(args):
    """Convert a single result directory."""
    if not args.model_path:
        print("Error: --model_path is required in single-directory mode.")
        return 1

    convert_single(
        result_dir=args.result_dir,
        model_path=args.model_path,
        task_type=args.task_type or "livecodebench",
        output_file=args.output_file,
        output_format=args.output_format,
        n_trajectories=args.n_trajectories,
        force=args.force,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert entropy_results to Self-Certainty input format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single directory (model_path required)
  python scripts/convert_to_self_certainty.py \\
      --result_dir outputs/results/Qwen_Qwen3-14B_livecodebench_release_v6_n32 \\
      --model_path Qwen/Qwen3-14B \\
      --task_type livecodebench

  # Batch: auto-discover all livecodebench result dirs
  python scripts/convert_to_self_certainty.py \\
      --batch --benchmark livecodebench

  # Batch: with custom root directory
  python scripts/convert_to_self_certainty.py \\
      --batch --benchmark livecodebench \\
      --results_root /other/server/outputs/results

  # Batch: override model for all dirs (e.g., all same model)
  python scripts/convert_to_self_certainty.py \\
      --batch --benchmark livecodebench \\
      --model_path Qwen/Qwen3-14B

  # Batch: deeper recursive search (for nested dirs)
  python scripts/convert_to_self_certainty.py \\
      --batch --benchmark aime --max_depth 4

  # Limit trajectories per problem
  python scripts/convert_to_self_certainty.py \\
      --batch --benchmark livecodebench --n_trajectories 16
        """,
    )

    # --- Mode selection ---
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--batch", action="store_true",
        help="Batch mode: auto-discover result dirs matching --benchmark"
    )
    mode_group.add_argument(
        "--result_dir", type=str, default=None,
        help="Single-dir mode: path to one result directory"
    )

    # --- Batch options ---
    batch_group = parser.add_argument_group("Batch options")
    batch_group.add_argument(
        "--benchmark", type=str, default=None,
        help="Benchmark keyword to filter directories "
             "(e.g., livecodebench, aime, olympiadbench)"
    )
    batch_group.add_argument(
        "--results_root", type=str, default="outputs/results",
        help="Root directory to search for result dirs "
             "(default: outputs/results)"
    )
    batch_group.add_argument(
        "--max_depth", type=int, default=3,
        help="Max directory depth for recursive search (default: 3)"
    )

    # --- Common options ---
    common_group = parser.add_argument_group("Common options")
    common_group.add_argument(
        "--model_path", type=str, default=None,
        help="HuggingFace model path. Required in single-dir mode. "
             "In batch mode, auto-detected from dir name if not specified."
    )
    common_group.add_argument(
        "--task_type", type=str, default=None,
        choices=list(SYSTEM_PROMPTS.keys()),
        help="Task type for system prompt. In batch mode, auto-detected "
             "from --benchmark if not specified."
    )
    common_group.add_argument(
        "--output_file", type=str, default=None,
        help="Output file path (single-dir mode only; "
             "batch mode auto-generates per dir)"
    )
    common_group.add_argument(
        "--output_format", type=str, default="json",
        choices=["json", "parquet"],
        help="Output format (default: json)"
    )
    common_group.add_argument(
        "--n_trajectories", type=int, default=None,
        help="Max trajectories per problem (default: use all available)"
    )
    common_group.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files"
    )

    args = parser.parse_args()

    # Validate args
    if args.batch:
        if not args.benchmark:
            parser.error("--benchmark is required in batch mode")
        return run_batch(args)
    elif args.result_dir:
        return run_single(args)
    else:
        parser.error("Must specify either --batch or --result_dir")


if __name__ == "__main__":
    sys.exit(main())
