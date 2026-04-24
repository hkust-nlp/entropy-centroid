#!/usr/bin/env python3
"""
Run Self-Certainty confidence_list.py in data-parallel across multiple GPUs.

Splits the input file into N shards (one per GPU), launches N independent
processes each pinned to its own GPU via CUDA_VISIBLE_DEVICES, waits for
all to finish, then merges the results back into a single output file.

Does NOT modify confidence_list.py — purely orchestration.

Usage:
    # Single file: 8-GPU parallel (auto-detect available GPUs)
    python scripts/run_self_certainty_parallel.py \
        --input_file outputs/results/.../self_certainty_input.json \
        --model_name Qwen/Qwen3-14B

    # Single file: specify GPU IDs and batch size
    python scripts/run_self_certainty_parallel.py \
        --input_file outputs/results/.../self_certainty_input.json \
        --model_name Qwen/Qwen3-14B \
        --gpu_ids 0,1,2,3,4,5,6,7 \
        --batch_size 8

    # Batch: auto-discover all livecodebench dirs and process each
    python scripts/run_self_certainty_parallel.py \
        --batch --benchmark livecodebench --batch_size 8

    # Batch: custom root, explicit GPU IDs
    python scripts/run_self_certainty_parallel.py \
        --batch --benchmark livecodebench \
        --results_root outputs/results \
        --gpu_ids 0,1,2,3,4,5,6,7

    # Resume interrupted runs (both single and batch)
    python scripts/run_self_certainty_parallel.py \
        --batch --benchmark livecodebench --resume
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

# Import directory discovery and model detection from convert script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_self_certainty import (
    find_result_dirs,
    detect_model_path_from_dirname,
    MISTRAL_NATIVE_MODELS,
)


def is_model_compatible(model_name: str) -> tuple:
    """
    Check if a model is compatible with confidence_list.py.

    Returns (compatible: bool, reason: str).
    Note: large models (>80B) are compatible but need multi-GPU mode;
    use needs_multi_gpu() to check for that.
    Mistral-native models (Ministral, Pixtral, mistral-large) are now
    supported via Mistral3ForConditionalGeneration in confidence_list.py.
    """
    return True, ""


def needs_multi_gpu(model_name: str, threshold_b: int = 80) -> bool:
    """
    Check if a model is too large for a single GPU and needs multi-GPU
    deployment via device_map="auto" (pipeline parallelism).

    Detection based on parameter count in model name (e.g., 120B, 70b).
    """
    import re
    size_match = re.search(r'(\d+)[bB]', model_name)
    if size_match:
        size_b = int(size_match.group(1))
        return size_b > threshold_b
    return False


def get_available_gpus():
    """Detect available CUDA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpu_ids = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return gpu_ids
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return [0]


def get_shard_dir(input_file):
    """Return the shard directory path derived from the input file."""
    base_dir = os.path.dirname(os.path.abspath(input_file))
    return os.path.join(base_dir, "self_certainty_shards")


def split_input(input_file, num_shards, shard_dir):
    """
    Split input JSON into N shard files.
    Each shard gets a contiguous slice of the original item list.
    Returns list of shard file paths.
    """
    print(f"  Loading input file: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    total = len(data)
    shard_size = math.ceil(total / num_shards)
    print(f"    Total items: {total}, shards: {num_shards}, ~{shard_size} items/shard")

    os.makedirs(shard_dir, exist_ok=True)

    shard_paths = []
    for i in range(num_shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, total)
        if start >= total:
            break

        shard_data = data[start:end]
        shard_path = os.path.join(shard_dir, f"shard_{i}.json")
        with open(shard_path, 'w') as f:
            json.dump(shard_data, f, ensure_ascii=False)

        size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        print(f"    Shard {i}: items [{start}:{end}] ({len(shard_data)} items, {size_mb:.1f} MB)")
        shard_paths.append(shard_path)

    return shard_paths


def _count_jsonl_lines(path):
    """Count non-empty lines in a JSONL file."""
    count = 0
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        pass
    return count


def _count_json_array_len(path):
    """
    Count items in a JSON array file without loading the entire file.
    Counts top-level '{' at depth 1 (items in the root array).
    Falls back to json.load() for small files.
    """
    size = os.path.getsize(path)
    # For small files (< 10MB), just load normally
    if size < 10 * 1024 * 1024:
        with open(path, 'r') as f:
            return len(json.load(f))

    # For large files, count top-level objects by tracking brace depth
    count = 0
    depth = 0
    in_string = False
    escape_next = False
    with open(path, 'r') as f:
        for line in f:
            for ch in line:
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '[' or ch == '{':
                    depth += 1
                    if depth == 2 and ch == '{':
                        count += 1
                elif ch == ']' or ch == '}':
                    depth -= 1
    return count


def launch_workers(shard_paths, gpu_ids, model_name, batch_size,
                   input_field_name, output_field_name, confidence_script,
                   stagger_delay=30):
    """
    Launch one process per shard, each pinned to its own GPU.
    Staggers launches to avoid simultaneous CPU RAM spike from model loading
    (8 workers × 64GB model = 512GB peak CPU RAM if launched simultaneously).
    Returns list of (process, gpu_id, shard_path, output_path, log_path) tuples.
    """
    workers = []
    for i, shard_path in enumerate(shard_paths):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        output_path = os.path.splitext(shard_path)[0] + "-confidence-list.json"
        progress_path = output_path + '.progress.jsonl'
        log_path = os.path.splitext(shard_path)[0] + ".log"

        # Check if this shard is already fully processed (for resume)
        shard_total = None
        try:
            with open(shard_path, 'r') as f:
                shard_data = json.load(f)
                shard_total = len(shard_data)
        except Exception:
            pass

        if shard_total and os.path.exists(output_path):
            try:
                n_existing = _count_json_array_len(output_path)
                if n_existing >= shard_total:
                    print(f"    GPU {gpu_id} | Shard {i}: already complete "
                          f"({n_existing}/{shard_total} items), skipping.")
                    workers.append((None, gpu_id, shard_path, output_path, log_path))
                    continue
                else:
                    print(f"    GPU {gpu_id} | Shard {i}: resuming "
                          f"({n_existing}/{shard_total} items done)")
            except Exception:
                pass
        elif shard_total and os.path.exists(progress_path):
            # Check JSONL progress file
            done = _count_jsonl_lines(progress_path)
            if done >= shard_total:
                print(f"    GPU {gpu_id} | Shard {i}: JSONL progress complete "
                      f"({done}/{shard_total}), will assemble on launch.")
            elif done > 0:
                print(f"    GPU {gpu_id} | Shard {i}: resuming from JSONL "
                      f"({done}/{shard_total} items done)")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable, confidence_script,
            "--input_file", shard_path,
            "--output_file", output_path,
            "--batch_size", str(batch_size),
            "--model_name", model_name,
            "--input_field_name", input_field_name,
            "--output_field_name", output_field_name,
        ]

        log_file = open(log_path, 'w')

        print(f"    GPU {gpu_id} | Shard {i}: launching (log: {log_path})")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )
        workers.append((proc, gpu_id, shard_path, output_path, log_path))

        # Stagger launches to avoid simultaneous CPU RAM spike.
        # Each model load creates a temporary CPU copy (~model_size bytes).
        if i < len(shard_paths) - 1:
            print(f"      (waiting {stagger_delay}s before next launch "
                  f"to stagger model loading...)")
            time.sleep(stagger_delay)

    return workers


def _tail_log(log_path, n_lines=3):
    """Return last n non-empty lines of a log file."""
    try:
        with open(log_path, 'rb') as f:
            # Seek to end, read backwards to find last lines
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 8192)
            f.seek(max(0, size - read_size))
            content = f.read().decode('utf-8', errors='replace')
        lines = [l for l in content.strip().split('\n') if l.strip()]
        return lines[-n_lines:] if lines else []
    except Exception:
        return []


def wait_for_workers(workers, progress_interval=60):
    """
    Wait for all worker processes to finish.
    Periodically prints progress from worker logs.
    Returns (success_count, fail_count).
    """
    active = [(i, w) for i, w in enumerate(workers) if w[0] is not None]
    completed = set()
    success = 0
    failed = 0

    if not active:
        print("  All shards already complete.")
        return len(workers), 0

    total_active = len(active)
    print(f"\n  Waiting for {total_active} workers...", flush=True)

    last_progress_time = time.time()

    while len(completed) < total_active:
        for idx, (proc, gpu_id, shard_path, output_path, log_path) in active:
            if idx in completed:
                continue
            ret = proc.poll()
            if ret is not None:
                completed.add(idx)
                shard_name = os.path.basename(shard_path)
                if ret == 0:
                    success += 1
                    print(f"    GPU {gpu_id} | {shard_name}: finished OK "
                          f"[{len(completed)}/{total_active}]", flush=True)
                else:
                    failed += 1
                    print(f"    GPU {gpu_id} | {shard_name}: FAILED (exit code {ret}) "
                          f"[{len(completed)}/{total_active}]", flush=True)
                    # Print last few lines of the log for quick diagnosis
                    tail = _tail_log(log_path, n_lines=5)
                    if tail:
                        print(f"      Last log lines:", flush=True)
                        for line in tail:
                            print(f"        {line}", flush=True)
                    print(f"      Full log: {log_path}", flush=True)

        # Periodic progress report from active workers
        now = time.time()
        if now - last_progress_time >= progress_interval and len(completed) < total_active:
            last_progress_time = now
            print(f"\n  --- Progress ({len(completed)}/{total_active} workers done) ---",
                  flush=True)
            for idx, (proc, gpu_id, shard_path, output_path, log_path) in active:
                if idx in completed:
                    continue
                # Check JSONL progress for item count
                progress_jsonl = output_path + '.progress.jsonl'
                done_items = _count_jsonl_lines(progress_jsonl)
                shard_name = os.path.basename(shard_path)
                last_line = _tail_log(log_path, n_lines=1)
                last_msg = last_line[0][:120] if last_line else "(no output yet)"
                print(f"    GPU {gpu_id} | {shard_name}: {done_items} items done | {last_msg}",
                      flush=True)
            print(flush=True)

        time.sleep(5)

    # Count pre-completed shards as success
    pre_completed = len(workers) - total_active
    return success + pre_completed, failed


def merge_results(shard_paths, output_file):
    """Merge all shard outputs into a single output file, preserving original order."""
    print(f"\n  Merging {len(shard_paths)} shard outputs...", flush=True)
    all_results = []

    for i, shard_path in enumerate(shard_paths):
        output_path = os.path.splitext(shard_path)[0] + "-confidence-list.json"
        progress_path = output_path + '.progress.jsonl'

        # Try JSON output first, then JSONL progress file
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                shard_results = json.load(f)
            all_results.extend(shard_results)
            print(f"    Shard {i}: {len(shard_results)} items (from JSON)", flush=True)
        elif os.path.exists(progress_path):
            # Worker crashed before assembling — read from JSONL progress
            shard_results = []
            with open(progress_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        shard_results.append(json.loads(line))
            all_results.extend(shard_results)
            print(f"    Shard {i}: {len(shard_results)} items (from JSONL progress)",
                  flush=True)
        else:
            print(f"    WARNING: Shard {i} output missing: {output_path}", flush=True)
            # Load original shard to preserve items without confidence
            with open(shard_path, 'r') as f:
                shard_data = json.load(f)
            for item in shard_data:
                item["confidence_list"] = [float('-inf')] * len(item.get("output", []))
            all_results.extend(shard_data)

    # Write merged output
    tmp_path = output_file + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, output_file)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Merged output: {output_file} ({len(all_results)} items, {size_mb:.1f} MB)")
    return all_results


def cleanup_shards(shard_dir):
    """Remove shard directory and all temporary files."""
    import shutil
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
        print(f"  Cleaned up shard directory: {shard_dir}")


# ============================================================================
# Core: process one result directory
# ============================================================================

def process_single_input(input_file, output_file, model_name, gpu_ids, batch_size,
                         input_field_name, output_field_name, confidence_script,
                         resume, keep_shards, force,
                         stagger_delay=30, progress_interval=60):
    """
    Run parallel confidence computation for a single input file.
    Returns 0 on success, 1 on failure.
    """
    # Check if final output already exists (skip mechanism).
    # Count items without loading entire files into memory.
    if not force and os.path.exists(output_file):
        try:
            n_input = _count_json_array_len(input_file)
            n_output = _count_json_array_len(output_file)
            if n_input > 0 and n_output >= n_input:
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"  SKIP: output already complete "
                      f"({n_output}/{n_input} items, {size_mb:.1f} MB)")
                print(f"    {output_file}")
                return 0
            elif n_output > 0:
                print(f"  Partial output found ({n_output}/{n_input}), "
                      f"will resume.")
                resume = True
        except Exception:
            pass

    num_gpus = len(gpu_ids)
    shard_dir = get_shard_dir(input_file)

    # Step 1: Split
    if resume and os.path.exists(shard_dir):
        shard_files = sorted(
            [os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
             if f.startswith("shard_") and f.endswith(".json")
             and "confidence" not in f and "log" not in f.split('.')[-1]],
            key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
        )
        print(f"  Resuming with {len(shard_files)} existing shards")
    else:
        shard_files = split_input(input_file, num_gpus, shard_dir)

    if not shard_files:
        print("  No shards to process.")
        return 1

    # Step 2: Launch workers
    print(f"\n  Launching {len(shard_files)} workers...")
    workers = launch_workers(
        shard_paths=shard_files,
        gpu_ids=gpu_ids,
        model_name=model_name,
        batch_size=batch_size,
        input_field_name=input_field_name,
        output_field_name=output_field_name,
        confidence_script=confidence_script,
        stagger_delay=stagger_delay,
    )

    # Step 3: Wait
    success, failed = wait_for_workers(workers, progress_interval=progress_interval)
    print(f"\n  Results: {success} succeeded, {failed} failed")

    # Close log file handles
    for w in workers:
        proc = w[0]
        if proc is not None and hasattr(proc, 'stdout') and proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass

    if failed > 0:
        print(f"  WARNING: {failed} shards failed. Merging available results.")

    # Step 4: Merge
    merge_results(shard_files, output_file)

    # Step 5: Cleanup
    if not keep_shards and failed == 0:
        cleanup_shards(shard_dir)
    elif failed > 0:
        print(f"  Keeping shards for retry (--resume). Dir: {shard_dir}")

    return 0 if failed == 0 else 1


# ============================================================================
# Multi-GPU mode (for large models like 120B+ that don't fit on a single GPU)
# ============================================================================

def process_single_input_multi_gpu(input_file, output_file, model_name, gpu_ids,
                                   batch_size, input_field_name, output_field_name,
                                   confidence_script, resume, force,
                                   progress_interval=60):
    """
    Run confidence computation as a single process using device_map="auto"
    for models too large to fit on a single GPU.

    Instead of sharding data across N GPUs (data parallelism), this deploys
    the model across ALL GPUs via HuggingFace pipeline parallelism and
    processes items sequentially through the multi-GPU model.

    Returns 0 on success, 1 on failure.
    """
    # Check if final output already exists
    if not force and os.path.exists(output_file):
        try:
            n_input = _count_json_array_len(input_file)
            n_output = _count_json_array_len(output_file)
            if n_input > 0 and n_output >= n_input:
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"  SKIP: output already complete "
                      f"({n_output}/{n_input} items, {size_mb:.1f} MB)")
                return 0
            elif n_output > 0:
                print(f"  Partial output found ({n_output}/{n_input}), will resume.")
        except Exception:
            pass

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    cmd = [
        sys.executable, confidence_script,
        "--input_file", input_file,
        "--output_file", output_file,
        "--batch_size", str(batch_size),
        "--model_name", model_name,
        "--input_field_name", input_field_name,
        "--output_field_name", output_field_name,
        "--device_map", "auto",
    ]

    log_path = os.path.splitext(output_file)[0] + "-multi_gpu.log"
    print(f"  Multi-GPU mode: single process with {len(gpu_ids)} GPUs "
          f"(device_map='auto')")
    print(f"  CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_path}")

    log_file = open(log_path, 'w')
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    # Wait with periodic progress reporting
    last_progress_time = time.time()
    try:
        while proc.poll() is None:
            time.sleep(5)
            now = time.time()
            if now - last_progress_time >= progress_interval:
                last_progress_time = now
                progress_path = output_file + '.progress.jsonl'
                done = _count_jsonl_lines(progress_path)
                tail = _tail_log(log_path, 1)
                last_msg = tail[0][:120] if tail else "(no output)"
                print(f"    Progress: {done} items done | {last_msg}",
                      flush=True)
    except KeyboardInterrupt:
        print(f"\n  Interrupted. Terminating worker...")
        proc.terminate()
        proc.wait(timeout=30)
        return 1
    finally:
        log_file.close()

    if proc.returncode != 0:
        print(f"  FAILED (exit code {proc.returncode})")
        tail = _tail_log(log_path, 5)
        if tail:
            print(f"  Last log lines:")
            for line in tail:
                print(f"    {line}")
        print(f"  Full log: {log_path}")
        return 1

    print(f"  Completed successfully.")
    return 0


# ============================================================================
# Batch mode
# ============================================================================

def run_batch(args, gpu_ids, confidence_script):
    """
    Auto-discover result dirs matching --benchmark, then process each
    that has self_certainty_input.json but no completed confidence output.
    """
    results_root = args.results_root
    benchmark = args.benchmark

    print(f"Batch mode: searching for '{benchmark}' result dirs "
          f"under {results_root}")
    print(f"  (recursive search up to depth {args.max_depth})\n")

    dirs = find_result_dirs(results_root, benchmark, max_depth=args.max_depth)

    if not dirs:
        print(f"No result directories found matching benchmark '{benchmark}'")
        return 1

    # Filter: only dirs that have self_certainty_input.json
    eligible = []
    for result_dir, rel_path in dirs:
        input_file = os.path.join(result_dir, "self_certainty_input.json")
        if not os.path.exists(input_file):
            input_parquet = os.path.join(result_dir, "self_certainty_input.parquet")
            if os.path.exists(input_parquet):
                input_file = input_parquet
            else:
                continue
        eligible.append((result_dir, rel_path, input_file))

    if not eligible:
        print(f"No directories with self_certainty_input.json found.")
        print(f"Run convert_to_self_certainty.py --batch --benchmark {benchmark} first.")
        return 1

    print(f"Found {len(eligible)} directories with self_certainty_input:")
    for result_dir, rel_path, input_file in eligible:
        output_file = os.path.splitext(input_file)[0] + "-confidence-list.json"
        model_path = (args.model_name
                      if args.model_name
                      else detect_model_path_from_dirname(rel_path, benchmark))
        status = ""
        if os.path.exists(output_file):
            try:
                n_input = _count_json_array_len(input_file)
                n_output = _count_json_array_len(output_file)
                if n_output >= n_input:
                    status = " [DONE]"
                else:
                    status = f" [PARTIAL {n_output}/{n_input}]"
            except Exception:
                status = " [CORRUPT]"
        print(f"  - {rel_path}  model={model_path}{status}")
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, (result_dir, rel_path, input_file) in enumerate(eligible, 1):
        print(f"\n{'#' * 80}")
        print(f"# [{idx}/{len(eligible)}] {rel_path}")
        print(f"{'#' * 80}")

        model_name = (args.model_name
                      if args.model_name
                      else detect_model_path_from_dirname(rel_path, benchmark))
        output_file = os.path.splitext(input_file)[0] + "-confidence-list.json"

        print(f"  Model: {model_name}")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")

        # Check model compatibility with confidence_list.py
        compatible, reason = is_model_compatible(model_name)
        if not compatible:
            print(f"  SKIP (incompatible): {reason}")
            skip_count += 1
            continue

        use_multi_gpu = needs_multi_gpu(model_name)
        if use_multi_gpu:
            print(f"  Mode: MULTI-GPU (model too large for single GPU, "
                  f"using device_map='auto' across {len(gpu_ids)} GPUs)")

        try:
            if use_multi_gpu:
                ret = process_single_input_multi_gpu(
                    input_file=input_file,
                    output_file=output_file,
                    model_name=model_name,
                    gpu_ids=gpu_ids,
                    batch_size=args.batch_size,
                    input_field_name=args.input_field_name,
                    output_field_name=args.output_field_name,
                    confidence_script=confidence_script,
                    resume=args.resume,
                    force=args.force,
                    progress_interval=args.progress_interval,
                )
            else:
                ret = process_single_input(
                    input_file=input_file,
                    output_file=output_file,
                    model_name=model_name,
                    gpu_ids=gpu_ids,
                    batch_size=args.batch_size,
                    input_field_name=args.input_field_name,
                    output_field_name=args.output_field_name,
                    confidence_script=confidence_script,
                    resume=args.resume,
                    keep_shards=args.keep_shards,
                    force=args.force,
                    stagger_delay=args.stagger_delay,
                    progress_interval=args.progress_interval,
                )

            if ret == 0:
                success_count += 1
            else:
                fail_count += 1
        except KeyboardInterrupt:
            print(f"\n  Interrupted by user. Stopping batch.")
            fail_count += 1
            break
        except Exception as e:
            print(f"\n  ERROR processing {rel_path}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            print(f"  Continuing to next model...")
            continue

    print(f"\n{'=' * 80}")
    print(f"Batch complete: {success_count} succeeded, {fail_count} failed "
          f"(total: {len(eligible)})")
    print(f"{'=' * 80}")
    return 0 if fail_count == 0 else 1


# ============================================================================
# Single-file mode
# ============================================================================

def run_single(args, gpu_ids, confidence_script):
    """Process a single input file."""
    if not args.model_name:
        print("Error: --model_name is required in single-file mode.")
        return 1

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.splitext(args.input_file)[0] + "-confidence-list.json"

    use_multi_gpu = needs_multi_gpu(args.model_name)
    if use_multi_gpu:
        print(f"Multi-GPU mode: model too large for single GPU, "
              f"using device_map='auto' across {len(gpu_ids)} GPUs")
        return process_single_input_multi_gpu(
            input_file=args.input_file,
            output_file=output_file,
            model_name=args.model_name,
            gpu_ids=gpu_ids,
            batch_size=args.batch_size,
            input_field_name=args.input_field_name,
            output_field_name=args.output_field_name,
            confidence_script=confidence_script,
            resume=args.resume,
            force=args.force,
            progress_interval=args.progress_interval,
        )

    return process_single_input(
        input_file=args.input_file,
        output_file=output_file,
        model_name=args.model_name,
        gpu_ids=gpu_ids,
        batch_size=args.batch_size,
        input_field_name=args.input_field_name,
        output_field_name=args.output_field_name,
        confidence_script=confidence_script,
        resume=args.resume,
        keep_shards=args.keep_shards,
        force=args.force,
        stagger_delay=args.stagger_delay,
        progress_interval=args.progress_interval,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Self-Certainty confidence computation in data-parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python scripts/run_self_certainty_parallel.py \\
      --input_file outputs/results/.../self_certainty_input.json \\
      --model_name Qwen/Qwen3-14B --batch_size 8

  # Batch: auto-discover all livecodebench dirs
  python scripts/run_self_certainty_parallel.py \\
      --batch --benchmark livecodebench --batch_size 8

  # Batch: specify GPUs
  python scripts/run_self_certainty_parallel.py \\
      --batch --benchmark livecodebench \\
      --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 8

  # Resume interrupted batch run
  python scripts/run_self_certainty_parallel.py \\
      --batch --benchmark livecodebench --resume

  # Force re-run (ignore existing outputs)
  python scripts/run_self_certainty_parallel.py \\
      --batch --benchmark livecodebench --force
        """,
    )

    # --- Mode selection ---
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--batch", action="store_true",
        help="Batch mode: auto-discover result dirs matching --benchmark"
    )
    mode_group.add_argument(
        "--input_file", type=str, default=None,
        help="Single-file mode: path to self_certainty_input.json"
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
        help="Root directory to search (default: outputs/results)"
    )
    batch_group.add_argument(
        "--max_depth", type=int, default=3,
        help="Max directory depth for recursive search (default: 3)"
    )

    # --- Common options ---
    common_group = parser.add_argument_group("Common options")
    common_group.add_argument(
        "--model_name", type=str, default=None,
        help="HuggingFace model path. Required in single-file mode. "
             "In batch mode, auto-detected from dir name if not specified."
    )
    common_group.add_argument(
        "--output_file", type=str, default=None,
        help="Output file path (single-file mode only)"
    )
    common_group.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated GPU IDs (default: auto-detect all)"
    )
    common_group.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size per GPU (default: 4)"
    )
    common_group.add_argument(
        "--input_field_name", type=str, default="model_input",
        help="Field name for input text (default: model_input)"
    )
    common_group.add_argument(
        "--output_field_name", type=str, default="output",
        help="Field name for output text list (default: output)"
    )
    common_group.add_argument(
        "--confidence_script", type=str, default=None,
        help="Path to confidence_list.py "
             "(default: Self-Certainty/src/confidence_list.py)"
    )
    common_group.add_argument(
        "--resume", action="store_true",
        help="Resume from existing shards (skip re-splitting)"
    )
    common_group.add_argument(
        "--keep_shards", action="store_true",
        help="Keep shard files after merging"
    )
    common_group.add_argument(
        "--force", action="store_true",
        help="Force re-run even if output already exists"
    )
    common_group.add_argument(
        "--stagger_delay", type=int, default=30,
        help="Seconds between worker launches to stagger model loading "
             "and avoid CPU RAM spike (default: 30)"
    )
    common_group.add_argument(
        "--progress_interval", type=int, default=60,
        help="Seconds between progress reports during worker execution "
             "(default: 60)"
    )

    args = parser.parse_args()

    # Resolve confidence_list.py path
    if args.confidence_script is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.confidence_script = os.path.join(
            project_root, "Self-Certainty", "src", "confidence_list.py"
        )

    if not os.path.exists(args.confidence_script):
        print(f"Error: confidence_list.py not found at {args.confidence_script}")
        return 1

    # Detect GPUs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = get_available_gpus()
    print(f"GPUs: {gpu_ids} ({len(gpu_ids)} total)\n")

    # Dispatch
    if args.batch:
        if not args.benchmark:
            parser.error("--benchmark is required in batch mode")
        return run_batch(args, gpu_ids, args.confidence_script)
    elif args.input_file:
        return run_single(args, gpu_ids, args.confidence_script)
    else:
        parser.error("Must specify either --batch or --input_file")


if __name__ == "__main__":
    sys.exit(main())
