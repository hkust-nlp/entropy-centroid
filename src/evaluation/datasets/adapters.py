"""Dataset adapters for unified lowest-centroid runner."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from evaluation.pipeline.cache_paths import canonical_cache_path


@dataclass
class PreparedInputs:
    """Intermediate artifacts produced by dataset preparation."""

    result_dir: str
    intermediate_files: List[str] = field(default_factory=list)


class BaseDatasetAdapter:
    """Common adapter interface."""

    name: str = "base"

    def discover_result_dirs(self, args) -> List[str]:
        raise NotImplementedError

    def prepare_inputs(self, result_dir: str, args) -> PreparedInputs:
        raise NotImplementedError


class BigCodeBenchAdapter(BaseDatasetAdapter):
    """Prepare BigCodeBench inputs and canonical evaluation cache."""

    name = "bigcodebench"

    def discover_result_dirs(self, args) -> List[str]:
        if not args.batch:
            return [args.result_dir]
        matched = sorted(glob.glob(args.pattern))
        result_dirs: List[str] = []
        for path in matched:
            if not os.path.isdir(path):
                continue
            entropy = os.path.join(path, "entropy_results.json")
            if os.path.exists(entropy):
                result_dirs.append(path)
        return result_dirs

    def prepare_inputs(self, result_dir: str, args) -> PreparedInputs:
        from evaluation.datasets.bigcodebench_pipeline import (
            build_evaluation_cache,
            convert_to_bigcodebench_jsonl,
            run_bigcodebench_evaluation,
            sanitize_samples,
        )

        prepared = PreparedInputs(result_dir=result_dir)
        force = bool(getattr(args, "force", False))

        samples_path = os.path.join(result_dir, "bigcodebench_samples.jsonl")
        if force or not os.path.exists(samples_path):
            samples_path = convert_to_bigcodebench_jsonl(
                result_dir=result_dir,
                output_path=samples_path,
                select_strategy=args.select_strategy,
                trajectory_index=args.trajectory_index,
            )
        prepared.intermediate_files.append(samples_path)

        sanitized_path = samples_path.replace(".jsonl", "-sanitized.jsonl")
        if force or not os.path.exists(sanitized_path):
            sanitized_path = sanitize_samples(samples_path, calibrate=False)
        prepared.intermediate_files.append(sanitized_path)

        eval_input = sanitized_path if os.path.exists(sanitized_path) else samples_path
        eval_results_path = eval_input.replace(".jsonl", "_eval_results.json")
        pass_at_k_path = eval_results_path.replace("eval_results.json", "pass_at_k.json")
        prepared.intermediate_files.extend([eval_results_path, pass_at_k_path])

        eval_results: Optional[dict] = None
        if not force and os.path.exists(eval_results_path):
            with open(eval_results_path, "r") as f:
                eval_results = json.load(f)
        else:
            eval_results = run_bigcodebench_evaluation(
                samples_path=eval_input,
                result_dir=result_dir,
                split=args.split,
                subset=args.subset,
                execution=args.execution,
                pass_k=args.pass_k,
                calibrated=args.calibrated,
                parallel=args.parallel,
                min_time_limit=args.min_time_limit,
                no_gt=args.no_gt,
            )

        if eval_results is None:
            raise RuntimeError("BigCodeBench evaluation did not produce eval_results.")

        cache_path = canonical_cache_path(result_dir)
        if force or not os.path.exists(cache_path):
            build_evaluation_cache(
                eval_results=eval_results,
                result_dir=result_dir,
            )

        return prepared


class LiveCodeBenchAdapter(BaseDatasetAdapter):
    """Prepare LiveCodeBench inputs and canonical evaluation cache."""

    name = "livecodebench"

    def discover_result_dirs(self, args) -> List[str]:
        if not args.batch:
            return [args.result_dir]
        matched = sorted(glob.glob(args.pattern))
        result_dirs: List[str] = []
        for path in matched:
            if not os.path.isdir(path):
                continue
            if (
                os.path.exists(os.path.join(path, "entropy_results.json"))
                or os.path.exists(os.path.join(path, "entropy_results.jsonl"))
            ):
                result_dirs.append(path)
        return result_dirs

    def prepare_inputs(self, result_dir: str, args) -> PreparedInputs:
        from evaluation.datasets.livecodebench_pipeline import (
            build_evaluation_cache,
            convert_trajectories,
            detect_release_version,
            run_livecodebench_evaluation,
        )

        prepared = PreparedInputs(result_dir=result_dir)
        force = bool(getattr(args, "force", False))

        release_version = args.release_version
        if release_version == "auto":
            detected = detect_release_version(result_dir)
            release_version = detected if detected else "release_v6"

        extracted_path = os.path.join(result_dir, "livecodebench_extracted.json")
        if force or not os.path.exists(extracted_path):
            convert_trajectories(result_dir)
        prepared.intermediate_files.append(extracted_path)

        eval_results_path = os.path.join(result_dir, "livecodebench_eval_results.json")
        prepared.intermediate_files.append(eval_results_path)
        if force or not os.path.exists(eval_results_path):
            k_list = [int(k) for k in args.pass_k.split(",")] if args.pass_k else None
            run_livecodebench_evaluation(
                result_dir=result_dir,
                release_version=release_version,
                num_process=args.num_process,
                timeout=args.timeout,
                k_list=k_list,
            )

        cache_path = canonical_cache_path(result_dir)
        if force or not os.path.exists(cache_path):
            build_evaluation_cache(result_dir)

        return prepared


class Tau2Adapter(BaseDatasetAdapter):
    """Prepare tau2 inputs to canonical entropy/cache format."""

    name = "tau2_bench"

    def discover_result_dirs(self, args) -> List[str]:
        if not args.batch:
            return [args.result_dir]
        base_dir = args.batch_base_dir
        if not os.path.isdir(base_dir):
            return []
        result_dirs: List[str] = []
        for model in sorted(os.listdir(base_dir)):
            model_dir = os.path.join(base_dir, model)
            if not os.path.isdir(model_dir):
                continue
            for domain in sorted(os.listdir(model_dir)):
                domain_dir = os.path.join(model_dir, domain)
                if not os.path.isdir(domain_dir):
                    continue
                if os.path.isfile(os.path.join(domain_dir, "entropy_results.jsonl")):
                    result_dirs.append(domain_dir)
        return result_dirs

    def prepare_inputs(self, result_dir: str, args) -> PreparedInputs:
        from evaluation.datasets.tau2_pipeline import load_jsonl_deduplicated, step_cache

        prepared = PreparedInputs(result_dir=result_dir)
        jsonl_path = os.path.join(result_dir, "entropy_results.jsonl")
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"entropy_results.jsonl not found: {jsonl_path}")

        trajectories = load_jsonl_deduplicated(jsonl_path)
        if not trajectories:
            raise RuntimeError("No tau2 trajectories loaded from JSONL.")

        # Convert to canonical entropy_results.json used by common centroid runner.
        entropy_json_path = os.path.join(result_dir, "entropy_results.json")
        if getattr(args, "force", False) or not os.path.exists(entropy_json_path):
            temp_path = entropy_json_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(trajectories, f)
            os.replace(temp_path, entropy_json_path)
        prepared.intermediate_files.append(entropy_json_path)

        step_cache(
            result_dir=result_dir,
            trajectories=trajectories,
            force=bool(getattr(args, "force", False)),
            filter_llm_error=bool(getattr(args, "filter_llm_error", False)),
        )
        return prepared

