import argparse
import json

from tau2.config import (
    DEFAULT_AGENT_IMPLEMENTATION,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_TEMPERATURE_AGENT,
    DEFAULT_LLM_TEMPERATURE_USER,
    DEFAULT_LLM_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_SEED,
    DEFAULT_USER_IMPLEMENTATION,
)
from tau2.data_model.simulation import RunConfig
from tau2.run import get_options, run_domain
from tau2.scripts.leaderboard.verify_trajectories import VerificationMode


def add_run_args(parser):
    """Add run arguments to a parser."""
    domains = get_options().domains
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=domains,
        help="The domain to run the simulation on",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="The number of times each task is run. Default is 1.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=DEFAULT_AGENT_IMPLEMENTATION,
        choices=get_options().agents,
        help=f"The agent implementation to use. Default is {DEFAULT_AGENT_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=DEFAULT_LLM_AGENT,
        help=f"The LLM to use for the agent. Default is {DEFAULT_LLM_AGENT}.",
    )
    parser.add_argument(
        "--agent-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_AGENT},
        help=f"The arguments to pass to the LLM for the agent. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_AGENT}}}'.",
    )
    parser.add_argument(
        "--user",
        type=str,
        choices=get_options().users,
        default=DEFAULT_USER_IMPLEMENTATION,
        help=f"The user implementation to use. Default is {DEFAULT_USER_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default=DEFAULT_LLM_USER,
        help=f"The LLM to use for the user. Default is {DEFAULT_LLM_USER}.",
    )
    parser.add_argument(
        "--user-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_USER},
        help=f"The arguments to pass to the LLM for the user. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_USER}}}'.",
    )
    parser.add_argument(
        "--task-set-name",
        type=str,
        default=None,
        choices=get_options().task_sets,
        help="The task set to run the simulation on. If not provided, will load default task set for the domain.",
    )
    parser.add_argument(
        "--task-split-name",
        type=str,
        default="base",
        help="The task split to run the simulation on. If not provided, will load 'base' split.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="(Optional) run only the tasks with the given IDs. If not provided, will run all tasks.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="The number of tasks to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"The maximum number of steps to run the simulation. Default is {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f"The maximum number of tool errors allowed in a row in the simulation. Default is {DEFAULT_MAX_ERRORS}.",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        required=False,
        help="The path to save the simulation results. Will be saved to data/simulations/<save_to>.json. If not provided, will save to <domain>_<agent>_<user>_<llm_agent>_<llm_user>_<timestamp>.json. If the file already exists, it will try to resume the run.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=False,
        help="Automatically resume from existing save file without interactive prompt. Requires --save-to.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"The maximum number of concurrent simulations to run. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"The seed to use for the simulation. Default is {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help=f"The log level to use for the simulation. Default is {DEFAULT_LOG_LEVEL}.",
    )
    parser.add_argument(
        "--enforce-communication-protocol",
        action="store_true",
        default=False,
        help="Enforce communication protocol rules (e.g., no mixed messages with text and tool calls). Default is False.",
    )

    # Local vLLM mode options
    parser.add_argument(
        "--local-vllm",
        action="store_true",
        default=False,
        help="Enable local vLLM mode. Launches a vLLM OpenAI-compatible server for local inference.",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="Port for the vLLM server. Default is 8000.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1,2,3",
        help='GPU IDs for vLLM (e.g., "4,5,6,7"). Default is "0,1,2,3".',
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM. Default is 4.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM. Default is 0.9.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length for vLLM. Default is auto-detect.",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="Number of top logprobs to collect for entropy calculation. Default is 20.",
    )

    # Prompt-based tool calling
    parser.add_argument(
        "--prompt-tool-calling",
        action="store_true",
        default=False,
        help="Use prompt-based tool calling for models without native function calling support.",
    )

    # Entropy collection
    parser.add_argument(
        "--entropy-output-dir",
        type=str,
        default="outputs/",
        help='Directory for entropy output (entropy_results.jsonl). Default is "outputs/".',
    )


def _run_with_config(args):
    """Apply CLI args to config module and run domain."""
    import tau2.config as config_module

    # Apply local vLLM settings
    if args.local_vllm:
        config_module.LOCAL_VLLM_ENABLED = True
        config_module.LOCAL_VLLM_PORT = args.vllm_port
        config_module.LOCAL_VLLM_API_BASE = f"http://localhost:{args.vllm_port}/v1"
        config_module.LOCAL_VLLM_GPU_IDS = args.gpu_ids
        config_module.LOCAL_VLLM_TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
        config_module.LOCAL_VLLM_GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization
        config_module.LOCAL_VLLM_MAX_MODEL_LEN = args.max_model_len
        config_module.LOCAL_VLLM_TOP_LOGPROBS = args.top_logprobs
        config_module.ENTROPY_COLLECTION_ENABLED = True
        config_module.ENTROPY_OUTPUT_DIR = args.entropy_output_dir

    # Apply prompt-based tool calling
    if args.prompt_tool_calling:
        config_module.PROMPT_TOOL_CALLING = True

    # Auto-resume mode
    if args.auto_resume:
        config_module.AUTO_RESUME = True

    run_config = RunConfig(
        domain=args.domain,
        task_set_name=args.task_set_name,
        task_split_name=args.task_split_name,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        agent=args.agent,
        llm_agent=args.agent_llm,
        llm_args_agent=args.agent_llm_args,
        user=args.user,
        llm_user=args.user_llm,
        llm_args_user=args.user_llm_args,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        save_to=args.save_to,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        enforce_communication_protocol=args.enforce_communication_protocol,
    )
    return run_domain(run_config)


def main():
    parser = argparse.ArgumentParser(description="Tau2 command line interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    add_run_args(run_parser)
    run_parser.set_defaults(
        func=lambda args: _run_with_config(args)
    )

    # Play command
    play_parser = subparsers.add_parser(
        "play", help="Play manual mode - interact with a domain as the agent"
    )
    play_parser.set_defaults(func=lambda args: run_manual_mode())

    # View command
    view_parser = subparsers.add_parser("view", help="View simulation results")
    view_parser.add_argument(
        "--dir",
        type=str,
        help="Directory containing simulation files. Defaults to data/simulations if not specified.",
    )
    view_parser.add_argument(
        "--file",
        type=str,
        help="Path to the simulation results file to view",
    )
    view_parser.add_argument(
        "--only-show-failed",
        action="store_true",
        help="Only show failed tasks.",
    )
    view_parser.add_argument(
        "--only-show-all-failed",
        action="store_true",
        help="Only show tasks that failed in all trials.",
    )
    view_parser.set_defaults(func=lambda args: run_view_simulations(args))

    # Domain command
    domain_parser = subparsers.add_parser("domain", help="Show domain documentation")
    domain_parser.add_argument(
        "domain",
        type=str,
        help="Name of the domain to show documentation for (e.g., 'airline', 'mock')",
    )
    domain_parser.set_defaults(func=lambda args: run_show_domain(args))

    # Start command
    start_parser = subparsers.add_parser("start", help="Start all servers")
    start_parser.set_defaults(func=lambda args: run_start_servers())

    # Check data command
    check_data_parser = subparsers.add_parser(
        "check-data", help="Check if data directory is properly configured"
    )
    check_data_parser.set_defaults(func=lambda args: run_check_data())

    # Evaluate trajectories command
    evaluate_parser = subparsers.add_parser(
        "evaluate-trajs", help="Evaluate trajectories and update rewards"
    )
    evaluate_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    evaluate_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save updated trajectory files with recomputed rewards. If not provided, only displays metrics.",
    )
    evaluate_parser.set_defaults(func=lambda args: run_evaluate_trajectories(args))

    # Submit command with subcommands
    submit_parser = subparsers.add_parser(
        "submit", help="Submission management for the leaderboard"
    )
    submit_subparsers = submit_parser.add_subparsers(
        dest="submit_command", help="Submit subcommands", required=True
    )

    # Submit prepare subcommand
    submit_prepare_parser = submit_subparsers.add_parser(
        "prepare", help="Prepare a submission for the leaderboard"
    )
    submit_prepare_parser.add_argument(
        "input_paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    submit_prepare_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory to save the submission and trajectories",
    )
    submit_prepare_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip trajectory verification step",
    )
    submit_prepare_parser.set_defaults(func=lambda args: run_prepare_submission(args))

    # Submit validate subcommand
    submit_validate_parser = submit_subparsers.add_parser(
        "validate", help="Validate an existing submission directory"
    )
    submit_validate_parser.add_argument(
        "submission_dir",
        help="Path to the submission directory to validate",
    )
    submit_validate_parser.add_argument(
        "--mode",
        type=VerificationMode,
        choices=[mode.value for mode in VerificationMode],
        default=VerificationMode.PUBLIC,
        help=f"Verification mode. Default is '{VerificationMode.PUBLIC.value}'",
    )
    submit_validate_parser.set_defaults(func=lambda args: run_validate_submission(args))

    # Submit verify-trajs subcommand
    submit_verify_parser = submit_subparsers.add_parser(
        "verify-trajs", help="Verify trajectory files"
    )
    submit_verify_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    submit_verify_parser.add_argument(
        "--mode",
        type=VerificationMode,
        choices=[mode.value for mode in VerificationMode],
        default=VerificationMode.PUBLIC,
        help=f"Verification mode. Default is '{VerificationMode.PUBLIC.value}'",
    )
    submit_verify_parser.set_defaults(func=lambda args: run_verify_trajectories(args))

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


def run_view_simulations(args):
    from tau2.scripts.view_simulations import main as view_main

    view_main(
        sim_file=args.file,
        only_show_failed=args.only_show_failed,
        only_show_all_failed=args.only_show_all_failed,
        sim_dir=args.dir,
    )


def run_show_domain(args):
    from tau2.scripts.show_domain_doc import main as domain_main

    domain_main(args.domain)


def run_start_servers():
    from tau2.scripts.start_servers import main as start_main

    start_main()


def run_check_data():
    from tau2.scripts.check_data import main as check_data_main

    check_data_main()


def run_verify_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.leaderboard.verify_trajectories import verify_trajectories

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    verify_trajectories(args.paths, args.mode)


def run_evaluate_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.evaluate_trajectories import evaluate_trajectories

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    evaluate_trajectories(args.paths, args.output_dir)


def run_prepare_submission(args):
    """Run the prepare submission command."""
    from tau2.scripts.leaderboard.prepare_submission import prepare_submission

    prepare_submission(
        input_paths=args.input_paths,
        output_dir=args.output,
        run_verification=not args.no_verify,
    )


def run_validate_submission(args):
    """Run the validate submission command."""
    from tau2.scripts.leaderboard.prepare_submission import validate_submission

    validate_submission(submission_dir=args.submission_dir, mode=args.mode)


def run_manual_mode():
    from tau2.scripts.manual_mode import main as manual_main

    manual_main()


if __name__ == "__main__":
    main()
