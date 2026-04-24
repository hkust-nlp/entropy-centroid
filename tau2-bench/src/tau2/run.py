import json
import threading
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.agent.llm_agent import LLMAgent, LLMGTAgent, LLMSoloAgent
import tau2.config as _cfg
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    RunConfig,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.gym.gym_agent import GymAgent
from tau2.metrics.agent_metrics import compute_metrics
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import RegistryInfo, registry
from tau2.user.user_simulator import DummyUser, get_global_user_sim_guidelines
from tau2.utils.display import ConsoleDisplay, Text
from tau2.utils.pydantic_utils import get_pydantic_hash
from tau2.utils.utils import DATA_DIR, get_commit_hash, get_now, show_dict_diff


def get_options() -> RegistryInfo:
    """
    Returns options for the simulator.
    """
    return registry.get_info()


def get_environment_info(
    domain_name: str, include_tool_info: bool = False
) -> EnvironmentInfo:
    """Get information about the environment for a registered Domain"""
    global registry
    env_constructor = registry.get_env_constructor(domain_name)
    return env_constructor().get_info(include_tool_info=include_tool_info)


def load_task_splits(task_set_name: str) -> Optional[dict[str, list[str]]]:
    """
    Loads the task splits for the given domain.
    """
    global registry
    task_split_loader = registry.get_task_splits_loader(task_set_name)
    if task_split_loader is None:
        return None
    return task_split_loader()


def load_tasks(task_set_name: str, task_split_name: Optional[str] = None) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    global registry
    task_loader = registry.get_tasks_loader(task_set_name)
    tasks = task_loader(task_split_name=task_split_name)
    return tasks


def get_tasks(
    task_set_name: str,
    task_split_name: Optional[str] = None,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    if task_ids is None:
        tasks = load_tasks(task_set_name=task_set_name, task_split_name=task_split_name)
    else:
        tasks = [
            task
            for task in load_tasks(
                task_set_name=task_set_name, task_split_name=task_split_name
            )
            if task.id in task_ids
        ]
    if task_ids is not None and len(tasks) != len(task_ids):
        missing_tasks = set(task_ids) - set([task.id for task in tasks])
        raise ValueError(
            f"Not all tasks were found for task set {task_set_name} - {task_split_name}: {missing_tasks}"
        )
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


def make_run_name(config: RunConfig) -> str:
    """
    Make a run name from the run config.

    When AUTO_RESUME is enabled, the name is deterministic (no timestamp)
    so that re-running the same config will find the existing save file
    and resume automatically.  Otherwise, a timestamp is included for
    backward compatibility (each run creates a new file).
    """
    clean_llm_agent_name = [x for x in config.llm_agent.split("/") if x][-1]
    agent_name = f"{config.agent}_{clean_llm_agent_name}"

    clean_llm_user_name = [x for x in config.llm_user.split("/") if x][-1]
    user_name = f"{config.user}_{clean_llm_user_name}"

    if _cfg.AUTO_RESUME:
        return f"{config.domain}_{agent_name}_{user_name}"
    return f"{get_now()}_{config.domain}_{agent_name}_{user_name}"


def run_domain(config: RunConfig) -> Results:
    """
    Run simulations for a domain
    """
    config.validate()
    ConsoleDisplay.display_run_config(config)
    if config.task_set_name is None:
        task_set_name = config.domain
    else:
        task_set_name = config.task_set_name
    tasks = get_tasks(
        task_set_name=task_set_name,
        task_split_name=config.task_split_name,
        task_ids=config.task_ids,
        num_tasks=config.num_tasks,
    )
    if "gt" in config.agent:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if LLMGTAgent.check_valid_task(task)]
        num_tasks = len(tasks)
        console_text = Text(
            text=f"Running {num_tasks} out of {total_num_tasks} tasks for GT agent.",
            style="bold green",
        )
        ConsoleDisplay.console.print(console_text)
    if "solo" in config.agent:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if LLMSoloAgent.check_valid_task(task)]
        num_tasks = len(tasks)
        console_text = Text(
            text=f"Running {num_tasks} out of {total_num_tasks} tasks for solo agent.",
            style="bold green",
        )
        ConsoleDisplay.console.print(console_text)

    num_trials = config.num_trials
    save_to = config.save_to
    if save_to is None:
        save_to = make_run_name(config)
    save_to = DATA_DIR / "simulations" / f"{save_to}.jsonl"

    # Determine entropy output directory
    entropy_output_dir = None
    if _cfg.ENTROPY_COLLECTION_ENABLED:
        entropy_output_dir = Path(_cfg.ENTROPY_OUTPUT_DIR)

    # Manage vLLM server lifecycle if enabled
    vllm_server = None
    if _cfg.LOCAL_VLLM_ENABLED:
        from tau2.utils.vllm_server import VLLMServerManager

        vllm_server = VLLMServerManager(
            model_name=config.llm_agent,
            tensor_parallel_size=_cfg.LOCAL_VLLM_TENSOR_PARALLEL_SIZE,
            gpu_ids=_cfg.LOCAL_VLLM_GPU_IDS,
            port=_cfg.LOCAL_VLLM_PORT,
            gpu_memory_utilization=_cfg.LOCAL_VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=_cfg.LOCAL_VLLM_MAX_MODEL_LEN,
            auto_tool_call_parser=True,
            log_dir=str(_cfg.ENTROPY_OUTPUT_DIR) if _cfg.ENTROPY_COLLECTION_ENABLED else None,
        )

    try:
        if vllm_server is not None:
            vllm_server.start()

        simulation_results = run_tasks(
            domain=config.domain,
            tasks=tasks,
            agent=config.agent,
            user=config.user,
            llm_agent=config.llm_agent,
            llm_args_agent=config.llm_args_agent,
            llm_user=config.llm_user,
            llm_args_user=config.llm_args_user,
            num_trials=num_trials,
            max_steps=config.max_steps,
            max_errors=config.max_errors,
            save_to=save_to,
            console_display=True,
            evaluation_type=EvaluationType.ALL,
            max_concurrency=config.max_concurrency,
            seed=config.seed,
            log_level=config.log_level,
            enforce_communication_protocol=config.enforce_communication_protocol,
            entropy_output_dir=entropy_output_dir,
        )
    finally:
        if vllm_server is not None:
            vllm_server.stop()

    metrics = compute_metrics(simulation_results)
    ConsoleDisplay.display_agent_metrics(metrics)

    return simulation_results


def run_tasks(
    domain: str,
    tasks: list[Task],
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    num_trials: int = 1,
    max_steps: int = 100,
    max_errors: int = 10,
    save_to: Optional[str | Path] = None,
    console_display: bool = True,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    max_concurrency: int = 1,
    seed: Optional[int] = 300,
    log_level: Optional[str] = "INFO",
    enforce_communication_protocol: bool = False,
    entropy_output_dir: Optional[Path] = None,
) -> Results:
    """
    Runs tasks for a given domain.
    If llm_as_judge is True, the LLM will be used to annotate the simulation run.
    Calculates the reward for the simulation run.
    Args:
        domain (str): The domain to run the simulation on.
        tasks (list[Task]): The tasks to run.
        agent (str): The agent to run the simulation on.
        user (str): The user to run the simulation on.
        llm_agent (str): The model to use for the agent.
        llm_args_agent (dict): The arguments to pass to the LLM for the agent.
        llm_user (str): The model to use for the user.
        llm_args_user (dict): The arguments to pass to the LLM for the user.
        max_steps (int): The maximum number of steps to run the simulation.
        max_errors (int): The maximum number of errors to allow in the simulation.
        save_to (str | Path): The path to json file where to save the simulation results. If the file already exists, it will try to resume the run.
        evaluation_type (EvaluationType): The type of evaluation to use.
        max_concurrency (int): The maximum number of concurrent simulations to run.
        seed (int): The seed to use for the simulation.
        log_level (str): The log level to use.
        enforce_communication_protocol (bool): Whether to enforce communication protocol rules.
    Returns:
        The simulation results and the annotations (if llm_review is True).
    """
    if isinstance(save_to, str):
        save_to = Path(save_to)
    # Set log level from config
    logger.remove()
    logger.add(lambda msg: print(msg), level=log_level)
    if len(tasks) == 0:
        raise ValueError("No tasks to run")
    if num_trials <= 0:
        raise ValueError("Number of trials must be greater than 0")
    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")

    random.seed(seed)

    seeds = [random.randint(0, 1000000) for _ in range(num_trials)]
    if "seed" in llm_args_agent:
        logger.warning("Each trial will modify the seed for the agent")

    if "seed" in llm_args_user:
        logger.warning("Each trial will modify the seed for the user")

    save_lock = threading.Lock()

    info = get_info(
        domain=domain,
        agent=agent,
        user=user,
        llm_agent=llm_agent,
        llm_args_agent=llm_args_agent,
        llm_user=llm_user,
        llm_args_user=llm_args_user,
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
    )
    simulation_results = Results(
        info=info,
        tasks=tasks,
        simulations=[],
    )
    done_runs = set()
    if save_to is not None:
        legacy_json_path = save_to.with_suffix(".json")

        if save_to.exists():
            # ── Resume from existing JSONL file ──
            if _cfg.AUTO_RESUME:
                response = "y"
            else:
                response = (
                    ConsoleDisplay.console.input(
                        "[yellow]File [bold]{}[/bold] already exists. Do you want to resume the run? (y/n)[/yellow] ".format(
                            save_to
                        )
                    )
                    .lower()
                    .strip()
                )
            if response != "y":
                raise FileExistsError(
                    f"File {save_to} already exists. Please delete it or use a different save_to name."
                )
            with open(save_to, "r") as fp:
                for line_str in fp:
                    line_str = line_str.strip()
                    if not line_str:
                        continue
                    try:
                        entry = json.loads(line_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSONL line in {save_to}")
                        continue
                    if entry.get("_type") == "header":
                        prev_info = Info.model_validate(entry["info"])
                        if get_pydantic_hash(prev_info) != get_pydantic_hash(info):
                            diff = show_dict_diff(
                                prev_info.model_dump(),
                                info.model_dump(),
                            )
                            ConsoleDisplay.console.print(
                                f"The run config has changed.\n\n{diff}\n\n"
                            )
                            if _cfg.AUTO_RESUME:
                                logger.warning(
                                    "Config changed but AUTO_RESUME is enabled, continuing."
                                )
                            else:
                                cfg_response = (
                                    ConsoleDisplay.console.input(
                                        "[yellow]Config changed. Continue anyway? (y/n)[/yellow] "
                                    )
                                    .lower()
                                    .strip()
                                )
                                if cfg_response != "y":
                                    raise ValueError(
                                        "The run config has changed. Please delete the existing file or use a different save_to name."
                                    )
                    elif entry.get("_type") == "simulation":
                        sim_data = {k: v for k, v in entry.items() if k != "_type"}
                        sim = SimulationRun.model_validate(sim_data)
                        simulation_results.simulations.append(sim)
                        done_runs.add((sim.trial, sim.task_id, sim.seed))

            console_text = Text(
                text=f"Resuming from JSONL: {len(done_runs)} runs done. "
                f"{len(tasks) * num_trials - len(done_runs)} remaining.",
                style="bold yellow",
            )
            ConsoleDisplay.console.print(console_text)

        elif legacy_json_path.exists():
            # ── Resume from legacy JSON and migrate to JSONL ──
            if _cfg.AUTO_RESUME:
                response = "y"
            else:
                response = (
                    ConsoleDisplay.console.input(
                        "[yellow]Legacy file [bold]{}[/bold] found. "
                        "Migrate to JSONL and resume? (y/n)[/yellow] ".format(
                            legacy_json_path
                        )
                    )
                    .lower()
                    .strip()
                )
            if response != "y":
                raise FileExistsError(
                    f"Legacy file {legacy_json_path} exists. "
                    "Delete it or use a different save_to name."
                )
            # ── Streaming migration: avoid loading entire file as string ──
            # json.load(fp) parses in C without creating an intermediate
            # Python string copy, cutting peak memory roughly in half
            # compared to Results.model_validate_json(fp.read()).
            logger.info(f"Reading legacy JSON file: {legacy_json_path}")
            raw_data = None
            try:
                with open(legacy_json_path, "r") as fp:
                    raw_data = json.load(fp)
            except (json.JSONDecodeError, Exception) as e:
                logger.error(
                    f"Legacy JSON file is corrupted or unreadable: "
                    f"{legacy_json_path}: {e}"
                )
                ConsoleDisplay.console.print(
                    f"[bold red]Legacy file {legacy_json_path} is corrupted. "
                    f"Starting fresh run (old data will not be migrated).[/bold red]"
                )

            if raw_data is None:
                # Corrupted / unreadable — create fresh JSONL
                if not save_to.parent.exists():
                    save_to.parent.mkdir(parents=True, exist_ok=True)
                header = {
                    "_type": "header",
                    "info": info.model_dump(mode="json"),
                }
                with open(save_to, "w") as fp:
                    fp.write(json.dumps(header) + "\n")
                ConsoleDisplay.console.print(Text(
                    text=f"Created new JSONL (legacy JSON corrupted). "
                    f"{len(tasks) * num_trials} runs to execute.",
                    style="bold yellow",
                ))
            else:
                # Validate config
                prev_info = Info.model_validate(raw_data.get("info", {}))
                if get_pydantic_hash(prev_info) != get_pydantic_hash(info):
                    diff = show_dict_diff(
                        prev_info.model_dump(),
                        info.model_dump(),
                    )
                    ConsoleDisplay.console.print(
                        f"The run config has changed.\n\n{diff}\n\n"
                    )
                    if _cfg.AUTO_RESUME:
                        logger.warning(
                            "Config changed but AUTO_RESUME is enabled, continuing."
                        )
                    else:
                        cfg_response = (
                            ConsoleDisplay.console.input(
                                "[yellow]Config changed. Continue anyway? (y/n)[/yellow] "
                            )
                            .lower()
                            .strip()
                        )
                        if cfg_response != "y":
                            raise ValueError("The run config has changed.")
                del prev_info

                # Validate tasks (warn only, don't block)
                prev_task_ids = sorted(
                    t.get("id", "") if isinstance(t, dict) else t.id
                    for t in raw_data.get("tasks", [])
                )
                curr_task_ids = sorted(t.id for t in tasks)
                if prev_task_ids != curr_task_ids:
                    logger.warning(
                        f"Task set changed: prev={len(prev_task_ids)} tasks, "
                        f"curr={len(curr_task_ids)} tasks. "
                        "Continuing with current task set."
                    )

                # ── Incremental migration ──
                # Pull the simulations list out of raw_data so we can free
                # the info/tasks dicts immediately, then consume simulations
                # one-by-one — each raw dict is replaced with None after
                # conversion, so memory is released progressively.
                simulations_raw = raw_data.pop("simulations", [])
                del raw_data  # free info + tasks raw dicts
                num_sims = len(simulations_raw)

                if not save_to.parent.exists():
                    save_to.parent.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"Migrating {num_sims} simulations from legacy JSON "
                    f"to JSONL: {save_to}"
                )

                with open(save_to, "w") as fp:
                    header = {
                        "_type": "header",
                        "info": info.model_dump(mode="json"),
                    }
                    fp.write(json.dumps(header) + "\n")

                    for i in range(num_sims):
                        sim_raw = simulations_raw[i]
                        simulations_raw[i] = None  # free raw dict

                        sim = SimulationRun.model_validate(sim_raw)
                        del sim_raw

                        simulation_results.simulations.append(sim)
                        done_runs.add((sim.trial, sim.task_id, sim.seed))

                        sim_dict = sim.model_dump(mode="json")
                        sim_dict["_type"] = "simulation"
                        fp.write(json.dumps(sim_dict) + "\n")

                        if (i + 1) % 50 == 0:
                            logger.info(
                                f"  migrated {i + 1}/{num_sims} simulations"
                            )

                del simulations_raw

                console_text = Text(
                    text=f"Migrated {len(done_runs)} runs from legacy JSON "
                    f"to JSONL. "
                    f"{len(tasks) * num_trials - len(done_runs)} remaining.",
                    style="bold yellow",
                )
                ConsoleDisplay.console.print(console_text)

        else:
            # ── Create new JSONL with header ──
            if not save_to.parent.exists():
                save_to.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving simulation batch to {save_to}")
            header = {
                "_type": "header",
                "info": info.model_dump(mode="json"),
            }
            with open(save_to, "w") as fp:
                fp.write(json.dumps(header) + "\n")

    def _save(simulation: SimulationRun):
        """Append a single simulation to the JSONL file (O(1) per save)."""
        if save_to is None:
            return
        sim_dict = simulation.model_dump(mode="json")
        sim_dict["_type"] = "simulation"
        line = json.dumps(sim_dict) + "\n"
        with save_lock:
            with open(save_to, "a") as fp:
                fp.write(line)

    # Entropy results accumulator
    entropy_results = []
    entropy_lock = threading.Lock()

    def _save_entropy(entropy_entry: dict):
        """Append a single entropy entry to the JSONL file and in-memory list."""
        if entropy_output_dir is None:
            return
        with entropy_lock:
            entropy_results.append(entropy_entry)
            entropy_path = entropy_output_dir / "entropy_results.jsonl"
            with open(entropy_path, "a") as fp:
                fp.write(json.dumps(entropy_entry) + "\n")

    _MAX_TASK_RETRIES = 3

    def _run(task: Task, trial: int, seed: int, progress_str: str) -> Optional[SimulationRun]:
        console_text = Text(
            text=f"{progress_str}. Running task {task.id}, trial {trial + 1}",
            style="bold green",
        )
        ConsoleDisplay.console.print(console_text)

        for attempt in range(1, _MAX_TASK_RETRIES + 1):
            try:
                simulation, entropy_entry = run_task(
                    domain=domain,
                    task=task,
                    agent=agent,
                    user=user,
                    llm_agent=llm_agent,
                    llm_args_agent=llm_args_agent,
                    llm_user=llm_user,
                    llm_args_user=llm_args_user,
                    max_steps=max_steps,
                    max_errors=max_errors,
                    evaluation_type=evaluation_type,
                    seed=seed,
                    enforce_communication_protocol=enforce_communication_protocol,
                    collect_entropy=entropy_output_dir is not None,
                    trial=trial,
                )
                simulation.trial = trial
                if console_display:
                    ConsoleDisplay.display_simulation(simulation, show_details=False)

                # When entropy collection is enabled, only persist the
                # simulation if we actually collected entropy data.  This
                # keeps the simulation JSONL and entropy_results.jsonl in
                # 1-to-1 correspondence.  Simulations without entropy (e.g.
                # single tool-call turn where logprobs are absent) are NOT
                # saved, so the resume mechanism will re-attempt them on
                # the next run.
                if entropy_output_dir is not None and entropy_entry is None:
                    ConsoleDisplay.console.print(
                        f"[bold yellow]⚠ Task {task.id} trial {trial}: "
                        f"no entropy collected (n_assist_msgs may be too few), "
                        f"skipping save so resume will retry[/bold yellow]"
                    )
                    return simulation
                _save(simulation)
                if entropy_entry is not None:
                    _save_entropy(entropy_entry)

                if simulation.termination_reason == TerminationReason.LLM_ERROR.value:
                    ConsoleDisplay.console.print(
                        f"[bold yellow]⚠ Task {task.id} trial {trial}: context window exceeded, "
                        f"partial trajectory saved[/bold yellow]"
                    )
                return simulation
            except Exception as e:
                if attempt < _MAX_TASK_RETRIES:
                    logger.warning(
                        f"Task {task.id} trial {trial} failed (attempt {attempt}/{_MAX_TASK_RETRIES}): {e}. "
                        f"Retrying..."
                    )
                    ConsoleDisplay.console.print(
                        f"[bold yellow]⚠ Task {task.id} trial {trial} failed (attempt {attempt}/{_MAX_TASK_RETRIES}): {e}. "
                        f"Retrying...[/bold yellow]"
                    )
                else:
                    logger.error(
                        f"Task {task.id} trial {trial} failed after {_MAX_TASK_RETRIES} attempts: {e}"
                    )
                    ConsoleDisplay.console.print(
                        f"[bold red]⚠ Task {task.id} trial {trial} failed after {_MAX_TASK_RETRIES} attempts: {e}[/bold red]"
                    )
                    return None
        return None

    # Ensure entropy output directory exists
    if entropy_output_dir is not None:
        entropy_output_dir.mkdir(parents=True, exist_ok=True)

    args = []
    for trial in range(num_trials):
        for i, task in enumerate(tasks):
            if (trial, task.id, seeds[trial]) in done_runs:
                console_text = Text(
                    text=f"Skipping task {task.id}, trial {trial} because it has already been run.",
                    style="bold yellow",
                )
                ConsoleDisplay.console.print(console_text)
                continue
            progress_str = f"{i}/{len(tasks)} (trial {trial + 1}/{num_trials})"
            args.append((task, trial, seeds[trial], progress_str))

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        res = list(executor.map(_run, *zip(*args)))
        # Filter out None results (failed tasks)
        successful = [r for r in res if r is not None]
        failed_count = len(res) - len(successful)
        simulation_results.simulations.extend(successful)

    if failed_count > 0:
        ConsoleDisplay.console.print(
            f"\n⚠️  [bold yellow]{failed_count} task(s) failed out of {len(res)}. "
            f"{len(successful)} completed successfully.[/bold yellow]\n"
            "To review the simulations, run: [bold blue]tau2 view[/bold blue]"
        )
    else:
        ConsoleDisplay.console.print(
            "\n✨ [bold green]Successfully completed all simulations![/bold green]\n"
            "To review the simulations, run: [bold blue]tau2 view[/bold blue]"
        )

    # Report entropy collection results
    if entropy_output_dir is not None and entropy_results:
        entropy_path = entropy_output_dir / "entropy_results.jsonl"
        ConsoleDisplay.console.print(
            f"\n[bold cyan]Entropy data written to {entropy_path} "
            f"({len(entropy_results)} entries)[/bold cyan]"
        )

    return simulation_results


def run_task(
    domain: str,
    task: Task,
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    seed: Optional[int] = None,
    enforce_communication_protocol: bool = False,
    collect_entropy: bool = False,
    trial: int = 0,
) -> tuple[SimulationRun, Optional[dict]]:
    """
    Runs a single task for a given domain.
    Calculates the reward for the simulation run.

    Args:
        domain: The domain to run the simulation on.
        task: The task to run.
        agent: The agent to run the simulation on.
        user: The user to run the simulation on.
        llm_agent: The model to use for the agent.
        llm_args_agent: The arguments to pass to the LLM for the agent.
        llm_user: The model to use for the user.
        llm_args_user: The arguments to pass to the LLM for the user.
        max_steps: The maximum number of steps to run the simulation.
        max_errors: The maximum number of errors to allow in the simulation.
        evaluation_type: The type of evaluation to use.
        seed: The seed to use for the simulation.
        enforce_communication_protocol: Whether to enforce communication protocol rules.
        collect_entropy: Whether to collect per-token entropy from agent generations.

    Returns:
        Tuple of (SimulationRun, entropy_entry_or_None).
        entropy_entry is a dict in centroid format if collect_entropy=True and data was collected.
    """

    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")
    global registry
    logger.info(
        f"STARTING SIMULATION: Domain: {domain}, Task: {task.id}, Agent: {agent}, User: {user}"
    )
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()
    AgentConstructor = registry.get_agent_constructor(agent)

    solo_mode = False
    if issubclass(AgentConstructor, LLMAgent):
        agent_instance = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
        )
    elif issubclass(AgentConstructor, LLMGTAgent):
        agent_instance = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            task=task,
        )
    elif issubclass(AgentConstructor, LLMSoloAgent):
        solo_mode = True
        environment: Environment = environment_constructor(solo_mode=True)
        user_tools = environment.get_user_tools() if environment.user_tools else []
        agent_instance = AgentConstructor(
            tools=environment.get_tools() + user_tools,
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            task=task,
        )
    elif issubclass(AgentConstructor, GymAgent):
        agent_instance = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
        )
    else:
        raise ValueError(
            f"Unknown agent type: {AgentConstructor}. Should be LLMAgent or LLMSoloAgent"
        )

    # Enable entropy collection on agent
    entropy_collector = None
    if collect_entropy and hasattr(agent_instance, "collect_entropy"):
        from tau2.utils.entropy_collector import EntropyCalculator, EntropyCollector

        agent_instance.collect_entropy = True
        entropy_calculator = EntropyCalculator(top_k=_cfg.LOCAL_VLLM_TOP_LOGPROBS)
        entropy_collector = EntropyCollector(entropy_calculator)

    try:
        user_tools = environment.get_user_tools()
    except Exception:
        user_tools = None

    UserConstructor = registry.get_user_constructor(user)
    if issubclass(UserConstructor, DummyUser):
        assert isinstance(
            agent_instance, LLMSoloAgent
        ), "Dummy user can only be used with solo agent"

    user_instance = UserConstructor(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=llm_user,
        llm_args=llm_args_user,
    )

    orchestrator = Orchestrator(
        domain=domain,
        agent=agent_instance,
        user=user_instance,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        solo_mode=solo_mode,
        validate_communication=enforce_communication_protocol,
        entropy_collector=entropy_collector,
    )
    simulation = orchestrator.run()

    reward_info = evaluate_simulation(
        domain=domain,
        task=task,
        simulation=simulation,
        evaluation_type=evaluation_type,
        solo_mode=solo_mode,
    )

    simulation.reward_info = reward_info

    # Build entropy entry in centroid format
    entropy_entry = None
    if entropy_collector is not None and entropy_collector.num_turns > 0:
        # Concatenate all agent turn text for generated_text field
        agent_texts = []
        for msg in simulation.messages:
            if isinstance(msg, AssistantMessage) and msg.content:
                agent_texts.append(msg.content)
        generated_text = "\n".join(agent_texts)

        reward = reward_info.reward if reward_info else 0.0
        problem = str(task.user_scenario) if task.user_scenario else ""
        entropy_entry = entropy_collector.to_centroid_format(
            task_id=task.id,
            problem=problem,
            solution=None,
            reward=reward,
            generated_text=generated_text,
            domain=domain,
            trial=trial,
        )
        # Tag with termination reason so downstream can filter error trajectories
        entropy_entry["termination_reason"] = simulation.termination_reason.value

        logger.info(
            f"Collected entropy for task {task.id}: "
            f"{entropy_collector.num_turns} turns, "
            f"{entropy_collector.total_tokens} tokens, "
            f"termination={simulation.termination_reason.value}"
        )

    logger.info(
        f"FINISHED SIMULATION: Domain: {domain}, Task: {task.id}, "
        f"Agent: {agent_instance.__class__.__name__}, "
        f"User: {user_instance.__class__.__name__}. "
        f"Reward: {reward_info.reward}"
    )
    return simulation, entropy_entry


def get_info(
    domain: str,
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    num_trials: int = 1,
    max_steps: int = 100,
    max_errors: int = 10,
    seed: Optional[int] = None,
) -> Info:
    user_info = UserInfo(
        implementation=user,
        llm=llm_user,
        llm_args=llm_args_user,
        global_simulation_guidelines=get_global_user_sim_guidelines(),
    )
    agent_info = AgentInfo(
        implementation=agent,
        llm=llm_agent,
        llm_args=llm_args_agent,
    )
    environment_info = get_environment_info(
        domain, include_tool_info=False
    )  # NOTE: Not saving tool info to avoid clutter.
    return Info(
        git_commit=get_commit_hash(),
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        user_info=user_info,
        agent_info=agent_info,
        environment_info=environment_info,
        seed=seed,
    )
