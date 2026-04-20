#!/usr/bin/env python3
"""Run simulation evaluation across multiple GPUs with a per-GPU task queue.

Each GPU gets a policy server and processes tasks one at a time. When a GPU
finishes a task, it immediately picks up the next one — no waiting for other
GPUs to finish their current task.

Usage:
    # All tasks, 1 GPU
    uv run python vla_foundry/eval/run_evaluation.py experiments/my_checkpoint

    # Specific tasks, 3 GPUs
    uv run python vla_foundry/eval/run_evaluation.py experiments/my_checkpoint \
        --num_gpus 3 --tasks PutMugOnSaucer TurnCupUpsideDown

    # Override defaults
    uv run python vla_foundry/eval/run_evaluation.py experiments/my_checkpoint \
        --num_gpus 3 --num_episodes 100:110 --num_processes 4
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from vla_foundry.eval.data_loading import collect_scenario_indices

# fmt: off
DEFAULT_TASKS = [
    # Main tasks (16 total, split across two platforms)
    #   Cabot — 9:
    "PlaceCupByCoaster", "PushCoasterToCenterOfTable", "PushCoasterToMug",
    "PutBananaOnSaucer", "PutKiwiInCenterOfTable", "PutMugOnSaucer",
    "PutSpatulaInUtensilCrock", "TurnCupUpsideDown", "TurnMugRightsideUp",
    #   Riverway — 7:
    "BimanualPlaceAppleFromBowlIntoBin", "BimanualPlaceFruitFromBowlIntoBin",
    "BimanualPutRedBellPepperInBin", "BimanualPutSpatulaOnPlateFromDryingRack",
    "BimanualPutSpatulaOnPlateFromTable", "BimanualStackPlatesOnTableFromDryingRack",
    "BimanualStoreCerealBoxUnderShelf",
    # Unseen (novel) tasks — 3:
    "BimanualPlaceAvocadoFromBowlIntoBin", "BimanualPutSpatulaOnPlateFromUtensilCrock",
    "PutMugInCenterOfTable",
]
# fmt: on

DEFAULT_DOCKER_IMAGE = "toyotaresearch/lbm-eval-oss:vla-foundry"

# Managed subprocesses for cleanup
_managed_procs: list[subprocess.Popen] = []
_print_lock = Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def parse_episode_range(num_episodes: str) -> set[int]:
    """Parse 'start:end' into a set of episode indices."""
    start, end = num_episodes.split(":")
    return set(range(int(start), int(end)))


def check_no_overlapping_results(
    task_dir: Path,
    requested_indices: set[int],
    max_sample_size: int,
) -> None:
    """Error out if task_dir already has results that conflict with the new run.

    Checks two conditions:
    1. No existing episode indices overlap with the requested range.
    2. Existing episodes + requested episodes do not exceed ``max_sample_size``.
    """
    if not task_dir.exists():
        return
    existing_indices = collect_scenario_indices(task_dir)
    overlap = requested_indices & existing_indices
    if overlap:
        sample = sorted(overlap)[:5]
        raise SystemExit(
            f"ERROR: {task_dir} already contains results for episode indices {sample}"
            f"{'...' if len(overlap) > 5 else ''} ({len(overlap)} overlapping). "
            f"Change --output_dir or remove stale results before re-running."
        )
    combined = len(existing_indices) + len(requested_indices)
    if combined > max_sample_size:
        raise SystemExit(
            f"ERROR: {task_dir} already has {len(existing_indices)} episode(s). "
            f"Adding {len(requested_indices)} would give {combined}, "
            f"exceeding --max_sample_size {max_sample_size}. "
            f"Reduce --num_episodes or increase --max_sample_size."
        )


def cleanup() -> None:
    """Kill all managed subprocesses (policy servers + docker containers)."""
    # Kill managed subprocesses and their entire process groups.
    # Policy servers are launched via `uv run` which spawns a child python
    # process — killing just the uv process can leave the child orphaned.
    for p in _managed_procs:
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        with contextlib.suppress(OSError):
            p.kill()
    # Kill docker containers we labelled
    containers = subprocess.run(
        ["docker", "ps", "-q", "--filter", "label=vla-foundry-eval"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if containers:
        subprocess.run(["docker", "kill"] + containers.split(), capture_output=True)


def wait_for_port(port: int, timeout: int = 1200) -> bool:
    """Wait until a TCP port is accepting connections.

    The timeout covers the full policy-server startup, which on first run
    includes downloading the model checkpoint from HuggingFace — expect
    this to take several minutes for large models on a cold cache.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(2)
    return False


def launch_policy_server(
    gpu: int,
    port: int,
    checkpoint_dir: str,
    args: argparse.Namespace,
    output_dir: Path,
    log_name: str | None = None,
) -> subprocess.Popen:
    """Start a policy server on the given GPU and port."""
    log_path = output_dir / (log_name or f".policy_server_gpu{gpu}.log")
    cmd = [
        "uv",
        "run",
        "--group",
        "inference",
        "python",
        "vla_foundry/inference/robotics/inference_policy.py",
        "--checkpoint_directory",
        checkpoint_dir,
        "--num_flow_steps",
        str(args.num_flow_steps),
        "--open_loop_steps",
        str(args.open_loop_steps),
        "--device",
        args.device,
        "--server-uri",
        f"0.0.0.0:{port}",
    ]
    if args.checkpoint_name:
        cmd.extend(["--checkpoint_name", args.checkpoint_name])
    log_file = open(log_path, "w")  # noqa: SIM115 — must outlive Popen
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env, start_new_session=True)
    _managed_procs.append(proc)
    return proc


def get_render_gid() -> str | None:
    """Get the group ID of /dev/dri/renderD128 for Docker --group-add."""
    try:
        return str(os.stat("/dev/dri/renderD128").st_gid)
    except OSError:
        return None


def run_task(
    task: str,
    gpu: int,
    port: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> bool:
    """Run a single evaluation task in a Docker container. Returns True on success."""
    task_dir = output_dir / args.model_name / task / "rollouts"
    check_no_overlapping_results(task_dir, parse_episode_range(args.num_episodes), args.max_sample_size)
    task_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(task_dir, 0o777)
    docker_log = task_dir.parent / ".docker.log"

    group_add = ["--group-add", "video"]
    render_gid = get_render_gid()
    if render_gid:
        group_add.extend(["--group-add", render_gid])

    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--label",
        "vla-foundry-eval",
        "--runtime=nvidia",
        "--gpus",
        f"device={gpu}" if args.num_gpus > 1 else "all",
        "--device",
        "/dev/dri",
        *group_add,
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
        "-e",
        f"LAUNCH_DEMONSTRATION_INDICES={args.num_episodes}",
        "-e",
        f"NUM_PROCESSES={args.num_processes}",
        "-e",
        f"POLICY_HOST={args.policy_host}",
        "-e",
        f"POLICY_PORT={port}",
        "-e",
        f"USE_EVAL_SEED={args.use_eval_seed}",
        "-e",
        f"MAX_RETRIES={args.max_retries}",
        "-e",
        f"RECORD_VIDEO={args.record_video}",
        "-e",
        f"VIDEO_FPS={args.video_fps}",
        "-e",
        f"MAX_SAMPLE_SIZE_PER_MODEL={args.max_sample_size}",
        "-v",
        f"{task_dir.resolve()}:/tmp/lbm/rollouts",
        args.docker_image,
        "bash",
        "/opt/lbm_eval/launch_sim.sh",
        task,
    ]

    with open(docker_log, "w") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        _managed_procs.append(proc)
        proc.wait()
    return proc.returncode == 0


def read_progress(log_path: Path) -> str | None:
    """Extract the latest tqdm progress from a docker log file."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="replace")
        matches = re.findall(r"(\d+/\d+ \[[\d:]+<[\d:?]+, [\d.?]+(?:s/eval|eval/s)\])", text)
        if matches and not matches[-1].startswith("0/"):
            return matches[-1]
    except OSError:
        pass
    return None


_active_tasks: set[str] = set()
_active_lock = Lock()
_last_progress: dict[str, str] = {}


def gpu_worker(
    gpu: int,
    tasks: list[str],
    port: int,
    args: argparse.Namespace,
    output_dir: Path,
    total: int,
    counter: list[int],
) -> list[tuple[str, bool]]:
    """Worker function for a single GPU — processes tasks sequentially."""
    results = []
    for task in tasks:
        counter[0] += 1
        task_log = output_dir / args.model_name / task / ".docker.log"
        log(f"[{counter[0]}/{total}] Evaluating: {task} (GPU {gpu}, port {port}, log: {task_log})")
        with _active_lock:
            _active_tasks.add(task)
            _last_progress.pop(task, None)
        ok = run_task(task, gpu, port, args, output_dir)
        with _active_lock:
            _active_tasks.discard(task)
        status = "Done" if ok else "FAILED"
        log(f"  {status}: {task}")
        results.append((task, ok))
    return results


def progress_monitor(
    output_dir: Path,
    model_name: str,
    stop_event,
) -> None:
    """Periodically print a progress snapshot for active tasks only."""
    while not stop_event.is_set():
        stop_event.wait(30)
        if stop_event.is_set():
            break
        with _active_lock:
            active = set(_active_tasks)
        lines = []
        for task in sorted(active):
            log_path = output_dir / model_name / task / ".docker.log"
            prog = read_progress(log_path)
            if prog and prog != _last_progress.get(task):
                _last_progress[task] = prog
                lines.append(f"  {task}: {prog}")
        if lines:
            timestamp = time.strftime("%H:%M:%S")
            log(f"--- Progress ({timestamp}) ---\n" + "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run simulation evaluation across multiple GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint_dir", help="Path to model checkpoint directory")
    parser.add_argument("--tasks", nargs="+", default=None, help="Task names to evaluate (default: all paper tasks)")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument(
        "--tasks_per_gpu", type=int, default=1, help="Concurrent tasks per GPU (each gets its own policy server)"
    )
    parser.add_argument("--model_name", default="foundry_model")
    parser.add_argument("--num_episodes", default="0:200")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--num_flow_steps", type=int, default=8)
    parser.add_argument("--open_loop_steps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint_name", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("rollouts"))
    parser.add_argument("--docker_image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--policy_host", default="localhost")
    parser.add_argument("--policy_port", type=int, default=50051)
    parser.add_argument("--use_eval_seed", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--record_video", type=int, default=1)
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument(
        "--max_sample_size",
        type=int,
        required=True,
        help=(
            "Maximum number of policy rollouts (per checkpoint, per task) you ever intend to run. "
            "Required for sequential statistical testing. Set based on your experimental budget."
        ),
    )
    args = parser.parse_args()

    tasks = args.tasks or DEFAULT_TASKS
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Register cleanup — both atexit and signal handlers to cover all exit paths
    atexit.register(cleanup)

    def _signal_handler(*_):
        log("\nInterrupted — cleaning up...")
        cleanup()
        os._exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    num_workers = args.num_gpus * args.tasks_per_gpu
    log(
        f"Evaluation: {len(tasks)} task(s), {args.num_gpus} GPU(s), "
        f"{args.tasks_per_gpu} task(s)/GPU = {num_workers} worker(s), "
        f"episodes {args.num_episodes}"
    )

    # Launch policy servers — one per worker (GPU, slot) pair
    # Workers on the same GPU share CUDA memory but have independent servers.
    worker_ports: list[tuple[int, int]] = []  # (gpu, port) per worker
    log(f"Launching {num_workers} policy server(s)...")
    for gpu in range(args.num_gpus):
        for slot in range(args.tasks_per_gpu):
            port = args.policy_port + gpu * args.tasks_per_gpu + slot
            log_name = f".policy_server_gpu{gpu}_slot{slot}.log"
            launch_policy_server(gpu, port, args.checkpoint_dir, args, output_dir, log_name=log_name)
            log(f"  GPU {gpu} slot {slot} → port {port} (log: {output_dir / log_name})")
            worker_ports.append((gpu, port))

    log("Waiting for policy servers...")
    for _gpu, port in worker_ports:
        if not wait_for_port(port):
            log(f"ERROR: Policy server on port {port} failed to start.")
            return 1
    log(f"  All {num_workers} servers ready")
    log("")

    # Distribute tasks round-robin across workers
    worker_tasks: list[list[str]] = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        worker_tasks[i % num_workers].append(task)

    # Start progress monitor
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=progress_monitor,
        args=(output_dir, args.model_name, stop_event),
        daemon=True,
    )
    monitor.start()

    # Run workers
    counter = [0]  # shared mutable counter
    failed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {}
        for w, (gpu, port) in enumerate(worker_ports):
            if not worker_tasks[w]:
                continue
            f = pool.submit(gpu_worker, gpu, worker_tasks[w], port, args, output_dir, len(tasks), counter)
            futures[f] = w

        for f in as_completed(futures):
            for _task, ok in f.result():
                if not ok:
                    failed += 1

    stop_event.set()
    monitor.join(timeout=5)

    # Summary
    log(f"\nEvaluation complete: {len(tasks) - failed}/{len(tasks)} succeeded")
    if failed:
        log(f"  {failed} task(s) failed")

    log("\nInteractive dashboard:")
    log(f"  uv run --group dashboard python vla_foundry/eval/results_explorer.py {output_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
