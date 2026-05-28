"""Run LIBERO baseline evaluations for pi0.5 and GR00T N1.7 on Modal.

Usage examples:

    modal run scripts/modal_libero_baselines.py::run_pi05 --suite libero_spatial --trials-per-task 50
    modal run scripts/modal_libero_baselines.py::run_gr00t --suite libero_10 --episodes-per-task 20
    modal run scripts/modal_libero_baselines.py::run_all --episodes-per-task 20

The functions write raw logs and a small metrics JSON file under
`/results/libero_baselines` in the `vla-quant-libero-results` Modal volume.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import modal


MINUTES = 60
HOURS = 60 * MINUTES

RESULTS_DIR = Path("/results/libero_baselines")
OPENPI_DIR = Path("/workspace/openpi")
GROOT_DIR = Path("/workspace/Isaac-GR00T")

OPENPI_PORT = 8000
GROOT_PORT = 5555

RESULTS_VOLUME = modal.Volume.from_name("vla-quant-libero-results", create_if_missing=True)
CACHE_VOLUME = modal.Volume.from_name("vla-quant-model-cache", create_if_missing=True)

COMMON_ENV = {
    "HF_HOME": "/cache/huggingface",
    "HF_HUB_CACHE": "/cache/huggingface/hub",
    "OPENPI_DATA_HOME": "/cache/openpi",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
    "NVIDIA_DRIVER_CAPABILITIES": "all",
    "TOKENIZERS_PARALLELISM": "false",
}

OPENPI_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "clang",
        "git",
        "git-lfs",
        "libegl1",
        "libegl1-mesa-dev",
        "libglew-dev",
        "libgl1-mesa-glx",
        "libgles2-mesa-dev",
        "libglib2.0-0",
        "libglfw3-dev",
        "libglu1-mesa",
        "libosmesa6-dev",
        "libsm6",
        "libxext6",
        "libxrender1",
        "linux-headers-generic",
        "make",
        "tmux",
    )
    .pip_install("uv==0.5.1")
    .add_local_dir(
        "third_party/openpi",
        str(OPENPI_DIR),
        copy=True,
        ignore=[
            ".git",
            "**/.git",
            "**/__pycache__",
            "**/*.pyc",
            "data",
            "checkpoints",
            "wandb",
        ],
    )
    .env(COMMON_ENV | {"UV_LINK_MODE": "copy"})
    .workdir(str(OPENPI_DIR))
    .run_commands(
        "uv venv --python 3.11.9 /.venv-openpi",
        "UV_PROJECT_ENVIRONMENT=/.venv-openpi GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev",
        "uv pip install --python /.venv-openpi -e . --no-deps",
        "/.venv-openpi/bin/python -c \"import transformers, pathlib, shutil; d=pathlib.Path(transformers.__file__).parent; src=pathlib.Path('/workspace/openpi/src/openpi/models_pytorch/transformers_replace'); shutil.copytree(src, d, dirs_exist_ok=True)\"",
        "uv venv --python 3.8 /.venv-libero",
        "uv pip sync --python /.venv-libero examples/libero/requirements.txt third_party/libero/requirements.txt packages/openpi-client/pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match",
        "mkdir -p /tmp/libero && printf 'benchmark_root: /workspace/openpi/third_party/libero/libero/libero\\nbddl_files: /workspace/openpi/third_party/libero/libero/libero/bddl_files\\ninit_states: /workspace/openpi/third_party/libero/libero/libero/init_files\\ndatasets: /workspace/openpi/third_party/libero/libero/datasets\\nassets: /workspace/openpi/third_party/libero/libero/libero/assets\\n' > /tmp/libero/config.yaml",
        "mkdir -p /usr/share/glvnd/egl_vendor.d && printf '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libEGL_nvidia.so.0\"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
)

GROOT_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "cmake",
        "ffmpeg",
        "git",
        "git-lfs",
        "libegl1-mesa-dev",
        "libgl1",
        "libglib2.0-0",
        "libglu1-mesa",
        "libsm6",
        "libxext6",
        "libxrender1",
        "tmux",
    )
    .pip_install("uv==0.5.1", "huggingface_hub[cli]")
    .add_local_dir(
        "third_party/Isaac-GR00T",
        str(GROOT_DIR),
        copy=True,
        ignore=[
            ".git",
            "**/.git",
            "**/__pycache__",
            "**/*.pyc",
            "checkpoints",
            "outputs",
            "wandb",
        ],
    )
    .env(COMMON_ENV | {"UV_LINK_MODE": "copy"})
    .workdir(str(GROOT_DIR))
    .run_commands(
        # Modal L40S runs x86_64. GR00T's default pyproject asks uv to
        # validate aarch64 too, including local Git-LFS wheels that are not
        # present in this vendored checkout. Drop aarch64-only requirements
        # inside the image before resolving the x86_64 environment.
        "sed -i \"s/required-environments/environments/\" pyproject.toml",
        "sed -i \"/platform_machine == 'aarch64'/d\" pyproject.toml",
        "sed -i '/^torchcodec = \\[/,/^\\]/d' pyproject.toml",
        "sed -i '/^\\[tool\\.uv\\.extra-build-dependencies\\]/,/^\\[/ { /^\\[tool\\.uv\\.extra-build-dependencies\\]/d; /^\\[/!d; }' pyproject.toml",
        "sed -i '/git submodule update --init/d' gr00t/eval/sim/LIBERO/setup_libero.sh",
        "sed -i 's/uv pip install --requirements /uv pip install --requirement /g' gr00t/eval/sim/LIBERO/setup_libero.sh",
        "uv sync --python 3.10",
    )
)

app = modal.App("vla-quant-libero-baselines")

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

GROOT_TASKS: dict[str, list[str]] = {
    "libero_10": [
        "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ],
    "libero_goal": [
        "libero_sim/open_the_middle_drawer_of_the_cabinet",
        "libero_sim/put_the_bowl_on_the_stove",
        "libero_sim/put_the_wine_bottle_on_top_of_the_cabinet",
        "libero_sim/open_the_top_drawer_and_put_the_bowl_inside",
        "libero_sim/put_the_bowl_on_top_of_the_cabinet",
        "libero_sim/push_the_plate_to_the_front_of_the_stove",
        "libero_sim/put_the_cream_cheese_in_the_bowl",
        "libero_sim/turn_on_the_stove",
        "libero_sim/put_the_bowl_on_the_plate",
        "libero_sim/put_the_wine_bottle_on_the_rack",
    ],
    "libero_object": [
        "libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_butter_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_milk_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_spatial": [
        "libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
}


def _run_streaming(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    log_path: Path,
    check: bool = True,
) -> str:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    output: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
            output.append(line)
        returncode = proc.wait()
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, cmd, "".join(output))
    return "".join(output)


def _start_process(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    log_path: Path,
) -> subprocess.Popen[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=merged_env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    proc._modal_log_handle = log  # type: ignore[attr-defined]
    return proc


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=30)
    log = getattr(proc, "_modal_log_handle", None)
    if log is not None:
        log.close()


def _tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def _wait_for_port(
    host: str,
    port: int,
    timeout_s: int,
    *,
    proc: subprocess.Popen[str] | None = None,
    log_path: Path | None = None,
) -> None:
    import socket

    deadline = time.time() + timeout_s
    last_report = 0.0
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            log_tail = _tail(log_path) if log_path is not None else ""
            raise RuntimeError(
                f"Process exited before opening {host}:{port} with code {proc.returncode}.\n"
                f"Last server log lines:\n{log_tail}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            if sock.connect_ex((host, port)) == 0:
                return
        if log_path is not None and time.time() - last_report > 60:
            log_tail = _tail(log_path, max_chars=2000)
            if log_tail:
                print(f"\nStill waiting for {host}:{port}. Recent server log:\n{log_tail}\n", flush=True)
            else:
                print(f"\nStill waiting for {host}:{port}; server log is empty so far.\n", flush=True)
            last_report = time.time()
        time.sleep(2)
    log_tail = _tail(log_path) if log_path is not None else ""
    raise TimeoutError(f"Timed out waiting for {host}:{port}.\nLast server log lines:\n{log_tail}")


def _sample_gpu(stop: threading.Event, samples: list[dict[str, Any]], interval_s: float = 2.0) -> None:
    query = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    while not stop.is_set():
        try:
            raw = subprocess.check_output(query, text=True)
            for line in raw.strip().splitlines():
                timestamp, index, name, mem_used, mem_total, util = [part.strip() for part in line.split(",")]
                samples.append(
                    {
                        "timestamp": timestamp,
                        "gpu_index": int(index),
                        "gpu_name": name,
                        "memory_used_mb": int(mem_used),
                        "memory_total_mb": int(mem_total),
                        "utilization_gpu_percent": int(util),
                    }
                )
        except Exception as exc:  # nvidia-smi can be temporarily unavailable during container startup.
            samples.append({"error": str(exc)})
        stop.wait(interval_s)


def _parse_openpi_metrics(output: str) -> dict[str, Any]:
    rates = re.findall(r"Total success rate:\s*([0-9.]+)", output)
    episodes = re.findall(r"Total episodes:\s*(\d+)", output)
    return {
        "success_rate": float(rates[-1]) if rates else None,
        "episodes": int(episodes[-1]) if episodes else None,
    }


def _parse_gr00t_metrics(output: str) -> dict[str, Any]:
    rates = re.findall(r"success rate:\s*([0-9.]+)", output)
    return {"success_rate": float(rates[-1]) if rates else None}


def _write_summary(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    RESULTS_VOLUME.commit()
    return payload | {"summary_path": str(path)}


def _max_memory_mb(samples: list[dict[str, Any]]) -> int | None:
    values = [sample["memory_used_mb"] for sample in samples if "memory_used_mb" in sample]
    return max(values) if values else None


def _missing_gr00t_checkpoint_files(model_path: Path) -> list[str]:
    required = [
        "config.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "statistics.json",
    ]
    missing = [name for name in required if not (model_path / name).exists()]
    if not list(model_path.glob("model-*.safetensors")):
        missing.append("model-*.safetensors")
    return missing


@app.function(
    image=OPENPI_IMAGE,
    gpu="A100-40GB",
    timeout=24 * HOURS,
    startup_timeout=2 * HOURS,
    volumes={"/results": RESULTS_VOLUME, "/cache": CACHE_VOLUME},
)
def run_pi05(
    suite: str = "libero_spatial",
    trials_per_task: int = 50,
    seed: int = 7,
    replan_steps: int = 5,
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero",
    dump_quant: bool = False,
    live_quant_bits: int = 0,
    live_quant_mode: str = "wht_lloydmax",
) -> dict[str, Any]:
    """Evaluate pi0.5-LIBERO through OpenPI's native LIBERO client/server."""
    if suite not in LIBERO_SUITES:
        raise ValueError(f"Unknown suite {suite!r}. Expected one of {LIBERO_SUITES}.")

    run_name = f"pi05_{suite}_{int(time.time())}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    quant_env = (
        {
            "VLA_QUANT_DUMP_NPZ": str(run_dir / "kv_drift.npz"),
            "VLA_QUANT_LAYERS": "0",
            "VLA_QUANT_JAX_LAYER_STRIDE": "18",
            "VLA_QUANT_JAX_LAYER_OFFSET": "0",
            "VLA_QUANT_LIVE_BITS": str(live_quant_bits),
            "VLA_QUANT_MODE": live_quant_mode,
            "VLA_QUANT_MAX_STEPS": "12",
            "VLA_QUANT_MAX_VALUES": "65536",
        }
        if dump_quant
        else {}
    )
    stop = threading.Event()
    gpu_samples: list[dict[str, Any]] = []
    monitor = threading.Thread(target=_sample_gpu, args=(stop, gpu_samples), daemon=True)
    monitor.start()

    server_log = run_dir / "server.log"
    server = _start_process(
        [
            "/.venv-openpi/bin/python",
            "scripts/serve_policy.py",
            "--env",
            "LIBERO",
            "policy:checkpoint",
            "--policy.config",
            "pi05_libero",
            "--policy.dir",
            checkpoint,
        ],
        cwd=OPENPI_DIR,
        env=COMMON_ENV | quant_env,
        log_path=server_log,
    )
    try:
        _wait_for_port("127.0.0.1", OPENPI_PORT, timeout_s=7200, proc=server, log_path=server_log)
        client_output = _run_streaming(
            [
                "/.venv-libero/bin/python",
                "examples/libero/main.py",
                "--args.host",
                "127.0.0.1",
                "--args.port",
                str(OPENPI_PORT),
                "--args.task-suite-name",
                suite,
                "--args.num-trials-per-task",
                str(trials_per_task),
                "--args.seed",
                str(seed),
                "--args.replan-steps",
                str(replan_steps),
                "--args.video-out-path",
                str(run_dir / "videos"),
            ],
            cwd=OPENPI_DIR,
            env=COMMON_ENV
            | {
                "PYTHONPATH": f"{OPENPI_DIR}:{OPENPI_DIR / 'packages/openpi-client/src'}:{OPENPI_DIR / 'third_party/libero'}",
                "LIBERO_CONFIG_PATH": "/tmp/libero",
            }
            | quant_env,
            log_path=run_dir / "client.log",
        )
    finally:
        stop.set()
        monitor.join(timeout=10)
        _terminate_process(server)
        RESULTS_VOLUME.commit()

    metrics = _parse_openpi_metrics(client_output)
    return _write_summary(
        run_name,
        {
            "run_name": run_name,
            "model": "pi0.5",
            "checkpoint": checkpoint,
            "suite": suite,
            "trials_per_task": trials_per_task,
            "live_quant_bits": live_quant_bits,
            "live_quant_mode": live_quant_mode,
            "seed": seed,
            "metrics": metrics,
            "max_gpu_memory_mb": _max_memory_mb(gpu_samples),
            "quant_drift_npz": str(run_dir / "kv_drift.npz") if dump_quant else None,
            "gpu_samples": gpu_samples,
            "logs": {"server": str(run_dir / "server.log"), "client": str(run_dir / "client.log")},
        },
    )

#pass in huggingface secret
#set via command line: modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
@app.function(
    image=GROOT_IMAGE,
    gpu="H100",
    timeout=24 * HOURS,
    startup_timeout=2 * HOURS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/results": RESULTS_VOLUME, "/cache": CACHE_VOLUME},
)
def run_gr00t(
    suite: str = "libero_spatial",
    episodes_per_task: int = 20,
    n_envs: int = 5,
    max_episode_steps: int = 720,
    n_action_steps: int = 8,
    seed: int = 7,
    checkpoint_root: str = "/cache/gr00t/GR00T-N1.7-LIBERO",
    dump_quant: bool = False,
    live_quant_bits: int = 0,
    live_quant_mode: str = "wht_lloydmax",
    task_limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate NVIDIA GR00T N1.7-LIBERO over the tasks in a LIBERO suite."""
    if suite not in GROOT_TASKS:
        raise ValueError(f"Unknown suite {suite!r}. Expected one of {tuple(GROOT_TASKS)}.")

    run_name = f"gr00t_n17_{suite}_{int(time.time())}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    quant_env = (
        {
            "VLA_QUANT_DUMP_NPZ": str(run_dir / "kv_drift.npz"),
            "VLA_QUANT_LIVE_BITS": str(live_quant_bits),
            "VLA_QUANT_MODE": live_quant_mode,
            "VLA_QUANT_MAX_STEPS": "12",
            "VLA_QUANT_MAX_VALUES": "65536",
            "GR00T_POLICY_CLIENT_TIMEOUT_MS": "180000",
        }
        if dump_quant
        else {}
    )
    stop = threading.Event()
    gpu_samples: list[dict[str, Any]] = []
    monitor = threading.Thread(target=_sample_gpu, args=(stop, gpu_samples), daemon=True)
    monitor.start()

    model_path = Path(checkpoint_root) / suite
    missing = _missing_gr00t_checkpoint_files(model_path)
    if missing:
        print(f"Downloading GR00T checkpoint files for {suite}; missing: {missing}", flush=True)
        filenames = [
            f"{suite}/config.json",
            f"{suite}/embodiment_id.json",
            f"{suite}/model-00001-of-00002.safetensors",
            f"{suite}/model-00002-of-00002.safetensors",
            f"{suite}/model.safetensors.index.json",
            f"{suite}/processor_config.json",
            f"{suite}/statistics.json",
        ]
        cmd = [
            "uv",
            "run",
            "hf",
            "download",
            "nvidia/GR00T-N1.7-LIBERO",
            *filenames,
            "--local-dir",
            checkpoint_root,
        ]
        _run_streaming(cmd, cwd=GROOT_DIR, env=COMMON_ENV, log_path=run_dir / "download.log")
        CACHE_VOLUME.commit()
        missing = _missing_gr00t_checkpoint_files(model_path)
        if missing:
            listing = _run_streaming(
                ["find", str(model_path), "-maxdepth", "2", "-type", "f"],
                cwd=GROOT_DIR,
                env=COMMON_ENV,
                log_path=run_dir / "checkpoint_listing.log",
                check=False,
            )
            raise FileNotFoundError(
                f"GR00T checkpoint at {model_path} is still missing {missing} after download.\n"
                f"Files found:\n{listing}"
            )

    setup_marker = GROOT_DIR / "gr00t/eval/sim/LIBERO/libero_uv/.setup_complete"
    if not setup_marker.exists():
        _run_streaming(
            ["bash", "gr00t/eval/sim/LIBERO/setup_libero.sh"],
            cwd=GROOT_DIR,
            env=COMMON_ENV,
            log_path=run_dir / "setup_libero.log",
        )
        setup_marker.write_text("ok\n", encoding="utf-8")

    server_log = run_dir / "server.log"
    server = _start_process(
        [
            "uv",
            "run",
            "python",
            "gr00t/eval/run_gr00t_server.py",
            "--model-path",
            str(model_path),
            "--embodiment-tag",
            "LIBERO_PANDA",
            "--use-sim-policy-wrapper",
            "--port",
            str(GROOT_PORT),
        ],
        cwd=GROOT_DIR,
        env=COMMON_ENV | quant_env,
        log_path=server_log,
    )

    task_results: dict[str, dict[str, Any]] = {}
    try:
        _wait_for_port("127.0.0.1", GROOT_PORT, timeout_s=7200, proc=server, log_path=server_log)
        tasks = GROOT_TASKS[suite][:task_limit] if task_limit else GROOT_TASKS[suite]
        for task in tasks:
            task_slug = task.removeprefix("libero_sim/").replace("/", "_")
            log_path = run_dir / f"client_{task_slug}.log"
            try:
                output = _run_streaming(
                    [
                        "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python",
                        "gr00t/eval/rollout_policy.py",
                        "--n-episodes",
                        str(episodes_per_task),
                        "--policy-client-host",
                        "127.0.0.1",
                        "--policy-client-port",
                        str(GROOT_PORT),
                        "--max-episode-steps",
                        str(max_episode_steps),
                        "--env-name",
                        task,
                        "--n-action-steps",
                        str(n_action_steps),
                        "--n-envs",
                        str(n_envs),
                        "--seed",
                        str(seed),
                        "--video-dir",
                        str(run_dir / "videos" / task_slug),
                    ],
                    cwd=GROOT_DIR,
                    env=COMMON_ENV | quant_env,
                    log_path=log_path,
                )
                task_results[task] = _parse_gr00t_metrics(output) | {"status": "ok", "log": str(log_path)}
            except subprocess.CalledProcessError as exc:
                task_results[task] = {
                    "status": "failed",
                    "success_rate": None,
                    "returncode": exc.returncode,
                    "log": str(log_path),
                    "tail": "".join(exc.output.splitlines(keepends=True)[-80:]),
                }
                print(f"GR00T rollout failed for {task}; continuing so logs and partial outputs are saved.", flush=True)
                break
    finally:
        stop.set()
        monitor.join(timeout=10)
        _terminate_process(server)
        RESULTS_VOLUME.commit()

    rates = [result["success_rate"] for result in task_results.values() if result["success_rate"] is not None]
    return _write_summary(
        run_name,
        {
            "run_name": run_name,
            "model": "GR00T N1.7",
            "checkpoint": "nvidia/GR00T-N1.7-LIBERO",
            "suite": suite,
            "episodes_per_task": episodes_per_task,
            "n_envs": n_envs,
            "max_episode_steps": max_episode_steps,
            "n_action_steps": n_action_steps,
            "live_quant_bits": live_quant_bits,
            "live_quant_mode": live_quant_mode,
            "task_limit": task_limit,
            "seed": seed,
            "metrics": {
                "success_rate": (sum(rates) / len(rates)) if rates else None,
                "tasks": task_results,
            },
            "max_gpu_memory_mb": _max_memory_mb(gpu_samples),
            "quant_drift_npz": str(run_dir / "kv_drift.npz") if dump_quant else None,
            "gpu_samples": gpu_samples,
            "logs": {"server": str(run_dir / "server.log"), "task_logs_dir": str(run_dir)},
        },
    )


@app.local_entrypoint()
def run_all(episodes_per_task: int = 20, trials_per_task: int = 50, suite: str = "all") -> None:
    """Run pi0.5 and GR00T baselines. Use suite=all for all four suites."""
    suites = list(LIBERO_SUITES) if suite == "all" else [suite]
    for suite_name in suites:
        print(json.dumps(run_pi05.remote(suite=suite_name, trials_per_task=trials_per_task), indent=2))
        print(json.dumps(run_gr00t.remote(suite=suite_name, episodes_per_task=episodes_per_task), indent=2))


if __name__ == "__main__":
    print(
        "Use Modal to run this file, e.g. "
        "`modal run scripts/modal_libero_baselines.py::run_pi05 --suite libero_spatial`.",
        file=sys.stderr,
    )
