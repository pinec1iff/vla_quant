"""Run LIBERO baseline evaluations for pi0.5, GR00T N1.7, and OpenVLA on Modal.

Usage examples:

    modal run scripts/modal_libero_baselines.py::run_pi05 --suite libero_spatial --trials-per-task 50
    modal run scripts/modal_libero_baselines.py::run_gr00t --suite libero_10 --episodes-per-task 20
    modal run scripts/modal_libero_baselines.py::run_all --episodes-per-task 20

The functions write raw logs and a small metrics JSON file under
`/results/libero_baselines` in the `vla-quant-libero-results` Modal volume.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
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
OPENVLA_DIR = Path("/workspace/openvla")
OPENVLA_CACHE_DIR = Path("/model-cache")

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
    "PYTHONUNBUFFERED": "1",
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
            ".venv",
            "**/.venv",
            "checkpoints",
            "outputs",
            "wandb",
        ],
    )
    .env(COMMON_ENV | {"UV_LINK_MODE": "copy"})
    .workdir(str(GROOT_DIR))
    .run_commands(
        # Modal runs x86_64. Resolve a fresh x86-only lock instead of asking
        # uv to validate vendored aarch64 Git-LFS wheel placeholders.
        "sed -i \"/platform_machine == 'aarch64'/d\" pyproject.toml",
        # uv 0.5.1 used by this image calls this setting `environments`.
        "sed -i \"s/required-environments/environments/\" pyproject.toml",
        # FlashAttention is optional at runtime; GR00T falls back to SDPA.
        # Removing it avoids both the aarch64 placeholder and a GitHub-hosted
        # wheel during image construction.
        "sed -i '/\"flash-attn==/d' pyproject.toml",
        "sed -i '/^flash-attn = \\[/,/^\\]/d' pyproject.toml",
        "sed -i '/^torchcodec = \\[/,/^\\]/d' pyproject.toml",
        "sed -i '/^\\[tool\\.uv\\.extra-build-dependencies\\]/,/^\\[/ { /^\\[tool\\.uv\\.extra-build-dependencies\\]/d; /^\\[/!d; }' pyproject.toml",
        "python -c \"from pathlib import Path; Path('uv.lock').unlink(missing_ok=True)\"",
        "sed -i '/git submodule update --init/d' gr00t/eval/sim/LIBERO/setup_libero.sh",
        "sed -i 's/uv pip install --requirements /uv pip install --requirement /g' gr00t/eval/sim/LIBERO/setup_libero.sh",
        "uv sync --python 3.10 --no-dev",
    )
)

OPENVLA_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "clang",
        "cmake",
        "git",
        "git-lfs",
        "libegl1",
        "libegl1-mesa-dev",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libglfw3-dev",
        "libglew-dev",
        "libglu1-mesa",
        "libosmesa6-dev",
        "libsm6",
        "libxext6",
        "libxrender1",
        "ninja-build",
    )
    .pip_install("uv==0.5.1")
    .env(COMMON_ENV | {"UV_LINK_MODE": "copy", "OPENVLA_IMAGE_REV": "4"})
    .run_commands(
        "clang --version && cmake --version && ninja --version",
        "git clone https://github.com/openvla/openvla.git /workspace/openvla && cd /workspace/openvla && git checkout c8f03f4",
        "uv pip install --system torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121",
        "uv pip install --system 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5%2Bcu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'",
        "cd /workspace/openvla && uv pip install --system -r experiments/robot/libero/libero_requirements.txt",
        "uv pip uninstall --system opencv-python opencv-python-headless || true",
        "uv pip install --system numpy==1.26.4 opencv-python==4.10.0.84 'accelerate>=0.25.0' draccus==0.8.0 einops json-numpy jsonlines matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 tensorflow==2.15.0 tensorflow-datasets==4.9.3 tensorflow-graphics==2021.12.3 timm==0.9.10 tokenizers==0.19.1 transformers==4.40.1 wandb",
        "uv pip install --system --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla",
        "cd /workspace/openvla && uv pip install --system -e . --no-deps",
        "git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /workspace/LIBERO",
        "cd /workspace/LIBERO && uv pip install --system -e .",
        "mkdir -p /root/.libero && printf 'benchmark_root: /workspace/LIBERO/libero/libero\\nbddl_files: /workspace/LIBERO/libero/libero/bddl_files\\ninit_states: /workspace/LIBERO/libero/libero/init_files\\ndatasets: /workspace/LIBERO/libero/datasets\\nassets: /workspace/LIBERO/libero/libero/assets\\n' > /root/.libero/config.yaml",
        "cd /workspace/openvla && PYTHONPATH=/workspace/openvla:/workspace/LIBERO python -c \"import accelerate, dlimp, draccus, einops, flash_attn, json_numpy, jsonlines, matplotlib, numpy, peft, rich, sentencepiece, tensorflow as tf, tensorflow_datasets, tensorflow_graphics, torch, torchaudio, wandb; assert numpy.__version__ == '1.26.4'; assert tf.__version__ == '2.15.0'; assert torch.__version__.startswith('2.2.0'); assert flash_attn.__version__.startswith('2.5.5'); assert torch.ones(1).numpy()[0] == 1; assert tuple(tf.image.resize(tf.zeros((4, 4, 3)), (2, 2)).shape) == (2, 2, 3); from libero.libero import benchmark; assert benchmark.get_benchmark_dict()['libero_spatial']().n_tasks == 10; import experiments.robot.libero.run_libero_eval\"",
        "mkdir -p /usr/share/glvnd/egl_vendor.d && printf '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libEGL_nvidia.so.0\"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .workdir(str(OPENVLA_DIR))
)

app = modal.App("vla-quant-libero-baselines")

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TABLE_SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10")
TABLE_SUITE_LABELS = {
    "libero_goal": "Goal",
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_10": "Long",
}
MODEL_LABELS = {
    "gr00t": "GR00T N1.7",
    "pi05": "Pi 0.5",
    "openvla": "OpenVLA 7B Default",
}
OPENVLA_CHECKPOINTS = {
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "libero_object": "openvla/openvla-7b-finetuned-libero-object",
    "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla/openvla-7b-finetuned-libero-10",
}
if set(OPENVLA_CHECKPOINTS) != set(LIBERO_SUITES):
    raise ValueError("OPENVLA_CHECKPOINTS must define exactly one checkpoint per LIBERO suite.")
for suite_name, checkpoint_name in OPENVLA_CHECKPOINTS.items():
    expected_suffix = suite_name.replace("_", "-")
    if not checkpoint_name.endswith(expected_suffix):
        raise ValueError(f"OpenVLA checkpoint for {suite_name} should end with {expected_suffix!r}.")
QUANTIZABLE_TRANSFORMER_LAYERS = {
    "gr00t": 12,
    "pi05": 18,
}


@dataclass(frozen=True)
class QuantPreset:
    name: str
    method: str
    precision: str
    enabled: bool
    weight_bits: int
    act_bits: int


QUANT_PRESETS: dict[str, QuantPreset] = {
    "fp16": QuantPreset("fp16", "Full Precision", "FP16", False, 0, 0),
    "w4a4": QuantPreset("w4a4", "QuantVLA", "W4A4", True, 4, 4),
    "w4a8": QuantPreset("w4a8", "QuantVLA", "W4A8", True, 4, 8),
    "a8": QuantPreset("a8", "QuantVLA", "A8", True, 0, 8),
}

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
    line_callback: Any | None = None,
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
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            output.append(line)
            if line_callback is not None:
                line_callback(line)
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
                        "monotonic_s": time.perf_counter(),
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


class _EpisodeMemoryTracker:
    def __init__(self, gpu_samples: list[dict[str, Any]], *, suite: str, task: str | None = None):
        self.gpu_samples = gpu_samples
        self.suite = suite
        self.task = task
        self.next_sample = 0
        self.episode = 0
        self.records: list[dict[str, Any]] = []

    def consume_line(self, line: str) -> None:
        task_match = re.search(r"Task:\s*(.+)", line)
        if task_match:
            self.task = task_match.group(1).strip()
        if re.search(r"Starting episode\s+\d+", line):
            self.next_sample = len(self.gpu_samples)

        match = re.search(r"# episodes completed so far:\s*(\d+)", line)
        if match is None:
            match = re.search(r"Episode\s+(\d+)/\d+\s+complete:", line)
        if match is None:
            return

        reported_episode = int(match.group(1))
        segment = [
            sample
            for sample in self.gpu_samples[self.next_sample :]
            if "memory_used_mb" in sample
        ]
        self.next_sample = len(self.gpu_samples)
        self.episode += 1
        current_mb = segment[-1]["memory_used_mb"] if segment else None
        peak_mb = max((sample["memory_used_mb"] for sample in segment), default=None)
        self.records.append(
            {
                "episode": self.episode,
                "reported_episode": reported_episode,
                "suite": self.suite,
                "task": self.task,
                "memory_used_gb_at_completion": _memory_gb(current_mb),
                "peak_memory_gb_during_episode": _memory_gb(peak_mb),
                "gpu_sample_count": len(segment),
            }
        )


def _parse_openpi_metrics(output: str) -> dict[str, Any]:
    rates = re.findall(r"Total success rate:\s*([0-9.]+)", output)
    episodes = re.findall(r"Total episodes:\s*(\d+)", output)
    task_names = list(dict.fromkeys(re.findall(r"Task:\s*(.+)", output)))
    task_rates = [float(rate) for rate in re.findall(r"Current task success rate:\s*([0-9.]+)", output)]
    return {
        "success_rate": float(rates[-1]) if rates else None,
        "episodes": int(episodes[-1]) if episodes else None,
        "tasks": [
            {"task": task_name, "success_rate": task_rate}
            for task_name, task_rate in zip(task_names, task_rates, strict=False)
        ],
    }


def _parse_gr00t_metrics(output: str) -> dict[str, Any]:
    rates = re.findall(r"success rate:\s*([0-9.]+)", output)
    return {"success_rate": float(rates[-1]) if rates else None}


def _parse_openvla_metrics(output: str) -> dict[str, Any]:
    rates = re.findall(r"Total success rate:\s*([0-9.]+)", output)
    if not rates:
        rates = re.findall(r"Total success rate:\s*\d+/\d+\s*=\s*([0-9.]+)", output)
    if not rates:
        rates = re.findall(r"Current total success rate:\s*([0-9.]+)", output)
    episodes = re.findall(r"Total episodes:\s*(\d+)", output)
    if not episodes:
        episodes = re.findall(r"Total # episodes completed:\s*(\d+)", output)
    if not episodes:
        episodes = re.findall(r"# episodes completed so far:\s*(\d+)", output)
    task_names = list(dict.fromkeys(re.findall(r"Task:\s*(.+)", output)))
    task_rates = [float(rate) for rate in re.findall(r"Current task success rate:\s*([0-9.]+)", output)]
    success_rate = float(rates[-1]) if rates else None
    if success_rate is None and task_rates:
        success_rate = sum(task_rates) / len(task_rates)
    return {
        "success_rate": success_rate,
        "episodes": int(episodes[-1]) if episodes else None,
        "tasks": [
            {"task": task_name, "success_rate": task_rate}
            for task_name, task_rate in zip(task_names, task_rates, strict=False)
        ],
    }


def _write_summary(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    RESULTS_VOLUME.commit()
    return payload | {"summary_path": str(path)}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _parse_presets(value: str) -> list[QuantPreset]:
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one quantization preset is required.")
    unknown = [name for name in names if name not in QUANT_PRESETS]
    if unknown:
        raise ValueError(f"Unknown presets {unknown}. Expected a comma-separated subset of {tuple(QUANT_PRESETS)}.")
    return [QUANT_PRESETS[name] for name in names]


def _success_percent(payload: dict[str, Any]) -> float | None:
    rate = payload.get("metrics", {}).get("success_rate")
    return round(100.0 * rate, 3) if rate is not None else None


def _quantized_layer_count(model_key: str, preset: QuantPreset) -> int | None:
    if model_key not in QUANTIZABLE_TRANSFORMER_LAYERS:
        return None
    return QUANTIZABLE_TRANSFORMER_LAYERS[model_key] if preset.enabled else 0


def _matrix_rows(
    runs: list[dict[str, Any]],
    presets: list[QuantPreset],
    models: list[str],
    quantization_setting: str,
) -> list[dict[str, Any]]:
    by_key = {
        (run["model_key"], run["preset"], run["suite"]): _success_percent(run["result"])
        for run in runs
        if run.get("status") == "ok"
    }
    rows: list[dict[str, Any]] = []
    for preset in presets:
        row: dict[str, Any] = {
            "Method": preset.method,
            "Precision": preset.precision,
            "Quantization Setting": "No quantization" if not preset.enabled else quantization_setting,
        }
        for model_key, model_label in MODEL_LABELS.items():
            quantized_layers = _quantized_layer_count(model_key, preset)
            total_layers = QUANTIZABLE_TRANSFORMER_LAYERS.get(model_key)
            row[f"{model_label} Quantized Layers"] = quantized_layers
            row[f"{model_label} Total Transformer Layers"] = total_layers
            row[f"{model_label} Layer Coverage Percent"] = (
                round(100.0 * quantized_layers / total_layers, 3)
                if quantized_layers is not None and total_layers
                else None
            )
            values = []
            for suite in TABLE_SUITES:
                value = by_key.get((model_key, preset.name, suite)) if model_key in models else None
                row[f"{model_label} {TABLE_SUITE_LABELS[suite]}"] = value
                if value is not None:
                    values.append(value)
            row[f"{model_label} Average"] = round(sum(values) / len(values), 3) if values else None
        rows.append(row)
    return rows


def _write_matrix_outputs(
    output_dir: Path,
    runs: list[dict[str, Any]],
    presets: list[QuantPreset],
    models: list[str],
    quantization_setting: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _matrix_rows(runs, presets, models, quantization_setting)
    payload = {
        "models": models,
        "presets": [preset.name for preset in presets],
        "quantization_setting": quantization_setting,
        "runs": runs,
        "table": rows,
    }
    (output_dir / "runs.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "accuracy_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    memory_rows = []
    for run in runs:
        if run.get("status") != "ok":
            continue
        result = run["result"]
        for episode in result.get("episode_memory", []):
            memory_rows.append(
                {
                    "model": run["model_key"],
                    "preset": run["preset"],
                    "suite": run["suite"],
                    "run_name": result.get("run_name"),
                    **episode,
                }
            )
    memory_fields = [
        "model",
        "preset",
        "suite",
        "run_name",
        "episode",
        "reported_episode",
        "task",
        "memory_used_gb_at_completion",
        "peak_memory_gb_during_episode",
        "gpu_sample_count",
    ]
    with (output_dir / "episode_memory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=memory_fields)
        writer.writeheader()
        writer.writerows(memory_rows)
    layer_rows = []
    for preset in presets:
        for model_key in ("gr00t", "pi05"):
            if model_key not in models:
                continue
            quantized_layers = _quantized_layer_count(model_key, preset)
            total_layers = QUANTIZABLE_TRANSFORMER_LAYERS[model_key]
            layer_rows.append(
                {
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "preset": preset.name,
                    "precision": preset.precision,
                    "quantization_setting": (
                        "No quantization" if not preset.enabled else quantization_setting
                    ),
                    "quantized_transformer_layers": quantized_layers,
                    "total_transformer_layers": total_layers,
                    "layer_coverage_percent": round(100.0 * quantized_layers / total_layers, 3),
                }
            )
    with (output_dir / "quantized_layers.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "model",
            "model_label",
            "preset",
            "precision",
            "quantization_setting",
            "quantized_transformer_layers",
            "total_transformer_layers",
            "layer_coverage_percent",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(layer_rows)


def _max_memory_mb(samples: list[dict[str, Any]]) -> int | None:
    values = [sample["memory_used_mb"] for sample in samples if "memory_used_mb" in sample]
    return max(values) if values else None


def _memory_gb(memory_mb: int | None) -> float | None:
    return round(memory_mb / 1024, 3) if memory_mb is not None else None


def _mean_episode_peak_gb(records: list[dict[str, Any]]) -> float | None:
    values = [
        record["peak_memory_gb_during_episode"]
        for record in records
        if record.get("peak_memory_gb_during_episode") is not None
    ]
    return round(sum(values) / len(values), 3) if values else None


def _estimated_quantized_component_savings(bits: int) -> float | None:
    if bits <= 0:
        return None
    return round(100.0 * (1.0 - bits / 16.0), 2)


def _timing_summary(
    *,
    server_start_s: float | None,
    server_ready_s: float | None,
    eval_start_s: float | None,
    eval_end_s: float | None,
    run_start_s: float,
    run_end_s: float,
    episodes: int | None,
) -> dict[str, float | None]:
    eval_wall_s = (eval_end_s - eval_start_s) if eval_start_s is not None and eval_end_s is not None else None
    server_startup_s = (
        (server_ready_s - server_start_s)
        if server_start_s is not None and server_ready_s is not None
        else None
    )
    return {
        "server_startup_s": round(server_startup_s, 3) if server_startup_s is not None else None,
        "eval_wall_s": round(eval_wall_s, 3) if eval_wall_s is not None else None,
        "total_remote_wall_s": round(run_end_s - run_start_s, 3),
        "episodes_per_second": round(episodes / eval_wall_s, 6) if episodes and eval_wall_s and eval_wall_s > 0 else None,
        "seconds_per_episode": round(eval_wall_s / episodes, 3) if episodes and eval_wall_s and eval_wall_s > 0 else None,
    }


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


def _assert_gr00t_checkpoint_loaded(log_path: Path) -> None:
    log = log_path.read_text(encoding="utf-8", errors="replace")
    failure_markers = (
        "were not used when initializing Gr00tN1d7",
        "were not initialized from the model checkpoint",
        "are newly initialized",
    )
    if any(marker in log for marker in failure_markers):
        raise RuntimeError(
            "GR00T checkpoint did not load cleanly; refusing to run evaluation with "
            f"uninitialized weights. Inspect {log_path}."
        )


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
    channel_stats: bool = False,
    live_quant_bits: int = 0,
    live_quant_mode: str = "wht_lloydmax",
    quant_scope: str = "action_expert_mlp",
    quant_weight_bits: int = 3,
    quant_act_bits: int = 8,
    task_limit: int | None = None,
    quant_label: str = "",
) -> dict[str, Any]:
    """Evaluate pi0.5-LIBERO through OpenPI's native LIBERO client/server."""
    if suite not in LIBERO_SUITES:
        raise ValueError(f"Unknown suite {suite!r}. Expected one of {LIBERO_SUITES}.")

    run_start_s = time.perf_counter()
    server_start_s: float | None = None
    server_ready_s: float | None = None
    eval_start_s: float | None = None
    eval_end_s: float | None = None
    if quant_weight_bits < 0 or quant_act_bits < 0:
        raise ValueError("Quantization bit widths must be non-negative.")
    label = f"_{_slug(quant_label)}" if quant_label else ""
    run_name = f"pi05_{suite}{label}_{int(time.time())}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    quant_env = {
        "VLA_QUANT_LIVE_BITS": str(live_quant_bits),
        "VLA_QUANT_MODE": live_quant_mode,
        "VLA_QUANT_SCOPE": quant_scope,
        "VLA_QUANT_WEIGHT_BITS": str(quant_weight_bits),
        "VLA_QUANT_ACT_BITS": str(quant_act_bits),
    }
    if dump_quant:
        quant_env |= {
            "VLA_QUANT_DUMP_NPZ": str(run_dir / "kv_drift.npz"),
            "VLA_QUANT_LAYERS": "0",
            "VLA_QUANT_JAX_LAYER_STRIDE": "18",
            "VLA_QUANT_JAX_LAYER_OFFSET": "0",
            "VLA_QUANT_MAX_STEPS": "12",
            "VLA_QUANT_MAX_VALUES": "65536",
        }
    if channel_stats:
        quant_env |= {
            "VLA_QUANT_CHANNEL_STATS": str(run_dir / "channel_stats.json"),
            "VLA_QUANT_JAX_LAYER_STRIDE": "18",
            "VLA_QUANT_JAX_LAYER_OFFSET": "0",
            "VLA_QUANT_MAX_STEPS": "12",
        }
    stop = threading.Event()
    gpu_samples: list[dict[str, Any]] = []
    episode_memory = _EpisodeMemoryTracker(gpu_samples, suite=suite)
    monitor = threading.Thread(target=_sample_gpu, args=(stop, gpu_samples), daemon=True)
    monitor.start()

    server_log = run_dir / "server.log"
    server_start_s = time.perf_counter()
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
        server_ready_s = time.perf_counter()
        eval_start_s = time.perf_counter()
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
                "--args.task-limit",
                str(task_limit) if task_limit is not None else "0",
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
            line_callback=episode_memory.consume_line,
        )
        eval_end_s = time.perf_counter()
    finally:
        if eval_start_s is not None and eval_end_s is None:
            eval_end_s = time.perf_counter()
        stop.set()
        monitor.join(timeout=10)
        _terminate_process(server)
        RESULTS_VOLUME.commit()

    metrics = _parse_openpi_metrics(client_output)
    run_end_s = time.perf_counter()
    max_memory_mb = _max_memory_mb(gpu_samples)
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
            "quant_scope": quant_scope,
            "quant_weight_bits": quant_weight_bits,
            "quant_act_bits": quant_act_bits,
            "quant_label": quant_label,
            "task_limit": task_limit,
            "seed": seed,
            "metrics": metrics,
            "timing": _timing_summary(
                server_start_s=server_start_s,
                server_ready_s=server_ready_s,
                eval_start_s=eval_start_s,
                eval_end_s=eval_end_s,
                run_start_s=run_start_s,
                run_end_s=run_end_s,
                episodes=metrics.get("episodes"),
            ),
            "max_gpu_memory_mb": max_memory_mb,
            "max_gpu_memory_gb": _memory_gb(max_memory_mb),
            "mean_peak_memory_gb_per_episode": _mean_episode_peak_gb(episode_memory.records),
            "episode_memory": episode_memory.records,
            "estimated_quantized_component_relative_savings_percent": _estimated_quantized_component_savings(
                quant_weight_bits if live_quant_mode == "quantvla_itq3s" and live_quant_bits > 0 else live_quant_bits
            ),
            "quant_drift_npz": str(run_dir / "kv_drift.npz") if dump_quant else None,
            "channel_stats_json": str(run_dir / "channel_stats.json") if channel_stats else None,
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
    channel_stats: bool = False,
    live_quant_bits: int = 0,
    live_quant_mode: str = "wht_lloydmax",
    quant_scope: str = "action_expert_mlp",
    quant_weight_bits: int = 3,
    quant_act_bits: int = 8,
    task_limit: int | None = None,
    quant_label: str = "",
) -> dict[str, Any]:
    """Evaluate NVIDIA GR00T N1.7-LIBERO over the tasks in a LIBERO suite."""
    if suite not in GROOT_TASKS:
        raise ValueError(f"Unknown suite {suite!r}. Expected one of {tuple(GROOT_TASKS)}.")

    run_start_s = time.perf_counter()
    server_start_s: float | None = None
    server_ready_s: float | None = None
    eval_start_s: float | None = None
    eval_end_s: float | None = None
    if quant_weight_bits < 0 or quant_act_bits < 0:
        raise ValueError("Quantization bit widths must be non-negative.")
    label = f"_{_slug(quant_label)}" if quant_label else ""
    run_name = f"gr00t_n17_{suite}{label}_{int(time.time())}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    quant_env = {
        "VLA_QUANT_LIVE_BITS": str(live_quant_bits),
        "VLA_QUANT_MODE": live_quant_mode,
        "VLA_QUANT_SCOPE": quant_scope,
        "VLA_QUANT_WEIGHT_BITS": str(quant_weight_bits),
        "VLA_QUANT_ACT_BITS": str(quant_act_bits),
    }
    if dump_quant:
        quant_env |= {
            "VLA_QUANT_DUMP_NPZ": str(run_dir / "kv_drift.npz"),
            "VLA_QUANT_GROOT_LOG_LAYER": "0",
            "VLA_QUANT_MAX_STEPS": "12",
            "VLA_QUANT_MAX_VALUES": "65536",
            "GR00T_POLICY_CLIENT_TIMEOUT_MS": "600000",
        }
    if channel_stats:
        quant_env |= {
            "VLA_QUANT_CHANNEL_STATS": str(run_dir / "channel_stats.json"),
            "VLA_QUANT_GROOT_LOG_LAYER": "0",
            "VLA_QUANT_MAX_STEPS": "12",
        }
    stop = threading.Event()
    gpu_samples: list[dict[str, Any]] = []
    episode_memory = _EpisodeMemoryTracker(gpu_samples, suite=suite)
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
    server_start_s = time.perf_counter()
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
        _assert_gr00t_checkpoint_loaded(server_log)
        server_ready_s = time.perf_counter()
        eval_start_s = time.perf_counter()
        tasks = GROOT_TASKS[suite][:task_limit] if task_limit else GROOT_TASKS[suite]
        for task in tasks:
            episode_memory.task = task.removeprefix("libero_sim/")
            task_slug = task.removeprefix("libero_sim/").replace("/", "_")
            log_path = run_dir / f"client_{task_slug}.log"
            task_start_s = time.perf_counter()
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
                    line_callback=episode_memory.consume_line,
                )
                task_end_s = time.perf_counter()
                task_elapsed_s = task_end_s - task_start_s
                task_metrics = _parse_gr00t_metrics(output)
                task_results[task] = task_metrics | {
                    "status": "ok",
                    "log": str(log_path),
                    "episodes": episodes_per_task,
                    "elapsed_s": round(task_elapsed_s, 3),
                    "episodes_per_second": round(episodes_per_task / task_elapsed_s, 6)
                    if task_elapsed_s > 0
                    else None,
                    "seconds_per_episode": round(task_elapsed_s / episodes_per_task, 3)
                    if episodes_per_task > 0
                    else None,
                }
                task_success_rate = task_metrics["success_rate"]
                completed_rates = [
                    result["success_rate"]
                    for result in task_results.values()
                    if result["success_rate"] is not None
                ]
                print(f"Current task success rate: {task_success_rate}", flush=True)
                print(
                    f"Current total success rate: {sum(completed_rates) / len(completed_rates)}",
                    flush=True,
                )
            except subprocess.CalledProcessError as exc:
                task_elapsed_s = time.perf_counter() - task_start_s
                task_results[task] = {
                    "status": "failed",
                    "success_rate": None,
                    "returncode": exc.returncode,
                    "log": str(log_path),
                    "episodes": 0,
                    "elapsed_s": round(task_elapsed_s, 3),
                    "tail": "".join(exc.output.splitlines(keepends=True)[-80:]),
                }
                print(f"GR00T rollout failed for {task}; continuing so logs and partial outputs are saved.", flush=True)
                break
        eval_end_s = time.perf_counter()
    finally:
        if eval_start_s is not None and eval_end_s is None:
            eval_end_s = time.perf_counter()
        stop.set()
        monitor.join(timeout=10)
        _terminate_process(server)
        RESULTS_VOLUME.commit()

    rates = [result["success_rate"] for result in task_results.values() if result["success_rate"] is not None]
    completed_episodes = sum(result.get("episodes", 0) for result in task_results.values() if result.get("status") == "ok")
    run_end_s = time.perf_counter()
    max_memory_mb = _max_memory_mb(gpu_samples)
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
            "quant_scope": quant_scope,
            "quant_weight_bits": quant_weight_bits,
            "quant_act_bits": quant_act_bits,
            "quant_label": quant_label,
            "task_limit": task_limit,
            "seed": seed,
            "metrics": {
                "success_rate": (sum(rates) / len(rates)) if rates else None,
                "episodes": completed_episodes,
                "tasks": task_results,
            },
            "timing": _timing_summary(
                server_start_s=server_start_s,
                server_ready_s=server_ready_s,
                eval_start_s=eval_start_s,
                eval_end_s=eval_end_s,
                run_start_s=run_start_s,
                run_end_s=run_end_s,
                episodes=completed_episodes,
            ),
            "max_gpu_memory_mb": max_memory_mb,
            "max_gpu_memory_gb": _memory_gb(max_memory_mb),
            "mean_peak_memory_gb_per_episode": _mean_episode_peak_gb(episode_memory.records),
            "episode_memory": episode_memory.records,
            "estimated_quantized_component_relative_savings_percent": _estimated_quantized_component_savings(
                quant_weight_bits if live_quant_mode == "quantvla_itq3s" and live_quant_bits > 0 else live_quant_bits
            ),
            "quant_drift_npz": str(run_dir / "kv_drift.npz") if dump_quant else None,
            "channel_stats_json": str(run_dir / "channel_stats.json") if channel_stats else None,
            "gpu_samples": gpu_samples,
            "logs": {"server": str(run_dir / "server.log"), "task_logs_dir": str(run_dir)},
        },
    )


@app.function(
    image=OPENVLA_IMAGE,
    gpu="A100-80GB",
    timeout=24 * HOURS,
    startup_timeout=2 * HOURS,
    # The OpenVLA image contains build-time files under /cache, so Modal
    # requires this volume to use a separate, empty mount point.
    volumes={"/results": RESULTS_VOLUME, str(OPENVLA_CACHE_DIR): CACHE_VOLUME},
)
def run_openvla(
    suite: str = "libero_spatial",
    trials_per_task: int = 1,
    seed: int = 7,
    checkpoint: str = "",
) -> dict[str, Any]:
    """Run the official OpenVLA LIBERO checkpoint without custom quantization."""
    if suite not in LIBERO_SUITES:
        raise ValueError(f"Unknown suite {suite!r}. Expected one of {LIBERO_SUITES}.")
    if trials_per_task <= 0:
        raise ValueError("trials_per_task must be positive.")

    checkpoint = checkpoint or OPENVLA_CHECKPOINTS[suite]
    run_start_s = time.perf_counter()
    run_name = f"openvla_{suite}_default_{int(time.time())}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    gpu_samples: list[dict[str, Any]] = []
    episode_memory = _EpisodeMemoryTracker(gpu_samples, suite=suite)
    monitor = threading.Thread(target=_sample_gpu, args=(stop, gpu_samples), daemon=True)
    monitor.start()
    eval_start_s = time.perf_counter()
    eval_end_s: float | None = None
    try:
        output = _run_streaming(
            [
                "python",
                "experiments/robot/libero/run_libero_eval.py",
                "--model_family",
                "openvla",
                "--pretrained_checkpoint",
                checkpoint,
                "--task_suite_name",
                suite,
                "--center_crop",
                "True",
                "--num_trials_per_task",
                str(trials_per_task),
                "--seed",
                str(seed),
                "--use_wandb",
                "False",
            ],
            cwd=OPENVLA_DIR,
            env=COMMON_ENV
            | {
                "HF_HOME": str(OPENVLA_CACHE_DIR / "huggingface"),
                "HF_HUB_CACHE": str(OPENVLA_CACHE_DIR / "huggingface/hub"),
                "TRANSFORMERS_CACHE": str(OPENVLA_CACHE_DIR / "huggingface/hub"),
                "PYTHONPATH": f"{OPENVLA_DIR}:/workspace/LIBERO",
            },
            log_path=run_dir / "eval.log",
            line_callback=episode_memory.consume_line,
        )
        eval_end_s = time.perf_counter()
    finally:
        if eval_end_s is None:
            eval_end_s = time.perf_counter()
        stop.set()
        monitor.join(timeout=10)
        RESULTS_VOLUME.commit()
        CACHE_VOLUME.commit()

    metrics = _parse_openvla_metrics(output)
    run_end_s = time.perf_counter()
    max_memory_mb = _max_memory_mb(gpu_samples)
    return _write_summary(
        run_name,
        {
            "run_name": run_name,
            "model": "OpenVLA 7B Default",
            "checkpoint": checkpoint,
            "suite": suite,
            "trials_per_task": trials_per_task,
            "seed": seed,
            "precision": "Official checkpoint default",
            "metrics": metrics,
            "timing": _timing_summary(
                server_start_s=None,
                server_ready_s=None,
                eval_start_s=eval_start_s,
                eval_end_s=eval_end_s,
                run_start_s=run_start_s,
                run_end_s=run_end_s,
                episodes=metrics.get("episodes"),
            ),
            "max_gpu_memory_mb": max_memory_mb,
            "max_gpu_memory_gb": _memory_gb(max_memory_mb),
            "mean_peak_memory_gb_per_episode": _mean_episode_peak_gb(episode_memory.records),
            "episode_memory": episode_memory.records,
            "gpu_samples": gpu_samples,
            "logs": {"eval": str(run_dir / "eval.log")},
        },
    )


@app.local_entrypoint()
def run_all(episodes_per_task: int = 20, trials_per_task: int = 50, suite: str = "all") -> None:
    """Run pi0.5 and GR00T baselines. Use run_quant_matrix to include OpenVLA."""
    suites = list(LIBERO_SUITES) if suite == "all" else [suite]
    for suite_name in suites:
        print(json.dumps(run_pi05.remote(suite=suite_name, trials_per_task=trials_per_task), indent=2))
        print(json.dumps(run_gr00t.remote(suite=suite_name, episodes_per_task=episodes_per_task), indent=2))


@app.local_entrypoint()
def run_quant_matrix(
    models: str = "gr00t,pi05,openvla",
    presets: str = "fp16,w4a4,w4a8,a8",
    suites: str = "all",
    trials_per_task: int = 50,
    openvla_trials_per_task: int = 1,
    episodes_per_task: int = 20,
    n_envs: int = 5,
    seed: int = 7,
    replan_steps: int = 5,
    quant_scope: str = "action_expert_mlp",
    task_limit: int = 0,
    output_dir: str = "",
    resume: bool = True,
) -> None:
    """Run a reproducible LIBERO model/precision matrix and write a paper-style table."""
    model_names = [name.strip().lower() for name in models.split(",") if name.strip()]
    aliases = {"pi0.5": "pi05", "pi0_5": "pi05", "groot": "gr00t"}
    model_names = [aliases.get(name, name) for name in model_names]
    unknown_models = [name for name in model_names if name not in MODEL_LABELS]
    if not model_names or unknown_models:
        raise ValueError(f"Models must be a comma-separated subset of {tuple(MODEL_LABELS)}; got {models!r}.")

    selected_presets = _parse_presets(presets)
    suite_names = list(LIBERO_SUITES) if suites.strip().lower() == "all" else [
        name.strip() for name in suites.split(",") if name.strip()
    ]
    unknown_suites = [name for name in suite_names if name not in LIBERO_SUITES]
    if not suite_names or unknown_suites:
        raise ValueError(f"Suites must be 'all' or a comma-separated subset of {LIBERO_SUITES}.")
    if quant_scope not in {"action_expert_mlp", "full"}:
        raise ValueError("quant_scope must be 'action_expert_mlp' or 'full'.")

    setting = "w/o DiT attention" if quant_scope == "action_expert_mlp" else "Full quantization"
    destination = Path(output_dir) if output_dir else Path("reports") / f"libero_quant_matrix_{int(time.time())}"
    runs: list[dict[str, Any]] = []
    checkpoint_path = destination / "runs.json"
    if resume and checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        runs = list(payload.get("runs", []))
        completed = sum(1 for run in runs if run.get("status") == "ok")
        print(f"Resuming from {checkpoint_path}; found {completed} completed run(s).", flush=True)
    print(f"Writing matrix checkpoints to {destination.resolve()}", flush=True)

    for preset in selected_presets:
        for model_name in model_names:
            for suite_name in suite_names:
                if model_name == "openvla" and preset.enabled:
                    continue
                existing_index = next(
                    (
                        idx
                        for idx, run in enumerate(runs)
                        if run.get("model_key") == model_name
                        and run.get("preset") == preset.name
                        and run.get("suite") == suite_name
                    ),
                    None,
                )
                if existing_index is not None and runs[existing_index].get("status") == "ok":
                    print(
                        f"Skipping completed {model_name} {preset.precision} on {suite_name}",
                        flush=True,
                    )
                    continue
                record: dict[str, Any] = {
                    "model_key": model_name,
                    "preset": preset.name,
                    "suite": suite_name,
                    "status": "running",
                }
                if existing_index is None:
                    runs.append(record)
                else:
                    runs[existing_index] = record
                _write_matrix_outputs(destination, runs, selected_presets, model_names, setting)
                print(f"\nRunning {model_name} {preset.precision} on {suite_name}", flush=True)
                kwargs = {
                    "suite": suite_name,
                    "seed": seed,
                    "live_quant_bits": 1 if preset.enabled else 0,
                    "live_quant_mode": "quantvla_itq3s",
                    "quant_weight_bits": preset.weight_bits,
                    "quant_act_bits": preset.act_bits,
                    "task_limit": task_limit or None,
                    "quant_label": preset.name,
                }
                try:
                    if model_name == "openvla":
                        result = run_openvla.remote(
                            suite=suite_name,
                            trials_per_task=openvla_trials_per_task,
                            seed=seed,
                        )
                    elif model_name == "pi05":
                        result = run_pi05.remote(
                            **kwargs,
                            trials_per_task=trials_per_task,
                            replan_steps=replan_steps,
                            quant_scope=quant_scope,
                        )
                    else:
                        result = run_gr00t.remote(
                            **kwargs,
                            episodes_per_task=episodes_per_task,
                            n_envs=n_envs,
                            quant_scope=quant_scope,
                        )
                    record.update(status="ok", result=result)
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    print(f"Run failed: {record['error']}", flush=True)
                _write_matrix_outputs(destination, runs, selected_presets, model_names, setting)

    print(f"\nWrote {destination / 'accuracy_table.csv'}", flush=True)
    print(f"Wrote {destination / 'runs.json'}", flush=True)


if __name__ == "__main__":
    print(
        "Use Modal to run this file, e.g. "
        "`modal run scripts/modal_libero_baselines.py::run_pi05 --suite libero_spatial`.",
        file=sys.stderr,
    )
