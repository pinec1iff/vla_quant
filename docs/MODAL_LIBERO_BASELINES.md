# Modal LIBERO Baselines

This workspace includes a Modal runner for LIBERO baseline evaluations:

- pi0.5 via OpenPI's `pi05_libero` checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- Isaac GR00T N1.7 via NVIDIA's LIBERO checkpoint: `nvidia/GR00T-N1.7-LIBERO`

The runner records native success-rate output plus `nvidia-smi` GPU memory samples. Summaries and raw logs are written to the `vla-quant-libero-results` Modal volume under `/results/libero_baselines`.

## One-time local setup

```bash
pip install modal
modal setup
```

The current runner uses public checkpoint locations. If Hugging Face access policy changes for your account, add a Modal secret and wire it into `run_gr00t` with `secrets=[modal.Secret.from_name("huggingface-secret")]`.

## Quick smoke tests

Run tiny jobs first to validate build, downloads, EGL, and policy server/client wiring:

```bash
modal run scripts/modal_libero_baselines.py::run_pi05 --suite libero_spatial --trials-per-task 1
modal run scripts/modal_libero_baselines.py::run_gr00t --suite libero_spatial --episodes-per-task 1 --n-envs 1
```

## Full default baselines

OpenPI's published LIBERO numbers use 50 trials per task:

```bash
modal run scripts/modal_libero_baselines.py::run_pi05 --suite libero_spatial --trials-per-task 50
modal run scripts/modal_libero_baselines.py::run_pi05 --suite libero_object --trials-per-task 50
modal run scripts/modal_libero_baselines.py::run_pi05 --suite libero_goal --trials-per-task 50
modal run scripts/modal_libero_baselines.py::run_pi05 --suite libero_10 --trials-per-task 50
```

NVIDIA's GR00T LIBERO README reports 200 episodes per suite. With ten tasks per suite, that is 20 episodes per task:

```bash
modal run scripts/modal_libero_baselines.py::run_gr00t --suite libero_spatial --episodes-per-task 20
modal run scripts/modal_libero_baselines.py::run_gr00t --suite libero_object --episodes-per-task 20
modal run scripts/modal_libero_baselines.py::run_gr00t --suite libero_goal --episodes-per-task 20
modal run scripts/modal_libero_baselines.py::run_gr00t --suite libero_10 --episodes-per-task 20
```

Run all suites for both models:

```bash
modal run scripts/modal_libero_baselines.py::run_all --suite all --trials-per-task 50 --episodes-per-task 20
```

## Results

List summaries:

```bash
modal volume ls vla-quant-libero-results /libero_baselines
```

Download all logs and JSON:

```bash
modal volume get vla-quant-libero-results /libero_baselines ./modal_libero_results
```

Each summary JSON includes:

- `metrics.success_rate`
- `max_gpu_memory_mb`
- `gpu_samples`
- log paths for server and client output

## Notes

- The default GPU is `L40S`, which has 48 GB VRAM and is a good first target for inference. If you want stricter apples-to-apples memory measurements, change the `gpu=` argument in `scripts/modal_libero_baselines.py` and rerun both models on the same GPU type.
- The first run will be slow because Modal builds images and downloads checkpoints. Later runs reuse the `vla-quant-model-cache` volume.
- pi0.5 uses OpenPI's native two-process LIBERO evaluation path. GR00T uses NVIDIA's native GR00T server plus `rollout_policy.py` over every task in the requested suite.
