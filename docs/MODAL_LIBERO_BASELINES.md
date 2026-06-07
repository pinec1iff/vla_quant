# Modal LIBERO Baselines

This workspace includes a Modal runner for LIBERO baseline evaluations:

- pi0.5 via OpenPI's `pi05_libero` checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- Isaac GR00T N1.7 via NVIDIA's LIBERO checkpoint: `nvidia/GR00T-N1.7-LIBERO`

The runner records native success-rate output, `nvidia-smi` GPU memory samples, and wall-clock inference timing. Summaries and raw logs are written to the `vla-quant-libero-results` Modal volume under `/results/libero_baselines`.

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
- `metrics.episodes`
- `max_gpu_memory_mb` and `max_gpu_memory_gb`
- `timing.server_startup_s`
- `timing.eval_wall_s`
- `timing.seconds_per_episode`
- `timing.episodes_per_second`
- `gpu_samples`
- log paths for server and client output

Compare a baseline JSON against a quantized JSON:

```bash
python tools/compare_libero_runs.py \
  --baseline-json reports/comparison_pi05/base/pi05_baseline.json \
  --quantized-json reports/comparison_pi05/quant/pi05_quantized.json \
  --out reports/comparison_pi05/accuracy_memory_timing.csv
```

## Notes

- The current runner uses `A100-40GB` for pi0.5 and `H100` for GR00T. If you want stricter apples-to-apples memory or latency measurements, change the `gpu=` argument in `scripts/modal_libero_baselines.py` and rerun both baseline and quantized conditions on the same GPU type.
- The first run will be slow because Modal builds images and downloads checkpoints. Later runs reuse the `vla-quant-model-cache` volume.
- pi0.5 uses OpenPI's native two-process LIBERO evaluation path. GR00T uses NVIDIA's native GR00T server plus `rollout_policy.py` over every task in the requested suite.


# Modal List
modal volume ls vla-quant-libero-results /libero_baselines
modal volume ls vla-quant-libero-results /libero_baselines/pi05_libero_spatial_1779652538

# Modal Download
modal volume get vla-quant-libero-results /libero_baselines/pi05_libero_spatial_1779652538 ./pi05_libero_spatial_1779652538
modal volume get vla-quant-libero-results /libero_baselines/pi05_libero_spatial_1779652538.json ./pi05_libero_spatial_1779652538.json

# Run Smoke Test
modal run scripts/modal_libero_baselines.py::run_gr00t \
    --suite libero_spatial \
    --episodes-per-task 1 \
    --n-envs 1 \
    --task-limit 1 \
    --dump-quant \
    --live-quant-bits 3 \
    --live-quant-mode wht_lloydmax

# Get Plots for Smoke
MPLCONFIGDIR=/tmp/mpl python tools/vla_quant_drift_probe.py \
  --baseline-npz reports/comparison_gr00t/base/kv_drift.npz \
  --quantized-npz reports/comparison_gr00t/quant/kv_drift.npz \
  --out reports/comparison_gr00t/gr00t_unquantized_vs_wht_lloydmax.png \
  --csv reports/comparison_gr00t/gr00t_unquantized_vs_wht_lloydmax.csv \
  --bits 3
