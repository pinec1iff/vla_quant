# Modal LIBERO Baselines

This workspace includes a Modal runner for LIBERO baseline evaluations:

- pi0.5 via OpenPI's `pi05_libero` checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- Isaac GR00T N1.7 via NVIDIA's LIBERO checkpoint: `nvidia/GR00T-N1.7-LIBERO`
- OpenVLA 7B via the official suite-specific `openvla/openvla-7b-finetuned-libero-*` checkpoints

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

## QuantVLA accuracy matrix

Run full precision, W4A4, W4A8, and activation-only A8 over all four suites. OpenVLA is evaluated only with its official default checkpoint and inference settings; none of the custom QuantVLA hooks or presets are applied to it:

```bash
modal run scripts/modal_libero_baselines.py::run_quant_matrix \
  --models gr00t,pi05,openvla \
  --presets fp16,w4a4,w4a8,a8 \
  --suites all \
  --trials-per-task 50 \
  --openvla-trials-per-task 1 \
  --episodes-per-task 20
```

Run just one model or a smoke-test subset:

```bash
modal run scripts/modal_libero_baselines.py::run_quant_matrix \
  --models pi05 \
  --presets w4a4,w4a8,a8 \
  --suites libero_spatial \
  --trials-per-task 1 \
  --task-limit 1
```

The local output directory is `reports/libero_quant_matrix_<timestamp>/` by default:

- `accuracy_table.csv` has `Goal`, `Spatial`, `Object`, `Long`, and `Average` columns for all three models.
- `runs.json` stores every raw Modal result, run status, checkpoint path, timing, and memory measurement.
- `episode_memory.csv` contains one row per completed episode with GPU memory in GB at completion and the peak observed during that episode.
- `quantized_layers.csv` contains quantized-layer counts and coverage percentages for GR00T and Pi 0.5 under every precision preset.

The runner checkpoints these files after every condition, so completed results remain available if a later Modal job fails. Use `--output-dir PATH` for a stable destination. The built-in presets are `fp16`, `w4a4`, `w4a8`, and `a8`; `a8` leaves weights untouched. Add `--quant-scope full` to label and run the full-quantization condition instead of the default action-expert/DiT-MLP condition.

Layer coverage counts unique transformer blocks containing an enabled quantization operation:

- GR00T: 12 DiT transformer blocks.
- Pi 0.5: 18 transformer blocks containing the action-expert MLP.
- FP16: zero quantized layers.
- W4A4, W4A8, and A8: all eligible layers are counted. A8 counts because activation quantization is active even though weights remain unchanged.
- Full quantization adds attention operations within the same blocks, so it does not double-count those blocks.
- OpenVLA is excluded because it runs unmodified.

Plot the layer coverage:

```bash
MPLCONFIGDIR=/tmp/mpl python tools/plot_quantized_layers.py \
  --layers-csv reports/libero_quant_matrix_<timestamp>/quantized_layers.csv \
  --out reports/quantized_layers.png
```

Run only the OpenVLA sanity check:

```bash
modal run scripts/modal_libero_baselines.py::run_quant_matrix \
  --models openvla \
  --presets fp16 \
  --suites all \
  --openvla-trials-per-task 1
```

`--openvla-trials-per-task` defaults to one because OpenVLA is included as a quick sanity check. Set it to `50` for the official 500-rollout-per-suite protocol. OpenVLA runs only once per suite under the `fp16` matrix row for table organization, but uses the official evaluator's default model dtype and has no custom quantization environment variables. Each run JSON also includes `episode_memory` and `mean_peak_memory_gb_per_episode`. Memory is sampled every two seconds, so very short episodes may have zero samples; `gpu_sample_count` makes those rows easy to filter before plotting.

## Low-overhead channel diagnostics

Use `--channel-stats` instead of `--dump-quant` when you only need channel RMS and quantization-MSE plots:

```bash
modal run scripts/modal_libero_baselines.py::run_gr00t \
  --suite libero_spatial \
  --episodes-per-task 1 \
  --n-envs 1 \
  --task-limit 1 \
  --live-quant-bits 1 \
  --live-quant-mode quantvla_itq3s \
  --quant-weight-bits 4 \
  --quant-act-bits 4 \
  --channel-stats
```

This reduces tensors to per-channel RMS, maximum magnitude, and MSE on the accelerator. It writes a small `channel_stats.json`, does not store full activations, and does not create an `.npz`. Collection stops after the first 12 instrumented steps.

Plot the downloaded statistics:

```bash
MPLCONFIGDIR=/tmp/mpl python tools/plot_channel_stats.py \
  --stats-json path/to/channel_stats.json \
  --out reports/channel_stats.png
```

Keep both `--channel-stats` and `--dump-quant` disabled for full accuracy, latency, and memory runs.

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
- `episode_memory`
- `mean_peak_memory_gb_per_episode`
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

# Run Smoke Test (Used to get the .png)
# WARNING Dump Quant will dump .npz file which was fetched from modal under 
# /reports/comparison_gr00t and /reports/comparison_pi05
# --dump quant will add SIGNIFICANT OVERHEAD TO RUNNING!!!!!!!!!!!!!
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

# Get accuracy, episode_memory, quantized_layers for pi and groot
# If you want to reproduce. Note, you must have modal because GCP
# is milking my money for not running anything!!!!!!!!!!!!!!
modal run scripts/modal_libero_baselines.py::run_quant_matrix \
  --models pi05 \
  --presets fp16,w4a4,w4a8,a8 \
  --suites all \
  --trials-per-task 20 \

modal run --detach scripts/modal_libero_baselines.py::run_quant_matrix \    
  --models gr00t \               
  --presets fp16,w4a4,w4a8,a8 \
  --suites all \                                                             
  --episodes-per-task 15 \       
  --n-envs 5 \
  --output-dir reports/gr00t_matrix


# AI Use: used to generate scaffolding for Modal and wrapper to run modal and gr00t baselines. Also used to debug the JAX hookup issues and weird wheel dependencies that came with OpenVLA (the default F16 baseline) and Gr00t + generalist pyplot/matplotlib infra plotting syntax infra since I can't remember matplotlib syntax for plotting especially the fancy stuff. Core quantization implementation done by me under the third_party_repos (pi and groot). Note to self: look into writing kernels next time.  

# Output data of interest for ease of access:
# Note: Plots of interest are .png files for drift under reports/comparison_* 
# Also gr00t_matrix and pi05_matrix and openvla_sanity contain memory, quantized layers, and accuracy info mentioned in report