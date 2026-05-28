"""Quantization drift probe for VLA KV/activation tensors.

This script is intentionally lightweight: it can plot a synthetic sanity check
immediately, or consume an `.npz` produced by future attention hooks.

Expected NPZ format:
    arrays named like `<component>__step<idx>`, e.g.
    `pi05_action_k__step0`, `pi05_action_k__step9`,
    `gr00t_cross_k__step0`.

Each array can have any shape; the last dimension is treated as channel/head
dimension and everything else is flattened.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class StepStats:
    component: str
    step: int
    mean: float
    std: float
    rms: float
    kurtosis: float
    static_mse: float
    per_step_mse: float


@dataclass
class CompareStats:
    condition: str
    component: str
    step: int
    rms: float
    relative_drift: float
    static_mse: float
    per_step_mse: float
    live_mse: float | None


def _flatten(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1)


def symmetric_quantize(x: np.ndarray, bits: int, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Symmetric fake quantization with signed integer range."""
    qmax = (2 ** (bits - 1)) - 1
    if scale is None:
        scale = float(np.max(np.abs(x)) / max(qmax, 1))
    scale = max(scale, 1e-8)
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return q * scale, scale


def hadamard(x: np.ndarray) -> np.ndarray:
    """Apply normalized Walsh-Hadamard transform over the last dimension."""
    n = x.shape[-1]
    if n & (n - 1):
        return x
    y = x.astype(np.float32, copy=True)
    h = 1
    while h < n:
        y = y.reshape(-1, n)
        for i in range(0, n, h * 2):
            a = y[:, i : i + h].copy()
            b = y[:, i + h : i + 2 * h].copy()
            y[:, i : i + h] = a + b
            y[:, i + h : i + 2 * h] = a - b
        h *= 2
    return (y / np.sqrt(n)).reshape(x.shape)


def demo_tensors(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    shape = (8, 16, 256)
    for step in range(10):
        data[f"pi05_prefix_k__step{step}"] = rng.normal(0.0, 1.0, shape)
        drift = step / 9
        data[f"pi05_action_k__step{step}"] = rng.normal(0.35 * drift, 1.0 + 0.9 * drift, shape)
    for step in range(4):
        drift = step / 3
        data[f"gr00t_cross_k__step{step}"] = rng.normal(0.2 * drift, 1.0 + 0.65 * drift, shape)
    return data


def load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path)
    return {k: raw[k] for k in raw.files}


def parse_key(name: str) -> tuple[str, int]:
    component, step = name.rsplit("__step", 1)
    return component, int(step)


def compute_stats(data: dict[str, np.ndarray], bits: int, rotate: bool) -> list[StepStats]:
    grouped: dict[str, dict[int, np.ndarray]] = {}
    for name, arr in data.items():
        component, step = parse_key(name)
        x = hadamard(arr) if rotate else arr
        grouped.setdefault(component, {})[step] = _flatten(x)

    out: list[StepStats] = []
    for component, by_step in grouped.items():
        first_step = min(by_step)
        _, static_scale = symmetric_quantize(by_step[first_step], bits)
        for step in sorted(by_step):
            x = by_step[step]
            x_static, _ = symmetric_quantize(x, bits, static_scale)
            x_step, _ = symmetric_quantize(x, bits, None)
            centered = x - np.mean(x)
            std = float(np.std(x) + 1e-8)
            out.append(
                StepStats(
                    component=component,
                    step=step,
                    mean=float(np.mean(x)),
                    std=std,
                    rms=float(np.sqrt(np.mean(x * x))),
                    kurtosis=float(np.mean((centered / std) ** 4)),
                    static_mse=float(np.mean((x - x_static) ** 2)),
                    per_step_mse=float(np.mean((x - x_step) ** 2)),
                )
            )
    return out


def _component_steps(data: dict[str, np.ndarray], component: str) -> dict[int, np.ndarray]:
    out = {}
    prefix = f"{component}__step"
    for name, arr in data.items():
        if name.startswith(prefix):
            out[int(name.removeprefix(prefix))] = arr
    return out


def _pair_components(data: dict[str, np.ndarray]) -> list[str]:
    components = sorted({parse_key(name)[0] for name in data})
    return [component for component in components if not component.endswith("_quant")]


def _kv_components(data: dict[str, np.ndarray]) -> tuple[str | None, str | None]:
    components = _pair_components(data)
    key = next((c for c in components if c.endswith("_k")), None)
    value = next((c for c in components if c.endswith("_v")), None)
    return key, value


def compute_comparison_stats(
    baseline: dict[str, np.ndarray],
    quantized: dict[str, np.ndarray],
    bits: int,
    rotate: bool,
) -> list[CompareStats]:
    rows: list[CompareStats] = []
    for condition, data in (("unquantized", baseline), ("wht_lloydmax", quantized)):
        for component in _pair_components(data):
            by_step = {
                step: _flatten(hadamard(arr) if rotate else arr)
                for step, arr in _component_steps(data, component).items()
            }
            if not by_step:
                continue
            first_step = min(by_step)
            _, static_scale = symmetric_quantize(by_step[first_step], bits)
            base_rms = float(np.sqrt(np.mean(by_step[first_step] * by_step[first_step])))
            q_component = f"{component}_quant"
            q_by_step = {
                step: _flatten(hadamard(arr) if rotate else arr)
                for step, arr in _component_steps(data, q_component).items()
            }
            for step in sorted(by_step):
                x = by_step[step]
                x_static, _ = symmetric_quantize(x, bits, static_scale)
                x_step, _ = symmetric_quantize(x, bits, None)
                rms = float(np.sqrt(np.mean(x * x)))
                live_mse = None
                if step in q_by_step:
                    qx = q_by_step[step]
                    n = min(x.shape[0], qx.shape[0])
                    live_mse = float(np.mean((x[:n] - qx[:n]) ** 2))
                rows.append(
                    CompareStats(
                        condition=condition,
                        component=component,
                        step=step,
                        rms=rms,
                        relative_drift=abs(rms - base_rms) / max(base_rms, 1e-8),
                        static_mse=float(np.mean((x - x_static) ** 2)),
                        per_step_mse=float(np.mean((x - x_step) ** 2)),
                        live_mse=live_mse,
                    )
                )
    return rows


def plot_comparison(
    baseline: dict[str, np.ndarray],
    quantized: dict[str, np.ndarray],
    rows: list[CompareStats],
    out_path: Path,
) -> None:
    components = sorted({row.component for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), constrained_layout=True)
    key_component, value_component = _kv_components(baseline)
    model_name = "GR00T" if any(c.startswith("gr00t") for c in components) else "pi0.5"
    condition_colors = {"unquantized": "#1f77b4", "wht_lloydmax": "#d62728"}
    series_colors = {
        ("unquantized", key_component): "#1f77b4",   # blue
        ("unquantized", value_component): "#ff7f0e",  # orange
        ("wht_lloydmax", key_component): "#d62728",  # red
        ("wht_lloydmax", value_component): "#2ca02c",  # green
    }
    component_styles = {
        key_component: {"linestyle": "-", "marker": "o", "suffix": "K"},
        value_component: {"linestyle": "--", "marker": "s", "suffix": "V"},
    }

    ax = axes[0, 0]
    for condition in ("unquantized", "wht_lloydmax"):
        for component in components:
            vals = [r for r in rows if r.condition == condition and r.component == component]
            if vals:
                ax.plot(
                    [r.step for r in vals],
                    [r.rms for r in vals],
                    marker=component_styles.get(component, {}).get("marker", "o"),
                    color=series_colors.get((condition, component), condition_colors[condition]),
                    linestyle=component_styles.get(component, {}).get("linestyle", "-"),
                    label=f"{condition} {component_styles.get(component, {}).get('suffix', component)}",
                )
    ax.set_title("RMS energy across denoising step")
    ax.set_xlabel("denoising step")
    ax.set_ylabel("RMS")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for condition in ("unquantized", "wht_lloydmax"):
        for component in components:
            vals = [r for r in rows if r.condition == condition and r.component == component]
            if vals:
                ax.plot(
                    [r.step for r in vals],
                    [r.relative_drift for r in vals],
                    marker=component_styles.get(component, {}).get("marker", "o"),
                    color=series_colors.get((condition, component), condition_colors[condition]),
                    linestyle=component_styles.get(component, {}).get("linestyle", "-"),
                    label=f"{condition} {component_styles.get(component, {}).get('suffix', component)}",
                )
    ax.set_title("Relative drift from first denoising step")
    ax.set_xlabel("denoising step")
    ax.set_ylabel("|RMS_t - RMS_0| / RMS_0")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    metrics = [
        ("static", lambda vals: [r.static_mse for r in vals]),
        ("per-step", lambda vals: [r.per_step_mse for r in vals]),
        ("live WHT-LM", lambda vals: [r.live_mse for r in vals if r.live_mse is not None]),
    ]
    x = np.arange(len(metrics))
    width = 0.34
    for offset, condition in ((-width / 2, "unquantized"), (width / 2, "wht_lloydmax")):
        vals = [r for r in rows if r.condition == condition]
        heights = []
        for _, getter in metrics:
            metric_vals = getter(vals)
            heights.append(float(np.mean(metric_vals)) if metric_vals else 0.0)
        ax.bar(
            x + offset,
            heights,
            width,
            color=condition_colors[condition],
            alpha=0.85,
            label=condition,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in metrics])
    ax.set_ylabel("mean MSE")
    ax.set_title("Static vs step-indexed vs live quantization error")
    ax.legend(fontsize=8, title="condition")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    quant_key_component, quant_value_component = f"{key_component}_quant", f"{value_component}_quant"
    hist_specs = [
        ("unquantized K", baseline, key_component, "-", series_colors[("unquantized", key_component)]),
        ("unquantized V", baseline, value_component, "--", series_colors[("unquantized", value_component)]),
        ("WHT-LM K", quantized, quant_key_component, "-", series_colors[("wht_lloydmax", key_component)]),
        ("WHT-LM V", quantized, quant_value_component, "--", series_colors[("wht_lloydmax", value_component)]),
    ]
    for label, data, component, style, color in hist_specs:
        by_step = _component_steps(data, component)
        if not by_step:
            continue
        step = max(by_step)
        x = by_step[step].reshape(-1)
        bins = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 80)
        ax.hist(x, bins=bins, density=True, histtype="step", linestyle=style, color=color, label=label)
    ax.set_title("Late-step K/V distribution: unquantized vs WHT-Lloyd-Max")
    ax.set_xlabel("K/V value")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)

    fig.suptitle(f"{model_name} KV quantization comparison", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


def plot_report(data: dict[str, np.ndarray], stats: list[StepStats], out_path: Path) -> None:
    components = sorted({s.component for s in stats})
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    for component in components:
        xs = [s.step for s in stats if s.component == component]
        ys = [s.rms for s in stats if s.component == component]
        ax.plot(xs, ys, marker="o", label=component)
    ax.set_title("RMS energy across denoising step")
    ax.set_xlabel("step")
    ax.set_ylabel("RMS")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for component in components:
        xs = [s.step for s in stats if s.component == component]
        base = next(s.rms for s in stats if s.component == component and s.step == min(xs))
        ys = [abs(s.rms - base) / max(base, 1e-8) for s in stats if s.component == component]
        ax.plot(xs, ys, marker="o", label=component)
    ax.set_title("Relative drift from first step")
    ax.set_xlabel("step")
    ax.set_ylabel("|RMS_t - RMS_0| / RMS_0")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    width = 0.35
    xloc = np.arange(len(components))
    static = []
    adaptive = []
    for component in components:
        vals = [s for s in stats if s.component == component]
        static.append(np.mean([s.static_mse for s in vals]))
        adaptive.append(np.mean([s.per_step_mse for s in vals]))
    ax.bar(xloc - width / 2, static, width, label="static scale/codebook")
    ax.bar(xloc + width / 2, adaptive, width, label="per-step scale")
    ax.set_xticks(xloc)
    ax.set_xticklabels(components, rotation=15, ha="right")
    ax.set_ylabel("mean fake-quant MSE")
    ax.set_title("Static vs step-indexed quantization error")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    hist_components = [
        component
        for component in components
        if "action" in component or "cross" in component or component.endswith("_quant")
    ]
    if not hist_components:
        hist_components = components[: min(2, len(components))]
    for component in hist_components:
        steps = sorted(s.step for s in stats if s.component == component)
        first = data[f"{component}__step{steps[0]}"].reshape(-1)
        last = data[f"{component}__step{steps[-1]}"].reshape(-1)
        bins = np.linspace(
            np.percentile(np.concatenate([first, last]), 0.5),
            np.percentile(np.concatenate([first, last]), 99.5),
            80,
        )
        ax.hist(first, bins=bins, density=True, histtype="step", label=f"{component} first")
        ax.hist(last, bins=bins, density=True, histtype="step", linestyle="--", label=f"{component} last")
    ax.set_title("Early vs late drifting component histogram")
    ax.set_xlabel("activation / KV value")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)

    fig.suptitle("VLA quantization drift probe", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


def write_csv(stats: list[StepStats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("component,step,mean,std,rms,kurtosis,static_mse,per_step_mse\n")
        for s in stats:
            f.write(
                f"{s.component},{s.step},{s.mean:.8g},{s.std:.8g},{s.rms:.8g},"
                f"{s.kurtosis:.8g},{s.static_mse:.8g},{s.per_step_mse:.8g}\n"
            )
    print(f"Wrote {out_path}")


def write_comparison_csv(rows: list[CompareStats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("condition,component,step,rms,relative_drift,static_mse,per_step_mse,live_mse\n")
        for r in rows:
            live = "" if r.live_mse is None else f"{r.live_mse:.8g}"
            f.write(
                f"{r.condition},{r.component},{r.step},{r.rms:.8g},{r.relative_drift:.8g},"
                f"{r.static_mse:.8g},{r.per_step_mse:.8g},{live}\n"
            )
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, help="Optional real hooked tensor dump.")
    parser.add_argument("--baseline-npz", type=Path, help="Unquantized run tensor dump for comparison plot.")
    parser.add_argument("--quantized-npz", type=Path, help="WHT+Lloyd-Max run tensor dump for comparison plot.")
    parser.add_argument("--out", type=Path, default=Path("reports/fig_quant_drift_probe.png"))
    parser.add_argument("--csv", type=Path, default=Path("reports/quant_drift_probe.csv"))
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--no-rotate", action="store_true", help="Disable WHT rotation before stats/quant.")
    args = parser.parse_args()

    if args.baseline_npz and args.quantized_npz:
        baseline = load_npz(args.baseline_npz)
        quantized = load_npz(args.quantized_npz)
        rows = compute_comparison_stats(baseline, quantized, bits=args.bits, rotate=not args.no_rotate)
        write_comparison_csv(rows, args.csv)
        plot_comparison(baseline, quantized, rows, args.out)
        return

    data = load_npz(args.npz) if args.npz else demo_tensors()
    stats = compute_stats(data, bits=args.bits, rotate=not args.no_rotate)
    write_csv(stats, args.csv)
    plot_report(data, stats, args.out)


if __name__ == "__main__":
    main()
