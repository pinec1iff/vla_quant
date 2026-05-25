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
        all_values = np.concatenate([by_step[s] for s in sorted(by_step)])
        _, static_scale = symmetric_quantize(all_values, bits)
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
    for component in components:
        steps = sorted(s.step for s in stats if s.component == component)
        first = data[f"{component}__step{steps[0]}"].reshape(-1)
        last = data[f"{component}__step{steps[-1]}"].reshape(-1)
        bins = np.linspace(
            np.percentile(np.concatenate([first, last]), 0.5),
            np.percentile(np.concatenate([first, last]), 99.5),
            80,
        )
        if "action" in component or "cross" in component:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, help="Optional real hooked tensor dump.")
    parser.add_argument("--out", type=Path, default=Path("reports/fig_quant_drift_probe.png"))
    parser.add_argument("--csv", type=Path, default=Path("reports/quant_drift_probe.csv"))
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--no-rotate", action="store_true", help="Disable WHT rotation before stats/quant.")
    args = parser.parse_args()

    data = load_npz(args.npz) if args.npz else demo_tensors()
    stats = compute_stats(data, bits=args.bits, rotate=not args.no_rotate)
    write_csv(stats, args.csv)
    plot_report(data, stats, args.out)


if __name__ == "__main__":
    main()
