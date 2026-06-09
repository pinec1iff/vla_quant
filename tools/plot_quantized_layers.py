from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.layers_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No layer coverage rows found.")

    presets = list(dict.fromkeys(row["precision"] for row in rows))
    models = list(dict.fromkeys(row["model_label"] for row in rows))
    x = np.arange(len(presets))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, model in enumerate(models):
        by_preset = {
            row["precision"]: float(row["quantized_transformer_layers"])
            for row in rows
            if row["model_label"] == model
        }
        values = [by_preset.get(preset, 0.0) for preset in presets]
        offset = (index - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model)
        ax.bar_label(bars, fmt="%.0f", padding=2)

    ax.set_xticks(x, presets)
    ax.set_ylabel("Quantized transformer layers")
    ax.set_title("Quantized Layer Coverage by Precision Preset")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
