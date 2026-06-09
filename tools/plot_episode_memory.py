from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PRESET_ORDER = ("fp16", "a8", "w4a8", "w4a4")


def _task_label(task: str) -> str:
    return task.replace("_", " ")


def _ordered_presets(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    ordered = [preset for preset in DEFAULT_PRESET_ORDER if preset in present]
    return ordered + sorted(present.difference(ordered))


def _write_heatmap(table: pd.DataFrame, model: str, output_path: Path) -> None:
    row_count = len(table)
    figure_height = max(8.0, 0.34 * row_count + 2.5)
    figure_width = max(8.0, 1.7 * len(table.columns) + 4.5)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))

    values = table.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)
    image = axis.imshow(masked_values, aspect="auto", cmap="viridis")

    axis.set_xticks(range(len(table.columns)), table.columns)
    axis.set_yticks(range(len(table.index)), table.index)
    axis.set_xlabel("Preset")
    axis.set_ylabel("Task")
    axis.set_title(f"{model} mean peak GPU memory by task and preset")

    finite_values = values[np.isfinite(values)]
    midpoint = (
        float(finite_values.min() + finite_values.max()) / 2.0
        if finite_values.size
        else 0.0
    )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = "N/A" if np.isnan(value) else f"{value:.2f}"
            color = "white" if np.isnan(value) or value < midpoint else "black"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )

    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Mean episode peak memory (GB)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a task-by-preset memory table and heatmap."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=("peak_memory_gb_during_episode", "memory_used_gb_at_completion"),
        default="peak_memory_gb_during_episode",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)
    required_columns = {"model", "preset", "suite", "task", args.metric}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        parser.error(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    model_values = data["model"].dropna().unique()
    model = str(model_values[0]) if len(model_values) == 1 else args.input_csv.parent.name
    data["task"] = data["task"].astype(str).map(_task_label)
    data[args.metric] = pd.to_numeric(data[args.metric], errors="coerce")
    presets = _ordered_presets(data["preset"])

    grouped = (
        data.groupby(["suite", "task", "preset"], as_index=False, dropna=False)
        .agg(
            mean_peak_memory_gb=(args.metric, "mean"),
            measured_episodes=(args.metric, "count"),
            total_episodes=(args.metric, "size"),
        )
    )
    grouped["task_label"] = grouped["suite"] + " | " + grouped["task"]

    memory_table = grouped.pivot(
        index="task_label", columns="preset", values="mean_peak_memory_gb"
    ).reindex(columns=presets)
    coverage_table = grouped.pivot(
        index="task_label", columns="preset", values="measured_episodes"
    ).reindex(index=memory_table.index, columns=presets)
    total_table = grouped.pivot(
        index="task_label", columns="preset", values="total_episodes"
    ).reindex(index=memory_table.index, columns=presets)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / f"{model}_memory_by_task.csv"
    coverage_path = args.out_dir / f"{model}_memory_sample_coverage.csv"
    heatmap_path = args.out_dir / f"{model}_memory_by_task.png"

    memory_table.round(3).to_csv(table_path, index_label="task")
    coverage = coverage_table.fillna(0).astype(int).astype(str)
    totals = total_table.fillna(0).astype(int).astype(str)
    (coverage + "/" + totals).to_csv(coverage_path, index_label="task")
    _write_heatmap(memory_table, model, heatmap_path)

    print(f"Wrote {table_path}")
    print(f"Wrote {coverage_path}")
    print(f"Wrote {heatmap_path}")


if __name__ == "__main__":
    main()
