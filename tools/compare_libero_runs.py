from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _success_rate(payload: dict) -> float | None:
    metrics = payload.get("metrics", {})
    if "success_rate" in metrics:
        return metrics["success_rate"]
    tasks = metrics.get("tasks", {})
    rates = [task.get("success_rate") for task in tasks.values() if task.get("success_rate") is not None]
    return sum(rates) / len(rates) if rates else None


def _episodes(payload: dict) -> int | None:
    metrics = payload.get("metrics", {})
    if "episodes" in metrics:
        return metrics["episodes"]
    return None


def _timing(payload: dict) -> dict:
    return payload.get("timing", {})


def _relative_change_percent(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return 100.0 * (value - baseline) / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--quantized-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_json.read_text())
    quantized = json.loads(args.quantized_json.read_text())
    base_mem = baseline.get("max_gpu_memory_gb")
    quant_mem = quantized.get("max_gpu_memory_gb")
    measured_savings = None
    if base_mem and quant_mem:
        measured_savings = 100.0 * (base_mem - quant_mem) / base_mem
    base_timing = _timing(baseline)
    quant_timing = _timing(quantized)
    base_seconds_per_episode = base_timing.get("seconds_per_episode")
    quant_seconds_per_episode = quant_timing.get("seconds_per_episode")
    inference_time_change = _relative_change_percent(quant_seconds_per_episode, base_seconds_per_episode)
    throughput_change = _relative_change_percent(
        quant_timing.get("episodes_per_second"), base_timing.get("episodes_per_second")
    )

    rows = [
        {
            "condition": "baseline",
            "run_name": baseline.get("run_name"),
            "model": baseline.get("model"),
            "suite": baseline.get("suite"),
            "success_rate_percent": None if _success_rate(baseline) is None else 100.0 * _success_rate(baseline),
            "episodes": _episodes(baseline),
            "max_gpu_memory_gb": base_mem,
            "server_startup_s": base_timing.get("server_startup_s"),
            "eval_wall_s": base_timing.get("eval_wall_s"),
            "total_remote_wall_s": base_timing.get("total_remote_wall_s"),
            "seconds_per_episode": base_seconds_per_episode,
            "episodes_per_second": base_timing.get("episodes_per_second"),
            "live_quant_mode": baseline.get("live_quant_mode"),
            "live_quant_bits": baseline.get("live_quant_bits"),
            "estimated_quantized_component_relative_savings_percent": baseline.get(
                "estimated_quantized_component_relative_savings_percent"
            ),
            "measured_peak_memory_relative_savings_percent_vs_baseline": 0.0,
            "seconds_per_episode_relative_change_percent_vs_baseline": 0.0,
            "episodes_per_second_relative_change_percent_vs_baseline": 0.0,
        },
        {
            "condition": "quantized",
            "run_name": quantized.get("run_name"),
            "model": quantized.get("model"),
            "suite": quantized.get("suite"),
            "success_rate_percent": None if _success_rate(quantized) is None else 100.0 * _success_rate(quantized),
            "episodes": _episodes(quantized),
            "max_gpu_memory_gb": quant_mem,
            "server_startup_s": quant_timing.get("server_startup_s"),
            "eval_wall_s": quant_timing.get("eval_wall_s"),
            "total_remote_wall_s": quant_timing.get("total_remote_wall_s"),
            "seconds_per_episode": quant_seconds_per_episode,
            "episodes_per_second": quant_timing.get("episodes_per_second"),
            "live_quant_mode": quantized.get("live_quant_mode"),
            "live_quant_bits": quantized.get("live_quant_bits"),
            "estimated_quantized_component_relative_savings_percent": quantized.get(
                "estimated_quantized_component_relative_savings_percent"
            ),
            "measured_peak_memory_relative_savings_percent_vs_baseline": measured_savings,
            "seconds_per_episode_relative_change_percent_vs_baseline": inference_time_change,
            "episodes_per_second_relative_change_percent_vs_baseline": throughput_change,
        },
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
