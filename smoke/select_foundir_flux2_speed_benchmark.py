#!/usr/bin/env python3
"""Select the fastest scientifically equivalent FLUX training schedule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


def relative_loss_delta(reference: Sequence[float], candidate: Sequence[float]) -> float:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("loss trajectories must have equal non-zero length")
    return sum(
        abs(float(left) - float(right)) / max(abs(float(left)), 1e-8)
        for left, right in zip(reference, candidate)
    ) / len(reference)


def select(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(results) < 2:
        raise RuntimeError("speed selection requires a baseline and at least one candidate")
    baseline = results[0]
    identity_keys = (
        "config_sha256",
        "git_revision",
        "gpu_family",
        "world_size",
        "global_batch_size",
        "optimizer_steps",
        "warmup_steps_excluded",
    )
    for result in results:
        if (
            result.get("status") != "speed_benchmark_complete"
            or result.get("protected_foundir_test_accessed") is not False
            or result.get("checkpoint_written") is not False
            or result.get("per_presentation_randomness_independent_of_microbatch_grouping") is not True
            or any(result.get(key) != baseline.get(key) for key in identity_keys)
        ):
            raise RuntimeError("benchmark identity or safety contract changed")
        values = [
            float(result["mean_seconds_per_step"]),
            float(result["global_samples_per_second"]),
            *[float(value) for value in result["mean_losses"]],
        ]
        if not all(math.isfinite(value) for value in values) or min(values[:2]) <= 0:
            raise RuntimeError("benchmark contains a non-finite or non-positive measurement")
        peaks = [int(value) for value in result["peak_memory_allocated_bytes_per_rank"]]
        totals = [int(value) for value in result["total_memory_bytes_per_rank"]]
        if (
            len(peaks) != int(result["world_size"])
            or len(totals) != int(result["world_size"])
            or any(peak <= 0 or total <= 0 or peak > total for peak, total in zip(peaks, totals))
        ):
            raise RuntimeError("benchmark memory measurements are invalid")

    baseline_losses = [float(value) for value in baseline["mean_losses"]]
    comparisons = []
    eligible = [baseline]
    for result in results:
        delta = relative_loss_delta(
            baseline_losses, [float(value) for value in result["mean_losses"]]
        )
        speedup = float(baseline["mean_seconds_per_step"]) / float(
            result["mean_seconds_per_step"]
        )
        equivalent = delta <= 0.05
        maximum_peak_memory_fraction = max(
            int(peak) / int(total)
            for peak, total in zip(
                result["peak_memory_allocated_bytes_per_rank"],
                result["total_memory_bytes_per_rank"],
            )
        )
        memory_safe = maximum_peak_memory_fraction <= 0.9
        comparisons.append(
            {
                "benchmark_name": result["benchmark_name"],
                "micro_batch_size_per_rank": result["micro_batch_size_per_rank"],
                "gradient_accumulation_steps": result["gradient_accumulation_steps"],
                "data_loader_workers_per_rank": result["data_loader_workers_per_rank"],
                "mean_seconds_per_step": result["mean_seconds_per_step"],
                "global_samples_per_second": result["global_samples_per_second"],
                "speedup_vs_baseline": speedup,
                "mean_relative_loss_delta_vs_baseline": delta,
                "short_run_loss_trajectory_equivalent": equivalent,
                "maximum_peak_memory_fraction": maximum_peak_memory_fraction,
                "memory_headroom_pass": memory_safe,
                "maximum_peak_memory_allocated_bytes": max(
                    int(value) for value in result["peak_memory_allocated_bytes_per_rank"]
                ),
            }
        )
        if result is not baseline and equivalent and memory_safe:
            eligible.append(result)

    fastest = min(eligible, key=lambda result: float(result["mean_seconds_per_step"]))
    fastest_speedup = float(baseline["mean_seconds_per_step"]) / float(
        fastest["mean_seconds_per_step"]
    )
    selected = fastest if fastest_speedup >= 1.05 else baseline
    return {
        "schema_version": 1,
        "status": "speed_benchmark_selection_complete",
        "config_sha256": baseline["config_sha256"],
        "git_revision": baseline["git_revision"],
        "gpu_family": baseline["gpu_family"],
        "world_size": baseline["world_size"],
        "global_batch_size": baseline["global_batch_size"],
        "baseline": baseline["benchmark_name"],
        "selected": selected["benchmark_name"],
        "selected_micro_batch_size_per_rank": selected["micro_batch_size_per_rank"],
        "selected_gradient_accumulation_steps": selected["gradient_accumulation_steps"],
        "selected_data_loader_workers_per_rank": selected["data_loader_workers_per_rank"],
        "selected_speedup_vs_baseline": (
            float(baseline["mean_seconds_per_step"])
            / float(selected["mean_seconds_per_step"])
        ),
        "minimum_required_speedup": 1.05,
        "loss_equivalence_maximum_mean_relative_delta": 0.05,
        "maximum_allowed_peak_memory_fraction": 0.9,
        "comparisons": comparisons,
        "quality_interpretation": (
            "All candidates preserve the frozen global batch, total presentation plan, "
            "optimizer-step/LR schedule, loss, data order, per-presentation randomness, "
            "checkpoint selection, and evaluation protocol. This short benchmark checks "
            "runtime stability and loss-trajectory agreement; it is not a final quality result."
        ),
        "protected_foundir_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [json.loads(path.read_text()) for path in args.result]
    payload = select(results)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(content)
    print("FOUNDIR_FLUX_SPEED_BENCHMARK_SELECTION_PASS " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
