#!/usr/bin/env python3
"""Free-generation diagnostics for existing real-adaptation checkpoints.

This is validation-only: it never loads the consumed real test for inference,
never selects a checkpoint, and never writes into the historical adaptation
output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_pilot import metric_gate_checks  # noqa: E402
from lowlight_restoration.qwen_real_adaptation import (  # noqa: E402
    load_adaptation_config,
    validate_adaptation_contract,
)
from lowlight_restoration.qwen_real_validation import (  # noqa: E402
    real_metric_gate_checks,
)
from lowlight_restoration.qwen_supervised import record_bundle_sha256  # noqa: E402
from smoke.qwen_lowlight_supervised_preflight import (  # noqa: E402
    load_inventory,
    sha256,
    workspace_path,
)
from smoke.train_qwen_lowlight_real_adaptation_ddp import (  # noqa: E402
    _load_data,
    distributed_context,
    evaluate_split,
    load_selected_visual,
    validate_runtime,
)
from smoke.train_qwen_lowlight_supervised import (  # noqa: E402
    atomic_write_new_json,
    load_model_and_processor,
    read_json,
    seed_everything,
)


def validated_output_dir(path: Path) -> Path:
    output_dir = path.resolve()
    try:
        output_dir.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("diagnostic output must stay inside the workspace") from exc
    historical = (ROOT / "checkpoints/qwen_lowlight_real_adaptation_step600").resolve()
    if output_dir == historical or historical in output_dir.parents:
        raise ValueError("diagnostics must not write into the historical result")
    if output_dir.exists():
        raise RuntimeError(f"immutable diagnostic output already exists: {output_dir}")
    return output_dir


def validated_steps(values: list[int], allowed: list[int]) -> list[int]:
    steps = list(dict.fromkeys(values))
    allowed_set = set(allowed)
    if not steps or any(step not in allowed_set for step in steps):
        raise ValueError(f"diagnostic steps must be a subset of {sorted(allowed_set)}")
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[25, 50, 100, 150, 200])
    args = parser.parse_args()

    config, config_sha, _ = load_adaptation_config(args.config)
    validate_adaptation_contract(config)
    data = config["data"]
    training = config["training"]
    generation = config["generation"]
    gate = config["gate"]
    runtime = config["runtime"]
    assert all(
        isinstance(section, dict)
        for section in (data, training, generation, gate, runtime)
    )
    steps = validated_steps(
        args.steps, [int(value) for value in training["checkpoint_steps"]]
    )
    output_dir = validated_output_dir(args.output_dir)

    rank, world_size, local_rank, device = distributed_context(config)
    try:
        validate_runtime(config, local_rank=local_rank, world_size=world_size)
        seed_everything(int(training["seed"]))
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        inventory, _ = load_inventory(config)
        real_splits, _, synthetic_validation = _load_data(config)
        if real_splits["test"]:
            # Loading the manifest is harmless, but this driver must never pass
            # the consumed test records to generation.
            test_ids = {record.pair_id for record in real_splits["test"]}
        else:
            test_ids = set()
        validation = list(real_splits["validation"])
        if test_ids & {record.pair_id for record in validation}:
            raise RuntimeError("real validation/test records overlap")
        model, processor, _ = load_model_and_processor(config, inventory, device)
        model.requires_grad_(False)
        model.eval()
        model.config.use_cache = True
        model_revision = str(inventory["model"]["revision"])
        real_config = data["real"]
        assert isinstance(real_config, dict)
        schedule_sha = str(training["schedule_sha256"])
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=False)
            atomic_write_new_json(
                output_dir / "diagnostic_contract.json",
                {
                    "schema_version": 1,
                    "scope": "validation_only_free_generation_existing_visual_checkpoints",
                    "config_sha256": config_sha,
                    "config": str(args.config.resolve().relative_to(ROOT)),
                    "steps": steps,
                    "real_validation_bundle_sha256": record_bundle_sha256(validation),
                    "synthetic_validation_bundle_sha256": record_bundle_sha256(
                        synthetic_validation
                    ),
                    "real_test_inference_forbidden": True,
                    "real_test_pair_count_loaded_but_not_inferred": len(test_ids),
                    "official_foundir_used": False,
                    "checkpoint_selection_performed": False,
                },
            )
        dist.barrier()

        for step in steps:
            checkpoint = (
                workspace_path(runtime["output_dir"])
                / "checkpoints"
                / f"step_{step:06d}"
            )
            metadata = read_json(checkpoint / "metadata.json")
            checkpoint_world_size = metadata.get("world_size")
            checkpoint_global_batch = metadata.get("global_batch_size")
            if (
                isinstance(checkpoint_world_size, bool)
                or not isinstance(checkpoint_world_size, int)
                or isinstance(checkpoint_global_batch, bool)
                or not isinstance(checkpoint_global_batch, int)
            ):
                raise RuntimeError(f"checkpoint runtime metadata invalid at step {step}")
            loaded_metadata = load_selected_visual(
                checkpoint=checkpoint,
                model=model,
                config_sha256=config_sha,
                manifest_sha256=str(real_config["manifest_sha256"]),
                schedule_sha256=schedule_sha,
                model_revision=model_revision,
                world_size=checkpoint_world_size,
                global_batch_size=checkpoint_global_batch,
                selected_step=step,
            )
            if loaded_metadata != metadata:
                raise RuntimeError(f"checkpoint metadata changed at step {step}")
            step_output = output_dir / f"step_{step:06d}"
            if rank == 0:
                step_output.mkdir(parents=True, exist_ok=False)
                (step_output / "evaluation").mkdir()
            dist.barrier()
            real_metrics, real_hashes = evaluate_split(
                model=model,
                processor=processor,
                records=validation,
                output_dir=step_output,
                split="real_validation",
                metric_kind="real",
                config_sha256=config_sha,
                selected_step=step,
                rank=rank,
                world_size=world_size,
                device=device,
                batch_size=int(generation["batch_size"]),
                max_new_tokens=int(generation["max_new_tokens"]),
                maximum_failures=int(config["gate"]["maximum_failure_examples"]),
            )
            synthetic_metrics, synthetic_hashes = evaluate_split(
                model=model,
                processor=processor,
                records=synthetic_validation,
                output_dir=step_output,
                split="synthetic_validation_replay",
                metric_kind="synthetic",
                config_sha256=config_sha,
                selected_step=step,
                rank=rank,
                world_size=world_size,
                device=device,
                batch_size=int(generation["batch_size"]),
                max_new_tokens=int(generation["max_new_tokens"]),
                maximum_failures=int(config["gate"]["maximum_failure_examples"]),
            )
            real_checks = real_metric_gate_checks(
                real_metrics,
                gate["real_validation"],
                prefix="real_validation",
            )
            synthetic_checks = metric_gate_checks(
                synthetic_metrics,
                gate["synthetic_replay"],
                prefix="synthetic_replay",
            )
            if rank == 0:
                atomic_write_new_json(
                    step_output / "result_summary.json",
                    {
                        "schema_version": 1,
                        "status": "diagnostic_complete",
                        "config_sha256": config_sha,
                        "selected_checkpoint_step": step,
                        "selected_visual_sha256": sha256(
                            checkpoint / "visual.safetensors"
                        ),
                        "checkpoint_metadata": loaded_metadata,
                        "real_validation_bundle_sha256": record_bundle_sha256(
                            validation
                        ),
                        "synthetic_validation_bundle_sha256": record_bundle_sha256(
                            synthetic_validation
                        ),
                        "real_validation_metrics": real_metrics,
                        "synthetic_validation_replay_metrics": synthetic_metrics,
                        "real_validation_gate_checks": real_checks,
                        "synthetic_validation_replay_gate_checks": synthetic_checks,
                        "all_validation_safeguards_pass": all(
                            [*real_checks.values(), *synthetic_checks.values()]
                        ),
                        "artifact_sha256": {
                            "real_validation": real_hashes,
                            "synthetic_validation_replay": synthetic_hashes,
                        },
                        "checkpoint_selection_performed": False,
                        "real_test_inferred": False,
                        "official_foundir_used": False,
                    },
                )
            dist.barrier()
        if rank == 0:
            step_summaries = [
                read_json(output_dir / f"step_{step:06d}" / "result_summary.json")
                for step in steps
            ]
            passing_steps = [
                int(summary["selected_checkpoint_step"])
                for summary in step_summaries
                if summary["all_validation_safeguards_pass"] is True
            ]
            atomic_write_new_json(
                output_dir / "result_summary.json",
                {
                    "schema_version": 1,
                    "status": "diagnostic_complete",
                    "config_sha256": config_sha,
                    "evaluated_checkpoint_steps": steps,
                    "validation_safeguard_passing_steps": passing_steps,
                    "earliest_validation_safeguard_passing_step": (
                        min(passing_steps) if passing_steps else None
                    ),
                    "checkpoint_results": step_summaries,
                    "checkpoint_selection_performed": False,
                    "cannot_retroactively_change_consumed_test_failure": True,
                    "real_test_inferred": False,
                    "official_foundir_used": False,
                },
            )
            print(
                "QWEN_REAL_ADAPTATION_DIAGNOSTICS_COMPLETE "
                f"steps={steps} validation_passing_steps={passing_steps}",
                flush=True,
            )
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
