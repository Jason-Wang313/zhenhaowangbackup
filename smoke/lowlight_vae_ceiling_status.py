#!/usr/bin/env python3
"""Fail-closed verification of a completed VAE-ceiling diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STATUSES = {
    "completed_pixel_frequency_bypass_required",
    "completed_latent_only_ceiling_clear",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_core(
    summary: Mapping[str, object],
    *,
    config_sha256: str,
    expected_records: int,
    expected_scenes: int,
    expected_unique_targets: int,
) -> None:
    status = summary.get("status")
    if status not in ALLOWED_STATUSES:
        raise RuntimeError(f"invalid VAE-ceiling status: {status}")
    if summary.get("config_sha256") != config_sha256:
        raise RuntimeError("VAE-ceiling config identity differs")
    if summary.get("official_foundir_inference_performed") is not False:
        raise RuntimeError("VAE ceiling improperly reports FoundIR inference")
    if summary.get("official_foundir_images_opened") is not False:
        raise RuntimeError("VAE ceiling improperly reports FoundIR image access")
    if int(summary.get("records", -1)) != expected_records:
        raise RuntimeError("VAE-ceiling record count differs")
    if int(summary.get("unique_targets", -1)) != expected_unique_targets:
        raise RuntimeError("VAE-ceiling unique-target count differs")
    metric_summary = summary.get("summary")
    if not isinstance(metric_summary, dict):
        raise RuntimeError("VAE-ceiling metric summary is missing")
    if int(metric_summary.get("records", -1)) != expected_records:
        raise RuntimeError("VAE-ceiling metric rows are incomplete")
    if int(metric_summary.get("scenes", -1)) != expected_scenes:
        raise RuntimeError("VAE-ceiling scene count differs")
    automatic = summary.get("automatic_decision")
    if not isinstance(automatic, dict):
        raise RuntimeError("VAE-ceiling decision is missing")
    bypass = automatic.get("pixel_frequency_bypass_required")
    if not isinstance(bypass, bool):
        raise RuntimeError("VAE-ceiling bypass decision is invalid")
    expected_status = (
        "completed_pixel_frequency_bypass_required"
        if bypass
        else "completed_latent_only_ceiling_clear"
    )
    if status != expected_status:
        raise RuntimeError("VAE-ceiling status contradicts its decision")
    inventory = summary.get("device_inventory")
    if not isinstance(inventory, list) or len(inventory) != 2:
        raise RuntimeError("VAE-ceiling two-GPU inventory is incomplete")
    if {int(item["rank"]) for item in inventory} != {0, 1}:
        raise RuntimeError("VAE-ceiling GPU ranks differ")


def validate_artifacts(config_path: Path, summary_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    input_config = config["input"]
    output = config["output"]
    model = config["model"]
    validate_core(
        summary,
        config_sha256=sha256(config_path),
        expected_records=int(input_config["expected_records"]),
        expected_scenes=int(input_config["expected_scenes"]),
        expected_unique_targets=int(input_config["expected_unique_targets"]),
    )
    expected_hashes = {
        "evaluator_sha256": sha256(
            ROOT / "smoke/evaluate_lowlight_vae_ceiling_ddp.py"
        ),
        "representation_module_sha256": sha256(
            ROOT / "lowlight_restoration/representation_ceiling.py"
        ),
        "paired_development_module_sha256": sha256(
            ROOT / "lowlight_restoration/paired_development.py"
        ),
        "autoencoder_sha256": str(model["autoencoder_sha256"]),
    }
    for key, expected in expected_hashes.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"VAE-ceiling code/model identity differs: {key}")
    expected_git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if summary.get("git_head") != expected_git:
        raise RuntimeError("VAE-ceiling Git identity differs")
    rows_path = (ROOT / str(output["rows_csv"])).resolve()
    if not rows_path.is_file() or summary.get("result_rows_sha256") != sha256(
        rows_path
    ):
        raise RuntimeError("VAE-ceiling result-row artifact differs")
    manifest_path = (ROOT / str(input_config["manifest"])).resolve()
    with manifest_path.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    target_hashes = {row["target_sha256"] for row in manifest_rows}
    sidecar_dir = (ROOT / str(output["directory"]) / "per_target").resolve()
    sidecars = sorted(sidecar_dir.glob("*.json"))
    if {path.stem for path in sidecars} != target_hashes:
        raise RuntimeError("VAE-ceiling sidecar target set differs")
    for path in sidecars:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("target_sha256") != path.stem:
            raise RuntimeError(f"invalid VAE-ceiling sidecar: {path}")
        for key, expected in {
            "config_sha256": sha256(config_path),
            **expected_hashes,
        }.items():
            if payload.get(key) != expected:
                raise RuntimeError(f"VAE-ceiling sidecar identity differs: {path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_artifacts(args.config.resolve(), args.summary.resolve())
    print(
        "LOWLIGHT_VAE_CEILING_STATUS_PASS "
        f"status={summary['status']} records={summary['records']} "
        f"unique_targets={summary['unique_targets']}"
    )


if __name__ == "__main__":
    main()
