#!/usr/bin/env python3
"""Freeze or verify the complete FoundIR five-type prelaunch decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.export_foundir_five_type_qwen_conditions import sha256, write_once  # noqa: E402


def workspace_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def build_payload(config_path: Path) -> tuple[dict[str, object], Path]:
    config = json.loads(config_path.read_text())
    if (
        config.get("status") != "frozen_training_only_prelaunch_gate"
        or config.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("prelaunch config is not frozen or permits protected data")
    qwen_path = workspace_path(config["output"])
    qwen = json.loads(qwen_path.read_text())
    config_sha = sha256(config_path)
    restoration_config = workspace_path(config["restoration_config"])
    if sha256(restoration_config) != config["restoration_config_sha256"]:
        raise RuntimeError("frozen restoration config changed")
    selected = qwen.get("selected_candidate")
    if (
        qwen.get("status") != "pass"
        or qwen.get("prelaunch_config_sha256") != config_sha
        or qwen.get("manifest_sha256") != config["manifest_sha256"]
        or selected not in {"joint", "per_type"}
        or qwen.get("protected_foundir_test_accessed") is not False
        or qwen.get("severity_evaluated") is not False
        or qwen.get("candidates", {}).get(selected, {}).get("automatic_gate_pass")
        is not True
    ):
        raise RuntimeError("Qwen training-domain gate did not pass its frozen contract")
    payload = {
        "schema_version": 1,
        "status": "pass",
        "prelaunch_config": config_path.relative_to(ROOT).as_posix(),
        "prelaunch_config_sha256": config_sha,
        "restoration_config_sha256": config["restoration_config_sha256"],
        "manifest_sha256": config["manifest_sha256"],
        "qwen_training_domain_gate": qwen_path.relative_to(ROOT).as_posix(),
        "qwen_training_domain_gate_sha256": sha256(qwen_path),
        "selected_candidate": selected,
        "runtime_preflight": "passed_in_same_fail_closed_runner",
        "distributed_canary": "passed_in_same_fail_closed_runner",
        "implementation_sha256": {
            relative: sha256(workspace_path(relative))
            for relative in (
                "smoke/run_foundir_five_type_prelaunch_8gpu.sh",
                "smoke/foundir_five_type_runtime_preflight.py",
                "smoke/foundir_five_type_distributed_canary.py",
                "smoke/export_foundir_five_type_qwen_per_type_pilot.py",
                "smoke/evaluate_foundir_five_type_qwen_training_gate.py",
            )
        },
        "restoration_optimizer_steps_started": 0,
        "protected_foundir_test_accessed": False,
    }
    return payload, workspace_path(config["completion_output"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prelaunch-config",
        type=Path,
        default=ROOT / "configs/foundir_five_type_prelaunch_gate_v1.json",
    )
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--require-selected", choices=("joint", "per_type"))
    args = parser.parse_args()
    config_path = workspace_path(args.prelaunch_config)
    payload, output = build_payload(config_path)
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if args.verify_existing:
        if not output.is_file() or output.read_bytes() != content:
            raise RuntimeError("complete prelaunch artifact is missing or changed")
    else:
        write_once(output, content)
    if (
        args.require_selected is not None
        and payload["selected_candidate"] != args.require_selected
    ):
        print(
            "FOUNDIR_FIVE_TYPE_PRELAUNCH_SELECTION_MISMATCH "
            f"required={args.require_selected} selected={payload['selected_candidate']}",
            flush=True,
        )
        raise SystemExit(2)
    print("FOUNDIR_FIVE_TYPE_PRELAUNCH_FREEZE_PASS " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
