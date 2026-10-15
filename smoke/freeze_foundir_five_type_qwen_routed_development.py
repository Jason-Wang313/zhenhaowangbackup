#!/usr/bin/env python3
"""Freeze the integrated development result before internal evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import resolve_workspace_path  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import write_once  # noqa: E402
from smoke.evaluate_foundir_five_type_qwen_routed_integrated_ddp import (  # noqa: E402
    validate_router_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-config", type=Path, required=True)
    args = parser.parse_args()
    router_path = resolve_workspace_path(ROOT, args.router_config)
    router, _, _, checkpoint = validate_router_config(router_path)
    development = resolve_workspace_path(ROOT, router["outputs"]["development"]) / "result.json"
    payload = json.loads(development.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "qwen_routed_integrated_development_complete"
        or payload.get("router_config_sha256") != file_sha256(router_path)
        or payload.get("blur_checkpoint_sha256") != file_sha256(checkpoint)
        or payload.get("split") != "validation"
        or int(payload.get("records", -1)) != int(router["evaluation"]["development_records"])
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("integrated development result cannot be frozen")
    freeze = {
        "schema_version": 1,
        "status": "frozen_qwen_routed_integrated_development_result",
        "router_config": router_path.relative_to(ROOT).as_posix(),
        "router_config_sha256": file_sha256(router_path),
        "development_result": development.relative_to(ROOT).as_posix(),
        "development_result_sha256": file_sha256(development),
        "development_gate_pass": payload["gate_pass"],
        "development_worst_normalized_margin": payload["worst_normalized_margin"],
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    output = resolve_workspace_path(ROOT, router["outputs"]["development_freeze"])
    write_once(output, (json.dumps(freeze, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_QWEN_ROUTED_DEVELOPMENT_FREEZE_PASS " + json.dumps(freeze, sort_keys=True))


if __name__ == "__main__":
    main()
