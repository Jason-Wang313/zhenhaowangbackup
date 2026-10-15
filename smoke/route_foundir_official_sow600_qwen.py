#!/usr/bin/env python3
"""Apply the frozen final four-way router to authorized official Qwen features."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.final_eleven_runtime import FrozenFinalElevenRouter  # noqa: E402
from lowlight_restoration.qwen_eleven_class_gate import LABEL_BY_CATEGORY  # noqa: E402
from smoke.export_foundir_official_sow600_qwen_features_ddp import (  # noqa: E402
    load_contract,
    workspace,
)


def atomic_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to replace differing router artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(payload)
    os.link(temporary, path)
    temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows = load_contract(args.config.resolve())
    router = FrozenFinalElevenRouter()
    if args.check_only:
        print(
            f"FOUNDIR_OFFICIAL_SOW600_FINAL_ROUTER_PREFLIGHT_PASS "
            f"records={len(rows)} feature_dim={router.feature_dim} config_sha256={config_sha}"
        )
        return

    feature_dir = workspace(config["runtime"]["feature_dir"])
    features = []
    for row in rows:
        path = feature_dir / f"{row['record_id']}.safetensors"
        if not path.is_file():
            raise RuntimeError(f"missing official Qwen feature: {row['record_id']}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            required = {
                "config_sha256": config_sha,
                "manifest_sha256": config["data"]["manifest_sha256"],
                "adapter_sha256": config["qwen"]["adapter_sha256"],
                "record_id": row["record_id"],
                "lq_sha256": row["lq_sha256"],
                "pooling": "qwen_visual_token_mean_and_std",
                "feature_dim": str(config["qwen"]["feature_dim"]),
            }
            if any(metadata.get(key) != value for key, value in required.items()):
                raise RuntimeError(f"official Qwen feature identity changed: {row['record_id']}")
            features.append(handle.get_tensor("feature").numpy())
    decisions = router.route(
        np.stack(features), [workspace(row["lq_path"]) for row in rows]
    )

    fields = (
        "record_id", "category", "pair_id", "truth", "route", "expert_plan",
        "status", "compositional_route", "eleven_route", "dual_route",
        "physical_route", "five_proposal", "five_confidence", "eleven_proposal",
        "eleven_confidence", "exposure", "dark_noise", "fallback_applied",
    )
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    counts: Counter[str] = Counter()
    per_category: dict[str, Counter[str]] = {
        category: Counter() for category in config["data"]["records_by_category"]
    }
    for source, decision in zip(rows, decisions, strict=True):
        truth = LABEL_BY_CATEGORY[source["category"]]
        route = decision["route"]
        status = "fallback" if decision["fallback_applied"] else "abstain" if route is None else "correct" if route == truth else "wrong"
        counts[status] += 1
        per_category[source["category"]][status] += 1
        writer.writerow(
            {
                "record_id": source["record_id"],
                "category": source["category"],
                "pair_id": source["pair_id"],
                "truth": truth,
                "route": route or "abstain",
                "expert_plan": "+".join(decision["expert_plan"]) or "identity",
                "status": status,
                **{
                    key: decision[key] if decision[key] is not None else "abstain"
                    for key in (
                        "compositional_route", "eleven_route", "dual_route",
                        "physical_route", "five_proposal", "five_confidence",
                        "eleven_proposal", "eleven_confidence", "exposure",
                        "dark_noise", "fallback_applied",
                    )
                },
            }
        )
    output = workspace(config["runtime"]["router_decisions"])
    atomic_new(output, buffer.getvalue().encode())
    summary = {
        "schema_version": 1,
        "status": "official_sow600_qwen_routing_complete",
        "records": len(rows),
        "counts": dict(counts),
        "per_category": {key: dict(value) for key, value in per_category.items()},
        "truth_used_as_router_input": False,
        "category_path_used_as_router_input": False,
        "config_sha256": config_sha,
    }
    atomic_new(
        output.with_suffix(".json"),
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    print("FOUNDIR_OFFICIAL_SOW600_ROUTING_COMPLETE " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
