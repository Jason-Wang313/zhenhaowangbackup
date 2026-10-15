#!/usr/bin/env python3
"""Build the route-aware 660-record table with the direct Restormer expert."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    result = (path if path.is_absolute() else ROOT / path).resolve()
    result.relative_to(ROOT.resolve())
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, payload: Mapping[str, object]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable replay artifact differs: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def checked_json(path: Path, expected_sha256: str) -> dict:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise RuntimeError(f"hash-bound artifact changed: {path}")
    return json.loads(path.read_text())


def same_metrics(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return all(abs(float(left[key]) - float(right[key])) <= 1e-12 for key in ("psnr", "lpips"))


def summarize(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    roles = (
        "project_route_aware",
        "released_foundir_generalist",
        "full_foundir_generalist_specialist",
    )
    result: dict[str, object] = {"records": len(values)}
    for role in roles:
        result[role] = {
            metric: float(np.mean([float(item[role][metric]) for item in values]))
            for metric in ("psnr", "lpips")
        }
    project = result["project_route_aware"]
    generalist = result["released_foundir_generalist"]
    full = result["full_foundir_generalist_specialist"]
    result["project_minus_released_generalist"] = {
        "psnr_db": project["psnr"] - generalist["psnr"],
        "lpips": project["lpips"] - generalist["lpips"],
    }
    result["project_minus_full_foundir"] = {
        "psnr_db": project["psnr"] - full["psnr"],
        "lpips": project["lpips"] - full["lpips"],
    }
    result["project_win_fraction_vs_full_foundir"] = {
        "psnr": float(np.mean([
            float(item["project_route_aware"]["psnr"])
            > float(item["full_foundir_generalist_specialist"]["psnr"])
            for item in values
        ])),
        "lpips": float(np.mean([
            float(item["project_route_aware"]["lpips"])
            < float(item["full_foundir_generalist_specialist"]["lpips"])
            for item in values
        ])),
    }
    return result


def read_contract(config_path: Path) -> tuple[dict, list[dict[str, str]], str]:
    config = json.loads(config_path.read_text())
    config_sha = sha256(config_path)
    scope = config.get("scope", {})
    if (
        config.get("schema_version") != 1
        or config.get("status") != "user_authorized_route_aware_integrated_composition_replay_v2"
        or scope.get("records") != 660
        or scope.get("fresh_inference") is not False
        or scope.get("artifact_replay") is not True
        or scope.get("model_weight_updates") is not False
        or scope.get("official_foundir_test_accessed") is not False
    ):
        raise RuntimeError("integrated replay scope changed")
    source = config["source"]
    for prefix in ("project_config", "project_result", "full_foundir_config", "full_foundir_result"):
        checked_json(resolve(source[prefix]), source[prefix + "_sha256"])
    expert = config["replacement_expert"]
    for prefix in ("config", "development_selection", "capacity_result"):
        checked_json(resolve(expert[prefix]), expert[prefix + "_sha256"])
    checkpoint = resolve(expert["checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != expert["checkpoint_sha256"]:
        raise RuntimeError("replacement checkpoint changed")
    manifest = resolve(source["manifest"])
    if sha256(manifest) != source["manifest_sha256"]:
        raise RuntimeError("composition manifest changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if (
        len(rows) != 660
        or len({row["record_id"] for row in rows}) != 660
        or Counter(row["category"] for row in rows) != Counter({key: 60 for key in {
            row["category"] for row in rows
        }})
    ):
        raise RuntimeError("composition manifest cardinality changed")
    target = [row for row in rows if row["category"] == expert["category"]]
    replaced = [
        row for row in target
        if row["final_route"] == expert["accepted_final_route"]
        and row["routing_status"] == expert["accepted_routing_status"]
    ]
    statuses = Counter(row["routing_status"] for row in target)
    if (
        len(replaced) != expert["expected_replaced_records"]
        or statuses != Counter({"correct": 53, "fallback": 5, "abstain": 2})
    ):
        raise RuntimeError(f"replacement routing cardinality changed: {len(replaced)} {statuses}")
    return config, rows, config_sha


def old_project_output(config: Mapping[str, object], row: Mapping[str, str]) -> Path:
    order = json.loads(row["expert_order_json"])
    if not order:
        return resolve(row["lq_path"])
    return resolve(Path(config["source"]["project_output_dir"]) / "ours" / "stages" / order[-1] / f"{row['record_id']}.png")


def run(config_path: Path, *, preflight: bool) -> None:
    config, rows, config_sha = read_contract(config_path)
    if preflight:
        print(json.dumps({
            "status": "FOUNDIR_COMPOSITION_INTEGRATED_RESTORMER_V2_PREFLIGHT_PASS",
            "records": 660,
            "route_accepted_replacement_records": 53,
            "fallback_records": 5,
            "abstain_records": 2,
            "fresh_inference": False,
            "official_foundir_test_accessed": False,
        }, sort_keys=True))
        return

    source = config["source"]
    expert = config["replacement_expert"]
    old_sidecars = resolve(source["project_sidecar_dir"])
    full_sidecars = resolve(source["full_foundir_sidecar_dir"])
    replacement_sidecars = resolve(expert["sidecar_dir"])
    replacement_images = resolve(expert["image_dir"])
    output = resolve(config["runtime"]["output_dir"])
    payloads: list[dict[str, object]] = []
    route_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        record_id = row["record_id"]
        old_path = old_sidecars / f"{record_id}.json"
        full_path = full_sidecars / f"{record_id}.json"
        if not old_path.is_file() or not full_path.is_file():
            raise RuntimeError(f"missing parent metric sidecar: {record_id}")
        old = json.loads(old_path.read_text())
        foundir = json.loads(full_path.read_text())
        if (
            old.get("status") != "complete"
            or foundir.get("status") != "complete"
            or old.get("record_id") != record_id
            or foundir.get("record_id") != record_id
            or old.get("category") != row["category"]
            or foundir.get("category") != row["category"]
            or not same_metrics(old["ours"], foundir["project"])
            or not same_metrics(old["foundir"], foundir["released_foundir_generalist"])
            or old.get("official_foundir_test_accessed") is not False
            or foundir.get("official_foundir_test_accessed") is not False
        ):
            raise RuntimeError(f"parent metric identity changed: {record_id}")

        replaced = (
            row["category"] == expert["category"]
            and row["final_route"] == expert["accepted_final_route"]
            and row["routing_status"] == expert["accepted_routing_status"]
        )
        if replaced:
            replacement_path = replacement_sidecars / f"{record_id}.json"
            replacement = json.loads(replacement_path.read_text())
            image_path = replacement_images / f"{record_id}.png"
            if (
                replacement.get("status") != "complete"
                or replacement.get("record_id") != record_id
                or replacement.get("category") != expert["category"]
                or replacement.get("output_sha256") != sha256(image_path)
                or not same_metrics(replacement["full_foundir"], foundir["full_foundir_generalist_specialist"])
                or replacement.get("official_foundir_test_accessed") is not False
            ):
                raise RuntimeError(f"replacement artifact identity changed: {record_id}")
            project_metrics = replacement["project_restormer"]
            project_source = "route_accepted_lowlight_haze_direct_restormer_v1"
            project_source_sidecar = replacement_path
            deployed_plan = [expert["name"]]
            route_counts["replacement"] += 1
        else:
            image_path = old_project_output(config, row)
            project_metrics = old["ours"]
            project_source = "frozen_parent_project_output"
            project_source_sidecar = old_path
            deployed_plan = json.loads(row["expert_order_json"])
            route_counts[row["routing_status"] if row["category"] == expert["category"] else "unchanged_other"] += 1
        if not image_path.is_file():
            raise RuntimeError(f"missing project output: {image_path}")
        with Image.open(image_path) as opened:
            if opened.size != (int(row["width"]), int(row["height"])):
                raise RuntimeError(f"project output dimensions changed: {record_id}")
        payload = {
            "schema_version": 1,
            "status": "complete",
            "config_sha256": config_sha,
            "record_id": record_id,
            "source_group": row["source_group"],
            "category": row["category"],
            "composition_size": int(row["composition_size"]),
            "router_truth": row["router_truth"],
            "final_route": row["final_route"],
            "routing_status": row["routing_status"],
            "deployed_expert_plan": deployed_plan,
            "replacement_applied": replaced,
            "project_source": project_source,
            "project_source_sidecar": project_source_sidecar.relative_to(ROOT).as_posix(),
            "project_source_sidecar_sha256": sha256(project_source_sidecar),
            "project_output": image_path.relative_to(ROOT).as_posix(),
            "project_output_sha256": sha256(image_path),
            "project_route_aware": {key: float(project_metrics[key]) for key in ("psnr", "lpips")},
            "released_foundir_generalist": {
                key: float(foundir["released_foundir_generalist"][key]) for key in ("psnr", "lpips")
            },
            "full_foundir_generalist_specialist": {
                key: float(foundir["full_foundir_generalist_specialist"][key]) for key in ("psnr", "lpips")
            },
            "fresh_inference": False,
            "model_weight_updates": False,
            "official_foundir_test_accessed": False,
        }
        write_once(output / "per_record" / f"{record_id}.json", payload)
        payloads.append(payload)
        if index % 50 == 0 or index == len(rows):
            print(f"FOUNDIR_INTEGRATED_REPLAY record={index}/{len(rows)}", flush=True)

    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_composition: dict[int, list[dict[str, object]]] = defaultdict(list)
    for payload in payloads:
        by_category[str(payload["category"])].append(payload)
        by_composition[int(payload["composition_size"])].append(payload)
    expected_route_counts = Counter({"replacement": 53, "fallback": 5, "abstain": 2, "unchanged_other": 600})
    if route_counts != expected_route_counts:
        raise RuntimeError(f"terminal route counts changed: {route_counts}")
    result = {
        "schema_version": 1,
        "status": "foundir_composition_integrated_restormer_v2_complete",
        "config_sha256": config_sha,
        "records": len(payloads),
        "route_counts": dict(sorted(route_counts.items())),
        "overall_metrics": summarize(payloads),
        "composition_metrics": {
            str(key): summarize(value) for key, value in sorted(by_composition.items())
        },
        "category_metrics": {
            key: summarize(value) for key, value in sorted(by_category.items())
        },
        "metric_protocol": config["metric_protocol"],
        "fresh_inference": False,
        "artifact_replay": True,
        "model_weight_updates": False,
        "development_pixels_accessed": True,
        "official_foundir_test_accessed": False,
        "limitation": config["scope"]["limitation"],
    }
    write_once(output / "result.json", result)
    print("FOUNDIR_COMPOSITION_INTEGRATED_RESTORMER_V2_COMPLETE " + json.dumps(result["overall_metrics"], sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    run(config_path, preflight=args.preflight)


if __name__ == "__main__":
    main()
