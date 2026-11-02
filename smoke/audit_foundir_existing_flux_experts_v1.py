#!/usr/bin/env python3
"""Read-only CPU audit; no CUDA, image decode, training, or automatic launch.

Code/configs come from this checkout. --asset-root may be the frozen migration
source for READ ONLY inspection; such a report is not destination qualification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def bound_asset(root, relative, expected):
    path = inside(root, relative)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"asset identity mismatch: {relative}: {actual} != {expected}")
    return path, {"path": relative, "sha256": actual, "bytes": path.stat().st_size}


def expected_modules():
    return ([f"double_blocks.{i}.img_attn.{part}" for i in range(5) for part in ("qkv", "proj")]
            + [f"single_blocks.{i}.{part}" for i in range(20) for part in ("linear1", "linear2")])


def inspect_lora(path, base_shapes, expected_rank=32):
    import torch
    from safetensors import safe_open
    modules = expected_modules()
    expected = {f"{m}.lora_{p}.weight" for m in modules for p in ("A", "B")}
    summary = {"modules": len(modules), "tensors": len(expected), "parameters": 0,
               "dtypes": [], "all_finite": True, "shapes": {}}
    dtypes = set()
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if set(f.keys()) != expected:
            raise ValueError(f"unexpected LoRA keys in {path.name}")
        for m in modules:
            a, b = (f.get_tensor(f"{m}.lora_{p}.weight") for p in ("A", "B"))
            out_features, in_features = base_shapes[f"{m}.weight"]
            if tuple(a.shape) != (expected_rank, in_features) or tuple(b.shape) != (out_features, expected_rank):
                raise ValueError(f"LoRA/base shape mismatch: {m}")
            if a.dtype != b.dtype or a.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(f"unexpected LoRA dtype: {m}")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f"nonfinite LoRA: {m}")
            dtypes.add(str(a.dtype))
            summary["parameters"] += a.numel() + b.numel()
            summary["shapes"][m] = [list(a.shape), list(b.shape)]
    summary["dtypes"] = sorted(dtypes)
    return summary


def rain_source_audit(root, config, bank_config):
    """Manifest-only identity audit. Never opens image paths, including dev."""
    data = config["data"]
    path, binding = bound_asset(root, data["manifest"], data["manifest_sha256"])
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    train = [r for r in rows if r["role"] == "direct_endpoint_train" and r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "validation"]
    if len(train) != 38334 or len(dev) != 100 or len(train) + len(dev) != len(rows):
        raise ValueError("Rain manifest count or role changed")
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate Rain record")
    if any(r["category"] != "10Rain" or r["composition_size"] != "1" for r in rows):
        raise ValueError("unexpected Rain category")
    for field in ("source_group", "gt_sha256"):
        if {r[field] for r in train} & {r[field] for r in dev}:
            raise ValueError(f"Rain training/development overlap: {field}")
    counts = {"train_records": len(train), "train_sources": len({r["source_group"] for r in train}),
              "development_records": len(dev), "development_sources": len({r["source_group"] for r in dev})}
    if counts["train_sources"] != 5780 or counts["development_sources"] != 100:
        raise ValueError("Rain source counts changed")
    # Cross-task validation identities are forbidden to the future Rain training.
    main_path, main_binding = bound_asset(root, "manifests/foundir_five_type_pairs_v1.csv",
        "e8c018879029c0bcc2da8d074cc4999541158ca97d07f48aadbcdddcdccc3ca7")
    blocked_groups, blocked_gt = set(), set()
    pools = {"blur": [], "noise": [], "haze": [], "lowlight": []}
    bindings = {}
    for name, spec in bank_config["historical_training_manifests"].items():
        frozen, bindings[name] = bound_asset(root, spec["path"], spec["sha256"])
        with frozen.open(newline="") as f:
            subset = list(csv.DictReader(f))
        for task, category in (("noise", "05Noise"), ("lowlight", "14Lowlight")) if name == "capacity" else (("haze", "08Haze"),):
            role = "tiny_overfit_train" if name == "capacity" else "pilot_train"
            pools[task] = [r for r in subset if r["category"] == category and r["role"] == role]
    selected = {r["record_id"]: r for pool in pools.values() for r in pool}
    matched = set()
    with main_path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != "train":
                blocked_groups.add(r["source_group"])
                blocked_gt.add(r["gt_sha256"])
            elif r["category"] == "01Blur" and r["composition_size"] == "1":
                pools["blur"].append(r)
            if r["record_id"] in selected:
                old = selected[r["record_id"]]
                for key in ("split", "source_group", "gt_sha256", "lq_sha256", "category", "composition_size"):
                    if old[key] != r[key] or r["split"] != "train":
                        raise ValueError(f"historical training record identity mismatch: {r['record_id']}")
                matched.add(r["record_id"])
    if matched != set(selected):
        raise ValueError("historical training rows absent from source manifest")
    overlap = [r["record_id"] for r in train
               if r["source_group"] in blocked_groups or r["gt_sha256"] in blocked_gt]
    if overlap:
        raise ValueError(f"Rain training overlaps other development metadata: {len(overlap)}")
    historical = {}
    for name, expected in {"blur": (105057, 7339), "noise": (16, 16),
                           "haze": (1580, 1580), "lowlight": (16, 16)}.items():
        pool = pools[name]
        actual = (len(pool), len({r["source_group"] for r in pool}))
        if actual != expected or len({r["record_id"] for r in pool}) != len(pool):
            raise ValueError(f"historical training cardinality mismatch: {name}: {actual}")
        # Include Rain's frozen development in addition to all source-manifest nontrain rows.
        all_blocked_groups = blocked_groups | {r["source_group"] for r in dev}
        all_blocked_gt = blocked_gt | {r["gt_sha256"] for r in dev}
        if any(r["source_group"] in all_blocked_groups or r["gt_sha256"] in all_blocked_gt for r in pool):
            raise ValueError(f"historical expert training overlaps current development: {name}")
        historical[name] = {"train_records": actual[0], "train_sources": actual[1],
                            "cross_task_nontrain_overlap": 0}
    return {**counts, "manifest": binding, "all_task_split_manifest": main_binding,
            "cross_task_nontrain_overlap": 0, "pixels_read": 0,
            "historical_training_manifests": bindings, "historical_training_pools": historical,
            "other_experts_declared_training_manifest_provenance_audited": True,
            "claim_scope": "declared_local_training_pools_only_not_FLUX_pretraining_data"}


def audit(asset_root):
    from safetensors import safe_open
    import torch
    torch.set_num_threads(1)
    source_bindings = {}

    def read(relative):
        p = inside(ROOT, relative)
        source_bindings[relative] = sha256(p)
        return json.loads(p.read_text())

    config = read("configs/foundir_existing_flux_experts_v1.json")
    source_bindings["smoke/audit_foundir_existing_flux_experts_v1.py"] = sha256(Path(__file__))
    source_bindings["lowlight_restoration/existing_flux_experts.py"] = sha256(ROOT / "lowlight_restoration/existing_flux_experts.py")
    first = read(config["experts"]["blur"]["training_config"])
    common_model = first["model"]
    assets = {}
    for key in ("flow_weights", "autoencoder_weights"):
        p, assets[key] = bound_asset(asset_root, common_model[key], common_model[key + "_sha256"])
        if key == "flow_weights":
            with safe_open(str(p), framework="pt", device="cpu") as f:
                base_shapes = {f"{m}.weight": f.get_slice(f"{m}.weight").get_shape() for m in expected_modules()}
    experts = {}
    common_lora = first["lora"]
    for name in config["flow_expert_order"]:
        item = config["experts"][name]
        evidence, training = read(item["evidence_config"]), read(item["training_config"])
        if evidence["model"]["parent_config"] != item["training_config"]:
            raise ValueError(f"wrong parent configuration: {name}")
        if sha256(ROOT / item["training_config"]) != evidence["model"]["parent_config_sha256"]:
            raise ValueError(f"parent configuration identity changed: {name}")
        for key in ("flow_weights_sha256", "autoencoder_weights_sha256"):
            if training["model"][key] != common_model[key]:
                raise ValueError(f"different shared backbone: {name}")
        for key in ("rank", "alpha", "dropout", "target_patterns", "expected_module_count"):
            if training["lora"][key] != common_lora[key]:
                raise ValueError(f"incompatible LoRA contract: {name}/{key}")
        if training["training"]["objective"] != "pure_noise_to_clean_rectified_flow_with_native_reference":
            raise ValueError(f"not a compatible flow objective: {name}")
        inf = evidence["inference"]
        if (inf["num_steps"] != 50 or inf["guidance"] != 1
                or inf["reference_time_coordinate"] != 10
                or inf.get("frozen_alpha", 1.0) != item["output_alpha"]):
            raise ValueError(f"original inference protocol changed: {name}")
        p, binding = bound_asset(asset_root, evidence["model"]["checkpoint"], evidence["model"]["checkpoint_sha256"])
        prompt = training["model"].get("prompt") or training["tasks"][item["category"]]["prompt"]
        experts[name] = {"checkpoint": binding, "weights": inspect_lora(p, base_shapes),
                         "inference": inf, "data_protocol": evidence["data"], "prompt": prompt,
                         "flow_contract": config["common_flow_contract"],
                         "full_rollout_equivalence_verified": False}
    rain = config["rain_teacher"]
    evidence, training = read(rain["evidence_config"]), read(rain["training_config"])
    precision = read(rain["precision_config"])
    for key in ("flow_weights_sha256", "autoencoder_weights_sha256"):
        if training["model"][key] != common_model[key]:
            raise ValueError("Rain backbone mismatch")
    p, binding = bound_asset(asset_root, evidence["rain_lora"], evidence["rain_lora_sha256"])
    rain_result = {"checkpoint": binding, "weights": inspect_lora(p, base_shapes),
                   "contract": rain["contract"], "allowed_in_flow_bank": False,
                   "precision": precision["training"],
                   "source_audit": rain_source_audit(asset_root, training, config)}
    for key in ("prompt_context_cache",):
        _, rain_result[key] = bound_asset(asset_root, training["model"][key], training["model"][key + "_sha256"])
    return {"schema_version": 1, "status": "asset_and_structure_audit_pass_only",
            "asset_root": str(asset_root),
            "destination_audit": asset_root.resolve() == Path(config["active_gpu_root"]).resolve(),
            "source_bindings": source_bindings, "shared_assets": assets, "experts": experts,
            "rain_teacher": rain_result, "pixels_read": 0, "gpu_used": False,
            "training_launch_ready": False, "scientific_success": False,
            "unresolved": ["full_model_one_hot_rollout_parity", "common_prompt_effect",
                           "mixed_output_alpha_contract",
                           "rain_multistep_learning_and_full_rollout_quality"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.asset_root.resolve())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        # Never overwrite an existing report or write into a frozen asset root.
        output = args.output.resolve()
        if output.is_relative_to(args.asset_root.resolve()):
            raise ValueError("report must be outside the read-only asset tree; use stdout there")
        with output.open("x") as f:
            f.write(encoded)
    else:
        print(encoded, end="")
    print("EXISTING_FLUX_EXPERTS_CPU_AUDIT_PASS; NOT_GPU_OR_TRAINING_READY", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
