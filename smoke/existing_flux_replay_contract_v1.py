"""CPU contracts for no-training FLUX expert replay. Never decode GT images."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/foundir_existing_flux_replay_v1.json"
SOURCES = [CONFIG, "configs/foundir_existing_flux_experts_v1.json",
    "smoke/existing_flux_replay_cluster_v1.py",
    "smoke/existing_flux_replay_contract_v1.py", "smoke/replay_foundir_existing_flux_experts_v1.py",
    "smoke/run_foundir_existing_flux_replay_v1_gpu.sh", "smoke/start_foundir_existing_flux_replay_v1_tmux.sh",
    "lowlight_restoration/existing_flux_experts.py", "lowlight_restoration/lora.py",
    "lowlight_restoration/scalar_residual.py", "lowlight_restoration/flow.py",
    "smoke/evaluate_foundir_blur_only_flux2_full_development_ddp.py",
    "smoke/train_foundir_blur_only_flux2_overfit_ddp.py"]
CATEGORIES = dict(blur="01Blur", noise="05Noise", haze="08Haze", lowlight="14Lowlight")
TOPOLOGY = {"nnodes": 4, "nproc_per_node": 2, "world_size": 8}


def worker_rank(node, local, world, local_world, rank):
    if (node not in range(4) or local not in range(2) or world != 8
            or local_world != 2 or rank != node * 2 + local):
        raise ValueError("requires four nodes x two GPUs; invalid rank mapping")
    return rank, local


def validate_topology(receipt):
    if receipt.get("topology") != TOPOLOGY or receipt.get("world") != 8:
        raise ValueError("requires four-node topology; old single-node receipt rejected")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def path(root, relative):
    p = (Path(root) / relative).resolve()
    p.relative_to(Path(root).resolve())
    return p


def read(p):
    return json.loads(Path(p).read_text())


def write_new(p, value):
    """Publish complete immutable JSON; interrupted temporary files are preserved."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if p.exists():
        if p.read_bytes() != content:
            raise ValueError(f"immutable artifact conflict: {p}")
        return
    fd, name = tempfile.mkstemp(prefix=".pending-", suffix=".json", dir=p.parent)
    with os.fdopen(fd, "wb") as f:
        f.write(content); f.flush(); os.fsync(f.fileno())
    try:
        os.link(name, p)
    except FileExistsError:
        if p.read_bytes() != content:
            raise ValueError(f"concurrent artifact conflict: {p}")
    else:
        os.unlink(name)
    directory = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def run_path(root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run ID")
    return path(root, f"checkpoints/foundir_existing_flux_replay_v1/{run_id}")


def assigned(tasks, rows, rank, world):
    if world not in (4, 8) or not 0 <= rank < world or len(tasks) != 4:
        raise ValueError("invalid total rank partition")
    task = tasks[rank % 4]
    return task, rows[task][rank // 4::world // 4]


def stable(*values):
    return hashlib.sha256(":".join(map(str, values)).encode()).hexdigest()


def select_rows(manifest, config, experts):
    blocked_source, blocked_gt = set(), set()
    with Path(manifest).open(newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != "train":
                blocked_source.add(r["source_group"]); blocked_gt.add(r["gt_sha256"])
    candidates = {t: {} for t in config["tasks"]}
    reverse = {v: k for k, v in CATEGORIES.items()}
    with Path(manifest).open(newline="") as f:
        for r in csv.DictReader(f):
            task = reverse.get(r["category"])
            if task not in candidates or r["split"] != "train" or r["composition_size"] != "1":
                continue
            if r["source_group"] in blocked_source or r["gt_sha256"] in blocked_gt:
                continue
            if min(int(r["width"]), int(r["height"])) < config["crop_size"]:
                continue
            if not r["lq_path"].startswith(f"data/foundir/MillionIRData/Train/{r['category']}/LQ/"):
                raise ValueError("unexpected training LQ path")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", r["record_id"]):
                raise ValueError("unsafe record ID")
            if not all(re.fullmatch(r"[0-9a-f]{64}", r[k]) for k in ("lq_sha256", "gt_sha256", "source_group")):
                raise ValueError("invalid source/image hashes")
            key = stable(config["selection_seed"], task, r["record_id"])
            previous = candidates[task].get(r["source_group"])
            if previous is None or key < previous[0]:
                candidates[task][r["source_group"]] = (key, dict(r))
    result = {}
    for task in config["tasks"]:
        selected = [v[1] for v in sorted(candidates[task].values(), key=lambda v: v[0])[:config["records_per_task"]]]
        if len(selected) != config["records_per_task"]:
            raise ValueError(f"insufficient eligible training sources: {task}")
        for r in selected:
            protocol = experts[task]
            d = bytes.fromhex(stable(protocol["data_protocol"]["crop_seed"], "full_development_crop", r["record_id"]))
            crop = config["crop_size"]
            r["crop_xy"] = [int.from_bytes(d[8:16], "big") % (int(r["width"]) - crop + 1),
                            int.from_bytes(d[:8], "big") % (int(r["height"]) - crop + 1)]
            r["inference_seed"] = int(stable(protocol["inference"]["seed"], protocol["inference"]["seed_namespace"], r["record_id"])[:16], 16)
            # Carry GT identity only, not a GT path that an accidental decoder could open.
            r.pop("gt_path", None)
        result[task] = selected
    return result


def make_receipt(root, assets_root, world):
    cfg = read(root / CONFIG)
    if world not in cfg["world_sizes"] or cfg["topology"] != TOPOLOGY or cfg["read_gt"] or cfg["training"]:
        raise ValueError("invalid replay scope")
    audit_path = path(root, cfg["asset_audit"])
    if sha(audit_path) != cfg["asset_audit_sha256"]:
        raise ValueError("destination audit identity changed")
    audit = read(audit_path)
    if audit["destination_audit"] is not True or audit["training_launch_ready"] is not False:
        raise ValueError("wrong audit prerequisite")
    sources = {p: sha(path(root, p)) for p in SOURCES}
    sources.update({p.relative_to(root).as_posix(): sha(p)
                    for p in sorted((root / "third_party/flux2/src/flux2").rglob("*.py"))})
    if not any(p.startswith("third_party/") for p in sources):
        raise ValueError("FLUX source unavailable")
    for p, expected in audit["source_bindings"].items():
        if sha(path(root, p)) != expected:
            raise ValueError(f"audited expert source changed: {p}")
        sources[p] = expected
    manifest = audit["rain_teacher"]["source_audit"]["all_task_split_manifest"]
    bindings = {v["path"]: v["sha256"] for v in audit["shared_assets"].values()}
    bindings.update({v["checkpoint"]["path"]: v["checkpoint"]["sha256"] for v in audit["experts"].values()})
    bindings[manifest["path"]] = manifest["sha256"]
    bindings.update(cfg["qwen_artifacts"])
    asset_stats = {}
    for p, expected in bindings.items():
        asset = path(assets_root, p)
        before = asset.stat()
        if sha(asset) != expected:
            raise ValueError(f"runtime asset changed: {p}")
        after = asset.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"asset changed while hashing: {p}")
        asset_stats[p] = [after.st_size, after.st_mtime_ns]
    rows = select_rows(path(assets_root, manifest["path"]), cfg, audit["experts"])
    for pool in rows.values():
        for r in pool:
            if not path(assets_root, r["lq_path"]).is_file():
                raise FileNotFoundError(r["lq_path"])
    return {"schema_version": 2, "config": cfg, "world": world, "topology": TOPOLOGY, "sources": sources,
            "assets": bindings, "asset_stats": asset_stats, "experts": audit["experts"], "rows": rows,
            "selection_digest": digest(rows), "gt_pixels_read": False,
            "historical_pixel_reproduction_claimed": False,
            "comparison": "same_environment_original_lora_vs_static_bank_selection"}


def load_receipt(root, output, world):
    receipt = read(output / "receipt.json")
    if receipt["world"] != world:
        raise ValueError("resume at different world size is not qualified")
    validate_topology(receipt)
    for p, expected in receipt["sources"].items():
        if sha(path(root, p)) != expected:
            raise ValueError(f"run source changed: {p}")
    for p, expected in receipt["asset_stats"].items():
        current = path(root, p).stat()
        if [current.st_size, current.st_mtime_ns] != expected:
            raise ValueError(f"asset changed since prepare: {p}")
    if digest(receipt["rows"]) != receipt["selection_digest"]:
        raise ValueError("selection changed")
    return receipt


def validate_record(info, identity, output):
    if info["identity"] != identity:
        raise ValueError("record identity changed")
    for name, expected in info["files"].items():
        if sha(path(output, name)) != expected:
            raise ValueError(f"saved tensor changed: {name}")


def validate_summary(records, expected, tolerance):
    ids = [r["identity"]["key"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("missing/duplicate/extra replay records")
    import math
    for r in records:
        x = r["max_abs_rgb"]
        if not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0:
            raise ValueError("invalid replay difference")
        passed = x <= tolerance and r["uint8_equal"] is True
        if r["pass"] is not passed:
            raise ValueError("inconsistent replay verdict")
    return all(r["pass"] for r in records)
