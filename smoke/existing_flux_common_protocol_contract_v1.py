"""Fixed 2x2 prompt/output diagnosis. Only parent-selected training GT may be scored."""
import csv
import math
from pathlib import Path

from smoke import existing_flux_replay_contract_v1 as c

ROOT = c.ROOT
CONFIG = "configs/foundir_existing_flux_common_protocol_v1.json"
SOURCES = [CONFIG, "smoke/existing_flux_common_protocol_contract_v1.py",
           "smoke/diagnose_foundir_existing_flux_common_protocol_v1.py",
           "smoke/existing_flux_common_protocol_cluster_v1.py",
           "smoke/run_foundir_existing_flux_common_protocol_v1_gpu.sh",
           "smoke/start_foundir_existing_flux_common_protocol_v1_tmux.sh"]
ARMS = ("original_prompt_original_output", "common_prompt_original_output",
        "original_prompt_common_output", "common_prompt_common_output")


def run_path(root, run_id):
    c.run_path(root, run_id)  # Identifier validation only.
    return c.path(root, f"checkpoints/foundir_existing_flux_common_protocol_v1/{run_id}")


def select_gt(manifest, rows):
    expected = {row["record_id"]: row for pool in rows.values() for row in pool}
    if len(expected) != sum(map(len, rows.values())):
        raise ValueError("duplicate parent record")
    found = {}
    with Path(manifest).open(newline="") as f:
        for row in csv.DictReader(f):
            key = row["record_id"]
            if key not in expected:
                continue
            parent = expected[key]
            if key in found or any(row.get(k) != v for k, v in parent.items()
                                   if k not in ("crop_xy", "inference_seed")):
                raise ValueError("GT mapping differs from parent metadata")
            prefix = f"data/foundir/MillionIRData/Train/{row['category']}/GT/"
            if row["split"] != "train" or not row["gt_path"].startswith(prefix) or ".." in Path(row["gt_path"]).parts:
                raise ValueError("only selected training GT allowed")
            found[key] = {"path": row["gt_path"], "sha256": row["gt_sha256"]}
    if set(found) != set(expected):
        raise ValueError("missing selected training GT mapping")
    return found


def terminal(root, cfg):
    p = c.path(root, cfg["parent_report"])
    if c.sha(p) != cfg["parent_report_sha256"]:
        raise ValueError("parent terminal report changed")
    report = c.read(p)
    full = report["full_result"]
    pairs = full["comparisons"]
    if (report["run_id"] != cfg["parent_run_id"] or not full["pass"] or full["records"] != 128
            or full["topology"] != c.TOPOLOGY or len(pairs) != 128
            or len({v["identity"]["key"] for v in pairs}) != 128
            or any(not v["pass"] or not v["uint8_equal"] or v["max_abs_rgb"] != 0 for v in pairs)):
        raise ValueError("parent is not complete 128/128 static parity")
    return report


def make_receipt(root, assets_root, *, require_parent):
    cfg = c.read(root / CONFIG)
    if (cfg["topology"] != c.TOPOLOGY or cfg["training"] or cfg["router"]
            or cfg["automatic_follow_on"] or not cfg["read_gt"] or tuple(cfg["arms"]) != ARMS):
        raise ValueError("invalid protocol scope")
    report = terminal(root, cfg)
    parent = c.make_receipt(root, assets_root, 8)
    if parent["selection_digest"] != report["selection_digest"]:
        raise ValueError("different training source selection")
    manifest = next(k for k in parent["assets"] if k.endswith("foundir_five_type_pairs_v1.csv"))
    gt = select_gt(c.path(assets_root, manifest), parent["rows"])
    for item in gt.values():
        if not c.path(assets_root, item["path"]).is_file():
            raise FileNotFoundError(item["path"])
    metric_path = cfg["metric"]["alexnet_path"]
    p = c.path(assets_root, metric_path)
    before = p.stat()
    if c.sha(p) != cfg["metric"]["alexnet_sha256"]:
        raise ValueError("offline AlexNet weight changed")
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("metric weight changed while hashing")
    parent["assets"][metric_path] = cfg["metric"]["alexnet_sha256"]
    parent["asset_stats"][metric_path] = [after.st_size, after.st_mtime_ns]
    if require_parent:
        check_parent(root, report)
    parent["sources"].update({name: c.sha(root / name) for name in SOURCES + [cfg["parent_report"]]})
    return {**parent, "schema_version": 1, "config": cfg, "gt": gt,
            "parent_artifact_hashes": report["artifact_hashes"],
            "parent_runtimes": report["full_result"]["node_runtimes"],
            "comparison": "fixed_2x2_prompt_output_training_source_diagnosis",
            "gt_pixels_read": False, "gt_will_be_read_for_metrics": True,
            "training_fit_only": True, "parent_runtime_artifacts_verified": require_parent}


def check_parent(root, report):
    for name, expected in report["artifact_hashes"].items():
        path = c.path(root, name)
        if isinstance(expected, dict):
            if (set(expected) != {"sha256", "bytes"}
                    or type(expected["bytes"]) is not int
                    or path.stat().st_size != expected["bytes"]):
                raise ValueError(f"parent artifact size/receipt changed: {name}")
            expected = expected["sha256"]
        if c.sha(path) != expected:
            raise ValueError(f"parent artifact changed: {name}")
    return c.load_receipt(root, c.run_path(root, report["run_id"]), 8)


def load_receipt(root, output):
    receipt = c.load_receipt(root, output, 8)
    if not receipt["parent_runtime_artifacts_verified"]:
        raise ValueError("CPU-only source preflight is not destination preparation")
    report = terminal(root, receipt["config"])
    check_parent(root, report)
    if c.digest(receipt["gt"]) != c.digest(select_gt(
            c.path(root, next(k for k in receipt["assets"] if k.endswith("foundir_five_type_pairs_v1.csv"))),
            receipt["rows"])):
        raise ValueError("GT mapping changed")
    return receipt


def aggregate(records, expected_keys, tolerance):
    if len(records) != len(expected_keys) or {v["identity"]["key"] for v in records} != set(expected_keys):
        raise ValueError("incomplete/duplicate metrics")
    tasks = sorted({v["identity"]["task"] for v in records})
    result = {}
    for task in tasks:
        subset = [v for v in records if v["identity"]["task"] == task]
        means = {}
        for row in subset:
            if set(row["metrics"]) != set(ARMS):
                raise ValueError("missing or extra metric arm")
            for arm in ARMS:
                for metric in ("psnr", "lpips"):
                    value = row["metrics"][arm][metric]
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("nonfinite metric")
        for arm in ARMS:
            means[arm] = {m: sum(v["metrics"][arm][m] for v in subset)/len(subset) for m in ("psnr","lpips")}
        base = means[ARMS[0]]
        deltas = {arm: {m: means[arm][m]-base[m] for m in base} for arm in ARMS[1:]}
        result[task] = {"records":len(subset), "means":means, "deltas_vs_original":deltas,
            "descriptive_mean_nonregression": {arm: d["psnr"] >= -tolerance["psnr_db"]
                and d["lpips"] <= tolerance["lpips"] for arm,d in deltas.items()},
            "per_arm_harmed_images": {arm: sum(v["metrics"][arm]["psnr"] < v["metrics"][ARMS[0]]["psnr"]-tolerance["psnr_db"]
                or v["metrics"][arm]["lpips"] > v["metrics"][ARMS[0]]["lpips"]+tolerance["lpips"] for v in subset)
                for arm in ARMS[1:]}}
    return result
