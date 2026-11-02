"""Native-flow Rain initialization; matched full-pool schedule and frozen cache reuse."""
import csv
from functools import lru_cache
from smoke import existing_flux_replay_contract_v1 as c
from smoke import rain_flow_formal_stage1_contract_v1 as formal

ROOT=c.ROOT
CONFIG="configs/foundir_rain_native50_warmstart_v1.json"
NATIVE={"path":"checkpoints/foundir_rain_flux2_source_diverse_pilot_v1/step_002000.safetensors",
        "sha256":"2c4537258783f51ad15163b1a3377ce811d1418d51b4da68b5d75ef502bd45af"}
MANIFEST={"path":"manifests/foundir_rain_flux2_source_diverse_pilot_v1.csv",
          "sha256":"ace5c8cb6da1a1628fe3ab3f7eec9e0a50feb1a12bb2b26732f06726f552636b"}
PARENT={"path":"checkpoints/foundir_rain_flow_formal_stage1_v1/formal_stage1_4x2_v1",
        "receipt_sha256":"8b2209f880307af41eec03c571a8c98cf0661f7714ac8d4f63d63cd96dfb75ae"}
NATIVE_CONFIG="configs/foundir_rain_flux2_source_diverse_pilot_v1.json"
SOURCES=[CONFIG,NATIVE_CONFIG,"smoke/rain_native50_warmstart_contract_v1.py",
 "smoke/train_foundir_rain_native50_warmstart_v1.py","smoke/rain_native50_warmstart_cluster_v1.py",
 "smoke/run_foundir_rain_native50_warmstart_v1_gpu.sh","smoke/start_foundir_rain_native50_warmstart_v1_tmux.sh",
 "smoke/verify_foundir_rain_native50_warmstart_v1_cpu.sh"]
validate_row=formal.validate_row
validate_plan=formal.validate_plan
build_plan=formal.build_plan


def run_path(root,run_id):
    c.run_path(root,run_id)
    return c.path(root,f"checkpoints/foundir_rain_native50_warmstart_v1/{run_id}")


def validate_config(cfg):
    original=c.read(ROOT/formal.CONFIG);formal.validate_config(original)
    expected={**original,"phase":"rain_native50_warmstart_stage1",
        "initialization":"copy_native50_Rain_step2000_promote_FP32_fresh_Adam",
        "minimum_free_disk_gib":100,"native_checkpoint":NATIVE,"native_manifest":MANIFEST,
        "native_config":NATIVE_CONFIG,"cache_parent":PARENT,
        "final_verification":"separate_CPU_after_GPU_workers_exit"}
    if cfg!=expected:raise ValueError("frozen native50 warmstart scope changed")


def audit_native_rows(old_rows,development):
    train=[r for r in old_rows if r["split"]=="train"]
    dev=[r for r in old_rows if r["split"]=="validation"]
    if len(train)!=1580 or len(dev)!=100 or len(old_rows)!=1680:
        raise ValueError("native50 historical manifest cardinality changed")
    for key in ("source_group","gt_sha256","record_id"):
        if {r[key] for r in train}&{r[key] for r in development}:
            raise ValueError("native50 initialization trained on current development")
    by_id={r["record_id"]:r for r in development}
    if len(by_id)!=100 or {r["record_id"] for r in dev}!=set(by_id):
        raise ValueError("old and current development identities differ")
    for row in dev:
        if any(row[k]!=by_id[row["record_id"]][k] for k in
               ("source_group","lq_sha256","gt_sha256","crop_x","crop_y","width","height")):
            raise ValueError("old and current development crops/files differ")


def validate_parent(parent,receipt):
    formal.validate_config(parent["config"]);formal.validate_plan(parent)
    if (parent["rows"]!=receipt["rows"] or parent["selection_digest"]!=receipt["selection_digest"]
        or parent["rain"]!=receipt["rain"] or parent["metric"]!=receipt["metric"]
        or parent["model"]!=receipt["model"] or parent["lora"]!=receipt["lora"]):
        raise ValueError("cache source differs in crop/sampling/teacher/latent/metric contract")


@lru_cache(maxsize=2)
def cache_receipt(root):
    output=c.path(root,PARENT["path"])
    if c.sha(output/"receipt.json")!=PARENT["receipt_sha256"]:
        raise ValueError("parent cache receipt changed")
    # The parent training completed, but final verification timed out. Recheck the
    # exact producer contract now; every cache tensor is independently checked on use.
    return formal.load_receipt(root,output)


def make_receipt(root,assets_root,require_cache=False):
    cfg=c.read(root/CONFIG);validate_config(cfg)
    prior=formal.make_receipt(root,assets_root)
    old=c.read(root/NATIVE_CONFIG)
    if (any(old["model"][k]!=prior["model"][k] for k in
            ("name","flow_weights","flow_weights_sha256","autoencoder_weights","autoencoder_weights_sha256","text_encoder"))
        or old["model"]["prompt"]!="Remove rain streaks from the reference image and restore a clean image. Preserve the exact content, geometry, identity, color, and texture."
        or old["lora"]["rank"]!=32
        or old["lora"]["alpha"]!=32 or old["lora"]["target_patterns"]!=prior["lora"]["target_patterns"]
        or old["training"]["objective"]!="pure_noise_to_clean_rectified_flow_with_native_reference"
        or old["inference"]["num_steps"]!=50):
        raise ValueError("native50 base/prompt/LoRA/flow incompatible")
    assets=dict(prior["assets"]);stats=dict(prior["asset_stats"])
    for info in (NATIVE,MANIFEST):
        file=c.path(assets_root,info["path"]);before=file.stat()
        if c.sha(file)!=info["sha256"]:raise ValueError("native50 asset changed")
        after=file.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError("native50 asset changed during audit")
        assets[info["path"]]=info["sha256"];stats[info["path"]]=[after.st_size,after.st_mtime_ns]
    with c.path(assets_root,MANIFEST["path"]).open(newline="") as f:old_rows=list(csv.DictReader(f))
    audit_native_rows(old_rows,[r for r in prior["rows"] if r["usage"]=="development"])
    result={**prior,"config":cfg,"assets":assets,"asset_stats":stats,
        "sources":{**prior["sources"],**{n:c.sha(root/n) for n in SOURCES}},
        "native_checkpoint":NATIVE,"cache_parent":PARENT,"historical_optimizer_reused":False,
        "native_training_overlaps_current_development":False,
        "historical_development_same_images_and_crops":True}
    validate_plan(result)
    if require_cache:
        if root.resolve()!=assets_root.resolve():raise ValueError("cache audit requires destination root")
        parent=cache_receipt(root);validate_parent(parent,result)
        output=c.path(root,PARENT["path"])
        # Metadata-only preflight; full tensor checks are sharded across workers.
        for index,row in enumerate(result["rows"]):
            marker=output/"cache"/f"image_{index:03d}.json";entry=c.read(marker)
            expected={"receipt_sha256":PARENT["receipt_sha256"],"index":index,"row":row,"usage":row["usage"]}
            if entry["identity"]!=expected or entry["tensor"] not in entry["files"]:
                raise ValueError("cache identity changed")
            if entry["teacher"]!="frozen_original_Rain_FLUX_step4000":raise ValueError("wrong cache teacher")
            if not c.path(output,entry["tensor"]).is_file():raise FileNotFoundError(entry["tensor"])
    return result


def load_receipt(root,output):
    receipt=c.load_receipt(root,output,8);validate_config(receipt["config"]);validate_plan(receipt)
    if receipt["native_checkpoint"]!=NATIVE or receipt["cache_parent"]!=PARENT:
        raise ValueError("native/cache identity changed")
    if receipt["historical_optimizer_reused"] is not False:raise ValueError("fresh Adam required")
    validate_parent(cache_receipt(root),receipt)
    return receipt
