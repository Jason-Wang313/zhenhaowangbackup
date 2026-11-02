"""Metadata-only source selection and exact asset binding for Rain qualification."""
import csv
from smoke import existing_flux_replay_contract_v1 as c

ROOT=c.ROOT
CONFIG="configs/foundir_rain_flow_adaptation_qualification_v1.json"
SOURCES=[CONFIG,"lowlight_restoration/rain_flow_adaptation.py","lowlight_restoration/flux2_endpoint_precision.py",
 "smoke/rain_flow_adaptation_contract_v1.py","smoke/rain_flow_adaptation_state_v1.py",
 "smoke/train_foundir_rain_flow_adaptation_v1.py","smoke/rain_flow_adaptation_cluster_v1.py",
 "smoke/run_foundir_rain_flow_adaptation_v1_gpu.sh","smoke/start_foundir_rain_flow_adaptation_v1_tmux.sh",
 "smoke/diagnose_foundir_existing_flux_common_protocol_v1.py",
 "configs/foundir_existing_flux_common_protocol_v1.json",
 "configs/foundir_rain_flux2_direct_endpoint_distillation_formal_v1.json"]


def run_path(root,run_id):
    c.run_path(root,run_id)
    return c.path(root,f"checkpoints/foundir_rain_flow_adaptation_v1/{run_id}")


def validate_config(cfg):
    required={"topology":c.TOPOLOGY,"gpu_family":"H200","minimum_memory_gib":139,
        "training_sources":32,"probe_sources":8,"crop_size":512,"augmentation":False,"steps":25,
        "global_batch":32,"microbatch":1,"accumulation":4,"learning_rate":1e-5,"weight_decay":0.,
        "max_grad_norm":1.,"t_sampling":"uniform_0.0001_to_0.9999",
        "loss_weights":{"flow_mse":1.,"gt_latent_l1":.25,"gt_rgb_mse":10.,"gt_lpips":.05,"teacher_rgb_l1":.1},
        "inference_steps":50,"reference_time":10,"guidance":1.,"output_alpha":1.,"probe_steps":[0,25],
        "checkpoint_every_step":True,"qualification_may_seed_formal":False,"automatic_follow_on":False,
        "training_fit_only":True,"development_pixels_allowed":False,"wall_cap_seconds":3600}
    if any(cfg.get(k)!=v for k,v in required.items()):raise ValueError("frozen Rain scope changed")


def select_rows(manifest,main_manifest,seed):
    with manifest.open(newline="") as f:rows=list(csv.DictReader(f))
    train=[r for r in rows if r["split"]=="train" and r["role"]=="direct_endpoint_train"]
    dev=[r for r in rows if r["split"]=="validation" and r["role"]=="direct_endpoint_validation"]
    if len(train)!=38334 or len(dev)!=100 or len(rows)!=38434 or len({r["source_group"] for r in train})!=5780:
        raise ValueError("Rain manifest cardinality changed")
    if len({r["record_id"] for r in rows})!=len(rows):raise ValueError("duplicate record")
    blocked={k:{r[k] for r in dev} for k in ("source_group","gt_sha256")}
    selected_ids={r["record_id"]:r for r in train};matched=set()
    with main_manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"]!="train":
                for k in blocked:blocked[k].add(row[k])
            if row["record_id"] in selected_ids:
                original=selected_ids[row["record_id"]]
                if any(original[k]!=row[k] for k in ("split","source_group","gt_sha256","lq_sha256","category","lq_path","gt_path","width","height")):
                    raise ValueError("Rain/main manifest identity mismatch")
                matched.add(row["record_id"])
    if matched!=set(selected_ids):raise ValueError("Rain training record absent from main manifest")
    groups={}
    for row in train:
        if row["category"]!="10Rain" or row["composition_size"]!="1" or any(row[k] in blocked[k] for k in blocked):
            raise ValueError("training source overlaps forbidden development identities")
        for side in ("lq","gt"):
            prefix=f"data/foundir/MillionIRData/Train/10Rain/{side.upper()}/"
            if not row[side+"_path"].startswith(prefix) or ".." in row[side+"_path"].split("/"):
                raise ValueError("not a training image path")
        if min(int(row["width"]),int(row["height"]))>=512:
            groups.setdefault(row["source_group"],[]).append(row)
    key=lambda value:c.stable(seed,"rain_flow_v1",value)
    chosen=[]
    for group in sorted(groups,key=key)[:32]:
        row=dict(min(groups[group],key=lambda r:key(r["record_id"])))
        row["crop_xy"]=[(int(row["width"])-512)//2,(int(row["height"])-512)//2]
        chosen.append(row)
    if len(chosen)!=32:raise ValueError("insufficient training sources")
    return chosen


def make_receipt(root,assets_root):
    cfg=c.read(root/CONFIG);validate_config(cfg)
    if c.sha(root/cfg["parent_report"])!=cfg["parent_report_sha256"]:raise ValueError("parent report changed")
    report=c.read(root/cfg["parent_report"])
    if report["status"]!="protocol_routed_engineering_pass" or any(n["status"] for n in report["nodes"]):
        raise ValueError("parent interface did not pass")
    prior=c.make_receipt(root,assets_root,8)
    audit=c.read(root/prior["config"]["asset_audit"]);rain=audit["rain_teacher"]
    old=c.read(root/"configs/foundir_rain_flux2_direct_endpoint_distillation_formal_v1.json")
    metric=c.read(root/"configs/foundir_existing_flux_common_protocol_v1.json")["metric"]
    assets={**prior["assets"],rain["checkpoint"]["path"]:rain["checkpoint"]["sha256"],
        rain["prompt_context_cache"]["path"]:rain["prompt_context_cache"]["sha256"],
        old["data"]["manifest"]:old["data"]["manifest_sha256"],metric["alexnet_path"]:metric["alexnet_sha256"]}
    stats=dict(prior["asset_stats"])
    for name,digest in assets.items():
        if name in stats:continue
        file=c.path(assets_root,name);before=file.stat()
        if c.sha(file)!=digest:raise ValueError("asset changed: "+name)
        after=file.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError("asset changed during hash")
        stats[name]=[after.st_size,after.st_mtime_ns]
    rows=select_rows(c.path(assets_root,old["data"]["manifest"]),
        c.path(assets_root,rain["source_audit"]["all_task_split_manifest"]["path"]),cfg["selection_seed"])
    for row in rows:
        for side in ("lq","gt"):
            if not c.path(assets_root,row[side+"_path"]).is_file():raise FileNotFoundError(row[side+"_path"])
    sources={**prior["sources"],**{name:c.sha(root/name) for name in SOURCES+[cfg["parent_report"]]}}
    return {**prior,"config":cfg,"metric":metric,"assets":assets,"asset_stats":stats,"sources":sources,
        "rows":rows,"selection_digest":c.digest(rows),"rain":rain,"model":old["model"],"lora":old["lora"],
        "training_fit_only":True,"teacher_is_FLUX_not_Restormer":True,"protected_pixels_read":False}


def load_receipt(root,output):
    receipt=c.load_receipt(root,output,8);validate_config(receipt["config"])
    if len(receipt["rows"])!=32 or len({r["source_group"] for r in receipt["rows"]})!=32:
        raise ValueError("training subset changed")
    return receipt
