"""Full-pool Rain training, frozen development evaluation, bounded first 200 steps."""
import csv
import random
from smoke import existing_flux_replay_contract_v1 as c
from smoke import rain_flow_adaptation_contract_v1 as parent

ROOT=c.ROOT
CONFIG="configs/foundir_rain_flow_formal_stage1_v1.json"
REPORT="reports/foundir_rain_flow_adaptation_driverfix_terminal_2026-11-02.json"
REPORT_SHA="e99cdec5329eb6de6c74d230eb301f1eb9148e322a33bd85add53682b77624bb"
SOURCES=[CONFIG,"smoke/rain_flow_formal_stage1_contract_v1.py","smoke/rain_flow_formal_stage1_state_v1.py",
         "smoke/train_foundir_rain_flow_formal_stage1_v1.py","smoke/rain_flow_formal_stage1_cluster_v1.py",
         "smoke/run_foundir_rain_flow_formal_stage1_v1_gpu.sh","smoke/start_foundir_rain_flow_formal_stage1_v1_tmux.sh",REPORT]


def run_path(root,run_id):
    c.run_path(root,run_id)
    return c.path(root,f"checkpoints/foundir_rain_flow_formal_stage1_v1/{run_id}")


def validate_config(cfg):
    old=c.read(ROOT/parent.CONFIG);parent.validate_config(old)
    expected={**old,"phase":"rain_endpoint_to_native_flow_formal_stage1","steps":200,
        "training_records":38334,"training_sources":5780,"probe_sources":132,
        "training_probe_sources":32,"development_sources":100,"augmentation":True,
        "probe_steps":[0,50,100,200],"minimum_free_disk_gib":220,
        "qualification_report":REPORT,"qualification_report_sha256":REPORT_SHA,
        "training_fit_only":False,"development_pixels_allowed":True,
        "sampling":"uniform_source_then_record_absolute_position_random512_d4",
        "formal_continuation_requires_manual_authorization":True,
        "official_gate":{**old["official_gate"],"protocol":"frozen_Rain_100_source_disjoint_development","evaluated_this_run":True}}
    if cfg!=expected:raise ValueError("frozen formal first-segment scope changed")


def build_plan(training,development,train_probe,cfg):
    """Absolute-position RNG: every exposure can be resumed without advancing a sampler."""
    groups={}
    for row in training:groups.setdefault(row["source_group"],[]).append(row)
    keys=sorted(groups)
    for key in keys:groups[key].sort(key=lambda r:r["record_id"])
    rows=[]
    for position in range(cfg["steps"]*cfg["global_batch"]):
        rng=random.Random(cfg["seed"]+1000003*(position+1))
        row=dict(rng.choice(groups[rng.choice(keys)]))
        width,height=max(512,int(row["width"])),max(512,int(row["height"]))
        row.update(usage="train_update",sample_position=position,
                   crop_xy=[rng.randrange(width-512+1),rng.randrange(height-512+1)],augmentation_code=rng.randrange(8))
        rows.append(row)
    for usage,subset in (("train_probe",train_probe),("development",development)):
        for row in subset:
            row=dict(row)
            if usage=="development":row["crop_xy"]=[int(row["crop_x"]),int(row["crop_y"])]
            row.update(usage=usage,sample_position=None,augmentation_code=0)
            rows.append(row)
    return rows


def validate_row(row):
    usage=row.get("usage")
    split,role=("validation","direct_endpoint_validation") if usage=="development" else ("train","direct_endpoint_train")
    if (usage not in ("train_update","train_probe","development") or row["split"]!=split
        or row["role"]!=role or row["category"]!="10Rain" or row["composition_size"]!="1"):
        raise ValueError("only Rain train updates and frozen development evaluation allowed")
    w,h=int(row["width"]),int(row["height"]);x,y=row["crop_xy"]
    if min(w,h)<256 or min(x,y)<0 or x+512>max(w,512) or y+512>max(h,512):
        raise ValueError("invalid crop geometry")
    if usage!="train_update" and (min(w,h)<512 or row["augmentation_code"]!=0):
        raise ValueError("evaluation cannot pad/augment")
    if usage=="development" and [x,y]!=[int(row["crop_x"]),int(row["crop_y"])]:
        raise ValueError("frozen development crop changed")
    if not isinstance(row["augmentation_code"],int) or row["augmentation_code"] not in range(8):
        raise ValueError("invalid augmentation")
    for side in ("lq","gt"):
        prefix=f"data/foundir/MillionIRData/Train/10Rain/{side.upper()}/"
        if not row[side+"_path"].startswith(prefix) or ".." in row[side+"_path"].split("/"):
            raise ValueError("forbidden pixel path")


def validate_plan(receipt):
    cfg=receipt["config"];rows=receipt["rows"];count=cfg["steps"]*cfg["global_batch"]
    if len(rows)!=count+132:raise ValueError("wrong scheduled record count")
    for index,row in enumerate(rows):
        validate_row(row)
        usage="train_update" if index<count else "train_probe" if index<count+32 else "development"
        if row["usage"]!=usage or row["sample_position"]!=(index if index<count else None):
            raise ValueError("sampler/evaluation boundary changed")
    training=rows[:count+32];development=rows[count+32:]
    if len({r["source_group"] for r in development})!=100:raise ValueError("development sources changed")
    for key in ("source_group","gt_sha256","record_id"):
        if {r[key] for r in training}&{r[key] for r in development}:raise ValueError("development leaks into training")


def make_receipt(root,assets_root):
    cfg=c.read(root/CONFIG);validate_config(cfg)
    if c.sha(root/REPORT)!=REPORT_SHA:raise ValueError("qualification report changed")
    report=c.read(root/REPORT)
    if (report["status"]!="rain_flow_engineering_pass" or report["steps"]!=25
        or len(report["nodes"])!=4 or any(n["status"]!=0 for n in report["nodes"])
        or len(report["checkpoint_hashes"])!=26):raise ValueError("matching qualification required")
    # Parent audits ALL 38,334 train rows against the main manifest and all non-train source/GT hashes.
    prior=parent.make_receipt(root,assets_root)
    old=c.read(root/"configs/foundir_rain_flux2_direct_endpoint_distillation_formal_v1.json")
    with c.path(assets_root,old["data"]["manifest"]).open(newline="") as f:pool=list(csv.DictReader(f))
    training=[r for r in pool if r["split"]=="train"]
    development=sorted([r for r in pool if r["split"]=="validation"],key=lambda r:r["record_id"])
    if len(training)!=38334 or len({r["source_group"] for r in training})!=5780 or len(development)!=100:
        raise ValueError("full data pool changed")
    if sum(min(int(r["width"]),int(r["height"]))<512 for r in training)!=35:
        raise ValueError("padding population changed")
    rows=build_plan(training,development,prior["rows"],cfg)
    for row in rows:
        for side in ("lq","gt"):
            if not c.path(assets_root,row[side+"_path"]).is_file():raise FileNotFoundError(row[side+"_path"])
    result={**prior,"config":cfg,"rows":rows,"selection_digest":c.digest(rows),
        "sources":{**prior["sources"],**{name:c.sha(root/name) for name in SOURCES}},
        "qualification_report_sha256":REPORT_SHA,"qualification_weights_reused":False,
        "training_fit_only":False,"development_evaluation_only":True,
        "pool_records":38334,"pool_sources":5780,"training_exposures":6400,
        "sampled_unique_records":len({r["record_id"] for r in rows[:6400]}),
        "sampled_unique_sources":len({r["source_group"] for r in rows[:6400]})}
    validate_plan(result)
    return result


def load_receipt(root,output):
    receipt=c.load_receipt(root,output,8);validate_config(receipt["config"]);validate_plan(receipt)
    if receipt["qualification_weights_reused"] is not False or receipt["qualification_report_sha256"]!=REPORT_SHA:
        raise ValueError("formal stage identity changed")
    return receipt
