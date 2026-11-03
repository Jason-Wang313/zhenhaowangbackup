"""Bound continuation from native50 step200; never restart its optimizer or sampler."""
import csv
from functools import lru_cache
from smoke import existing_flux_replay_contract_v1 as c
from smoke import rain_native50_warmstart_contract_v1 as native
from smoke import rain_flow_formal_stage1_contract_v1 as formal
from smoke import rain_flow_formal_stage1_state_v1 as parent_state

ROOT=c.ROOT
CONFIG="configs/foundir_rain_native50_continue_v1.json"
REPORT="reports/foundir_rain_native50_terminal_2026-11-03.json"
REPORT_SHA="360b8713d5f9106df419a3f46432ab0b95bf340b144f990e542b212e790aee62"
PARENT={"path":"checkpoints/foundir_rain_native50_warmstart_v1/native50_stage1_4x2_v1",
 "receipt_sha256":"ac284c0ed2ffe1d7291c7c4462f31632187036f1885d29d221f41c9333be61c2",
 "checkpoint_receipt_sha256":"23e62b72b004e8b76714f2114b565d8c10d641c4e349ee307f6b3b53c90fcfcd",
 "checkpoint_sha256":"8fdd6a0d5c6682ab616ba65580477329f3155aecab9da94ec6f1f943720cedcb",
 "result_sha256":"df75640e33bf7725b8f66186fd682953fbf81502228b4b42a2cfcb88e280a0fa"}
SOURCES=[CONFIG,REPORT,"smoke/rain_native50_continue_contract_v1.py","smoke/rain_native50_continue_state_v1.py",
 "smoke/train_foundir_rain_native50_continue_v1.py","smoke/rain_native50_continue_cluster_v1.py",
 "smoke/run_foundir_rain_native50_continue_v1_gpu.sh","smoke/start_foundir_rain_native50_continue_v1_tmux.sh"]
validate_row=formal.validate_row
validate_plan=formal.validate_plan
build_plan=formal.build_plan


def run_path(root,run_id):
    c.run_path(root,run_id)
    return c.path(root,f"checkpoints/foundir_rain_native50_continue_v1/{run_id}")


def validate_config(cfg):
    old=c.read(ROOT/native.CONFIG);native.validate_config(old)
    expected={**old,"phase":"rain_native50_exact_continuation_200_to_2000","start_step":200,"steps":2000,
        "initialization":"restore_native50_step200_LoRA_Adam_all_rank_RNG_absolute_position",
        "probe_steps":[400,700,1000,1500,2000],"wall_cap_seconds":28800,"minimum_free_disk_gib":1800,
        "final_verification":"light_summary_and_key_checkpoint_only_no_exhaustive_CPU_audit",
        "pause_rule":"two_consecutive_milestones_PSNR_nonincreasing_and_LPIPS_nondecreasing",
        "continuation_parent":PARENT}
    if cfg!=expected:raise ValueError("frozen continuation scope changed")


def terminal_report(root):
    if c.sha(root/REPORT)!=REPORT_SHA:raise ValueError("parent terminal report changed")
    report=c.read(root/REPORT)
    if (report["status"]!="rain_native50_warmstart_complete" or report["steps"]!=200
        or report["cpu_verify_status"]!=0 or len(report["nodes"])!=4 or any(n["status"] for n in report["nodes"])
        or report["result_sha256"]!=PARENT["result_sha256"]):raise ValueError("parent did not finish")
    return report


def validate_parent_rows(parent,receipt):
    native.validate_config(parent["config"]);native.validate_plan(parent)
    if (receipt["rows"][:6400]!=parent["rows"][:6400] or receipt["rows"][64000:]!=parent["rows"][6400:]
        or any(receipt[k]!=parent[k] for k in ("rain","metric","model","lora","native_checkpoint"))):
        raise ValueError("continuation changed original sampler/evaluation/teacher/model")


@lru_cache(maxsize=2)
def parent_receipt(root):
    output=c.path(root,PARENT["path"])
    if c.sha(output/"receipt.json")!=PARENT["receipt_sha256"]:raise ValueError("parent receipt changed")
    return native.load_receipt(root,output)


def validate_parent_state(state,record,world=8):
    if (state["step"]!=200 or state["world"]!=world or state["next_global_slot"]!=6400
        or len(state["history"])!=200 or len(state["rank_rng"])!=world
        or state["receipt_hash"]!=PARENT["receipt_sha256"] or record["sha256"]!=PARENT["checkpoint_sha256"]):
        raise ValueError("need exact native50 step200 full training state")
    groups=state["optimizer"]["param_groups"]
    if len(groups)!=1 or groups[0]["lr"]!=1e-5 or groups[0]["weight_decay"]!=0:
        raise ValueError("parent optimizer recipe changed")


def load_parent_state(root):
    parent_receipt(root)
    output=c.path(root,PARENT["path"])
    marker=output/"steps"/"step_000200.json"
    if c.sha(marker)!=PARENT["checkpoint_receipt_sha256"]:raise ValueError("parent step200 receipt changed")
    # Lightweight resume check: metadata receipts and the one required step200 state,
    # not all previous state tensors, caches, or evaluation outputs.
    state,record=parent_state.load_latest(output,PARENT["receipt_sha256"],8)
    if state is None:raise ValueError("missing parent step200")
    validate_parent_state(state,record)
    return state,record


def make_receipt(root,assets_root,require_parent=False):
    cfg=c.read(root/CONFIG);validate_config(cfg);report=terminal_report(root)
    prior=native.make_receipt(root,assets_root)
    old=c.read(root/"configs/foundir_rain_flux2_direct_endpoint_distillation_formal_v1.json")
    with c.path(assets_root,old["data"]["manifest"]).open(newline="") as f:pool=list(csv.DictReader(f))
    train=[r for r in pool if r["split"]=="train"]
    dev=[r for r in prior["rows"] if r["usage"]=="development"]
    probes=[r for r in prior["rows"] if r["usage"]=="train_probe"]
    rows=build_plan(train,dev,probes,cfg)
    result={**prior,"config":cfg,"rows":rows,"selection_digest":c.digest(rows),
        "sources":{**prior["sources"],**{n:c.sha(root/n) for n in SOURCES}},
        "continuation_parent":PARENT,"parent_report_sha256":REPORT_SHA,
        "historical_optimizer_reused":True,"training_exposures":64000,"additional_exposures":57600,
        "sampled_unique_records":len({r["record_id"] for r in rows[:64000]}),
        "sampled_unique_sources":len({r["source_group"] for r in rows[:64000]}),
        "additional_unique_records":len({r["record_id"] for r in rows[6400:64000]}),
        "additional_unique_sources":len({r["source_group"] for r in rows[6400:64000]}),
        "parent_milestone":report["milestones"]["200"]}
    validate_plan(result);validate_parent_rows(prior,result)
    # Main manifest/source/GT separation for the complete pool was already audited
    # by native.make_receipt; check all new image paths exist without decoding pixels.
    for name in {r[k] for r in rows[6400:64000] for k in ("lq_path","gt_path")}:
        if not c.path(assets_root,name).is_file():raise FileNotFoundError(name)
    if require_parent:
        if root.resolve()!=assets_root.resolve():raise ValueError("parent audit requires destination root")
        validate_parent_rows(parent_receipt(root),result)
        state,_=load_parent_state(root);del state
    return result


def load_receipt(root,output):
    receipt=c.load_receipt(root,output,8);validate_config(receipt["config"]);validate_plan(receipt)
    if receipt["continuation_parent"]!=PARENT or receipt["historical_optimizer_reused"] is not True:
        raise ValueError("continuation identity changed")
    validate_parent_rows(parent_receipt(root),receipt)
    return receipt


def required_free_gib(output):
    # Reserve only remaining additions on resume. Existing snapshots/cache stay.
    steps=len(list((output/"steps").glob("step_*.json")))
    cache=len(list((output/"cache").glob("image_*.json")))
    return max(100.,1800.-min(steps,1801)*.37-min(cache,57600)*.016)
