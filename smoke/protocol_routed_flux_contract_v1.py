"""Qualification contract for explicit internal routes and inherited RGB/prompt protocols."""
import math
from smoke import existing_flux_replay_contract_v1 as c
from smoke import existing_flux_common_protocol_contract_v1 as parent

ROOT=c.ROOT
CONFIG="configs/foundir_protocol_routed_flux_qualification_v1.json"
SOURCES=[CONFIG,"lowlight_restoration/protocol_routed_flux.py",
         "smoke/protocol_routed_flux_contract_v1.py","smoke/qualify_foundir_protocol_routed_flux_v1.py",
         "smoke/protocol_routed_flux_cluster_v1.py","smoke/run_foundir_protocol_routed_flux_v1_gpu.sh",
         "smoke/start_foundir_protocol_routed_flux_v1_tmux.sh"]


def run_path(root,run_id):
    c.run_path(root,run_id)
    return c.path(root,f"checkpoints/foundir_protocol_routed_flux_v1/{run_id}")


def validate_config(cfg):
    required={"topology":c.TOPOLOGY,"gpu_family":"H200","minimum_memory_gib":139,
              "tasks":list(c.CATEGORIES),"records_per_task":2,"steps":50,"crop_size":512,
              "guidance":1.,"reference_time":10,"rgb_max_abs_tolerance":1e-6,
              "require_one_hot_uint8_equal":True,"read_gt":True,"wall_cap_seconds":1800,
              "arms":["one_hot","near_one_hot"],"near_one_hot_primary_weight":.9,
              "near_one_hot_other_weight":.1/3,"training":False,"learned_router":False,
              "automatic_follow_on":False,"anchor_policy":"argmax_fixed_expert_order",
              "prompt_policy":"inherit_anchor_original_prompt",
              "output_policy":"inherit_anchor_original_alpha_and_quantization_order"}
    if any(cfg.get(k)!=v for k,v in required.items()):
        raise ValueError("qualification scope changed")


def make_receipt(root,assets_root,*,require_parent):
    cfg=c.read(root/CONFIG);validate_config(cfg)
    evidence=c.path(root,cfg["diagnostic_report"])
    if c.sha(evidence)!=cfg["diagnostic_report_sha256"]:
        raise ValueError("common-protocol diagnosis changed")
    previous=parent.make_receipt(root,assets_root,require_parent=require_parent)
    rows={t:previous["rows"][t][:2] for t in cfg["tasks"]}
    keys={v["record_id"] for pool in rows.values() for v in pool}
    sources={**previous["sources"],**{name:c.sha(root/name) for name in SOURCES+[cfg["diagnostic_report"]]}}
    merged={**cfg,**{k:previous["config"][k] for k in ("parent_run_id","parent_report","parent_report_sha256","metric")}}
    return {**previous,"config":merged,"rows":rows,"gt":{k:v for k,v in previous["gt"].items() if k in keys},
            "parent_selection_digest":previous["selection_digest"],"selection_digest":c.digest(rows),"sources":sources,
            "comparison":"explicit_one_hot_parity_and_fixed_near_one_hot_execution",
            "routes_are_manual_engineering_fixtures":True,"training_fit_only":True}


def load_receipt(root,output):
    receipt=parent.load_receipt(root,output)
    validate_config(receipt["config"])
    original=c.read(c.run_path(root,receipt["config"]["parent_run_id"])/"receipt.json")
    expected={t:original["rows"][t][:2] for t in receipt["config"]["tasks"]}
    if (receipt["rows"]!=expected or receipt["parent_selection_digest"]!=original["selection_digest"]
            or not receipt["routes_are_manual_engineering_fixtures"] or not receipt["training_fit_only"]):
        raise ValueError("qualification source subset or scope changed")
    return receipt


def aggregate(records, expected_keys):
    keys=[r["identity"]["key"] for r in records]
    if len(keys)!=8 or len(set(keys))!=8 or set(keys)!=set(expected_keys):
        raise ValueError("expected exactly eight unique image metrics")
    arms=("original","one_hot","near_one_hot")
    result={}
    for task in c.CATEGORIES:
        rows=[r for r in records if r["identity"]["task"]==task]
        if len(rows)!=2:
            raise ValueError("expected two training sources per task")
        for row in rows:
            if set(row["metrics"])!=set(arms):
                raise ValueError("metric arms changed")
            for value in row["metrics"].values():
                if set(value)!={"psnr","lpips"} or any(type(x) not in (float,int) or not math.isfinite(x) for x in value.values()):
                    raise ValueError("invalid metrics")
            if row["metrics"]["one_hot"]!=row["metrics"]["original"]:
                raise ValueError("identical uint8 output must have identical metrics")
        means={a:{m:sum(r["metrics"][a][m] for r in rows)/2 for m in ("psnr","lpips")} for a in arms}
        result[task]={"records":2,"means":means,
                      "near_one_hot_minus_original":{m:means["near_one_hot"][m]-means["original"][m] for m in ("psnr","lpips")},
                      "near_one_hot_harmed_images":sum(r["metrics"]["near_one_hot"]["psnr"]<r["metrics"]["original"]["psnr"]
                          or r["metrics"]["near_one_hot"]["lpips"]>r["metrics"]["original"]["lpips"] for r in rows)}
    return result
