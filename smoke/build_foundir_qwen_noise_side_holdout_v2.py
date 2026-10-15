#!/usr/bin/env python3
"""Freeze a second fresh source-disjoint holdout for the Noise-side guard."""
from __future__ import annotations
import csv,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lowlight_restoration.qwen_eleven_class_gate import LABEL_BY_CATEGORY
from lowlight_restoration.qwen_five_class_gate import stable_digest
from smoke.run_foundir_qwen_five_class_gate_ddp import sha256
SOURCE=ROOT/"manifests/foundir_five_type_pairs_v1.csv";EXCLUDED=[ROOT/"manifests/foundir_qwen_eleven_class_train_only_v1.csv",ROOT/"manifests/foundir_qwen_eleven_class_fresh_holdout_v2.csv",ROOT/"manifests/foundir_qwen_lowlight_noise_nonlinear_holdout_v1.csv"];OUTPUT=ROOT/"manifests/foundir_qwen_noise_side_holdout_v2.csv";SUMMARY=OUTPUT.with_suffix(".json");TARGETS=("noise","noise+low_light");QUOTA=96;SEED=20260922
def main():
 used=set()
 for p in EXCLUDED:
  with p.open(newline="") as h:used.update(r["source_group"] for r in csv.DictReader(h))
 pools=defaultdict(dict)
 with SOURCE.open(newline="") as h:
  for r in csv.DictReader(h):
   if r.get("split")!="train" or r.get("category") not in LABEL_BY_CATEGORY or r["source_group"] in used:continue
   label=LABEL_BY_CATEGORY[r["category"]]
   if label not in TARGETS:continue
   old=pools[label].get(r["source_group"])
   if old is None or stable_digest(SEED,r["record_id"])<stable_digest(SEED,old["record_id"]):pools[label][r["source_group"]]=r
 selected=[];holdout_sources=set()
 for label in TARGETS:
  rows=sorted(pools[label].values(),key=lambda r:stable_digest(SEED,label,r["source_group"]));rows=[r for r in rows if r["source_group"] not in holdout_sources][:QUOTA]
  if len(rows)!=QUOTA:raise RuntimeError(f"insufficient {label} sources: {len(rows)}")
  holdout_sources.update(r["source_group"] for r in rows)
  for r in rows:selected.append({"record_id":r["record_id"],"label":label,"source_group":r["source_group"],"lq_path":r["lq_path"],"lq_sha256":r["lq_sha256"],"width":r["width"],"height":r["height"]})
 selected.sort(key=lambda r:(r["label"],r["record_id"]))
 if OUTPUT.exists() or SUMMARY.exists():raise FileExistsError("refusing overwrite")
 with OUTPUT.open("x",newline="") as h:w=csv.DictWriter(h,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
 out={"schema_version":1,"status":"frozen_before_feature_extraction","manifest":OUTPUT.relative_to(ROOT).as_posix(),"manifest_sha256":sha256(OUTPUT),"source_manifest":SOURCE.relative_to(ROOT).as_posix(),"source_manifest_sha256":sha256(SOURCE),"excluded_partitions":{p.relative_to(ROOT).as_posix():sha256(p) for p in EXCLUDED},"records":192,"records_per_class":96,"source_groups_per_class":{"noise":96,"noise+low_light":96},"all_holdout_sources_unique":True,"training_only":True,"development_internal_or_official_test_used":False};SUMMARY.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__":main()
