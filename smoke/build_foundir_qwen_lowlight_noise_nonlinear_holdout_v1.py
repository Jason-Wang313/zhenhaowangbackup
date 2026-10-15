#!/usr/bin/env python3
"""Freeze a fresh training-only holdout for the targeted nonlinear head."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_eleven_class_gate import LABEL_BY_CATEGORY
from lowlight_restoration.qwen_five_class_gate import stable_digest
from smoke.run_foundir_qwen_five_class_gate_ddp import sha256

SOURCE=ROOT/"manifests/foundir_five_type_pairs_v1.csv"
V1=ROOT/"manifests/foundir_qwen_eleven_class_train_only_v1.csv"
V2=ROOT/"manifests/foundir_qwen_eleven_class_fresh_holdout_v2.csv"
OUTPUT=ROOT/"manifests/foundir_qwen_lowlight_noise_nonlinear_holdout_v1.csv"
SUMMARY=OUTPUT.with_suffix(".json")
SEED=20260921
PER_CLASS=120
TARGETS=("low_light","noise+low_light")


def main():
    used=set()
    for path in (V1,V2):
        with path.open(newline="") as handle: used.update(row["source_group"] for row in csv.DictReader(handle))
    pools=defaultdict(lambda:defaultdict(list))
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            category=row.get("category"); source=row.get("source_group","")
            if row.get("split")!="train" or category not in LABEL_BY_CATEGORY or source in used: continue
            label=LABEL_BY_CATEGORY[category]
            if label in TARGETS: pools[label][source].append(row)
    selected=[]; selected_sources=set()
    for label in TARGETS:
        sources=sorted(pools[label],key=lambda source:stable_digest(SEED,label,"source",source))
        if not sources: raise RuntimeError(f"no fresh sources for {label}")
        candidates=[]
        for source in sources:
            rows=sorted(pools[label][source],key=lambda row:stable_digest(SEED,label,"record",row["record_id"]))
            for index,row in enumerate(rows): candidates.append((index,stable_digest(SEED,label,"round",index,source),row))
        candidates.sort(key=lambda item:(item[0],item[1]))
        picked=[]
        for _,_,row in candidates:
            if row["source_group"] in selected_sources: continue
            picked.append(row)
            if len(picked)==PER_CLASS: break
        if len(picked)<PER_CLASS:
            # Reuse a source within this holdout only after every available source contributed once.
            already={row["record_id"] for row in picked}
            for _,_,row in candidates:
                if row["record_id"] in already or row["source_group"] in selected_sources: continue
                picked.append(row); already.add(row["record_id"])
                if len(picked)==PER_CLASS: break
        if len(picked)!=PER_CLASS: raise RuntimeError(f"insufficient fresh records for {label}: {len(picked)}")
        selected_sources.update(row["source_group"] for row in picked)
        for row in picked:
            selected.append({"record_id":row["record_id"],"label":label,"source_group":row["source_group"],"lq_path":row["lq_path"],"lq_sha256":row["lq_sha256"],"width":row["width"],"height":row["height"]})
    selected.sort(key=lambda row:(row["label"],row["record_id"]))
    if OUTPUT.exists() or SUMMARY.exists(): raise FileExistsError("refusing to replace frozen holdout")
    with OUTPUT.open("x",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(selected[0])); writer.writeheader(); writer.writerows(selected)
    source_counts={label:len({row["source_group"] for row in selected if row["label"]==label}) for label in TARGETS}
    payload={"schema_version":1,"status":"frozen_before_feature_extraction","source_manifest":SOURCE.relative_to(ROOT).as_posix(),"source_manifest_sha256":sha256(SOURCE),"excluded_partitions":{path.relative_to(ROOT).as_posix():sha256(path) for path in (V1,V2)},"manifest":OUTPUT.relative_to(ROOT).as_posix(),"manifest_sha256":sha256(OUTPUT),"records":len(selected),"records_per_class":PER_CLASS,"source_groups_per_class":source_counts,"source_disjoint_from_v1_and_v2":True,"training_only":True,"development_internal_or_official_test_used":False}
    SUMMARY.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps(payload,indent=2,sort_keys=True))


if __name__=="__main__": main()
