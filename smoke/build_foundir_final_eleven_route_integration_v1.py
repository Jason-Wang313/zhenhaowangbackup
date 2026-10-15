#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DECISIONS=ROOT/'checkpoints/foundir_final_eleven_route_cpu_gate_v1/development_decisions.csv'
SOURCE=ROOT/'manifests/foundir_five_type_pairs_v1.csv'
OUTPUT=ROOT/'manifests/foundir_final_eleven_route_integration_v1.csv'
DECISIONS_SHA256='0342bf00bef98eccb9c3d3ad8e3c42c2368db2761f3c2ddd38f9b1342ad31225'
SOURCE_SHA256='e8c018879029c0bcc2da8d074cc4999541158ca97d07f48aadbcdddcdccc3ca7'
ROUTE_ORDERS={
 'blur':['blur'],'noise':['noise'],'haze':['haze'],'rain':['rain'],'low_light':['low_light'],
 'blur+noise':['blur_noise'],'haze+low_light':['haze','low_light'],
 'rain+haze':['rain','haze'],'blur+low_light':['low_light','blur'],
 'noise+low_light':['noise','low_light'],'blur+noise+low_light':['noise','low_light','blur']}
CATEGORIES={
 'blur':'01Blur','blur+noise':'02Blur_Noise','noise':'05Noise','haze':'08Haze',
 'haze+low_light':'09Lowlight_Haze','rain':'10Rain','rain+haze':'13Rain_Haze',
 'low_light':'14Lowlight','blur+low_light':'15Lowlight_Blur',
 'noise+low_light':'16Lowlight_Noise','blur+noise+low_light':'18Lowlight_Blur_Noise'}

def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()

def main():
 if sha(DECISIONS)!=DECISIONS_SHA256 or sha(SOURCE)!=SOURCE_SHA256:
  raise RuntimeError('frozen input identity changed')
 with DECISIONS.open(newline='') as f: decisions=list(csv.DictReader(f))
 with SOURCE.open(newline='') as f: source={r['record_id']:r for r in csv.DictReader(f)}
 selected=[]
 for route in ROUTE_ORDERS:
  seen=set()
  for d in decisions:
   if d['truth']!=route or d['status']!='correct':continue
   r=source[d['record_id']]
   if r['split']!='validation' or r['category']!=CATEGORIES[route]:
    raise RuntimeError(f"development row contract changed: {d['record_id']}")
   if min(int(r['width']),int(r['height']))<512:continue
   if r['source_group'] in seen:continue
   seen.add(r['source_group']);selected.append({**r,'router_truth':d['truth'],'final_route':d['final_route'],'expert_order_json':json.dumps(ROUTE_ORDERS[route],separators=(',',':')),'routing_status':d['status']})
   if len(seen)==16:break
  if len(seen)!=16:raise RuntimeError(f'not enough source groups for {route}: {len(seen)}')
 selected.sort(key=lambda r:r['record_id'])
 OUTPUT.parent.mkdir(parents=True,exist_ok=True)
 with OUTPUT.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
 meta={'schema_version':1,'status':'frozen_final_eleven_route_integration_development_v1','records':len(selected),'records_by_category':dict(sorted(Counter(r['category'] for r in selected).items())),'records_by_route':dict(sorted(Counter(r['final_route'] for r in selected).items())),'source_groups':len({r['source_group'] for r in selected}),'source_manifest_sha256':SOURCE_SHA256,'routing_decisions_sha256':DECISIONS_SHA256,'manifest_sha256':sha(OUTPUT),'official_or_sealed_test_accessed':False}
 OUTPUT.with_suffix('.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');print(json.dumps(meta,indent=2,sort_keys=True))
if __name__=='__main__':main()
