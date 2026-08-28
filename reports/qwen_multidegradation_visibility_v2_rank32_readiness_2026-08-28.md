# Qwen multi-degradation visibility-v2 rank-32 readiness — 2026-08-28

## Decision

The advisor-approved five-query Qwen2.5-VL dialogue route is ready for one
two- or eight-H100 launch. The official FoundIR test and the protected
synthetic fixed test have not been accessed by this correction.

## Corrected cause

The frozen causal audit showed that sequential composition suppressed earlier
degradation evidence. The authoritative normalized results were a first/last
recall ratio of `0.6723`, first/last median normalized survival of `0.4442`,
and signal-to-detection AUC of `0.7928`.

The new generator applies `haze -> blur -> rain -> low_light -> noise`. Every
active component must independently reach `0.85 *` the training-only isolated
q10 anchor and retain at least `20%` of its isolated signal in the final
composite. Bounded masking-operator attenuation is allowed only after attempt
256, and the same anchor and retention gates still apply.

## Frozen data

- manifest: `manifests/qwen_multidegradation_pilot_v2.jsonl`
- manifest SHA-256:
  `0ec9b367d3b795b01f8a2185b0ff6bfda8a8d21b29926929371bac4bf013ce7e`
- generation result SHA-256:
  `a485161a133c6516afd3a86ce6c6b9df94e914dd2ceeab875f1dd5071da0629a`
- 12,000 records and PNGs: 9,600 train, 1,200 validation, 1,200 protected
  fixed test
- exact clean/single/double/triple, combination, and L2--L5 balance
- 1,977 used sources with source-disjoint train/validation/test roles
- 21,600 active-component evidence rows passed
- mean/p95/maximum screening attempts: `3.9289 / 7 / 549`
- every generated PNG was checked against its frozen visibility hash

## Training-only calibration partition

- manifest:
  `manifests/qwen_multidegradation_visibility_v2_calibration.json`
- SHA-256:
  `2b97f62f9a8d0e24f770549feaede16284d49899f0f3b08b20fb27fccbfd229c`
- 1,183 records from 198 complete training sources are reserved for
  calibration and receive no gradient updates
- 8,417 gradient-eligible records remain; a deterministic 8,400 are used per
  epoch to preserve complete five-query global batches
- every degradation/L1--L5 calibration cell contains at least 60 records
- validation and test were not used to form or fit this partition

## Frozen model and decision rule

- config: `configs/qwen_multidegradation_visibility_v2_rank32.json`
- config SHA-256:
  `ec0526f35795fe4c8c8f4a1acb6b78abdb9cd194fe4d96b342bbaee931f79ae3`
- fresh Qwen2.5-VL-7B base initialization
- rank-32 LoRA on all 32 visual blocks and decoder attention, plus trainable
  visual merger; no classification head
- five separate questions per image for blur, noise, rain, haze, and low-light
- five epochs, 210,000 dialogue presentations, 1,050 optimizer steps, fixed
  global batch 200
- assistant-token cross entropy with weight 8 on the exact decision tokens
  `1..5` (token IDs `16..20`)
- inference scores those same five decoder tokens at their causal answer
  position
- a bounded per-type temperature and five-class bias are fitted only on the
  reserved training sources; the resulting answer is serialized back through
  the canonical Qwen dialogue/JSON interface

## Frozen validation gate

Validation is evaluated exactly once after the epoch-5 checkpoint and
training-only calibration are frozen. Required triple degradation-set exact
recognition is `>= 0.80`. The earlier structure, macro/micro-F1, overall set
exact, combined type/level exact, active severity, MAE, per-query accuracy,
and gradient-scope safeguards remain active. A failed validation exits before
selection or protected fixed-test access.

## CPU verification

- full Qwen inventory hashes passed
- all 12,000 generated-image hashes passed
- exact L1--L5 single-token identities and assistant decision positions passed
- rank-32 meta-architecture passed: 272 LoRA modules and 87,062,272 trainable
  parameters including the merger
- Python compilation, shell syntax, JSON parsing, and `git diff --check` passed
- 40 focused Qwen/launcher/calibration tests passed
- repository-wide suite: 252 tests passed; one unrelated pre-existing
  conditioned-FLUX authorization test failed because its frozen
  `lowlight_restoration/lora.py` hash is stale relative to the unchanged HEAD
  file. This Qwen work does not modify that file or authorization contract.

## GPU launch

The checked launcher accepts exactly `CUDA_VISIBLE_DEVICES=0,1` or
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, keeps global batch 200, records all
selected-GPU telemetry, prevents duplicate runs, and uses tmux when available
or detached nohup otherwise.
