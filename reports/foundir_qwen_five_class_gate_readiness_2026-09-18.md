# FoundIR Qwen five-class singleton gate readiness

Status: CPU preflight passed; GPU feature extraction has not started.

## Scope

- Classes: Blur, Noise, Haze, Rain, Low-light.
- Input data: isolated single-degradation records from the FoundIR training split only.
- Frozen representation: the existing project Qwen physical-v8 step-2500 adapter.
- New trainable component: one standardized linear five-way softmax head.
- Low-confidence predictions abstain and therefore never enter the deterministic expert router.
- Development and internal-test records are excluded from fitting, selection, and the terminal training-domain holdout.

## Frozen partition

- Fit: 450 records per class (2,250 total).
- Confidence calibration: 90 records per class (450 total).
- Terminal training-only holdout: 90 records per class (450 total).
- Each record represents a different `(class, source_group)` pair.
- A global source hash assigns each source to exactly one of the three splits, including sources shared by multiple degradation categories.
- Manifest: `manifests/foundir_qwen_five_class_train_only_v1.csv`.
- Manifest SHA-256: `af8798fab6ea340ba4589684ac02997b0cb01fe88a56a78bafcd92e75f36fc6c`.

## Gates frozen before GPU work

- Overall wrong-expert rate: at most 0.5%.
- Wrong-expert rate for every class: at most 1.0%.
- Overall non-abstained coverage: at least 65%.
- Non-abstained coverage for every class: at least 50%.
- Priority: minimize wrong expert routing before maximizing coverage.
- The 450-record holdout is accessed once after classifier and threshold selection on the separate calibration split.

## Verification

- New gate plus deterministic router tests: 36/36 passed.
- Qwen runtime preflight: passed for all 3,150 records.
- No GPU was used during preparation.
- No FoundIR development or internal-test record was used.

## GPU launch

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash smoke/launch_foundir_qwen_five_class_gate_v1_8gpu.sh
```

Expected complete wall time on one 8xH100 instance: 25--50 minutes, with a center estimate of 35 minutes. This is 3.3--6.7 GPU-hours, center 4.7 GPU-hours. The main uncertainty is shared-filesystem image/feature I/O and CPU logistic-regression convergence after feature extraction.
