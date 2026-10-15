# FoundIR stage-trajectory compatibility v1 — GPU readiness

Date: 2026-09-27 UTC
Status: first GPU launch failed before step 1; corrected retry is ready

> **DIAGNOSTIC ONLY / 非主线。** This tiny gate cannot replace the terminal official SOW600 result and cannot establish paper novelty by itself.

## Hypothesis and stop rule

A 257,235-parameter, identity-initialized bounded residual adapter receives the degraded input plus frozen Noise, Low-light, and Blur stage states. It is trained and selected only on source-disjoint `Low-light+Noise` and `Low-light+Blur` trajectories. No triple record enters gradients or checkpoint selection.

The launcher stops before the triple endpoint unless the selected checkpoint improves PSNR over its frozen route output in **both** double validation categories. If that gate passes, it performs exactly one frozen full-resolution evaluation on the existing 60-image `Low-light+Blur+Noise` validation endpoint. It never accesses the official FoundIR sealed test.

## Frozen data

| Split | Low-light+Blur | Low-light+Noise | Source groups |
|---|---:|---:|---:|
| Train | 49 | 40 | 51 total |
| Validation/selection | 13 | 6 | 13 total |

- All 221 source groups represented by the 660-image endpoint are excluded from train and validation.
- Triple records in gradients or checkpoint selection: `0`.
- Manifest: `manifests/foundir_stage_trajectory_compatibility_v1.csv`, SHA-256 `120b45c24fb92f7222adee1b0dcbf93c02234f883f1b283fe1fe03263b425397`.
- Config: `configs/foundir_stage_trajectory_compatibility_v1.json`, SHA-256 `4413728d0d4ffd10d9657e2bb9c4c3be266a36d58594c2524c0e09aab9236d26`.

This is deliberately a falsification pilot. The available source-isolated stage cache is only 108 records; even a positive result must later be confirmed on a larger frozen split.

## Implementation and verification

- Model: `lowlight_restoration/stage_trajectory_compatibility.py`, SHA-256 `1eb97f35ee3d8ff7262ebecc60703336326d9de6b39ec267113a074dbdb0d695`.
- Trainer: `smoke/train_foundir_stage_trajectory_compatibility_ddp.py`, SHA-256 `668a7807ccad385d31edc604e4a1b58a2a7966b105a93dc2c132d1a1a31ca769`.
- Endpoint evaluator: `smoke/evaluate_foundir_stage_trajectory_compatibility_endpoint_ddp.py`, SHA-256 `0e4cd67e971a8608cab71041f56fb98ecc85da6497bfed460a72826449a09462`.
- NCCL/DDP no-update rehearsal: `smoke/diagnose_foundir_stage_trajectory_nccl_ddp.py`, SHA-256 `715cc6dd24ae8f78add30a41fc2ee6b934b8baf48f56ff5d978ac4998e82454a`.
- Launcher: `smoke/run_foundir_stage_trajectory_compatibility_v1_8gpu.sh`, SHA-256 `cacbbab6020c45c54fd27084f1fca4ac3d0fa31d60e76ec12bbb176bfcbfed96`.
- Exact identity initialization: pass.
- Finite-gradient rehearsal: pass.
- Unit tests: `2 passed`.
- Python compilation, shell syntax, manifest hashes, image hashes/shapes, source isolation, config validation, governance binding, and `git diff --check`: pass.
- Governance hold: `foundir_stage_trajectory_compatibility_v1_design-e435439b0673fefc`, resolved by the advisor's explicit `execute` instruction for this documented scope.

## First-launch runtime failure and correction

The first external 8×H200 launch started at `2026-09-27T09:18:29Z` and failed during `dist.init_process_group()` at approximately `09:19:55Z`, before cache construction, forward/backward, optimizer step 1, checkpoint creation, or triple access. All ranks reported NCCL/CUDA error 401 (`operation cannot be performed in the present state`). No process remains and no terminal marker exists.

This matches the previously verified H200 NVLS initialization failure in this workspace. The launcher now exports `NCCL_NVLS_ENABLE=0`, explicitly keeps P2P enabled, and runs a bounded eight-rank scalar-all-reduce plus full adapter DDP forward/backward rehearsal with zero optimizer steps before formal training. The same settings remain in the formal process environment. This changes no data, model, loss, batch, steps, selection, or endpoint protocol.

## Runtime estimate and exact launch

One homogeneous 8-GPU H100/H200 node is sufficient. More nodes would add orchestration overhead and do not help this tiny DDP workload.

- H200 estimate: `0.20–0.40 hours` total wall time, approximately `1.6–3.2 GPU-hours`.
- H100 estimate: `0.25–0.50 hours` total wall time, approximately `2.0–4.0 GPU-hours`.
- These are pre-run estimates; the first cache/25-step timing will replace them with measured projection.

Retry exactly:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/run_foundir_stage_trajectory_compatibility_v1_8gpu.sh
```

Do not launch a duplicate. After launch, monitor `logs/foundir_stage_trajectory_compatibility_v1.log` and the terminal marker `logs/foundir_stage_trajectory_compatibility_v1.end`.
