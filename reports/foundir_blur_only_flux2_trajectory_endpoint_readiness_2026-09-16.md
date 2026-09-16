# FLUX.2 Blur-only trajectory-endpoint gate readiness — 2026-09-16

**DIAGNOSTIC ONLY / 非主线.** This bounded gate does not claim a SOW result,
does not access a protected evaluation split, and cannot launch full scale.

## Decision and purpose

The prior step-10k random-t predicted-clean RGB gate improved tiny-train PSNR
but damaged LPIPS and source-disjoint development. The user explicitly
authorized the materially different next gate with `execute` and `continue`.
The final hash-bound resolved governance event is
`flux2_blur_trajectory_endpoint_design_v1_final-6dab2f3e67857f10`; it records USER
authorization and does not fabricate an advisor response.

This gate supervises the endpoint that is actually deployed. It retains the
original random-t flow MSE, recomputes the official 50-step shifted CFG
trajectory on every candidate update, detaches only after transition 46, and
backpropagates through the last four transitions with non-reentrant activation
checkpointing. The final tokens are decoded by the frozen VAE and receive RGB
MSE weight `1.0` plus LPIPS weight `0.02`. It is not a four-step surrogate.

## Frozen protocol

- parent: `checkpoints/foundir_blur_only_flux2_full_v1/step_010000.safetensors`;
- parent SHA-256:
  `e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`;
- config: `configs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1.json`;
- config SHA-256:
  `d1824e219a5f46629e6d8cc3e964e5b676512e2bb8037c41c5bf88a01f9f7b23`;
- arms: `flow_control` and `trajectory_endpoint`, both with independently fresh
  AdamW and identical sample order, random-t flow noise, parent, and global
  batch;
- training: 200 optimizer steps per arm, checkpoints 50/100/200, global batch
  8 on homogeneous 8xH100 or 8xH200;
- evidence: 16 fixed train crops and 16 source-group-disjoint descriptive
  development crops, with fixed `Input | Output | Ground Truth` snapshots;
- development GT cannot select an arm, loss weight, or checkpoint;
- predeclared candidate is trajectory step 200; pass requires at least
  `+0.15 dB` over control and parent, LPIPS regression no more than `0.003`
  against both, PSNR wins on at least 10/16 train records, then human visual
  review.

## Verification

- governance check passes against final hold SHA
  `6cbfdf2c063ddb871361bc0f8945a93d359d620a5d8f386a9510aad5e69a5489`;
- 20 focused tests pass, including exact equality with official
  `denoise_cfg()` when gradients are disabled, exact 46+4 transition coverage,
  detached prefix, suffix model gradients, checkpointed/non-checkpointed
  output and gradient agreement, scope locks, source disjointness, parent
  identity, and H100/H200 global-batch equivalence;
- Python compilation, shell syntax, complete parent/model/manifest hashes,
  prior VAE evidence, and launcher `--check` pass;
- a real FLUX.2 4B CPU rehearsal loaded all 50 LoRA modules from the exact
  parent, ran a reduced 2-detached + 2-checkpointed-suffix trajectory, and
  backpropagated finite loss `1.0449200868606567`; all 100 trainable LoRA
  tensors received finite nonzero gradients and the output shape was
  `[1,16,128]`;
- the GPU runner performs a full-resolution 50-step forward/backward rehearsal
  before optimizer step 1, records peak memory, requires nonzero LoRA
  gradients, and rejects any frozen-VAE parameter gradient;
- Qwen degradation conditioning, mixed degradation, pixel refiner, decoder
  tuning, distillation, development-GT optimization, sealed internal, and the
  official FoundIR test all remain disabled.

## Launch

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
smoke/launch_foundir_blur_only_flux2_trajectory_endpoint_v1_8gpu.sh
```

Monitor:

```bash
tail -f logs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1.log
```

The candidate recomputes 50 model calls per update and backpropagates through
four, so the first full-resolution rehearsal is the memory gate. A rough
8xH100/H200 wall-time estimate is 35--70 minutes, but runtime must be inferred
from the live log rather than treated as guaranteed.
