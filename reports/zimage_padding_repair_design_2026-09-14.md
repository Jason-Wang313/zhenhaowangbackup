# Deterministic padding compatibility repair — round-1 supplement

DIAGNOSTIC ONLY / 非主线. User explicitly said `execute` after the consolidated
batch's deterministic CUDA failure. This is an implementation repair inside
round 1, not a new loss/LR sweep, model switch or round-2 quality gate.

## Evidence and bounded change

Original batch result SHA-256:
`97c3d855f447d3e24667401466cc1bff985efe1ad4304f954ac171e78474ca23`.
Ordinary workers 0/1/4/5 completed all 12 fits. Strict workers 2/3/6/7 failed
before optimizer step 1 in boolean-index broadcast assignment of text padding.
CUDA Indexing.cu:699 reports flattened sizes 115200 vs 14745600. This is a
runtime incompatibility in the current PyTorch execution path, not evidence
of restoration convergence failure or a proven FlashAttention-backward bug.

The pinned Base pipeline has two equivalent padding assignments (image/text).
Replace only those with `torch.where(mask[:, None], frozen_token, features)`.
Unmasked feature values/gradients pass through; masked values equal the pad
token and masked feature gradients are zero. Padding tokens remain frozen.
No timestep, attention, normalization, loss, seed, precision, optimizer,
schedule, training set or backbone change is introduced.

The repair is process-local and opt-in. A tracked compatibility module verifies
the entire upstream pipeline SHA-256
`ace84d674ea8356ffc0ebfeafefc4ceadba40590f52908071adb1e2e3a98a5e2`,
requires exactly one occurrence of each of the two assignments, and compiles
that already-imported pinned function with ONLY these two substitutions.
The ignored upstream checkout is not edited. The transformed function hash
is recorded. Unexpected source, shapes, token training or dtypes fail closed.

## Mandatory runtime checks BEFORE optimizer step 1

1. On each selected CUDA device, compare original ordinary-mode padding versus
   strict-deterministic replacement. Require EXACT output and feature-gradient
   equality for width 3840, FP32/BF16 and empty/partial/all padding masks.
   This catches CUDA-only failures before loading the 6B model.
2. On the actual fixed 512x512 crop, run full model/loss/backward four times:
   original with strict PyTorch flag off; repaired with flag off; repaired
   strict twice. FlashAttention deterministic backward and cuDNN deterministic
   remain enabled in ALL four. Require exact output and ALL trainable-gradient
   hashes across these probes. No optimizer update is performed. Preserve
   failures and do not silently relax exactness or disable determinism.
3. Only if both gates pass, restore initial LoRA/optimizer/RNG and run the same
   off_a/on/off_b 200-step controls. Phases remain on the same physical GPU.

CPU tests and reduced real-model tests cannot establish CUDA success. The
single GPU command automatically performs these checks before training; do
not describe the repair as GPU-validated before their artifacts are observed.

## Execution and reuse

Only original rank/device IDs 2/3/6/7 are launched: 12 new fits. Other four
GPUs do not retrain already-completed ordinary controls. This preserves the
same-GPU repetition design; distributing phases onto idle GPUs would defeat
that control. Parent config/result/checkpoint/log/image hashes are validated
before reuse. Parent ordinary results retain their ORIGINAL config identities
and paths in the merged report and are explicitly marked reused. No copying,
overwriting, relabeling or deletion of historical outputs is permitted.

New config: `configs/foundir_blur_only_zimage_repro_repair_v1.json`.
New output: `checkpoints/foundir_blur_only_zimage_repro_repair_v1/`.
Launcher: `smoke/launch_foundir_blur_only_zimage_repro_repair_v1_8gpu.sh`.
It requires an eight-H100/H200 node inventory, but trains only four independent
workers. The final report combines 12 reused and up to 12 new fits. Each new
worker must record its CUDA operator/model repair gates and original source
hash; a merged report alone never proves quality or admits full training.

Cross-execution-bundle comparisons now include the verified padding repair;
do not claim they isolate one CUDA kernel causally. Same-GPU strict repeats
and instrumentation checks remain the primary reproducibility control.
After these complete, inspect fixed metrics/visuals and freeze round 2 using
evidence. No advisor re-request, Qwen degradation conditioning, mixed
degradation, development-GT optimization, sealed internal or official FoundIR
test access. The overall teacher/student restoration objective remains open.
