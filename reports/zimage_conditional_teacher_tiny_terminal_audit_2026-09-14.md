# Conditional Z-Image teacher tiny stage: terminal audit

DIAGNOSTIC ONLY / 非主线. Fixed training-archive crops, not official FoundIR
test results. Audited at 2026-09-14T12:42:25Z.

The job finished all 1200 steps normally on 8x H100, from 12:22:36Z to
12:36:45Z (status 0). Training-loop elapsed time through step1200 was
672.7192 seconds, including its intermediate snapshot. Launch revision:
`e2d2c3cf7da23168cad61f10a540882be06e2874`.

The model is pinned Z-Image Base 6B with fresh FP32 rank-32 LoRA and a
zero-initialized aligned LQ projection, native Gaussian-noise-to-GT flow,
50-step shift-6 inference, CFG1, original frozen VAE and native text encoder.
Config: `configs/foundir_blur_only_zimage_conditional_teacher_v1.json`, SHA256
`c8bbf5d7dc85c2436f554a7e01e38b9d44d3f554686e001566c248d278da94db`.

## Quality: FAIL

| Split | Input PSNR / LPIPS | Step 0 | Step 400 | Step 1200 |
|---|---:|---:|---:|---:|
| 16 training crops | 24.7232 / .3085 | 8.1380 / .8391 | 18.1619 / .5004 | 21.6206 / .2783 |
| 16 development crops | 23.9045 / .3264 | 7.6022 / .8331 | 17.1693 / .5604 | 19.9942 / .4420 |

Final training PSNR loses 3.1026 dB against input. LPIPS improves .0302423,
just meeting its >.03 criterion, but the PSNR gain >1dB criterion fails.
The fixed training probe decreases .4649051 -> .3990823, a 14.1583% reduction;
final/initial=.8584168, failing the predeclared <.8 ratio. Two of three train
checks fail. Only 2/16 training images improve PSNR and 7/16 improve LPIPS.
Development loses 3.9104dB and worsens LPIPS by .1156743; 0/16 improves PSNR.
Development was descriptive for this tiny stage, so it is NOT the reason
the learnability admission gate fails. The training criteria already fail.

Numerical proxy comparison only: development fails both Stage-1 Blur targets
(20.7/.21), missing PSNR by .7058dB and exceeding LPIPS by .2320. It fails
both Stage-2 targets (25.9/.18), missing PSNR by 5.9058dB and exceeding LPIPS
by .2620. These 16 crops cannot establish contractual acceptance.

Visual review of the final fixed triptychs also FAILS:

- Train 0042483: coarse statue geometry sharpens, but its blue pattern is
  smeared/replaced and surface details differ from GT.
- Train 0091486: near-clean input is severely changed, with repainted grass,
  rocks and foreground stems; PSNR 36.8832 ->22.1782 and LPIPS .04686 ->.33958.
- Development 0055608: bus/front details and structural boundaries are
  replaced by distorted, painted-looking blocks; sharpness is not fidelity.
- Development 0092666: leaf positions/contours and texture are altered;
  PSNR 30.0970 ->21.9946 and LPIPS .15234 ->.35312.

The four reviewed PNGs reside at
`checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_tiny/step_001200/`
under their train/development roles. No passing visual-review artifact exists.

## Execution and artifact audit: PASS

Independently verified config/result/checkpoint hashes, 1200 ordered finite
training rows covering the expected 16 record IDs, two FP32 checkpoints with
477 finite tensors each, six recomputed aggregates, and 96 record triptychs
with matching sidecar image hashes and config identity. All eight CUDA
rehearsals passed exact native forward/LoRA-gradient equivalence, repeated
updates, synchronized post-clipping gradients/weights, and initial reset.
Condition-sensitivity maxima after rehearsal updates range .078125--.171875.
This establishes a functioning tested input path, not sufficient learned
restoration conditioning. Maximum logged preclip gradient was 10.38948 at
step29 during warmup, clipped at norm1. No nonfinite/runtime failure occurred.

- Result: `checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_tiny/result.json`,
  SHA256 `ed945a012bd39ac6d08cc3d08f5f9ce0ce4e662f229405ecc21fe06d4281359d`.
- Step400: `checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_tiny/step_000400.safetensors`,
  SHA256 `3e5f6a7091e991f36bf808262a2478d37c3b04460f4a5d54cce282d5ef15e62d`.
- Step1200: `checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_tiny/step_001200.safetensors`,
  SHA256 `52a61cd9412f494c2141d5f34ffd9cbb2dc66911755e008f2828a63071ea93a7`.
- Launcher governance report: `zimage_conditional_teacher_tiny-00140bbcd7f61c87`.

All protected-test, mixed-degradation, Qwen degradation-conditioning,
development-GT-optimization and decoder-tuning flags are false.

## Interpretation and next boundary

This recipe has learned substantial scene information from LQ: both train and
development improve from initialization and from step400 to1200. The sampled
curve therefore does not demonstrate late overfitting, and the result must
not be described as an optimizer crash or complete inability to learn.
It also does not establish that longer training, more data or another
conditioning architecture will succeed. Native semantics and a live LQ path
alone were insufficient for fidelity under this frozen budget. A unique cause
is unresolved; weak per-step conditional reconstruction and accumulated
sampling error are distinguishable hypotheses, not established diagnoses.

The prepared 2048/128 pilot is not admitted. Do not create a passing tiny
review, automatically launch pilot/full training, distill this checkpoint,
or tune the decoder from this failed gate. The next scientific decision
should use this failed training-learnability evidence and the earlier healthy
VAE/oracle evidence to identify where conditioning fails to preserve content,
before choosing a further experiment. No new advisor approval is requested.
This check turn performed an audit and updated handoff; it changed no training
code, model, loss, schedule, gate threshold or GPU job.
