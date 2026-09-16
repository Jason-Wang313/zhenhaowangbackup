# Z-Image corrective bridge v2 tiny result — 2026-09-16

## Decision

The frozen 1,000-step Blur-only tiny protocol completed normally on 8xH100,
but failed its scientific quality gate. This is a runtime success and a
teacher-quality failure. Do not full-scale, distill, or decoder-tune this
checkpoint.

## Frozen scope

- Pure Z-Image Base 6B Layer-1 corrective bridge teacher.
- Fixed 16 train and 16 source-disjoint development images.
- Analytic/on-policy state mixture; eight-step solver-aligned training and
  inference; corrective target `v=(state-GT)/sigma`.
- Endpoint latent/RGB/edge/LPIPS losses, near-clean identity loss, and internal
  spatial residual gate.
- No independent restoration branch, distillation, decoder tuning, Qwen
  degradation conditioning, mixed degradation, development-GT optimization,
  sealed internal access, or official FoundIR test access.

Config SHA-256:
`64c1b53518e3b429e11c33f62d84097375a1220b0ea0cf90703cee461d184ddf`.

## Metrics

| Step | Split | Input PSNR | Output PSNR | Input LPIPS | Output LPIPS |
|---:|---|---:|---:|---:|---:|
| 0 | train | 24.723185 | 24.642070 | 0.308547 | 0.311363 |
| 250 | train | 24.723185 | 24.171788 | 0.308547 | 0.306431 |
| 500 | train | 24.723185 | 24.212626 | 0.308547 | 0.309294 |
| 1000 | train | 24.723185 | 25.199975 | 0.308547 | 0.283527 |
| 0 | development | 23.904541 | 23.890001 | 0.326357 | 0.330076 |
| 250 | development | 23.904541 | 23.497512 | 0.326357 | 0.343599 |
| 500 | development | 23.904541 | 23.471814 | 0.326357 | 0.349269 |
| 1000 | development | 23.904541 | 23.125971 | 0.326357 | 0.342116 |

The required train gate was PSNR >=26.4, LPIPS <=0.16, and both better than
input. Final train misses it by 1.200025 dB and 0.123527 LPIPS. Only 10/16
train images improve PSNR and 8/16 improve LPIPS. Final development is worse
than input by 0.778570 dB and 0.015760 LPIPS; only 4/16 improve PSNR.

V2 also underperforms the v1 final train result (25.807990/0.294316) in PSNR,
although its LPIPS is 0.010789 lower. V1 development was 22.186115/0.343677;
v2 development is less bad but still materially below the unchanged input.

## Visual review

Representative final train and development triptychs were inspected, including
hard-blur and near-clean records. Outputs introduce conspicuous granular or
checker texture, edge ringing/echoes, and local structural corruption. The
failure is not hidden by aggregate averaging:

- development `foundir_train_0100870`: -3.328337 dB and +0.063953 LPIPS;
- development `foundir_train_0062371`: -2.437623 dB and +0.132400 LPIPS;
- development `foundir_train_0055120`: -2.338318 dB despite a 0.047543 LPIPS
  decrease; the output has severe false texture and distorted fence detail;
- train near-clean `foundir_train_0091486`: -3.771526 dB and +0.029995 LPIPS.

Near-clean identity protection therefore did not prevent destructive changes,
and the teacher did not robustly learn the 16 training images.

## Artifact verification

Every step/split directory at steps 0/250/500/1000 contains exactly 16 PNG
triptychs, 16 per-image JSON sidecars, and one aggregate `metrics.json`: 128
PNGs and 128 sidecars total. All metrics are finite. The result status is
`zimage_corrective_bridge_tiny_complete`; terminal launcher status is 0.

Checkpoint SHA-256 values:

- step 250: `4f9a3bb62bb1546216184d2460d2a965fb44929538d14c24c975bf04ed47ea70`
- step 500: `e529fcea183b4ab18555216b7fae2d0d3e8154eb2b61f62e037bd0fe77684f15`
- step 1000: `3d8c06b1e9dd2008587bacb145fbf6d827fb45bc60be3c6b91b5b8addbf63be7`

Result SHA-256:
`9d33c238eb3bb44adee15ed8dc6845dbddc665a7b811c6d6aae07a18f99bfcc1`.

## Conclusion

The off-trajectory/solver-alignment repair improves the late train result over
its own step-250/500 states, but does not solve the core restoration problem.
More steps or more images are not evidence-backed next actions: the teacher
cannot reliably overfit the fixed 16-image gate and harms unseen images. A
student is bounded by this failed teacher, while decoder tuning cannot recover
missing or incorrectly predicted structure. Any next experiment must change
the upstream restoration capacity/formulation; it must not scale or distill
this checkpoint.
