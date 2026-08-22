# FoundIR-Lowlight Gate 2 photometric-stage result

Date: 2026-08-22 UTC

## Outcome

`FOUNDIR_LOWLIGHT_GATE2_PHOTOMETRIC_PASS`

Seven raw/fixed/input-only/oracle proposals were evaluated on all 245 frozen
outer-development records after fitting only on the 429 training-role records.
The run cached 1,715 saved-and-reloaded uint8 RGB PNGs. No protected FoundIR
pixel was opened or inferred upon.

| Proposal | Equal-family PSNR | Worst-family PSNR | Equal-family SSIM |
|---|---:|---:|---:|
| Raw input | 10.4902 | 7.6175 | 0.3147 |
| Fixed train-mean exposure | 13.6246 | 9.4313 | 0.4743 |
| Input-only bounded predictor | 7.6557 | 5.5939 | 0.0408 |
| O1 target-fit exposure | 17.1650 | 12.3770 | 0.5272 |
| O2 target-fit monotone tone/WB | 25.7409 | 22.8427 | 0.6792 |
| O3 O2 plus smooth illumination | 25.8192 | 22.8886 | 0.6805 |
| O4 gamma plus full matrix | 25.2953 | 22.1219 | 0.6771 |

The input-only predictor is killed: it is worse than raw input on the frozen
outer families, including `5.5939 dB` on SICE. There is no correction attempt.
The oracle ladder proves that the failure is not lack of photometric headroom.
Instead, target rendering is not identifiable or transferable from the current
global input summaries. O2 also exceeds the less constrained historical O4
ceiling, showing that a stable monotone per-channel rendering explains more of
these targets than the older gamma/full-matrix parameterization.

This is an intermediate Gate 2 result. Both frozen LPIPS paths, released
teachers, historical FLUX, and target-accessing blend ceilings remain pending
the one-shot GPU bundle before a conservative base is selected.

## Identities

- Config SHA-256: `a1fed1aadfaf69db7a02b72bf9523cd85ada7f2b65a54f989a562ece2f38fdc2`
- Runner SHA-256: `788c6ada67f8f3dac990582548bb33a4f9ba465fb9a7e9d851e71381385a1685`
- Output manifest SHA-256: `1176ad4c481b0a1a0dc5fd1974bda0b9dcd25abfb39be89cd322c211b63c2680`
- Machine result SHA-256: `a097d0267256fcbc2a4363457dd98a84d9b90befa21b83dfc3c1857fcb296909`
