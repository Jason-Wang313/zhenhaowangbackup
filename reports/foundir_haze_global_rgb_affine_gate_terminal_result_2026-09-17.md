# FLUX.2 Haze global RGB-affine gate — terminal result

## Decision

**FAIL — do not proceed to the 1,580-record Haze full-development run.**

This was the final simple calibration-only photometric attempt. One global
per-channel affine map was fitted by ordinary least squares using only the 16
calibration triptychs, frozen, and then applied once to the 16
source-group-disjoint validation triptychs. It did not train or change FLUX,
use validation GT for fitting, access sealed/internal or official FoundIR test
data, or use a GPU.

## Frozen method

The candidate is:

`candidate_rgb = clip(scale_rgb * flux_rgb + bias_rgb, 0, 1)`

Fitted and bounded parameters:

- scale RGB: `[.95746851, .97136094, .98177325]`
- bias RGB: `[.03818231, .03508838, .02740940]`
- prospective bounds: scale `[.8, 1.2]`, normalized bias `[-.1, .1]`

Implementation:
`smoke/audit_foundir_haze_global_rgb_affine.py`.

## Current Haze result

| Split / arm | Images | PSNR | LPIPS |
|---|---:|---:|---:|
| Calibration base FLUX | 16 | 29.103513 | .049687 |
| Calibration affine | 16 | 30.334390 | .049665 |
| Validation base FLUX | 16 | 19.956481 | .138165 |
| Validation affine | 16 | **20.070569** | **.140397** |

Against Haze Stage-2 SOW (`PSNR >=22.1`, `LPIPS <=.19`):

- PSNR margin: `-2.029431 dB`, or **-9.1829%** relative to the floor — fail;
- LPIPS safety margin: `.049603`, or **26.1070%** relative to the cap — pass.

Relative to paired base FLUX, the candidate gains only `.114089 dB` PSNR and
worsens LPIPS by `.002231`. It wins PSNR and LPIPS on 9/16 records each. The
worst PSNR regression is `-2.353279 dB` on
`foundir_train_0215097`; the worst LPIPS regression is `.041199` on
`foundir_train_0217755`.

The automatic checks are therefore: PSNR SOW fail, LPIPS SOW pass, PSNR versus
base pass, LPIPS non-regression fail. Overall gate: fail.

## Visual review

All 16 fixed `Input | Affine Candidate | Ground Truth` images were inspected.
The correction preserves structure and introduces no new hallucination, but
its global brightening/color shift is inconsistent with scene-specific
targets. It helps several dark outputs, while over-brightening or moving other
already-good outputs away from their targets. Examples
`foundir_train_0215097` and `foundir_train_0217755` visibly demonstrate this
failure mode. A single global photometric transform cannot represent the
remaining scene-dependent Haze target mapping.

## Artifact identity and runtime

- result: `checkpoints/foundir_haze_global_rgb_affine_gate_v1/result.json`
- result SHA-256: `59ed87dbbea42379ce78fdf63ef122696634dd01690369b483e46517019503db`
- fixed candidate visuals: 16
- GPU wall time: `0`
- GPU-hours: `0`
- official FoundIR test accessed: false
- sealed internal accessed: false

No Haze full-development or Rain job was started. Further scalar/global
calibration sweeps are not justified by this evidence.
