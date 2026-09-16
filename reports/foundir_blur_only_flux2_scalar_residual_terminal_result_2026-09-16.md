# FLUX.2 Blur-only scalar residual feasibility terminal result — 2026-09-16

**DIAGNOSTIC ONLY / 非主线.** The frozen 512-record feasibility run completed
all intended inference and artifacts but failed its predeclared automatic gate.
It did not access development, sealed internal, or official FoundIR test data.

## Runtime and identities

- homogeneous `8 x NVIDIA H100 80GB HBM3`;
- start `2026-09-16T10:10:22Z`, result emitted around `10:18:07Z`;
- model wall time `439.5092 s`;
- all `512/512` immutable sidecars and `128/128` validation base PNGs exist;
- no OOM, NCCL failure, NaN, or inference exception;
- config SHA-256: `a2e49e2094a1f8f8911377729cd48297891dd3392435638d12faddb838844bac`;
- checkpoint SHA-256: `e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`;
- result SHA-256: `08e7b0220ea02e4067c1c4b0ef5834a93976857a46fc7c27e502e1891c41b6bc`.

Torchrun's nonzero exit is the intentional scientific-failure return after a
complete result. The shell wrapper expected status 2, but torch elastic exposed
status 1 and printed `unexpected torchrun status 1`. This is a wrapper status
normalization defect, not a GPU/runtime/data failure; the result and terminal
governance hold were already emitted correctly.

## Frozen calibration result

| alpha | PSNR | LPIPS |
|---:|---:|---:|
| .250 | 22.962656 | .325165 |
| .375 | **22.986674** | .284560 |
| .500 | 22.911411 | .256071 |
| .625 | 22.730487 | .238376 |
| .750 | 22.447922 | .228699 |
| .875 | 22.067155 | **.225097** |
| 1.000 (FLUX) | 21.621397 | .225653 |

No alpha satisfied the predeclared absolute calibration constraint LPIPS
`<=.17`, so no alpha was selected and the frozen automatic gate failed before
validation selection. No full-development evaluation is authorized.

## Read-only validation audit

All alpha metrics were already generated for the frozen 128 validation records.
The following audit is descriptive/post-hoc and cannot alter the failed gate:

| alpha | PSNR | LPIPS | PSNR delta vs FLUX | LPIPS delta vs FLUX |
|---:|---:|---:|---:|---:|
| .250 | 22.754873 | .332694 | +.877756 | +.103593 |
| .375 | 22.871236 | .292874 | +.994120 | +.063773 |
| .500 | **22.902913** | .263793 | **+1.025797** | +.034691 |
| .625 | 22.839735 | .245332 | +.962619 | +.016230 |
| .750 | 22.692221 | .234457 | +.815104 | +.005356 |
| .875 | 22.353720 | .229522 | +.476604 | +.000421 |
| 1.000 (FLUX) | 21.877117 | **.229102** | 0 | 0 |

At alpha `.875`, validation PSNR improves on 93.75% of records, gains
`+.476604 dB` on average, regresses LPIPS only `+.000421`, and has a worst
per-record PSNR regression of `.765213 dB`. This is promising relative evidence.
It does not pass the frozen experiment because the absolute `.17` calibration
constraint was not met.

## Interpretation and stop

The absolute `.17` threshold was transferred from the full-development target
to a different training-crop distribution whose unblended FLUX baseline LPIPS
is `.225653` calibration / `.229102` validation. It therefore made alpha
selection impossible even though `.875` shows a strong relative PSNR/LPIPS
tradeoff. This is a design/protocol mismatch, not evidence that blending has no
signal. Conversely, the post-hoc `.875` result cannot be promoted to a selected
candidate or used to justify automatic development access.

Active terminal hold:
`flux2_blur_scalar_residual_feasibility_result-fe6e894531bf58fa`. Stop here.
Any proposal to replace the absolute calibration constraint with a relative
constraint and evaluate a frozen alpha requires a new explicit decision.
