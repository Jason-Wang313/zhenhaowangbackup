# FLUX.2 Blur-only frozen scalar residual full-development result — 2026-09-16

The one-shot 2,595-record development confirmation completed normally. The
candidate passes both requested SOW average metrics and the fixed visual review,
but fails the separately predeclared worst-record regression safety gate.

## Runtime and identities

- homogeneous `8 x NVIDIA H100 80GB HBM3`;
- start `2026-09-16T10:53:06Z`, end `2026-09-16T11:31:11Z`;
- model wall time `2118.4764 s`;
- 2,595/2,595 sidecars and 16/16 fixed visuals complete;
- no OOM, NCCL, NaN, or runtime failure; exit 2 is intentional scientific-failure status;
- config SHA: `b756c7b223d8aed8e2a7018eec8e2e395042f772e477584b79ba2a98c7a12da5`;
- result SHA: `5b9dac81e11d338d17a36c24bba90f8e5c08299727d15883ebee0d5129e25daf`;
- step-10k checkpoint SHA: `e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`.

## Result

| System | PSNR | LPIPS | SOW PSNR >=25.9 | SOW LPIPS <=.18 |
|---|---:|---:|:---:|:---:|
| paired alpha=1 base replay | 25.730036 | .145439 | fail | pass |
| frozen alpha=.75 candidate | **26.559141** | **.151993** | **pass** | **pass** |

The base replay reproduces the prior `25.734236/.145406` within frozen
tolerances. Candidate versus base is `+.829105 dB / +.006554 LPIPS`, with
PSNR wins on 89.56% of all records. This is a substantial and directly paired
gain, not a seed or hardware artifact.

## Tail gate and visual review

The only failed automatic check is maximum per-record PSNR regression:
`12.569169 dB` versus the allowed `2 dB`. Three of 2,595 records regress by
more than 2 dB; 32 regress by more than 1 dB. The worst record is
`foundir_train_0004783`: base `44.049135/.002379`, candidate
`31.479966/.016354`, input `19.546748/.368966`. Thus the candidate remains an
excellent restoration on that record, but blending degrades an exceptionally
accurate base output and violates the frozen conservative rule.

All 16 fixed `Input | Candidate | GT` triptychs were reviewed. They are
coherent and artifact-free, with strong motion deblurring and no gross
hallucination. Hash-bound visual review:
`reports/foundir_blur_only_flux2_scalar_residual_full_development_visual_review_v1.json`,
SHA `7f2cbb91e1706f2ac1aa847a4d9fee960f4088e6d31cb2377191a3cffa8c18e9`.

## Conclusion

The project-level SOW average Blur metrics are achieved on the complete
source-disjoint development set: `26.559141 dB / .151993`. However, the
experiment's stricter scientific admission gate is false because of tail
regression. Report both facts; do not claim the complete gate passed or tune
alpha on development. Active hold:
`flux2_blur_scalar_residual_full_development_result-b484e042c294985c`.
