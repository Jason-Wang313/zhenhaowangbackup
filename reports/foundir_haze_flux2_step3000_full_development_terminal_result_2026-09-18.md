# FLUX.2 Haze step-3000 full-development terminal result

## Outcome

The immutable isolated-Haze FLUX step-3000 checkpoint completed evaluation on
all 1,580 source-group-disjoint development records across 158 source groups.
Both formal Stage-2 SOW mean metrics pass, and the 95% equal-source-group
bootstrap remains inside the SOW bounds. The separately required fixed visual
gate fails on one near-clean record, so this is a robust aggregate metric pass
but not a complete identity/visual-safety pass.

| System | PSNR | LPIPS | SOW PSNR >=22.1 | SOW LPIPS <=.19 |
|---|---:|---:|:---:|:---:|
| Input | 14.767724 | .320877 | fail | fail |
| FLUX Haze step 3000 | **23.311888** | **.101155** | **pass** | **pass** |

SOW margins:

- PSNR: `+1.211888 dB`, or **+5.4837%** relative to the 22.1 floor;
- LPIPS safety: `.088845`, or **46.7605%** relative to the .19 cap.

Versus the earlier 128-record pilot (`22.707914/.103458`), the complete result
improves PSNR by `.603974 dB` and improves LPIPS by `.002302`. The larger
validation set did not expose a mean-metric regression.

## Source-group confidence

The 5,000-resample, 95% equal-source-group bootstrap is:

- group-balanced mean: `23.311888/.101155`;
- PSNR interval: `[22.882761, 23.721181]`;
- LPIPS interval: `[.094556, .108192]`.

Even the conservative bounds pass SOW: PSNR lower-bound safety is
`+.782761 dB / +3.5419%`, and LPIPS upper-bound safety is
`.081808 / 43.0569%`.

## Tail and fixed visual review

Relative to input, PSNR improves on 92.4051% and LPIPS improves on 97.7848%
of records. However, 90 records regress by more than 1 dB PSNR, 70 regress by
more than 2 dB, and the worst regression is 21.448758 dB. These tails are
descriptive rather than part of the user's formal mean SOW gate, but they
identify incomplete near-clean protection.

Fifteen of sixteen prospectively fixed `Input | Output | GT` visuals pass for
effective haze removal, coherent structure, and absence of gross hallucination.
`foundir_train_0206837` fails: an already near-clean input at
`35.369958/.017752` is shifted strongly orange and becomes
`17.857776/.094846`, losing 17.512182 dB. Hash-bound review:
`reports/foundir_haze_flux2_step3000_full_development_visual_review_v1.json`.

Therefore:

- contractual mean PSNR/LPIPS: pass;
- source-group bootstrap against SOW: pass;
- fixed visual/near-clean identity safety: fail;
- no claim that Haze is visually perfect or tail-safe.

## Runtime and boundaries

- start: `2026-09-18T03:58:07Z`;
- end: `2026-09-18T04:23:58Z`;
- total end-to-end wall time: **25m51s** on homogeneous 8xH100;
- approximate aggregate GPU-hours: **3.45**;
- evaluator distributed wall time: `1382.0913 s`;
- result SHA-256:
  `66e8a552558f69c7e24830928bed2e9c13cae094e4ec3ecfa8f319d765f733fe`;
- 1,580/1,580 immutable sidecars and 16/16 fixed visuals complete;
- no training, tuning, released FoundIR baseline, sealed internal, or official
  FoundIR test access.

Stop here. Do not automatically train, route, or launch another evaluation.
The next decision is whether the user accepts the robust SOW aggregate pass or
authorizes one bounded, simple near-clean/strength protection intervention.
