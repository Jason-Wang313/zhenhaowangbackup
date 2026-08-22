# FoundIR-Lowlight Gate 3 structured-Qwen correction 1 result

Date: 2026-08-22 UTC

## Decision

Correction 1 completed successfully as an experiment and **failed** the frozen
scientific causal gate. Gate 4 remains blocked. Correction 2 is not justified
by the post-result residual attribution: posterior-temperature calibration is
not stable across training families, the correct-versus-shuffled result is at
chance, and the target-useful tone direction changes across domains.

No protected FoundIR-Lowlight ID 1101--1150 was accessed.

## Frozen result

| Variant | Equal-family PSNR | Equal-family LPIPS |
|---|---:|---:|
| measured-only | 17.3927 | 0.38648 |
| oracle + measured | 18.9932 | 0.38828 |
| structured Qwen + measured | 17.4044 | 0.38652 |
| shuffled structured Qwen + measured | 17.4044 | 0.38648 |

| Family | Oracle PSNR gain | Qwen gain over measured | Qwen gain over shuffled | Qwen LPIPS change over measured | Oracle gain recovered |
|---|---:|---:|---:|---:|---:|
| LOL-v2 | +2.1001 | +0.00883 | -0.00130 | +0.000137 | 0.42% |
| SDSD | +1.9823 | -0.00213 | -0.00561 | -0.000028 | -0.11% |
| SICE | +0.7193 | +0.02845 | +0.00691 | +0.000005 | 3.96% |

The output was conservative enough to avoid the large LPIPS regressions of the
initial free-form adapter, but it recovered essentially none of the oracle
headroom and did not causally beat shuffled Qwen on any family.

## What the learned parameters show

The optimizer actively suppressed the Qwen correction:

- gain raw slopes moved from `-2.0` to approximately
  `[-3.86, -3.07, -3.94]`;
- gamma raw slopes moved from `-2.0` to approximately
  `[-3.92, -3.36, -3.98]`;
- the measured-posterior residual scale moved from `-2.0` to `-3.40`;
- posterior-knot adjustments moved toward a compressed high-severity response.

Because slopes are passed through `softplus`, these changes shrink their
magnitude substantially. The result is not an under-trained adapter trying to
make a large correction; it is an optimizer finding that the safest
cross-domain action is nearly zero.

Per-record correct Qwen beat measured-only on 50.0% of LOL-v2, 49.6% of SDSD,
and 77.8% of SICE. More importantly, correct Qwen beat the within-family
group-shuffled condition on only 45.1%, 48.8%, and 38.9%, respectively. The
SICE gain over measured is therefore a mostly non-causal family-level shift,
not image-specific Qwen information.

## Correction-2 attribution

The only narrow remaining interface hypothesis was that the frozen posterior
temperature `1.0` left useful Qwen logit differences too diffuse. A deterministic
five-fold group-held-out attribution used only the 429 frozen training records,
the cached candidate NLLs, measured statistics, and the target-accessing oracle
tone delta. It did not access new images.

| Temperature | Median entropy | Median margin | Overall oracle-tone MSE improvement | SDSD | SICE |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1022 | 0.9242 | +0.585% | -0.068% | +1.875% |
| 0.10 | 0.3235 | 0.6622 | -0.140% | -0.296% | +0.166% |
| 0.20 | 0.5893 | 0.3448 | -0.022% | +0.164% | -0.391% |
| 0.35 | 0.7922 | 0.1733 | -0.136% | +0.183% | -0.766% |
| 0.50 | 0.8819 | 0.1080 | -0.175% | +0.211% | -0.936% |
| 0.75 | 0.9430 | 0.0638 | -0.107% | +0.328% | -0.964% |
| 1.00 | 0.9668 | 0.0440 | -0.014% | +0.436% | -0.903% |
| 1.50 | 0.9848 | 0.0267 | +0.129% | +0.583% | -0.766% |
| 2.00 | 0.9914 | 0.0191 | +0.222% | +0.671% | -0.663% |

No temperature improves both training families. The best aggregate value is
small and regresses SDSD. Temperature sharpening therefore cannot credibly
recover the frozen `0.20 dB` per-family causal margin, much less half of the
oracle increment.

The expected-severity association with oracle tone parameters is also
domain-dependent. For example, the mean correlation with the three gamma
parameters is `-0.399` on training SDSD, `+0.003` on training SICE, `-0.157`
on outer LOL-v2, `-0.787` on outer SDSD, and `-0.293` on outer SICE. A second
global monotone decoder would repeat correction 1; a family-specific decoder
would encode dataset rendering and is not a credible unknown-camera solution.

## Stop interpretation

The original oracle result still proves large restoration/tone headroom. The
two Qwen experiments now show that the available relative low-light severity
signal does not identify the target rendering after ordinary input statistics
are known. This is the identifiability problem anticipated by the strategy,
not a reason to add capacity blindly.

Using correction 2 would be justified only by new information, such as
objective real-world severity/formation supervision that changes the Qwen
signal itself. Reparameterizing the same cached scores, adding the unstable
32-D visual projection, using spatial Qwen injection, or weakening the causal
threshold is not justified. The current route must stop before Gate 4 for a
user/advisor decision on whether to acquire that supervision or formally
decouple the restoration-SOTA branch from the advisor-directed causal-Qwen
claim.

## Evidence identities

- correction checkpoint SHA-256:
  `6f23fb7cc0a5e6e0e1abe20d69c4080babb44bd7f27f74f2af9b25b4f6db20c2`;
- prediction manifest SHA-256:
  `649eeee3dea3988610c70e1c8647000b080962f4ac651b7f0e77588a4b2a9856`;
- metric rows SHA-256:
  `a02520808a50b40607b9c4c764acf8b8b6cc1128c29b5242359f369da20de50d`;
- result JSON SHA-256:
  `90bf88f1bea05887b415810c920747abbaac713152503a5690894e8487269e68`.

The run began at `2026-08-22T14:10:19Z` and completed at
`2026-08-22T14:26:14Z`. Training used all four H100 80 GB GPUs and took
`571.76` seconds; the complete bundle took 15 minutes 55 seconds.
