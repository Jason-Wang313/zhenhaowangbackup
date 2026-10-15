# FoundIR router nonlinear-head leave-one-out ablation

CPU-only factorial reaggregation of 16,461 frozen development features and frozen tiny-head checkpoints. No image, Qwen, restoration, training, GPU, or official/sealed-image access.

## Frozen thresholds

| Variant | Coverage | Correct | Wrong | Abstain | Accepted error |
|---|---:|---:|---:|---:|---:|
| atomic_plus_linear | 80.5966% | 13,243 | 24 | 3,194 | 0.1809% |
| plus_lowlight_nonlinear | 85.2439% | 14,006 | 26 | 2,429 | 0.1853% |
| plus_noise_nonlinear | 79.1264% | 13,001 | 24 | 3,436 | 0.1843% |
| plus_both_nonlinear | 83.7738% | 13,764 | 26 | 2,671 | 0.1885% |

## Matched coverage (13,025/16,461, 79.1264%)

| Variant | Correct | Wrong | Abstain | Accepted error |
|---|---:|---:|---:|---:|
| atomic_plus_linear | 13,004 | 21 | 3,436 | 0.1612% |
| plus_lowlight_nonlinear | 13,003 | 22 | 3,436 | 0.1689% |
| plus_noise_nonlinear | 13,001 | 24 | 3,436 | 0.1843% |
| plus_both_nonlinear | 13,001 | 24 | 3,436 | 0.1843% |

## Noise / Low-light focus at frozen thresholds

| Variant | Coverage | Correct | Wrong | Abstain | Accepted error |
|---|---:|---:|---:|---:|---:|
| atomic_plus_linear | 56.9972% | 2,841 | 10 | 2,151 | 0.3508% |
| plus_lowlight_nonlinear | 72.2911% | 3,604 | 12 | 1,386 | 0.3319% |
| plus_noise_nonlinear | 52.1591% | 2,599 | 10 | 2,393 | 0.3833% |
| plus_both_nonlinear | 67.4530% | 3,362 | 12 | 1,628 | 0.3557% |

## Limitations

- This is a frozen development-set reaggregation, not an independent lockbox result.
- Atomic+linear is the historical linear conditional baseline; it is not an unconditioned atomic-only head.
- Matched coverage uses post-hoc ranking of already accepted routes and does not recalibrate or retrain any head.
- No downstream restoration inference was run; results measure routing only.
