# FoundIR routing paper ablation — CPU reaggregation

Date: 2026-09-27 UTC
Status: complete; frozen-artifact reaggregation only

## Decision

The existing artifacts already support the routing contribution. At the
final router's accepted count (13,603 / 16,461,
coverage 82.6377%), the ordinary eleven-class head ranked
only by confidence makes **92** wrong accepted decisions,
whereas the final multi-head router makes **17**. This is a
reduction of **75 errors (81.52%)** at the same
accepted count. No new routing or restoration experiment is needed to
establish this matched-coverage comparison.

## Routing ablation on the aligned 16,461-record development set

| Stage | Coverage | Correct | Wrong | Accepted error | Pure eleven-head wrong at same coverage |
|---|---:|---:|---:|---:|---:|
| categorical_confidence_consensus | 87.6435% | 14,226 | 201 | 1.3932% | 153 |
| compositional_linear_v1 | 85.3715% | 13,928 | 125 | 0.8895% | 126 |
| conditional_noise_v2 | 83.4336% | 13,676 | 58 | 0.4223% | 101 |
| asymmetric_absence_v3 | 80.5966% | 13,243 | 24 | 0.1809% | 65 |
| dual_nonlinear | 83.7738% | 13,764 | 26 | 0.1885% | 103 |
| final_multi_head | 82.6377% | 13,586 | 17 | 0.1250% | 92 |

The historical stages are not merely a monotonic threshold sweep.
The initial `categorical_confidence_consensus` is actually worse than
the pure eleven-head ranking at the same coverage (201 versus 153
wrong routes), so consensus by itself is not automatically beneficial.
The evidence supports the final compositional/ambiguity/physical design,
not a generic claim that adding more heads always helps.
`dual_nonlinear` dominates `conditional_noise_v2`: it has higher
coverage and fewer wrong routes. The final router also dominates
`asymmetric_absence_v3`: it has higher coverage and fewer wrong routes.
This supports a contribution from representation/consensus changes, not
only from refusing more records.

## Physical-consensus endpoint for the Noise family

| Class | Records | Correct | Wrong | Abstain | Coverage |
|---|---:|---:|---:|---:|---:|
| noise | 1,859 | 674 | 0 | 1,185 | 36.2561% |
| noise+low_light | 1,743 | 1,168 | 0 | 575 | 67.0109% |

## Downstream restoration evidence

### 1,408-record development crop rehearsal

| Routing status | Records | Input PSNR | Restored PSNR | PSNR gain | Input LPIPS | Restored LPIPS | LPIPS change | PSNR worse | LPIPS worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abstain | 73 | 31.625220 | 31.625220 | 0.000000 | 0.171917 | 0.171917 | 0.000000 | 0 | 0 |
| correct | 1,247 | 18.990829 | 25.616073 | 6.625244 | 0.405331 | 0.166065 | -0.239267 | 199 | 44 |
| fallback | 87 | 9.541988 | 19.076769 | 9.534781 | 0.463830 | 0.198007 | -0.265823 | 3 | 2 |
| wrong | 1 | 15.981540 | 16.522522 | 0.540982 | 0.286962 | 0.227882 | -0.059080 | 0 | 0 |

Fallback is beneficial on average but is not an absolute no-harm
guarantee. This table reports matched input/output metrics by routing
status; it does not infer safety from the function name.

### Full-resolution 660 validation records

| Routing status | Records | Project PSNR / LPIPS | Released FoundIR PSNR / LPIPS |
|---|---:|---:|---:|
| abstain | 44 | 33.404717 / 0.137869 | 36.706342 / 0.072560 |
| correct | 562 | 24.948907 / 0.171727 | 24.726378 / 0.177745 |
| fallback | 54 | 17.642393 / 0.212348 | 15.347872 / 0.250039 |

For the 60-image Blur+Noise+Low-light triple category:

| Routing status | Records | Project PSNR / LPIPS | Released FoundIR PSNR / LPIPS |
|---|---:|---:|---:|
| correct | 53 | 16.715288 / 0.181321 | 24.986867 / 0.038079 |
| fallback | 7 | 18.651010 / 0.147128 | 24.632900 / 0.035290 |

The correctly routed triple subset remains poor, confirming that the
dominant triple failure is restoration composition rather than route
recognition.

## Paper interpretation

The evidence supports a narrow claim: cross-representation and physical
consensus reduce wrong restoration actions substantially beyond ordinary
single-head confidence selection at matched coverage. It does not support
claims of universal safety, open-set robustness, or successful triple
restoration.

## Provenance and limitations

- GPU used: 0; model updates: none; new inference: none.
- Official/sealed images accessed: false.
- All routing stages are aligned to the same 16,461 development records.
- This development set was inspected during router iteration and is not
  an independent lockbox; final paper wording must disclose that boundary.
- The full-resolution sidecars compare per-record against Released FoundIR
  Generalist. Full FoundIR specialist metrics are available only as
  aggregate frozen results, not in these per-record sidecars.

Machine-readable artifacts:

- `reports/foundir_routing_paper_ablation_2026-09-27.json`
- `reports/foundir_routing_paper_ablation_2026-09-27.csv`
- `reports/foundir_routing_risk_coverage_curve_2026-09-27.csv`
