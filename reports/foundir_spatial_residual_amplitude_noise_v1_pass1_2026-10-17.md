# FoundIR spatial residual amplitude Noise V1 pass 1 — 2026-10-17

## Outcome

The live `formal1` run completed its first effective pass at step `1,706` on
`8 x H100 80GB`. Training is finite and continues normally, but the deployable
spatial predictor is slightly worse than frozen fidelity on both validation
metrics.

## Development validation

| Arm | PSNR | Gain vs proposal | LPIPS | Reduction vs proposal |
|---|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` |
| Frozen fidelity | `43.39715355634689` | `+.27299337834119797` | `.028955043449059303` | `+.02140702431097452` |
| Spatial pass 1 | `43.395900413393974` | `+.27174023538827896` | `.029359601011492487` | `+.021002466748541337` |

Relative to frozen fidelity, spatial changes are `-.00125314295291574 dB`
PSNR and `+.000404557562433184` LPIPS (worse).

The predicted amplitude map has mean `1.0358318537473679`, standard deviation
`.11071481758944479`, and zero boundary fraction. It has therefore moved away
from identity and is learning; the learned variation has simply not yet
generalized into better metrics.

## Runtime and decision

- Formal launch artifact: `2026-10-17T13:24:38Z`
- Step-1,706 validation: `2026-10-17T13:51:11Z`
- Approximate pass-1 wall time: `.443` hours
- Approximate allocated GPU-hours: `3.54`
- Validation-reported seconds per optimizer step: `.9339068072857848`
- Protected-data access: none

Continue to step `8,530` under the unchanged frozen recipe. Do not interpret
this small first-pass regression as proof that the local-amplitude hypothesis
fails, and do not retune from this interim checkpoint.
