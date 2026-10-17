# FoundIR spatial residual amplitude Noise V1 pass 3 — 2026-10-17

## Outcome

The live `formal1` run completed pass 3 at step `5,118` on `8 x H100 80GB`.
Spatial prediction keeps a tiny PSNR advantage over frozen fidelity, but the
gain is flattening and remains far below the GT-local oracle.

## Development validation

| Arm | PSNR | Gain vs proposal | LPIPS | Reduction vs proposal |
|---|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` |
| Frozen fidelity | `43.39715355634689` | `+.27299337834119797` | `.028955043449059303` | `+.02140702431097452` |
| Spatial pass 3 | `43.400297187268734` | `+.27613700926303864` | `.0317960577121994` | `+.018566010047834425` |

Relative to fidelity, spatial changes are `+.0031436309218444425 dB` PSNR
and `+.0028410142631400966` LPIPS (worse). It needs another
`.023862990736961353 dB` to cross the PSNR gate. Pass-2 to pass-3 PSNR gain
grew by only `.0007849782705344696 dB`.

The amplitude mean is `.9737586602568626`, standard deviation
`.21764669148700272`, and boundary fraction zero. The map remains nontrivial
and unsaturated, but the learned inference-time mapping has recovered only a
small fraction of the GT-local upper bound.

The step-5,118 checkpoint and validation were written at
`2026-10-17T14:46:37Z`. All protected-data flags remain false. Continue the
frozen run through steps `6,824` and `8,530` without changes.
