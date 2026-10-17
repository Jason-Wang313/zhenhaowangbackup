# FoundIR spatial residual amplitude Noise V1 pass 2 — 2026-10-17

## Outcome

The live `formal1` run completed pass 2 at step `3,412` on `8 x H100 80GB`.
The deployable spatial predictor now has a very small PSNR advantage over
frozen fidelity, but pays for it with worse LPIPS. Relative to proposal, LPIPS
still clears the joint gate and PSNR remains short.

## Development validation

| Arm | PSNR | Gain vs proposal | LPIPS | Reduction vs proposal |
|---|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` |
| Frozen fidelity | `43.39715355634689` | `+.27299337834119797` | `.028955043449059303` | `+.02140702431097452` |
| Spatial pass 2 | `43.3995122089982` | `+.27535203099250793` | `.033109975053776` | `+.017252092706257827` |

Relative to frozen fidelity, spatial changes are `+.00235865265130997 dB`
PSNR and `+.004154931604716697` LPIPS (worse). The predictor still needs
`.02464796900749206 dB` more PSNR to cross the joint target.

The amplitude map mean is `.8973334357142448`, standard deviation
`.20313113427297755`, and boundary fraction zero. It has learned substantial
variation without saturating, but only a small fraction of the GT-local
oracle's headroom transfers to inference-time prediction so far.

The step-3,412 checkpoint and validation were written at
`2026-10-17T14:19:10Z`. All protected-data flags remain false. Continue to the
frozen step-5,118 checkpoint without changes.
