# FoundIR proposal residual-basis Noise V1 pass 2 — 2026-10-17

## Outcome

The live `formal1` run completed pass 2 at step `3,412` on `8 x H100 80GB`.
Fidelity continues to improve both PSNR and LPIPS and has entered the
predeclared hopeful pass-2 PSNR range. The run remains healthy and continues
unchanged.

## Development validation

| Arm | PSNR | PSNR gain | LPIPS | LPIPS reduction |
|---|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` |
| Fidelity | `43.22963780909777` | `+.10547763109207153` | `.03339703518213355` | `+.016965032577900274` |
| Midpoint | `43.163344368338585` | `+.03918419033288956` | `.030206606008505332` | `+.02015546175152849` |
| Perceptual | `43.07285115867853` | `-.051309019327163696` | `.028950609410458128` | `+.021411458349575696` |

Positive `LPIPS reduction` means LPIPS is lower and better.

## Interpretation

- Fidelity PSNR gain grew from pass 1's `+.044231854379177094 dB` to
  `+.10547763109207153 dB`; LPIPS reduction grew from `.010349548166232125`
  to `.016965032577900274`.
- Fidelity comfortably passes the LPIPS target but still needs about
  `.194522 dB` additional PSNR to clear the joint gate.
- The observed point is inside the `.09–.12 dB` pass-2 trend range identified
  before this validation was seen. This materially improves feasibility, but
  later gains may saturate and must not be assumed linear.
- Midpoint now also improves both metrics, while perception retains the best
  LPIPS at a small PSNR cost. The basis diversity remains useful.

The step-3,412 checkpoint and validation were written at
`2026-10-17T11:07:44Z`. All protected-data access flags remain false. Continue
to the frozen pass-3 checkpoint at step `5,118` without recipe changes.
