# FoundIR proposal residual-basis Noise V1 pass 3 — 2026-10-17

## Outcome

The live `formal1` run completed pass 3 at step `5,118` on `8 x H100 80GB`.
Fidelity now sits only `.06934652030467986 dB` below the joint PSNR gate while
comfortably exceeding the LPIPS requirement. The frozen run continues through
passes 4 and 5 without changes.

## Development validation

| Arm | PSNR | PSNR gain | LPIPS | LPIPS reduction |
|---|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` |
| Fidelity | `43.354813657701015` | `+.23065347969532013` | `.030396450317311974` | `+.01996561744272185` |
| Midpoint | `43.2777451351285` | `+.15358495712280273` | `.027026652820495656` | `+.023335414939538168` |
| Perceptual | `43.174893632531166` | `+.050733454525470734` | `.02612984426014009` | `+.024232223499893735` |

Positive `LPIPS reduction` means lower and better LPIPS.

## Trend

- Fidelity PSNR gain by pass: `+.044231854379177094`,
  `+.10547763109207153`, `+.23065347969532013 dB`.
- The pass-2 to pass-3 increment is `+.1251758486032486 dB`; no validation
  saturation is visible yet.
- Fidelity LPIPS reduction by pass: `.010349548166232125`,
  `.016965032577900274`, `.01996561744272185`.
- The joint gate is not yet passed because PSNR remains `.06934652030467986
  dB` short. Do not claim success before a recorded validation crosses it.

The step-5,118 checkpoint and validation were written at
`2026-10-17T11:39:52Z`. All protected-data flags remain false. Crop size now
progresses from 192 to 256 under the frozen schedule; continue to step `6,824`
without changes.
