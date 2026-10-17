# FoundIR proposal residual-basis Noise V1 pass 1 — 2026-10-17

## Outcome

The live `formal1` run completed its first effective pass at step `1,706` on
one homogeneous `8 x NVIDIA H100 80GB HBM3` node. The fidelity head improves
both PSNR and LPIPS over the frozen proposal. This is the first evidence that
the restoration-native residual basis supplies correction information that
the old direct residual lacked.

The run remains active under the frozen five-pass schedule. Do not restart,
duplicate, retune, or interrupt it based on this interim result.

## Development validation

| Arm | PSNR | PSNR gain | LPIPS | LPIPS reduction |
|---|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` |
| Fidelity | `43.16839203238487` | `+.044231854379177094` | `.0400125195938017` | `+.010349548166232125` |
| Midpoint | `43.10491702705622` | `-.01924315094947815` | `.03484063790710934` | `+.015521429852924484` |
| Perceptual | `43.0077431127429` | `-.1164170652627945` | `.03262401805659465` | `+.017738049703439174` |

Positive `LPIPS reduction` means LPIPS is lower and therefore better.

## Interpretation

- Fidelity is the only current arm that improves both metrics.
- Fidelity already clears the LPIPS part of the target (`-.005`) by more than
  twofold, but its `+.0442 dB` PSNR gain is only about 15% of the `+.30 dB`
  requirement.
- The old direct residual's GT-only maximum-PSNR oracle achieved only
  `+.04245246201753616 dB/-.003946266945604293` LPIPS. The learned fidelity
  basis now exceeds that old oracle in both metrics without GT-time selection.
- The perceptual head's much lower LPIPS and lower PSNR show that the two heads
  have separated as designed. A later selector may use this diversity, but no
  energy/gate work is authorized until the frozen basis run completes.

## Runtime and data boundary

- Launch: `2026-10-17T10:08:46Z`
- Step-1,706 validation written: `2026-10-17T10:36:14Z`
- Approximate pass-1 wall time: `.458` hours
- Approximate allocated GPU-hours: `3.66`
- Validation-reported seconds per optimizer step: `.9659281564561993`
- Official/sealed/internal/frozen-660 data accessed: no

Continue to step `8,530` unchanged, validating at steps `3,412`, `5,118`,
`6,824`, and `8,530`.
