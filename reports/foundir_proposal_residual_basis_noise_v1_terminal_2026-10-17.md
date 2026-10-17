# FoundIR proposal residual-basis Noise V1 terminal — 2026-10-17

## Outcome

`formal1` completed its frozen five-pass schedule at step `8,530` on
`8 x NVIDIA H100 80GB HBM3` with launcher status `0`. Fidelity improves both
metrics substantially, but the predeclared joint gate is not passed because
PSNR remains `.02700662165880202 dB` below the `+.30 dB` requirement.

## Final development validation

| Arm | PSNR | PSNR gain | LPIPS | LPIPS reduction |
|---|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` |
| Fidelity | `43.39715355634689` | `+.27299337834119797` | `.028955043449059303` | `+.02140702431097452` |
| Midpoint | `43.314712010324` | `+.19055183231830597` | `.026323031037463807` | `+.024039036722570017` |
| Perceptual | `43.20691728591919` | `+.08275710791349411` | `.02608288897499733` | `+.024279178785036493` |

Positive `LPIPS reduction` means lower and better LPIPS.

## Gate decision and interpretation

- Fidelity passes the LPIPS requirement by a wide margin.
- Fidelity misses the PSNR gate by `.02700662165880202 dB`; the joint gate is
  therefore **not passed**.
- The final pass added `+.03130790591239929 dB` over pass 4, larger than the
  pass-3 to pass-4 increment. The model was not completely stationary despite
  the cosine learning rate reaching `1e-6`.
- In MSE terms, closing the remaining `.0270066 dB` requires about `.620%`
  further reduction relative to the final fidelity output. This is physically
  modest, but must not be claimed attainable without a frozen feasibility
  audit.

## Runtime and provenance

- Start: `2026-10-17T10:08:46Z`
- End: `2026-10-17T12:43:59Z`
- Wall-clock hours: `2.5869444444444443`
- Allocated GPU-hours: `20.695555555555554`
- Config SHA-256:
  `a41c885daa83401453ed31a881cb814af65051296566102c6e815443e6d3f145`
- Final checkpoint SHA-256:
  `9ebc6b73e627e94f35ef52f511696215285c7efb9f800936d0b3d873b8aac95a`
- Final validation SHA-256:
  `ef94b47cbe4bb10cbe6a8eab9dcb3bd277c32da12fc00d024fd503c3611f3d7d`
- Official/sealed/internal/frozen-660 data accessed: no

Do not resume or silently extend `formal1`. The next experiment should be a
no-training GT-only amplitude/local oracle over the frozen final fidelity
residual. If that oracle crosses the joint gate, a predeclared short
fidelity-only continuation may be justified; if it does not, further schedule
tuning alone is not supported.
