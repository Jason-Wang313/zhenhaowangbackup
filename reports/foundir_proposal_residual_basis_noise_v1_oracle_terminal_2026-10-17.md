# FoundIR residual-basis final oracle terminal — 2026-10-17

## Outcome

The frozen step-8,530 fidelity-residual feasibility audit completed normally
on `8 x H100 80GB`. It exactly reproduced the proposal and fidelity validation
metrics and did not access protected data.

The central result is asymmetric:

- No image-wide global amplitude crosses the joint gate. Alpha `1.000` remains
  the maximum-PSNR global setting at `+.27299337834119797 dB`.
- GT-derived local amplitudes have ample headroom. The 8x8 hard-patch upper
  bound reaches `+.5355773419141769 dB/-.01792761307387991 LPIPS`, exceeding
  both targets.

Therefore the learned residual direction is informative but needs spatially
varying application. Merely increasing a global scale or extending the same
training schedule is not supported by this audit.

## Key results

| Candidate | PSNR | Gain | LPIPS | Reduction | Joint gate |
|---|---:|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` | No |
| Global alpha 1.000 / deployed fidelity | `43.39715355634689` | `+.27299337834119797` | `.028955043449059303` | `+.02140702431097452` | No |
| GT-local 64x64 hard | `43.55702514201403` | `+.4328649640083313` | `.03434032116729213` | `+.016021746592741692` | Yes |
| GT-local 32x32 hard | `43.5728612318635` | `+.44870105385780334` | `.03446503311624838` | `+.015897034643785446` | Yes |
| GT-local 16x16 hard | `43.60152127593756` | `+.4773610979318619` | `.03399479424479068` | `+.01636727351524314` | Yes |
| GT-local 8x8 hard | `43.65973751991987` | `+.5355773419141769` | `.03243445468615391` | `+.01792761307387991` | Yes |

Every local row uses GT to compute patch amplitudes and is an upper bound, not
a deployable inference result. The per-image balanced Pareto selector chooses
8x8 hard on 255 records and 8x8 smooth on one, reaching
`+.5355988889932632 dB/-.01792761789056385 LPIPS`; this nearly identical result
confirms that candidate selection is not the main source of the gain.

## Protocol and cost

- Start: `2026-10-17T13:02:00Z`
- End: `2026-10-17T13:03:12Z`
- Wall-clock hours: `.020`
- Allocated GPU-hours: `.160`
- Proposal reproduction difference: exactly `0 dB/0 LPIPS`
- Fidelity reproduction difference: exactly `0 dB/0 LPIPS`
- Official/sealed/internal/frozen-660 accessed: no
- Result JSON:
  `reports/foundir_proposal_residual_basis_noise_v1_oracle_step008530.json`
- Result SHA-256:
  `ce6ab400c5003a630f3e05fed60a5e1fe9e6d6514af86cddc1fed6b9e65e2f66`

## Decision

The lack-of-corrective-information hypothesis is rejected for the new
residual basis: sufficient information exists locally. The next causal test
should learn an 8x8 spatial amplitude map from inference-available evidence
(degraded input, proposal, fidelity output and their residual), supervised on
training-only analytic amplitudes. It must be evaluated as a deployable model
without GT-time selection and compared against alpha `1.000`. Do not claim the
oracle numbers as achieved restoration performance.
