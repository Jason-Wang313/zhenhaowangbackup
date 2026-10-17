# FoundIR spatial residual amplitude Noise V1 qualification — 2026-10-17

## Outcome

The fresh 25-step `qualification_v1` run completed successfully on one
homogeneous `8 x NVIDIA H100 80GB HBM3` node. Training remained finite and the
launcher exited with status `0`; checkpointing and fixed 256-record
development validation completed. No protected data was accessed.

This is an engineering qualification only. Its checkpoint must not initialize
the formal run.

## Validation

| Arm | PSNR | Gain | LPIPS | Reduction |
|---|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` |
| Frozen fidelity | `43.39715355634689` | `+.27299337834119797` | `.028955043449059303` | `+.02140702431097452` |
| Spatial step 25 | `43.39715509861708` | `+.27299492061138153` | `.028954966111086833` | `+.02140710164894699` |

The spatial result differs from fidelity by only `+.00000154227018956 dB` and
`-.00000007733797247` LPIPS. Predicted amplitude has mean
`1.0000218451023102`, standard deviation `.0004146491084407974`, and zero
boundary fraction. This is expected after only 25 warm-up steps from an exact
alpha-1 initialization; it is neither success nor failure of the hypothesis.

## Runtime and next action

- Start: `2026-10-17T13:19:04Z`
- End: `2026-10-17T13:20:14Z`
- Wall-clock hours: about `.0194`
- Allocated GPU-hours: about `.1556`
- Validation-reported seconds per optimizer step: `1.353417037241161`
- Protected-data access: none

Launch a new-ID, from-zero five-pass run for `8,530` steps and validate every
`1,706` steps. A conservative H100 estimate is `2.5–3.5` wall-clock hours and
`20–28` allocated GPU-hours. Replace it with measured pass-1 timing.
