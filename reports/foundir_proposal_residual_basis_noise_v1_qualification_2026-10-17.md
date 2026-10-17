# FoundIR proposal residual-basis Noise V1 GPU qualification — 2026-10-17

## Outcome

`qualification_v1` passed its fresh 25-step engineering qualification on one
homogeneous `8 x NVIDIA H100 80GB HBM3` node. Training remained finite and the
launcher exited with status `0`; checkpointing and the fixed 256-record
source-disjoint development validation both completed. No official, sealed,
internal, or frozen-660 data was accessed.

This is an engineering result only. It does not establish adequate corrective
headroom and must not be used to select or initialize the formal run.

## Runtime and artifacts

- Start: `2026-10-17T09:57:50Z`
- End: `2026-10-17T09:59:08Z`
- Total wall time: about `78 s` (`.0217` hours)
- Allocated GPU-hours: about `.173`
- Logged optimizer-loop time: `31.16433133929968 s` for 25 steps
- Validation-reported seconds per optimizer step, including final validation
  amortization: `1.5732924649864435`
- Checkpoint:
  `checkpoints/foundir_proposal_residual_basis_noise_v1/qualification_v1/step_000025.pt`
- Validation:
  `checkpoints/foundir_proposal_residual_basis_noise_v1/qualification_v1/validation_step_000025.json`
- Config SHA-256:
  `a41c885daa83401453ed31a881cb814af65051296566102c6e815443e6d3f145`

## Development validation

| Arm | PSNR | PSNR gain | LPIPS | LPIPS reduction |
|---|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` |
| Fidelity | `43.13571598380804` | `+.011555805802345276` | `.05056087289904099` | `-.00019880513900716323` |
| Perceptual | `43.13552286475897` | `+.01136268675327301` | `.050557308552015456` | `-.00019524079198163236` |
| Midpoint | `43.135619446635246` | `+.011459268629550934` | `.05055906142661115` | `-.00019699366657732753` |

Here, a negative `LPIPS reduction` means LPIPS became worse. The useful early
signal is the roughly `+.0115 dB` PSNR movement after only 25 warm-up steps.
The approximately `.00020` LPIPS regression and near-identical two heads are
not yet scientific evidence: both heads share the same initialization and
have not had enough optimization to separate.

## Frozen next action

Launch a new-ID, from-zero five-pass run for `8,530` optimizer steps and
validate every `1,706` steps. Do not resume or seed from `qualification_v1`.
Do not change the loss because of this 25-step result, and do not train an
energy or gate before the residual basis demonstrates adequate headroom.

Based on the qualification and the expected slower 192/256-crop phases, the
formal run is conservatively estimated at `3.5–6.0` wall-clock hours on one
eight-GPU node, or `28–48` allocated GPU-hours. The central estimate is about
`4.7` wall-clock hours / `37.6` allocated GPU-hours. The pass-1 measured rate
must replace this estimate once available.
