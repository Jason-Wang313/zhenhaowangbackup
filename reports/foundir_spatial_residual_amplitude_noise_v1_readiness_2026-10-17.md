# FoundIR spatial residual amplitude Noise V1 readiness — 2026-10-17

## Purpose

The GT oracle showed that global scaling cannot pass the gate, while local 8x8
amplitudes can reach `+.53558 dB/-.01793 LPIPS`. This experiment tests the
smallest deployable mechanism implied by that evidence: predict the local
amplitudes without GT at inference time.

## Frozen method

- Frozen proposal:
  `checkpoints/foundir_noise_mse_lpips_continuation_v1/step_000250.pt`
- Frozen residual basis:
  `checkpoints/foundir_proposal_residual_basis_noise_v1/formal1/step_008530.pt`
- Trainable predictor: `223,393` parameters
- Inputs: observation, proposal, fidelity, fidelity minus proposal
- Output: one bounded `[0,2]` amplitude per non-overlapping 8x8 patch
- Initialization: exact alpha `1`, so the initial output equals fidelity
- Training-only teacher: analytic least-squares amplitude from training GT,
  weighted by residual energy
- Inference/validation: no GT, no oracle selection
- Loss: output MSE + weighted amplitude teacher + weak LPIPS + local no-harm
- Data: unchanged 54,585-record source-disjoint Noise training split and
  fixed 256-record development validation

## Frozen scale

- Engineering qualification: 25 steps, fresh run ID
- Scientific checkpoints: every 1,706 steps
- Primary schedule: five passes / 8,530 steps
- Global batch: 32 on one homogeneous 8xH100 or 8xH200 node
- The qualification checkpoint must not initialize the formal run

Formal runtime must be estimated from the exact qualification hardware. A
provisional qualification estimate is `.02–.04` wall-clock hours and
`.16–.32` allocated GPU-hours.

## Checks completed

- Config JSON parse: pass
- Python compilation: pass
- Bash syntax: pass
- Focused unit tests: 2/2 pass
- Initial amplitude identity: pass
- Analytic teacher recovery: pass
- Frozen step-8,530 state strict load: pass
- Diff check: pass
- Protected data accessed: no

Next command:
`smoke/run_foundir_spatial_residual_amplitude_noise_v1_8gpu.sh` with a fresh
qualification run ID and total steps `25`.
