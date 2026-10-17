# FoundIR residual-basis final oracle readiness — 2026-10-17

## Question

Does the frozen step-8,530 fidelity residual already contain enough corrective
direction to exceed the `+.30 dB/-.005 LPIPS` development gate when its global
or local amplitude is selected with access to GT?

This is a feasibility upper bound only. GT-selected outputs are not deployable
and cannot be reported as model inference results.

## Frozen protocol

- Checkpoint:
  `checkpoints/foundir_proposal_residual_basis_noise_v1/formal1/step_008530.pt`
- Checkpoint SHA-256:
  `9ebc6b73e627e94f35ef52f511696215285c7efb9f800936d0b3d873b8aac95a`
- Data: the same fixed 256-record source-disjoint development subset
- Global amplitudes: `0.000–2.000`, step `.025`
- Local analytic amplitudes: hard and bilinear-smoothed 8/16/32/64 patches,
  bounded to `[0, 2]`
- Required protocol reproduction: proposal and alpha-1 fidelity within
  `.001 dB/.00001 LPIPS`
- Protected data: prohibited

The output includes every fixed-candidate mean and a per-image Pareto sweep.
A fixed candidate crossing the gate establishes a simple amplitude headroom;
only a GT-selected Pareto crossing establishes oracle headroom. Neither is a
deployable result without a separately learned or frozen selection rule.

## Qualification and cost

Python compilation, Bash syntax, two focused CPU unit tests, and diff checks
pass. The active Codex container exposes no GPU. Based on the earlier 49-arm
oracle and this audit's 89 candidates, estimated runtime on one eight-GPU node
is `.10–.17` wall-clock hours and `.8–1.36` allocated GPU-hours.

Launcher:
`smoke/run_foundir_proposal_residual_basis_noise_v1_oracle_8gpu.sh`.
