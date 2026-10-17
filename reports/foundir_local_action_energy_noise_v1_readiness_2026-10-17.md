# FoundIR local action energy v1 readiness — 2026-10-17

## Decision and causal motivation

The user stopped continuous spatial-alpha run
`foundir_spatial_residual_amplitude_noise_v1/formal1`. Its launcher ended with
status `120` at `2026-10-17T14:53:05Z`; last logged step was `5,525`, and the
last complete checkpoint/validation was step `5,118`. It must not be resumed.

The continuous predictor recovered only `+.0031436309218444425 dB` beyond the
frozen fidelity result at its last complete validation. This is consistent
with continuous regression averaging incompatible local actions. In contrast,
the frozen residual's GT-only 8x8 oracle shows `+.5355773419141769 dB` versus
proposal, about `+.2626 dB` beyond fidelity. Only `.02700662165880202 dB` more
than fidelity is needed to cross the original proposal-relative `+.30 dB`
gate. The replacement therefore learns discrete local action comparisons
rather than another continuous alpha target.

## Frozen method contract

- Proposal checkpoint:
  `checkpoints/foundir_noise_mse_lpips_continuation_v1/step_000250.pt`
- Proposal SHA-256:
  `0ec585e4bfc07a7d85c4f26e8abcabc9b778299db4b83440cc5bbabf02f2b4a0`
- Residual-basis checkpoint:
  `checkpoints/foundir_proposal_residual_basis_noise_v1/formal1/step_008530.pt`
- Residual-basis SHA-256:
  `9ebc6b73e627e94f35ef52f511696215285c7efb9f800936d0b3d873b8aac95a`
- Candidate amplitudes: `{0,.5,.75,1,1.25,1.5,2}`.
- One decision is made per non-overlapping 8x8 region.
- The `1.0` action is the safe fidelity baseline.
- Initialization chooses `1.0` exactly. At inference, an alternative is used
  only when its predicted cost beats the baseline by more than `.05`.
- Training GT supplies relative local log-MSE and best-action labels. Neither
  validation nor inference uses GT to choose an action.
- The energy model has `1,369,575` trainable parameters. Proposal and residual
  basis remain frozen.

The loss contains relative-cost SmoothL1 regression, best-candidate ranking,
differentiable soft-action output MSE, weak LPIPS, local no-harm, and an
explicit penalty for assigning probability to candidates whose true local
cost is no better than the safe action.

## Data and access boundary

Training uses only the existing 54,585-record, 1,755-source-group Noise
training split. Development validation remains the fixed 256-record,
source-disjoint 256x256-crop subset. Official, sealed, internal, parent-test,
and frozen-660 pixels are not accessed and may not be used for training,
tuning, checkpoint selection, or early stopping.

## Completed CPU/static qualification

- JSON parse: pass.
- Python compilation: pass.
- Bash syntax: pass.
- Focused unit tests: 3/3 pass.
- Initial fail-closed output exactly reproduces frozen fidelity: pass.
- Candidate target identifies a known alpha `1.5` construction: pass.
- Soft candidate selection carries finite gradients: pass.
- Frozen step-8,530 residual basis strict state-dict load: pass.
- Diff whitespace check: pass.

The current Codex container has no `nvidia-smi`, so no GPU training was
launched here.

## Next run and cost

Run one fresh 25-step engineering qualification on a homogeneous local
8xH100 or 8xH200 node. Estimated cost is `.03–.08` wall-clock hours and
`.24–.64` allocated GPU-hours. This estimate must be replaced with measured
time. The qualification checks finite optimization, memory, DDP, checkpoint,
validation, exact access flags, and whether action choices leave the safe
initialization. It is not scientific evidence and its checkpoint must never
seed the formal run.

If qualification passes, the predeclared formal schedule is five effective
passes / 8,530 optimizer steps. Initial estimate is `3–5` wall-clock hours and
`24–40` allocated GPU-hours on one eight-GPU node; update it from measured
qualification/pass-1 throughput.
