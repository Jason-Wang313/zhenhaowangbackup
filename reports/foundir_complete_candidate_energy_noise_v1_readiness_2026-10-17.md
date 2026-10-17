# Complete candidate energy v1 readiness — 2026-10-17

## Decision and scope

The user explicitly paused the EBT Generalist gate and authorized the complete
candidate-conditioned Noise energy after the GT-only candidate oracle passed.
This implementation does not resume either failed local selector. It freezes
the proposal and step-8,530 residual basis and trains only a new `1,193,907`
parameter shared candidate scorer on the existing source-disjoint development
protocol.

No official, sealed, internal, parent-test, or frozen-660 pixels are used.

## Model contract

For every real deployable candidate, the scorer receives the observation,
proposal, candidate image, candidate-minus-proposal,
candidate-minus-observation, and that image's five structured VLM severity
levels. It predicts separate 32x32 relative log-MSE and spatial-LPIPS risks
plus uncertainty.

The deployable candidates are proposal, fidelity, perceptual, half-fidelity,
1.25x fidelity, and 75/25 and 50/50 fidelity/perceptual blends. One of four
negative roles is presented on each optimizer step: wrong expert,
under-restored external output, detail-damaged output, or repeated proposal.
Negatives supervise risk estimation but can never be selected at inference.

Training uses relative-risk regression and gap-weighted, tie-aware ranking.
Inference uses the same candidate scores, retains fidelity as the safe
baseline, and accepts another candidate only if its conservative MSE risk is
strictly lower and its conservative perceptual risk remains within the frozen
safety threshold. A freshly initialized model therefore returns fidelity
exactly.

## Static and CPU qualification

- Python compilation: pass.
- JSON parsing and Bash syntax: pass.
- Focused unit tests: `6/6` pass, including the existing oracle tests.
- Shared-scorer shape, initial fail-closed behavior, target construction,
  candidate composition, and tie-aware ranking: pass.
- Spatial LPIPS target forward/backward smoke: finite.
- Manifest SHA-256: pass.
- Frozen proposal SHA-256: pass.
- Frozen residual-basis SHA-256: pass.
- Three external negative checkpoint SHA-256 checks: pass.
- Diff whitespace check: pass.

## Next gate

Run one fresh 25-step qualification on exactly one homogeneous 8xH100 or
8xH200 node. The qualification checkpoint must not seed formal training. The
qualification is an engineering check only: finite optimization, artifact
writing, protected-data flags, initial/final selector behavior, runtime and
memory must be inspected before authorizing a fresh formal run.

Because this candidate scorer evaluates seven deployable candidates and
spatial LPIPS targets, the pre-run estimate is `.2–.6` wall-clock hours and
`1.6–4.8` allocated GPU-hours. Replace this range with measured runtime.

The eventual inference-only joint target remains PSNR
`43.424160178005695` or higher and LPIPS `.045362067760033824` or lower on
the fixed 256-record source-disjoint development subset.
