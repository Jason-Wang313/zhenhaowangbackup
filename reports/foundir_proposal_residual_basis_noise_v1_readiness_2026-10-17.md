# FoundIR Noise proposal residual basis V1 — readiness

Date: 2026-10-17 UTC

## Purpose

The completed GT oracle showed that local selection over the V2 direct
residual improves both metrics but cannot exceed `+.042452 dB/-.003946 LPIPS`.
Therefore gate learning is not the immediate bottleneck: the available
residual direction lacks enough corrective information.

This experiment freezes the existing proposal and trains a stronger residual
basis before any new energy/gate work. It uses a restoration-native Restormer
with `26,116,852` parameters instead of the prior `350,827`-parameter direct
control. Its input is degraded image, proposal, and proposal-minus-degraded.
Two independent RGB corrections are learned:

- fidelity direction: MSE plus weak LPIPS;
- perceptual direction: MSE plus stronger LPIPS;
- both: 8x8 local MSE no-harm penalty relative to the proposal.

This directly addresses the two measured failures: insufficient residual
capacity/information and an objective that did not explicitly prioritize MSE.
It does not train energy and is not yet a novelty-success claim.

## Frozen protocol

- Training: 54,585 Noise records / 1,755 source groups only.
- Validation: the same fixed 256-record source-disjoint development subset.
- Proposal: hash-bound Noise Restormer step 250, frozen.
- Hardware: one homogeneous 8xH100 or 8xH200 node, global batch 32.
- Qualification: fresh 25 steps, never reused for the formal run.
- Formal schedule after qualification: 1,706 steps/pass, five passes / 8,530
  total steps, validation after each pass.
- Gate: relative to the frozen proposal, at least `+.30 dB` PSNR and `.005`
  LPIPS reduction.
- Official, sealed, internal, and frozen-660 access: prohibited.

Static validation completed: JSON parse, Python compilation, Bash syntax,
diff check, focused forward/initialization unit test, and actual adapted
step-250 checkpoint initialization all pass. The real checkpoint preflight
reports `26,116,852` trainable parameters and the expected `noise_mse_lpips`
task/step identity.
