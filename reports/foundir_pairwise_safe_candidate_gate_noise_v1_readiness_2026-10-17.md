# Pairwise safe candidate gate v1 readiness — 2026-10-17

## First-principles change

The previous model failed because it estimated seven absolute risks plus its
own uncertainty, then selected the minimum. This created overconfident
multiple-choice errors: even its highest-confidence accepted regions damaged
PSNR.

The new `1,199,043`-parameter model asks the deployment question directly for
each alternative versus fidelity:

1. Does this candidate improve local MSE by more than the frozen margin?
2. Does it cause perceptual harm beyond the frozen margin?
3. What is its relative local log-MSE change?

It uses balanced binary supervision, tie exclusion, gain regression and
within-image candidate ranking. Only the six actual deployable alternatives
are trained. The failed scorer and its weights are not reused. Obvious
external negative models are excluded because they do not match the
deployment decision distribution.

## Safe deployment

Candidate gain/harm probability thresholds are selected only on 17 source-
disjoint calibration groups. They are applied unchanged to 17 separate
evaluation groups. If calibration finds no policy that improves PSNR within
the perceptual proxy bound, deployment accepts zero regions and exactly
returns fidelity.

## Readiness

- Python compilation, JSON parsing and Bash syntax: pass.
- Focused tests: `7/7` pass.
- Fresh initialization returns fidelity exactly: pass.
- Pairwise gain and harm target direction: pass.
- Calibration no-solution fallback: pass.
- Proposal and residual-basis hashes: pass.
- Qualification output path: fresh.
- Protected-data flags: all false.

The next action is a fresh 25-step engineering qualification on exactly one
homogeneous 8xH100/H200 node. Estimated runtime is `.08–.25` wall-clock hours
and `.64–2.0` allocated GPU-hours; replace this estimate with measured data.
The qualification checkpoint must not seed formal training.
