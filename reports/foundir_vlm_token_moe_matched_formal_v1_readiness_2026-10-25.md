# Minimal VLM-token MoE matched formal readiness

Date: 2026-10-25 UTC

## Question

Does adding the frozen VLM feature through one small score projector improve
the same restoration DiT-MoE over its ordinary image-only router?

This experiment does not add LoRA experts, a second MoE, hard priors, energy
refinement, atomic composition, or block/time schedules.

## Matched arms

1. `image_only`: image features produce the two expert weights; VLM adjustment
   is exactly zero.
2. `vlm_assisted`: the same image router and expert bank are used, while the
   7,168-dimensional frozen Qwen visual feature passes through the small MLP
   and adds a correction to the two router logits.

Both arms start independently from the same pretrained FLUX.2 Klein 4B
weights. Neither 25-step qualification checkpoint is allowed. Both use the
same generic text context, so diagnosis cannot leak through the prompt.

## Frozen scale and evaluation

- Fit: 2,250 records from training-only sources.
- Calibration: 450 source-disjoint records.
- Holdout: 450 source-disjoint records.
- Training: 1,000 optimizer steps, global batch 32.
- Exposure: 32,000 fit presentations, approximately 14.2 fit-set passes.
- Training crop: 256x256.
- Evaluation crop: deterministic 256x256.
- Calibration checkpoints: steps 500 and 1,000.
- Holdout access: fixed final step 1,000 only; no holdout checkpoint selection.
- Inference: six matched degraded-anchored flow steps.
- Metrics: PSNR, AlexNet LPIPS, per-degradation means, expert weights, routing
  entropy, and VLM adjustment magnitude.

The primary hypothesis passes only if VLM-assisted improves both mean holdout
PSNR and mean holdout LPIPS over image-only. Magnitudes, paired bootstrap
intervals, record win rates, and all five degradation-specific results must
still be reported; a tiny numerical pass must not be described as a large
effect.

## Runtime and safety

Measured 25-step qualifications took `.046-.056` wall hours. Including the
larger training schedule and three evaluations, the conservative frozen budget
is `1.3-2.5` wall hours and `10.4-20.0` allocated GPU-hours per arm on one
homogeneous 8xH100/H200 node.

The launcher logs stdout to a shared arm-specific file and supports resume only
from that same formal arm's hash-bound checkpoint. It refuses qualification
checkpoints, does not auto-launch the other arm, and does not auto-launch any
later experiment. Official, sealed, parent-test, and internal-test access flags
remain false.

## Qualification evidence

- Python compilation: pass.
- Config JSON validation: pass.
- Launcher shell syntax: pass.
- Formal protocol unit tests: 4/4 pass.
- `image_only` CPU/data preflight: pass.
- `vlm_assisted` CPU/data preflight: pass.
- Fit/calibration/holdout counts: 2,250/450/450.
- Prompt context: `[1,512,7680]`; IDs: `[1,512,4]`.
- Protected data accessed: false.

## Identity bindings

- Config SHA-256: `00afcfe013fad31c61f0f477340d65921bc5d7c7e325439c33ea794ebe166379`
- Trainer SHA-256: `0540ff5f2e366ff5e6bfae3618d8286e7aae4918a4b9d459d035218e4c8bddac`
- Launcher SHA-256: `95cc42fbbc6e14fe26badec1c15f47fac26938f23f3ebe348228c64d826c8aae`
- Comparator SHA-256: `585bf3c9d558b123b31720e3159c4c4ed7ccd4879baa1a01c58fa825e687c52b`
- Manifest SHA-256: `af6a96dfaa21982535bc77b490f4c521960767afd5b5cf4ded4f9c4da2b749c7`

## Exact next action

Run only the fresh `image_only` arm first. Inspect normal model loading, finite
step progression, step-500 checkpoint/evaluation, and final artifacts. Then run
the independently initialized `vlm_assisted` arm. Finally execute the frozen
paired comparator. Do not interpret training loss as restoration performance.
