# Minimal VLM-token MoE image-only formal: terminal result

Date: 2026-10-25 UTC

## Outcome

The `image_only` arm completed its frozen 1,000-step schedule normally, but it
failed scientifically. The trained construction is not a usable ordinary
restoration MoE: its restoration output is substantially worse than the
degraded input, and its router has collapsed almost completely to one expert.

This is not a training crash, insufficient GPU allocation, or an evaluation
artifact. Step 500 already exhibited the same collapse and poor restoration;
another 500 steps did not resolve either problem.

## Runtime

- Status: `matched_formal_arm_complete`
- Steps: `1,000/1,000`
- Hardware: `8 x NVIDIA H100 80GB HBM3`
- Wall-clock hours: `.6436424450461022`
- Allocated GPU-hours: `5.149139560368818`
- Peak allocated memory per rank: `12,662,946,304` bytes
- Mean last-five training loss: `.9347534656524659`
- OOM/NCCL/runtime failures: none
- Protected-data access flags: all false

## Fixed holdout result

450 source-disjoint training-only records, 90 per degradation:

- Degraded input: `23.71294072670594` PSNR / `.3023991875624698` LPIPS
- Image-only MoE output: `15.634318159133873` PSNR / `.34884399509264363` LPIPS
- Change versus input: `-8.078622567572069` dB / `+.046444807530173826` LPIPS

The system worsens both metrics. By degradation, output PSNR is:

- Blur: `17.690864679177963`
- Haze: `14.85009344413457`
- Low-light: `8.583464338405483`
- Noise: `18.11601077864781`
- Rain: `18.931157555303532`

## Router collapse

Mean holdout expert weights are approximately:

- Expert 0: `5.887322805777887e-7`
- Expert 1: `.9999994105762906`

Mean routing entropy is `8.626073475094624e-6`, compared with `ln(2) ~= .693`
for balanced use. Thus the nominal two-expert model behaves as a one-expert
model. The step-500 calibration had the same pattern: expert 1 weight
`.9999990116` and output `15.9184/.345086`.

## First-principles diagnosis

The experiment replaced one MLP in raw pretrained FLUX.2 with two identical
clones and trained only that bank/router on the restoration objective. It did
not start from an already working restoration MoE. With identical experts and
no mechanism preserving meaningful utilization, the easiest optimization path
was to select one expert almost exclusively. More importantly, adjusting one
MLP bank inside an otherwise frozen raw generator was insufficient to create a
competent five-degradation restorer.

The additive VLM router branch can change expert scores, but it cannot create
missing restoration competence. A relative VLM-versus-image comparison on this
broken base would not answer the advisor's intended question about improving a
working MoE.

## Identity bindings

- Config SHA-256: `00afcfe013fad31c61f0f477340d65921bc5d7c7e325439c33ea794ebe166379`
- Final checkpoint SHA-256: `7883d970de6e4110a82abbcad076f29f148b498477fe7039bf590ec74b26956b`
- Final result: `checkpoints/foundir_vlm_token_moe_matched_formal_v1/image_only/result.json`
- Holdout: `checkpoints/foundir_vlm_token_moe_matched_formal_v1/image_only/holdout_step_001000.json`
- Calibration: `checkpoints/foundir_vlm_token_moe_matched_formal_v1/image_only/calibration_step_001000.json`

## Decision boundary

Do not report this as a working MoE and do not automatically launch the
VLM-assisted arm. The next valid experiment must first supply an actually
working restoration base/expert bank and prevent collapse. Only then does
image-only versus VLM-assisted routing become a meaningful matched comparison.
