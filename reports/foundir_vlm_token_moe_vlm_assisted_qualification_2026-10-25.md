# Minimal VLM-token MoE: VLM-assisted engineering qualification

Date: 2026-10-25 UTC

## Outcome

The independently initialized `vlm_assisted` arm completed all 25 optimizer
steps on one homogeneous 8x NVIDIA H100 80GB node. The small VLM-to-router
branch received gradients, updated, and produced nonzero router-logit
adjustments while the shared expert weights remained normalized. This proves
engineering viability only; it does not establish restoration improvement.

## Frozen result

- Status: `minimal_vlm_token_moe_engineering_qualification_pass`
- Steps: `25/25`
- World size: `8`
- Global batch size: `32`
- Trainable parameters: `172,512,772`
- Mean loss over the last five steps: `1.1534493446350098`
- Mean routing entropy: `.15699553223017573`
- Maximum expert-weight sum error: `1.1920928955078125e-07`
- Maximum VLM logit adjustment: `.75`
- VLM output-head L1 update: `.78411865234375`
- Peak allocated memory per rank: `12,669,009,408` bytes
- Wall-clock hours: `.046006957962219736`
- Allocated GPU-hours: `.3680556636977579`
- Protected-data access flags: all false

## Identity bindings

- Config SHA-256: `ca2c622c3062652895607b6276ca568fa8a7c0c917d9ac6be847562b3fe735a7`
- Checkpoint SHA-256: `eb120a62bfc5d888635adaed4fb15098034fe909be5df90540268230203cdc4f`
- Result: `checkpoints/foundir_vlm_token_moe_minimal_qualification_v1/vlm_assisted/result.json`
- Checkpoint: `checkpoints/foundir_vlm_token_moe_minimal_qualification_v1/vlm_assisted/step_000025.safetensors`

The checkpoint is an engineering artifact and must not initialize formal
training.

## Honest comparison with the engineering control

The image-only and VLM-assisted last-five training losses are respectively
`1.1534720182418823` and `1.1534493446350098`. This tiny difference is not a
quality metric, the runs are only 25 steps, and no holdout restoration
evaluation was performed. It must not be cited as evidence that VLM guidance
improves restoration.

## Next action

Before a longer GPU launch, freeze the matched scientific protocol: fresh
initialization for both arms, sufficient training exposure, identical data and
compute, and evaluation on the source-disjoint calibration/holdout roles using
restoration metrics plus routing behavior. Do not initialize from either
qualification checkpoint and do not auto-launch formal training.
