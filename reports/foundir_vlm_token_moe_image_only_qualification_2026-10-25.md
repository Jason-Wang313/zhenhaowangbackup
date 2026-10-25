# Minimal VLM-token MoE: image-only engineering qualification

Date: 2026-10-25 UTC

## Outcome

The frozen `image_only` arm completed all 25 optimizer steps successfully on
one homogeneous 8x NVIDIA H100 80GB node. This establishes that the ordinary
image-routed MoE control can load, train, synchronize, and save safely. It is
an engineering result only; it does not measure restoration quality and is not
evidence that VLM guidance helps.

## Frozen result

- Status: `minimal_vlm_token_moe_engineering_qualification_pass`
- Steps: `25/25`
- World size: `8`
- Global batch size: `32`
- Trainable parameters: `170,662,658`
- Mean loss over the last five steps: `1.1534720182418823`
- Mean routing entropy: `.16084940331958933`
- Maximum expert-weight sum error: `1.1920928955078125e-07`
- Maximum VLM logit adjustment: `0.0` (required for this control)
- VLM output-head L1 update: `0.0` (required for this control)
- Peak allocated memory per rank: `12,653,083,136` bytes
- Wall-clock hours: `.0563902092075053`
- Allocated GPU-hours: `.4511216736600424`
- Protected-data access flags: all false

## Identity bindings

- Config SHA-256: `ca2c622c3062652895607b6276ca568fa8a7c0c917d9ac6be847562b3fe735a7`
- Checkpoint SHA-256: `4dbf51c1d63bedd5afda796735a201db352252949074282fcb3437f33329d294`
- Result: `checkpoints/foundir_vlm_token_moe_minimal_qualification_v1/image_only/result.json`
- Checkpoint: `checkpoints/foundir_vlm_token_moe_minimal_qualification_v1/image_only/step_000025.safetensors`

The checkpoint is an engineering artifact and must not initialize formal
training.

## Next action

Run the independently initialized, matched `vlm_assisted` 25-step
qualification. Do not launch formal training automatically. The two arms must
share the same backbone, expert bank, data, optimizer, seed, and compute; only
the small VLM-to-router score adjustment differs.
