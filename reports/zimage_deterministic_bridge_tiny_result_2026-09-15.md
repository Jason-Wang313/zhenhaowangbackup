# Z-Image Layer-1 deterministic bridge tiny result — 2026-09-15

## Outcome

**TERMINAL TINY-GATE SCIENTIFIC FAILURE — DO NOT FULL-SCALE THIS RECIPE.**

The pure Z-Image Base 6B Layer-1 run completed normally on eight homogeneous
H100 80GB GPUs at `2026-09-15T13:17:54Z` (terminal status 0). It trained all
1,000 planned optimizer steps in about 12 minutes and finished the complete
fixed evaluation in about 19 minutes wall time. There was no traceback, OOM,
NCCL error, nonfinite value, or protected-data access.

This experiment is a deterministic `LQ latent -> GT latent` residual/bridge
teacher with three-scale, multi-depth LQ conditioning, rank-64 FP32 DiT LoRA,
and latent/RGB/LPIPS fidelity losses. It has no independent restoration branch,
distillation, decoder tuning, Qwen degradation conditioning, or mixed
degradation.

## Immutable identities

- Config: `configs/foundir_blur_only_zimage_deterministic_bridge_tiny_v1.json`
- Config SHA-256: `5a14b71e60ef8400cf87281b4991b15b897dad32828841638c24c0cb96e141cb`
- Result: `checkpoints/foundir_blur_only_zimage_deterministic_bridge_tiny_v1/result.json`
- Result SHA-256: `bee7290a15687b6f0fb2a57743cc92cdab9bd55e72efb9935a785bb8128aa403`
- Step-250 checkpoint SHA-256: `799196fd0058cc11755d351933f14b8a193763552660f9733f6c92e4fdc0fbff`
- Step-500 checkpoint SHA-256: `6321393cafc0cf186bf37331ab8691a5042a81e424e0b7276ab21f8f097f321d`
- Step-1000 checkpoint SHA-256: `5196fa78f5ce9ae614ea896395d0f36ce67adec6cedc58684d7e5f08d29c138c`
- Code revision: `4e0f37111f6e60b3815e5f438acc5ef803338811`

Every step 0/250/500/1000 split contains exactly 16 per-image PNG triptychs,
16 bound per-image JSON sidecars, and one aggregate metrics JSON: 128 PNGs and
128 sidecars across the fixed 16-train/16-source-disjoint-development protocol.

## Metrics

| Step | Split | Input PSNR / LPIPS | Output PSNR / LPIPS | PSNR gain / LPIPS improvement |
|---:|---|---|---|---|
| 0 | train | 24.7232 / .3085 | 24.6421 / .3114 | -.0811 / -.0028 |
| 250 | train | 24.7232 / .3085 | 22.7934 / .2776 | -1.9298 / +.0309 |
| 500 | train | 24.7232 / .3085 | 24.4854 / .2874 | -.2378 / +.0211 |
| 1000 | train | 24.7232 / .3085 | **25.8080 / .2943** | **+1.0848 / +.0142** |
| 0 | development | 23.9045 / .3264 | 23.8900 / .3301 | -.0145 / -.0037 |
| 250 | development | 23.9045 / .3264 | 21.9166 / .3048 | -1.9879 / +.0215 |
| 500 | development | 23.9045 / .3264 | 22.1493 / .3248 | -1.7552 / +.0016 |
| 1000 | development | 23.9045 / .3264 | **22.1861 / .3437** | **-1.7184 / -.0173** |

The train gate required PSNR >=26.4, LPIPS <=.16, both better than input, and
acceptable visuals. Step 1,000 exceeds the aggregate train input on both
metrics, demonstrating real restoration learning, but misses PSNR by .5920 dB
and LPIPS by .1343. Only 11/16 train images improve PSNR and 10/16 improve
LPIPS. The descriptive source-disjoint development set degrades on average;
only 2/16 improve PSNR and 8/16 improve LPIPS.

## Visual review

Visual review fails. Step-1,000 outputs broadly contain granular/checker-like
high-frequency texture, edge echoes, and local structural distortion. Strongly
blurred examples sometimes become sharper but do so through false texture;
near-clean inputs are often damaged. Representative failures include
`foundir_train_0042483`, `foundir_train_0001150`, `foundir_train_0091486`, and
development record `foundir_train_0100870`.

## Decision

Do not full-scale, distill, or tune the decoder from this checkpoint. More data
cannot repair failure to cleanly overfit 16 images, and the student/decoder
cannot add upstream restoration information absent from the teacher. Preserve
all artifacts. A further route requires a new formulation decision outside
this bounded Layer-1 tiny goal; no independent restoration branch is implied or
authorized by this report.
