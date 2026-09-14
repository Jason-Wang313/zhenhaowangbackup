# Blur-only Z-Image paired-flow teacher v2 result — 2026-09-14

> **DIAGNOSTIC ONLY / 非主线。** This is a fixed Blur-only teacher
> learnability gate, not a five-type FLUX.2 mainline or official FoundIR-test
> result.

## Outcome

The corrected Base-native paired latent-flow v2 completed 400/400 optimizer
steps on homogeneous 8x `NVIDIA H100 80GB HBM3` with
`flash_attention_2`. Training and evaluation ended normally with status 0.
The automatic teacher gate failed: LPIPS improved, but PSNR did not.

| Split / state | PSNR input -> output | LPIPS input -> output |
|---|---:|---:|
| fixed train / step 0 | 24.7232 -> 14.5875 | 0.3085 -> 0.6186 |
| fixed development / step 0 | 23.9045 -> 13.0060 | 0.3263 -> 0.6930 |
| fixed train / step 400 | 24.7232 -> 23.2828 | 0.3085 -> 0.2415 |
| fixed development / step 400 | 23.9045 -> 22.6414 | 0.3263 -> 0.2692 |

Step 400 improved LPIPS on 13/16 records in each split, but improved PSNR on
only 2/16 train and 1/16 development records. Mean PSNR fell `1.4404 dB` on
train and `1.2632 dB` on development.

Fixed triptych review confirms that v2 repaired v1's blue/green noise,
low-frequency block, content, and geometry collapse. It preserves the scene
substantially and often sharpens the input, but remains below GT detail and can
introduce grain, over-sharpening, and luminance changes. It unnecessarily
damages near-clean examples. Train `foundir_train_0091486` changes
`36.883 -> 26.250 dB` (`0.047 -> 0.151` LPIPS); development
`foundir_train_0062371` changes `27.230 -> 21.971 dB`
(`0.040 -> 0.152` LPIPS).

## Exact identities

- Config:
  `configs/foundir_blur_only_zimage_teacher_paired_flow_overfit_v2.json`,
  SHA-256 `d6f5946fcd75050a17d96c093f7590552f89d14caaf7dce0c659fe35fcb2fbb8`.
- Step 200 SHA-256:
  `95844e51228046664b2647f2a3f34800f9455f1b171bbd79f0627c9e2e7c73c5`.
- Step 400 SHA-256:
  `2a3e9b3392dfea20f9e6e869febce1ff99bcaa751e317e247c908f683a4ecd39`.
- Training result SHA-256:
  `b941e313b15f233ad7bc938ab9f558f83da2ed35c8e784c142c8abf4cc39d2c2`.
- Evaluation result:
  `checkpoints/foundir_blur_only_zimage_teacher_paired_flow_overfit_v2/evaluation/result.json`,
  SHA-256 `1c1bdf59c03dfd29773fd122cfb1a101bc6b45542816de2b9c7b6b053d9bd890`.
- Governance hold:
  `zimage_blur_teacher_paired_flow_tiny_overfit_v2-ea895f0c24c75257`,
  SHA-256 `c749f28a843c718fd474d714fa293a1ab77639a718d2d970116f51a1152abcd7`.
- Matching continuation decision SHA-256:
  `515e869861d305acb0a747c9b15c26dd1fb2abd4a5c1934ada035fdc0758809a`.

Sealed internal and official FoundIR test access are both false.

## Decision

Do not start full teacher training or distillation from v2. The user explicitly
authorized continued best-effort work on the same route without another
advisor request. The next bounded gate initializes an isolated v3 run from the
immutable step-400 LoRA, resets the optimizer, aligns training samples to the
actual 50-step shift-6 inference grid, removes the empirical T2I timestep loss
weight, and evaluates cumulative steps 800 and 1200 on the unchanged fixed
train/development crops. Decoder tuning remains gated until after teacher and
distillation evidence establishes a decoder/latent ceiling.
