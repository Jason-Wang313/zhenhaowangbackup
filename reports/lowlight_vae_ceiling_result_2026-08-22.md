# FLUX.2 VAE representation-ceiling result

Date: 2026-08-22 UTC

## Outcome

The two-H100 diagnostic completed successfully with terminal status 0. Its
predeclared automatic decision is:

`completed_pixel_frequency_bypass_required`

The frozen FLUX.2 VAE is a strong low-frequency representation, but a
signal/confidence-gated pixel-frequency bypass is required before training the
restoration bridge. The bypass decision was caused by one dataset-balanced
failure: LOL-v2 real-train validation scene-balanced high-frequency
correlation was `0.89812091`, below the frozen `0.90000000` minimum.

This is a representation diagnostic on reusable non-FoundIR paired-real
development data. No official FoundIR image was opened and no FoundIR
inference was performed.

## Verified scope and provenance

- Git revision: `f64088141b69635540ca823d85c1da4fd28aff77`
- Config: `configs/lowlight_vae_ceiling_v2.json`
- Config SHA-256:
  `bd8e62409eb556cc667fb0856c0631f21259c7d05b36238c8e7ecfe4dd9d003c`
- Summary: `checkpoints/lowlight_vae_ceiling_v2/result_summary.json`
- Summary SHA-256:
  `91db40f26f60e69b21ae767775dc7e169cf9b274e3dce4296b91662e2656e515`
- Result rows: `checkpoints/lowlight_vae_ceiling_v2/result_rows.csv`
- Result-row SHA-256:
  `9d8796385f3c1de190c29c8713e5930a6d94c85682ca7463816baa886ff67938`
- Records: 174 paired records / 82 scenes / 130 exact-unique targets
- Sidecars: 130/130 complete and identity-verified
- Devices: two NVIDIA H100 80GB HBM3 GPUs
- Main run: `2026-08-22T01:57:22Z` to `2026-08-22T02:04:40Z`
  (`438` seconds)
- Independent status helper:
  `LOWLIGHT_VAE_CEILING_STATUS_PASS`

## Scene-balanced aggregate metrics

| Metric | Value |
|---|---:|
| Full-resolution PSNR | 38.448800 dB |
| Full-resolution SSIM | 0.966559 |
| Resized full-resolution LPIPS | 0.005159 |
| Center-crop PSNR | 38.256216 dB |
| Center-crop SSIM | 0.965424 |
| Center-crop LPIPS | 0.009412 |
| High-frequency correlation | 0.908125 |
| High-frequency energy ratio | 0.992530 |
| Seam-error ratio | 1.027740 |
| Mean luma bias | 0.001270 |

The aggregate result is excellent, and SICE passed every threshold. The
decision remains fail-closed because every dataset-balanced metric had to pass;
LOL-v2 high-frequency correlation missed its floor by `0.00187909`.

## Architecture consequence

The next restoration implementation must use both:

1. the explicit bounded photometric head already prioritized by the
   `+15.6853 dB` global-photometric oracle gain; and
2. a conservative signal-to-noise/confidence-gated Laplacian or pixel residual
   bypass around the VAE.

The bypass must start as identity/conservative fusion and allow the learned
latent branch to replace only low-confidence detail. This is not evidence for
discarding FLUX.2 or substituting a conventional restoration model. It is an
information-preserving path inside the FLUX/Z-Image restoration adapter.
