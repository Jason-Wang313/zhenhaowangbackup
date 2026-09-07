# FoundIR five-type FLUX.2 pixel-refinement mainline v3 readiness — 2026-09-07

## Why v3 is required

The no-JPEG v2 run passed its corrected runtime preflight and completed the
44-record FoundIR-training-only VAE ceiling. The frozen VAE was strong overall,
but the predeclared ceiling returned `pixel_refinement_required`:

- Noise high-frequency correlation: `0.89056` versus `0.90000`;
- Low-light high-frequency correlation: `0.86160` versus `0.90000`;
- Low-light plus Haze: PSNR `34.72370 dB` versus `35.00000 dB`, and
  high-frequency correlation `0.82381` versus `0.90000`.

The v2 result is preserved at
`checkpoints/foundir_five_type_flux2_dit_v2/vae_ceiling_result.json`, SHA-256
`1702e40a8938fcbf659f0442a6bce51e2d87d69aa261e6c335da3dffa866eba9`.
No VAE parameter is unfrozen, no Qwen export ran, and no FLUX optimizer step
started in v2.

V3 reuses this measurement instead of repeating the identical 44 VAE round
trips. The reuse fails closed unless the source v2 config and result hashes,
current FoundIR manifest hash, VAE weights, VAE thresholds, target-selection
seed, five-type scope, record count, category coverage, and protected-test flag
all match exactly.

## Minimal architecture response

V3 keeps FLUX.2 Klein 4B, its frozen VAE, frozen native Qwen3 text encoder,
native reference-image tokens, and DiT LoRA rank/alpha 32/32. It adds one
33,251-parameter bounded pixel refiner after VAE decoding. The refiner:

- receives only the decoded FLUX restoration, aligned LQ image, and their
  difference;
- is initialized to exact identity;
- can change each normalized RGB value by at most `0.25` in the `[-1, 1]`
  representation;
- uses six lightweight local NAF-style blocks and has no independent image
  generation path;
- is supervised by paired RGB Charbonnier and fixed high-frequency losses,
  with a residual-size penalty;
- is saved in the same SafeTensors checkpoint as the FLUX LoRA and loaded as
  one deployable restoration system.

The 128-pair overfit gate trains the refiner jointly with the FLUX LoRA and
evaluates the actual six-step decoded-and-refined output. Its 32 evaluation
pairs are interleaved across all 11 admitted FoundIR categories rather than
taken from the category-sorted file prefix. The gate separately requires a
positive refiner PSNR gain and a high-frequency-correlation gain on Noise,
Low-light, and Low-light+Haze, the categories that triggered the VAE decision.
Full training applies the RGB/refinement objective every eighth optimizer step,
preserving the frozen 1.8-million-presentation curriculum and global batch size
while controlling wall-clock cost.

## Scope and stop policy

Only Blur, Noise, Rain, Haze, and Low-light FoundIR training categories are
admitted, including approved mixed categories. JPEG remains excluded. Qwen is
frozen and only its actual five-question predictions are used; there are no
oracle, generic, or shuffled/wrong condition evaluations. The official FoundIR
test remains forbidden until the complete internal protocol is frozen and the
user explicitly approves that final evaluation.
