# Blur-only Z-Image teacher architecture audit — 2026-09-13

## Decision

Use the released undistilled `Tongyi-MAI/Z-Image` foundation as the 50-step
teacher backbone and condition it on the aligned degraded image through a
frozen Z-Image ControlNet during the first tiny-overfit gate. Train DiT LoRA
rank 32. Do not start distillation or decoder fine-tuning during this gate.

## Evidence

- Advisor instruction: `换换Z Image吧。先训teacher，然后蒸馏一下，少步/单步。如果点还是不行，微调一下decoder。`
- Official Z-Image source revision:
  `26f23eda626ffadda020b04ff79488e1d72004cd` (Apache-2.0).
- DiffSynth-Studio source revision:
  `32ef37e4fc38400f9fe14b1affd335dad9cdf50d` (Apache-2.0).
- Released foundation revision:
  `Tongyi-MAI/Z-Image@04cc4abb7c5069926f75c9bfde9ef43d49423021`
  (Apache-2.0).
- Architecture: 6B S3-DiT, 30 single-stream layers, width 3,840, 30 attention
  heads, 16-channel VAE latent, Qwen3 text encoder, undistilled CFG-capable
  28--50-step model.
- The public `Z-Image` checkpoint is text-to-image. The official
  `Z-Image-Omni-Base` and `Z-Image-Edit` checkpoints are still marked
  unreleased, so neither is a reproducible current dependency.
- DiffSynth-Studio exposes a Z-Image ControlNet path that encodes a full-color
  control image with the same VAE and injects residual hints into two refiner
  blocks and 15 DiT layers. Base and Turbo publish identical DiT dimensions,
  so the interface is structurally compatible with the undistilled Base.
- Selected initialization is the non-distilled
  `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` ControlNet at revision
  `5155fc56d17821007d6f62ac192c09e0f0e72016` (Apache-2.0). It is frozen in
  the first gate; it supplies the LQ image path while DiT LoRA learns the
  Blur-only restoration mapping.

## Local runtime verification

The complete pinned artifacts were loaded through the actual training and
evaluation constructors on CPU on 2026-09-13. This verified the runtime tensor
schemas rather than only filenames and hashes:

- the loaded classes were `ZImageDiT`, `ZImageTextEncoder`,
  `FluxVAEEncoder`, `FluxVAEDecoder`, and `ZImageControlNet`;
- the training pipeline included `ZImageUnit_PAIControlNet`, so the aligned LQ
  image reached the ControlNet path;
- rank-32 LoRA injection produced 476 trainable tensors / 79,380,480 trainable
  parameters, all under the DiT and all LoRA-only;
- a structural 64x64 CPU run using PyTorch SDPA and fixed seed 0 produced a
  finite flow-matching loss of `2.9709219932556152`; backward produced finite
  gradients for all 476 trainable tensors and nonzero first-step gradients for
  the expected 238 `lora_B` tensors;
- the exact checkpoint export naming used by the trainer hot-loaded into all
  238 evaluator LoRA targets with zero mismatches, and `clear_lora()` removed
  them cleanly;
- a two-step CPU evaluator-interface smoke exercised CFG 4.0, the LQ
  ControlNet path, VAE decoding, PIL conversion, PSNR, and LPIPS and returned a
  64x64 RGB image.

The 64x64/two-step values are interface diagnostics only and are deliberately
not quality evidence. The frozen quality protocol remains 512x512 and 50
steps on the fixed train/development crops. CUDA FlashAttention, DDP, optimizer
updates, and quality behavior still require the authorized 8-GPU tiny-overfit
run.

## Scientific boundary

This is a Blur-only development diagnostic, not a five-type final system and
not an official FoundIR-test result. It uses the same frozen 16 training crops
and separate 16 development crops used by the earlier code sanity check. The
crop coordinates are shared exactly between each LQ/GT pair. Sealed internal
and official FoundIR test remain inaccessible.

The tiny-overfit gate asks only whether PSNR and LPIPS improve over the input
on the memorized 16 training crops and whether fixed triptychs are visually
valid. It does not claim generalization and does not use SOW thresholds.

## Ordered next gates

1. Run the 16-pair teacher tiny-overfit and inspect fixed triptychs.
2. If learnability and visuals pass, train the full Blur-only teacher and
   evaluate its frozen development curve.
3. Distill the accepted teacher to few-step, then test single-step.
4. Consider decoder fine-tuning only if distilled outputs remain inadequate
   and a decoder/latent ceiling audit identifies the decoder as a bottleneck.
