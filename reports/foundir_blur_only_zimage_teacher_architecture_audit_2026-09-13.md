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

The production CUDA preflight additionally executes a finite BF16 attention
operation on every visible GPU with DiffSynth's selected backend and records
the backend name. Distributed trainer/evaluator artifact verification performs
one complete SHA-256 pass on rank 0 per phase while all other ranks verify the
required file set and synchronize, preventing redundant multi-rank reads
without weakening model-identity enforcement.

## Scientific boundary

This is a Blur-only development diagnostic, not a five-type final system and
not an official FoundIR-test result. It uses the same frozen 16 training crops
and separate 16 development crops used by the earlier code sanity check. The
crop coordinates are shared exactly between each LQ/GT pair. Sealed internal
and official FoundIR test remain inaccessible.

The tiny-overfit gate asks only whether PSNR and LPIPS improve over the input
on the memorized 16 training crops and whether fixed triptychs are visually
valid. It does not claim generalization and does not use SOW thresholds.

## Tiny-overfit terminal result — 2026-09-14

The authorized 8-GPU run completed all 400 optimizer steps and all four fixed
evaluation snapshots. Training ran on eight `NVIDIA H100 80GB HBM3` devices
with `flash_attention_2`, global batch 8, and 79,380,480 trainable LoRA
parameters. The 400-step training wall time was `433.1167 s`; telemetry reached
98--100% peak utilization and 40,400--41,014 MiB peak GPU memory use on each
GPU. The launcher returned status 3 only after training and evaluation because
the then-current advisor emitter recognized only
`protected_foundir_test_accessed`, while this result correctly recorded
`official_foundir_test_accessed: false`.

- step-200 checkpoint SHA-256:
  `8412dfd3da2dec67d56bbec94556edb4fa38276c007c8a3b8aaebf27e6a18a38`;
- step-400 checkpoint SHA-256:
  `2505c79fd554e76297eaaaa6be0d6e5aabbd75a702a2c11666c19df87697177b`;
- immutable evaluation result SHA-256:
  `5e9b8b4a715382bb4d8d5a3e233314da9c34b7b3d7e4c4031663160dbea0e651`;
- training loss was `1.770904` at step 1 and `0.413092` at step 400, but the
  image-restoration gate failed both metrics.

| Role / state | Input PSNR | Output PSNR | Input LPIPS | Output LPIPS | Improved records |
|---|---:|---:|---:|---:|---:|
| memorized train / step 0 | 24.7232 | 7.6775 | 0.3085 | 0.8538 | 0/16 PSNR, 0/16 LPIPS |
| memorized train / step 400 | 24.7232 | 9.3654 | 0.3085 | 0.8408 | 0/16 PSNR, 0/16 LPIPS |
| source-disjoint visual / step 0 | 23.9045 | 8.1023 | 0.3263 | 0.8460 | 0/16 PSNR, 0/16 LPIPS |
| source-disjoint visual / step 400 | 23.9045 | 9.0932 | 0.3263 | 0.8949 | 0/16 PSNR, 0/16 LPIPS |

All 64 expected fixed triptychs are present: 16 train plus 16 source-disjoint
development images at both step 0 and step 400. Direct visual inspection found
step 0 to be blue-grey noise. Step 400 recovered some coarse spatial layout,
which shows that the LQ path is not wholly disconnected, but outputs retain
severe green/yellow casts, incorrect content, low-frequency blocks, and
collapsed texture. This is not Blur restoration, so the human visual gate also
fails. Full teacher training is prohibited by the predeclared ordering.

## Read-only failure diagnosis

The available evidence narrows the fault domain but does not prove one root
cause:

- The released ControlNet is explicitly trained for `Z-Image-Turbo`; the
  current experiment attaches it to the undistilled Base because their tensor
  shapes match. DiffSynth's own reference example pairs this ControlNet with
  Turbo. Shape compatibility does not establish feature or trajectory
  compatibility, making this the strongest architectural incompatibility
  candidate.
- The pinned official Base scheduler config uses static shift `6.0`
  (`use_dynamic_shifting: false`), whereas DiffSynth's `Z-Image` scheduler uses
  static shift `3.0` when `sigma_shift` is omitted. Training and evaluation in
  this gate therefore used shift 3 rather than the checkpoint's published
  shift 6 trajectory.
- The native Base implementation and DiffSynth both negate the raw transformer
  output before Euler integration, and their normalized timestep-to-embedding
  scaling is equivalent. The audit found no direct prediction-sign or
  timestep-scale inversion.
- Native CFG computes `positive + scale * (positive - negative)`, while
  DiffSynth computes `negative + scale * (positive - negative)`. At configured
  scale 4 these are not identical. Training used the positive path only
  (`cfg_scale=1`), while evaluation used DiffSynth CFG 4.
- Step 0 already collapses before any trained LoRA is loaded. Step 400's coarse
  geometry and the verified 238-target LoRA hot-load exclude a simple claim
  that the condition path or checkpoint reload was entirely absent, but they
  cannot distinguish ControlNet semantics, scheduler trajectory, CFG, or their
  interaction.

Resolving these candidates requires new bounded inference/training controls;
those are a new diagnostic experiment and are not authorized under the active
failure hold.

## Gate disposition and exact next action

The automatic metric gate and visual gate both fail. Do not start full teacher
training, few-step/single-step distillation, or decoder tuning. The active
governance hold is
`zimage_blur_teacher_tiny_overfit-a6becc1eff058fec`. Send the failure report to
the advisor and request authorization for one bounded corrected tiny diagnostic
that isolates Base/ControlNet compatibility, scheduler shift, CFG, and
prediction trajectory. Decoder work remains behind both the teacher and
distillation gates.
