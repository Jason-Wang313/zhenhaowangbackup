# Z-Image optimization root-cause audit

DIAGNOSTIC ONLY / 非主线. No official FoundIR or mainline success is claimed.

The user requested a deeper audit after unsuccessful teacher, endpoint student,
and decoder-tail experiments, then said execute. Existing authorization permits
bounded work on the same isolated-Blur restoration route without another advisor
round. This diagnostic changes no deployed architecture and uses training GT only.

## Findings established from code and terminal artifacts

- Same frozen crop coordinates are applied to LQ and GT; image hashes, geometry,
  isolated-Blur labels, and source-disjoint train/development roles are checked.
  This rules out a crop-coordinate code mismatch, not every possible source-data
  registration imperfection.
- Paired-flow derivative is LQ-GT and descending integration subtracts velocity.
  The local Base path uses model_fn_z_image_turbo by a structural dispatch
  (`siglip_embedder is None`); the function name is not evidence of Turbo weights.
  Its actual time embedding is 1000-timestep and its returned velocity is negated,
  consistent with the previously audited native sign convention.
- V3 optimized BF16 LoRA/AdamW; small updates can be rounded away. Endpoint v1
  uses FP32 LoRA/state and still has limited train fidelity. Numerical hazard is
  established, sole causation is not. The later decoder experiment exactly
  reproduces endpoint metrics after restoring its FP32 tensors.
- The deployed endpoint forward is the same LQ-latent -> predicted residual ->
  frozen decode mapping used for endpoint training. Nonreentrant checkpointing
  preserves gradients. Real CPU smoke showed finite gradients, not convergence.
- GT VAE roundtrip is about 41 dB. Decoder tail training improves LPIPS but harms
  PSNR and reconstruction retention; this does not establish a decoder ceiling.
- Endpoint loss combines RGB MSE, GT-latent MSE, LPIPS, and weak teacher imitation
  on only three training records. Scalar loss magnitudes do not establish the
  gradient balance or whether these objectives conflict.
- Sixteen fixed training crops cannot establish generalization. Requiring an
  unseen development gain from every tiny-overfit experiment confuses learning
  capability with data coverage. Persistent training-image artifacts nevertheless
  justify examining optimization before full-data training.

## Frozen discriminating experiment

Use two already selected training records: hard motion blur 0042483 and
near-clean 0091486. Both have rejected teacher targets, so the existing mixed
loss on these records is exactly RGB MSE + .1 latent MSE + .05 LPIPS.
Initialize every arm from immutable endpoint step 400, with original frozen VAE
and exact FP32 LoRA. Run four arms independently for each record:

1. Optimize the predicted latent itself with RGB MSE (LR .01).
2. Optimize the predicted latent itself with the existing mixed loss (LR .01).
3. Optimize original DiT LoRA with RGB MSE (LR 5e-5).
4. Optimize original DiT LoRA with the existing mixed loss (LR 5e-5).

One rank owns one arm/record; all eight GPUs run independent single-image
optimizations, not DDP averaging across conflicting objectives. AdamW FP32,
weight decay zero, 200 steps, snapshots 0/1/20/100/200. Frozen baseline must
reproduce the previous endpoint record within .02 dB/.002 LPIPS. Record actual
parameter updates and initial per-term latent-gradient norms/cosines, raw RGB
loss, quantized PSNR/LPIPS, latent error, saturation fraction, and GT VAE ceiling.
Save fixed Input|Output|GT images and final trainable tensors with hashes.

Latent optimization is an oracle fitted to training GT and is never deployable.
Neither a one-image LoRA fit nor its metrics demonstrate generalization. The
purpose is to find where learning fails, not to select a production checkpoint.

Interpretation: if latent RGB optimization works but LoRA RGB fails, investigate
DiT adaptation/optimizer conditioning; do not tune decoder further. If RGB beats
mixed within the same parameterization on both records, objective conflict is
supported. If all single-image arms work, pooled optimization/data coverage
becomes the next question. If latent RGB itself fails, inspect decoder gradients,
precision, step sizes, and local optimization before blaming DiT. None of these
outcomes alone proves a unique root cause; fixed learning rates and two examples
limit the inference. Negative gradient cosine is local evidence, not proof that
an auxiliary objective is globally harmful.

No development GT is optimized or used for filtering. Qwen degradation
conditioning, mixed degradations, sealed internal and official FoundIR test
remain prohibited. No new long training is authorized by this diagnostic result
alone without reviewing its evidence and freezing the next design.

## CPU readiness evidence

Three focused tests and full artifact/H100/H200 preflight passed. Real 6B CPU
64x64 two-update smokes passed for LoRA RGB and latent mixed. Both reproduce
the same initialization and initial gradient measurements. Weighted latent
gradient norms are RGB .005552, latent .002975, LPIPS .015705; pixel/LPIPS
cosine -.01389 is close to orthogonal, not strong antagonism. LoRA's first step
changes 77,364,194/79,380,480 FP32 parameters and raises this small crop's PSNR
23.4144 -> 25.0655. These are interface checks and local clues, not evidence
of full-resolution convergence. GPU four-arm results are still required.
