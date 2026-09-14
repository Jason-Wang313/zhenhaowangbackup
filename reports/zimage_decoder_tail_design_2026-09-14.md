# Blur-only Z-Image decoder intervention

DIAGNOSTIC ONLY / 非主线. This is not a five-type FLUX.2 or official FoundIR result.

The user explicitly authorized teacher, few/single-step distillation, then
decoder tuning if quality remains inadequate, without another advisor round.
The endpoint step-400 student satisfies that condition: train 25.4045/.2550
versus input 24.7232/.3085; development 22.8776/.3520 versus input
23.9045/.3263. Fixed images contain grain/web artifacts and insufficient
motion-blur recovery. Healthy GT VAE roundtrips (~41 dB) do not exclude
decoder sensitivity to predicted latents. No causal claim about that
sensitivity has yet been established.

Freeze the immutable endpoint student and preserve its exact FP32 LoRA values.
Cache its predicted latents. Train decoder blocks 15--17, output GroupNorm,
and output convolution in FP32; keep the earlier decoder layers and all other
model components frozen. Keep the same 16 train and 16 source-disjoint
development crops. Train 400 steps, global batch 8, LR 1e-5, AdamW FP32,
save/evaluate steps 0/200/400. Loss: RGB MSE + .05 LPIPS on predicted latents,
plus .1 RGB MSE on GT-latent reconstruction. GT anchoring uses training only.
Development GT is evaluated only, never used for gradients or target filtering.

Record the original frozen-decoder baseline before promoting the decoder tail,
then step 0 after promotion to expose precision-only changes. Evaluate output
PSNR/LPIPS and GT roundtrip at each checkpoint, with fixed triptychs. A candidate
must improve both development metrics versus its step-0 baseline AND input,
improve both training metrics versus input, and retain development GT roundtrip
PSNR >=35 and LPIPS <=.02. Visual review remains necessary; metric acceptance
alone does not authorize scaling. Failure is useful evidence against this
bounded decoder intervention, not proof that all decoder strategies fail.

The deployed bundle is the immutable endpoint LoRA plus the saved decoder-tail
tensors, pinned base weights, and this precision policy. Reload the final tail
checkpoint and require exact parameter equality before publication.

No Qwen degradation conditioning, mixed degradation, sealed internal, or
official FoundIR test access. Supports homogeneous 8x H100/H200. No full
teacher training or full student training is claimed completed.
