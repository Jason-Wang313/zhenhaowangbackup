# Controlled LoRA step-size and stability audit

DIAGNOSTIC ONLY / 非主线. Existing Blur-only user authorization; user said next.

The completed optimization audit demonstrates healthy frozen-decoder latent
optimization, while strong-blur LoRA RGB collapses near step 200 and near-clean
LoRA mixed regresses. The exact cause is not isolated. Reproduce these two
failing combinations with four learning rates: 5e-5, 2e-5, 1e-5, 5e-6. Keep
the exact FP32 endpoint initialization, frozen BF16 model/VAE, AdamW defaults,
gradient clipping at 1, single-image batch, objective, seed and 200 steps.
Eight independent ranks own four rates x two record/objective combinations.
There is no averaging across experiments. The two objectives are not compared
causally with each other; learning rate is the only changed variable within
each record/objective comparison.

Record every pre-update loss and pre-clipping gradient norm. At steps 1 and
every 20 steps, and every step >=160, record LoRA layer gradient norms, actual
update norms/relative update, and decoder/DiT normalization input RMS ranges.
Capture normalization statistics during the original forward only, not
checkpoint recomputation. Save fixed images at 0,1,20,40,60,80,100,120,140,
160,180,190,195,200. Save reference-rate checkpoints at 180,190,195,200 and
final checkpoints for every rank, allowing later forward/backward inspection.
Report CPU/GPU duration and instrumented overhead rather than assuming identical
speed. Compare the reference trajectory with the prior run; if it differs
materially, investigate before attributing differences to learning rate.

Lower LR and 200 steps may learn more slowly, so a stable but weaker fit is
not automatically superior. Judge stability, loss trajectory, PSNR and LPIPS
together; no development or full-data claim comes from these two fitted images.
If all rates show spikes, inspect recorded layer/normalization evidence before
changing precision or architecture. If smaller rates remove collapse but do
not yet fit, a separately documented longer fixed-budget learnability stage
can be considered. Never silently select a late failing checkpoint.

No decoder tuning, Qwen degradation conditioning, mixed degradation, sealed
internal, or official FoundIR test. No oracle is deployed. No additional advisor
round is required under the user's existing authorization.
