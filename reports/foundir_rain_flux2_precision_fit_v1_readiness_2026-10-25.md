# Rain FLUX precision and fitting diagnosis

User decision: first make Rain DiT pass 34.2 PSNR / .08 LPIPS, then build a
working image-only MoE, then add VLM tokens. The small pixel refiner proposal
is withdrawn from the active plan. No additional restoration model is used
at inference by this experiment.

## Evidence and limits

The completed 3,000-step direct-endpoint formal is not resumed. Best step
2,000 was 32.085315339143946/.08882607079111039; step 3,000 regressed.
Reading its actual weights and optimizer showed BF16 LoRA parameters and
BF16 `exp_avg`/`exp_avg_sq` at LR 1e-5. Relative to the parent step-500 LoRA,
64.21640107125947% of LoRA A entries remained exactly equal at step 3,000;
only 4.126790364583333% of B entries remained equal. This does not prove
the entire quality gap is caused by precision. A CPU numerical example does
verify that subtracting 1e-5 from BF16 weights near .005/.01/.02 can be lost
when stored back. FP32 trainable state retains such changes.

The VAE weights are FP32. Old training subtracted the predicted velocity
from the original FP32 encoded input; old inference first cast the input
latent to BF16. New training and inference share the same function and
keep that subtraction FP32. CUDA matmuls/convolutions still use BF16.
Latent target formation also avoids premature BF16 rounding. No architecture,
LoRA rank, expert bank, teacher, or loss coefficient is added/changed.

## Frozen diagnostic

- Start each mode from the old best step-2,000 checkpoint, SHA-256
  `eff05dccc0d9bcdf538305ac4095304348a876eb6cb3a4ded3ea27767901f36a`.
- FP32 LoRA weights and Adam moments; LR 1e-5, weight decay zero, batch 32,
  eight ranks, microbatch one and accumulation four; no cross-node use.
- Keep the parent's eight loss terms/coefficients and frozen Restormer
  training teacher. Deployment remains a single FLUX LoRA.
- Select one record from each of 32 eligible training source groups using
  deterministic hashes. Use fixed, centered 512 crops and no augmentation
  to diagnose fitting of identical inputs. This is a limited fitting test,
  not the final training dataset or a generalization experiment.
- Keep the original 100-record/100-source development rows and 512 crops
  unchanged. Training source overlap is zero. No protected pixels enter.
- Qualification: 25 steps, evaluate at 0/25. Independent fitting diagnostic:
  300 steps, evaluate at 0/100/300. Each fit image receives 300 presentations.
  Qualification weights are forbidden as fit initialization.
- Step 0 measures both the legacy and shared-FP32 inference paths. Every
  milestone records training-fit and development PSNR/LPIPS separately.
- Log all loss terms and sampled actual A/B parameter update fractions.
  Assert FP32 optimizer state, finite outputs/gradients and F32 saved LoRA
  tensors. Save optimizer checkpoints and config/code/sample identities.
- The 300-step mode requires a passing qualification with matching config,
  code and sample hashes. It never auto-launches full-data training.

This simultaneously repairs numerical state and removes the training versus
evaluation resolution difference. It does **not** isolate the causal gain
from FP32 training through a matched second BF16 training arm. The initial
legacy/shared comparison isolates inference arithmetic only.

## Verification and identities

Four CPU tests pass: real LoRA/Adam small updates and exact FP32 checkpoint
round trip; shared endpoint precision, train/inference equality and nonzero
gradients; all eight objective terms backpropagate; deterministic training-only
source selection. Both shell scripts pass `bash -n`. Complete asset/hash,
selected-image hash, crop-boundary and source-disjoint preflight passes.

- Config SHA-256: `0b3609a8203b472d2c76b5952e0045affcbeb72d0191d80063f9819d9d08ed2a`.
- Fit records/crops SHA-256: `cfd29a617d50b25c87ba8d3e246566cd789fae412b057a677d2a6c43afdd969a`.
- Development rows SHA-256: `1652d4e46634c1ee248e1984ea2aeaec271d8491a67dcc3acf12838cb181d783`.

No GPU result exists yet. Qualification must establish real 512-crop memory
safety and throughput. Initial conservative estimates on one 8xH100/H200
node: qualification .25–1.0 wall hours / 2–8 allocated GPU-hours; independent
300-step fit .8–3.5 wall hours / 6.4–28 GPU-hours. These are unmeasured
estimates; replace them using qualification step times and evaluation cost.

## Launch and interpretation

Qualification (detached tmux; launcher captures preflight failures in log):

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_flux2_precision_fit_v1_tmux.sh
```

Log: `logs/foundir_rain_flux2_precision_fit_v1_qualification_v1.log`.
Result: `checkpoints/foundir_rain_flux2_precision_fit_v1/qualification_v1/result.json`.
Session: `rain_precision_qualification_v1` on socket `zhenhaowang_jobs`.

After qualification is verified, independently start fit from step 2,000:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_flux2_precision_fit_v1_tmux.sh fit fit_v1
```

If training-fit PSNR/LPIPS cannot meet the numeric target, investigate
optimization/objective/trainable DiT layers before expanding data or steps.
If fitting succeeds but development fails, focus on generalization and a
full-data matched-resolution recipe. A passing 32-image fit does not pass
SOW. A developmental gain alone cannot establish that precision caused it.
Any longer run must have its protocol and measured budget frozen first.
