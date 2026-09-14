# Round 2: latent-first endpoint quality correction

DIAGNOSTIC ONLY / 非主线. User said `execute` after the completed round-1
reproducibility gate. This freezes ONE quality correction, not another LR or
reproducibility sweep. The overall multi-step teacher/full-data/distillation
goal is unfinished. This experiment repairs the supervised endpoint mapping;
it is NOT a new distillation result, nor a claim that the failed teacher passed.

## Evidence and hypothesis

The repaired deterministic route reproduces exact gradients/updates/weights
and snapshot latents across repeats and instrumentation. Pure-RGB strong-blur
fitting still has grainy artificial texture (0042483), whereas the training-GT
latent oracle through the SAME original frozen decoder looks close to GT.
The frozen VAE reconstructs GT at about 40--41 dB on the fixed train/dev crops.
During deterministic RGB fitting, latent MSE rises .27390 -> .28909 despite
RGB improvement. Near-clean mixed fits remain below input PSNR. These facts
motivate anchoring the learned latent to a decoder-compatible target; they do
not prove latent anchoring will cure generalization or identify a unique cause.

## Frozen correction

Same Z-Image Base, native fixed text prompt, original frozen VAE and exact
endpoint step-400 FP32 LoRA initialization. Same LR 5e-5, AdamW defaults,
weight decay 0, norm clipping 1, global batch 8, aligned frozen 512x512 crops,
same endpoint input/output mapping and LoRA scope. No new architecture/head,
pixel bypass, decoder tuning, augmentation or data. Reset optimizer explicitly.

- Steps 1--800: GT latent MSE only, weight 1. No decoder backward/LPIPS loss.
- Steps 801--1200: latent MSE (1) + RGB MSE (1) + LPIPS (.05).
- No teacher imitation: the historical teacher qualifies only 3/16 records;
  this is supervised recovery, never relabel it successful teacher distillation.
- The stronger persistent latent weight is 10x the earlier .1 coefficient.
  Warmup plus stronger anchor and removing weak teacher imitation constitute
  one training-objective recipe; results cannot isolate their individual effects.
  1,200 total steps is a predeclared bounded budget, not an indefinite extension.

All 16 training and 16 source-group-disjoint development crops are unchanged.
Development targets enter metric calculation ONLY, never training losses,
teacher filtering, rehearsal or optimizer updates. Images remain fixed at all
seven snapshots: 0/200/400/600/800/1000/1200 (224 triptychs). Latent-only-phase
evaluations still decode and measure BOTH image metrics, so a falling latent
loss cannot be misreported as visual progress.

## Runtime validation

Use the verified padding repair, explicit deterministic FlashAttention backward,
strict PyTorch/cuDNN determinism, TF32 off and CUBLAS_WORKSPACE_CONFIG=:4096:8.
Eight-rank synchronous data parallelism is one shared model/optimizer, not
independent image memorization. Gradients are packed in a fixed parameter order
into one FP32 vector, all-reduced and divided by eight before global clipping.
This avoids changing DDP bucket layouts across the two objective stages.
Fix NCCL Ring/Simple and data presentation order. Before actual optimization,
repeat two distributed rehearsal updates (one latent-only and one joint loss)
from identical restored weights, RNG and new optimizer; require exact per-rank
outputs/POST-clipping averaged gradients/updated parameters across repeats and synchronized
parameters across ranks. Reset all state again before step 1. Round-1 single-
GPU equality does not alone prove DDP repeatability; this rehearsal tests it.
Verify nonzero FP32 updates and frozen-model gradient exclusion in rehearsal.

Startup also verifies inherited endpoint per-record PSNR/LPIPS against the
immutable parent within .02 dB/.002 LPIPS. All inference uses the SAME deployed
one-step mapping; no training-GT latent oracle is used at inference. Save all
checkpoint identities, fixed Input|Output|GT images, per-record PSNR/LPIPS,
GT latent MSE, raw RGB MSE, clipping fractions and per-step loss/gradient logs.
Track near-clean (input >=30 dB) preservation and largest per-record regressions
as diagnostics, without selecting or dropping examples using model outcomes.

## Decision after this gate

The predeclared candidate is final step 1200; no retrospective best-checkpoint
selection. Automatic bounded metric admission requires lower mean training
latent MSE than step 0 AND mean PSNR/LPIPS improvement over input on BOTH
train and development. Human fixed-visual review remains necessary and the
complete quality gate remains pending until reviewed. Original SOW thresholds
are unchanged; this 16/16 crop gate is not contractual or statistical acceptance.
No automatic full-data launch follows a numerical pass. Earlier snapshots are
evidence of convergence/overfitting, not permission to cherry-pick a replacement.

If latent fitting improves but visual quality does not, inspect predicted versus
GT latent/decoder behavior rather than declaring success. If train improves but
development fails, treat generalization as unresolved. If this bounded recipe
still cannot learn a credible mapping, stop the two-round decision phase and
revisit the training formulation with the evidence; do not add another series
of LR/step-count sweeps. No advisor re-request is needed under the existing user
authority, but no model-family change or protected-test access is implied.

Prohibited throughout: Qwen degradation conditioning, mixed degradation,
sealed internal, official FoundIR test and development-GT optimization.

## CPU readiness evidence

Config SHA-256: `cc4cb5e26803fa20a4ea92fc91ae2b6185619685b1e04e0a071e908aa89da904`.
Design/governance report: `zimage_latent_first_design-e0bd267718808756`;
records existing user authority without a new advisor request.
Full six model-weight hashes, inherited config/checkpoint/result hashes and
fixed-image preflight passed, including H100 and H200 inventories.
Sixty focused tests, Python compilation, shell syntax and diff whitespace
checks passed; test evidence is `logs/zimage_latent_first_tests_v1.log`.

The real 6B CPU 64x64 rehearsal completed both repeats, latent then joint;
outputs, post-clipped averaged gradients and parameter hashes match exactly.
Initial FP32 weights are restored exactly, AdamW state is FP32, both steps
change parameters, and all frozen gradients remain absent. Losses are
5.0682168 and .2708406 on two DIFFERENT training records; these are interface
smoke values, not a learning curve. Preclip gradient norms are 10.3766603 and
.8073046 (clip=1 applies on the first). CPU evidence does not establish
full-resolution CUDA quality, memory or distributed kernel reproducibility.

Evidence: `logs/zimage_latent_first_cpu_smoke_v1.log`,
`logs/zimage_latent_first_cpu_9f9hciau/rehearsal_rank_0.json`,
`logs/zimage_latent_first_preflight_v1.log`.
The focused test suite also compares two-process Gloo averaged gradients and
AdamW updates against a single-process global-batch reference for BOTH stages,
then checks exact repeated rehearsal and initial-state reset.
GPU rehearsal runs within the one new job; no separate GPU diagnostic launch.
