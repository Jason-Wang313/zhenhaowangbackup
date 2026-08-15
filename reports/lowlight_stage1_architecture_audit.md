# Lowlight Stage 1 FLUX.2 architecture audit

Date: 2026-08-15 UTC
Workspace: `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`
Scope: Lowlight-only Phase 1 restoration baseline

## Decision and boundary

The baseline uses the local FLUX.2 [klein] Base 4B checkpoint as one
reference-conditioned restoration backbone. The FLUX.2 transformer,
autoencoder, and FLUX Qwen3 text encoder are frozen. Only rank-8 LoRA weights
on selected transformer image-path Linear layers are trainable.

This checkout is an inference implementation. It documents that Base models
are suitable for fine-tuning and LoRA, but it contains no BFL training loop,
paired-restoration loss, optimizer recipe, or official restoration LoRA
implementation. Consequently, the flow-training formulation in this project
is **project-derived**, not an official BFL training recipe. It must be judged
by the fail-closed overfit and restoration evaluation gates below.

This audit does not establish GPU training success, visual identity
preservation, FoundIR performance, or Phase 1 acceptance.

## Verified local model stack

- FLUX.2 source: `third_party/flux2`, revision
  `50fe5162777813d869182b139e83b10743caef15`, with the tracked offline patch
  `patches/flux2-offline-smoke.patch` exactly applied.
- Flow weights:
  `models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors`,
  7,751,105,712 bytes, SHA-256
  `9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92`.
- Autoencoder weights: `models/flux2-klein-base-4b/ae.safetensors`,
  336,211,292 bytes, SHA-256
  `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`.
- FLUX text encoder: `models/qwen3-4b-fp8`, with both indexed SafeTensors
  shards checked by exact size and SHA-256 during full preflight.
- The Qwen3 checkpoint above is FLUX's text encoder. It is not a
  vision-capable Qwen-VL checkpoint and cannot satisfy the later structured
  degradation-prediction role.
- Default Base-4B inference settings in the local model registry are 50 steps
  and guidance 4.0; the checkpoint is not guidance-distilled.

The full CPU preflight on 2026-08-15 verified all large hashes, 149 flow
tensors, 251 autoencoder tensors, the exact FLUX revision/patch, imports,
manifest files, LoRA shapes, and free disk.

## Transformer architecture

The local `Klein4BParams` configuration has:

- token width 128 in and out;
- text-context width 7,680;
- hidden width 3,072;
- 24 attention heads;
- five double-stream blocks followed by 20 single-stream blocks;
- four positional axes with dimensions `[32, 32, 32, 32]` and theta 2,000;
- MLP ratio 3.0;
- no guidance embedding in the transformer.

The model projects image and text tokens separately. Double-stream blocks
update image and text streams with attention and gated residual MLP paths.
The streams are then concatenated for the single-stream blocks, and the final
layer maps image tokens back to the 128-channel velocity representation.
Timestep embeddings drive the model's modulation layers.

## Native reference-image path

The official inference code encodes a reference image with the FLUX.2
autoencoder, flattens its latent grid with `batched_prc_img`, and assigns the
first reference a time coordinate of 10. During non-cached sampling,
`denoise_cfg` appends the reference tokens and IDs to the current noisy image
tokens at every step, runs classifier-free guidance, keeps only the predicted
target-image tokens, and integrates from the schedule's `t=1` endpoint toward
`t=0`.

The baseline deliberately reuses this native token interface:

```text
clean RGB ------------> frozen VAE ---> clean latent ---+
                                                       +--> project-derived flow pair
fixed noise --------------------------------------------+

degraded RGB ---------> frozen VAE ---> reference latent
                                         |
                                         +--> reference tokens, time ID 10

prompt ---------------> frozen Qwen3 ---> context tokens

[noisy target tokens, degraded reference tokens] + context
    ---> frozen FLUX.2 plus trainable LoRA ---> target-token velocity
```

At evaluation, the script returns to the official 50-step CFG sampling path
with the degraded latent supplied as reference tokens. No second restoration
model is chained into the result.

## Project-derived training objective

For clean latent `x_0`, sampled noise `epsilon`, and uniform finite
`t in [0,1]`, the project constructs:

```text
x_t = (1 - t) * x_0 + t * epsilon
velocity target = epsilon - x_0
```

The degraded latent is conditioning only; it is never the velocity target.
The loss is mean-squared error between the transformer's target-token output
and `epsilon - x_0`. Clean/degraded crops are aligned and deterministic.
Training positions, timestep/noise draws, and seeds are reproducible.

The loss-decrease gate uses a fixed probe: the same record ordering, crop,
noise seed, and timestep 0.5 are evaluated before training and at every
checkpoint. It therefore does not compare unrelated stochastic minibatches.

## Trainable scope

All base model, VAE, and text-encoder parameters are frozen. LoRA is injected
into these image-path modules:

- `double_blocks.*.img_attn.qkv`
- `double_blocks.*.img_attn.proj`
- `single_blocks.*.linear1`
- `single_blocks.*.linear2`

The exact SafeTensors-shape audit finds 50 matched Linear modules. Rank 8 and
alpha 8.0 produce exactly 8,110,080 trainable parameters. LoRA B matrices are
zero-initialized, so injection initially preserves the unmodified baseline
output. A CPU test verifies both exact output preservation and gradient scope.

## Data and gates

The first authorized run uses `manifests/low_light_overfit16.csv`: 16 unique
training sources with level quotas L1=3, L2=3, L3=3, L4=3, and L5=4, using
deterministic aligned 512-pixel crops. It is only an overfit gate and cannot be
used to claim generalization.

Automatic authorization for later full training requires all of:

- deterministic probe loss decreases;
- restored L2-L5 mean PSNR exceeds degraded-input mean PSNR;
- restored L2-L5 mean SSIM exceeds degraded-input mean SSIM;
- restored L1 mean SSIM is at least 0.70;
- every metric is finite and every restored/contact-sheet artifact exists;
- a human contact-sheet review confirms scene identity and no hallucination.

The evaluator refuses to reuse a failed gate. Full 8,000-pair training remains
unauthorized until both automatic and visual overfit gates pass.

## Artifact and interruption integrity

Each checkpoint consists of the LoRA SafeTensors file, optimizer state, and a
sidecar completion marker containing their SHA-256 hashes, config hash, step,
and metric record. Files are published without replacing an existing target;
the sidecar is published last. A partial set, hash mismatch, conflicting
metric, differing frozen config, or non-finite value fails closed.

If interruption occurs after a complete checkpoint is published but before
its JSONL metric append, resume validates both artifact hashes and reconstructs
that exact metric from the sidecar. The final summary verifies and records the
checkpoint SHA-256, optimizer SHA-256, top-level Git revision, FLUX revision,
exact config, command, Python/PyTorch/CUDA/GPU environment, and output paths.

## CPU verification evidence

The following passed on 2026-08-15 using `.venv/flux2-smoke/bin/python`:

- Python compilation for trainer, package, preflight, evaluator, builder, and
  tests;
- shell syntax for `smoke/run_lowlight_baseline_overfit.sh`;
- `git diff --check`;
- nine unit tests, including config rejection, deterministic ordering,
  interrupted-summary recovery, non-finite metric rejection, and failed
  evaluation reuse;
- CPU preflight with `--skip-large-hashes`;
- CPU preflight with all large SHA-256 checks enabled.

Both preflights reported config SHA-256
`62740e0e0f26c4001057adf9d955259b7c8d89553245a2c1565a22599e7d0804`,
50 LoRA modules, and 8,110,080 trainable parameters.

## Remaining evidence before the Lowlight gate

The immediate next evidence is the offline H100 16-pair run, its training
summary, automatic metrics, restored images, and identity contact sheets.
After that gate passes, the fixed 8,000/1,000/1,000 baseline run, real Qwen-VL
selection and conditioning ablation, and official FoundIR-Lowlight evaluation
remain required. The final Lowlight threshold is PSNR at least 15.1 dB and
LPIPS at most 0.34 on the official FoundIR-Lowlight evaluation split.
