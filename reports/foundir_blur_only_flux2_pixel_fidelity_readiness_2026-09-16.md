# FLUX.2 Blur-only step-10k pixel-fidelity gate readiness — 2026-09-16

## Decision

The complete 2,595-record continuation result closed further step/LR scaling:
step 10,000 remains best at `25.734236 / 0.145406`, while step 12,000
regressed to `25.709168 / 0.145691`. The user explicitly authorized changing
the fidelity mechanism while retaining the exact step-10k parent.

The new bounded experiment is a paired tiny ablation, not another continuation:

- both arms load exact parent
  `checkpoints/foundir_blur_only_flux2_full_v1/step_010000.safetensors`, SHA-256
  `e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`;
- both arms use the same 16 isolated-Blur training crops, sample order, noise,
  timesteps, pure noise-to-clean flow, native LQ reference, 50-step official
  inference, LR `1e-5`, global batch 8, and 400 optimizer steps;
- `flow_control` uses only flow MSE;
- `rgb_mse` adds only direct RGB MSE from
  `predicted_clean = noise - predicted_velocity`, decoded through the frozen
  VAE; no Charbonnier, edge loss, LPIPS loss, pixel refiner, or decoder tuning;
- every arm starts with a fresh AdamW state. Old optimizer moments are not
  compatible with the changed objective;
- checkpoints and fixed train triptychs are saved at 0/50/100/200/400;
- the 16 source-group-disjoint development crops are evaluated only at 0/400
  and are explicitly prohibited from choosing the arm, weight, or checkpoint.

Config:
`configs/foundir_blur_only_flux2_pixel_fidelity_gate_v1.json`, SHA-256
`e366396c9189dbd03dda45a501d41552cb80ccfb2ebc60748848dcd22284a9b1`.

## Why this is not a rerun of the failed five-type recipe

The historical five-type trainer combined a degraded-anchored flow, mixed
isolated/composed degradations, Qwen conditions, a six-step non-official
evaluation solver, and a jointly trained 33,251-parameter pixel refiner. Its
RGB Charbonnier term had weight 0.1 and ran only every eighth optimizer step.
Its 32-image overfit aggregate spanned 11 categories; the two isolated Blur
examples changed `19.78098 -> 19.38375 dB` and
`19.88767 -> 20.03899 dB`. It therefore never isolated or validated direct
RGB gradients for the successful Blur-only pure-flow formulation.

The old implementation's local identity was mathematically valid for its
anchored source (`source - velocity`). The reusable part is only the tested
token-unpack, frozen-VAE decode, and differentiable gradient path. The new gate
uses the correct pure-flow identity (`noise - velocity`) and removes every
historical confound listed above.

## VAE ceiling and acceptance

Existing independent Blur VAE evidence is `41.106046 dB / 0.010297 LPIPS`,
including a sampled minimum near `35.12 dB`, far above the `25.9 / 0.18`
target. The GPU job nevertheless recomputes quantized encode/decode metrics on
all frozen 16 train and 16 descriptive-development crops and refuses training
unless both roles have mean PSNR at least 35 dB and mean LPIPS at most 0.03.

The predeclared step-400 RGB candidate must:

- beat the paired flow control by at least `0.15 dB` train PSNR;
- beat its own step-10k parent baseline by at least `0.15 dB` train PSNR;
- regress LPIPS by at most `0.003` versus both control and parent;
- beat control PSNR on at least 10/16 training records;
- then pass human review of fixed `Input | Output | Ground Truth` images.

No full-scale run starts automatically even if these checks pass.

## Verification

- governance hold
  `flux2_blur_pixel_fidelity_design_v1_final-d37d35b71035af98` is resolved by
  the exact latest user authorization;
- 14 focused tests pass, covering flow identity, token geometry, real loss
  gradient direction, frozen decoder parameters, protected scope, manifest
  source disjointness, paired gate logic, parent identity, and H100/H200 global
  batch equivalence;
- Python compilation, shell syntax, full model/parent/manifest hashes, prior
  VAE evidence, and the complete launcher `--check` pass;
- a real FLUX VAE CPU forward/backward smoke produced latent shape
  `[1,128,4,4]`, finite RGB loss `0.00143477`, nonzero latent-gradient L1
  `0.07469666`, and no decoder-parameter gradients;
- protected flags remain false for Qwen degradation conditioning, mixed
  degradation, development-GT optimization, sealed internal, and official
  FoundIR test access.

## Launch

On one homogeneous 8xH100/H200 GPU instance:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && \
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
smoke/launch_foundir_blur_only_flux2_pixel_fidelity_v1_8gpu.sh
```

Monitor:

```bash
tail -f logs/foundir_blur_only_flux2_pixel_fidelity_gate_v1.log
```

Expected runtime is roughly 20--35 minutes, dominated by fixed 50-step image
snapshots and the RGB arm's differentiable VAE decode. Runtime is an estimate,
not a quality claim. Inspect `result.json` and all fixed visuals before any
full-scale decision.
