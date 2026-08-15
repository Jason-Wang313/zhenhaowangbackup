# Lowlight FLUX.2 16-pair overfit result

Date: 2026-08-15 UTC
Workspace: `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`

## Result

The FLUX.2 Lowlight baseline **passed the bounded 16-pair overfit gate at
guidance 1.0** after the originally configured guidance 4.0 evaluation failed.
This authorizes preparation of the fixed full baseline training/evaluation
pipeline. It does not establish generalization, FoundIR performance, or
Lowlight Phase 1 completion.

## Bound evidence

- Training Git revision:
  `265050404ab2a36cd9f11514067ec97c1a5c951a`
- Guidance-diagnostic Git revision:
  `02144fe2ff506f45ebdb607f9951082854a434fa`
- FLUX revision: `50fe5162777813d869182b139e83b10743caef15`
- Config SHA-256:
  `62740e0e0f26c4001057adf9d955259b7c8d89553245a2c1565a22599e7d0804`
- Final checkpoint:
  `checkpoints/lowlight_baseline_overfit16/step_000100.safetensors`
- Checkpoint SHA-256:
  `94ecd1c32e7a137ed153b581ea0b12c879d7e5e3c0698375a3d2293ecb6481c7`
- Optimizer SHA-256:
  `a9d2256d7ebb9d588816dc9442461d9764d8dd832ff05bf977f59e21f446bba2`
- Passing metrics:
  `checkpoints/lowlight_baseline_overfit16/evaluation_guidance_1/metrics.json`
- Passing metrics SHA-256:
  `9ffb568a500eb378a2ad81ccd4ff4f329bf47b3dcb589dbe5d12623aeba96edc`
- Contact-sheet SHA-256 values:
  - sheet 1: `3e11dc0a58e1041bad1eab1826b54d5576897ef844635a77387bf8267d884412`
  - sheet 2: `a0879f7297069c18284612a3625e62699f9583a603aaa11af892f65200cc28b0`
- GPU: NVIDIA H100 80GB HBM3; peak reported allocation approximately
  17.61 GiB.

## Training evidence

The fixed probe used the same L5 pair, crop, noise seed, and timestep 0.5 at
every checkpoint. Its loss decreased monotonically:

- step 0: `0.2863059044`
- step 25: `0.2833456993`
- step 50: `0.2750599980`
- step 75: `0.2715889513`
- step 100: `0.2700676620`

All four checkpoint round-trip checks passed. The final adapter and optimizer
hashes match the training summary and sidecar.

## Guidance diagnosis

Training always used the conditional restoration prompt. The first evaluation
used CFG 4.0, which amplified the conditional/unconditional difference beyond
the path used during training. It passed aggregate degraded-image improvement
but failed L1 identity:

- L2-L5 restored mean PSNR: `16.2327 dB`
- L2-L5 restored mean SSIM: `0.7069`
- L1 restored mean SSIM: `0.6585` (required `>=0.70`)
- automatic gate: failed

The bounded guidance-1 diagnostic changed only CFG guidance and the output
directory. It reused the exact checkpoint, 16 crops, per-image seeds, 50-step
schedule, prompt, and model stack. Guidance 1.0 is the conditional-only path
that matches training. It produced:

- L2-L5 input/restored mean PSNR: `9.2784 -> 22.1979 dB`
- L2-L5 input/restored mean SSIM: `0.4287 -> 0.8537`
- L1 restored mean PSNR: `23.3290 dB`
- L1 restored mean SSIM: `0.7731`
- loss decrease: pass
- PSNR improvement: pass
- SSIM improvement: pass
- L1 identity floor: pass
- automatic gate: pass

Guidance 1.0 is therefore the frozen inference setting for the next baseline
stage unless later validation evidence requires a documented change.

## Human identity review

Both guidance-1 contact sheets were inspected at original resolution. All 16
restorations preserve the depicted scene, geometry, object layout, and semantic
identity. No object insertion, removal, scene replacement, or structural
hallucination was observed. The human identity gate passes.

Recorded limitations:

- `div2k_0360_L1` preserves the cathedral geometry and decoration but changes
  tone/contrast substantially; its SSIM is `0.5199`.
- `lsdir_0001916_L5` preserves the Christmas scene but remains the weakest
  dark-sample restoration at `14.5789 dB` PSNR.
- Some outputs retain small brightness or color differences from the clean
  targets. These must be monitored on validation and internal-test sources.

## Authorization boundary

The 16-pair automatic and visual gates are passed at guidance 1.0. Full
8,000-pair baseline code/configuration may now be prepared and independently
preflighted. The full run must retain the fixed 1,000-record validation and
1,000-record internal-test sets. No Qwen-VL gain, official FoundIR metric, or
Phase 1 SOW threshold is claimed by this result.
