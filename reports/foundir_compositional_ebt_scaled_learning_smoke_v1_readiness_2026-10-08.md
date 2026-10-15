# FoundIR scaled EBT real-data learning smoke v1 readiness — 2026-10-08

## Status

**CPU PREFLIGHT PASS; TWO-GPU SMOKE NOT YET STARTED.**

This bounded diagnostic trains matched Plain and Compositional 26,043,648-
parameter EBT arms for 400 updates each on real frozen singleton/double data.
It is not a 100k run, does not save checkpoints, and cannot access triple,
sealed, or official FoundIR data.

## Frozen identities

- Config: `configs/foundir_compositional_ebt_scaled_learning_smoke_v1.json`
  - SHA-256: `2eead7bacca889db2bc704510df0ae30465c3752b9c2cd74df417a94033735c2`
- Trainer: `smoke/train_foundir_compositional_ebt_scaled_learning_smoke_v1.py`
  - SHA-256: `cf63afbc8b166f7f685b63f09bdad5cbd60a3755eb3cdc68f4ff0ecae90681ef`
- Runner: `smoke/run_foundir_compositional_ebt_scaled_learning_smoke_v1_2gpu.sh`
  - SHA-256: `c89ef9760dd787f16633a05f08345de48d5ea975624666728d6826c887e40c48`
- Data manifest SHA-256:
  `d7add2be0ec4edff95034caee1287b4746acffcec76d43ab3f288503bbf3b449`

## Frozen protocol

- Two independent arms: Plain and Compositional.
- One homogeneous H100/H200 GPU per arm; no DDP.
- 26,043,648 parameters per arm, 128x128 crops, patch size 16.
- One differentiable candidate-gradient step with fixed step size 10.
- Qualification: this is the project one-step adaptation, not the complete
  official learnable-alpha recipe.
- Batch 128, 400 optimizer steps, 51,200 unique train records per arm.
- AdamW, peak LR `1e-4`, 50-step linear warmup, weight decay `.01`.
- Before/after evaluation on the same frozen 512-record source-disjoint
  lower-order validation subset.
- Automatic promotion requires positive validation PSNR change for both arms.

## Preflight evidence

- Train schedule: 51,200 unique records; 25,653 singleton and 25,547 double.
- Validation subset: 512 records; 260 singleton and 252 double.
- All records have composition size one or two; no triple/test records enter.
- `none`, `jpeg`, and `resample` non-target nuisance target reconstruction
  passes with finite 128x128 tensors.
- The renderer's OpenCV BGR output is explicitly converted to RGB. A
  no-nuisance source-crop identity test passes exactly.
- The required HDD3 project interpreter loads host headless OpenCV 4.10.0
  explicitly; it does not switch to system Python or alter the environment.
- JSON, Python, shell syntax, exact parameter count, and whitespace checks
  pass.

## Launch

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_SCALED_LEARNING_SMOKE_RUN_ID=real400_v1 CUDA_VISIBLE_DEVICES=0,1 bash smoke/run_foundir_compositional_ebt_scaled_learning_smoke_v1_2gpu.sh
```

- Estimated wall time: `0.10–0.75 hours`.
- Estimated allocated compute: `0.20–1.50 GPU-hours`.
- Terminal result:
  `checkpoints/foundir_compositional_ebt_scaled_learning_smoke_v1/real400_v1/result.json`.

At completion, use measured real-data wait/compute time and validation change
to decide whether to design full training. Do not automatically launch 100k.
