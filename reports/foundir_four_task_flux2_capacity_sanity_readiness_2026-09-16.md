# Four-task FLUX.2 capacity sanity readiness

## Status

CPU-side implementation and complete preflight pass. No GPU job has started from the Codex shell.

The authorized phase-1 suite runs four independent isolated-degradation FLUX.2 Klein Base 4B LoRA arms for `05Noise`, `08Haze`, `10Rain`, and `14Lowlight`. It does not use Qwen degradation predictions, mixed degradation, a pixel refiner, sealed internal, or the official FoundIR test. It cannot automatically launch joint training.

## Frozen identities

- Config: `configs/foundir_four_task_flux2_capacity_sanity_v1.json`
- Config SHA-256: `995eae82efd24306139b2d91680beff9a5fa82076c34ec7f1b8821e74ffcd21c`
- Frozen manifest: `manifests/foundir_four_task_flux2_capacity_sanity_v1.csv`
- Manifest SHA-256: `3abbaad9bb7a476d9fd673f76542a984471c72f265a5914170efe8151e30ec98`
- Manifest metadata SHA-256: `cebe76cd13e93c4f68db5e6c83fa059c430cdbd1135d29f0422769e53849ef34`
- Governance design hold: `flux2_four_task_capacity_sanity_design_v1-37a4459ff7574823`
- Hold SHA-256: `1d6d2812fb89df95553eaae7e6b48a1cd96fa431cca1c04c603e31511a4edf0e`
- Decision source: explicit USER authorization after the advisor's joint-training direction; not a fabricated advisor reply.

The manifest contains exactly 128 records: per task, 16 unique training source groups and 16 unique source-disjoint validation source groups. Every LQ/GT path, stored SHA-256, image dimensions, deterministic crop, category, split, composition size, and source-group separation passed complete preflight.

## Execution design

One homogeneous 8×H100/H200 host runs four independent two-GPU arms concurrently:

| Task | GPUs | Per-task global batch |
|---|---:|---:|
| Noise | 0,1 | 8 |
| Haze | 2,3 | 8 |
| Rain | 4,5 | 8 |
| Low-light | 6,7 | 8 |

Each task uses microbatch 1 per rank, accumulation 4, 400 optimizer steps, fresh rank-32 LoRA, task-fixed prompt, checkpoints 50/100/200/400, and official 50-step inference. Fixed train and descriptive validation `Input | Output | GT` triptychs plus PSNR/LPIPS are saved at every checkpoint.

## Verification

- New focused suite: `10 passed`.
- New plus legacy Blur sanity compatibility: `16 passed`.
- Python compilation: passed.
- Shell syntax: passed.
- Governance check: passed.
- Model hash preflight: passed once for the shared FLUX/VAE artifacts.
- Four complete task/image preflights: passed.
- Launcher `--check`: `FOUNDIR_FOUR_TASK_CAPACITY_LAUNCHER_CHECK_PASS`.

## Launch

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
smoke/launch_foundir_four_task_flux2_capacity_sanity_v1_8gpu.sh
```

Monitor the master log with:

```bash
tail -f logs/foundir_four_task_flux2_capacity_sanity_v1.log
```

Per-task logs end in `_noise.log`, `_haze.log`, `_rain.log`, and `_lowlight.log`. The terminal aggregate always stops for joint visual/scientific audit and emits a new governance hold. A failed arm does not trigger a retry or block the other three arms from completing.
