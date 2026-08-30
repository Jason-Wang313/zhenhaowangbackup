# FoundIR five-type Qwen-guided FLUX.2 full-pipeline readiness — 2026-08-30

## Outcome

The complete development and internal-freeze pipeline is implemented for one
Qwen-guided FLUX.2 [klein] 4B Base restorer covering Blur, Noise, Rain, Haze,
Lowlight, and the admitted single/double/triple FoundIR combinations. It does
not access the official FoundIR test during development.

The code is ready for one resumable launch on exactly eight homogeneous NVIDIA
H100 80GB or eight NVIDIA H200 GPUs. Scientific global batch size remains 32:
H100 uses microbatch 1 and accumulation 4; H200 uses microbatch 2 and
accumulation 2.

The GPU run cannot start yet because the authorized official FoundIR training
archive is absent at `data/foundir/MillionIRData/Train`. Metadata-only
preflight passes for all 1,010,064 official metadata rows and the selected
685,495 five-type rows. No protected official-test pixels were accessed.

## Frozen execution

- Joint Qwen condition export is eight-way sharded, append-only, hash-checked,
  fsynced, and resume-safe.
- Canonical restoration text is encoded once into verified SafeTensors using
  FLUX.2's frozen Qwen3 text encoder and consumed through a bounded LRU cache.
- The restoration LoRA starts fresh after the stage-0 architecture gate and
  uses rank/alpha 32 on the declared image and image/text interaction paths.
- Training uses 1.8 million balanced presentations and 56,250 optimizer steps:
  single, then single+double, then single+double+triple.
- The trainer preserves the exact deterministic curriculum suffix and AdamW
  state across resume, restores optimizer tensors to their parameter device,
  records completed versus in-progress stage history separately, and runs an
  RGB VAE memory canary before the first optimizer step.
- Losses are degraded-anchored flow MSE, predicted-clean latent L1, and RGB
  Charbonnier every eight steps. Five percent identity pairs and five percent
  generic-condition dropout are retained.
- Development selection evaluates six predeclared checkpoints and maximizes
  the worst normalized PSNR/LPIPS margin across all five tasks.
- Generic, Joint-Qwen, shuffled/wrong, and type-oracle conditions are compared
  using identical selected restoration weights.
- Native-resolution development uses fixed 512-pixel tiles, 128-pixel overlap,
  deterministic window blending, native PSNR/SSIM, resized LPIPS, and a seam
  error ratio.
- Released FoundIR generalist, Lowlight specialist, and weather specialist are
  evaluated on the same frozen record contract. Specialists are used only for
  semantically valid categories.
- The sealed internal evaluation is single-use. Its original contract stays
  immutable; a separate canonical consumption record prevents a second run in
  another output directory.
- The final report fails scientifically with status 2 unless the sealed
  Stage-2-caliber thresholds and the Qwen causal non-regression/improvement
  checks pass. Infrastructure failures remain status 1.

## Static evidence

- JSON parsing passed.
- Python compilation passed for all new full-pipeline modules.
- Shell syntax passed for the full runner and detached launcher.
- Twenty focused FoundIR protocol, prompt-cache, manifest, curriculum-resume,
  checkpoint-selection, contract-isolation, and tiled-blending tests passed.
- Metadata-only preflight passed both with large hashes skipped and with full
  local model/checkpoint hashes enabled.
- `git diff --check` passed.

## Launch action after training data is present

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/launch_foundir_five_type_flux2_full_8gpu.sh
```

Persistent log:
`logs/foundir_five_type_flux2_full_v1.log`.

GPU telemetry:
`logs/foundir_five_type_flux2_full_v1_gpu.csv`.

The launcher deliberately fails before detaching if the training archive is
missing, the host is not exactly eight homogeneous H100/H200 devices, or any
required source differs from the committed revision.
