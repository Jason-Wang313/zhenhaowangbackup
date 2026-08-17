# Lowlight Qwen-conditioned FLUX.2 comparison readiness

Date: 2026-08-17 UTC

## Scope

The expanded frozen-Qwen ridge assessor passed its predeclared train-selection
and previously unexposed validation gates. This milestone prepares the fair
FLUX.2 baseline-versus-Qwen restoration comparison authorized by that result.
It does not claim that Qwen conditioning improves restoration, does not report
an official FoundIR result, and does not claim Phase 1 completion.

The scientific question is restricted to whether the frozen Qwen-predicted
Lowlight severity signal improves restoration while the FLUX.2 backbone, data,
source split, crops, LoRA restoration path, optimizer, learning rate, training
steps, batch size, precision, seeds, GPU budget, and inference settings remain
fixed.

## Frozen assessor evidence

- Assessor: frozen `Qwen2.5-VL-7B-Instruct` features plus scalar ridge head
- Selected ridge alpha: `3000`
- Unexposed validation accuracy / MAE: `0.68` / `0.332`
- Luminance baseline accuracy / MAE: `0.64` / `0.432`
- Per-level recall L1-L5: `0.88`, `0.60`, `0.68`, `0.44`, `0.80`
- Assessor checkpoint SHA-256:
  `27f58645d992c97f8b3e44af82a2a9ae11d93d0fa4bc5090b3d1bde39271b368`
- Assessor metrics SHA-256:
  `18285e16fad062d8653e9468159f6679531a7f97ea1d5b794bc12fe4994cc29a`
- Internal test used for assessor metrics: `false`
- Official FoundIR used: `false`

The prediction export is label-blind: it records only the frozen assessor's
predicted Lowlight level and provenance for each pair. It does not export true
levels or compute assessor accuracy on the protected internal-test split.

## Fair conditioned run

- Config: `configs/lowlight_flux2_qwen_conditioned_full.json`
- Config SHA-256:
  `ae50f3b289f6f3733bf7904f9659c93147896b7280771e0a2a66d3455fd3d41b`
- Baseline config SHA-256:
  `c71235efe88e8f3bc98ca863350af372a78aa35f1a2cc065a1867c5bb6bfde18`
- Training: `8,000` records / `1,600` sources / `4,000` steps
- Validation: fixed `1,000` records / `200` sources
- Internal test: fixed `1,000` records / `200` sources
- FLUX.2 LoRA: 50 rank-8 modules / `8,110,080` parameters
- Condition adapter: one learned 7,680-dimensional severity token /
  `38,400` parameters
- Total trainable parameters: `8,148,480`
- Inference: 50 steps, guidance `1.0`, seed `2026081503`
- Runtime output: `checkpoints/lowlight_qwen_conditioned_full/`
- Runtime log: `logs/lowlight_qwen_conditioned_full.log`

The condition adapter is the only intended model-path addition. It maps one of
the five frozen predicted levels to a learned token and appends that token to
the same frozen prompt context used by the baseline. The authorization layer
requires exact equality between the conditioned and baseline `model`, `data`,
`lora`, `training`, and `inference` sections. It also re-hashes the baseline
gate, baseline checkpoint and metrics, assessor checkpoint and metrics, frozen
prompt context, feature extractor, and every runtime implementation file.

## Predeclared comparison gate

The primary quantitative scope is internal-test L2-L5. Relative to the frozen
baseline, the conditioned system must:

- improve mean PSNR;
- improve mean SSIM;
- reduce mean LPIPS; and
- regress L1 mean SSIM by no more than `0.01`.

The comparison also produces paired baseline/Qwen contact sheets for fixed
stratified cases, baseline worst cases, and the largest improvements and
regressions. Passing the quantitative gate is not sufficient by itself; scene
identity and structural preservation require human review.

## Restart and evidence behavior

The launcher holds an `flock` lock and appends preflight, prediction export,
training, both evaluations, and comparison output to one log. The 10,000-record
prediction export, 4,000-step training checkpoints, and per-image evaluation
outputs are provenance-bound and resumable. Existing matching artifacts are
verified before reuse, while differing artifacts are rejected rather than
overwritten.

At readiness verification, the expanded assessor cache contained `8,250` of
the `10,000` required features. The remaining `1,750` features are extracted
before training. No conditioned prediction manifest, training output, metrics,
comparison result, or conditioned log existed yet.

## CPU verification

- Full repository test suite: `56` passed
- Python compilation: pass
- Launcher shell syntax: pass
- Git whitespace check: pass
- Conditioned CPU preflight: pass
- Frozen implementation hashes: pass
- Condition adapter parameters: `38,400`
- Total trainable parameters: `8,148,480`
- Existing prediction sidecars: `0`
- Conditioned config SHA-256:
  `ae50f3b289f6f3733bf7904f9659c93147896b7280771e0a2a66d3455fd3d41b`

GPU execution remains pending until this tracked milestone is committed and
pushed. Lowlight is the only authorized task; Blur, Noise, Haze, Rain, the
100,000-record dataset, and the separate `亮色大模型` workstream remain out of
scope.

## Offline H100 command

```bash
CUDA_VISIBLE_DEVICES=0 smoke/run_lowlight_qwen_conditioned_full.sh
```

Expected total runtime is approximately 6-7 hours on the H100. A detached tmux
session is recommended if the browser connection may close.
