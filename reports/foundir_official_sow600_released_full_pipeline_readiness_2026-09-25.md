# DIAGNOSTIC ONLY / 非主线 — Official SOW600 Released FoundIR comparison readiness

Date: 2026-09-25

The user explicitly authorized a frozen, no-training comparison on the same
600 official SOW singleton pairs. This comparison does not replace the
terminal project v8 result and does not update any model weights.

## Frozen systems and routing

- Project: terminal official v8 category metrics from canonical commit
  `5003e1e51bf9373fbdb352cc0420d712d396f114`.
- Released FoundIR Generalist: `third_party/foundir/premodel/model-2000.pt`.
- Full FoundIR: Generalist followed by Weather Specialist for all 200 Haze
  records and Low-light Specialist for all 50 Low-light records. Blur, Noise,
  and Rain retain Generalist output.
- Metric protocol: original-size RGB, no resize/crop/border shave, `pyiqa`
  0.1.16 PSNR/LPIPS/SSIM, unweighted per-image mean within each category.

## Frozen artifacts

- Authorization:
  `reports/advisor_governance/foundir_official_sow600_released_baseline_authorization_2026-09-25.json`,
  SHA-256 `1b9df7984dfa6d198b5dbf9c1c1070387367b1aca589d43b8516215c125a5307`.
- Config: `configs/foundir_official_sow600_released_full_pipeline_v1.json`,
  SHA-256 `3dff5bb4341d3f97c15470fe2d8bfaab7c9b22af051c13171d38fa986224802d`.
- Evaluator:
  `smoke/evaluate_foundir_official_sow600_released_full_pipeline_v1.py`,
  SHA-256 `81ed5d0bf96cf58901ad1abe5a11aff9c45d22451dc46edb6ff57fb42c8313c6`.
- Runner:
  `smoke/run_foundir_official_sow600_released_full_pipeline_v1_8gpu.sh`,
  SHA-256 `2ba3768ad86d77022fc2a736845f5c4b5c8f2812852c22d8f4cb2ea27c430972`.
- Launcher:
  `smoke/launch_foundir_official_sow600_released_full_pipeline_v1_8gpu.sh`,
  SHA-256 `aab5989b721a499f0961a5ae0e863b95d80cc96bfa15c71fc5f435f2b96855e6`.

## Preflight and estimate

CPU preflight passed governance, syntax, the 600-record/category contract,
all 1,200 official LQ/GT hashes and dimensions, all released checkpoint hashes,
the canonical project-result binding, the official metric-script hash, and
local LPIPS weights. No GPU work or output artifact existed at readiness time.

- One homogeneous 8xH200: 24--38 minutes wall time, center 30 minutes;
  approximately 3.2--5.1 GPU-hours, center 4.0.
- One homogeneous 8xH100: 30--48 minutes wall time, center 38 minutes;
  approximately 4.0--6.4 GPU-hours, center 5.1.
- Main uncertainty: official resolution distribution, GPFS throughput, and
  full-image LPIPS runtime.

The current Codex shell has no `nvidia-smi`, so it did not launch the GPU job.
On one 8-GPU node:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
smoke/launch_foundir_official_sow600_released_full_pipeline_v1_8gpu.sh
```

Monitor:

```bash
tail -f logs/foundir_official_sow600_released_full_pipeline_v1.log
```
