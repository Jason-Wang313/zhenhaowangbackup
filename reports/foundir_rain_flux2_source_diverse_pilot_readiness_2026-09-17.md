# FoundIR Rain FLUX.2 source-diverse pilot readiness — 2026-09-17

The user explicitly authorized the same minimal source-diverse route that made
Haze learn, after a read-only scalar diagnosis showed that Rain cannot reach
the Stage-2 PSNR target through `Input + alpha * (FLUX - Input)`. On the
existing sixteen source-disjoint Rain records, the best global unquantized
alpha reaches only 31.634688 PSNR; even a nondeployable per-image GT oracle
reaches only 33.192510, below the 34.2 target.

The frozen pilot uses FLUX.2 Klein Base 4B with a fresh rank-32 LoRA, isolated
Rain only, the unchanged pure-noise-to-clean rectified-flow objective, global
batch 32, 2,000 optimizer steps, and official 50-step inference. It adds no
alpha correction, Qwen degradation condition, mixed degradation, refiner, or
new loss. There is no automatic continuation.

The FoundIR development corpus contains 116 Rain validation source groups.
Sixteen were already consumed by the earlier capacity gate and oracle audit.
The frozen gate therefore uses every one of the remaining 100 untouched
source groups exactly once rather than reusing consumed sources or repeating
groups to claim 128 records. Training uses 1,580 unique source groups selected
from 5,729 available groups after the same exclusion. Train/validation overlap
is zero.

Frozen identities:

- config: `configs/foundir_rain_flux2_source_diverse_pilot_v1.json`, SHA-256
  `ab55489d25a03c1801c7d425e9fb42b0aa92d81b5b0d433bd9c9f4fefb1a2f71`;
- manifest: `manifests/foundir_rain_flux2_source_diverse_pilot_v1.csv`, SHA-256
  `ace5c8cb6da1a1628fe3ab3f7eec9e0a50feb1a12bb2b26732f06726f552636b`;
- metadata: `manifests/foundir_rain_flux2_source_diverse_pilot_v1.json`, SHA-256
  `e115d29f2fe51fbe84001c2d0a9dbc35ba41f2eacc36283ada9e7cb3b0c8b624`.

Fifteen combined focused tests pass. Python compilation, shell syntax,
governance, source isolation, all 3,360 frozen LQ/GT image and crop decodes,
parent exclusions, and model hashes pass. Official FoundIR test and sealed
internal data were not accessed.

One homogeneous 8xH100 instance is required. Estimated complete launcher wall
time is 42--58 minutes, center about 50 minutes, including preflight, loading,
training, fixed inference, aggregation, and reporting. Aggregate GPU usage is
5.6--7.7 GPU-hours, center about 6.7. Shared-storage decoding and official
50-step inference are the main uncertainty. Parallel duplicate instances do
not shorten this single sequential trajectory.

This Codex shell has no `nvidia-smi`, so no GPU process was started here.
