# FoundIR local action energy v1 qualification — 2026-10-17

Fresh `qualification_v1` completed successfully on eight NVIDIA H100 80GB
GPUs. The launcher ran from `2026-10-17T15:08:02Z` through `15:11:53Z` and
ended with status `0`. This is `.0642` wall-clock hours and `.5133` allocated
GPU-hours.

The 25-step run produced finite loss terms, a checkpoint, and the frozen
256-record source-disjoint development validation. It used global batch 32,
microbatch 2 per GPU and gradient accumulation 2. No official, sealed,
internal, parent-test, or frozen-660 data was accessed.

## Step-25 validation

| Output | PSNR | LPIPS | PSNR gain vs proposal | LPIPS reduction vs proposal |
|---|---:|---:|---:|---:|
| Proposal | 43.124160178005695 | .050362067760033824 | 0 | 0 |
| Frozen fidelity | 43.39715355634689 | .028955043449059303 | +.27299337834119797 | .02140702431097452 |
| Local action | 43.39715355634689 | .028955043449059303 | +.27299337834119797 | .02140702431097452 |

The selector accepted `0%` of non-safe alternatives and chose alpha `1.0` on
`100%` of regions. This confirms exact fail-closed behavior after a very short
qualification. It is not a method-quality conclusion: the network has not yet
had enough exposure to overcome the deliberately initialized `.05` decision
margin.

Checkpoint SHA-256:
`ec98308fefd05fffa92a2dd6d05e76dbc2c38363b7ad251ce220d23c2e8be5f4`.
Validation JSON SHA-256:
`36512bcf48f63aea88e161d611f3bf7888056aba6f737a99a0a37db69794d48f`.

Engineering qualification passes. The next permitted run is a fresh-ID,
from-zero formal schedule of 8,530 optimizer steps. Never initialize it from
the qualification checkpoint. Based on the measured run, startup/validation
overhead dominates 25 steps; retain the prior conservative formal estimate of
`3–5` wall-clock hours and `24–40` allocated GPU-hours until pass-1 supplies a
better measurement.
