# FoundIR scaled EBT H100/H200 kernel benchmark v1 readiness — 2026-10-08

## Status

**CPU PREFLIGHT PASS; H100/H200 BENCHMARK NOT YET COMPLETED.**

The first launch attempt at `2026-10-08T09:50:21Z` stopped during hardware
preflight because the allocated node exposed four NVIDIA H100 80GB HBM3 GPUs,
while v1 initially required H200. It stopped before workers, data access, or
optimizer updates and produced no result. The failed run directory/log are
preserved under run ID `h200_benchmark_v1`. The corrected hardware gate accepts
four homogeneous H100 or H200 GPUs and still rejects mixed/unsupported nodes.

This is a bounded synthetic compute benchmark, not a scientific training run.
It does not read the scaled training manifest, any triple image, sealed data,
or the official FoundIR test. It saves timing/memory JSON only and cannot save
a model checkpoint.

## Frozen identities

- Config: `configs/foundir_compositional_ebt_scaled_kernel_benchmark_v1.json`
  - SHA-256: `cf13efdb3b04a7708591cb3994d9ed9b4ac3b9b9e0ec8ee1405bb822468d91e7`
- Benchmark worker:
  `smoke/benchmark_foundir_compositional_ebt_scaled_kernel_v1.py`
  - SHA-256: `15f12f7100096642248abe23d4d96a9d97e4264804ff73ab243494c3bd289db0`
- Four-GPU runner:
  `smoke/run_foundir_compositional_ebt_scaled_kernel_benchmark_v1_4gpu.sh`
  - SHA-256: `2394f04f8d1356d88b632cb73e838646d175f733701c9f5ad95835675ceb2301`

## Profiles

| Model | Exact parameters | Image / patch | Energy descent |
|---|---:|---:|---:|
| xxs | 11,266,048 | 128 / 16 | official-style 1 step |
| xxs | 11,266,048 | 128 / 16 | project 3 steps |
| smallish | 26,043,648 | 128 / 16 | official-style 1 step |
| smallish | 26,043,648 | 128 / 16 | project 3 steps |

Each profile scans local batches 8, 16, 32, 64, and 128, stopping larger
batches after an OOM. Each passing batch performs one warm-up and three timed
full optimizer updates, including differentiable candidate descent, loss
backpropagation through the energy gradient, gradient clipping, and AdamW.
Outputs include full-step time, images/s, peak allocated memory, and peak
reserved memory.

Four homogeneous H100/H200 GPUs are intentional: there are exactly four
independent profiles. Any remaining GPUs are not allocated merely to duplicate
measurements. Full training GPU topology will be decided from measured data.

## Preflight evidence

- Config JSON, Python compilation, shell syntax, and whitespace checks pass.
- Instantiated parameter counts match exactly for both model sizes.
- A CPU miniature full higher-order optimizer update returns finite loss and
  gradient norm.
- A no-CUDA worker invocation fails closed before creating an output JSON.
- The Codex shell exposes no CUDA device, so measured GPU results do not yet
  exist.

## Launch command and cost bound

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_SCALED_BENCHMARK_RUN_ID=h100_benchmark_v1 CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/run_foundir_compositional_ebt_scaled_kernel_benchmark_v1_4gpu.sh
```

- Estimated wall time: `0.08–0.75 hours` (about 5–45 minutes).
- Estimated allocated compute: `0.32–3.00 GPU-hours` on four H100/H200 GPUs.
- Terminal result:
  `checkpoints/foundir_compositional_ebt_scaled_kernel_benchmark_v1/h100_benchmark_v1/result.json`
- Main log:
  `logs/foundir_compositional_ebt_scaled_kernel_benchmark_v1_h100_benchmark_v1.log`

Do not infer a full 100k runtime until this benchmark finishes. Do not launch
full scaled training automatically after it; first choose model size, descent
rule, microbatch/accumulation, and GPU topology from the measured result and
record the resulting training decision.
