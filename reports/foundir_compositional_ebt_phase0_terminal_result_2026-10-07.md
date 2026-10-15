# Compositional EBT Phase-0 terminal result — 2026-10-07

> **DIAGNOSTIC ONLY / 非主线。** This zero-update readiness result is not a
> trained restoration model, a FoundIR metric result, or proof that the paper
> hypothesis succeeds.

## Terminal status

- Status: `zero_update_phase0_pass`.
- Start: `2026-10-07T18:14:47Z`.
- End: `2026-10-07T18:15:05Z`.
- Wall time: 18 seconds (`0.005` hours).
- Hardware: one `NVIDIA H100 80GB HBM3`.
- Approximate compute: `0.005 GPU-hours`.
- PyTorch: `2.7.0+cu128`.
- Optimizer steps: `0`.
- Model updates/checkpoints: none.
- Triple endpoint accessed: false.
- Sealed internal accessed: false.
- Official FoundIR test accessed: false.

## Frozen identities

- Config: `configs/foundir_compositional_ebt_phase0_v1.json`.
- Config SHA-256:
  `b79d3cfec14f55a26f8563e9488fa8f6ed57c1f55ac15467ea5642caabaf0cea`.
- Official EBT source revision:
  `19420cbeae655bbf11930219a675ade6897019e8`.
- Result:
  `checkpoints/foundir_compositional_ebt_phase0_v1/gpu_smoke_v1/result.json`.
- Result SHA-256:
  `74d19448da5a498e0e5cc6e18fa2fcf78ddb19f87a88c3fb211c265b01ac22ff`.
- Persistent log:
  `logs/foundir_compositional_ebt_phase0_v1_gpu_smoke_v1.log`.

## Gate results

Both matched arms contain exactly `143,360` trainable parameters.

| Arm | Max absolute candidate gradient | Energy decreased on all samples | Candidate gradients finite | Second-order parameter gradients finite |
|---|---:|---:|---:|---:|
| Plain EBT | `.0017837645` | yes | yes | yes |
| Compositional EBT | `.0017837645` | yes | yes | yes |

Plain sample energies changed from `.0090869591/.0423856676` to
`.0090741534/.0423767902`. Compositional energies were identical at
initialization and changed by the same amounts. This equality is expected:
the two models start from the same weights and the Plain dense descriptor
mixer is identity-initialized. The experiment verifies matched mechanics, not
a learned difference between the methods.

## Interpretation and stop

The Phase-0 implementation is mechanically viable on the target CUDA stack:
energy differentiation through image candidates and second-order training
gradients both work. It does not demonstrate restoration performance, OOD
generalization, or compositional advantage because no optimizer update ran.

Stop here. The next scientific action is to freeze a matched Plain versus
Compositional EBT pilot trained only on singleton/double data, including exact
data partitions, model size, update budget, checkpoint selection, initialization,
and a one-time unseen-triple policy. Do not launch training automatically from
this Phase-0 result.
