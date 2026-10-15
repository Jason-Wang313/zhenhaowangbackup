# FoundIR matched Compositional EBT lower-order pilot v1 — readiness

> **DIAGNOSTIC ONLY / 非主线。** This is a small-data learnability/causal
> pilot, not a FLUX.2 mainline result, SOW result, or paper-performance claim.

## Frozen question

Can matched Plain and atomic/pairwise Compositional Energy-Based Transformer
arms first learn restoration from singleton/double examples before either arm
is exposed to a held-out triple image?

The two arms use the same 146,432 tensors/parameters, initialization, binary
five-factor descriptor, training records, crop/augmentation presentation
schedule, optimizer, 300-step budget, three differentiable candidate-descent
steps, and final-step checkpoint rule. The only intervention is dense
descriptor-to-energy mixing versus diagonal atomic/pairwise mixing.

## Frozen data boundary and gate

- Gradient data: 60 parents (20 singleton + 40 double).
- Development data: 30 source-disjoint parents (10 singleton + 20 double).
- Training/checkpoint selection triple count: zero.
- This run may not open the 40 R0 held-out-triple images, the FoundIR 60-image
  triple endpoint, sealed internal data, or the official FoundIR test.
- Each arm must gain at least `0.05 dB` development PSNR over degraded input,
  improve development Smooth L1, reduce last-25 versus first-25 train loss by
  at least 10%, remain finite, and avoid all-black/all-white collapse.
- If either arm fails, stop without triple access. If both pass, freeze both
  checkpoints and seek a separate decision for one-time held-out-triple
  evaluation.

## Verification

- CPU preflight: `pass`; Plain and Compositional initial loss both
  `0.0009887457126751542`, with finite outputs and higher-order gradients.
- Existing focused EBT tests: `3/3 pass`.
- Python compilation, shell syntax, governance check, and `git diff --check`:
  pass.
- Governance:
  `foundir_compositional_ebt_lower_order_pilot_v1_design-5d467aa6f7d278cd`,
  resolved by the advisor/user's explicit `sure, execute new step now`.
- Config SHA-256:
  `a69e48b35fd1e58b03a864e8d1da15bfc5fdd1e3a8f5401c43b5ae67f6844d87`.
- Trainer SHA-256:
  `25ff4925d62adc36609645a80f240dc8cffc31c8e57df071666675cec83b80b7`.

## GPU execution status

The current Codex shell has neither `nvidia-smi` nor `/dev/nvidia0`, so the
authorized optimizer-update run has not started here. No GPU process or GPU
hours were created.

On one H100 80GB or H200 node:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && \
EBT_LOWER_ORDER_RUN_ID=pilot_v1 CUDA_VISIBLE_DEVICES=0 \
bash smoke/launch_foundir_compositional_ebt_lower_order_pilot_v1_gpu.sh
```

Prelaunch estimate: 10–30 minutes total wall time and `0.17–0.50 GPU-hours`
on exactly one GPU. This is an engineering estimate; the runner logs a measured
per-arm projection every ten optimizer steps.
