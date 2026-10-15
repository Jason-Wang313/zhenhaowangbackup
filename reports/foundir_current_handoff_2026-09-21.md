# FoundIR current handoff — 2026-09-21

> **SUPERSEDED:** This snapshot is historical. Current status and next action
> are in `reports/foundir_current_handoff_2026-09-23.md`. In particular, do not
> reuse this file's statements that Noise or Blur are unresolved.

## Mandatory compute-sizing rule

Future agents must recommend the number of GPU nodes from the actual workload, not from the number of available nodes.

Before proposing multi-node execution, estimate all of the following:

1. single-node wall time for the complete remaining job;
2. model/gradient size and compute per optimizer step;
3. cross-node bandwidth and expected communication time per step;
4. whether changing world size changes global batch, optimizer-step count, learning-rate schedule, or checkpoint comparability;
5. engineering time for code changes, smoke tests, rendezvous, debugging, and user coordination;
6. expected net wall-clock saving after including that engineering time.

Use multi-node training only when the expected end-to-end saving materially exceeds implementation and coordination cost, and when the optimization change is scientifically acceptable. If coding, validation, and launch overhead is comparable to simply finishing on one 8-GPU node, recommend one node immediately. Do not optimize GPU utilization at the expense of total user time or experiment reliability.

Practical default:

- small/medium models and jobs under roughly 1–2 hours on 8 GPUs: use one 8-GPU node unless an already-tested multi-node launcher exists;
- embarrassingly parallel inference/evaluation: shard across nodes without NCCL;
- long training jobs with an existing verified launcher and favorable networking: consider multi-node after a short communication benchmark;
- do not rewrite a live or ready-to-run job merely because extra nodes are temporarily available.

## Multi-node audit and current decision

A 3-node × 8-H100 NCCL TCP smoke completed successfully on `eth0`:

- 24 ranks, correctness pass;
- 16 MiB all-reduce: 0.06335 s;
- 64 MiB all-reduce: 0.07832 s;
- 256 MiB all-reduce: 0.25309 s;
- estimated 256 MiB bus bandwidth: 1.8933 GiB/s;
- network interface reports 10 Gb/s; RDMA was disabled.

The smoke proves feasibility, not that multi-node is optimal. The current Noise Restormer is only about 26M parameters and the original 8-GPU job is expected to finish in approximately 45–70 minutes. Time already spent implementing and validating a 24-GPU variant is comparable to a substantial fraction of the original run. Only two nodes remain available, and creating another 16-GPU variant would add more engineering and validation delay while changing the training batch/step trajectory.

**Current optimal decision: do not continue adapting the job to 16 GPUs. Run the already-tested single-node 8-GPU job now.** The 24-GPU files remain experimental evidence and are not the recommended current launch path.

## Current Noise experiment

Goal: one bounded Noise-only Restormer round after official v3 failure.

- training data: FoundIR training-only `05Noise`;
- train records/source groups: 54,585 / 1,755;
- initialization: project-trained Noise step-250, not a released FoundIR specialist;
- loss: RGB MSE, PSNR-prioritized;
- schedule: 1,000 steps, global batch 32, 8×H100/H200;
- official images/GT are not used in gradients;
- no alpha sweep;
- after training, test the fixed step-1000 checkpoint once on all 100 full-resolution official Noise images with the official `pyiqa` protocol; pass means freeze immediately.

Recommended launch:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash smoke/launch_foundir_noise_mse_restormer_v1_8gpu.sh
```

## Official status before this Noise round

- Low-light: pass;
- Haze: pass after frozen global correction (`22.2106 / .12359`);
- Rain: pass after Qwen-supported Rain fallback (`37.8092 / .05290`);
- Noise: unresolved;
- Blur: unresolved and should be handled only after Noise.

Do not touch or regress the frozen Low-light, Haze, or Rain paths while working on Noise.
