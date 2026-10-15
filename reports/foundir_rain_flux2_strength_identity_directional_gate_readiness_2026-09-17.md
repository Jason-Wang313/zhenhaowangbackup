# Rain FLUX.2 strength/identity directional gate readiness

Status: CPU-ready; GPU not started from the current shell because
`nvidia-smi` is unavailable.

The authorized experiment is frozen to the exact Rain step-2000 LoRA and
optimizer, 250 additional optimizer steps (absolute step 2250), LR 2.5e-5,
and the unchanged rectified-flow objective. Sampling deterministically cycles
through heavy, moderate, and near-clean training crops. Heavy/moderate targets
remain GT; only near-clean targets are changed to LQ identity targets. There
is no new model, loss, alpha, refiner, router, or automatic continuation.

The manifest reuses all 1,580 parent training records and exact crops. Its new
train-split development set contains 48 source-disjoint records: 16 heavy,
16 moderate, and 16 near-clean. It excludes the parent training groups and
the earlier capacity-gate groups. Validation, sealed internal, and official
FoundIR test data were not accessed.

Directional checks compare the exact parent and candidate on the same 48
records:

- near-clean PSNR: at least +0.5 dB;
- moderate PSNR regression: at most 0.1 dB;
- heavy PSNR regression: at most 0.1 dB;
- overall PSNR: at least +0.2 dB;
- overall LPIPS regression: at most 0.005;
- fixed visuals remain a required human review.

Verification completed:

- focused tests: 7/7 passed;
- shell syntax and Python compilation: passed;
- complete data/model hash and image preflight: passed;
- advisor governance check: hold resolved;
- terminal result: absent.

Frozen identities:

- config SHA-256: `fa9e5fe097be5e18b46de2f904d94374771dd6fa3a686c5f3bf624459e2c8fc2`;
- manifest SHA-256: `6e39096a93418a54bd573dee2affdee9421f2167060e7e011a3af070d45acfd9`;
- metadata SHA-256: `ebc232836196febe78be29baf1a53b0833bf3a84179fb04366f6051aceeac124`.

Expected total wall time is 40-55 minutes from launcher invocation (center
47), including the repeated CPU/HDD preflight. Expected GPU-occupied time is
28-40 minutes (center 34) on one 8xH100/H200 instance, or 3.7-5.3 GPU-hours
(center 4.5). Import/model loading, HDD cache, and the two 48-record 50-step
evaluations dominate uncertainty.

Launch only with:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && bash smoke/launch_foundir_rain_flux2_strength_identity_directional_gate_v1_8gpu.sh
```
