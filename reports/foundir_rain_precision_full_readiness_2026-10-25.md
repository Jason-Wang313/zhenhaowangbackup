# Full-data FP32 Rain FLUX: launch readiness

## Decision and scope

User authorized execution after the completed 32-image fitting diagnosis:
the train set reached 35.09303906168583/.04928288975497708 but development
remained 32.44553214852097/.08547242213506251. The next hypothesis is that
the corrected FP32 state and matched 512 training resolution should be
tested on the full source-diverse Rain pool. This is a hypothesis, not a
promise of 34.2/.08 or proof that precision alone caused prior improvement.
No pixel refiner or new inference branch is introduced.

## Frozen recipe

- Config: `configs/foundir_rain_flux2_precision_full_v1.json`.
- Initialize independently from the original direct-endpoint step-2,000
  LoRA, SHA-256 `eff05dccc0d9bcdf538305ac4095304348a876eb6cb3a4ded3ea27767901f36a`.
  Fresh AdamW, learning rate 1e-5, zero weight decay, gradient clipping1.
  Never initialize from the 32-image fit or full-data qualification outputs.
- All 38,334 training records / 5,780 source groups remain eligible. Use
  deterministic shuffled source cycles and shuffled within-source record
  cycles. This weights sources equally, not each individual record equally.
  The 3,000-step schedule gives 96,000 presentations; it is not claimed to
  be a fixed number of exhaustive record epochs. Actual unique coverage is
  recorded at completion.
- Random paired 512 crops, paired D4 geometry, no resizing. Exactly 35
  short-sided training records need reflection padding before augmentation.
  Their original pixels remain intact; padding is training-only.
- Frozen 100-source development rows/crops and metric implementation unchanged.
  The same 32 fixed center-crop training probes remain separate from the
  development gate and do not define training data exposure.
- FP32 trainable LoRA and Adam states; BF16 compute; shared FP32 endpoint.
  Same rank32, 50 LoRA target modules, eight loss terms and coefficients.
  Same frozen Restormer behavior teacher, training only. Inference remains
  one FLUX endpoint forward.
- One homogeneous 8xH100/H200 node, batch32 = 8 ranks x 4 accumulated microbatches.
  No cross-node training. No official/sealed/internal/parent-test access.

## Memory and integrity

Training loading uses two spawned CPU workers per rank with prefetch factor2.
It does not retain all decoded pairs. The first byte-read of each image is
SHA-256 checked against the frozen manifest; size/mtime/ctime changes force
revalidation. Geometry is checked during decode. Preflight checks all unique
image paths once in parallel on rank0, hashes model/init/teacher assets and
all fixed probe/development images. It does NOT eagerly hash all training
pixels; those are verified when used. This distinction is intentional.

The frozen input manifest SHA-256 remains
`2a65ff9b353f4537ac103da387059bba80124b4722c91c0ac1bc77e6fb569317`.
Canonical training-row digest:
`0207fd65e5a37628023393a93bc9e80d1f1df85708d6677a94cc656dffea9fb0`.
Development digest:
`1652d4e46634c1ee248e1984ea2aeaec271d8491a67dcc3acf12838cb181d783`.
Probe digest:
`cfd29a617d50b25c87ba8d3e246566cd789fae412b057a677d2a6c43afdd969a`.
Full-data config SHA-256:
`382bb767dfc6dae743f8c70d690f2307b640acc1eb1ff06d25ce557aef981237`.

## Gates and budgets

Fresh qualification: 25 steps, evaluations 0/25. Must prove full-data loading,
finite gradients, observed FP32 updates, safe memory, F32 saved LoRA,
optimizer artifacts, and valid matched metrics. Not a scientific gate.
Formal: separate initialization, 3,000 steps, evaluations at
0/250/500/1000/2000/3000. It requires matching qualification config, source,
training/probe hashes. No automatic transition. New IDs preserve prior output.
No new data, loss, threshold or schedule tuning during either frozen run.

Prior 32-image diagnostic ran at median 1.5413306262344122 seconds/step.
This is only a compute reference; random full-data IO may cost more.
Provisional full-data qualification .10-.50 wall hours / .8-4 GPU-hours;
formal 1.5-3.0 wall hours / 12-24 allocated GPU-hours on eight GPUs.
Replace estimates with measured qualification throughput before formal launch.

## Verification

All eight CPU tests pass: four pre-existing precision/loss/checkpoint tests
plus four full-data tests covering source balance/within-source cycling,
rank partitioning, train-only restriction, deterministic paired crops,
reflection padding, changed-file rejection, spawned DataLoader behavior,
no decoded cache, and exact unchanged development/probe identities.
Both launcher shell syntax checks and Python compilation pass.
Full preflight passed with 38,334 records / 5,780 sources / 100 development
records / 35 padding records. GPU qualification has NOT run yet.

## Commands on the user's GPU node

Qualification:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_flux2_precision_full_v1_tmux.sh
```

Log: `logs/foundir_rain_flux2_precision_full_v1_qualification_v1.log`.
Session: `rain_full_qualification_v1`, socket `zhenhaowang_jobs`.
Output: `checkpoints/foundir_rain_flux2_precision_full_v1/qualification_v1/`.

Only after confirming qualification and updating the budget, separate formal:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_flux2_precision_full_v1_tmux.sh formal formal_v1
```

The Codex environment has no GPU; a tmux session created there would not
constitute a GPU launch. Inspect shared logs after the user starts on their node.
