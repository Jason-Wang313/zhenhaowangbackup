# FoundIR VLM-energy real-data Generalist continuation readiness — 2026-10-11

## Scope and governance

This package continues the Generalist only. It contains no specialist branch.
Optimizer input is restricted to the frozen `train` split; checkpoint selection
uses only the frozen `validation` split. No `internal_test`, official, sealed, or
parent-test pixels are loaded.

Frozen manifest:

- path: `manifests/foundir_five_type_pairs_v1.csv`
- SHA-256: `e8c018879029c0bcc2da8d074cc4999541158ca97d07f48aadbcdddcdccc3ca7`
- train: `654,881`
- validation: `16,461`
- protected internal test: `14,153` (never loaded)

## Implemented continuation contract

- weight-only initialization from the selected synthetic joint-phase
  checkpoint; AdamW is intentionally reset for the real-domain continuation;
- three real-data passes;
- equal sampling mass for singleton, double, and triple tiers, followed by
  equal category sampling inside each tier;
- deterministic crop/augmentation and deterministic balanced schedules;
- EMA with decay `0.9999`, updated after every optimizer step;
- raw and EMA validation after passes 1, 2, and 3;
- resumable pass-boundary checkpoints containing raw weights, EMA weights,
  optimizer state, exact initialization hash, world size, and global batch;
- 8/24/32/40-GPU profiles; the recommended formal profile remains 24 GPUs.

Core files:

- `configs/foundir_vlm_energy_real_generalist_v1.json`
- `smoke/preflight_foundir_vlm_energy_real_generalist_v1.py`
- `smoke/benchmark_foundir_vlm_energy_real_io_v1.py`
- `smoke/train_foundir_vlm_energy_real_generalist_v1.py`
- `smoke/run_foundir_vlm_energy_real_io_v1.sh`
- `smoke/run_foundir_vlm_energy_real_generalist_v1_qualification_8gpu.sh`
- `smoke/run_foundir_vlm_energy_real_generalist_v1_multinode.sh`
- `smoke/qualify_foundir_vlm_energy_real_generalist_v1_8gpu.sh`

## Completed CPU checks

The CPU preflight passed against the epoch-17 checkpoint:

- frozen initialization:
  `checkpoints/foundir_vlm_energy_intersection_v1/train/formal_3node_v1/model_epoch_017_resume.pt`;
- checkpoint SHA-256:
  `46048543d3112e8b47ffeb86d17fcb9800857d422e3f8056de0e197578b24bd9`
- checkpoint epoch / step: `17 / 21,250`
- model tensors: `767`
- full manifest SHA-256 and all split/category counts verified;
- representative path-existence checks: `176`;
- report: `reports/foundir_vlm_energy_real_generalist_v1_preflight.json`.

A 1,024-pair, single-worker full-image decode benchmark on the Codex-side shared filesystem
completed in `265.231030` seconds:

- `3.860785` pairs/s for one process;
- `0.104320` decoded GiB/s;
- report:
  `checkpoints/foundir_vlm_energy_real_generalist_v1/io_benchmark/codex_readiness_v1.json`.

This one-process diagnostic is not a substitute for per-GPU-node concurrent
I/O measurement. Every formal node must run the 32-worker benchmark and exceed
`10 pairs/s`; its launcher requires that exact PASS result.

The same shared filesystem then passed a 32-worker concurrency qualification
over 264 pairs:

- `71.718757` pairs/s;
- `1.883744` decoded GiB/s;
- throughput gate: PASS;
- report:
  `checkpoints/foundir_vlm_energy_real_generalist_v1/io_benchmark/codex_concurrent_readiness_v1.json`.

CPU resume compatibility also passed: the epoch-17 state loaded strictly into
all `40,197,726` parameters with zero missing or unexpected keys, and an EMA
copy initialized and updated successfully.

## GPU qualification — PASS

The isolated qualification completed successfully on one `8 x NVIDIA H100
80GB HBM3` node (the allocated node was H100, not H200):

- status: `foundir_vlm_energy_real_generalist_qualification_pass`;
- four optimizer steps, global batch `24`;
- wall time inside the trainer: `32.080162` seconds;
- initial checkpoint SHA-256:
  `46048543d3112e8b47ffeb86d17fcb9800857d422e3f8056de0e197578b24bd9`;
- qualification checkpoint SHA-256:
  `d2e7cef883f3d95f44a0c2d08d94e10ffcfe4831efb84185522961f3c2668642`;
- no traceback, OOM, NCCL, non-finite, or protected-data access;
- result:
  `checkpoints/foundir_vlm_energy_real_generalist_v1/qualification_epoch17_v1/result.json`.

The 264-record raw/EMA validation is a functional diagnostic only and is not
the matched 660-image comparison. Raw was `24.713677 / .188829`; EMA was
`24.711906 / .188869` after only four updates, as expected.

After that qualification, synthetic epoch 18 completed and improved every
development tier over epoch 13. The selected formal initialization is now
`model_epoch_018_resume.pt`, SHA-256
`666fb152b50024357b6c44ea96469012e5405acb67a4ed08ad5618da1aca182e`.
The epoch-17 qualification proves the pipeline but cannot hash-qualify the new
initialization; rerun the same four-step qualification using epoch 18 before
formal launch.

### Epoch-18 qualification update — PASS

The hash-bound epoch-18 rerun passed on 8 x NVIDIA H200:

- config SHA-256:
  `3c6c8f50092f35331e39c498332f538ba0d5e9ef942f7bbacd028a0c6e79e73e`;
- initialization SHA-256:
  `666fb152b50024357b6c44ea96469012e5405acb67a4ed08ad5618da1aca182e`;
- four optimizer steps, raw/EMA validation, and checkpoint write: PASS;
- trainer wall time: `31.209208` seconds;
- no traceback, OOM, NCCL, non-finite, or protected-data access;
- result:
  `checkpoints/foundir_vlm_energy_real_generalist_v1/qualification_epoch18_v1/result.json`.

All four current H200 nodes passed their 4,096-pair, 32-worker I/O gates at
`121.886`, `121.029`, `154.242`, and `136.736` pairs/s. A current-IP four-node
NCCL/profile benchmark is the only remaining prelaunch gate.

Per-node I/O qualification:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && VLM_ENERGY_REAL_IO_RUN_ID="$(hostname)_real_v1" VLM_ENERGY_REAL_IO_RECORDS=4096 VLM_ENERGY_REAL_IO_WORKERS=32 bash smoke/run_foundir_vlm_energy_real_io_v1.sh
```

One-node GPU qualification:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && VLM_ENERGY_REAL_RUN_ID=qualification_epoch17_v1 VLM_ENERGY_REAL_INIT_CHECKPOINT=checkpoints/foundir_vlm_energy_intersection_v1/train/formal_3node_v1/model_epoch_017_resume.pt VLM_ENERGY_REAL_MAX_STEPS=4 VLM_ENERGY_REAL_MAX_VALIDATION_RECORDS=264 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/run_foundir_vlm_energy_real_generalist_v1_qualification_8gpu.sh
```

The combined one-command node qualification (I/O followed by GPU) is:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/qualify_foundir_vlm_energy_real_generalist_v1_8gpu.sh
```

The user selected the four-node / 32-GPU profile on 2026-10-12: batch 2 per
GPU, global batch 64, 10,233 steps per pass, and 30,699 steps across three
passes. The initial estimate is `25–29 wall hours` and `800–928 GPU-hours`.
Replace this range with an estimate derived from the actual four-node launch
or a matching benchmark once available.
