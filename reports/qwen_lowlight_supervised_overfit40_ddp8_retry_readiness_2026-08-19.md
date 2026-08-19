# Qwen Lowlight supervised overfit40 DDP8 retry readiness — 2026-08-19

## Current gate result

The original advisor-route supervised Qwen-VL overfit40 gate **failed**. It
completed all 400 optimizer steps, passed JSON validity, gradient-scope,
frozen-parameter, checkpoint/resume, and loss-decrease checks, but memorized
only 35 of 40 severity labels (`87.5%`). The 8,000-record supervised pilot
remains unauthorized.

Authoritative failed-run evidence:

- log: `logs/qwen_lowlight_supervised_overfit40.log`
- summary: `checkpoints/qwen_lowlight_supervised_overfit40/training_summary.json`
- predictions: `checkpoints/qwen_lowlight_supervised_overfit40/predictions.json`
- failed-run Git revision: `c218c6093e78a2b0d43ec73956bad841bb9d8cbc`

The five misses are valid adjacent middle-level predictions, not malformed
labels or corrupt images:

| Pair | True | Predicted |
|---|---:|---:|
| `lsdir_0000057_L2` | 2 | 1 |
| `lsdir_0000057_L3` | 3 | 2 |
| `div2k_0273_L2` | 2 | 3 |
| `div2k_0273_L3` | 3 | 4 |
| `div2k_0599_L3` | 3 | 4 |

The manifest hashes and calibrated target ratios for these records are valid.
The fixed-order batch-one run repeatedly showed middle-level loss spikes and
late interference. Its final full generation therefore did not match its low
average teacher-forced probe loss. This supports a bounded balanced-batch
memorization retry; it does not support changing labels or starting the pilot.

## DDP8 retry contract

The retry is a new output path and does not overwrite or reinterpret the
failed evidence.

- same exact 40-record selection bundle:
  `4317f00856ec590361c44660445208251f9db8e3e28c699fe9487daff1537e69`
- same 8 sources and exactly 8 records at each L1–L5 level
- all eight H200 GPUs, one complete five-level source group per rank
- per-rank microbatch: 5 (one source's L1–L5)
- gradient accumulation: 1
- effective/global batch: 40
- every optimizer update sees every selected record exactly once
- 200 optimizer updates / 8,000 declared sample presentations
- frozen language decoder and LM head; full visual path trainable
- deterministic math SDPA, BF16, fused AdamW, gradient clipping
- five-step linear warmup followed by cosine decay from `2e-5` to `2e-6`
- real stop-at-step-2 checkpoint and fresh-process resume
- strict required result: 100% valid JSON, type accuracy, severity accuracy,
  and per-level accuracy, plus all gradient/freeze/resume/loss checks

The eight-GPU layout removes the eight serial source microbatches that an
equivalent one-GPU global update would require. That makes an eight-GPU
wall-time benefit plausible, but it is not yet measured. The launcher records
two-second utilization, memory, power, and SM-clock evidence for all GPUs; no
claim of full utilization or maximum efficiency is authorized before those
measurements are inspected.

## CPU validation completed

- Python compilation passed.
- Launcher shell syntax passed.
- Eleven supervised/distributed unit tests passed.
- Workspace tmux `3.7` and the exact interpreter/module launcher were verified.
- Offline preflight passed with all 16 model files fully SHA-256 checked.
- Preflight config SHA-256:
  `a19ead5a89975292ebc83604b42989d37553d90db707838b1db6ca8f6f5bec2b`
- Qwen model revision:
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- No official FoundIR image and no synthetic test-split record is used.

## GPU launch and evidence

From the workspace root, the one-line launch is:

```bash
bash smoke/launch_qwen_lowlight_supervised_overfit40_ddp8_tmux.sh
```

Attach after launch with:

```bash
.tools/tmux-env/bin/tmux attach -t qwen_vlm_overfit40_ddp8
```

The terminal and GPU evidence persist at:

- `logs/qwen_lowlight_supervised_overfit40_ddp8.log`
- `logs/qwen_lowlight_supervised_overfit40_ddp8_gpu.csv`
- `checkpoints/qwen_lowlight_supervised_overfit40_ddp8/`

The tmux shell remains present after either success or failure. The runner
always writes `QWEN_SUPERVISED_DDP8_TERMINAL_EXIT_STATUS <n>` and a UTC end
marker, so a nonzero automatic gate result remains inspectable and is not
mistaken for a missing log or an unexplained crash.
