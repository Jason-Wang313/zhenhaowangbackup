# Qwen Lowlight supervised 1/2/4/8-H200 scaling-gate readiness — 2026-08-19

## Outcome

CPU-side implementation and fail-closed preflight pass. GPU timing has not yet
run in the coding-agent terminal because that terminal exposes no NVIDIA device
or `nvidia-smi`. This report authorizes only the bounded scaling gate. It does
not authorize or claim completion of the 8k supervised pilot, real-domain
validation, diffusion conditioning, FoundIR evaluation, or a 100k corpus run.

## Frozen benchmark contract

- Scaling config:
  `configs/qwen_lowlight_supervised_scaling_1_2_4_8.json`
- Scaling config SHA-256:
  `fbbaf2a4063f96096f32bd08df34ebe29c0ae915c72e8d0667b684df3684dff3`
- Passing DDP overfit base-config SHA-256:
  `a19ead5a89975292ebc83604b42989d37553d90db707838b1db6ca8f6f5bec2b`
- Exact 40-record selection-bundle SHA-256:
  `4317f00856ec590361c44660445208251f9db8e3e28c699fe9487daff1537e69`
- Authoritative full-hash preflight SHA-256:
  `abf49c48e8188128a70268112698de9722673a8891cee54906ac07689cbd4df0`
- World sizes: `1`, `2`, `4`, and `8` H200 GPUs, run sequentially.
- Per-rank microbatch: one complete L1–L5 source group, or 5 records.
- Gradient accumulation: `8`, `4`, `2`, and `1`, respectively.
- Global batch: exactly 40 records for every optimizer step at every world
  size.
- Each world size uses 3 warmup optimizer steps and 10 measured optimizer
  steps with the same seed, BF16, deterministic math-SDPA, gradient
  checkpointing, AdamW settings, and trainable/frozen parameter scope.
- The selection projection is 1,000 optimizer steps. Fixed startup excludes
  the benchmark-only warmup, and the primary criterion is minimum projected
  end-to-end training wall time.

The gate fails unless all four correctness audits pass, all eight physical
GPUs have persistent telemetry for every stage, a multi-GPU configuration is
at least 1.05 times the measured one-GPU throughput, a multi-GPU configuration
reduces projected wall time, the selected configuration uses more than one
GPU, and each multi-GPU measured-loss trajectory stays within 5% relative
deviation of the deterministic one-GPU reference.

## Persistent execution and evidence

- Benchmark worker: `smoke/benchmark_qwen_lowlight_supervised_scaling.py`
- Preflight and selector:
  `smoke/validate_qwen_lowlight_supervised_scaling.py`
- GPU runner:
  `smoke/run_qwen_lowlight_supervised_scaling_1_2_4_8.sh`
- Persistent tmux launcher:
  `smoke/launch_qwen_lowlight_supervised_scaling_tmux.sh`
- Workspace-local tmux: `.tools/tmux-env/bin/tmux` (`tmux 3.7` verified)
- Main log: `logs/qwen_lowlight_supervised_scaling_1_2_4_8.log`
- Terminal-event log:
  `logs/qwen_lowlight_supervised_scaling_1_2_4_8_events.log`
- Two-second all-GPU telemetry:
  `logs/qwen_lowlight_supervised_scaling_1_2_4_8_gpu.csv`
- Immutable output root:
  `checkpoints/qwen_lowlight_supervised_scaling_1_2_4_8/`

The tmux launcher creates a persistent shell, enables `remain-on-exit`, and
sends the short checked runner to that shell. The runner appends UTC launch,
per-world, selector, and terminal-exit markers even when a command fails. A
missing tmux session is therefore not used as a proxy for job status.

## CPU verification completed

- Python compilation passed for the scaling helper, benchmark, validator, and
  tests.
- Bash syntax passed for the GPU runner and tmux launcher.
- 18 supervised-schema, DDP-scheduling, scaling-selection, and fail-closed
  summary-validation unit tests passed.
- Runner `--check` and tmux launcher `--check` passed.
- Skip-hash preflight passed.
- Full preflight verified all 16 model files, 729 checkpoint tensors,
  8,292,166,656 parameters, exact prompt/tokenization, all 40 selected images,
  and the 1/2/4/8 global-batch contracts.
- The coding terminal reports PyTorch `2.7.0+cu128`, CUDA runtime `12.8`,
  `torch.cuda.is_available() == False`, zero CUDA devices, and no
  `nvidia-smi`; GPU execution must therefore use the user's shared-workspace
  H200 interface.

## Exact GPU handoff

Run these two launch lines one at a time in the GPU interface:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
bash smoke/launch_qwen_lowlight_supervised_scaling_tmux.sh
```

Attach or monitor with:

```bash
.tools/tmux-env/bin/tmux attach -t qwen_vlm_scaling_1_2_4_8
tail -n 100 -f logs/qwen_lowlight_supervised_scaling_1_2_4_8.log
```

After the selector writes `scaling_summary.json`, stop and review the selected
GPU count and telemetry before preparing or launching the 8k pilot.
