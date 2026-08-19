# Qwen Lowlight supervised overfit40 readiness — 2026-08-19

## Decision

**CPU readiness: PASS. GPU gate: NOT RUN.** One H200 may now be used for the
bounded advisor-route code/gradient/memorization smoke. Do not start the 8k
supervised pilot until this gate completes and the result is reviewed.

This replaces none of the preserved frozen-feature/ridge evidence. It tests a
different architecture: supervised token-level cross-entropy with the entire
Qwen visual path trainable and the language decoder frozen.

## Frozen experiment

- Config: `configs/qwen_lowlight_supervised_overfit40.json`
- Config SHA-256:
  `95671500cd2470444722db31bc8d9d648e6cdb57dea09574014936fc0f883b5b`
- Manifest: `manifests/low_light_pilot_10000.csv`
- Manifest SHA-256:
  `e1ea9132bc0d5e55b30998c51c8e37a0363f29b2a23e7851ee804f3100d4a784`
- Selection bundle SHA-256:
  `4317f00856ec590361c44660445208251f9db8e3e28c699fe9487daff1537e69`
- Selection: 8 train sources x all five levels = 40 records
- Per-level count: 8 each for L1, L2, L3, L4, and L5
- Selected sources: `lsdir_0000764`, `lsdir_0000057`, `lsdir_0000626`,
  `div2k_0273`, `lsdir_0000496`, `div2k_0599`, `div2k_0514`, and
  `lsdir_0000648`
- Validation split touched: no
- Internal test split touched: no
- Official FoundIR touched: no

## Training and restart protocol

- Trainable: all 676,550,144 parameters in `model.visual`, including the
  44,574,464-parameter merger/connector
- Frozen: 7,615,616,512 parameters in `model.language_model` and `lm_head`
- Precision: BF16
- Device count: one GPU; GPU 0 by default
- Visual tokens per audited image: 117
- Micro-batch: 1
- Gradient accumulation: 1
- Optimizer: fused AdamW
- Learning rate: `2e-5`, constant
- Weight decay: `0`
- Optimizer steps: 400 (10 passes over 40 records)
- Gradient checkpointing: enabled
- Attention: deterministic math SDPA
- Sampling for evaluation: greedy, at most 64 new tokens

The launcher executes two separate Python processes. The first process trains
through step 2, proves gradient/freeze scope, writes hashed visual and optimizer
checkpoints, and exits intentionally. The second process verifies and reloads
that checkpoint, restores optimizer and RNG state, and continues to step 400.
No existing differing artifact is overwritten.

## Automatic pass conditions

- step-2 checkpoint was resumed in a fresh process;
- every trainable visual tensor received a finite gradient and all three visual
  groups had nonzero gradient norm;
- frozen modules had no gradients or detected updates before or after restart;
- final deterministic probe loss is at most `0.5` and at most 25% of its
  initial value;
- valid structured-output rate is 100%;
- degradation-type accuracy is 100%;
- severity/combined-label accuracy is 100%; and
- every L1-L5 subgroup reaches 100% accuracy.

Failure is a legitimate gate result. A failure does not authorize the 8k pilot,
diffusion conditioning, or FoundIR evaluation; it must be diagnosed and
reviewed first.

## Artifacts produced by the GPU run

All runtime artifacts are written below
`checkpoints/qwen_lowlight_supervised_overfit40/`, including:

- exact config, preflight result, and selected-record manifest;
- one immutable JSON metric per optimizer step;
- first-backward gradient audit;
- frozen-parameter audits before and after restart;
- step-2 and step-400 visual/optimizer checkpoints with SHA-256 metadata;
- initial/final deterministic loss evidence;
- raw generated output for all 40 records;
- structured validity, type/severity metrics, per-level accuracy, and confusion
  matrix; and
- terminal `training_summary.json` with environment and peak GPU allocation.

The persistent log is `logs/qwen_lowlight_supervised_overfit40.log`.
