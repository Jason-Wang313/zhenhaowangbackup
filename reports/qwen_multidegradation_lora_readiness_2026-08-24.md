# Qwen multi-degradation LoRA readiness — 2026-08-24

The advisor-approved Qwen stage is CPU-complete and ready for one autonomous
four-H100 run. No FoundIR training or test data was accessed.

## Frozen data

- Contract SHA-256: `b2a28d44bb599a7044463438f03944503cbc3def0c7c5532af952a6b3d65df49`.
- Manifest: 12,000 records; SHA-256
  `0d5fa12abc387ad857c85ad574dcf8ecf7dc158c443d470c096610de04255599`.
- Split: 9,600 train / 1,200 validation / 1,200 fixed test.
- Every split is 10% clean; its degraded remainder is exactly 1:1:1
  single/double/triple. Combinations and every active L2--L5 severity are
  exactly balanced.
- Original source identity is split before cropping. All haze records use only
  MiO100 sources with verified aligned depth maps.
- All 12,000 PNGs and 12,000 per-record immutable generation records exist.
  The aggregate generation result SHA-256 is
  `73653c445e2253e472838d8dd912777891a5f7809d0a45724c41ca0a9037a90c`.

One documented generation edge case was corrected: AgenticIR's linear mapping
is undefined when `vmax == vmin`. Only for that mathematical limit, direct
multiplicative luma calibration is used. The first attempt preserved 11,985
valid records; recovery generated the remaining 15 without overwriting the
existing records.

## Frozen training and evaluation

The configuration SHA-256 is
`f0e0eb66910571ad6585259fae7f9063fc362989e3911f96bfe1152a508a4d20`.
Qwen2.5-VL-7B remains offline and frozen except for rank-16 LoRA on all 160
visual attention/MLP linear modules. The ablation compares the same LoRA with
the 44,574,464-parameter visual merger frozen versus trainable. The tokenizer,
language decoder, LM head, patch embed, and all non-adapter visual weights are
frozen.

The progressive schedule is one retained-data epoch at each stage:

1. 3,840 clean + single records;
2. 6,720 retained clean/single + double records;
3. 9,600 retained clean/single/double + triple records.

This is 630 optimizer steps at global batch 32. Four H100s each process eight
images per step, minimizing synchronization and accumulation overhead. The
two variants are trained sequentially so each uses all four GPUs. Validation
selects the winner; only that checkpoint is then evaluated once on the fixed
synthetic test.

The full model identity/hash preflight passed, including all five 16 GB Qwen
shards. A meta-device architecture audit matched exactly 160 LoRA modules and
11,151,360 adapter parameters; the trainable-merger variant has 55,725,824
trainable parameters total. Twenty-nine focused contract/data/LoRA/trainer
tests, Python compilation, shell syntax, and `git diff --check` passed.

This coding terminal has no visible CUDA device, so no GPU result or final
checkpoint is claimed yet. The single external launch action is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/launch_qwen_multidegradation_lora.sh
```

The launcher uses system tmux if executable and otherwise a detached nohup
fallback. It records persistent training and four-GPU telemetry logs. Training
resumes from immutable curriculum-stage checkpoints after interruption.
