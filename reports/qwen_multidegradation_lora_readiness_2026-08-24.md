# Qwen multi-degradation LoRA readiness — 2026-08-24

The advisor-approved Qwen stage is CPU-complete and ready for one autonomous
two- or four-H100 run. No FoundIR training or test data was accessed.

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
`b9b576ddf7723bf0c74027d35ce2489c9a6806f7baf1a5df06f244df4124f922`.
Qwen2.5-VL-7B remains offline and frozen except for rank-16 LoRA on all 160
visual attention/MLP linear modules. The ablation compares the same LoRA with
the 44,574,464-parameter visual merger frozen versus trainable. The tokenizer,
language decoder, LM head, patch embed, and all non-adapter visual weights are
frozen.

The progressive schedule is one retained-data epoch at each stage:

1. 3,840 clean + single records;
2. 6,720 retained clean/single + double records;
3. 9,600 retained clean/single/double + triple records.

This is 630 optimizer steps at effective global batch 32. Two H100s each
process eight images for two accumulated microsteps; four H100s each process
eight images for one microstep. Thus either topology sees the same 32 records
per optimizer update without increasing per-GPU activation memory. The two
variants are trained sequentially so each uses every visible GPU. Validation
selects the winner; only that checkpoint is then evaluated once on the fixed
synthetic test.

Before full training, a separate 200-step GPU overfit gate uses one immutable
32-record batch: eight clean, eight single, eight double, and eight triple
records, collectively covering every one of the 20 type/level pairs. It starts
from the untouched base model, exercises visual LoRA plus the trainable merger,
and must demonstrate loss reduction, structured generation, type-set
memorization, independent-severity memorization, and exact gradient scope. Its
weights are gate-only and never initialize either full variant.

The full model identity/hash preflight passed, including all five 16 GB Qwen
shards and every one of the 12,000 generated image/evidence hashes. A
meta-device architecture audit matched exactly 160 LoRA modules and
11,151,360 adapter parameters; the trainable-merger variant has 55,725,824
trainable parameters total. Thirty-six focused contract/data/LoRA/trainer
tests, Python compilation, shell syntax, and `git diff --check` passed.
The shared environment now resolves Transformers 5.14.1. Its processor target
token counts, Qwen module hierarchy, all 160 LoRA targets, merger scope, and
trainable parameter count were revalidated before freezing that exact runtime
version; software versions are checked at the start of preflight.

The predeclared real-domain sanity pool contains five visually screened images
for each of blur, noise, rain, haze, and low light. It is qualitative only: no
absolute real L2--L5 truth is invented, and no FoundIR data is used. A source
audit found that one public SIDD-Small mirror PNG had different container bytes
from its official SIDD Full member even though the decoded RGB pixels were
exactly identical. The mirror is preserved as mismatch evidence; real-sanity
v3 instead freezes the official member. All five SIDD noise files now match
their official archive members byte-for-byte. The frozen real-sanity v3
manifest SHA-256 is
`59f5302aa3b58ba185c1d0fd0deaf82dbf7c0dc2c7b9345ca388854e666b6fad`;
the five-way official verification SHA-256 is
`d19ab870183f55ce72ddebbac63b5d920e15cf3b169ee1528de0a0ec96a717de`.

This coding terminal has no visible CUDA device, so no GPU result or final
checkpoint is claimed yet. The single external launch action is:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash smoke/launch_qwen_multidegradation_lora.sh
```

The launcher uses system tmux if executable and otherwise a detached nohup
fallback. It records persistent training and GPU telemetry logs. Training
resumes from immutable curriculum-stage checkpoints after interruption.
