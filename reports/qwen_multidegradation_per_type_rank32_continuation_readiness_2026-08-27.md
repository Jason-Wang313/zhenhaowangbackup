# Qwen per-type rank-32 continuation readiness — 2026-08-27

## Decision

The advisor judged three epochs potentially underfit, asked for longer
training, and considered the 55% strict triple-degradation type-set exact rate
too low. The epoch-3 training losses were still descending
(`0.0774334 -> 0.0405060 -> 0.0297576`). This continuation therefore adds five
complete epochs and evaluates total epochs 5, 6, and 8.

The protected synthetic fixed test and FoundIR are not used for training or
checkpoint selection. The fixed test remains inaccessible unless a frozen
validation candidate passes every gate, including the continuation's stricter
80% triple exact threshold.

## Immutable initialization

The continuation writes a new output root and does not alter the v1 result.
It resumes both the rank-32 adapter and exact AdamW state from epoch 3:

- adapter SHA-256:
  `5c7df484672088806cca56c83947cc3a3c7fa695b608bba79ed3a9ee11d2399b`;
- optimizer SHA-256:
  `a7c198465d29370b758c99a36f083454e9191957c83703dc4157844c54bb9aa1`;
- source validation SHA-256:
  `c3939afab873a34b744aeab961699732345eb0da573c9641d232a1ddd4099051`;
- initial optimizer steps: 1,440;
- initial dialogue presentations: 144,000.

The continuation learning-rate maxima equal the v1 schedule's terminal rates:
visual LoRA `5e-6`, merger `5e-7`, and language LoRA `2e-6`. A new five-epoch
cosine decay starts from those values, avoiding an upward learning-rate jump.

## Data and hardware equivalence

Every added epoch contains all 9,600 training images and all five per-type
questions, for 48,000 dialogue presentations. Five added epochs contribute
240,000 presentations; the eight-epoch cumulative total is 384,000.

The effective global batch is 200 on both supported systems:

- 2 H100: microbatch 25 per GPU, gradient accumulation 4;
- 8 H100: microbatch 25 per GPU, gradient accumulation 1.

Every global batch contains exactly 40 questions for each of blur, noise,
rain, haze, and low-light. The same global batch and sample schedule are
preserved if a stage checkpoint is resumed on the other supported GPU count.

## Validation and selection

Epochs 5, 6, and 8 are evaluated on the same frozen, source-disjoint synthetic
validation split. Only candidates passing all gates are eligible. Eligible
candidates are ranked by strict triple type-set exact first, followed by
combined type-level exact, active severity accuracy, overall set exact,
macro-F1, and severity MAE. The protected test is consumed only once after
selection.

## Reproducibility evidence

- config: `configs/qwen_multidegradation_per_type_rank32_continue_v2.json`;
- config SHA-256:
  `ec46b18c72ec9d2df18b725f0fd2dde39d42136a6008a677b45ba3559bf6cd4b`;
- runner SHA-256:
  `e3331661c43142e269ff27a6c168df8cc6c397b12e3c5594aca46079be8311ce`;
- launcher SHA-256:
  `95454de4933529ca616178f0f9bec698966630d99a9521adfc43f3c9f98107e7`;
- full-hash preflight: passed for all 12,000 generated records, Qwen model
  inventory, rank-32 272-module scope, and 25 frozen real-sanity images;
- 34 focused tests: passed;
- Python compilation, JSON parsing, shell syntax, and `git diff --check`:
  passed.

The broader historical test suite retains an unrelated pre-existing
authorization error: `configs/lowlight_flux2_qwen_conditioned_full.json` binds
an obsolete SHA-256 for the unchanged HEAD version of
`lowlight_restoration/lora.py`. This continuation does not modify either file,
and all tests covering the changed Qwen paths pass.

## Launch action

From the workspace root, choose exactly one visible-device set and run the
checked launcher. It retains a tmux pane when system tmux is available and
otherwise uses a detached nohup fallback. Persistent logs and GPU telemetry
are written under `logs/`.
