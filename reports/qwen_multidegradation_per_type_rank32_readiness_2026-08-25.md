# Qwen per-type rank-32 dialogue retraining readiness

## Advisor decision

The advisor rejected independent classification heads because they would make
the system behave like an ordinary classifier instead of a dialogue VLM. The
approved diagnostic is the second dialogue route: submit the same degraded
image separately with one question for blur, noise, rain, haze, and low-light,
and let the Qwen text decoder answer one L1--L5 level each time.

The advisor-provided framework image is
`f95684090e025c3de176065a2dfdf043.png`, a 2088 x 1524 RGB PNG with SHA-256
`5b50d6e6d28a55949d2dda2217b152cdb8ae25af95db4400e5a23c037c2cfc12`.
It explicitly shows LoRA on the vision encoder and pretrained language model,
a frozen tokenizer/output decoder, assistant response supervision, and
cross-entropy loss. The advisor additionally requested Qwen LoRA rank 32, a
substantially larger batch, and longer training with evidence of whether loss
continues to decrease.

## Frozen method

No classification head exists. Each image produces five independent dialogue
records. For example, the noise record asks:

`Considering the noise degradation, how would you rate its level in this image? ...`

Its only valid targets are `The noise level is L1.` through
`The noise level is L5.` L1 means that the queried degradation is absent. The
five generated answers are parsed independently and then deterministically
aggregated into the existing five-slot JSON and canonical restoration text.
Different prompt lengths use left padding during decoder generation; training
continues to use right padding and assistant-answer-only CE.

The model starts fresh from the hash-pinned Qwen2.5-VL-7B-Instruct base. It
does not initialize from any of the rank-16/rank-8 correction checkpoints.
The trainable scope is:

- 160 visual-encoder LoRA modules at rank 32 / alpha 64;
- 112 language self-attention `q/k/v/o` LoRA modules at rank 32 / alpha 64;
- the visual merger;
- 87,062,272 trainable parameters total.

All original Qwen base weights, visual patch embedding, LM head, and tokenizer
remain frozen. Plain assistant-token cross-entropy is used; there is no
auxiliary classifier or semantic-token reweighting.

## Data, batch, and schedule

The original source-disjoint AgenticIR corpus remains unchanged: 9,600 train,
1,200 validation, and 1,200 protected fixed-test images. Dialogue expansion
produces 48,000/6,000/6,000 image-question pairs. Every training image is
asked all five questions exactly once per epoch.

The full run uses microbatch 25 per GPU and fixed global batch 100: two GPUs
use accumulation 2 and four GPUs use accumulation 1. Each global batch has
exactly 20 examples of each queried type. This is 3.125 times the prior local
microbatch and is checked on the actual H100 before full training. Three
complete shuffled dialogue epochs produce 144,000 presentations and 1,440
optimizer steps. The log records every tenth loss and exact first-window,
whole-epoch, and last-window mean losses for all three epochs.

Before full training, a 32-image/160-dialogue memorization gate uses the same
rank-32 scope and full microbatch 25. Fifty global-batch-100 steps yield 5,000
dialogue presentations. It must pass loss reduction, structure, set exact,
combined exact, severity, and gradient-scope gates. Thus it also fails early
if the proposed batch does not fit H100 memory.

## Validation and protected-test discipline

The final checkpoint is evaluated by 6,000 separate Qwen decoder calls on the
unchanged 1,200-image validation split. In addition to the existing structure,
type, severity, composition, and gradient gates, the run requires at least
99% valid per-type responses and at least 80% exact five-class query accuracy.
Triple-degradation set exact remains a separate frozen gate.

Any overfit or validation failure exits status 2 before selection. The
protected synthetic fixed test is called exactly once only if every validation
gate passes. FoundIR training and official test data are not used by this
stage.

Frozen config:
`configs/qwen_multidegradation_per_type_rank32_v1.json`, SHA-256
`43af6977452d4d62a5845d18c3d55cbcd30c56d9e9c1e9f8019b76c3983d5f58`.

CPU/full-hash evidence before launch:

- all Qwen model hashes and all 12,000 generated-image hashes passed;
- 47 focused legacy/new Qwen and launcher tests passed;
- representative type-specific targets have 9--10 supervised tokens;
- the rank-32 meta-architecture audit passed exactly;
- shell syntax and Python compilation passed;
- no new output, validation, selection, fixed-test, or real-sanity artifact
  exists.

This document is readiness evidence, not a GPU result or a validation pass.
