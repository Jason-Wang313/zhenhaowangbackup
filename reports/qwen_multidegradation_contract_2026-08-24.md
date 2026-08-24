# Qwen multi-degradation experiment contract — 2026-08-24

This is the frozen Qwen-side contract approved through advisor review. It
replaces the Lowlight-only prototype as the active VLM experiment; it does not
rewrite or relabel that historical result.

Qwen2.5-VL must identify every visible type among Blur, Noise, Rain, Haze, and
Lowlight and independently assign L2, L3, L4, or L5 to every active type. L1 is
clean only and is serialized as `{"degradations":[],"levels":{}}`. Single,
double, and triple degradations are supported. All structured answers use one
fixed ordering and exact JSON schema.

AgenticIR synthetic data is used only to teach and objectively evaluate Qwen.
Original sources are split before cropping, so no source can cross train,
validation, or fixed test roles. The pilot contains 12,000 records: 10% clean,
with the remaining records equally divided among single, double, and triple
compositions. Combination and severity quotas are balanced within each split.
Haze is generated only for MiO100 sources with an existing aligned depth map;
grayscale pseudo-depth is forbidden.

Qwen is trained with LoRA and assistant-answer-token cross entropy while the
tokenizer, language decoder, and LM head remain frozen. Two otherwise identical
runs compare a frozen visual merger against a trainable visual merger. Model
selection uses synthetic validation only. The fixed synthetic test is consumed
once after selection.

Quantitative synthetic evaluation covers strict JSON validity, per-type and
aggregate precision/recall/F1, exact degradation-set match, exact full-label
match, independent severity accuracy/MAE, and clean/single/double/triple
strata. Real images provide a separate manual sanity check: adjacent L3/L4
ambiguity is acceptable, but obvious missing types, false positives, or
severity contradictions fail inspection.

After passing, one Qwen checkpoint is frozen. Its JSON is deterministically
converted to canonical restoration text for FLUX's existing Qwen3 text encoder
and multilayer attention. FLUX training is a later stage and uses FoundIR's
official training LQ/GT pairs only if dataset authorization is verified.
FoundIR official test pixels remain protected until the complete system and
protocol are frozen.
