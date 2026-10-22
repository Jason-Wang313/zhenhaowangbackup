# Explicit-VLM DiT-MoE novelty and implementation audit — 2026-10-19

## Bottom line

The broad proposal **"use a VLM to recognize degradation and guide a MoE
restoration model" is not novel**. The stronger broad proposal **"VLM-guided
MoE inside a DiT for all-in-one restoration" is also too close to the union of
recent work to support a first-of-kind claim**.

The strongest overlap evidence is:

- LoRA-IR uses a CLIP-based degradation router to predict degradation
  probabilities and uses those probabilities for top-k selection and weighted
  merging of low-rank restoration experts. It also reports tests on unseen
  mixed degradations.
- M2IR uses DA-CLIP image/text degradation priors to produce spatial routing
  weights for specialized and shared restoration experts.
- VLMoE-Restore explicitly claims VLM parsing of degradation type, degree and
  physical cause, followed by both feature injection and MoE gating guidance
  for composite infrared degradations.
- MiM-DiT already places hierarchical MoE routing inside a pretrained DiT for
  all-in-one restoration.
- SLER-IR already performs layer-wise expert routing and explicitly motivates
  hybrid layer paths for composite degradation.
- UARE already performs explicit analyze-then-restore vision-language
  restoration, although it does not use degradation-specific MoE routing.

Therefore the current idea is a **conditional go only after narrowing the
mechanism**. The most defensible remaining hypothesis found in this audit is:

> Compile the VLM's explicit atomic multi-label diagnosis and per-atom severity
> into a constrained, block- and diffusion-time-dependent schedule over atomic
> DiT LoRA experts. A learned image router may make only bounded residual
> corrections inside the VLM-declared support. Train and test whether this
> structure composes singleton skills into coupled degradations more reliably
> than image-only routing, generic VLM feature routing, and static VLM top-k
> merging.

Even this is not guaranteed paper-level novelty. It is a credible, testable
mechanism-level hypothesis, not a safe claim, because a reviewer can still
describe it as combining LoRA-IR's VLM/LoRA routing, SLER-IR's layer paths and
FoundIR-v2's timestep-conditioned scheduler. Its credibility must come from a
precise formulation and matched compositional evidence, not from the names of
the components.

No GPU experiment was launched during this audit: `0` wall-clock GPU hours and
`0` allocated GPU-hours.

## What the closest papers actually do

| Method | Restoration backbone | Routing signal | Expert operation | Coupled/OOD evidence | Important boundary |
|---|---|---|---|---|---|
| FoundIR-v2, arXiv:2512.09282 | SDXL latent diffusion | Concatenated LQ latent and noisy HQ latent at diffusion timestep | Soft weighted sum of attention experts | More than 50 subtasks; coupled-degradation results | LLaVA captions enter cross-attention separately; explicit VLM type/severity does not constrain the MoE gate |
| MiM-DiT, arXiv:2603.02710 | Pretrained SD3.5 DiT | Linearly projected LQ features, without external labels | Dense fusion of four heterogeneous attention groups; sparse top-k sub-experts inside each group | FoundIR and real benchmarks | Occupies broad `MoE + DiT + all-in-one restoration`; no explicit VLM diagnosis |
| UARE, arXiv:2512.06750 | Bagel-style mixture of two full transformers | Generated IQA/quality-analysis text remains in a shared attention stream | IQA expert conditions a restoration expert; not a degradation-expert router | Single, multiple and high-order degradations | Occupies broad explicit analysis-guided restoration, but not diagnosis-constrained degradation MoE |
| LoRA-IR, arXiv:2410.15385 | NAFNet-like U-Net | High-resolution CLIP degradation embedding and class probabilities | Static per-image top-k weighted mixture of degradation-specific LoRA experts | Three unseen mixed benchmarks; none used in mixed training | Very close to `VLM diagnosis -> LoRA expert merge`; no explicit per-atom severity or DiT-time schedule |
| MEASNet, arXiv:2407.19139 | Transformer restoration network | Learned task-specific prompts and image content | Pixel-level and global/frequency-level top-k experts | Three/five-task all-in-one tests | Not a VLM; already covers prompt-guided spatial/global expert selection |
| M2IR, arXiv:2603.14816 | Mamba-style Transformer encoder/decoder | DA-CLIP degradation feature plus similarity to text descriptors such as rainy/hazy/noisy, aligned spatially with decoder features | Per-pixel top-k specialized experts plus one shared expert | Includes CDD11 fundamental and composite degradations | Occupies VLM-like text/image prior guided spatial expert collaboration |
| SLER-IR, arXiv:2603.05940 | Eight-layer restoration backbone | CLIP-derived degradation embedding on a hypersphere | One expert per layer; up to `3^8=6561` paths; global/local degradation maps | Claims progressive hybrid paths for composite degradation | Occupies layer-wise compositional routing, but routing is implicit rather than explicit VLM type/severity |
| VLMoE-Restore, DOI:10.1109/ICIGP68997.2026.11619648 | Three-expert infrared restoration network | VLM semantic parsing of degradation type, degree and physical cause | Dual-path feature injection and gating guidance | Composite infrared degradation; abstract claims average `+2.8 dB` | Closest wording to the advisor idea, but infrared-only and full text is closed |

### FoundIR-v2 details

FoundIR-v2's router input is

\[
z_t^{(k)}=\phi(f_k^{LQ},x_{t,k}^{HQ}),
\]

where `phi` is concatenation. Softmax gating produces

\[
w_i^{(k)}=
\frac{\exp(g_i^{(k)\top}z_t^{(k)})}
{\sum_j\exp(g_j^{(k)\top}z_t^{(k)})},
\qquad
\mathcal F^{(k)}(z_t)=\sum_iw_i^{(k)}E_i(z_t^{(k)}).
\]

The experts are attention mechanisms such as spatial, channel and sparse
attention. LLaVA captions are injected through cross-attention as a separate
condition. The MoE ablation compares no scheduler, hard top-k MoE and soft MoE;
the paper reports soft MoE as best. This is an implicit latent-feature router,
not an explicit type/severity-constrained router.

### MiM-DiT details and reproducibility limit

MiM-DiT uses four Inter-MoE expert groups: spatial self-attention, channel
self-attention, Swin attention and SE attention. A dense image-feature router
mixes all four groups. Each group contains same-architecture sub-experts and
uses sparse top-k Intra-MoE routing. The resulting conditioning tokens are
injected into SD3.5 DiT blocks through a zero-initialized linear projection.

The paper trains for `2,000,000` iterations, patch size `512`, batch size
`128`, fixed learning rate `5e-5`, and uses `40` inference steps. Its router
ablation reports average LPIPS `0.1636` for dense-Inter/sparse-Intra, versus
`0.2457` for sparse/sparse and `0.2348` for sparse/dense.

Reproducibility is currently weak. The public GitHub repository at commit
`aac9421` contains only a two-line README. The arXiv source contains
`main.tex`, figures and tables but no code and no separate supplement despite
the paper repeatedly referring to supplementary material. Therefore only the
published equations/results, not unreported implementation behavior, are used
in this audit.

### UARE overlap

UARE contains an IQA expert and a restoration expert in one interleaved token
stream with shared self-attention. In assessment-guided restoration, the IQA
expert generates quality analysis and enhancement recommendations; those text
tokens remain in the stream and condition restoration latents. UARE trains a
single-to-multiple-to-high-order restoration curriculum, then jointly
fine-tunes IQA and restoration with interleaved text/image data.

Its inference ablation on RealSR reports:

| Condition | PSNR | LPIPS | MUSIQ |
|---|---:|---:|---:|
| Simple prompt | 21.79 | 0.2772 | 66.52 |
| External Q-Insight prompt | 21.26 | 0.3175 | 64.43 |
| UARE analysis-then-restore | 21.38 | 0.3095 | 69.67 |

This supports a perceptual benefit claim, not a universal fidelity gain: the
simple prompt has better PSNR and LPIPS. The UARE GitHub repository currently
states that code and weights remain unreleased.

### LoRA-IR overlap

LoRA-IR is the strongest open collision with static expert merging. Its
DG-Router runs one downsampled global image and four high-resolution crops
through frozen CLIP, pools the crop features, concatenates global/local
features, and trains an MLP/classification head. It produces a degradation
embedding and class probabilities. Those probabilities select top-k LoRA
experts and re-normalize their weights:

\[
x_{out}=PreMod(x_{in})+
\sum_{i=1}^{k}w'_{\varphi(i)}E_{\varphi(i)}(x_{in}).
\]

The released code confirms the paper: `ConvMixLoRA.forward` masks all but
top-k CLIP probabilities and sums the corresponding low-rank weight updates.
This is not merely prompt conditioning; it is actual VLM-feature-guided expert
weight composition.

The paper evaluates rain+haze, low-light+blur and blur+JPEG without training on
mixed degradation. Its ablation reports low-light+blur PSNR `20.59` with the
LoRA mixture versus `16.79` without LoRA experts, and blur+JPEG `25.05` versus
`24.55`. Thus a proposed paper cannot cite coupled/unseen expert composition as
an untouched problem.

### M2IR, SLER-IR and VLMoE-Restore overlap

M2IR's ADEC obtains a DA-CLIP degradation feature and similarities to a
predefined text descriptor set. It concatenates these priors, spatially aligns
them to decoder features through cross-attention, and computes per-pixel
softmax routing over specialized experts. Each position uses top-k specialized
experts plus a shared expert. Its public GitHub repository currently contains
only a license and a README saying that code is being sorted out.

SLER-IR uses a CLIP degradation encoder, emits one routing vector per layer,
and makes independent expert selections across eight layers. It explicitly
argues that composite degradations can activate hybrid expert paths. It also
constructs a 4x4 crop-wise degradation map for local guidance. It does not use
explicit VLM type/severity labels, but it already occupies layer-wise and
spatial compositional expert routing.

VLMoE-Restore is closed access. Its Semantic Scholar/IEEE abstract, not a full
paper audit, states that a VLM parses type, degree and physical cause, and that
the result controls both feature injection and MoE gating over raindrop,
defogging and low-illumination experts. Because the full text is unavailable,
no finer claim is made here. The title and abstract alone are sufficient to
forbid a broad `first VLM-guided MoE restoration` claim.

## What the current repository really provides

### VLM outputs

The Qwen2.5-VL branch produces five explicit slots:

- `blur`, `noise`, `rain`, `haze`, `low_light`;
- level `1` means absent and levels `2–5` encode increasing strength;
- singleton, eleven-class composition and atomic multi-label heads exist;
- separate nonlinear low-light/noise heads and physical checks also exist.

The repository persists both structured outputs and frozen visual features.
The feature artifacts are dimension `7168`, formed from mean and standard
deviation pooling of `3584`-dimensional Qwen visual tokens. The production
router consumes these features through linear/nonlinear heads and emits a
route/expert plan.

### How FLUX currently uses the diagnosis

The existing five-type FLUX path does **not** use diagnosis to select internal
experts. It converts the five levels into canonical text such as `Blur: L3;
Noise: absent; ...`, encodes this text with Qwen3, and passes the resulting
context tokens to FLUX. The degraded image separately enters as native FLUX
reference-image tokens. Training uses degraded-anchored rectified flow and a
bounded pixel refiner.

Thus direct structured expert control would be a real code change. It would
not be an accurate description of the current model.

### Existing expert compatibility

The strongest Blur, Haze and Low-light FLUX checkpoints use the same FLUX.2
Klein 4B base, the same 50 target Linear modules, rank/alpha `32/32`, and the
same 100 tensor keys. Direct inspection confirms identical key sets and tensor
shapes (`32,440,320` LoRA parameters each), so they are mathematically
compatible candidates for a shared-base LoRA mixture.

The strongest current Noise and Rain experts are Restormers, not FLUX LoRAs.
They cannot be inserted or weight-merged inside the same DiT. Blur+Noise also
uses a Restormer. Therefore a genuine five-atom DiT-MoE requires training at
least Noise and Rain FLUX LoRA experts under a matched base/protocol, or
changing the scope. Calling the current heterogeneous sequential system an
internal DiT-MoE would be false.

### Feasible insertion point

Current FLUX LoRA targets are:

- `double_blocks.*.img_attn.qkv`;
- `double_blocks.*.img_attn.proj`;
- `single_blocks.*.linear1`;
- `single_blocks.*.linear2`.

The present `LoRALinear` stores one `(A,B)` pair and has no route input. A MoE
implementation would need multiple atomic `(A_d,B_d)` pairs and per-sample
weights set before each FLUX forward. Block- and timestep-specific weights are
feasible because module names identify block depth and each model call already
receives diffusion timesteps. This is an implementation conclusion only; no
such module currently exists.

## Novelty claims that are not allowed

Do not claim any of the following:

- first MoE all-in-one restoration;
- first MoE + DiT restoration;
- first VLM-guided restoration;
- first VLM-guided MoE restoration;
- first VLM degradation classifier controlling LoRA experts;
- first expert composition for unseen mixed degradation;
- first layer-wise or spatial degradation-aware expert routing.

## Narrow claim that remains testable

The remaining mechanism must differ operationally, not only by backbone name.
A lean formulation is **diagnosis-constrained atomic expert scheduling**:

1. The VLM emits a calibrated multi-label atomic support vector `p_d` and
   continuous/ordinal severity `s_d` for each degradation atom.
2. Each atom owns a compatible FLUX LoRA expert. Absent atoms have exactly
   zero support; an unconstrained router cannot silently activate them.
3. Severity controls expert magnitude, while a learned low-capacity residual
   router may redistribute only a bounded fraction of mass inside the active
   support.
4. The mixing coefficients vary by DiT block and diffusion timestep, so early
   and late restoration can allocate different operations without executing
   heterogeneous experts sequentially.
5. Training includes an explicit compositional consistency objective: a mixed
   diagnosis must preserve the active atomic support and must not collapse to
   an unrelated single expert.

The core scientific question is not whether Qwen can classify degradation.
It is whether an explicit atomic/severity diagnosis is a useful structural
constraint on DiT computation and whether that constraint improves coupled or
unseen composition over learned image-only routing.

## Required matched evidence before a paper claim

Use one backbone, identical experts, identical data and identical training for
all router arms:

1. image-feature-only learned router;
2. generic frozen Qwen embedding concatenated to the router;
3. explicit type-only static top-k prior;
4. explicit type plus severity static merge;
5. explicit type/severity/composition with constrained block/time schedule;
6. oracle diagnosis, used only as an upper-bound diagnostic and never as model
   performance.

Report singleton, double and triple results separately. The decisive evidence
is not router classification accuracy by itself. It is restoration improvement
on coupled and preferably unseen compositions, plus routing traces showing
that the claimed constraint changes expert use as intended. Include LoRA-IR,
M2IR, FoundIR-v2, MiM-DiT, SLER-IR and UARE in related work even if released
weights are unavailable.

## Go/no-go decision

- **No-go** for implementing a generic VLM-conditioned router or a static
  VLM-weighted LoRA mixture; LoRA-IR/M2IR already make that contribution.
- **No-go** for presenting the existing Qwen router alone as the paper's main
  architecture contribution; it currently selects external heterogeneous
  experts and has no matched published-router comparison.
- **Conditional go** for the constrained atomic severity-to-block/time
  scheduling hypothesis, after the advisor accepts this narrower wording.
- **No GPU launch yet.** First freeze the exact mathematical mechanism,
  identify training data for matched Noise/Rain FLUX LoRA experts, and define
  a bounded pilot with the six controls above. Hardware scale cannot rescue an
  overlapping claim.

## Sources and material status

- FoundIR-v2: arXiv:2512.09282, full main text read; no code link found in the
  paper page used for this audit.
- MiM-DiT: arXiv:2603.02710, full six-page main text and arXiv source read;
  public GitHub `kkkls/MIM-DiT` contains README only; referenced supplement is
  not present in the source archive.
- UARE: arXiv:2512.06750, full main text and supplement read; public GitHub
  `lwq20020127/UARE` says code/weights are not released.
- LoRA-IR: arXiv:2410.15385, full main text read; public source code inspected,
  including `classification/model.py`, `PriorFtModel_arch.py`, and
  `loralib/layers.py`.
- MEASNet: arXiv:2407.19139, full main text read.
- M2IR: arXiv:2603.14816, full arXiv HTML read; public GitHub `Im34v/M2IR`
  currently contains only README/license.
- SLER-IR: arXiv:2603.05940, full arXiv HTML read.
- VLMoE-Restore: DOI:10.1109/ICIGP68997.2026.11619648; only bibliographic
  record and abstract were accessible. It is explicitly marked as an
  abstract-level overlap check, not a full-paper reading.

