# FoundIR triple-composition literature and reproduction memo

> **Superseded on 2026-09-26:** The proposed CSR novelty claim is withdrawn
> after full-text verification of CURE (arXiv:2607.03044). CURE already uses
> matched selective-restoration targets, per-factor intermediate supervision,
> order consistency, and held-out triple evaluation after training without
> triples. See `reports/foundir_triple_novelty_cure_prior_art_correction_2026-09-26.md`.

Date: 2026-09-26 UTC
Status: literature/code audit complete; reproduction protocol proposed; no new training authorized or launched
Scientific role: paper-novelty planning, not an official FoundIR result

## 1. Decision in one sentence

Do **not** claim factorized continuous degradation intensity prediction by itself as the paper novelty. That space is already occupied by 2026 prior work. The defensible research question to test is narrower and harder:

> Can a single restoration backbone learn an explicit degradation-intervention algebra from singleton and double compositions, and thereby generalize to held-out triple compositions without seeing triple restoration targets?

The proposed contribution is therefore **held-out compositional generalization through counterfactual subset supervision**, not merely type/severity prompting, dynamic routing, a new specialist, or a new expert order.

This is still a working hypothesis. It must pass the frozen reproduction and novelty gates below before it can become a paper claim.

## 2. Local empirical motivation

The hash-bound 660-image route-aware comparison is complete:

| Composition size | Project PSNR / LPIPS | Released FoundIR Generalist | Full FoundIR |
|---|---:|---:|---:|
| Single, 300 images | 28.714078 / .141957 | 27.232394 / .174412 | 28.852611 / .168653 |
| Double, 300 images | 24.321537 / .192313 | 22.245454 / .206662 | 24.184117 / .194418 |
| Triple, 60 images | 16.941122 / .177332 | 24.945571 / .037753 | 33.677913 / .031967 |

The double aggregate beats Full FoundIR on both metrics and is frozen. The Low-light+Blur+Noise triple is the isolated failure: project minus Full FoundIR is `-16.736791 dB / +.145365 LPIPS`.

The existing degradation-aware Qwen experiment independently shows a composition-complexity failure on a source-disjoint synthetic validation set:

| Composition size | Type-set exact | Combined type+severity exact |
|---|---:|---:|
| Clean | .9917 | .9917 |
| Single | .9611 | .7389 |
| Double | .8944 | .6333 |
| Triple | .7500 | .4667 |

This does not prove the restoration model and VLM have the same root cause. It does establish a reproducible project-wide pattern: flat/categorical representations degrade as the number of simultaneous factors increases.

Evidence:

- `checkpoints/foundir_composition_integrated_restormer_v2/result.json`, SHA-256 `511940015fb3e3898f7da4ce7f31958dfb7f0bf9175062002de801cd2fcf7c1f`;
- `checkpoints/qwen_multidegradation_visibility_v2b_hard_triple/merger_trainable/validation_result.json`;
- no official FoundIR test image is part of this novelty-development evidence.

## 3. Literature findings and novelty exclusions

### FoundIR (arXiv 2412.01427)

FoundIR already contains a degradation-agnostic diffusion Generalist, isolated-to-coupled incremental learning, and a degradation-aware Generalist-to-Specialist pipeline. Its ablation explicitly compares Generalist only, Specialist only, Specialist-to-Specialist, Specialist-to-Generalist, and Generalist-to-Specialist.

Therefore the following are not novel here:

- adding another category specialist;
- Generalist-to-Specialist execution;
- changing a fixed specialist order;
- claiming that a Generalist alone is the complete FoundIR baseline.

### OneRestore (ECCV 2024, arXiv 2407.04621)

OneRestore is the closest public, fully inspectable composite-restoration implementation found in this audit.

Verified paper/code mechanism:

- one 5.98M-parameter transformer restorer;
- 12 fixed scene classes (`clear` plus 11 degradation compositions);
- text descriptors derived from GloVe component names and refined by an MLP;
- an automatic ResNet-18 visual embedder selects one descriptor by top-1 cosine similarity;
- the selected 324-dimensional descriptor is injected into every restoration block through Scene Descriptor-guided Cross-Attention;
- training uses Smooth L1, MS-SSIM, and Composite Degradation Restoration Loss (CDRL);
- CDRL treats the clean image as positive and the input plus other degraded variants as negatives.

Important implementation limits:

- automatic inference collapses the image to one global top-1 class descriptor;
- restoration training uses fixed categorical composition labels, not independent continuous intensities;
- all 11 listed composition classes are directly represented during training;
- the paper explicitly states that it controls degradation type but not restoration intensity for each type;
- it reports failures on extreme/high-complexity corruptions and unconsidered degradation types.

Public code audited at `https://github.com/gy65896/OneRestore`, current remote HEAD observed as `23a2e6af066864dca013aeed7f6c6f5958aa7259`.

### GRIDS (arXiv 2407.12273)

GRIDS estimates degradation similarity, groups related tasks, trains a model per group, and performs adaptive grouped-model selection. It does not establish unseen simultaneous-composition generalization from singleton/double training.

Therefore degradation-similarity grouping alone is not novel.

### RestoreAgent (arXiv 2407.18035)

RestoreAgent uses an MLLM to select restoration tools and order, supports iterative replanning, rollback, stop decisions, and execution history, and obtains supervision through exhaustive order/model search.

Therefore dynamic expert order, MLLM routing, replanning, and rollback are not novel.

### Learning Dynamic Prompts / DPPD (TIP 2025, DOI 10.1109/TIP.2025.3567205)

DPPD already argues that fixed prompts average degradation distributions. It uses degradation prototype assignment and prompt-distribution learning to generate instance-adaptive prompts.

Therefore replacing a fixed prompt with a dynamic prompt is not sufficient novelty.

### CDG-IR (TMM 2026, DOI 10.1109/TMM.2026.3702504)

The paper is titled **Composite Degradation Image Restoration Based on Multi-Dimensional Continuous Degradation Intensity Prediction**. Its abstract verifies:

- a Composite Prompt Generator that identifies intertwined degradation characteristics;
- an Image Restoration Transformer with self- and cross-attention;
- continuous composite degradation factors;
- MD-83W, 830,000 samples with continuous variation over five degradation dimensions.

This directly invalidates the earlier tentative claim that a five-dimensional continuous type/severity vector is itself novel.

### F2D-Net (Knowledge-Based Systems 2026, DOI 10.1016/j.knosys.2026.116411)

F2D-Net reports two failure modes: deterministic transport over-smoothing and low-selectivity image-level expert routing. It factorizes the restoration velocity into a shared backbone plus degradation-specific expert increments, then assembles them with learned pixel-wise spatial gating. It also uses state-dependent multiplicative noise.

Therefore factorized experts and spatial degradation gating alone are not novel.

### CDIR (TIP 2026, DOI 10.1109/TIP.2026.3682023)

CDIR already uses efficient dual-branch attention, dynamic operations guided by local/context priors, and iterative prior updates for composite degradation restoration.

Therefore local/context-aware dynamic processing alone is not novel.

### UniCoRN (WACV 2026, DOI 10.1109/WACV61042.2026.00156)

UniCoRN is a unified controllable latent-diffusion system with a multi-head control network, mixture-of-experts adaptation, low-level visual cues, and curriculum learning for simultaneous degradations.

Therefore multi-head controllable diffusion plus MoE is not sufficient novelty.

### MDIR (ICASSP 2026, DOI 10.1109/ICASSP55912.2026.11461462)

MDIR conditions restoration using a predicted Fourier magnitude-difference signature and adaptive layer normalization.

Therefore compact continuous degradation signatures and AdaLN injection alone are not novel.

### Unresolved literature risks

The following 2026 papers were identified but their full methods were not yet available through the sources used in this audit and must be checked before a novelty claim:

- CURE: Controllable Unified Image Restoration for Complex Degradations;
- TSCIR: Towards Composite Weather Degradation Image Restoration via a Two-Stage Framework;
- Fine-grained Structure-Preserving Composite Degradation Image Restoration;
- Degradation-aware feature disentanglement for task-conditioned all-in-one restoration;
- MDIR: Multi-Dimensional Degradation Representation Learning for All-in-One Image Restoration (distinct title from the ICASSP magnitude-signature work).

## 4. Revised candidate contribution

### Working name

Counterfactual Subset Restoration (CSR), used only as an internal working name.

### Claim to test

A restoration model trained on clean, singleton, and double compositions can generalize to a held-out triple if training explicitly teaches what each conditioning component is allowed to remove.

The distinction from prior work is the supervision and evaluation regime:

- continuous intensity prompting is a baseline, not the contribution;
- factorized/spatial experts are a baseline, not the contribution;
- the triple composition and every triple restoration target are excluded from training and checkpoint selection;
- matched counterfactual targets supervise selective removal of arbitrary degradation subsets;
- the main endpoint is held-out composition generalization, not average performance on composition classes seen during training.

### Candidate representation

The frozen VLM emits a five-component control object:

`[(presence_k, severity_k, confidence_k)]` for Blur, Noise, Haze, Rain, and Low-light.

Each component is encoded separately. The restoration backbone must preserve component identity through injection; concatenating or averaging all active types into one undifferentiated prompt is not allowed in the proposed arm.

### Candidate intervention supervision

For a clean source `y`, fixed operator parameters, and degradation set `S`, synthesize a matched lattice containing `x_T` for every `T subseteq S`. For input `x_S` and requested removal subset `A subseteq S`, supervise the model toward the explicitly rendered counterfactual `x_(S\\A)`.

This creates three measurable constraints:

1. **Selective removal:** requesting `A` removes only factors in `A`.
2. **Subset consistency:** restoring `A` then `B` agrees with restoring `A union B`, within a predeclared tolerance.
3. **Inactive-factor identity:** setting component `k` to absent should not alter the image through the `k` branch.

Because image degradations are non-commutative, every record must preserve operator parameters and application order. Counterfactual targets are explicitly rendered; they are never inferred by subtracting images or pretending that degradation operators commute.

## 5. Data truth and restrictions

### What is available

The AgenticIR synthetic manifests store:

- active degradation set;
- independent discrete levels;
- physical operator parameters;
- application order;
- clean source identity and hash;
- source-disjoint split information.

This is sufficient to generate matched intervention lattices without inventing labels.

### What is not available

The 60 FoundIR Low-light+Blur+Noise validation images contain composition labels but no authoritative per-component continuous severity truth. Severity must not be reverse-engineered from filenames, GT differences, or a model prediction and then presented as ground truth.

### Protected-data rule

- Official sealed singleton SOW600 remains terminal and untouched.
- No official FoundIR test pixel may enter training, prompt fitting, severity calibration, checkpoint selection, or hyperparameter selection.
- The current 60-image FoundIR triple validation category may be used only once as a frozen external endpoint after synthetic selection is complete.

## 6. Bounded reproduction protocol

### R0: public-mechanism reproduction gate

Reproduce the public OneRestore descriptor path before testing the proposed loss:

- fixed categorical composition descriptor;
- global descriptor injected through cross-attention;
- restoration objective with and without CDRL;
- source-disjoint synthetic development only.

The purpose is not to claim OneRestore numbers on CDD-11. It is to verify that the local implementation can recover the expected directional ablation: descriptor guidance and CDRL should improve over an unconditioned backbone on seen composite degradations.

Pass condition: both changes have the expected direction on a frozen development set and all output/provenance checks pass. If R0 fails, stop at implementation diagnosis.

### R1: strongest prior-art-style baseline

Using the same backbone, optimizer budget, data, augmentations, and selection rule:

- factorized five-dimensional oracle type/severity input;
- one global continuous prompt, CDG-IR-style in scientific role;
- train on singleton and double compositions only;
- all triple compositions excluded from gradients and selection.

This is a reproduction baseline, not the proposed method.

### R2: proposed CSR arm

Keep R1 unchanged except for:

- component-preserving conditioning;
- counterfactual subset targets;
- selective-removal, subset-consistency, and inactive-factor identity losses.

No extra triple examples, model capacity, optimizer steps, or validation accesses are allowed in R2.

### R3: VLM substitution

Only after oracle-conditioning R2 succeeds, replace oracle synthetic labels with frozen Qwen predictions. Report both results. This separates restoration compositionality from VLM recognition error.

### Frozen split design

- Training: source-disjoint clean/single/double synthetic records only.
- Development: source-disjoint clean/single/double records for checkpoint selection.
- Synthetic challenge: held-out triples, never used for selection.
- External challenge: the 60 FoundIR Low-light+Blur+Noise validation images, accessed once after freeze.
- Official sealed singleton set: no access.

### Primary endpoints

The method is scientifically interesting only if all are true:

1. R2 beats R1 on held-out triple PSNR and LPIPS under paired source-group bootstrap.
2. R2 does not materially regress the frozen singleton/double development aggregate.
3. Selective-removal leakage is lower than R1 on counterfactual targets.
4. The improvement persists when oracle controls are replaced by frozen VLM controls, or the residual gap is explicitly attributed to VLM error.
5. The one-time FoundIR triple endpoint improves over `16.941122/.177332`; beating Full FoundIR is a stretch outcome, not required to establish the mechanism.

Failure to beat R1 means the proposed novelty is rejected, even if a larger model improves absolute metrics.

## 7. What not to do next

- Do not train another Low-light+Blur+Noise specialist.
- Do not use the 60 FoundIR triple images iteratively.
- Do not call continuous severity conditioning novel.
- Do not compare only against Released FoundIR Generalist.
- Do not attribute the restoration collapse solely to router error; the current evidence does not support that causal claim.
- Do not launch a large FLUX experiment before R0/R1/R2 configs, hashes, data exclusion, loss definitions, and stop gates are frozen.

## 8. Exact next implementation task

Prepare a CPU-only reproducibility package for R0 and the future R1/R2 comparison:

1. vendor or hash-pin the exact OneRestore source revision without modifying its scientific logic;
2. build a source-disjoint synthetic mini-manifest containing matched clean/single/double records and held-out triples;
3. implement a counterfactual-lattice validator that proves parameter/order consistency and triple exclusion;
4. write unit tests for fixed categorical descriptors, five-component controls, and subset-target construction;
5. emit a governance-bound design/readiness report with total wall-time and GPU-hour estimates;
6. stop for user/advisor review before any GPU launch.

No GPU training was started while producing this memo. Wall time and GPU-hours for this literature/code audit are not charged as a GPU experiment; GPU-hours are exactly zero.

## 9. Source links

- FoundIR: https://arxiv.org/abs/2412.01427
- OneRestore: https://arxiv.org/abs/2407.04621
- OneRestore code: https://github.com/gy65896/OneRestore
- GRIDS: https://arxiv.org/abs/2407.12273
- RestoreAgent: https://arxiv.org/abs/2407.18035
- DPPD: https://doi.org/10.1109/TIP.2025.3567205
- CDG-IR: https://doi.org/10.1109/TMM.2026.3702504
- F2D-Net: https://doi.org/10.1016/j.knosys.2026.116411
- CDIR: https://doi.org/10.1109/TIP.2026.3682023
- UniCoRN: https://doi.org/10.1109/WACV61042.2026.00156
- MDIR: https://doi.org/10.1109/ICASSP55912.2026.11461462
