# Current-method novelty audit

Date: 2026-09-27 UTC
Scope: repository/method/prior-art audit only; no training or evaluation was run

## Bottom line

### Evidence update after frozen-artifact reaggregation

The paper-facing reaggregation in
`reports/foundir_routing_paper_ablation_2026-09-27.md` strengthens the routing
claim. On the same 16,461 development records and at exactly the final router's
13,603 accepted decisions (82.6377% coverage), an ordinary eleven-class head
ranked only by confidence makes 92 wrong accepted decisions; the final
multi-head router makes 17. This is 75 fewer errors, an 81.52% reduction at
matched coverage. The final accepted-error rate is .1250%, versus .6763% for
the ordinary head. Therefore the final improvement cannot be explained only
by abstaining on more images.

This is existing development evidence, not a newly independent lockbox. It
supports the current method claim but does not establish open-set or universal
safety. No additional routing experiment is needed to reproduce this
matched-coverage result; the remaining paper work is framing, independent
evaluation design, and the separately identified triple-restoration problem.

The effective terminal system is potentially publishable as a **risk-controlled
VLM-mediated restoration orchestration system**, but it is not defensible to
claim that its novelty is simply "VLM + FLUX.2 + Restormer experts" or
"All-in-One restoration by routing specialists." Those broad ideas are already
covered by RestoreAgent, AgenticIR, FoundIR, GRIDS, and several unified/MoE
restoration methods.

The strongest current candidate contribution is narrower:

> A fail-closed restoration router that projects a fine-tuned VLM visual
> representation into several independently calibrated degradation hypothesis
> spaces, executes a heterogeneous restoration plan only under cross-head and
> physical-evidence agreement, and otherwise abstains or takes a conservative
> partial action.

This is more than ordinary top-1 classification in the current implementation.
It combines five-way and eleven-way categorical heads, an atomic compositional
head, ambiguity-specific nonlinear heads, a physical exposure/noise check, and
an explicit execution contract. Nevertheless, it is currently a **candidate
method claim**, not cleared novelty: confidence thresholding, selective
prediction, ensemble agreement, and abstention are established ideas outside
image restoration. A paper must establish what is restoration-specific and
compare against simpler selective-routing baselines.

## 1. What the successful system actually is

The terminal official result is produced by a routed portfolio, not by one
monolithic VLM-conditioned restorer:

1. A fine-tuned `Qwen2.5-VL-7B-Instruct` adapter is loaded and its visual-token
   mean and standard deviation form a 7,168-dimensional feature.
2. Frozen learned heads create several route proposals:
   - a five-class singleton head;
   - an eleven-class singleton/composition head;
   - a five-factor compositional presence head;
   - low-light/noise ambiguity-specific nonlinear heads.
3. A physical exposure/dark-noise rule supplies another independent piece of
   evidence for the difficult Noise versus Noise+Low-light family.
4. `final_eleven_route()` accepts only a small set of agreement patterns.
   Otherwise it returns `None`.
5. `safe_expert_plan()` converts an accepted route into a fixed plan, or uses
   a conservative Low-light partial fallback for specified dark abstentions.
6. The expert portfolio is heterogeneous:
   - Blur, Noise, Haze, and Low-light use separately trained FLUX.2 experts;
   - Rain and direct Blur+Noise use Restormer experts;
   - admitted mixed classes use fixed sequential plans.

Primary implementation evidence:

- `lowlight_restoration/final_eleven_runtime.py`
- `lowlight_restoration/final_eleven_route_router.py`
- `configs/foundir_composition_whole_image_v1.json`
- `smoke/export_foundir_official_sow600_qwen_features_ddp.py`
- `smoke/run_foundir_qwen_five_class_gate_ddp.py`

The official 600-image run reports 538 correct accepted routes, 62 fallbacks,
zero wrong routes, and all ten SOW checks passing. This is strong terminal
system evidence, but it is not by itself proof that the routing rule is novel.

## 2. Critical identity split: proposed mainline versus successful deployment

The repository contains two related but scientifically distinct methods.

### A. Earlier monolithic VLM-conditioned FLUX.2

`configs/foundir_five_type_flux2_dit_v3.json` and
`smoke/train_foundir_five_type_flux2_full_ddp.py` show that the fine-tuned Qwen
produced structured type/severity text. A frozen Qwen3 text encoder converted
that text to context tokens passed into the trainable FLUX.2 DiT. This is real
language-conditioned restoration rather than routing only.

That model's sealed internal result passed only 1 of 10 metric checks. It is a
negative/failed mainline result and cannot supply the successful official-v8
numbers.

### B. Terminal routed expert system

The terminal system uses the fine-tuned Qwen visual representation to select
an expert/plan. The FLUX.2 experts in the verified composition config receive
fixed task prompts; Qwen features and predicted severities are not injected
inside those restoration backbones. This system produced the terminal official
pass.

A paper must not combine A's architectural claim with B's metrics as though
one evaluated model produced both. A can appear as motivation or a negative
ablation; B is the effective method.

## 3. Atomic novelty matrix

| Current element | Closest overlap | Audit verdict |
|---|---|---|
| Fine-tuned VLM recognizes degradation type/severity | VLM quality analysis, dynamic prompts, continuous degradation prediction, and degradation-aware visual prompts are well populated | Not novel alone |
| VLM/MLLM selects restoration tools | RestoreAgent and AgenticIR directly do this | Not novel |
| Multiple task-specific experts | FoundIR specialists, GRIDS groups, RestoreAgent/AgenticIR toolboxes | Not novel |
| Heterogeneous FLUX.2 and Restormer portfolio | Agent/toolbox systems already allow heterogeneous models and choose among several tools per task | Engineering contribution, unlikely method novelty alone |
| Fixed sequential plans for mixed degradation | Earlier methods detect degradations and apply tools in predetermined order; RestoreAgent and AgenticIR go further with adaptive ordering and replanning | Not novel; current method is less general |
| Route-specific residual alphas | Ordinary output/residual fusion and calibration | Not novel |
| Five-way plus eleven-way agreement | Classifier ensembles and consensus are generic; no exact restoration match was found in the closest four system papers | Possible implementation distinction, weak alone |
| Atomic compositional head plus ambiguity-specific nonlinear heads | DAME-Net/MDUR predicts atomic multi-label factors; specialized ambiguity classifiers are conventional | Partial overlap; not enough alone |
| Learned-router plus physical-rule agreement | Hybrid learned/physical gating is a plausible distinction, but the rule covers a narrow ambiguity and is not a general restoration theory | Possible supporting contribution |
| Fail-closed execution with explicit abstention | The inspected RestoreAgent, AgenticIR, GRIDS, and FoundIR full texts do not define selective routing by coverage/wrong-route risk or use explicit abstention before restoration | Strongest current distinction, but generic selective-prediction prior art remains |
| Conservative partial fallback after abstention | Agentic systems have rollback/reflection; CURE has selective restoration. The current fallback is different but only Low-light-specific and is not proven no-harm | Supporting mechanism, not yet standalone novelty |
| All five official categories pass | Strong application result | Performance/validation contribution, not method novelty |

## 4. Closest prior methods and exact boundary

### RestoreAgent (arXiv:2407.18035)

RestoreAgent uses an MLLM to assess degradation type/extent, determine tasks,
optimize task order, select a model for each task, execute tools, re-plan, and
rollback. It directly preempts broad claims about VLM routing, expert/model
selection, heterogeneous tool use, and dynamic order. Its paper does not use
the terms abstention, rejection, fail-safe, or confidence-based routing in the
restoration decision rule inspected here.

### AgenticIR (ICLR 2025, arXiv:2410.17809)

AgenticIR uses VLM/LLM perception, scheduling, execution, reflection, rollback,
and rescheduling over a restoration toolbox. It also studies consistency of
LLM scheduling. It is broader and more adaptive than the current fixed plans.
Its uncertainty discussion concerns scheduling variation, not a calibrated
coverage-risk abstaining router.

### FoundIR (ICCV 2025, arXiv:2412.01427)

FoundIR already combines a degradation-agnostic diffusion Generalist with a
degradation-aware specialist expert pool. Thus generalist/specialist ensemble
restoration and specialist refinement are not available novelty claims.

### GRIDS (arXiv:2407.12273)

GRIDS estimates degradation relationships, groups restoration tasks, and
adaptively selects grouped models. It preempts degradation-similarity grouping
and grouped-model selection, but does not establish the current fail-closed
multi-evidence execution contract.

### Unified/composite restoration methods

OneRestore, AllRestorer, DPPD, CDG-IR, F2D-Net, CDIR, UniCoRN, DAME-Net/MDUR,
and CURE collectively cover categorical/global descriptors, adaptive factor
weights, dynamic prompts, continuous per-factor degradation intensity,
factorized experts, pixel-wise gating, multi-head control, unseen
compositions, selective restoration, and order consistency. These features
must not be repackaged as the current paper's novelty.

## 5. What can be claimed now

The following are defensible as factual system contributions, with restrained
wording:

1. An implemented, reproducible VLM-mediated restoration system joining a
   fine-tuned degradation representation to a heterogeneous expert portfolio.
2. A fail-closed routing contract based on agreement across categorical,
   compositional, ambiguity-specific, and physical evidence.
3. Explicit separation of route acceptance, abstention, and conservative
   fallback instead of forcing every input through a predicted expert.
4. Full-resolution terminal validation showing all ten official singleton
   SOW checks pass, with 538 correct routes, 62 fallbacks, and zero wrong
   accepted routes in that authorized 600-image scope.
5. A 660-image composition characterization showing strong singleton/double
   behavior and an honest triple-composition failure boundary.

Do not currently claim:

- the first VLM-guided or agentic image restoration system;
- the first expert-routed or heterogeneous All-in-One restorer;
- the first degradation-aware specialist pipeline;
- a unified single restoration model;
- VLM severity-conditioned restoration for the terminal-v8 system;
- general mixed-degradation superiority, because the one triple category
  remains far below Full FoundIR;
- guaranteed safety or zero error beyond the evaluated distribution.

## 6. Best paper framing

The best framing based on the method that actually worked is:

> **Selective-risk restoration orchestration:** when the cost of choosing a
> wrong restoration expert is larger than the cost of abstaining, route only
> under cross-representation evidence agreement and expose the resulting
> accuracy/coverage/restoration-quality trade-off.

The scientific point is not that several classifiers vote. It is that a wrong
restoration action can irreversibly destroy image information, so routing must
be treated as a selective decision problem whose utility is measured after
restoration. The paper should connect routing confidence to downstream
restoration harm, not stop at classification accuracy.

A cautious working title would be:

> **Fail-Closed VLM Routing for Risk-Controlled All-in-One Image Restoration**

Avoid claiming "first" until the targeted selective-prediction literature
search is complete.

## 7. Evidence still required for a defensible method paper

The existing system is not yet sufficient for a strong method-novelty claim.
The minimum missing experiments are:

1. **Routing ablation:** top-1 five-way, top-1 eleven-way, confidence-only,
   probability averaging/ordinary ensemble, cross-head consensus, + nonlinear
   ambiguity head, + physical guard, + fallback.
2. **Risk-coverage curves:** wrong-route risk and accepted coverage over frozen
   thresholds, with confidence intervals and per-class results.
3. **Downstream harm:** PSNR/LPIPS change caused by correct route, wrong route,
   abstention/identity, and fallback. This establishes why selective routing is
   restoration-specific.
4. **No-harm/fallback audit:** measure the fraction of fallback outputs worse
   than the input. The current name `safe_expert_plan` is a code contract, not
   empirical proof of safety.
5. **OOD/open-set test:** unknown degradation types, clean images, unseen
   compositions, and severity/domain shift. A fail-closed claim is weak if
   tested only on the admitted eleven classes.
6. **Closest-system comparison:** RestoreAgent, AgenticIR, FoundIR, and a simple
   confidence-gated router under aligned data/tool constraints where feasible.
7. **Frozen calibration/test separation:** demonstrate that thresholds and
   fallback rules are selected without reusing the final reporting set.
8. **Efficiency:** latency, GPU memory, and expert-call count versus agentic
   replanning. Deterministic low-overhead safety is the clearest practical
   advantage over RestoreAgent/AgenticIR.

If these experiments show that the consensus rule materially lowers
downstream damage at useful coverage, the current method can support a real
paper without inventing a completely new restoration backbone. If a simple
max-softmax threshold performs equally well, the supposed method novelty
collapses to system engineering.

## 8. Overall verdict

The user's intuition is partly correct: the complete implementation is more
distinctive than a generic "VLM + experts" summary suggests. It already has a
coherent paper-worthy idea in its **risk-controlled fail-closed routing
contract** and strong official singleton evidence.

Brutally honestly, it is not yet safe to say "our current method is already
novel" without qualification. The broad architecture is heavily occupied,
the successful system is not the monolithic VLM-conditioned FLUX model, the
triple result is poor, and the strongest distinction lacks the ablations and
OOD evidence needed to prove it matters. The right move is not to invent an
unrelated mechanism; it is to turn the existing fail-closed system into a
precise selective-risk method and test whether its extra machinery beats a
simple confidence-gated baseline.

## Sources

- RestoreAgent: https://arxiv.org/html/2407.18035
- AgenticIR: https://arxiv.org/html/2410.17809
- FoundIR: https://arxiv.org/html/2412.01427
- GRIDS: https://arxiv.org/html/2407.12273
- OneRestore: https://arxiv.org/abs/2407.04621
- CURE: https://arxiv.org/html/2607.03044
- AllRestorer: https://arxiv.org/html/2411.10708
- DAME-Net/MDUR: https://arxiv.org/html/2604.09313
- Existing project literature matrix:
  `reports/foundir_triple_composition_literature_reproduction_memo_2026-09-26.md`
- CURE boundary audit:
  `reports/cure_capability_gap_evidence_table_2026-09-27.md`
