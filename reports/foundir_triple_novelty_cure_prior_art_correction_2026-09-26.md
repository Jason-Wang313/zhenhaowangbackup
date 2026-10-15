# Prior-art correction: CURE overlaps the proposed CSR claim

Date: 2026-09-26 UTC
Status: full-text verification complete for CURE; prior CSR novelty claim withdrawn

## Finding

The proposal in `foundir_triple_composition_literature_reproduction_memo_2026-09-26.md`
treated matched subset-image supervision, selective removal of degradation
components, order consistency, and generalization from singleton/double
training to held-out triples as a potentially novel combination. That claim
is **not defensible** after reading the full text of:

> CURE: Controllable Unified Image Restoration for Complex Degradations,
> arXiv:2607.03044, July 2026, https://arxiv.org/html/2607.03044

| Proposed CSR feature | CURE full-text evidence |
|---|---|
| Matched subset targets | Section 5.2 states that subset-degraded images reuse exactly the same masks and parameters as composite images; Section 5.1 calls these selective-restoration pairs. |
| Remove selected factors and retain others | Section 3.3 introduces an intermediate loss to supervise removal of an individual factor; Section 3.6 evaluates selective restoration with the remaining degradation as GT. |
| Order consistency | Section 3.3 introduces a permutation-invariant loss for two removal orders. |
| Exclude triples in CURE fine-tuning, test on triples | Section 3.4 excludes triple-degradation images from the CURE fine-tuning stage and includes them in testing. Supplement Section 8.1 and Table 9 report selective restoration on triple composites. The preceding baseline and embedder training do see triple classes; see the code audit below. |
| Multiple degradation families | Section 5.3 adds Blur-Noise-JPEG to the weather-related CCDD-11 setting. |

This does not mean no distinct contribution is possible. It means the current
CSR description cannot be claimed as new on the basis of these features or
their combination. A different claim would require a concrete mechanism and
evaluation distinction from CURE, followed by another targeted prior-art
check. Different image categories or a higher score alone are not sufficient
method novelty.

### Qualification from released CURE code

The statement "CURE trains without triples" applies only to the CURE
fine-tuning stage. At released repository commit
`a294116ad3d7dce58723a21bdd64f74b085e7b13`,
`03_train_CURE.py` loads a pretrained OneRestore restorer and a pretrained
embedder. `02_train_Onerestore_baseline.py` uses `BASELINE_TYPES`, which
include `low_haze_rain` and `low_haze_snow`, and
`01_train_OneRestore_embedder.py` uses `EMBEDDER_TYPES`, which also include
those triples. `tools/prepare_h5.py` binds the baseline HDF5 builder to
`BASELINE_TYPES`, while the CURE fine-tuning HDF5 builder uses
`RESTORATION_TYPES` without triples. Sources:

- https://github.com/bo-oseng/CURE/blob/a294116ad3d7dce58723a21bdd64f74b085e7b13/03_train_CURE.py
- https://github.com/bo-oseng/CURE/blob/a294116ad3d7dce58723a21bdd64f74b085e7b13/02_train_Onerestore_baseline.py
- https://github.com/bo-oseng/CURE/blob/a294116ad3d7dce58723a21bdd64f74b085e7b13/01_train_OneRestore_embedder.py
- https://github.com/bo-oseng/CURE/blob/a294116ad3d7dce58723a21bdd64f74b085e7b13/cure/constants.py
- https://github.com/bo-oseng/CURE/blob/a294116ad3d7dce58723a21bdd64f74b085e7b13/tools/prepare_h5.py

The paper's supplement also says its baseline models are trained to
saturation before CURE is applied and describes 11 CCDD-11 degradation
classes. This is a pretraining exposure distinction, not an allegation of
improper test-set access: the triple test images themselves are separate.
An end-to-end triple-free protocol would be stricter than CURE's reported
setup, but DAME-Net already studies genuinely unseen higher-order
combinations. Protocol stringency alone is not established method novelty.

## Relation to the completed R0

R0's three arms collapsed to nearly black output, so its metrics do not test
the proposed research hypothesis. A read-only checkpoint probe on one 96×96
train crop found initially unsaturated outputs, but after training the
pre-clamp residuals were strongly negative: A mean `-3.66408`, B `-2.10201`,
C `-2.78312`. For A, 100% of pre-clamp pixels were negative; for B, 99.837%;
for C, 99.986%. The hard output clamp then maps almost all pixels to zero.
This supports a saturation/gradient-trap diagnosis; it does not isolate the
upstream reason the residual drifted negative. Input and target pairs are not
trivially reversed or black: train input/clean means are 0.4204/0.4252 and
development input/clean means are 0.3661/0.4057.

The user's subsequent `continue` requested investigation, not an automatic
new GPU experiment. R0 v1 artifacts remain frozen and no R1/R2/CSR run has
been launched. The next useful scientific step is to formulate a claim with a
specific difference from CURE, then validate it with the closest baselines.
Restoring the R0 implementation may still be useful as a baseline engineering
task, but it cannot validate the withdrawn CSR novelty claim by itself.

## Search status and limits

The previous memo audited FoundIR, OneRestore, GRIDS, RestoreAgent, DPPD,
CDG-IR, F2D-Net, CDIR, UniCoRN, and MDIR. This correction adds a full-text
check of CURE. AllRestorer and causal distortion-invariant restoration were
also screened by abstract; their abstracts do not overturn the CURE overlap.
This is a targeted search, not a claim of exhaustive novelty clearance.

## Possible next question, not a novelty claim

CURE's Section 4 states that performance may drop beyond three concurrent
degradation types, especially with complex spatially varying degradations.
That is a concrete limitation to investigate. It does not automatically create
novelty: F2D-Net already uses pixel-wise spatial gating, and other composite
restoration work may address four or more factors. The next bounded literature
screen should compare CURE, F2D-Net, AllRestorer, and methods explicitly
testing four-plus/spatially varying compositions before defining a new
experiment. Our current 660-image FoundIR comparison has only one triple
composition and no four-factor category, so it cannot validate this question
by itself.

## Additional targeted screen completed on 2026-09-26

The prior search covered ten named works at varying depth: FoundIR,
OneRestore, GRIDS, RestoreAgent, DPPD, CDG-IR, F2D-Net, CDIR, UniCoRN, and
the Fourier-signature MDIR. This follow-up substantively checked four more:
CURE, AllRestorer, DAME-Net, and DIL. Therefore **14 distinct related works**
have been reviewed at varying depth; this does not mean 14 full reproductions
or 14 full-paper audits. Four titles from the original memo remain only
identified, not method-verified: TSCIR, Fine-grained Structure-Preserving
Composite Degradation Image Restoration, Degradation-aware Feature
Disentanglement, and the distinct Multi-Dimensional Degradation
Representation Learning MDIR.

| Work | Verified overlap or boundary | Source |
|---|---|---|
| CURE (2026) | Matched selective subset targets, order consistency, train without triple, test on triple; directly preempts old CSR claim. | https://arxiv.org/html/2607.03044, Sections 3.3, 3.4, 5.2, 8.1 |
| AllRestorer (2024) | Models relationships among all degradation factors with adaptive factor weights/intensity; ordinary factor weighting is not new. Its CDD-11 setup is not evidence of a held-out triple claim. | https://arxiv.org/html/2411.10708, Sections 1, 4, 5 |
| DAME-Net / MDUR (2026) | Eight atomic types, 43 configurations, up to four factors, 21 seen/22 unseen configurations, factor-wise cues and decoupled MoE; merely adding four-factor or unseen-combination tests is not new. | https://arxiv.org/html/2604.09313, Section IV-A |
| DIL (2023) | Causal/counterfactual distortion augmentation and unseen distortion type/degree generalization; not the same matched subset restoration task, but preempts broad claims about introducing counterfactual reasoning to image restoration. | https://arxiv.org/html/2303.06859, Sections 1, 4 |

The search should now be **claim-driven**. More broad searches will not make
the withdrawn CSR mechanism novel. Before another GPU experiment, articulate
one precise proposed difference from CURE and DAME-Net, then check the closest
papers and code for exactly that difference. If no such difference is clear,
the appropriate outcome is to change the research question, not to keep
training the same arm.
