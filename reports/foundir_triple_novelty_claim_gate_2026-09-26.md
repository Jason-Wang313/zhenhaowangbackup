# Triple-composition novelty claim gate

Date: 2026-09-26 UTC
Status: literature/code audit; no new model training or protected-set access

## Decision

The old Counterfactual Subset Restoration (CSR) method claim is rejected as
novelty. CURE already trains with selective-restoration pairs, single-factor
intermediate losses, and removal-order consistency. A further code audit
identified a narrower, testable **evaluation gap**: CURE excludes triple
degradations only during its CURE fine-tuning stage. Its initial OneRestore
restorer and prompt embedder are trained with triple classes. The paper's
held-out-triple experiment is therefore not an end-to-end triple-free test.
This observation is not a methodological contribution by itself.

## Closest-work matrix

| Work | Selective partial restoration | Strictly unseen combinations from initial training | Operator history/order | Consequence |
|---|---|---|---|---|
| CURE (2026) | Yes: matched partial targets, intermediate/permutation losses | No for its released two-stage protocol: initial restorer and embedder include triples; CURE fine-tuning excludes triples | Randomizes removal order for two factors; weather and Blur-Noise-JPEG synthesis each use a fixed physical order | Direct method baseline and critical exposure-control ablation |
| DAME-Net / MDUR (2026) | Factor-specific correction, but no verified user-requested subset-to-subset target task in the inspected full text | Yes: 21 of 43 configurations train, 22 held out, up to four factors | Stage-wise spatial/frequency experts; no verified operator-history intervention | Strong unseen-composition baseline; four-factor testing is not novel alone |
| OneRestore (2024) | Composition descriptor and CDRL, not explicit arbitrary partial removal | No: listed composition classes appear in training | No verified formation-order control | Base architecture and baseline |
| AllRestorer (2024) | Adaptive per-factor weighting, not verified selective subset targets | Trains/evaluates on CDD-11; no verified strict unseen-composition claim | No verified formation-order control | Adaptive factor weighting alone is old |
| Multiple degraded image restoration via degradation history estimation (ICME 2023) | Not established by its abstract | Not established by its abstract | Explicitly predicts the degradation operation chain and parameters | Formation-order modeling alone is old; full method needs checking before claiming an order-based contribution |
| DIL (CVPR 2023) | No matched subset-to-subset target task in the inspected experiments | Tests unseen distortion types/degrees | Counterfactual distortion augmentation, not verified physical operator-order interventions | Broad counterfactual restoration terminology is old |

Primary evidence:

- CURE paper: https://arxiv.org/html/2607.03044, Sections 3.3, 3.4, 5.2,
  6, 8.1; released code commit
  `a294116ad3d7dce58723a21bdd64f74b085e7b13`, especially
  `03_train_CURE.py`, `02_train_Onerestore_baseline.py`,
  `01_train_OneRestore_embedder.py`, `cure/constants.py`, and
  `tools/prepare_h5.py`.
- DAME-Net paper: https://arxiv.org/html/2604.09313, Section IV-A.
- AllRestorer paper: https://arxiv.org/html/2411.10708, Sections 1 and 4.
- DIL paper: https://arxiv.org/html/2303.06859, Sections 1 and 4.
- Degradation history estimation: DOI `10.1109/ICME55011.2023.00097`;
  abstract inspected, full method pending.

## Candidate research question, not an accepted novelty claim

> Under an end-to-end composition holdout, where no prompt encoder, backbone,
> checkpoint, or hyperparameter-selection stage has seen any triple example,
> does selective restoration learned from singleton/double compositions still
> generalize to triple inputs? If it fails, can an explicitly order-conditioned
> intervention mechanism recover the gap without triple targets?

This question has two parts. The first is a **reproduction/protocol audit** of
CURE. The second would require a distinct method. The first must be answered
before designing the second. It remains possible that CURE works equally well
under strict exclusion, in which case the exposure hypothesis is rejected.

The proposed order-conditioned mechanism also has unresolved prior-art risk:
the 2023 degradation-history method already estimates chains and parameters;
RestoreAgent optimizes restoration order; and CURE randomizes two-factor
removal order. A new method must show an actual mechanism and outcome not
covered by these works. Merely adding an order token, changing the number of
degradations, or combining existing losses is not enough.

## Smallest decisive reproduction before new method work

Use the released CURE code and the same source split, operators, architecture,
optimizer budget, and evaluation protocol for two tracks:

1. Published-style track: train the OneRestore baseline and embedder with
   the published class list, including triples; then fine-tune CURE without
   triples.
2. Strict-holdout track: exclude all triple examples from baseline training,
   embedder training, CURE fine-tuning, and model selection. Keep the same
   source split and training-update budget; document any necessary
   resampling of singleton/double classes.

Freeze both tracks before one triple evaluation. Compare overall restoration,
selective restoration, and per-composition PSNR/SSIM; add LPIPS if the metric
implementation is fixed in advance. If the strict track drops materially,
that is a concrete, reproducible gap to investigate. If it does not, stop this
branch. The current clean-room R0 cannot substitute for this comparison:
its outputs collapsed and its architecture/descriptors differ from released
CURE/OneRestore.

No GPU run is launched by this document. The official FoundIR singleton test
and the 60-image FoundIR triple category remain outside tuning. Any new GPU
experiment needs a frozen config, data hash audit, bounded wall-time/GPU-hour
estimate, and an output-collapse preflight based on actual images.
