# CURE capability/gap evidence table

Date: 2026-09-27 UTC
Scope: paper/repository audit only; no experiment result is used to establish a
novelty claim
CURE paper: *CURE: Controllable Unified Image Restoration for Complex
Degradations*, ICPR 2026, arXiv:2607.03044
Code: official repository commit
`a294116ad3d7dce58723a21bdd64f74b085e7b13`

## Bottom line

CURE has already solved much more than ordinary All-in-One restoration. It
explicitly covers identity/no-operation control, user-selected restoration
strength, selective removal of individual factors, matched subset targets,
and consistency across two restoration orders. These cannot be reused as this
project's novelty.

The clearest remaining gaps are: (1) controllability for spatially varying and
more-than-three-factor degradations, (2) dependence on expensive matched
selective/half-intensity ground truth, and (3) real-world validation of
selective/intensity/order control rather than only ordinary full restoration.
Each is a genuine CURE boundary, but none is yet cleared as a novel method.
Nearby work already covers four-factor unseen compositions, pixel-wise
spatial gating, automatic factor detection, adaptive per-factor weighting,
and degradation-history/order modeling.

## What CURE already solves

| Capability | Status in CURE | Direct evidence | Consequence for our paper |
|---|---|---|---|
| Unified restoration of known single/composite degradations | Solved within its fixed vocabularies | Paper Sec. 3.1 and Tables 1/8; repository defines 12 CCDD scene classes and a second Blur-Noise-JPEG setting | “One model handles multiple degradations” is not novel. |
| Explicit identity/no-restoration operation | Solved | Paper Secs. 3.2–3.3, identity embedding/loss; repository `PromptEncoder.identity()` | Identity preservation or a no-op token is not novel. |
| Continuous user control of restoration strength | Solved for embedding interpolation | Paper Eq. 2/Eq. 4 trains at `w=0.5` and evaluates other values; repository linearly interpolates identity and degradation embeddings | Prompt interpolation or a scalar strength control is not novel. This is user control, not automatic calibrated severity estimation. |
| Selective removal while preserving other factors | Solved on synthetic matched pairs | Paper Secs. 3.3, 3.6, 5.1–5.2 and Tables 5/9/12/13; subset targets reuse the same masks/parameters | Selective restoration, matched counterfactual targets, and “remove A but keep B” are not novel. |
| Component-wise intermediate supervision | Solved | Paper Eq. 5; repository supervises two partial outputs against `type1`/`type2` targets | Per-factor intermediate loss is not novel. |
| Restoration-order consistency | Solved for two-factor removal sequences | Paper Eq. 6 and Table 7; repository evaluates both component orders and penalizes disagreement | Reversing expert/restoration order or adding an order-consistency loss is not novel. |
| Joint and sequential restoration | Solved | Paper Sec. 3.6; official repo exposes one-step and two-stage inference | Merely adding a sequential execution path is not novel. |
| Selective restoration on triple composites | Demonstrated, with an important qualification | Paper Supplement Sec. 8.1/Table 9 evaluates triple selective restoration after CURE fine-tuning excludes triples | “Train CURE without triples and test triples” is already reported. The initial baseline still sees triple classes, so this is not an end-to-end holdout. |
| Generalization outside weather degradations | Demonstrated on Blur-Noise-JPEG | Paper Sec. 3.6 and Supplement Sec. 8.2 | Changing from weather to digital degradations is not a contribution by itself. |
| Ordinary real-world full restoration | Evaluated | Paper Sec. 9 evaluates NPE, RTTS, RS and Snow100K-R with NIQE and qualitative images | CURE is not purely synthetic in its full-restoration evaluation; the remaining real-world gap is specifically controllability. |

## What CURE does not solve, and whether the gap has paper value

| Candidate gap | Evidence that CURE leaves it open | Closest-work collision/risk | Paper value assessment |
|---|---|---|---|
| **Spatially varying control with more than three simultaneous factors** | CURE explicitly states in Sec. 4 that performance may drop beyond three types, particularly for complex spatially varying degradations. Its official control is a global 324-D prompt, not a spatial intervention map. | DAME-Net/MDUR already tests 43 configurations up to four factors with seen/unseen splits and mask-constrained routing. F2D-Net already claims pixel-wise spatial gating. | **High-value problem, medium/high novelty risk.** A defensible contribution would need spatially localized *selective intervention/preservation*, not merely four factors or spatial gating. |
| **Controllability without matched subset and half-intensity GT** | Paper Sec. 5.1 says CURE requires pairs absent from existing datasets. The HDF5 stores clean, full, two selective targets and a half target. One CURE update performs eight restorer passes (full, identity, ratio, ratio-twice, two partial, two sequential). | The targeted audit has not yet cleared unpaired/weakly supervised controllable restoration, consistency distillation, or efficient intervention learning. | **High practical and potential method value; novelty unknown.** “Learn the same controls without explicitly rendered counterfactual/half targets” is a sharper question than another prompt design. Requires a dedicated prior-art search before claiming it. |
| **Real-world selective/intensity/order benchmark** | CURE's selective, ratio and order ground truths are synthetic CCDD/Blur-Noise-JPEG. Sec. 9 real-world evaluation is ordinary full restoration on predominantly single-degradation benchmarks using NIQE; it does not validate real selective targets or control fidelity. | Real composite paired/control datasets may exist outside the current 14-work screen. Dataset contribution quality depends on credible capture/GT, not synthetic mixtures renamed “real.” | **High benchmark value, uncertain algorithm novelty.** Strong if it supplies authoritative real control targets and metrics; weak if it only adds qualitative examples. |
| **Blind/open-vocabulary control and calibrated uncertainty** | Paper motivates the problem “even when degradation types are known.” Official controllable inference requires `--source-prompt`; prompt names must belong to a fixed 12-class vocabulary. The restoration runtime loads only three frozen text tensors from the embedder, not its visual branch. | DAME-Net predicts atomic multi-label factors; AllRestorer uses image+text descriptors and adaptive weights; DPPD, CDG-IR, FoundIR and VLM-guided restoration already occupy dynamic/continuous degradation conditioning. | **Project-relevant but crowded.** A VLM that merely predicts type/severity is not enough. A credible distinction would require open-set recognition, calibrated uncertainty, and demonstrably safer intervention under wrong/unknown factors. |
| **Strict end-to-end unseen-composition evaluation** | `RESTORATION_TYPES` excludes triples only during CURE fine-tuning, while `BASELINE_TYPES` includes both triple classes. The paper initializes from saturated baselines. Thus the reported triple test is not end-to-end triple-free, although it is not test-image leakage. | DAME-Net already has standardized seen/unseen composition splits up to four factors. | **Useful protocol/ablation, low method novelty.** It can correct interpretation or strengthen evaluation, but by itself is unlikely to be a paper method contribution. |
| **Formation-history-aware restoration for non-commuting degradations** | CCDD synthesis uses a fixed physical application order, whereas CURE's permutation loss addresses the *removal* order and encourages the final result to be invariant. It does not infer or intervene on the unknown degradation-formation chain. | ICME 2023 explicitly studies degradation-history estimation; RestoreAgent studies restoration ordering/planning. Their full boundaries must be checked. | **Scientifically meaningful, high prior-art risk.** “Add an order token” is not novel. Potential value requires showing why formation history changes the correct selective intervention and how the method differs from history estimation/planning. |
| **Automatic physical per-factor severity estimation** | CURE's `w` is a user-selected interpolation coefficient. It is supervised only with a half-degradation target and is not an estimate of the input's physical severity or uncertainty. | AllRestorer has adaptive per-factor weights; CDG-IR claims multi-dimensional continuous degradation intensity prediction; DPPD learns prompt distributions. | **Real gap from CURE, but low novelty confidence.** Continuous severity vectors or adaptive weights alone are already occupied. |
| **Scaling to four or more factors, without any other distinction** | Explicit CURE limitation beyond three factors | DAME-Net already evaluates up to four-factor unseen configurations | **Not a viable novelty claim alone.** |

## Claims that are already ruled out

The following should not be proposed as the paper's main novelty:

- selective restoration or matched subset targets;
- identity/no-restoration control;
- scalar or interpolated restoration strength;
- component-wise intermediate supervision;
- restoration-order consistency;
- testing triples after singleton/double CURE fine-tuning;
- four-factor evaluation by itself;
- dynamic prompts, continuous type/severity vectors, factor weighting,
  pixel-wise gating, or automatic factor detection by themselves;
- a stricter train/test split presented as if it were a new restoration
  mechanism.

## Most defensible directions after this audit

These are research questions, not novelty claims:

1. **Spatial intervention:** Can a model remove a user-selected degradation
   only where it occurs, while preserving other degradations and unaffected
   regions, under unseen higher-order real compositions? This must be
   distinguished from F2D-Net spatial gating and DAME-Net mask-constrained
   expert routing.
2. **Pair-free controllability:** Can identity, intensity, selective and
   order-consistent control be learned without CURE's matched subset and
   half-intensity targets, with substantially less storage/compute? This is
   the clearest repo-derived systems/method gap, but its prior art has not yet
   been exhaustively checked.
3. **Real controllability benchmark:** Can selective removal, strength
   calibration and preservation leakage be measured on genuinely real
   composite degradations with authoritative targets? This is more naturally
   a dataset/evaluation contribution than a standalone model novelty.

The current evidence does **not** justify selecting one of these as novel.
The next literature step, if requested, should be claim-driven: fully audit
the closest papers for only one candidate rather than broadly collecting more
All-in-One papers.

## Evidence and source boundaries

Primary sources:

- CURE full text: https://arxiv.org/html/2607.03044, especially Secs. 3.2–3.6,
  4, 5.1–5.2, 8.1–8.2, 9 and 10.
- Official CURE repository at commit
  `a294116ad3d7dce58723a21bdd64f74b085e7b13`:
  `cure/constants.py`, `cure/embeddings.py`, `03_train_CURE.py`,
  `tools/prepare_h5.py`, `inference_selective_control.py`, and
  `inference_ratio_control.py`.
- Existing closest-work audits:
  `reports/foundir_triple_novelty_claim_gate_2026-09-26.md` and
  `reports/foundir_triple_novelty_cure_prior_art_correction_2026-09-26.md`.

Closest-work boundaries used here:

- DAME-Net/MDUR: arXiv:2604.09313, atomic multi-label detection, 43
  configurations, up to four factors, standardized seen/unseen splits.
- AllRestorer: arXiv:2411.10708, image+text composite descriptors and adaptive
  per-factor weights.
- F2D-Net: DOI `10.1016/j.knosys.2026.116411`, factorized experts and
  pixel-wise spatial gating.
- Multiple degraded image restoration via degradation history estimation:
  DOI `10.1109/ICME55011.2023.00097`.
- RestoreAgent: arXiv:2407.18035, tool/order planning and replanning.
- CDG-IR: DOI `10.1109/TMM.2026.3702504`, multi-dimensional continuous
  degradation intensity prediction.

This is a targeted audit, not exhaustive novelty clearance. “Paper value” in
the table means the gap is meaningful enough to investigate; it does not mean
the method or claim is novel.
