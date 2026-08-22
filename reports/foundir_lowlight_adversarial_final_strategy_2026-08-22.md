# FoundIR-Lowlight adversarially reviewed execution strategy

Date: 2026-08-22 UTC

## Status and outcome

This document is the frozen plan after 29 autonomous adversarial-review rounds.
No experiment described here may start until the applicable stage contract and
the first-stage items in the pre-experiment checklist are complete. Later-stage
contracts are frozen before their own execution as specified in attack 27;
they are not guessed before their prerequisites exist. Freezing the plan does
not claim that its external dependencies or scientific gates have passed. This
is planning closure: further progress comes from executing the seven gates, not
another speculative strategy round. New empirical evidence follows the frozen
branches and stop rules; it does not authorize new disposable tests or an
unbounded module search.

The 29 attacks are a risk catalog, not 29 execution tasks. The executable plan
is the seven-gate critical path below. The strongest defensible route is not a
monolithic stack of every promising idea. It is a non-regressing proposal
ladder whose smallest advisor-compliant deployed system is:

```text
degraded RGB -> one selected conservative pixel/teacher base proposal ---+
degraded RGB -> measured statistics + frozen real-validated Qwen condition +
degraded RGB -------------------------------------------------------------+
                                                                          v
                                      degraded-anchored FLUX.2 bounded delta
                                                                          v
                                       base + delta -> native-resolution output
```

The released FoundIR proposals are mandatory to measure because their weights
are already present and benchmark-relevant; they are not mandatory at deployed
inference. A learned photometric head, learned proposal fusion, new camera
simulator, pixel-frequency bypass, stronger spatial VLM conditioning,
camera-forward consistency, quantization-aware training, and inference
ensembles are conditional research proposals, not default steps. The basic
causally validated VLM condition and genuine FLUX residual contribution remain
mandatory for the advisor-directed deliverable. A conditional proposal is not
implemented unless a cheaper diagnostic identifies its exact bottleneck and
predicts enough headroom to cross the existing promotion margin.

This plan addresses the known avoidable weaknesses found in these reviews. It
does not claim literal perfection and cannot eliminate the
irreducible uncertainty caused by the protected FoundIR test distribution and
the unavailable official training archive. No official FoundIR test image is
used during development.

## Autonomous attack 1: benchmark and objective

Attacks:

- A system can cross the contractual `18.9 dB / 0.28 LPIPS` threshold without
  being SOTA. The competitive target must be the best result under exactly the
  same isolated `14Lowlight`, full-resolution RGB protocol, not a number from a
  different crop, category mixture, or color space.
- The FoundIR comparison table itself reports `21.90 dB` for AutoDIR and
  `20.04 dB` for InstructIR on isolated Lowlight, versus `18.98 dB` for
  FoundIR. Therefore beating the FoundIR row is not a SOTA claim.
- PSNR rewards the conditional mean; LPIPS can reward plausible detail that
  harms pixel fidelity. Pure diffusion sampling is therefore misaligned with
  the primary metric.
- The released FoundIR environment exists and pins `lpips==0.1.4`, but it does
  not include or pin `pyiqa`; the README requests an unversioned install.
  Metric-weight and implementation differences could look like progress.
- Fifty pairs have high variance. A mean-only decision can hide catastrophic
  camera or darkness-regime failures.

Decisions:

- Freeze the official evaluator revision and a finite two-path LPIPS check: one
  pinned compatible `pyiqa` release through the FoundIR script plus direct
  `lpips==0.1.4` AlexNet. Freeze backbone/weights hashes, RGB range, image
  decoder, resize behavior, and aggregation before the protected run. Because the
  released FoundIR environment pins `lpips==0.1.4` but not `pyiqa`, one locally
  chosen `pyiqa` version is not treated as the Huawei contract evaluator.
  Cross-check PSNR with two independent implementations and require the final
  candidate to satisfy the LPIPS target under both frozen paths, unless Huawei
  supplies the exact evaluator, which then becomes authoritative.
- Select checkpoints on paired real development by PSNR/LPIPS Pareto dominance,
  worst-dataset risk, per-scene paired deltas, and scene bootstrap intervals.
- Use deterministic or very low-variance transport. Stochastic candidates and
  averages earn inclusion only on held-out development data.
- The protected run is all 50 pairs once, never sequential public-test slices.
- The minimum unambiguous published-table target is greater than `21.90 dB`.
  The engineering target is at least `22.50 dB` and contract-evaluator LPIPS at
  most `0.25`, leaving PSNR margin over the table and LPIPS margin under the
  contract. The public table's second metric is SSIM, so `0.25` is not claimed
  as an LPIPS-SOTA threshold. A dated audit of newer comparable methods is
  repeated immediately before final freeze; the target moves upward if a
  stronger identical-protocol result is verified.

## Autonomous attack 2: data and identifiability

Attacks:

- The `+15.6853 dB` gamma-plus-color oracle sees each target. Its parameters
  are non-unique and can encode the dataset's preferred target rendering. The
  dark input may not identify that rendering.
- LOL-v2 and SICE contain only two target-render distributions, alignment
  imperfections, repeated exposures per scene, clipping, and proxy metadata.
  Random record splits would leak scene and camera style.
- Sophisticated synthetic camera degradation can still be distributionally
  wrong. More synthetic volume can amplify a wrong simulator.
- Teacher outputs are proposals, not truth. Blind distillation can inherit
  their artifacts and suppress a better paired target.
- FoundIR official training access is an external dependency. A strategy that
  assumes access is not executable today.

Decisions:

- First measure whether bounded, identifiable target summaries are predictable
  from input statistics under nested scene-grouped and leave-one-dataset-out
  evaluation. This is a diagnostic, not the final head. The unconstrained
  gamma-plus-matrix fit remains only a target-accessing upper ceiling.
- Train all deployable components from input alone. Targets are allowed only
  as ordinary training/development supervision, never at inference.
- Use leave-one-dataset-out development and scene grouping. Alignment-confidence
  masks gate strict pixel losses; no development pair is silently warped or
  removed to improve a score.
- Fit RAW/camera simulator distributions only from allowed training images and
  validate each simulated statistic against held-out real inputs. Synthetic
  data remains a coverage/replay stream, not the sole domain.
- Maintain two data branches:
  - Plan A: authorized official FoundIR training pairs, beginning with 39,962
    isolated-Lowlight pairs, plus released teachers and independent real pairs.
  - Plan B: independent paired real data, existing controlled synthetic replay,
    and confidence-weighted released-teacher proposals, with no official test
    use. A new camera simulator is conditional on a measured coverage gap.

## Autonomous attack 3: architecture

Attacks:

- FLUX.2 is a contractual/advisor-prioritized backbone, but a large generative
  transformer is not automatically better at pixel-exact inverse imaging than
  a compact deterministic restoration path.
- Starting from random noise forces scene regeneration and increases variance.
- The VAE round trip is excellent (`38.4488 dB`, `0.005159 LPIPS`), and its only
  predeclared failure is a narrow LOL-v2 high-frequency correlation miss of
  `0.00187909`. This does not prove that copying high frequencies from a noisy
  input helps restoration.
- An unconstrained photometric matrix can create negative cross-channel mixes,
  clipping, color instability, and misleading parameter-regression targets.
- A camera-forward cycle can be satisfied by a wrong clean image paired with
  compensating degradation parameters.
- Qwen's earlier one-token path was causally near-null, and the real adaptation
  gate failed. Keeping Qwen in the critical path by default would repeat the
  same mistake at higher cost.

Decisions:

- Prefer an information-preserving degraded-anchored bridge, but freeze its
  endpoint only after the pure-noise, deterministic interpolation, and
  low-noise residual-flow comparison. Every candidate receives decoded-RGB
  conditional-mean supervision.
- The photometric head predicts bounded exposure, monotone tone curves,
  white balance, a near-identity color residual, and a smooth illumination
  field. It does not directly regress the ill-conditioned oracle matrix as the
  final design.
- FLUX predicts only the residual the conservative control path cannot explain;
  fusion is zero-initialized and bounded so every added branch starts from the
  previous proposal.
- The frequency bypass is signal/noise/confidence gated and tested in three
  forms: off, fixed conservative Laplacian, and learned bounded fusion. It is
  removed unless degraded-image restoration ablations improve held-out PSNR
  and LPIPS on every development dataset.
- Camera consistency is auxiliary only. Paired reconstruction and identity
  constraints prevent the cycle from defining the solution by itself.
- Qwen is mandatory for the advisor-directed deliverable, but it enters FLUX
  only after oracle-plus-measured/measured-only/measured-plus-Qwen/
  measured-plus-shuffled-Qwen causal ablations on the
  compact control show incremental downstream value. Until then, only the
  non-diffusion control/interface diagnostic is authorized.

## Autonomous attack 4: optimization and losses

Attacks:

- Adding L2, Charbonnier, SSIM, LPIPS, edge, frequency, color, identity, flow,
  distillation, and cycle losses at once makes the objective unidentifiable.
- Average validation improvement can be driven by the easier dataset while the
  difficult camera domain regresses.
- Full-resolution images are too large for naive end-to-end training, while
  crop-only training misses global tone and stitching failures.
- Larger adapters and longer training can memorize target style without
  learning a camera-general inverse.

Decisions:

- Stage objectives: photometric RGB loss; latent bridge plus decoded RGB loss;
  then one auxiliary family at a time. Loss weights are selected inside
  training datasets and frozen before omitted-dataset evaluation.
- Use scene/camera-balanced sampling, darkness/noise stratification, real-pair
  replay, and hard-example queues based on teacher/model disagreement.
- Select by a robust score combining mean and worst-dataset PSNR/LPIPS, with an
  explicit identity gate on already-good inputs.
- Train on matched local crops plus a low-resolution global view. Validate the
  exact native-resolution tiler before scaling capacity.
- Scale capacity only when a measured proposal ceiling shows under-capacity;
  do not use parameter count as a proxy for restoration quality.

## Autonomous attack 5: inference and integration

Attacks:

- Tiling can erase the crop-level gain through seams and inconsistent global
  tone. FoundIR images are 2.7K to 3.7K wide, so this is a primary failure mode.
- D4, multi-seed, checkpoint soups, test-time adaptation, teachers, and VLM can
  multiply runtime and failure surfaces.
- Candidate selection by a no-reference metric can prefer attractive but
  pixel-wrong outputs.
- A large module stack makes it impossible to attribute a final failure.

Decisions:

- Use one low-resolution global prediction for tone/color/illumination and
  overlapping local tiles with halos, coordinate channels, and windowed fusion.
- Freeze a runtime/memory budget before the protected test. Optional inference
  mechanisms are retained only when their gain per cost is material and stable.
- No-reference metrics and camera consistency may reject pathological
  candidates, but cannot choose among ordinary outputs unless that rule wins on
  every held-out paired dataset.
- Maintain exact ablation provenance so the deployed system is the smallest
  proposal that reaches the development gate.

## Autonomous attack 6: the official reference implementation

The official source invalidates a weak assumption in the earlier draft.
FoundIR is already an input-preserving residual system:

- it predicts the pixel residual `LQ - GT`, not an image independently from
  pure noise;
- the degraded RGB is concatenated as a spatial condition;
- it trains with direct L1 residual supervision;
- released real-image inference uses four diffusion steps;
- full-resolution inference uses 1024-pixel patches with stride 512;
- its optional Lowlight route is a cascade: generalist output, then the compact
  1.51M-parameter Lowlight specialist. The specialist is not applied to the raw
  input in the documented route.

Consequences:

- The official generalist and documented generalist-to-specialist cascade are
  mandatory reference proposals before designing a more expensive bridge.
- A compact deterministic pixel-residual control with the same information
  advantage is a mandatory architecture control. FLUX must beat it
  incrementally; model size is not evidence.
- The new FLUX objective must compare three endpoint constructions on a bounded
  gate: the current pure-noise target with degraded reference, a deterministic
  degraded-to-clean latent interpolation, and a degraded-anchored low-noise
  residual flow. The winner is selected without FoundIR test access.
- A bridge endpoint is not frozen from intuition. Starting directly from a
  degraded latent may be out of distribution for pretrained FLUX, while pure
  noise discards too much identity. The bounded comparison resolves that risk.

## Autonomous attack 7: the strongest use of released FoundIR knowledge

Restricting the released models to offline distillation may throw away the
most benchmark-relevant prior available. The stronger candidate is a
teacher-anchored residual booster:

```text
raw degraded RGB ------------------------------+
                                                |
raw -> FoundIR generalist -> Lowlight specialist+-> proposal encoder
                                                |
bounded photometric proposal ------------------+
                                                v
                           FLUX residual/fusion model
                                                |
                           base proposal + bounded delta
```

The fusion delta is zero-initialized, confidence gated, and frequency bounded.
At initialization the output is exactly the selected conservative proposal.
This gives FLUX access to the released FoundIR distribution while making it
learn corrections rather than regenerate the scene.

The teacher cascade is allowed at deployed inference only if all of the
following hold:

- checkpoint licenses and redistribution requirements are compatible with the
  deliverable;
- the documented cascade is reproducible and its runtime fits the frozen
  budget;
- it is strong or complementary on every omitted real development dataset;
- teacher-anchored FLUX beats the same FLUX system without teacher proposals;
- the final output is genuinely changed and improved by the trained FLUX path,
  rather than merely relabeling the public teacher as a FLUX result.

If any condition fails, teacher-at-inference is removed. Training-only
distillation is also conditional on its own cheap admission gate; otherwise the
deployable base is the compact pixel path or another retained proposal.

FoundIR is not the only reference proposal. The paper's stronger isolated
Lowlight rows make AutoDIR and InstructIR mandatory table references, not
mandatory executions:

- InstructIR has an MIT-licensed public repository and may be audited and used
  as a reference/teacher, never the final advisor-directed backbone.
- AutoDIR has public inference code and weights but its repository exposes no
  license metadata. It cannot enter a redistributable training or inference
  path without explicit license/author clearance. Its `21.90 dB` reported
  result remains the minimum competitive reference even if clearance is not
  obtained.
- Any newer method, including one-step degradation-aware diffusion with a
  high-fidelity decoder, is added to the dated table audit after
  identical-protocol verification. Code execution still requires attack 28's
  admission rule. Its useful design principles are already covered by the
  one/few-step residual endpoint, degradation-aware adapters, and gated
  high-frequency decoder/bypass in this plan.

## Autonomous attack 8: development coverage is still too narrow

Two paired datasets and 82 scene components cannot establish camera-general
SOTA readiness. Before a final system freeze, development must contain at
least three independent target-render/camera families. Candidate additions are
accepted only after provenance, license, alignment, scene grouping, and
protected-overlap audits. Priority is:

1. another aligned RGB paired-real dataset such as LSRW or SDSD;
2. a raw short/long-exposure dataset such as SID or SMID, rendered through a
   frozen disclosed pipeline and grouped by camera;
3. the advisor/Huawei hidden real set for final independent confirmation.

Raw-derived and RGB targets are reported separately because their render
pipelines are not interchangeable. Unpaired phone images remain qualitative
robustness data and never select PSNR-oriented changes.

New candidates are compared against existing protected hashes and the already
frozen overlap index. No additional FoundIR test image is opened to perform
this screening.

The primary third-family acquisition is now SDSD because its official release
provides mechatronically aligned low/normal dynamic-video pairs at 1920 by
1080, with 70 indoor and 80 outdoor sequences. Indoor and outdoor are separate
strata inside one SDSD family, not two independent datasets; sequence, not
frame, is the grouping unit, and temporally adjacent frames are aggressively
subsampled. LOL-v2, SICE, and SDSD are therefore the minimum three independent
families. The repository and data do not expose a clear redistribution license,
so they may be used only for internal research after terms review and may not
be packaged in the deliverable without permission. The official processed-SMID
release is the next raw-derived fallback.

## Autonomous attack 9: gates can create their own endless loop

Requiring a positive 95% interval for every exploratory idea can reject useful
components on a small development suite; allowing unlimited corrections can
produce the endless loop the user identified. The process therefore has two
levels and a fixed budget:

- promotion to one larger confirmation run requires a useful point estimate on
  every outer omitted family and no material worst-family regression;
- deployment retention requires a positive scene-bootstrap paired interval,
  improvement on at least three independent development families, and no
  material PSNR or LPIPS regression in any family;
- the mandatory Qwen/interface gets at most two corrections, the FLUX endpoint
  gets one, and an optional proposal gets one pilot with no automatic rescue;
- no failed branch is rescued by acquiring a new test set; the same reusable
  leave-one-dataset-out suites diagnose every correction;
- only one architecture branch is changed per comparison.

This budget does not prevent later scaling of a branch that already passed. It
prevents repeated threshold chasing with new data names.

## Autonomous attack 10: failure attribution before scaling

Every failed proposal is assigned to exactly one leading bottleneck using
ceilings and counterfactuals:

- weak oracle/proposal ceiling -> data or proposal capacity;
- strong oracle proposal but weak learned output -> prediction/optimization;
- strong candidate exists but fused output is weak -> gating/fusion;
- crop gain but native-resolution loss -> global context/tiling;
- oracle condition helps but Qwen does not -> VLM estimation;
- oracle and shuffled conditions match -> conditioning interface;
- simulated validation improves while omitted real data falls -> simulator
  domain error;
- all real development improves but hidden/final test falls -> irreducible
  domain shift, reported honestly rather than followed by public-test tuning.

## Autonomous attack 11: benchmark artifacts must not become leakage

The official repository contains example Lowlight inputs/results and links an
archive of the authors' restored test outputs. Those artifacts are especially
tempting for learning target style or blend weights, but doing so would tune on
the protected benchmark. Therefore:

- do not open or run project inference on bundled IDs `1118` or `1127`;
- do not download or inspect the released FoundIR test-output archive;
- do not use any public method's precomputed outputs for IDs `1101--1150`;
- public checkpoints may be run on protected IDs only once as part of the final
  frozen system/evaluator after all development decisions are locked;
- all teacher development inference is on non-FoundIR images.

## Autonomous attack 12: simulator realism must be measurable

Adding every camera effect is not enough. A simulator can contain shot noise,
read noise, black-level error, Bayer/ISP operations, white-balance shifts,
clipping, vignetting, tone curves, quantization, and JPEG while using the wrong
joint distribution. Simulator acceptance therefore requires:

- marginal and joint-statistic matching to allowed real low-light inputs by
  camera/dataset, including signal-noise curves, dark-pixel histograms, channel
  ratios, clipping, spatial illumination, spectral slopes, and compression;
- a frozen real-versus-synthetic domain-classifier test whose held-out accuracy
  must move toward chance without reducing physical parameter coverage;
- leave-one-simulator-family-out training so one renderer cannot dominate;
- downstream omitted-real improvement. Statistical similarity alone never
  authorizes scale-up.

## Autonomous attack 13: loss alignment under imperfect registration

Pure L2 best matches PSNR but can punish a correct output when pairs move;
pure L1 is robust but does not directly optimize the conditional mean. The
training objective is therefore confidence partitioned:

- high-confidence aligned pixels use RGB MSE plus a small Charbonnier term;
- uncertain motion/occlusion boundaries use masked Charbonnier and structural
  losses, never unmasked MSE;
- low-frequency tone/color receives explicit global supervision;
- LPIPS and frequency losses enter only after a PSNR-strong checkpoint and are
  bounded so they cannot erase more PSNR than the frozen equivalence margin;
- the clean/already-good identity stream receives an exact residual-to-zero
  constraint.

The loss stack is built sequentially. Each auxiliary loss is ablated alone;
there is no single run containing all unproven losses.

## Autonomous attack 14: VLM usefulness may be intrinsically redundant

For isolated Lowlight the restoration network already observes the degraded
pixels and the degradation type is constant. A coarse five-level token may add
no information. The VLM route is made harder to ignore and easier to falsify:

- oracle condition contains continuous exposure deficit, noise parameters,
  clipping, color shift, and illumination maps, not only L1--L5;
- measured-statistics condition is a cheap non-VLM control;
- Qwen supplies soft canonical posteriors, uncertainty, structured tokens, and
  semantic-preservation features through a compact global FiLM adapter;
- condition dropout, swaps, and monotone counterfactuals force sensitivity;
- the four-way oracle-plus-measured/measured-only/measured-plus-Qwen/
  measured-plus-shuffled-Qwen matrix uses identical restoration weights and
  paired scene-bootstrap deltas.

If oracle does not help, the injection is redesigned. If measured-only matches
measured-plus-Qwen, Qwen has not earned benchmark-critical complexity. If
measured-plus-Qwen does not beat measured-plus-shuffled-Qwen and measured-only
after at most two genuine redesigns,
the branch may continue only as a restoration research baseline. It is not the
advisor-directed VLM-guided deliverable, no VLM-guided Lowlight gain is
claimed, and the protected FoundIR run remains blocked pending advisor/user
review. This is a hard honesty rule, not an invitation to keep finding new
severity test sets.

## Autonomous attack 15: fusion can overfit even when proposals are strong

Zero initialization prevents damage at step zero; it does not guarantee
non-regression on a new camera after training. A gate trained on in-sample
proposal outputs can simply memorize which proposal matches each target.
Controls are:

- train fusion only from scene-grouped out-of-fold proposal outputs;
- restrict the base mixture to a convex combination, then add a separately
  bounded residual whose norm and frequency response are regularized;
- calibrate proposal disagreement and restoration confidence on omitted
  datasets, with an input-only out-of-distribution fallback to the conservative
  teacher/pixel proposal;
- report proposal, convex-mixture, and residual contributions separately;
- never choose a proposal or blend per protected image using a target or a
  no-reference aesthetic score.

These controls reduce regression risk but do not mathematically guarantee
performance under unseen FoundIR shift. That remaining uncertainty is why the
final public test stays single-use.

The final fusion is explicitly:

```text
p(x) = sum_k softmax(g(x))_k * proposal_k(x)
y(x) = clip(p(x) + a(x, c) * bounded_delta_flux(x, p(x), c), 0, 1)
```

`g` is trained only from out-of-fold proposals; `a` is a bounded confidence
map; `c` is the frozen Qwen structured condition. Proposal weights are
non-negative and sum to one. The FLUX delta starts at zero. If the optional
frequency path passes, it can replace only bands whose predicted clean-signal
confidence is higher than the conservative proposal. The low-resolution global
path predicts tone/color/illumination controls, not replacement scene content.

## Autonomous attack 16: evaluator reproduction can create a false win

The official repository's unpinned `pyiqa` dependency means that a locally
stable number can still disagree with the acceptance evaluator. The defense is
not to select the most favorable package version:

- build uint8 RGB canaries for identity, one-code-value error, clipping,
  channel permutation, resize, and a licensed non-protected known pair;
- compute PSNR independently with the contract wrapper and a literal NumPy
  reference, requiring numerical agreement after the exact save/reload path;
- evaluate development outputs with the two frozen LPIPS paths and recorded
  weight hashes; use the worse value for promotion;
- obtain Huawei's evaluator/container before the final run if possible. If it
  remains unavailable, report implementation sensitivity and preserve margin
  under every reproduced variant rather than claiming exact evaluator parity.

Metric variants are diagnostic implementations of one frozen output; they are
not extra objectives to tune independently.

## Autonomous attack 17: teachers and development sets may share training data

FoundIR, InstructIR, AutoDIR, Qwen, or other public checkpoints may have trained
on LOL, SICE, SDSD, their source photographs, or semantically near-identical
images. A teacher's strong score could therefore be memorization rather than
transfer. Before any teacher result supports a generalization claim:

- audit papers, configs, manifests, release notes, and author disclosures for
  training provenance;
- screen exact hashes, perceptual hashes, scene/source names, video sequences,
  and semantic near-neighbors wherever the underlying training list is
  available;
- label every teacher/dataset relation `verified_disjoint`, `known_overlap`, or
  `unverifiable`;
- treat `known_overlap` and `unverifiable` teacher results as in-domain proposal
  evidence only, never independent generalization evidence;
- require at least one provenance-independent outer family or hidden sentinel
  for a finalist that depends on a teacher.

Base-model pretraining provenance can rarely be proven exhaustively. That
uncertainty is disclosed rather than converted into a false clean split.

## Autonomous attack 18: the proposal ladder itself can overfit development

Two corrections per branch do not control the combined multiplicity from many
branches, losses, endpoints, gates, seeds, and optional modules. The experiment
process therefore becomes a predeclared, cost-aware selection protocol:

- assign every run a ledger entry before launch: hypothesis, changed factor,
  data folds, seed set, metrics, gate, compute, and correction count;
- for cheap cached proposals and photometric diagnostics, choose settings only
  in scene/sequence-grouped inner folds and measure once on each omitted outer
  family;
- for expensive Qwen/FLUX branches, use one-seed bounded pilots with frozen
  source-disjoint train/validation strata from all three families and select by
  worst-family behavior. Do not pay for full three-fold training of branches
  that will be killed at pilot scale;
- never promote a module from a pooled mean that hides a failing family;
- reserve a provenance-independent sentinel or Huawei/advisor hidden set for
  one final confirmation after the architecture is locked;
- track the total number of compared proposals and report all attempts, not
  only the winner;
- after the finalist architecture is declared, allow only implementation-bug
  fixes whose expected output equivalence is tested on canaries. A metric-driven
  change reopens development and consumes a new confirmation opportunity.

With only three families, outer-fold estimates remain noisy. They are a
disciplined development estimate, not proof of universal camera generalization.

## Autonomous attack 19: a lucky optimization seed is not a SOTA system

A deterministic seed is useful for debugging but insufficient for selecting a
high-variance diffusion finalist. Seed policy is frozen before results:

- exploratory controls use one predeclared seed and may not be rescued by
  changing it;
- only the single architecture selected for deployment is trained with three
  predeclared seeds at the same data/compute budget;
- never select the best training seed. Report mean, standard deviation,
  scene-bootstrap uncertainty, and the worst seed/family result;
- either ensemble a predeclared fixed set only if the full set passes the cost
  and held-out gate, or deploy a seed fixed before finalist results;
- promotion intervals incorporate both scene sampling and training-seed
  variation. A gain smaller than seed variance is not treated as architectural
  progress.

Inference stochasticity is separately controlled: fixed sampler seeds and
deterministic settings are part of the frozen checkpoint identity.

## Autonomous attack 20: photometric parameters remain non-identifiable

Even ridge regression does not make gamma and a full color matrix uniquely
predictable. Multiple parameter combinations can render nearly identical
images, so parameter error can reject a good restoration or reward an unstable
one. Replace parameter regression with a nested ceiling ladder:

1. exposure-only scalar oracle;
2. bounded per-channel white balance plus monotone piecewise-linear tone oracle;
3. the same plus a low-rank smooth illumination field;
4. the unconstrained gamma-plus-matrix oracle, retained only as a ceiling.

The deployable head predicts bounded tone-knot values, positive normalized
white balance, and smooth illumination directly, or predicts target image
summaries such as luminance quantiles and chromaticity moments. It is judged on
the rendered output, monotonicity, clipping, and transfer—not raw parameter
distance. The smallest nested oracle that explains most of the `+15.6853 dB`
gain defines the learnable target; a large gap to the full matrix is reported
as target-render ambiguity rather than hidden inside a more flexible head.

## Autonomous attack 21: Qwen should not estimate sensor maps from language

Qwen's useful comparative advantage is global degradation semantics,
uncertainty, scene content, and semantic-preservation features. It is not
assumed to recover pixel-scale shot/read noise, clipping masks, or spatial
illumination maps from language tokens. Conditioning is divided by information
scale:

- a deterministic measured-statistics path and compact spatial CNN estimate
  SNR/noise proxies, saturation/clipping, local illumination, and confidence;
- Qwen outputs soft global type/severity posteriors, exposure-order uncertainty,
  scene semantics, and features that protect salient content;
- the necessary causal matrix is oracle-plus-measured, measured-only,
  measured-plus-Qwen, and measured-plus-shuffled-Qwen, using the same
  restoration weights;
- the advisor-directed claim requires measured-plus-Qwen to beat measured-only
  and measured-plus-shuffled-Qwen on omitted paired data;
- spatial Qwen injection is optional and must show information beyond the
  compact CNN rather than merely adding parameters.

This keeps the VLM genuinely causal while assigning low-level inverse imaging
to the component that observes the relevant pixels.

## Autonomous attack 22: crop retention and saved-output semantics are too weak

Retaining only 95% of a crop gain could discard roughly `0.7 dB` when the gain
is large and erase the engineering margin. Full-resolution validation instead
uses matched whole-image reference inference and tiled inference on canaries:

- output dimensions, RGB channel order, EXIF orientation behavior, clipping,
  and file count must match exactly;
- tiled output may lose at most `0.10 dB` PSNR and `0.002` LPIPS versus the
  feasible whole-image reference on matched canaries;
- overlap seams, tilewise color jumps, border behavior, and global luminance
  drift have separate predeclared absolute limits derived from clean and target
  canaries before candidate inference;
- native-resolution performance must also retain the development promotion
  gain within its normal statistical equivalence margin, not a percentage that
  scales with an oracle-sized gain;
- evaluation, checkpoint selection, fusion, and any ensemble operate on the
  exact saved sRGB output after clipping and uint8 round-to-nearest. Averaging
  occurs in metric sRGB space, followed by one frozen quantization step;
- late quantization-aware fine-tuning or a straight-through estimator is an
  optional isolated proposal only if it improves omitted-family saved-PNG
  metrics. Float-only gains do not count.

## Autonomous attack 23: a real-versus-synthetic classifier is gameable

A classifier may exploit scene content rather than degradation, while a
simulator can drive accuracy toward chance by destroying useful information.
Simulator validation is therefore conditional and causal:

- compare residual/noise statistics on content-matched clean sources or
  semantic/brightness-matched strata, not arbitrary real and synthetic scenes;
- condition discrimination on the clean source where paired data exists, and
  audit which image regions/features drive the decision;
- test recovery/calibration of simulator parameters that are identifiable,
  joint-distribution coverage, and support holes rather than only binary
  accuracy;
- accept a simulator family only when adding it improves an omitted real family
  and removing it produces the expected degradation-specific loss;
- keep a clean-information and identity floor so making synthetic images less
  distinguishable by corruption cannot pass;
- never claim physical shot/read-noise recovery from sRGB alone. Such values
  are effective sRGB noise parameters unless RAW metadata makes them
  identifiable.

## Autonomous attack 24: input-only fallback still has deployment uncertainty

Cross-fitted fusion and an out-of-distribution score reduce risk but cannot
guarantee non-regression on a new camera. The fallback itself must be calibrated
without target or benchmark feedback:

- define OOD features and thresholds only on training/inner-validation inputs;
- stress the gate with camera shifts, exposure extremes, compression, clipping,
  and proposal disagreement while targets remain hidden during threshold fit;
- compare always-conservative, always-learned, and gated policies on every
  omitted family; retain gating only if it improves worst-family risk;
- bound per-pixel and global deviation from the conservative proposal, with an
  emergency identity/conservative route for non-finite values or unsupported
  dimensions;
- disclose every checkpoint, teacher, license, runtime, memory cost, and
  network dependency in the composite system. No undeclared remote service or
  unavailable teacher is part of the deliverable.

This is risk control, not an OOD guarantee. Protected-distribution uncertainty
remains irreducible.

## Autonomous attack 25: target-render style can masquerade as restoration

Different paired datasets encode different normal-exposure rendering, white
balance, denoising, sharpening, and local-tone preferences. A model can improve
one target family by learning its style while becoming less physically correct.
Controls are:

- record camera/render provenance and keep dataset/sequence identities out of
  the deployable input unless they are real test-time metadata;
- report low-frequency photometric, structural/high-frequency, clipping, and
  perceptual deltas separately by family;
- use target-render latent variables only when inferred from input metadata or
  appearance; never choose a render style from the target dataset label at
  inference;
- include clean/already-good and exposure-bracket consistency tests so a target
  style cannot justify hallucination or needless restyling;
- prefer the robust conditional-mean solution across plausible renderings when
  render intent is not identifiable. A stochastic attractive rendering is not
  the PSNR-optimal answer.

This attack also limits what the official score proves: a FoundIR win is an
identical-protocol benchmark result, not evidence of universally preferred
mobile rendering.

## Autonomous attack 26: one number cannot support every SOTA claim

The published FoundIR quantitative table shows two rows per method whose
scales correspond to PSNR and SSIM, while the contractual Lowlight gate in this
project is PSNR and LPIPS. The released `cal_metrics.py` can calculate all
three, but the public table does not provide a comparable LPIPS leaderboard.
Therefore:

- `>21.90 dB` supports only an isolated-`14Lowlight` PSNR comparison against
  the audited table under identical preprocessing;
- `LPIPS <=0.25` is an engineering/contract-margin objective, not a verified
  LPIPS-SOTA threshold;
- also report SSIM with the table-compatible evaluator, but do not optimize it
  by sacrificing the frozen PSNR/LPIPS contract objective;
- claim joint PSNR/LPIPS SOTA only if reference outputs or author-reported
  identical-protocol LPIPS values become legitimately available without
  protected-test leakage. Otherwise use the precise claim `PSNR exceeds the
  audited published FoundIR table while satisfying the frozen LPIPS target`;
- compare systems only when category, image count, resize/crop, RGB space,
  saved-bit depth, and aggregation match exactly.

This wording prevents a strong PSNR result from being inflated into an
unsupported universal or multi-metric SOTA claim.

## Autonomous attack 27: an all-at-once contract freeze is circular

Exact full-scale sample budgets depend on the authorized data branch, measured
throughput, and winning pilot. Requiring those exact values before even a
no-training proposal diagnostic is impossible; leaving them unfrozen before a
large run is equally dangerous. Contracts are therefore stage-gated:

- before any model inference, freeze protocol, data/provenance, proposal-cache,
  evaluator, and experiment-ledger contracts;
- before each training stage, freeze its data folds, seed set, sample
  presentations, optimizer semantics, correction budget, promotion rule, and
  immutable output identity;
- before scaling, freeze the winning branch, scaling probe, exact full-run
  sample budget, distributed equivalence, and stop rule;
- before confirmation/protected inference, freeze the entire code/checkpoint/
  dependency graph, native-resolution path, metric container, runtime, and
  reporting language;
- a downstream contract may be drafted earlier but cannot be finalized using
  results from data it is supposed to protect.

No stage starts merely because an earlier contract exists. This resolves the
dependency loop without permitting results-driven rewrites.

## Autonomous attack 28: scientific safeguards can become useless complexity

A plan can fail without a bad model: it can spend months implementing elegant
controls that never affect the deployed decision. Every retained operation
must answer one of four necessary questions:

1. Are the metric, data, and benchmark claim valid?
2. What is the cheapest proposal that already works on real paired data?
3. Does Qwen add causal information, and does FLUX add genuine restoration
   value beyond that proposal?
4. Does the frozen primary system survive seed, full-resolution, and protected
   evaluation relative to its minimal predeclared controls?

If an operation cannot change a go/kill/design decision for one of those
questions, it is removed. The resulting feasibility rules are:

- use exactly one bounded third real family: sequence-subsampled SDSD after a
  short terms audit, or one disclosed fallback if SDSD cannot be used; do not
  acquire several alternatives in parallel;
- run full nested leave-one-family-out analysis only for cheap cached proposals
  and photometric ceilings. Expensive Qwen/FLUX candidates use bounded
  cross-family pilots on all frozen strata; only the single winner receives a
  full-scale run;
- evaluate only the already-present FoundIR generalist and documented cascade
  as mandatory teacher proposals. InstructIR is optional under a fixed small
  setup budget of one environment smoke and one cached inference pass, with no
  training or rescue work. AutoDIR remains a table reference and is not
  executed while its license is absent;
- use two predeclared LPIPS paths, not an open-ended version sweep: the pinned
  FoundIR/`pyiqa` path and direct `lpips==0.1.4` AlexNet cross-check. If Huawei
  supplies its evaluator, that evaluator becomes authoritative;
- Plan B starts immediately if official training authorization is absent when
  scaling begins. The project does not wait for the 4.85 TB archive;
- optional modules require a named measured failure, a cheap counterfactual or
  oracle showing at least the applicable architecture-promotion margin, and a
  fixed implementation/pilot budget. Otherwise they are killed before coding;
- the default optional-module count is zero. Attack 29 defines a bounded
  evidence-driven budget instead of an arbitrary one-module cap;
- only the one selected architecture receives three-seed replication. Pilot
  branches use one predeclared seed and are never rescued by seed search;
- CPU audits, cache hashing, and canaries are combined into the gate they
  protect; they are not separate ceremonial milestones.
- literature, license, and teacher-provenance audits are time-bounded to the
  available release documentation and local metadata. `Unverifiable` is a
  valid disclosed outcome; only missing legal permission for actual use is a
  blocker.

Concrete conditional admissions:

| Module | Implement only when |
|---|---|
| Learned photometric head | The identifiable tone/WB/illumination ceiling is strong and input-only prediction transfers in the cheap grouped diagnostic |
| Learned proposal fusion | An out-of-fold oracle convex blend beats the best single proposal by an architecture-level margin on every family |
| New camera simulator | Plan B has a measured coverage gap and a small simulator stream improves an omitted real family without identity loss |
| Frequency bypass | Bypass-off restoration, not merely the VAE round trip, shows a frequency bottleneck and a fixed bounded bypass has a positive ceiling |
| Spatial Qwen or camera cycle | The basic causal interface passes information tests but localization or forward inconsistency remains the attributed bottleneck |
| Quantization-aware training | Float-to-saved-uint8 conversion alone causes a material measured regression |
| Ensemble | Fixed constituent averaging improves every family enough to justify its runtime; no per-image selection is used |

The practical consequence is that the likely execution builds only a compact
pixel control, one causally conditioned FLUX bridge, and the native-resolution
wrapper. Teachers, photometric controls, and oracles are mostly cached
diagnostics. The larger architecture appears only if measured evidence makes
it necessary.

## Autonomous attack 29: aggressive pruning can discard complementary gains

The previous one-optional-module cap was operationally simple but not derived
from the image-formation problem. Two independent bottlenecks can coexist—for
example, global target rendering and high-frequency recovery—and two proven
corrections may be better than either alone. Conversely, restoring every
removed idea would recreate the unbounded loop. The final rule constrains
experiments and interactions rather than pretending only one useful idea can
exist:

- after Gate 4, rank optional work only from measured residual bottlenecks and
  cheap ceilings, not intuition;
- freeze a total budget of at most three one-seed bounded optional pilots before
  any optional result is observed. Each proposal gets one pilot and no
  correction; unused budget expires;
- retain at most two non-redundant optional changes in the architecture/data
  recipe. Each must pass alone, and the pair must pass one joint ablation
  against the mandatory winner. If their effects are redundant or antagonistic,
  keep only the stronger gain-per-runtime proposal;
- the optional pair cannot trigger another dataset acquisition, a larger model,
  a new loss search, or extra protected-test access;
- a fixed seed ensemble is evaluated only after the three mandatory finalist
  seeds already exist. It consumes no extra training branch, must pass the
  frozen runtime and every-family gate, and never selects per image;
- the exact pilot GPU-hour ceiling is frozen from Gate 4 throughput before
  optional launches. It cannot be enlarged because an attractive idea fails.

Priority follows current evidence, not novelty:

1. Input-predictable bounded photometric correction and teacher/proposal
   complementarity, because the real photometric oracle has large headroom and
   the FoundIR teachers are already available.
2. Camera-fitted simulation only if Plan B exposes a real coverage gap.
3. Frequency bypass only if the trained restorer, not merely the VAE, shows
   recoverable high-frequency loss.
4. Spatial Qwen or camera consistency only after the basic causal VLM path
   passes and localization/forward inconsistency is the remaining bottleneck.
5. QAT only for measured uint8 loss and an ensemble only for stable free gain
   from the already-required seed replicas.

Thus the removed ideas remain available where they can be valuable, but none
becomes work merely because it sounds sophisticated. This maximizes option
value under a finite experiment budget.

## Plan A and Plan B data details

**Plan A -- official training access:** acquire only author-authorized training
content. Begin with the 39,962 `14Lowlight` pairs, verify every file against
the official metadata, and build source/scene groups from GT identity,
exact/perceptual hashes, and provenance before splitting. Cache FoundIR
generalist/specialist proposals. Train the compact residual and FLUX bridge on
this distribution plus camera-diverse real replay. Add Lowlight-containing
compound categories only after the isolated gate and project scope permit it.
No official test image, public restored output, or test-derived statistic enters
the split, simulator, checkpoint selection, or target-render model.

**Plan B -- archive unavailable:** use LOL-v2, SICE, one sequence-grouped third
family after terms review, the existing controlled AgenticIR replay, and the
released FoundIR teachers. Do not make a new camera simulator a prerequisite.
Add one only if the mandatory-path error attribution demonstrates a coverage
gap and its bounded pilot improves omitted real data. Plan B can produce a
strong frozen candidate but cannot remove uncertainty about the official
training distribution; this limitation is carried into the final report.

## Frozen seven-gate execution sequence after plan approval

Each gate ends with a pass/kill decision and user review. No gate exists only
to produce paperwork.

1. **Protocol and development freeze:** pin the finite two-path LPIPS contract,
   independent PSNR reference, exact uint8 save/reload behavior, and canaries.
   Freeze LOL-v2, SICE, and one bounded sequence-grouped third family with
   overlap/alignment/provenance audits. Audit the FoundIR teacher cascade,
   freeze the experiment ledger, and date the `21.90 dB` literature target.
2. **Cache the cheap proposal/ceiling lattice:** evaluate raw input, current
   historical baseline, FoundIR generalist, documented cascade, fixed
   photometric controls, the four nested photometric oracles, and oracle blends.
   Select one provisional conservative base; the compact control in Gate 3 may
   replace it. Run only the cheap input-predictability diagnostic for a possible
   photometric head and record the fusion ceiling. Do not build a deployable
   photometric head or learned fusion before Gate 5's optional budget and the
   compact/FLUX proposals exist.
3. **Compact control and credible VLM gate:** train the compact deterministic
   pixel-residual control. Run the cheap same-scene Qwen exposure diagnostic
   and the four necessary causal variants on bounded paired-real pilots.
   Correct the basic Qwen/interface at most twice. Freeze Qwen only if
   measured-plus-Qwen adds paired-real value over measured-only and
   measured-plus-shuffled-Qwen; otherwise stop the advisor-directed
   route for review before any FLUX scaling.
4. **One bounded FLUX decision:** with the frozen VLM and identical pilot
   data/compute, compare pure-noise reference flow, deterministic LQ-to-GT
   latent flow, and degraded-anchored low-noise residual flow. Allow at most one
   endpoint/interface correction. Select one endpoint only if FLUX genuinely
   beats the better of the compact control and best base; otherwise stop instead
   of wrapping a weaker model around it.
5. **Bound optional work, then scale exactly one winner:** use Plan A only if
   authorization already exists; otherwise execute Plan B immediately. From
   Gate 4 residuals, freeze and run at most three optional pilots under attacks
   28--29; retain no more than two individually and jointly passing changes.
   Freeze sample presentations after one scaling probe, cache reusable latents/
   teacher outputs, and run one final full-scale architecture.
6. **Replicate and freeze deployment:** train two additional predeclared seeds
   for the one selected architecture, never selecting the best seed. Freeze the
   deployable seed or predeclared fixed ensemble, native-resolution global plus
   halo-tile path, exact dimensions/uint8 RGB output, seam/color/detail/runtime/
   memory gates, complete dependency hashes, and precise claim language. Run
   an ensemble of the already-required seeds only if its predeclared average
   passes every development family and runtime gate. Run one independent
   hidden/sentinel confirmation if available; it is not a blocker when no such
   set exists and its absence is disclosed.
7. **One protected evaluation event:** run the predeclared primary system and
   minimal frozen controls—the selected conservative base, compact pixel
   control, final FLUX with measured-only condition, and primary final FLUX with
   measured-plus-Qwen—on all 50 FoundIR-Lowlight pairs in one locked event.
   The primary is selected before this event; never choose among systems or
   correct from individual protected images afterward. Preserve and report all
   results whether they win or fail.

Gates 1 and the non-inference parts of Gate 2 may be prepared together. All
training gates are serial because each kills or selects the next branch.

## Compute and experiment budget

- Teacher outputs and VAE latents are immutable, hash-addressed caches so they
  are never recomputed for each branch.
- A new training implementation gets one 16-pair code/identity smoke. Branches
  sharing the verified code path reuse that evidence instead of repeating a
  ceremonial overfit. All candidate branches stop at bounded cross-family
  pilots; only one architecture gets a full-scale run.
- The mandatory Qwen/interface gate gets at most two genuine corrections and
  the FLUX endpoint gate at most one. Optional work is capped at three total
  one-seed pilots, one attempt each, with at most two jointly validated changes
  retained.
  Changing only a seed never counts as a correction and is not used to hunt a
  pass.
- Every launch is registered before execution in the experiment ledger. Cheap
  diagnostics use inner/outer folds; expensive pilots use the frozen
  source-disjoint cross-family strata and worst-family selection. The ledger
  records failed and killed proposals so the overall search multiplicity is
  visible.
- Only the selected architecture uses three fixed seeds: the primary full run
  plus two replications. No best-seed checkpoint is deployed.
- FLUX training must support two H100s and eight H200s with identical effective
  batch, sample count, optimizer-step semantics, and learning-rate policy.
- Mixed precision, activation checkpointing, latent/proposal caching, fused
  attention, and data prefetch are default engineering controls; their numeric
  equivalence is checked before a large run.
- The final runtime budget is frozen from proposal-lattice timing. An optional
  module whose gain is negligible relative to its latency is removed even if
  its mean metric moves slightly.
- Exact full-run steps are not guessed before Plan A/Plan B data and throughput
  exist. One short scaling probe freezes sample presentations and the full-run
  budget; there is no open-ended sequence of budget doublings.

## Error budget and quantitative gates

The following are non-additive safeguards rather than a fictional decomposition
of PSNR decibels:

| Risk | Existing evidence | Gate before protected test |
|---|---|---|
| Metric mismatch | Official script/environment found; `pyiqa` remains unpinned | Independent PSNR agreement and LPIPS compliance under both frozen paths, or the exact authoritative Huawei evaluator |
| Representation | VAE `38.4488 dB`, LPIPS `0.005159` | Target round trip retains large headroom; bypass is deployed only if it beats bypass-off on every development family |
| Global rendering | Full-matrix oracle gain `+15.6853 dB`, but parameters are non-identifiable | Nested exposure/tone-WB/illumination ceilings isolate an identifiable target; input-only learned head must transfer across omitted families and beat fixed-transform control |
| Spatial restoration | Not yet measured on paired real suite | Winning bridge must beat the best conservative proposal under the constrained PSNR/LPIPS gate on every development family |
| Teacher value | FoundIR checkpoints hash-audited; AutoDIR/InstructIR table evidence audited; training overlap not yet resolved | Provenance status disclosed; inference anchoring or distillation only if proposal/complementarity transfers on a provenance-independent outer family |
| VLM value | Previous narrow condition near-null | Frozen ordinal thresholds pass (`0.75` concordance, `0.80` non-inversion, `0.60` median stack Spearman, `0.80` positive-stack fraction, `1.0` span); oracle-plus-measured beats measured-only by at least `0.50 dB`; measured-plus-Qwen beats measured-only and measured-plus-shuffled-Qwen by at least `0.20 dB` per family with no LPIPS regression and recovers at least half of the incremental oracle gain |
| Optimization variance | No finalist replication yet | At least three predeclared seeds; promotion survives combined scene/seed uncertainty and worst seed/family is inside non-regression margins |
| Full resolution and output | VAE seam ratio `1.027740` only | Matched tiled inference loses no more than `0.10 dB` PSNR and `0.002` LPIPS versus feasible whole-image reference, passes frozen seam/color gates, and exact saved uint8 RGB outputs preserve the development gain |
| Domain robustness | Only LOL-v2 and SICE paired suite today | Freeze at least three independent families; all retained components improve the mean and do not materially regress the worst omitted family |

The constrained selection objective is fit only in grouped inner folds and
maximizes worst-family scene-balanced PSNR subject to LPIPS and identity
non-inferiority. Outer omitted-family results promote or kill the frozen
proposal; they do not select its settings. A material exploratory
PSNR gain is at least `+0.20 dB` on each omitted dataset with LPIPS change at
most `+0.005`. A perceptual-only promotion requires LPIPS at least `-0.010`
with PSNR change no worse than `-0.05 dB`. Larger architecture decisions
require at least `+0.50 dB` or that perceptual-only rule. Deployment retention
also requires a positive 95% scene-bootstrap interval after the suite has at
least three independent families, no family outside the equivalence margins,
and no material L1/already-good identity regression. For finalists, the
interval and non-regression decision also include training-seed variation.
Metric canaries establish
implementation determinism; the fixed practical margins above are not derived
from deterministic canary variance and are never changed after results appear.

## Proposal ladder and stop rules

1. `P0`: degraded input and fixed training-set photometric controls.
2. `P1`: released FoundIR generalist and documented
   generalist-to-specialist cascade. Other licensed references are conditional.
3. `P2`: conditional bounded learned photometric/illumination head.
4. `P3`: compact deterministic pixel-residual control.
5. `P4`: causally validated and frozen Qwen condition on `P3`.
6. `P5`: FLUX residual refinement under the three endpoint constructions,
   zero-initialized around the best conservative proposal.
7. `P6`: zero by default, or at most two non-redundant changes admitted and
   jointly validated under attacks 28--29.

Stop rules:

- If the identifiable nested photometric ceiling is weak, do not train a
  flexible photometric predictor merely because the full matrix oracle is
  strong. If `P2` has a strong identifiable ceiling but its cheap input-only
  predictor does not transfer, kill `P2` for this execution; do not acquire a
  new dataset or add DiT capacity merely to rescue it.
- If neither released teacher route beats the photometric control nor
  complements it in an oracle blend, skip teacher-at-inference. Distillation
  still requires its own omitted-dataset gain and is skipped if that also fails.
- If `P3` cannot beat `P0/P2` or complement `P1`, inspect
  alignment/data/objective before enabling FLUX residual capacity. It need not
  independently beat a strong licensed teacher to serve as the compact control.
- If `P4` misses the ordinal or measured-plus-Qwen causal gate after two
  redesigns, block the
  advisor-directed deliverable and protected test; do not hide the failure
  behind FLUX capacity or a new severity set.
- If `P5` cannot beat the best conservative `P1--P4` proposal, perform at most
  one endpoint/interface correction. A non-improving FLUX wrapper is not a
  credible final system.
- If frequency bypass, camera consistency, stronger VLM injection, or an
  ensemble misses its isolated gate, remove that optional extension.
- If official training access remains unavailable, execute Plan B; do not wait
  indefinitely and do not compensate by tuning on FoundIR test.
- If native-resolution behavior fails, fix it before any benchmark inference.
- Once code, checkpoints, evaluator, primary, controls, and inference are
  frozen, run every predeclared system on all 50 FoundIR-Lowlight pairs in one
  locked event. Any later use is disclosed as benchmark iteration and cannot
  support a fresh unbiased claim.

If the protected run misses the target, preserve and report it. Do not tune on
its individual images. Continued work returns to the reusable development
suites and newly authorized non-benchmark data; any later FoundIR run is
explicitly a benchmark iteration, and a fresh unbiased claim requires a new
hidden advisor set.

## First model-inference milestone: proposal-lattice diagnostic

No part of this experiment starts until its contracts are frozen. The first
post-plan milestone measures the existing proposal lattice on
reusable non-FoundIR pairs before training the full architecture. It compares:

- the raw input;
- the released FoundIR generalist;
- the documented generalist-to-Lowlight-specialist cascade;
- InstructIR only if its licensed checkpoint is runnable inside the frozen
  small setup budget; AutoDIR is not executed without license clearance;
- the current synthetic-trained FLUX baseline as historical evidence;
- a transform whose parameters are the training-scene mean;
- bounded input-only predictors for luminance quantiles, monotone tone knots,
  positive white balance, and smooth illumination, with nested scene-grouped
  selection;
- exposure-only, monotone-tone/WB, spatial-illumination, and unconstrained
  full-matrix per-image GT-fitted oracles, only as ordered upper bounds.

These cheap diagnostics use nested leave-one-dataset-out transfer and grouped
within-training-family inner folds. Parameter/summary error is supporting
evidence, but the decision is made on exact saved-uint8 restored-image PSNR,
the worse of the two frozen LPIPS paths, and supporting SSIM because oracle
matrix parameters are non-identifiable. The predictor sees only the degraded
input at inference.

It also measures oracle convex and frequency-selective blends among proposals.
Those target-accessing blends are ceilings only. They decide whether the
bottleneck is proposal quality or proposal selection/fusion and whether a
shallow global head is useful or the first trained component must be spatial.
The diagnostic does not open an official FoundIR test image, authorize the full
bridge, or claim deployable quality.

## Pre-experiment and stage-contract checklist

Before the first proposal inference, the protocol, development/provenance,
proposal-lattice, evaluator, and experiment-ledger items below must be explicit,
machine-readable, hash-frozen, and internally consistent. The remaining items
are mandatory before their corresponding training, scaling, or final-inference
stage; none may be retrofitted after seeing that stage's protected result:

- verify `21.90 dB` as the current table target, audit later identical-protocol
  results, and freeze the competitive target separately from the contractual
  floor;
- freeze the two-path LPIPS reproduction, exact uint8 save/reload semantics,
  independent PSNR cross-checks, worse-of-two LPIPS rule, and
  canaries; record whether the exact Huawei evaluator was obtained;
- verify teacher licenses, documented cascade semantics, training provenance,
  overlap status, and inference cost;
- freeze the three-family development acquisition/split policy without using
  a new disposable test set; predeclare nested folds and reserve a
  provenance-independent sentinel/hidden confirmation when available;
- freeze the experiment ledger, total compared-proposal disclosure, finalist
  seed set, and combined scene/seed uncertainty method;
- freeze proposal-lattice inputs, hashes, metrics, blend ceilings, and gates;
- freeze Plan A and Plan B budget policies and Gate 3/Gate 4 correction caps;
  freeze exact sample presentations for a branch before that branch trains and
  freeze the exact full-run budget only after the declared scaling probe;
- after Gate 4 and before any optional pilot, freeze the three-pilot/GPU-hour
  budget, candidate ranking, individual gates, two-change retention cap, and
  joint-ablation rule;
- freeze the FLUX endpoint comparison and the compact pixel-residual control;
- freeze the VLM causal matrix and its rule for removal/redesign;
- freeze the measured-statistics/CNN versus Qwen information split;
- freeze native-resolution whole-versus-tiled, exact-dimension, quantization,
  seam/color, runtime, memory, and identity safeguards;
- if a simulator is admitted under attack 28, freeze its content-matched
  validation, identifiable-parameter wording, and omitted-real downstream gate
  before implementation;
- if learned fusion/OOD fallback is admitted, freeze its input-only calibration
  before implementation; always freeze composite-system dependency/license
  disclosure before deployment;
- review the final dependency graph so no optional module silently becomes a
  prerequisite for diagnosing an earlier one.

## Irreducible uncertainties after freeze

No further planning round can remove these without an external state change:

- author authorization for the official 4.85 TB training archive;
- license/author clearance for AutoDIR and redistribution terms for SDSD;
- exact Huawei/contract evaluator availability and the correct unpinned
  `pyiqa` behavior if it is not supplied;
- incomplete training provenance for public/base checkpoints and any
  unverifiable overlap with development families;
- whether a newer method has published a stronger identical-protocol result by
  final-evaluation time;
- the exact target-render/camera distribution of the protected 50 pairs;
- availability of a genuinely hidden Huawei/advisor confirmation set.

Each uncertainty has a branch or disclosure above. None is handled by opening
the FoundIR test early, inventing another five-level label set, or adding an
unablated module.

## Evidence identities used to freeze this plan

- FoundIR quantitative table PNG SHA-256:
  `fc0e88b363b80d30271569ad8f1a1c2e96cefed068113163af909c95127441e8`.
- FoundIR environment SHA-256:
  `fd4ba0d46d8d2bc01848dcf836d245c43e09bbe80299f156c675a0132c018c63`.
- FoundIR README SHA-256:
  `324070e32e144e784d00840b9af57ed926c628c5e06e0849774ccdd46d8107bb`.
- FoundIR residual-diffusion implementation SHA-256:
  `6f53c2ae7ad8f9881a3a2c9418878a07b10c2dadf2b197ef231e5d69455cbad1`.
- FoundIR Lowlight-specialist inference SHA-256:
  `7e54ec014690855c3d45e40f4ea338af0c3ca2ff88675587155b1bf6979558ef`.
- Existing paired-real photometric ceiling and VAE ceiling identities are
  recorded in their immutable result reports.

The table establishes AutoDIR `21.90 dB`, InstructIR `20.04 dB`, and FoundIR
`18.98 dB` for isolated Lowlight. A 2026 literature search also identified
one-step degradation-aware diffusion with high-fidelity decoder work, but no
stronger identical-`14Lowlight` result was verified; the mandatory dated audit
is repeated before the protected run.
