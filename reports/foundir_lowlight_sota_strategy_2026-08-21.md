# FoundIR-Lowlight first-principles SOTA strategy

Date: 2026-08-21 UTC

## Decision

The current pipeline is a verified proof of training and evaluation plumbing,
not yet a credible SOTA attempt. Development is therefore re-centered on the
actual endpoint: paired, full-resolution restoration PSNR and LPIPS on unseen
real Lowlight images. Project L1--L5 classification remains useful for
controlled synthetic experiments, but it is not an official FoundIR target and
will not block restoration based on invented absolute labels for unrelated
real photographs.

No project model has been run on the official FoundIR-Lowlight pairs. The 50
official pairs remain protected.

## Evidence forcing the strategy change

- Current restoration training used 8,000 simple synthetic low-light crops,
  512 by 512 pixels, 4,000 optimizer steps, and approximately 8.11 million
  rank-8 LoRA parameters.
- Its internal synthetic test result is strong but distribution-specific:
  PSNR 24.7723 dB and LPIPS 0.0440. It says nothing quantitative about
  FoundIR generalization.
- The earlier frozen-Qwen ridge plus one-token condition changed internal test
  PSNR by only +0.03683 dB. It disproves that narrow interface, not the
  advisor's full supervised-VLM route.
- Real Qwen adaptation improved its proxy-label validation objective but
  forgot synthetic structured behavior. Every saved checkpoint had the same
  trade-off. Earlier checkpoint selection cannot rescue it.
- FoundIR's released metadata lists 472,046 single-degradation pairs and
  1,010,064 full-training pairs. It contains 39,962 isolated-Lowlight pairs
  and 378,440 pairs in categories containing Lowlight.
- Official FoundIR trains for 500,000 single-degradation steps and then up to
  2,000,000 all-degradation steps. Its training archive is approximately
  4.85 TB.
- Official FoundIR-Lowlight test images are 2720x1528, 3008x1688, or
  3680x2456. A 512-crop-only development loop does not validate the deployed
  global tone, stitching, or detail-preservation behavior.

The dominant gap is therefore data distribution, fidelity architecture,
objective alignment, and high-resolution deployment. It is not a shortage of
arbitrary real-image five-level labels.

## First-principles objective

Let a low-light observation be

```text
x = camera_pipeline(exposure_field * scene_radiance, sensor_parameters) + noise
```

and let `y` be the paired normal-light target. The system must approximate a
conservative estimate of `p(y | x)` while preserving every recoverable fact in
`x`. Mean squared error, hence PSNR, favors the conditional mean. LPIPS favors
perceptually plausible local structure. The system must therefore learn a
controlled Pareto compromise rather than act as an unconstrained image
generator.

Every experiment below must answer one of these questions:

1. Can the representation reproduce the required target fidelity at all?
2. Can the restoration architecture use degradation information causally?
3. Does the training distribution cover independent real cameras and scenes?
4. Does the training objective improve the exact downstream metrics?
5. Does the full-resolution implementation preserve the crop-level gain?

An experiment that answers none of these questions is not run.

## Additional improvements from the second first-principles pass

### Degraded-to-clean transport, not generation from pure noise

The current FLUX path starts from random noise and treats the degraded image
only as a reference. That unnecessarily asks a restoration model to regenerate
the whole scene. A paired restoration bridge will instead start from the
degraded latent and learn a deterministic or low-stochasticity transport to the
clean latent. Training will compare:

```text
current path: random noise -> clean, degraded image is a reference
bridge path:  degraded latent -> clean latent, degraded image remains observable
```

The bridge uses direct clean-latent and decoded-RGB supervision in addition to
flow loss. It is expected to improve identity and PSNR and to reduce sampling
variance.

### Explicit global photometric factorization

Global exposure, spatial illumination, white balance, color correction, and
tone response are lower-dimensional than texture restoration and dominate
many pixel errors. A bounded, interpretable photometric head will predict an
exposure field, white-balance/color transform, and monotone tone curve. FLUX
then predicts only the residual that the physical/global head cannot explain.
This reduces the burden on generative attention and makes over-correction
measurable.

### High-frequency fidelity bypass

A latent VAE can discard sensor detail before the DiT sees it. The system will
measure the clean-image VAE round-trip ceiling first. If it is limiting, a
signal-to-noise-gated Laplacian/pixel skip will carry recoverable high-frequency
detail around the VAE, while the learned path denoises only low-confidence
bands. The final output remains the FLUX/Z-Image restoration system; this is a
fidelity path inside its restoration adapter, not a conventional model
substitution.

### Learned forward-camera consistency

The model will also estimate camera/degradation parameters and reapply a
differentiable forward model to its restoration. A candidate that cannot
reproduce the observed low-light input under any plausible exposure, noise,
black-level, white-balance, clipping, ISP, and compression setting is rejected
or penalized. This gives an input-only anti-hallucination signal during
training and optional test-time candidate selection without using test GT.

### Conditional mean and conservative ensembles

For PSNR, averaging valid samples approximates the conditional mean. Fixed D4
geometric self-ensemble, low-variance multi-seed averaging, checkpoint/LoRA
soups, and a fixed perceptual-versus-fidelity blend may be evaluated on real
development data. They are retained only if both PSNR and LPIPS improve across
held-out datasets. No test-set-driven candidate selection is allowed.

## Additional improvements from the third first-principles pass

This pass asks a stricter question: how can each new component be made to earn
its place while preserving a strong fallback? The answer is to develop the
system as an information-preserving proposal ladder rather than a monolithic
generator.

### Proposal-lattice ceiling and non-regressing residuals

On paired development data, produce aligned candidates from the raw input, the
oracle/learned photometric transform, the released FoundIR teachers, and the
FLUX bridge. Measure both the best single candidate and an oracle convex blend.
This separates three failure modes:

- all proposals are weak, so better data or restoration capacity is needed;
- a good proposal exists but selection is weak, so uncertainty/gating is the
  bottleneck;
- a convex blend is much better than every proposal, so frequency- or
  region-selective fusion has measurable value.

The deployable FLUX system will be residual and zero-initialized around its
photometric/input fidelity path. Its learned blend masks are bounded and
regularized toward the conservative proposal, so a new branch starts as an
identity rather than destroying information. Released FoundIR models remain
development baselines and training-only teachers; the final deployed output
remains FLUX/Z-Image-based.

### Teacher consensus as uncertainty, not blind imitation

The public FoundIR generalist and Lowlight specialist encode useful training
distribution knowledge, but either can be wrong. Generalist, specialist, and
paired-GT agreement will define confidence-weighted distillation targets on
allowed training data. Agreement receives stronger distillation; disagreement
is treated as uncertainty and is resolved by paired supervision rather than
averaged blindly. Teacher disagreement also supplies hard examples for the
residual bridge.

### Learn the target rendering distribution

Low-light restoration is not only inverse exposure estimation. Paired targets
also define a dataset-specific choice of brightness, white balance, contrast,
and tone. These low-frequency target statistics will be learned from allowed
training pairs, including official FoundIR training pairs once authorized.
The photometric head will predict a distribution over plausible renders and
the fidelity branch will choose its conditional mean for PSNR, while retaining
a bounded perceptual residual for LPIPS. No statistic is fitted on FoundIR
test targets.

### Counterfactual conditioning that cannot be ignored

For the same clean scene rendered at multiple known degradations, swap the
oracle condition while holding pixels fixed and require predictable monotone
changes in exposure/noise correction. Add condition dropout and shuffled
condition negatives. This makes the conditioning interface answerable by
intervention: correct metadata must reduce paired loss, and wrong metadata
must not produce the same result. Qwen is upgraded only after this oracle
counterfactual test passes.

### Robust-risk selection instead of average-score chasing

Every checkpoint is scored by dataset, camera/scene cluster, darkness/noise
regime, and resolution. Selection minimizes a predeclared combination of mean
risk and worst-dataset risk with scene-bootstrap uncertainty. A change that
raises the mean by exploiting one dataset while materially harming another is
rejected. Alignment/occlusion confidence controls pixel losses so the model is
not trained to reproduce registration errors.

### Input-only adaptation stays low dimensional

If test-time adaptation is retained, it may update only bounded exposure,
white-balance, monotone-tone, and illumination-field controls using the frozen
camera-forward consistency objective. It may not update the generative model
or use benchmark targets. The complete rule, iteration count, and rejection
threshold must first improve every held-out development dataset and then be
frozen.

## Executable strategy

### Gate A: representation and simple-restoration ceilings

On non-FoundIR paired real development images, measure:

- clean RGB -> frozen VAE -> RGB round-trip PSNR/LPIPS;
- exact identity/reference pass-through;
- best per-image oracle exposure/gamma/color transform;
- learned global photometric transform;
- current FLUX restoration at 512 and through the proposed full-resolution
  tiler.

Decision rules:

- If the VAE ceiling is too close to the target model result, implement the
  high-frequency bypass before training more LoRA.
- If an oracle global transform explains most of the attainable gain, build
  the explicit photometric head before increasing DiT capacity.
- If native-resolution stitching loses the crop-level gain, fix global/local
  conditioning and tiling before any benchmark run.

### Gate B: causal degradation-conditioning matrix

Train or evaluate the same restoration backbone under four conditions:

1. no degradation condition;
2. oracle continuous condition from known simulator/pair metadata;
3. predicted VLM condition;
4. condition shuffled between images.

A simple measured-statistics condition is included as a fifth reference.

Decision rules:

- Oracle approximately equal to baseline means the injection path is ignored,
  redundant, or too weak. Redesign conditioning; do not retrain Qwen.
- Oracle better, predicted approximately baseline means VLM estimation is the
  bottleneck.
- Predicted approximately shuffled means the restoration model is not using
  the VLM causally.
- Predicted better than baseline and shuffled, with paired confidence intervals
  on dataset-held-out real images, is the only evidence that authorizes a VLM
  restoration-gain claim.

### Gate C: objective real-domain VLM development

Real severity development uses same-scene exposure ordering, not absolute
labels on unrelated scenes. For each stack, darker exposure must imply
non-decreasing predicted severity. Soft expected severity is derived from the
five canonical target NLLs to expose discrete collapse and ties.

Primary metrics are pairwise concordance for pairs separated by at least one
EV, hard-level non-inversion, per-stack Spearman correlation, adjacent
inversions, prediction span, and confidence. Passing proves only relative
exposure sensitivity. Decisive evidence remains paired downstream restoration
gain from Gate B.

If Qwen needs corrective adaptation, restart from synthetic step 600 with
limited adapters or partial unfreezing, a lower learning rate, stronger
synthetic replay, feature/logit distillation to prevent forgetting, continuous
or ordinal real losses, and multi-objective checkpoint selection. Full visual
path adaptation is not repeated unchanged.

The VLM condition is upgraded from one hard token to a soft program containing:

- degradation-type posterior;
- continuous exposure deficit and uncertainty;
- signal-dependent/read-noise estimates;
- black-level and clipping indicators;
- white-balance/color-shift estimates;
- spatial illumination and confidence maps;
- semantic preservation masks or tokens.

The program is injected through multiple tokens plus FiLM/AdaLN or a spatial
control branch in multiple DiT blocks. A soft mixture of restoration adapters
for illumination-, noise-, color-, and compression-dominant regimes is a
candidate only after the basic oracle test passes.

### Gate D: benchmark-relevant training data

Highest priority is legitimate access to the official FoundIR training set and
released checkpoints, never the official test targets.

Training begins with the 39,962 isolated-Lowlight pairs. Lowlight-containing
mixtures are considered only after the current Lowlight-only gate and scope
permit them. The official generalist and lowlight specialist are used as
baselines and training-only teachers; the final deployed restoration output
remains FLUX/Z-Image-based.

The remaining data streams are:

- genuinely aligned real low/normal exposure pairs, partitioned by scene,
  camera, and dataset;
- physically realistic synthetic pairs generated in linear/RAW-like space;
- the existing controlled AgenticIR hierarchy for clean causal labels and
  replay, not as a model of all real low light.

The realistic forward simulator includes exposure, spatial illumination and
vignetting, Poisson shot noise, read/dark noise, black-level error, Bayer/ISP
effects, white balance and color response, nonlinear tone mapping, highlight
clipping, quantization, and JPEG. Parameter distributions are fitted only from
allowed training data. Clean source identities are screened against every
protected set.

Real pairs are aligned with explicit motion/occlusion masks. Misaligned pixels
do not receive strict pixel loss.

### Gate E: fidelity-first model and objective

The model uses a strong spatial degraded-image encoder/control path rather than
relying on one context token. Capacity is increased only after the ceiling and
oracle diagnostics identify under-capacity.

Training is staged:

1. global photometric reconstruction;
2. deterministic degraded-to-clean latent bridge;
3. pixel/latent residual refinement and high-frequency bypass;
4. real-pair and official-training fine-tuning;
5. optional VLM/expert routing after causal authorization.

Losses are aligned to the endpoint: RGB L2/Charbonnier, multi-scale SSIM,
LPIPS, edge/frequency preservation, color/tone error, identity loss, and
forward-camera consistency. Checkpoint selection uses a Pareto rule over PSNR,
LPIPS, identity, and worst-dataset performance rather than a single training
loss.

### Gate F: real paired development without endless test consumption

Build multiple paired real datasets with scene/camera/source separation. Use
leave-one-dataset-out development: train on all but one dataset, evaluate on
the omitted dataset, rotate, and select changes that improve mean and
worst-dataset results with paired bootstrap confidence intervals.

These are declared reusable development sets. They are not repeatedly renamed
"fresh tests." A separate hidden Huawei/advisor holdout remains the source of
unbiased real-world confirmation.

### Gate G: full-resolution frozen inference

Use a low-resolution global pass for tone, color, and semantic context plus
overlapping 1024-or-larger local tiles with halos, coordinate consistency, and
windowed blending. Preserve exact original dimensions. Validate seam energy,
global color consistency, and detail retention on non-FoundIR 2K/4K pairs.

Only development-proven, GT-free options may be enabled at inference:

- deterministic D4 self-ensemble;
- conditional-mean multi-seed averaging;
- fixed fidelity/perceptual blending;
- low-dimensional input-only control refinement selected by forward-camera
  consistency.

### Gate H: one protected FoundIR run

Freeze code, checkpoint hashes, evaluator, seeds, tiling, ensemble, and all
hyperparameters. Run all 50 official `14Lowlight` pairs together once. Never
iterate over small public-test sections.

Report RGB PSNR, LPIPS, SSIM, per-image rows, bootstrap uncertainty, contact
sheets, exact environment, and hashes. The contractual Stage 2 reference is
PSNR at least 18.9 dB and LPIPS at most 0.28; the competitive objective is to
beat the best published result under the identical protocol by more than
measurement uncertainty, not merely cross the contract floor.

After that run, any future public-test use is transparently a benchmark
iteration. A truly unbiased final claim then requires the hidden Huawei or
advisor set.

## Current execution state

Gate C has begun without model inference. The screened Retinex-Net pool was
reinterpreted correctly:

- singleton unrelated scenes remain qualitative robustness images only;
- one mixed/fused component is excluded;
- 15 author-provided MEF exposure stacks are retained;
- 67 exact-unique images provide 210 total same-scene comparisons, including
  151 comparisons separated by at least one EV;
- ordering is frozen from within-stack mean luma before predictions;
- these data have no absolute project L1--L5 labels and require no human
  severity annotation.

The next executable actions are to validate the frozen ordinal manifest, run
existing frozen Qwen checkpoints through canonical soft scoring on two H100s,
and implement Gate A's VAE/global-photometric ceiling diagnostic on paired
non-FoundIR real development data.

### Execution update after the third pass

Gate A is now active. A reusable paired-real suite contains 174 records across
82 scene components: 102 LOL-v2 real-adaptation validation records and 72 SICE
reusable validation records. It contains no consumed real-test record and has
zero exact FoundIR overlap.

The first completed ceiling is decisive. The strict scene-balanced input score
is 8.4356 dB PSNR and 0.2475 SSIM. A target-fitted global gamma plus color
matrix reaches 24.1209 dB and 0.6571, a paired gain of 15.6853 dB with a 95%
scene-bootstrap interval of 14.7717 to 16.6036 dB. This is an oracle diagnostic,
not an inference method, but it moves the learned photometric head ahead of
additional DiT capacity in the implementation order.

The two-H100 VAE ceiling is CPU-preflight ready. It measures 130 exact-unique
targets with a direct center crop and full-resolution overlapping tiles,
including PSNR, SSIM, resized LPIPS, high-frequency retention, and seam error.
Its predeclared decision either clears latent-only restoration or requires the
pixel-frequency bypass per dataset. The public FoundIR generalist and Lowlight
specialist checkpoint identities are also frozen; teacher inference has not
yet run.

## External dependency

The official FoundIR training archive requires the request form linked in the
official repository. Storage is not the blocker: the workspace filesystem had
approximately 900 TB free at audit time. Access authorization and transfer are
the external dependency. The released generalist checkpoint is public and can
be acquired independently; the lowlight specialist checkpoint is already
present locally and hash-audited.
