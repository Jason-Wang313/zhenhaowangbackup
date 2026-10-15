# Frozen-proposal energy-TTS novelty audit

Date: 2026-10-14 UTC
Scope: repository and targeted prior-art audit; no training or evaluation run

## Bottom line

The broad claim **"the first VLM-guided All-in-One image restoration system
with gradient-based iterative inference" is not defensible**. VLU-Net (CVPR
2025) already combines a VLM-derived degradation representation with an
eight-stage proximal-gradient deep unfolding network. Likewise, iterative
test-time refinement, restoration test-time scaling, energy-based imaging, and
proposal refinement each have direct prior art.

The narrower mechanism below was not found in the targeted audit and remains a
credible working novelty hypothesis:

> Given an immutable output from a heterogeneous FLUX.2/Restormer AiOIR
> system, learn a candidate-dependent scalar compatibility energy conditioned
> jointly on the original degraded observation and a structured VLM
> degradation representation; use the gradient of this same intrinsic energy
> to refine the candidate at test time and use its value/trajectory for
> stopping or verification.

This is a **working "to our knowledge" claim**, not proof of world-first
priority. The contribution is the complete restoration-specific formulation,
not any one ingredient: EBM, VLM conditioning, iterative inference, TTS, and
proposal refinement all predate it.

## Exact proposed formulation

Let `x` be the degraded observation, `y0` the frozen routed
FLUX.2/Restormer proposal, and `c` the structured VLM degradation condition.
The new trainable component is a scalar energy

\[
E_\theta(y; x,c),
\]

and the candidate is updated without changing the frozen proposal network:

\[
y_{t+1}=y_t-\alpha_t\nabla_yE_\theta(y_t;x,c).
\]

The same learned energy must support three empirically verified roles:

1. a useful direction for candidate refinement;
2. a calibrated ordering of restoration candidates;
3. stopping or compute allocation without clean ground truth at inference.

If the implementation only learns a direct residual, repeatedly applies a
feed-forward refiner, or uses a separate IQA/reward model, it does not establish
this proposed distinction.

## Closest work and claim boundary

| Work | VLM-conditioned AiOIR | Iterative/TTS inference | Candidate-dependent scalar learned energy | Direct candidate-gradient update | Same intrinsic signal verifies/stops | Boundary |
|---|---|---|---|---|---|---|
| VLU-Net, CVPR 2025 | Yes | Fixed eight non-shared unfolding stages | No | PGD-inspired learned D-GDM, but not `grad_y E(y;x,c)` | No | Blocks broad "VLM + gradient/unfolding AiOIR" claim |
| SIPL, 2025 | AiOIR, no VLM | Yes; output is reused as pseudo-privileged information | No | No | No | Blocks "first iterative test-time self-refinement for AiOIR" claim |
| KDS, 2025 | No | Yes; multi-particle inference-time scaling | KDE log-density, not a trained observation/VLM-conditioned energy | Yes, density steering | No learned intrinsic verifier | Blocks "first verifier-free gradient TTS for restoration" claim |
| ResFlow-Tuner, 2026 | Multimodal FLUX Real-IR | Yes | No; external reward-model feedback | Steering inside flow inference | External RM/rank aggregation | Blocks "first restoration TTS" and FLUX-TTS claims |
| TTPO, AAAI 2026 | Backbone-agnostic IR, no VLM requirement | Yes | No; selected preference images and NR-IQA/human feedback | Gradient guidance in diffusion optimization | External metrics/human preferences | Blocks broad proposal-refinement/test-time optimization claim |
| TAO, ICML 2024 | Open-set IR, no VLM | Test-time parameter/degradation adaptation | No | Updates adapter/model behavior, not candidate under intrinsic energy | No | Distinguishes parameter adaptation from prediction-space optimization |
| EBT, 2025 | No AiOIR/VLM | Yes | Yes | Yes | Yes | Supplies the core EBT principle; image evidence is 128x128 Gaussian/diffusion denoising only |
| ALPS, 2026 | No | Iterative posterior sampling/MAP | Multiscale EBM prior | Langevin/posterior optimization | Energy is intrinsic, but not VLM/AiOIR proposal verification | Blocks "first energy-based image restoration" claim |
| DAME-Net/MDUR, 2026 | Compositional AiOIR, no VLM requirement | Fixed feed-forward MoE | No | No | No | Blocks lower-order/higher-order compositional restoration as novelty alone |
| Proposed project mechanism | Yes | Fixed steps plus learned/frozen adaptive stopping test | Yes: `E(y;x,c)` | Yes | Yes | Exact intersection not found in this targeted audit |

## Full-text findings that materially change the claim

### VLU-Net: the most dangerous wording overlap

Zeng et al., *Vision-Language Gradient Descent-driven All-in-One Deep
Unfolding Networks*, CVPR 2025, DOI `10.1109/CVPR52734.2025.00705`, arXiv
`2503.16930`:

- fine-tunes CLIP on degradation image/text pairs;
- uses the VLM degradation vector in a degradation-guided gradient-descent
  module;
- unfolds proximal gradient descent into eight non-shared stages;
- evaluates multiple restoration tasks and unseen random noise.

The verified boundary is that VLU-Net uses a fixed trained unfolding network.
The inspected paper does not define a candidate-dependent scalar learned
energy `E(y;x,c)`, differentiate that scalar with respect to the candidate at
inference, expose a variable compute-quality curve, or use the scalar for
intrinsic candidate verification/stopping.

Therefore say **candidate-energy optimization**, not merely "VLM-guided
gradient descent".

### SIPL: the most dangerous iterative-AiOIR overlap

Wu et al., *Boosting All-in-One Image Restoration via Self-Improved Privilege
Learning*, arXiv `2505.24207`:

- feeds preliminary restoration outputs back as pseudo-privileged information;
- performs optional multiple test-time refinement iterations;
- reports `+4.58 dB` over its PromptIR baseline on composite degradation;
- reports a large OOD composite improvement from additional iterations.

It does not train a scalar candidate energy or update the candidate with
`grad_y E`. Nevertheless, a paper must compare against a matched recurrent or
SIPL-style self-refinement baseline. Otherwise reviewers can reasonably argue
that the gain comes from iterative feedback rather than the energy
formulation.

### KDS, ResFlow-Tuner, and TTPO: restoration TTS is already occupied

- KDS, arXiv `2507.05604`, steers multiple diffusion particles using gradients
  of a patch-wise kernel-density estimate. It needs neither retraining nor an
  external verifier.
- ResFlow-Tuner, arXiv `2603.22027`, applies reward-model feedback to steer a
  FLUX.1-dev restoration trajectory at inference.
- TTPO, arXiv `2511.19169`, begins with an arbitrary restoration proposal,
  creates preference candidates, selects them with no-reference IQA or human
  feedback, and performs gradient-guided diffusion optimization.

Consequently neither "test-time scaling for restoration", "FLUX refinement",
"proposal refinement", "gradient-guided restoration", nor "no external
verifier" is independently novel.

## Novelty strength if the method works

### Weak evidence

- one Noise-only gain;
- fixed one-step refinement only;
- comparison only against the frozen proposal;
- energy decreases but PSNR/LPIPS do not consistently improve;
- no matched feed-forward or recurrent-refinement baseline.

This would be an EBT application/engineering result, not a strong paper
contribution.

### Credible method evidence

- at least `+.50--.75 dB` PSNR and `-.010` LPIPS on the frozen Noise
  development protocol;
- 2/3 candidate-gradient steps improve over one step at matched checkpoint;
- a matched-parameter direct bounded refiner and a recurrent/SIPL-style
  refiner cannot match the gain;
- energy change and candidate ranking correlate with PSNR/LPIPS improvement;
- removing the VLM condition or original degraded observation materially
  harms performance.

This supports a real mechanism claim, but remains singleton evidence.

### Strong paper evidence

- the same frozen energy mechanism improves several singleton categories and
  Double compositions without per-task retraining;
- adaptive stopping beats fixed-step inference at matched average FLOPs or
  latency;
- the learned energy selects better candidates than random/first-candidate and
  approaches an evaluation-only GT oracle ranking;
- the complete system is compared against VLU-Net's fixed unfolding concept,
  SIPL-style recurrent refinement, and at least one external-feedback TTS
  baseline or faithful controlled surrogate;
- gains survive source-disjoint validation and a single frozen reporting-set
  evaluation without tuning on protected pixels.

Under that evidence, a defensible paper statement is:

> To our knowledge, this is the first VLM-conditioned AiOIR framework that
> treats refinement of a frozen heterogeneous restoration proposal as
> optimization of a learned candidate energy, using the same intrinsic energy
> for prediction-space descent and test-time verification/stopping.

Avoid shortening this to "the first EBM-TTS AiOIR method" because that phrase
hides important distinctions and is easier for VLU-Net, SIPL, KDS, EBT, ALPS,
or ResFlow-Tuner to invalidate.

## Staged causal experiment and adequate-scale requirement

The Noise-first pilot must freeze the data split, proposal outputs, metrics,
and selection rule before training. Its minimum comparison is:

1. frozen routed proposal (`0` refinement steps);
2. matched-parameter feed-forward bounded residual refiner;
3. matched recurrent/self-refinement baseline without scalar energy;
4. scalar-energy refinement at fixed `1/2/3` candidate steps;
5. ablations without VLM condition and without degraded-observation condition.

The pilot passes technically at `+.30 dB` and `-.005` LPIPS and becomes
paper-promising at `+.50--.75 dB` and `-.010` LPIPS relative to the frozen
proposal. No official, sealed, internal-test, parent-test, or frozen 660
reporting pixels may influence training, model selection, step count, or
stopping thresholds.

The first GPU qualification remains bounded to 500 optimizer steps on one
8-GPU node. Its wall time and allocated GPU-hours must be measured on the
actual H100/H200 family rather than capped by the obsolete `.5`-hour estimate.
It is only a systems qualification and cannot, by itself, reject the method. After
it passes, a formal Noise test must use the complete 54,585-record training
split for at least one effective pass. Additional passes must be predeclared
from the source-disjoint development trend and measured throughput. A negative
method conclusion requires adequate exposure followed by a saturated or
worsening development curve; it must not be inferred from an undertrained
500-step checkpoint. Every launch must report wall-clock hours and allocated
GPU-hours.

## Sources verified in this audit

- VLU-Net: https://arxiv.org/abs/2503.16930
- SIPL: https://arxiv.org/abs/2505.24207
- KDS: https://arxiv.org/abs/2507.05604
- ResFlow-Tuner: https://arxiv.org/abs/2603.22027
- TTPO: https://arxiv.org/abs/2511.19169
- TAO: https://arxiv.org/abs/2312.02197
- EBT: https://arxiv.org/abs/2507.02092
- ALPS: https://arxiv.org/abs/2601.02594
- DAME-Net/MDUR: https://arxiv.org/abs/2604.09313
- Existing system audit:
  `reports/foundir_current_method_novelty_audit_2026-09-27.md`
- Existing EBT audit:
  `reports/foundir_compositional_ebt_phase0_audit_2026-10-07.md`

## Audit limitation

This was a claim-targeted public-paper search, not a mathematical proof that no
equivalent unpublished, newly indexed, non-English, or differently titled work
exists. Re-run the exact-intersection search immediately before submission and
use "to our knowledge" rather than an unconditional first claim.
