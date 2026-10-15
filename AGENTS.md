# Project agent instructions

## Active workspace and runtime

- The sole active project root is `/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang`.
- Use `/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3` for all Python work.
- Preserve all dirty and untracked project files. Do not reset, clean, or overwrite unrelated work.
- Keep the old HDD workspace untouched.
- Never use official, sealed, parent-test, or internal-test pixels for training, hyperparameter selection, checkpoint selection, or early stopping.

## MANDATORY: regular focused GitHub progress tracking

GitHub is the durable project history. Do not allow material work to remain
only in the dirty local workspace for long periods. The following is a hard
operating rule for all later agents, not an optional cleanup suggestion.

Create a focused commit and push it to `origin/main` after each coherent,
material milestone, including:

- a new or materially changed model, training script, evaluation script,
  configuration, launcher, or reproducibility contract;
- a code failure that required a substantive fix once the corrected code has
  passed the available static/CPU/GPU qualification;
- an advisor/user research-priority decision that changes the active plan;
- a completed qualification, pass-level validation, selected checkpoint, or
  terminal experiment result after its report/handoff has been updated;
- before leaving a long-running experiment unattended, provided its launch
  code/config and current handoff form a coherent reproducible snapshot.

For an active run, do not commit every log line or optimizer step. Commit the
launch-ready code before or immediately after launch, then commit again at
meaningful validated milestones and terminal completion. A running job is not
an excuse to leave its source/config untracked.

Every push must follow this safety procedure:

1. Run `git fetch origin` and inspect local/remote divergence.
2. Preserve the dirty workspace. Never use `git reset --hard`, `git clean`, or
   broad checkout commands to prepare a push.
3. Stage an explicit allowlist of relevant source/config/report files. Never
   use `git add -A`, `git add .`, or a directory-wide add in this repository.
4. Exclude checkpoints, model weights, raw/generated images, datasets, caches,
   telemetry, live logs, temporary files, and large generated manifests. Keep
   data identity through small metadata summaries and SHA-256 bindings.
5. Scan the staged files for credentials, access tokens, private keys, and
   unintended machine-specific secrets. Never commit the previously exposed
   GitHub token or any replacement credential.
6. Run syntax/JSON checks and `git diff --cached --check`; inspect the staged
   file list and size before committing.
7. If the main workspace is heavily dirty or an untracked file conflicts with
   an incoming remote path, create an isolated temporary Git worktree from
   `origin/main`, copy only the explicit allowlist, commit there, and push from
   there. Do not disturb the live training workspace.
8. Push the focused commit to `origin/main`, verify the remote commit hash, and
   report the commit hash and included scope to the user.

If authentication, networking, a remote conflict, or a required secret scan
blocks the push, report that immediately and preserve the ready commit. Do not
silently defer Git tracking across later milestones. Do not claim the work is
backed up until the remote hash has been verified.

## STOP: advisor-approved two-stage novelty plan — 2026-10-15

This section supersedes the 2026-10-14 energy-TTS decision below wherever the
two differ. The phrase "PPT direction" means the restoration-energy method
already presented in the advisor slides; it does not mean making more slides.

The advisor's decision is:

1. **Try the PPT restoration-energy idea first.** Test whether a lightweight
   action-contrastive / proposal-relative energy can improve the already
   frozen FLUX.2/Restormer output. The energy is conditioned on the degraded
   input and explicit VLM degradation information. It should learn to
   distinguish a correct restoration from under-restoration, over-restoration,
   a wrong expert action, or newly damaged details, and then use that signal
   for a small number of candidate-image correction steps. This is currently a
   hypothesis, not an established working result.
2. **Use a staged but adequately scaled causal test.** Start from Noise on
   source-disjoint development data and compare against both the frozen
   proposal and a matched feed-forward refiner. Preserve the numerical gates
   and data restrictions in the 2026-10-14 section below. The first 500 steps
   are an engineering qualification only: they test correctness, memory,
   throughput, checkpoint/resume, and whether gradients and validation are
   finite. They are not enough to declare the research idea dead. After that
   qualification, train on the complete 54,585-record Noise training split.
   The first scientific decision point must include at least one complete
   effective pass over those records; if the validation trend is still
   improving, continue through a predeclared multi-pass schedule rather than
   stopping because the model is initially undertrained. Derive the exact
   optimizer-step schedule and GPU-hours from measured throughput before the
   formal launch. Do not claim success merely because learned energy decreases;
   PSNR/LPIPS and candidate ranking/refinement must improve.
3. **Only if the PPT energy idea does not work, switch to the advisor's DiT-MoE
   fallback.** The fallback is a FLUX/DiT restoration model whose internal
   experts are controlled using explicit VLM predictions of degradation type,
   severity, and composition. Do not begin this fallback in parallel with the
   first energy test unless the user explicitly changes the sequence.

Novelty boundaries for the fallback must be stated honestly:

- FoundIR-v2 already uses MoE, although its backbone is SDXL rather than DiT.
- MiM-DiT already uses MoE inside a diffusion transformer for All-in-One image
  restoration. Therefore **"MoE + DiT" is not a new contribution**.
- The remaining hypothesis is narrower: MiM-DiT directly looks at degraded
  image features to choose experts, while the proposed fallback first asks a
  VLM to state the degradation type, severity, and composition and then uses
  that explicit diagnosis to control DiT experts.
- "We added a VLM to the router" is not automatically enough novelty. It only
  becomes defensible if a matched comparison shows that explicit VLM guidance
  improves mixed/OOD degradation restoration or expert composition over an
  otherwise identical image-feature router.

Advisor-chat provenance, in plain language: the advisor said the PPT ideas are
reasonable but may not work and asked whether any already work. The advisor
then suggested MoE, initially noting FoundIR-v2 and its lack of this DiT
combination. After learning that MiM-DiT already covers the broad MoE+DiT idea
but does not explicitly output degradation type/severity through a VLM, the
advisor approved the sequence: **try the project's own PPT energy idea first;
if it fails, the VLM-guided DiT-MoE fallback is still acceptable because it is
not a complete overlap.** No method-success claim follows from that approval.

### Frozen scale and hardware plan for the first energy experiment

Only one homogeneous eight-GPU node may be used for this experiment. It may be
**8 x NVIDIA H200** or **8 x NVIDIA H100 80GB**, but never a mixed node and
never multiple nodes. Do not create or propose a cross-node launch. All
training, validation, checkpointing, and resume logic must work on either
supported single-node profile while preserving global batch 32.

- H200 profile: micro-batch 4/GPU, gradient accumulation 1.
- H100 profile: micro-batch 1/GPU, gradient accumulation 4.

These profiles have the same eight GPUs, global batch, optimizer-step schedule,
and training-record sampler. The smaller H100 micro-batch is only a memory
compatibility change; it does not authorize fewer data passes or different
method settings.

The Noise training split contains 54,585 paired records. Use global batch 32
unless the measured H200 memory/throughput qualification proves a different
batch is materially better. At global batch 32, one complete effective data
pass is `ceil(54,585 / 32) = 1,706` optimizer steps.

Use this frozen staged scale:

1. **Systems qualification:** 500 optimizer steps. This checks correctness,
   finite second-order gradients, memory, throughput, validation, checkpoint,
   and exact resume. It is not a scientific failure decision.
2. **First scientific checkpoint:** 1 complete effective pass, 1,706 total
   optimizer steps. Evaluate the frozen proposal, matched feed-forward refiner,
   and 1/2/3-step energy refinement on the fixed source-disjoint development
   validation set.
3. **Primary adequately scaled run:** 5 complete effective passes, 8,530 total
   optimizer steps. Save and validate at the end of every pass. This is the
   minimum scale for the main method judgment unless an earlier run has a
   numerical/code failure rather than merely weak metrics.
4. **Pre-authorized continuation:** extend the same run to at most 10 complete
   effective passes, 17,060 total optimizer steps, only when pass-4 to pass-5
   source-disjoint validation still shows a meaningful improving trend and no
   overfitting. Do not change the validation set, loss, candidate steps, or
   checkpoint rule based on protected reporting data.

Do not launch 100,000 steps by default: that would be approximately 58.6
passes over this split and is not currently justified. The 10-pass cap may be
changed only by a later explicit user decision backed by the pass-by-pass
development curve.

Exact wall-clock and allocated GPU-hours cannot be claimed before the H200
qualification measures seconds per optimizer step. After the 500-step run,
compute and report:

- `wall hours = remaining_steps * measured_seconds_per_step / 3600`;
- `allocated GPU-hours = wall hours * 8`.

Never transfer a timing estimate between H100 and H200, and never substitute
an old multi-node EBT benchmark. Measure seconds/step on the exact single-node
GPU family used for the run.

## Mandatory reporting style

All future user-facing reports must be easy to understand on the first read:

- Start with the practical answer or current outcome, then explain why.
- Use short, direct Mandarin sentences; retain English technical terms only
  where they prevent ambiguity.
- Translate each necessary term immediately into plain language. For example,
  say "模型直接看图选专家" before or alongside "latent-feature routing".
- Clearly separate **confirmed fact**, **our hypothesis**, and **next test**.
- For every comparison, name both sides, the dataset/protocol, and what each
  number means. Never use an unexplained phrase such as "coverage", "gate", or
  "better".
- Do not present an idea, advisor approval, lower energy, or a successful code
  run as evidence that image quality improved. State "not tested yet" when no
  metric result exists.
- Give time estimates in normal wall-clock hours and also report allocated
  GPU-hours for every proposed or completed GPU experiment.
- Prefer one concrete example over dense jargon. Add equations or internal
  implementation details only after the plain-language explanation.

## STOP: frozen FLUX/Restormer proposal + energy-TTS decision — 2026-10-14

This section supersedes the older Generalist-first and live-run sections below.
The from-scratch EBT Generalist is paused and must not be resumed, duplicated,
reconfigured, evaluated on the protected 660 endpoint, or used to select a
checkpoint. Its existing artifacts and failure logs remain historical evidence.

The only active restoration-model research objective is a lean test of
EBT-style test-time optimization applied to the already successful frozen
FLUX.2/Restormer routed system. FLUX.2/Restormer produces an immutable initial
proposal. A lightweight observation- and VLM-conditioned scalar energy
`E(y; x, c)` may then refine candidate `y` by a small number of gradients with
respect to the candidate image. Do not retrain or alter the frozen proposal
models during the first causal test.

Work source-disjoint development data only. Never use the official, sealed,
parent-test, internal-test, or frozen 660 reporting pixels for training,
hyperparameter selection, checkpoint selection, early stopping, or deciding
the number of test-time steps.

Proceed in this order:

1. Start with the Noise singleton, because the frozen routed result
   `40.133486/.096746` is behind Full FoundIR `42.655305/.067910` in both PSNR
   and LPIPS.
2. Compare the frozen proposal against a matched-parameter feed-forward bounded
   refiner and fixed 1/2/3-step candidate-gradient refinement. Require extra
   steps to improve restoration rather than merely lower an uncalibrated
   energy.
3. A minimum technical pass is at least `+.30 dB` and `-.005` LPIPS relative
   to the frozen proposal on protected source-disjoint development validation.
   Paper-worthy evidence is at least `+.50--.75 dB` and `-.010` LPIPS, a useful
   compute-quality curve, and superiority to the matched feed-forward refiner.
4. Only after the Noise gate passes may the same frozen mechanism be tested on
   the remaining singleton categories. A convincing singleton aggregate target
   is at least `29.2 dB` with LPIPS no worse than `.135`, with gains across
   multiple categories rather than one category dominating the mean.
5. Only after multi-category singleton evidence may the unchanged mechanism be
   evaluated on Double compositions. The frozen Double baseline is
   `24.321537/.192313`; an indicative paper-strength target is at least
   `24.8/.180`, subject to per-category no-harm evidence.
6. Do not reopen Triple, specialists, compositional EBT, or another from-scratch
   Generalist until this lean mechanism succeeds or fails under its frozen
   causal comparison.

The working novelty hypothesis is not simply "EBM", "more inference steps",
or "VLM + EBT". It is candidate-dependent, differentiable test-time refinement
of frozen heterogeneous AiOIR proposals under an observation- and structured
VLM-conditioned scalar energy, with the same energy used for iterative
restoration and stopping/verification. Treat "first" as unverified until the
targeted prior-art audit is complete. Current audit and exact claim boundary:
`reports/foundir_flux_restormer_energy_tts_novelty_audit_2026-10-14.md`.

The first GPU qualification, if the implementation and protocol pass CPU
review, is capped at 500 optimizer steps on one 8-GPU node. Its duration is
measured rather than constrained by the obsolete `.5` wall-hour estimate,
because second-order candidate gradients may take longer on H100 than H200.
**This is a systems check, not a method-failure gate.** After it passes, use
its measured throughput to freeze
an adequately scaled schedule over all 54,585 Noise training records, with at
least one complete effective data pass and further predeclared passes whenever
the source-disjoint validation trend is still improving. Report both
wall-clock hours and allocated GPU-hours. Stop only for a code/numerical
failure, a clearly saturated or worsening development curve after adequate
exposure, or completion of the frozen scale schedule—not merely because a
small run misses the final metric target.

## DEFERRED: Generalist-first research decision — 2026-10-11

This was the active objective on 2026-10-11 but is superseded by the
2026-10-14 decision above. Preserve it as historical protocol context only.

Do not design, implement, train, tune, benchmark, or launch any low-light,
weather, trajectory-energy, residual-energy, or other specialist. Preserve the
specialist prior-art/design notes as deferred research only. Do not use a
specialist result to claim that the Generalist gate passed.

The decisive comparison is Generalist versus Generalist on the existing 660
full-resolution whole-image composition set, using identical records and the
frozen metric protocol (`full_rgb_uint8_image` PSNR and
`area_weighted_nonoverlap_512_rgb_uint8` LPIPS). Neither side may run a
specialist. The released FoundIR Generalist reference is hash-bound in
`checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json`.

For an unambiguous pass, the EBT Generalist must beat the released FoundIR
Generalist in both PSNR (higher) and LPIPS (lower) for every composition tier,
not merely in a weighted aggregate:

| Tier | Released FoundIR Generalist PSNR | Released FoundIR Generalist LPIPS |
|---|---:|---:|
| Singleton, 300 images | 27.2323939564508 | 0.1744119563086327 |
| Double, 300 images | 22.245453630403492 | 0.2066620594635684 |
| Triple, 60 images | 24.9455711880497 | 0.03775338436922618 |
| Overall, 660 images (reporting only; not a substitute for tier gates) | 24.757709920211013 | 0.17664758756638468 |

## HISTORICAL FAILED RUN: real-data Generalist and pass-1 diagnostic — 2026-10-12

`real_generalist_4node_mixed_hopper_v1` ran on 8 x H200 plus 24 x H100. It used
global batch 64, 10,233 steps/pass, three planned passes, EMA `.9999`, and the
frozen 654,881-pair real training split. It failed during pass 2 at pass step
1,300 / global step 11,533 after a 600-second NCCL ALLREDUCE watchdog timeout.
Only pass 1 is complete. Do not resume or evaluate this run while the
2026-10-14 decision is active. Its 16,461-record 256x256-crop validation is
diagnostic only and is not comparable to the decisive 660-image protocol.

Pass-1 raw tier metrics were Singleton `28.829671/.172490`, Double
`23.916114/.242836`, and Triple `33.070977/.025842`. The apparent Double/Triple
inversion is not contradictory:

- Triple contains only `blur+low_light+noise` (579 validation records), whereas
  Double averages five heterogeneous categories.
- Double raw per-category LPIPS is `blur+noise .017810`,
  `haze+low_light .196602`, `low_light+noise .242655`, `haze+rain .277549`,
  and `blur+low_light .312665`. The last three drive the Double weakness.
- The sampler assigns one third of presentations to each composition tier. The
  sole Triple category therefore receives approximately five times the
  category-level exposure of each of the five Double categories.
- LPIPS has weight `.05` and is included only every four optimizer steps, so
  PSNR can improve before hard Double perceptual artifacts are resolved.

The former instruction to complete all three passes and run a matched 660
evaluation is deferred. Do not access the 660 endpoint for this failed run
while the 2026-10-14 decision is active.

## Historical FLUX/Restormer gap record for the monolithic/generalist route

The old routed FLUX.2/Restormer system's Triple gap to FoundIR was not a simple
parameter-capacity problem. Preserve these established causes. They motivate
the bounded energy-refinement test above but do not authorize specialist or
Triple work:

1. Most FLUX work used synthetic/task-specific data derived from roughly 1,600
   clean sources, versus FoundIR metadata covering 1,010,064 pairs and the
   frozen 654,881-pair real training split now used by EBT.
2. FLUX experts received hundreds to roughly 10,000 task-specific steps,
   whereas the released FoundIR Generalist is at step 1,300,000. Raw steps are
   not batch-normalized, but the exposure gap is material.
3. FLUX/Restormer was a heterogeneous routed expert system, not one jointly
   trained Generalist over all singleton/double/triple categories.
4. Sequential experts have measured inter-expert domain shift: on 53 Triple
   records, Noise was nearly inert (`+.069859 dB`), Low-light overshot GT luma
   by `+29.503`, and Blur worsened 49/53 records and lost `.272308 dB`.
5. Synthetic operator order/strength did not fully match FoundIR, including the
   documented Noise/JPEG visibility mismatch.
6. Short FLUX/Restormer selections lacked a mature EMA selection path; current
   EBT uses EMA `.9999` and validates raw and EMA after every pass.
7. FLUX is a large pretrained generative DiT, while FoundIR's approximately
   36.3M-parameter UNetRes is restoration-native. More FLUX parameters did not
   substitute for paired restoration alignment.
8. Full FoundIR also has Low-light and Weather specialist refinement. That is a
   separate full-system advantage and must not be mixed into the frozen
   Generalist-versus-Generalist gate. Specialist work remains paused.

After the active real-data run completes, the remaining path is:

1. Audit all pass/checkpoint/validation artifacts and hashes.
2. Select and freeze raw or EMA weights without protected-test access.
3. Run one matched 660 whole-image Generalist-versus-Generalist evaluation.
4. Only after all tier gates pass may specialist research be reopened.

Report wall-clock hours and allocated GPU-hours for every proposed or completed
GPU experiment.

## DEFERRED: potential SOTA follow-up after the Generalist gate — 2026-10-12

This records a candidate paper-evaluation program. It does not change the
active Generalist-first decision and does not authorize another training or
evaluation while `real_generalist_4node_mixed_hopper_v1` is running. First
finish all three frozen passes, select raw versus EMA using only the frozen
16,461-record validation evidence, and run exactly one matched 660-image gate.
Do not tune from that result. The project is not yet demonstrably SOTA:
pass-1 crop validation is encouraging, but Double LPIPS remains worse than the
released FoundIR Generalist and is not directly comparable to whole-image
metrics.

If and only if EBT beats the released FoundIR Generalist in both PSNR and LPIPS
for Singleton, Double, and Triple on the frozen 660 protocol, the lean
potential-SOTA program is:

1. Train/evaluate one matched CDD-11 model and report all eleven categories
   plus Singleton/Double/Triple aggregates with PSNR/SSIM. Main composite
   targets are AllRestorer `33.72/.9436`, DAR-Net `29.73/.888`, ClearAIR
   `29.34/.886`, MoCE-IR-S `29.05/.881`, and OneRestore `28.47/.878`.
   AllRestorer lacks a verified reproducible public implementation, but its
   published number cannot be omitted from an unrestricted SOTA claim.
2. Train/evaluate one matched standard 5D model on SOTS, Rain100L, BSD68 sigma
   25, GoPro, and LOL. Important average PSNR/SSIM targets include D3Net
   `32.36/.901`, DAR-Net `31.02/.921`, DPC-Net `31.00/.923`, DFPIR
   `30.64/.913`, MoCE-IR `30.58/.919`, and CyclicPrompt `30.52/.935`.
   Disclose metric/protocol tradeoffs and never compare unmatched numbers.
3. Run two matched retraining ablations: no VLM conditioning and no learned
   energy/direct restoration. Use the frozen full checkpoint for evaluation-
   only adaptive-TTS fixed-step sweeps and self-verification on/off tests.
4. Report parameters, FLOPs, peak memory, latency, and actual adaptive descent
   steps. Cite and discuss FoundIR-v2, VLU-Net, DFPIR, InstructIR, Perceive-IR,
   ClearAIR, and EBT even where benchmarks are not directly matched.

The minimum defensible follow-up has eight blocks: frozen 660 evaluation;
matched CDD-11 train/eval; matched 5D train/eval; two retraining ablations;
adaptive-TTS evaluation; self-verification evaluation; and efficiency
evaluation. This means four new training runs and roughly six to eight
evaluation jobs after the current gate. A broad AiOIR claim additionally needs
separately trained 3D, real-world, and robustness/OOD evidence, raising the
program to roughly 12--15 blocks and six to eight training runs.

The current 30,699-step 32-GPU schedule measures about 8.97 wall hours/pass,
or about 27 wall hours and 864 allocated heterogeneous GPU-hours for three
passes. Four mechanical repeats would be an unapproved conservative ceiling of
108 serial wall hours and 3,456 GPU-hours. Before any CDD-11, 5D, or ablation
launch, run a protocol-specific 500-step throughput qualification (target at
most 0.5 wall hour and 4 allocated GPU-hours on eight GPUs), then size training
by matched epochs and unique-sample exposure. Narrow the intended claim to
composite/mixed Generalist SOTA or best matched average; winning the
project-specific 660 comparison alone is not universal AiOIR SOTA. Specialist
work remains paused.
