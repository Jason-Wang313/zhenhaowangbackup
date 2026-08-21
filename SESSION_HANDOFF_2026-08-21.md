# Session handoff — 2026-08-21

This is the complete continuation guide for a fresh agent with no prior chat
context. Do not rely on an earlier model-provider history or assume that a GPU
job is still running. Every operational claim below was checked against this
workspace on 2026-08-21.

## Mandatory startup order and authority

1. Work only in
   `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`.
2. Begin every work session by telling the user exactly: `I am working in
   zhenhaowang, which is the correct workspace.`
3. Read `agents.md` completely, then read this file completely.
4. Run `git status --short --branch`, `git rev-parse HEAD`, and
   `git rev-parse origin/main` before changing anything. Preserve all unrelated
   tracked and untracked work.
5. Treat the advisor-authoritative architecture and safety rules in
   `agents.md` as higher authority than this handoff. This handoff supersedes
   `SESSION_HANDOFF_2026-08-18.md` only for current operational status and the
   next task. Older handoffs are historical evidence and must remain intact.
6. Verify relevant files and machine-readable artifacts directly before making
   new claims. Do not reconstruct status from memory or chat alone.

The implementation baseline immediately before this documentation handoff was
`3e43880ac99b9d18b5bba0a4d949d2b82398a856` on `main`, with local HEAD equal
to `origin/main`. Its subject is `Prepare source-disjoint Qwen real adaptation
gate`. The documentation handoff itself is committed after that baseline; a
fresh agent must use the actual current `git rev-parse HEAD` and verify that it
equals `git rev-parse origin/main`.

Known unrelated untracked files existed at handoff and must not be staged,
modified, or deleted as part of the next task:

- `manifests/low_light_pilot_smoke_500.csv`
- `manifests/low_light_pilot_smoke_500.json`
- `manifests/low_light_source_pool.json`
- `manifests/source_pool_mio_div2k_partial.json`
- `reports/model_audit_stage0.md`
- `smoke/calibrate_agenticir_noise_levels.py`
- `smoke/calibrate_agenticir_selected_002.py`
- `smoke/download_flux2_chunks.py`
- `smoke/download_qwen_part1.sh`
- `smoke/monitor_qwen_download.sh`
- `smoke/run_agenticir_noise_levels.py`
- `smoke/run_agenticir_selected_blur.py`
- `smoke/run_agenticir_selected_haze.py`
- `smoke/run_agenticir_selected_low_light.py`
- `smoke/run_agenticir_selected_noise.py`
- `smoke/run_agenticir_selected_rain.py`
- `smoke/watch_flux2_then_qwen.sh`

The destructive-change rule in `agents.md` requires two clarification rounds
before deletion, overwrite, replacement, reset, or another hard-to-recover
change. Do not interpret this handoff as permission for cleanup.

## Project scope and immutable architecture

The user owns only the `VLM Guided All-in-One` image-enhancement workstream.
The separate `亮色大模型` workstream belongs to another project member and is
out of scope unless the user explicitly changes responsibility.

Current execution scope is Lowlight only. The advisor-authoritative route is:

1. Train a degradation-aware VLM with supervised token-level cross-entropy.
   The degraded image and a degradation-assessment question are inputs; the
   structured degradation type and severity description is the target.
2. Train the vision encoder and multimodal path while keeping the tokenizer,
   language decoder, and LM head frozen.
3. Validate the VLM credibly on real/downstream data. Synthetic validation is
   not sufficient.
4. Only after a credible VLM pass, freeze the VLM and use its full structured
   degradation description to guide the trainable diffusion restoration model.
5. Freeze the complete system before the official FoundIR evaluation.

FoundIR is protected. It has not been used by the latest VLM adaptation and
must not be used for iterative selection, tuning, corrective adaptation, or a
replacement holdout. Diffusion is not authorized because the VLM has not
passed credible real-domain validation.

Do not revive the superseded frozen-feature ridge/one-token route, generate the
100,000-record corpus, expand to all five degradations, start diffusion, or run
FoundIR at the present gate.

## Completed synthetic supervised-VLM pilot

The genuine Qwen supervised synthetic Lowlight pilot passed and remains the
initialization source for the real-adaptation experiment:

- output: `checkpoints/qwen_lowlight_supervised_pilot_8k_ddp8/`
- selected checkpoint: step 600
- selected visual SHA-256:
  `b0e65a8597077973a2a103dc3369e0cb1ea703de95e2e4841d359c3d34eff77f`
- hardware: eight H200 GPUs
- optimizer steps: 1,000
- synthetic validation: severity accuracy `0.939`, severity MAE `0.061`,
  valid JSON `1.0`
- synthetic internal test: severity accuracy `0.944`, severity MAE `0.058`,
  valid JSON `1.0`

This is synthetic-domain evidence only. It never authorized diffusion by
itself.

## Preserved real-domain failures before adaptation

The synthetic step-600 VLM failed two real-domain evaluations. Both failures
are immutable evidence and must not be relabeled:

1. Original free-generation result:
   `checkpoints/qwen_lowlight_real_validation_step600/result_summary.json`
   with SHA-256
   `85542baf277c72a442e12d6da4f04e02051e45f7d586020472253437b6ebceda`.
2. Constrained-scoring diagnostic:
   `checkpoints/qwen_lowlight_real_validation_constrained_step600/result_summary.json`
   with SHA-256
   `8cb62790a423438d2d62e096d0985075a3c4c2e179e950d2beb5a0e61ab44552`.

The constrained diagnostic made SICE JSON validity `65.56% -> 100%`, severity
accuracy `21.11% -> 25.56%`, MAE `2.222 -> 1.444`, and within-one accuracy
`34.44% -> 46.67%`, while QWK changed `0.355 -> 0.299`. Predictions remained
collapsed toward low severity levels. LOL-v2 constrained severity accuracy was
slightly worse than free generation. Therefore malformed generation was a
secondary issue; real-domain calibration/generalization was the main failure.

Do not repeat constrained decoding alone and claim that the calibration
problem is solved.

## Frozen real-adaptation data and leakage rules

The adaptation dataset uses the official LOL-v2 Real-captured Train split with
reference-based proxy labels. These labels are not human ground truth.

- manifest: `manifests/qwen_lowlight_real_adaptation.csv`
- manifest SHA-256:
  `91f95644f852f628891513f116c0a3607b8de980bbe5252c360ff41d9bcc9ab7`
- metadata: `manifests/qwen_lowlight_real_adaptation.json`
- metadata SHA-256:
  `2b9f35ffdc2543a4a99bf95fbb4eb61938a022c54cba4569a14863c5b536f06f`
- annotation disclosure: `reference_based_proxy_not_human_ground_truth`
- 272 perceptual scene components
- no exact overlap with previously evaluated real images
- no LOL-v2 Train/Test Normal perceptual-hash edge at Hamming distance <= 4

Frozen source-disjoint composition:

| Split | Records | Scene components | L1/L2/L3/L4/L5 |
|---|---:|---:|---|
| Train | 673 | 190 | 190/14/18/52/399 |
| Validation | 143 | 41 | 41/3/3/11/85 |
| Former fresh test | 145 | 41 | 41/3/4/11/86 |

The 145-record fresh test was evaluated exactly once after checkpoint
selection. It is now consumed. Never run it again as a fresh holdout, select a
checkpoint from its results, or tune a corrective experiment from its sample
outcomes. It may be cited only as historical failed evidence.

Forbidden sources for a new holdout are:

- official FoundIR;
- SICE records already evaluated;
- LOL-v2 official Test records already evaluated;
- every record in `manifests/qwen_lowlight_real_adaptation.csv`;
- any exact or perceptually overlapping scene/source.

A corrective experiment requires a newly acquired, previously unseen,
source-disjoint and perceptually disjoint real holdout whose rubric, labels,
manifest, thresholds, and hashes are frozen before inference.

## Frozen real-adaptation experiment contract

The completed experiment used:

- config: `configs/qwen_lowlight_real_adaptation_step600.json`
- config SHA-256:
  `0a817d70d88f172d6d16d37853ce9c3cd77308446b3eed3d5e5cdab0c3b3b541`
- initialization: the synthetic step-600 full visual path
- trainable scope: all 390 visual tensors / 676,550,144 parameters
- frozen scope: language decoder and LM head, 7,615,616,512 parameters
- optimizer: fresh AdamW state
- 200 optimizer steps, global batch 40, 8,000 sample presentations
- each step: 20 L1-L5-balanced real presentations plus 20 L1-L5-balanced
  synthetic replay presentations
- every microbatch contains all L1-L5 levels
- exact schedule SHA-256:
  `8ae8e4e06a306c3828423374e39cecac4e3be2cf90b2b08564005b928a62c71f`
- two GPUs: microbatch 5 per rank, gradient accumulation 4
- eight GPUs: microbatch 5 per rank, gradient accumulation 1
- checkpoint candidates: steps 25, 50, 100, 150, and 200
- selection: minimum real-validation macro-level teacher-forced loss, with an
  earlier-step tiebreaker; the test was not used for selection
- synthetic replay validation: 40 complete sources / 200 records
- replay bundle SHA-256:
  `1981939ac92de5412df990f39f9d7c2ee8776ec537e76dec059a81378407cfc3`

The contract and data are frozen historical evidence. Do not silently alter
them and call the result a rerun of the same experiment.

## Latest GPU run: complete, verified scientific failure

The real-adaptation run has finished. No training process should be assumed
active. Do not launch it again.

Authoritative artifacts:

- output: `checkpoints/qwen_lowlight_real_adaptation_step600/`
- summary:
  `checkpoints/qwen_lowlight_real_adaptation_step600/training_summary.json`
- summary SHA-256:
  `924f3d552b5305783873f769c8e1c18053548ed24714511671892a5db7378f03`
- checkpoint selection:
  `checkpoints/qwen_lowlight_real_adaptation_step600/checkpoint_selection.json`
- selection SHA-256:
  `c7618cd719f63b8f08f0426ef62356cda4595428a199216c18a42ab5d6640324`
- persistent logs:
  `logs/qwen_lowlight_real_adaptation_step600.log`,
  `logs/qwen_lowlight_real_adaptation_step600_events.log`, and
  `logs/qwen_lowlight_real_adaptation_step600_gpu.csv`

Machine-readable result: `status="failed"` and
`automatic_gate_pass=false`. This is a legitimate scientific failure, not a
training crash.

Execution facts:

- hardware: two NVIDIA H100 80GB HBM3 GPUs
- all 200 optimizer steps completed
- 8,000 sample presentations completed
- selected checkpoint: step 150
- selected visual SHA-256:
  `32c0496ee758b990d312d67a6614d2f85ef8e87804c97e9f6158bad51ace9388`
- initial real macro teacher-forced loss: `0.1205462946`
- selected loss: `0.0713094496`
- selected/initial ratio: `0.5915524`, passing the required `<= 0.9`
- training time: `1929.23 s`
- trainer end-to-end time: `2285.78 s`
- throughput: `4.1467 samples/s`
- maximum torch allocation: approximately `23.9 GiB` per GPU
- telemetry maximum used memory: approximately `27.0 GiB` per GPU
- mean utilization while used memory exceeded 26,000 MiB: approximately `38%`
  per GPU

Do not claim full GPU utilization, maximum efficiency, or minimum possible
runtime from this run.

Integrity evidence passed:

- gradients reached all 390 visual tensors / 676,550,144 trainable parameters;
- no frozen parameter received a gradient;
- all 7,615,616,512 frozen parameters stayed unchanged;
- all five checkpoint freeze audits passed;
- 200/200 step-metric files exist;
- 6/6 validation-loss files exist for steps 0/25/50/100/150/200;
- all evaluation artifact hashes matched;
- independent recomputation from stored prediction rows reproduced the metrics
  and gate decision;
- FoundIR was not used.

## Exact gate results

### Real validation: all predeclared thresholds passed

- valid structured output: `1.0`
- degradation-type accuracy: `0.9580`
- degradation-type macro-F1: `0.9462`
- severity accuracy: `0.7552`
- severity macro-F1: `0.5594`
- severity MAE: `0.2517`
- within-one-level accuracy: `0.9930`
- QWK: `0.9553`
- per-level accuracy: L1 `35/41=0.8537`, L2 `2/3=0.6667`, L3
  `2/3=0.6667`, L4 `5/11=0.4545`, L5 `64/85=0.7529`

### Consumed fresh real test: failed only the required per-level floor

- valid structured output: `1.0`
- degradation-type accuracy: `0.9448`
- degradation-type macro-F1: `0.9299`
- severity accuracy: `0.7655`
- severity macro-F1: `0.5132`
- severity MAE: `0.2414`
- within-one-level accuracy: `0.9931`
- QWK: `0.9579`
- per-level accuracy: L1 `35/41=0.8537`, L2 `0/3=0.0`, L3
  `4/4=1.0`, L4 `2/11=0.1818`, L5 `70/86=0.8140`
- required per-level floor: `0.25`

All three L2 samples came from the single scene component
`lolv2_real_train_scene_00339_35d38eef8650`. That is historical diagnostic
evidence, not permission to tune on those three records. Never weaken the
floor, remove a level, or reclassify this as a pass.

### Synthetic replay: three safeguards failed

- valid structured output: `0.93`, required `>=0.99` — failed
- degradation-type accuracy: `0.93`, required `>=0.95` — failed
- degradation-type macro-F1: `0.9522` — passed
- severity accuracy: `0.89`, required `>=0.85` — passed
- severity macro-F1: `0.9224` — passed
- severity MAE: `0.32`, required `<=0.25` — failed
- per-level accuracy: `0.875/0.90/0.85/0.875/0.95` — all passed

The 14 malformed outputs were concentrated in five sources: `div2k_0235`
(5), `div2k_0387` (1), `lsdir_0000581` (2), `lsdir_0000723` (1), and
`lsdir_0000807` (5). The result indicates scene-specific synthetic
structured-output forgetting while real calibration remains imperfect.

## Launcher-status defect: fix before another GPU launch

The trainer in `smoke/train_qwen_lowlight_real_adaptation_ddp.py` intentionally
returns status 2 after writing a complete verified failed summary. The log
correctly contains `QWEN_REAL_ADAPTATION_FAILED`. However,
`python -m torch.distributed.run` converts that child exit into its generic
`ChildFailedError` and returns shell status 1. The current shell runner in
`smoke/run_qwen_lowlight_real_adaptation.sh` merely forwards that outer status,
so its events incorrectly record:

- `QWEN_REAL_ADAPTATION_TRAINING_END status=1`
- `QWEN_REAL_ADAPTATION_TERMINAL_EXIT_STATUS 1`

This explains why the tmux pane reported an error-looking exit. It does not
invalidate the completed summary and is not a reason to rerun training.

Before any future GPU launch, change the checked runner so that after
`torch.distributed.run` returns it inspects the completed summary. If the
summary is internally verified, belongs to the current config/run, has
`status="failed"`, and has `automatic_gate_pass=false`, normalize the outer
result to terminal status 2 and emit the legitimate automatic-gate-failure
marker. Preserve status 1 for real crashes, missing/incomplete summaries,
hash/config mismatches, or inconsistent state. Add focused CPU tests for all
of those branches. Do not alter the existing log or summary to hide the defect.

## Current authorization and stop conditions

The failed gate requires user review. At handoff creation, the user requested
documentation, not a new GPU run. A fresh agent must not infer permission to
continue training merely from opening a new session.

Until the user explicitly says to continue, do not:

- start diffusion;
- run FoundIR;
- rerun the completed adaptation;
- rerun inference on the consumed fresh test;
- select any checkpoint using consumed-test results;
- use SICE or LOL-v2 Test as a new holdout;
- lower thresholds or relabel the failure;
- call proxy labels human ground truth;
- generate the 100,000-record corpus;
- expand to Blur, Noise, Haze, Rain, or mixed degradations.

Even after the user says `continue`, finish the corrective VLM gate and stop
again for review. Only a credible VLM pass can authorize Lowlight diffusion
conditioning. Only a frozen complete system can authorize official FoundIR.

## Exact next task after explicit user authorization

Execute this sequence in order. Do not skip directly to corrective training.

1. Preserve and document the failed adaptation result and launcher-status
   defect. Never modify or overwrite the completed result directory.
2. Fix only the runner's status normalization first. Add CPU tests proving
   verified failure maps to 2, verified pass maps to 0, and incomplete or
   inconsistent output remains a true error.
3. Run validation-only free-generation diagnostics on existing visual
   checkpoints 25, 50, 100, 150, and 200:
   - evaluate real validation;
   - evaluate frozen synthetic replay validation;
   - do not run the consumed fresh test;
   - do not run FoundIR;
   - write results to new immutable diagnostic paths without overwriting the
     existing selected-checkpoint evaluation.
4. Determine from validation data only whether an earlier checkpoint retains
   synthetic structured-output behavior while still meeting all real
   validation thresholds. This is diagnostic evidence; it cannot retroactively
   turn the consumed-test run into a pass.
5. Design and predeclare a new corrective adaptation contract before training.
   It must use multi-objective checkpoint selection based on real validation
   and synthetic replay, rather than real loss alone. Candidate changes may
   include stronger synthetic replay, lower learning rate, fewer steps, lower
   adaptation strength, partial visual freezing/adapters, or canonical
   structured-output stabilization. Choose using validation evidence and
   scientific reasoning, never consumed-test outcomes.
6. Acquire and freeze a genuinely new real holdout before corrective
   inference. It must be never previously evaluated, source-disjoint, exact-
   and perceptual-overlap screened, and not FoundIR. Freeze the annotation
   rubric, annotation-kind disclosure, manifest, split hashes, thresholds, and
   selection logic before any inference. Human labels are preferred; if proxy
   labels are used, disclose them literally as proxy labels.
7. Run the relevant CPU unit tests, Python compilation, shell syntax checks,
   config/manifest validation, and CPU preflight. Run `git diff --check`.
8. Inspect `git status` and `git diff`; commit and push only the reviewed
   source/config/test/report changes. Verify local HEAD equals `origin/main`.
9. Only then provide the user one short, literal GPU launch action. Verify the
   workspace-local tmux executable first. Provide attach and persistent-log
   commands separately. Use an adaptive two-H100 or eight-H200 configuration
   that preserves the declared global scientific schedule and minimizes total
   wall-clock time; no ceremonial scaling benchmark is required.
10. After the corrective GPU gate, inspect the complete artifacts and report
    pass/fail. Stop for user review even if it passes.

## Files to inspect first for the next authorized implementation

- `agents.md`
- `SESSION_HANDOFF_2026-08-21.md`
- `configs/qwen_lowlight_real_adaptation_step600.json`
- `lowlight_restoration/qwen_real_adaptation.py`
- `smoke/train_qwen_lowlight_real_adaptation_ddp.py`
- `smoke/run_qwen_lowlight_real_adaptation.sh`
- `smoke/launch_qwen_lowlight_real_adaptation_tmux.sh`
- `checkpoints/qwen_lowlight_real_adaptation_step600/training_summary.json`
- `checkpoints/qwen_lowlight_real_adaptation_step600/checkpoint_selection.json`
- `checkpoints/qwen_lowlight_real_adaptation_step600/evaluation/`
- `logs/qwen_lowlight_real_adaptation_step600.log`
- `logs/qwen_lowlight_real_adaptation_step600_events.log`
- `logs/qwen_lowlight_real_adaptation_step600_gpu.csv`

## Reporting language for the next agent

Use these exact distinctions:

- The adaptation **completed training and failed its automatic scientific
  gate**.
- The shell launcher **misreported the intentional status-2 failure as status
  1 because of torchrun wrapping**.
- Real validation passed; the consumed fresh real test missed the per-level
  floor; synthetic replay missed structure, type-accuracy, and MAE safeguards.
- No FoundIR result exists from this adaptation.
- Diffusion is not authorized.
- The next experiment is corrective VLM adaptation with a genuinely new frozen
  holdout, not official FoundIR and not a rerun of the same consumed test.

Never call this a crash, a pass, an official FoundIR result, or evidence that
the complete All-in-One system is ready.

## Continuation update — checkpoint diagnostics and holdout screening

This update supersedes the stale `Exact next task` sequence above where work
has already completed. Commit `48057642a6b6d4d21c90b1945fcfd0903bc83b27`
was pushed to `origin/main`. It fixes torchrun status normalization, adds the
validation-only five-checkpoint diagnostic, provides two/eight-GPU launchers,
and records the failed-gate follow-up. Thirteen focused CPU tests passed.

The diagnostic ran to terminal status 0 on two H100s and independently
verified five checkpoints and 40 evaluation artifacts without FoundIR or the
consumed test. No checkpoint passed all safeguards. All checkpoints passed
real validation but failed the same synthetic structured-output, type
accuracy, and severity-MAE safeguards. The diagnostic root summary SHA-256 is
`aff8f2bb0388e57d23092c1b7a4ad92b533afe6f0043c6672090410a90b25af7`.
Earlier-checkpoint selection therefore cannot rescue the completed run.

Official LOL-v1 was acquired but all 500 scenes exactly overlap protected
LOL-v2/adaptation or prior-evaluation data and are rejected. Preserve both the
valid archive and the corrupt first download; deletion still requires the
two-round confirmation procedure.

The authors' separate official `Testing Images` archive has now been extracted
and screened without model inference:

- archive SHA-256:
  `0033eb74d4c964755f02d4b46e6db3eb457637c116415571fa2d0212577b3f29`;
- 335 image paths, 260 exact-unique images, and 190 internal scene/exposure
  components;
- zero exact or pHash-distance-at-most-4 overlaps with all 13,793 protected
  paths;
- minimum candidate/protected pHash distance: 10;
- candidate-manifest SHA-256:
  `c95b5069ad7756265ed0a3c71411a6cabbc0e877e8a2460122f8e93b293a0537`;
- status: screened but unlabeled; no holdout inference is authorized.

The immediate next task is external human annotation. Use the frozen rubric at
`reports/qwen_lowlight_corrective_holdout_annotation_rubric_2026-08-21.md`,
the blank 190-component CSV at
`manifests/qwen_lowlight_corrective_holdout_annotation.csv`, and the local
visual index under
`data/real_lowlight_corrective_holdout/human_annotation_package_20260821/`.
Two raters work independently and a third adjudicates. After labels return,
implement and run the fail-closed finalizer, freeze the new holdout and
evaluation thresholds, then predeclare the corrective adaptation. Any GPU
training launcher must support the user's current two H100s with the literal
device set `0,1`. Do not run the corrective model, consumed test, diffusion,
or FoundIR before the new holdout is frozen.

## Continuation correction — objective exposure ordering and SOTA execution

The `immediate next task` above is superseded. User inspection exposed that
unrelated photographs cannot receive defensible absolute project L1--L5 labels
without same-scene exposure anchors. Stop the 190-component human severity
annotation. The images remain an unlabeled qualitative robustness pool.

FoundIR-Lowlight is a paired restoration benchmark evaluated by PSNR and
LPIPS; it is not five-level labeled. The project synthetic L1--L5 convention
must not be mistaken for an official benchmark target. Real VLM severity
development now uses objective same-scene exposure ordering, while paired real
restoration improvement is the decisive downstream evidence.

The first-principles strategy is frozen in
`reports/foundir_lowlight_sota_strategy_2026-08-21.md`. Key additions are:

- official FoundIR training-set acquisition and released-model distillation;
- VAE reconstruction and global-photometric ceiling measurements;
- a deterministic degraded-to-clean FLUX/Z-Image bridge rather than pure-noise
  regeneration;
- a high-frequency fidelity bypass and differentiable camera-forward
  consistency;
- baseline/oracle/predicted/shuffled causal conditioning;
- physically realistic RAW/camera synthesis and aligned real pairs;
- native-resolution global-plus-tiled inference and conservative ensembles;
- leave-one-dataset-out paired real development, followed by one frozen
  all-50-pair FoundIR run.

The objective ordinal suite has been built without model inference:

- CSV: `manifests/qwen_lowlight_real_exposure_ordering.csv`
- metadata: `manifests/qwen_lowlight_real_exposure_ordering.json`
- 15 author-provided MEF exposure stacks;
- 67 exact-unique images;
- 210 total pairwise comparisons and 151 pairs separated by at least one EV;
- one component containing `test_out`, `zero`, and `test_0.2` was conservatively
  excluded;
- zero FoundIR use and zero protected overlap inherited from the screened
  source pool;
- labels are explicitly relative exposure order, not absolute L1--L5 ground
  truth.

The two-H100 frozen-step-600 diagnostic is CPU-ready at
`configs/qwen_lowlight_real_exposure_ordering_step600.json`. Its launcher is
`smoke/launch_qwen_lowlight_real_exposure_ordering_tmux.sh` and requires the
literal device set `CUDA_VISIBLE_DEVICES=0,1`. Six focused tests and the full
100-test CPU suite passed; compilation, shell syntax, workspace-local tmux,
manifest regeneration, full visual-checkpoint hashing, and `git diff --check`
passed. No Qwen inference has yet run on these stacks. Commit and push the
readiness milestone before giving the GPU launch command.

## Continuation update — third-pass design and Gate A execution

The first-principles strategy received another challenge pass. New frozen
ideas are a non-regressing proposal lattice, teacher-disagreement uncertainty,
target-render-distribution modeling, counterfactual conditioning losses,
worst-dataset robust selection, and bounded low-dimensional input-only
adaptation. The authoritative strategy is still
`reports/foundir_lowlight_sota_strategy_2026-08-21.md`.

The released FoundIR assets were identity-audited without inference:

- generalist: 580,829,530 bytes, stored step 1,300,000, SHA-256
  `2603a0be47d0844d4c9cea542ecca039e0e4aaa39d45d41064bfbdc13c0830c9`;
- Lowlight specialist: 6,294,432 bytes, iteration 60,000 / epoch 413,
  SHA-256
  `f834248e145d020d5a02f257b7ceb383b579add2522938b3e1e0bd460abe6`;
- machine audit: `manifests/foundir_released_teacher_audit.json`.

The reusable paired-real development suite is complete:

- CSV SHA-256:
  `fee42cbf440a62c494bccc49e6f25b871ce8ea0044bcd825472e538dc3f5300d`;
- 174 pairs / 82 scene components;
- LOL-v2 real-adaptation validation: 102 records / 41 components;
- SICE reusable validation: 72 records / 41 components;
- no L1 records, no consumed real-test records, and zero exact FoundIR overlap;
- levels are retained only as proxy metadata.

The CPU global-photometric ceiling completed. The target-fitted gamma plus 3
by 3 color matrix changed scene-balanced PSNR `8.4356 -> 24.1209 dB` and SSIM
`0.2475 -> 0.6571`. PSNR gain is `+15.6853 dB`, 95% scene-bootstrap CI
`+14.7717` to `+16.6036`. Seven SICE records exceeded the 8-pixel alignment
diagnostic; none were warped or excluded. This is a GT-accessing oracle only,
never a deployable inference method. Exact result:
`reports/lowlight_real_photometric_ceiling_result_2026-08-21.md`.

The immediate GPU gate is now the FLUX.2 VAE representation ceiling:

- config: `configs/lowlight_vae_ceiling_v1.json`;
- config SHA-256:
  `04d8c261633635dc57a7c79d0b109035e03801059fb2a18cf5899088d8db96a3`;
- 130 exact-unique targets, two H100/H200 GPUs, literal device set `0,1`;
- direct center-crop and full-resolution tiled VAE round trips;
- PSNR, SSIM, resized LPIPS, high-frequency retention, and seam metrics;
- immutable per-target resume sidecars and fail-closed result verification;
- no FoundIR access.

After all source/config/test/report changes are committed and pushed, the one
launch action is:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash smoke/launch_lowlight_vae_ceiling_tmux.sh
```

Inspect the result before preparing teacher inference or implementing the
pixel bypass. Do not run FoundIR. The separate Qwen exposure-ordering
diagnostic remains ready and unrun, but representation Gate A has priority
because it now directly determines the restoration architecture.
