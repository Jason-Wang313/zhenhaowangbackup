# Authoritative fresh-session handoff — Lowlight Phase 1

Date: 2026-08-16 UTC
Workspace: `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`
Branch: `main`

This file is the authoritative handoff for the next coding-agent session. It
supersedes the now-stale
`reports/lowlight_context_handoff_2026-08-15.md`. Do not resume from that older
file's pending-work section; all of its baseline work has since completed.

## Mandatory first actions

1. Work only in:
   `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`.
2. Before using tools, tell the user exactly:

   > I am working in zhenhaowang, which is the correct workspace.

3. Read `agents.md` completely. Its scope, workspace boundary, and two-round
   confirmation rule for destructive or overwriting changes are mandatory.
4. Read this entire handoff before proposing or running anything.
5. Do not use subagents unless the user explicitly asks for delegation.
6. Recheck `git status`, local HEAD, remote HEAD, and current runtime artifacts;
   do not assume that the state below remains unchanged after this handoff.

## Current repository state at handoff

- final implementation HEAD immediately before committing this handoff:
  `cd09eae25190ac3dcc66ef4d61e6e3a63ae75572`
- `origin/main` at the same point:
  `cd09eae25190ac3dcc66ef4d61e6e3a63ae75572`
- latest commit: `Add expanded gated Qwen ridge validation`
- previous commit: `af6baa8` (`Document Qwen ridge validation failure`)
- branch: `main`
- tracked implementation work is committed and pushed

This handoff is committed afterward, so the actual HEAD visible to the new
session should be one documentation commit later. Recheck local and remote HEAD
rather than treating `cd09eae` as the expected final documentation commit.

The following unrelated historical files are still untracked. They belong to
the existing workspace state. Preserve them and do not stage, modify, delete,
or claim them unless separately required:

```text
manifests/low_light_pilot_smoke_500.csv
manifests/low_light_pilot_smoke_500.json
manifests/low_light_source_pool.json
manifests/source_pool_mio_div2k_partial.json
reports/model_audit_stage0.md
smoke/calibrate_agenticir_noise_levels.py
smoke/calibrate_agenticir_selected_002.py
smoke/download_flux2_chunks.py
smoke/download_qwen_part1.sh
smoke/monitor_qwen_download.sh
smoke/run_agenticir_noise_levels.py
smoke/run_agenticir_selected_blur.py
smoke/run_agenticir_selected_haze.py
smoke/run_agenticir_selected_low_light.py
smoke/run_agenticir_selected_noise.py
smoke/run_agenticir_selected_rain.py
smoke/watch_flux2_then_qwen.sh
```

## Hard scope boundary

Work on **Lowlight only**.

Do not start or claim work on:

- Blur
- Noise
- Haze
- Rain
- the 100,000-record multimodal dataset
- the separate `亮色大模型` workstream

The generic safeguards know all five degradation names, but the four later
tasks remain `blocked_no_task_calibration`. Do not generate, train, or evaluate
them from this handoff.

Do not use the internal test split or official FoundIR as a tuning split. Do
not claim Phase 1 acceptance from internal synthetic metrics. The official
FoundIR-Lowlight Phase 1 target remains PSNR `>=15.1 dB` and LPIPS `<=0.34` for
the complete restoration system.

## Environment and execution model

The coding-agent terminal is CPU-only and has internet access. The user runs
GPU work manually in a separate offline NVIDIA H100 80 GB terminal. Both mount
this same workspace, but processes, GPU visibility, shell variables, and
packages outside workspace-local environments are not shared.

Use the workspace interpreter:

```bash
.venv/flux2-smoke/bin/python
```

Never claim a GPU run succeeded unless the user supplies its terminal output
or the shared filesystem contains verifiable logs, metrics, and checkpoints.
Prepare one exact offline launcher command at a time. Runtime outputs are
fail-closed and provenance-bound; preserve failed evidence rather than
overwriting it.

A workspace-local tmux 3.7 binary exists at `.tools/tmux-env/bin/tmux`. At
this handoff it reports no server and no sessions. If browser-disconnection
survival is needed, create a new tmux session in the GPU terminal and launch
the command inside it. Never claim an old session or process was resumed
without verifying the process and checkpoint/log state.

## Authoritative Lowlight data

Manifest: `manifests/low_light_pilot_10000.csv`

- 2,000 source-disjoint clean sources
- 10,000 aligned records
- train: 1,600 sources / 8,000 records
- validation: 200 sources / 1,000 records
- internal test: 200 sources / 1,000 records
- exactly five records per source, one each at L1-L5
- source datasets: MiO100 100, DIV2K train 800, LSDIR train 1,100
- L1 is the clean source
- L2-L5 use the frozen calibrated AgenticIR low-light recipe
- official FoundIR has not been used

## Completed FLUX.2 baseline — internal pass

The non-Qwen FLUX.2 Klein Base 4B restoration baseline is complete on the
synthetic internal validation and internal-test splits.

Authoritative report:
`reports/lowlight_full_baseline_result_2026-08-16.md`

Checkpoint:
`checkpoints/lowlight_baseline_full/step_004000.safetensors`

Important evidence:

- checkpoint SHA-256:
  `fa51deac872e3df57f147631361841039c386ef43f58a17a56fb68b9924e8f22`
- frozen training config SHA-256:
  `c71235efe88e8f3bc98ca863350af372a78aa35f1a2cc065a1867c5bb6bfde18`
- training-summary SHA-256:
  `4581852a5ed35a1654f239040cd050d5ee40b462a22b57215c50f8afb3cd538b`
- validation metrics SHA-256:
  `b1b78a9afa5174786702ac17fc2db44524ba374a2264c42067ba68fa3ff08e51`
- internal-test metrics SHA-256:
  `6bac93cb0a45556787364c553242d15463b2ea0a5a18d5738054b2a87e1a1da4`
- validation L2-L5: PSNR `24.9051`, SSIM `0.8913`, LPIPS `0.0411`
- internal-test L2-L5: PSNR `24.7723`, SSIM `0.8858`, LPIPS `0.0440`
- validation L1: PSNR `31.4672`, SSIM `0.9317`, LPIPS `0.0172`
- internal-test L1: PSNR `31.6674`, SSIM `0.9266`, LPIPS `0.0185`
- automatic gates passed
- human scene-identity review passed

These are strong **synthetic internal** results. They are not official FoundIR
results and must not be compared numerically as if they came from the official
FoundIR split.

Known baseline failure cases are intentionally retained, including extremely
dark or nearly uniform crops such as validation `div2k_0452_L5` at about
`3.01 dB`, plus some L1 tonal/texture changes. Preserve these in the eventual
fair baseline-versus-Qwen comparison.

## Frozen Qwen model

Model: `Qwen/Qwen2.5-VL-7B-Instruct`

- local path: `models/qwen2.5-vl-7b-instruct`
- revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- architecture: `Qwen2_5_VLForConditionalGeneration`
- parameters: `8,292,166,656`
- dtype: bfloat16
- attention: SDPA
- license: Apache-2.0
- all 16 local model files are hash-bound in
  `configs/qwen_lowlight_smoke25.json`

Target structured output for Lowlight remains exactly:

```json
{"degradations":["low_light"],"levels":{"low_light":3}}
```

The standalone Qwen assessor predicts type/severity. It does not itself
produce restored images or FoundIR PSNR/LPIPS.

## Qwen severity experiment history — preserve every failure

### 1. Zero-shot rubric: failed

- severity accuracy: `0.36`
- MAE: `0.72`
- valid structured output: `1.0`
- L5 recall: `0.0`; all five L5 images were predicted as L4
- degradation type accuracy: `1.0`
- this legacy smoke did read 25 records from the manifest's internal `test`
  split; those records and their labels are exposed and must not be used for
  later assessor tuning
- artifact: `checkpoints/qwen_lowlight_smoke25/metrics.json`
- artifact SHA-256:
  `8852884ad27afdb5b82b14e7683b97a5b616a0f3f7a2288f1c11b0d4f9afbdad`

Audit conclusion: the image/label/path/processor wiring was correct. The
failure came from an uncalibrated project-specific severity scale, not a
reversed label or missing-image bug.

### 2. Visual few-shot calibration: failed

- accuracy: `0.44`
- MAE: `1.04`
- L1 recall: `1.0`
- L5 recall: `1.0`
- L2-L4 collapsed heavily toward L5
- artifact:
  `checkpoints/qwen_lowlight_fewshot_validation25/metrics.json`
- artifact SHA-256:
  `9ed698200d5247eb58106758c91bef86f82d4d82ffb26f385ac63fef14058b16`
- report: `reports/qwen_lowlight_fewshot_failure_2026-08-16.md`

The original 25 validation sources used here are exposed.

### 3. Frozen-Qwen CORAL linear ordinal head: selection failed

- 240 train sources / 1,200 records
- 60 train-only selection sources / 300 records
- 7,168-dimensional pooled frozen-Qwen features
- 28,676 fitted parameters
- selection accuracy: `0.6667`
- selection MAE: `0.3967`
- luminance baseline: accuracy `0.6933`, MAE `0.3467`
- per-level recall gate passed, but both baseline-relative gates failed
- fixed validation was not touched by this experiment
- artifact:
  `checkpoints/qwen_lowlight_ordinal_head/selection_metrics.json`
- artifact SHA-256:
  `1e6fdd301326846be74b7c208c02dfd8b47dd080bae0a7bcc91a85f952d43b17`
- report: `reports/qwen_lowlight_ordinal_linear_failure_2026-08-16.md`

### 4. Frozen-Qwen scalar ridge head: selection passed, validation failed

Train-only selection:

- 7,169 fitted parameters
- alpha: `1000`
- accuracy: `0.74`
- MAE: `0.2633333`
- recall L1-L5: `0.90, 0.65, 0.70, 0.5667, 0.8833`
- luminance baseline: accuracy `0.6933`, MAE `0.3467`
- all selection gates passed

Fixed 25-source validation:

- accuracy: `0.48`
- MAE: `0.52`
- recall L1-L5: `0.80, 0.40, 0.60, 0.00, 0.60`
- validation baseline: accuracy `0.64`, MAE `0.40`
- all five output levels were represented
- all five true L4 samples missed: four became L3, one became L5
- internal test used: `false`
- official FoundIR used: `false`

Artifacts:

- metrics:
  `checkpoints/qwen_lowlight_ridge_head/metrics.json`
- metrics SHA-256:
  `4baf465aab731a845249a3bdbbdbec7e642439bae892686a0a7f5b09f6c5b0fd`
- selection metrics SHA-256:
  `c9c90778dae9af1e6e746b477fc8562417b4dc7d06052b8ab97b56dfd5863821`
- checkpoint SHA-256:
  `643ba5ab047b7930c3a1eaef002e1d62300ac66da69f8fdfbc534f2570ac3a93`
- log SHA-256:
  `a3cf05eb47f442524acbf99593f06eeec220942e9f7094d9e3f854f5c05813db`
- report: `reports/qwen_lowlight_ridge_failure_2026-08-16.md`

Meticulous diagnosis found no path, label, image-hash, source-isolation,
cache-provenance, or structured-output bug. The problem was high-dimensional
overfit plus an optimistic single selection split:

- input dimensions: `7,168`
- fitted examples: `1,200`
- fitted-set accuracy: `0.9392`
- five-fold source-grouped accuracy on 300 cached train sources (1,500
  records):
  `0.6767 +/- 0.0287`
- matching grouped luminance baseline accuracy: `0.6467`
- train-only threshold calibration improved grouped accuracy only slightly to
  about `0.6833`, so changing rounding boundaries was not a sufficient repair

The 25 validation source IDs in the ridge metrics artifact are now exposed.
They can support diagnosis only and must never again be presented as an
unbiased gate for a revised head.

## Immediate active experiment — prepared but NOT STARTED

The expanded frozen-Qwen ridge pipeline is implemented, CPU-verified,
committed, and pushed. At this handoff there is **no** directory
`checkpoints/qwen_lowlight_ridge_expanded_head` and **no** log
`logs/qwen_lowlight_ridge_expanded_head.log`. Therefore no expanded GPU feature
extraction, selection, checkpoint, or validation result exists yet.

Readiness report:
`reports/qwen_lowlight_ridge_expanded_readiness_2026-08-16.md`

Frozen config:
`configs/qwen_lowlight_ridge_expanded_head.json`

Config SHA-256:
`b04bf893e073cf50423a384456398b12ce92625a979453c8f8c61e210f8bd33c`

Design:

- fit: 1,280 train sources / 6,400 records
- train-only selection: 320 sources / 1,600 records
- exact dataset-stratified source quotas
- fit bundle SHA-256:
  `721eb7694fae0ba9222c4ef38729948358de63122ab9ce15cdb39dad802c4780`
- selection bundle SHA-256:
  `75fb05afe9a273c6d9d8b19c62760df58ca89438dbb34fb4ef39170a419ce944`
- frozen ridge alphas: `300, 1000, 3000, 10000, 30000`
- fixed nearest-integer L1-L5 decoder
- train-selection luminance baseline: accuracy `0.629375`, MAE `0.465`
- selection gate requires:
  - accuracy at least `0.60`
  - every output level represented
  - every per-level recall at least `0.40`
  - accuracy at least the train-only luminance baseline
  - MAE no worse than the train-only luminance baseline
- only a passing train-selection candidate may open new validation images

New validation gate:

- exactly 50 previously unused validation sources
- all five levels per source: 250 records
- excludes every one of the 25 exposed validation sources
- DIV2K 22, LSDIR 27, MiO100 1
- exact IDs are frozen in the config
- validation bundle SHA-256:
  `c8e017fd19b9527610890e3976ceb536f0ae6ff5ebc9680094bc0538542e00ea`
- internal test and FoundIR remain untouched

Feature extraction:

- new cache:
  `checkpoints/qwen_lowlight_ridge_expanded_head/features`
- the prior 1,525 cached feature files are not overwritten or relabeled
- new feature cache schema: 2
- visual extraction batch size: 20
- first run needs 8,000 train features
- after selection passes, it needs 250 validation features
- each completed feature file is atomic and resumable after interruption

Verification already completed:

- expanded CPU preflight: pass
- all 16 Qwen model files verified
- zero expanded train features existed at preflight
- zero new validation images opened
- all 25 exposed sources excluded
- all 48 repository unit tests passed
- Python compilation passed
- launcher shell syntax passed
- `git diff --check` passed
- train-only batch-20 processor check passed with 20 image grids

## Exact next action

First, check whether the user ran the expanded job after this handoff:

```bash
test -f checkpoints/qwen_lowlight_ridge_expanded_head/metrics.json && \
  jq '{status,selected_alpha,metrics,gate,test_split_used,official_foundir_used}' \
  checkpoints/qwen_lowlight_ridge_expanded_head/metrics.json
test -f logs/qwen_lowlight_ridge_expanded_head.log && \
  tail -80 logs/qwen_lowlight_ridge_expanded_head.log
ps -eo pid,args | \
  rg '[t]rain_qwen_lowlight_ridge_expanded_head.py|[r]un_qwen_lowlight_ridge_expanded_head.sh'
```

If neither artifact exists and no process is active, give the user exactly this
offline H100 command:

```bash
CUDA_VISIBLE_DEVICES=0 smoke/run_qwen_lowlight_ridge_expanded_head.sh
```

Expected first-run duration is roughly 20-40 minutes on the H100. The prior
batch-5 cache wrote 1,500 features in about 288 seconds; batch 20 is intended
to offset the larger 8,000-record extraction. CPU ridge fitting may add several
minutes.

Expected live markers:

1. `QWEN_LOWLIGHT_EXPANDED_RIDGE_PREFLIGHT_PASS`
2. `QWEN_EXPANDED_RIDGE_CUDA_READY`
3. `QWEN_ORDINAL_FEATURE_PROGRESS .../8000`
4. five `QWEN_EXPANDED_RIDGE_CANDIDATE` lines
5. either `QWEN_EXPANDED_RIDGE_SELECTION_FAILED`, or a passing selection
6. only after selection passes: `QWEN_ORDINAL_FEATURE_PROGRESS .../250`
7. final `QWEN_LOWLIGHT_EXPANDED_RIDGE_VALIDATION_PASS` or
   `QWEN_LOWLIGHT_EXPANDED_RIDGE_VALIDATION_FAILED`

The launcher logs to:
`logs/qwen_lowlight_ridge_expanded_head.log`.

Do not mistake an extraction counter reaching 8,000 for an experiment pass.
The selection gate and, if reached, final validation gate must both be read.

## How to handle the expanded outcome

### If train-only selection fails

- Preserve `selection_metrics.json` and the feature cache.
- Confirm that no new validation features were created.
- Document the failed train-only result.
- Do not loosen gates or inspect/tune against the frozen 250 validation images.
- The next scientific decision is whether to use the frozen features only as a
  continuous conditioning signal or escalate to a bounded Qwen vision LoRA.

### If selection passes but validation fails

- Preserve the checkpoint, per-record results, metrics, and log.
- The 250 validation records are then exposed and cannot be reused as a fresh
  unbiased gate for another tuned head.
- Do not post-hoc change thresholds, alpha, source selection, or pooling and
  claim the same 250 records as validation.
- Diagnose using the failed records only after recording hashes and a failure
  report.
- Prefer a bounded Qwen vision-LoRA decision over an endless sequence of
  linear heads.

### If both selection and validation pass

- Verify all artifact hashes, source exclusions, all five recalls, baseline
  comparisons, `test_split_used=false`, and `official_foundir_used=false`.
- Add a result report, run relevant tests, commit, push, and verify remote SHA.
- Freeze the assessor checkpoint and prediction interface.
- Then implement the fair restoration-conditioning experiment using the same
  FLUX.2 backbone, LoRA/restoration path, data, source split, optimizer,
  learning rate, steps, batch size, precision, seeds, GPU budget, and sampling
  parameters as the non-Qwen baseline. The intended difference is only the
  presence of Qwen type/severity conditioning.
- Keep a ground-truth-level condition run as an explicitly internal oracle
  diagnostic, never as the reported Qwen result.
- Do not claim Qwen improvement until restored-image PSNR/SSIM/LPIPS and visual
  identity are compared under the fair protocol.

## Scientific question that remains unanswered

The project must answer:

> With the same diffusion restoration model, data, source split,
> preprocessing, training budget, and evaluation conditions, does adding
> Qwen-VL degradation type and severity information improve restoration
> quality?

The completed FLUX.2 baseline answers only the non-Qwen half. None of the
standalone Qwen severity experiments has yet produced a validated conditioning
signal, and no fair baseline-versus-Qwen restoration comparison has run.

Required systems remain:

```text
Baseline: degraded image -> FLUX.2 restoration model -> restored image

Qwen: degraded image -> Qwen-VL type/severity -> condition adapter
      -> the same FLUX.2 restoration model -> restored image
```

Do not compare different restoration backbones, restore twice, alter the data
budget, or add captions/aesthetic/object descriptions in the first ablation.

## Official FoundIR status

No official FoundIR-Lowlight metric has been reported. Internal synthetic
metrics are not interchangeable with FoundIR metrics. The final Lowlight Phase
1 sub-target requires the complete restoration pipeline on the official
FoundIR-Lowlight evaluation split with:

- PSNR `>=15.1 dB`
- LPIPS `<=0.34`
- exact manifest, checkpoint, config, code revision, command, per-image output,
  and metric JSON/CSV retained

A Lowlight pass may be called `FoundIR-Lowlight Phase 1 sub-target passed`. It
must not be called complete five-task Phase 1 acceptance.

## Communication rules for the next agent

- Lead with the actual pass/fail state; never soften a failed gate into
  “promising.”
- Distinguish CPU readiness, GPU execution, internal validation, internal test,
  and official FoundIR evidence.
- Give one exact copy-paste GPU command at a time.
- State realistic time estimates and whether progress is resumable.
- Preserve hashes and failed evidence.
- Do not tell the user a stopped process was resumed unless a real checkpoint
  resume was verified.
- Do not claim a detached terminal session exists without checking it.
- Ignore the repeated `pkg_resources is deprecated` warning; it is noisy but
  not the cause of the model failures.

## Concise current truth

The synthetic FLUX.2 Lowlight baseline is strong and complete internally. The
standalone Qwen severity assessor is not yet validated. Four Qwen approaches
have failed at different gates. A fifth, expanded frozen-feature ridge attempt
is carefully prepared and leakage-safe, but has not started on the H100. The
single next task is to run or inspect that expanded attempt—nothing else.
