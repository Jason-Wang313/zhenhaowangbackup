# Session handoff — 2026-09-07

This is the current, self-contained continuation guide for a fresh coding
agent. It supersedes older handoffs for **operational status and the immediate
next task**, but it does not override the advisor-authoritative architecture,
workspace restrictions, destructive-change confirmation rule, or FoundIR-test
protection policy in `agents.md`.

## Continuation update — 2026-09-07 after first FLUX full-run failure

This section supersedes the earlier statement below that no full training had
started. The H200 pipeline launched at `2026-09-07T12:04:43Z`, passed every
completed CPU/Qwen/prompt/evaluation-contract gate, and printed step 1. It then
failed at `2026-09-07T12:20:15Z`; no full-training checkpoint was written.

The primary failure was not NCCL or a bad GPU. Rank 6's DataLoader selected
`foundir_train_0395700`, an aligned `1920x494` Rain LQ/GT pair that is valid for
the earlier 448 Qwen crop but shorter than the 512 FLUX training crop. Rank 6
raised the data error before the next backward pass, while the other seven
ranks waited in DDP gradient all-reduce until the 600-second NCCL watchdog
expired. GPU 6's persistent 0% utilization was therefore a real stalled-rank
signal. The complete manifest contains 37 such short-side-under-512 Rain pairs:
35 train, one validation, and one internal-test record.

The correction and independent speed-benchmark route are implemented. The
trainer deterministically applies the same aspect-ratio-preserving bicubic
resize to LQ and GT only for those undersized pairs, then uses the existing
aligned deterministic crop. A preflight scans all 685,495 manifest geometries
and opens all 37 affected LQ/GT files before model loading. Per-step batch
loading is synchronized across ranks before DDP forward/backward, DataLoader
wait is bounded, and NCCL desynchronization diagnostics are enabled. The actual
preflight passed and bound the affected record set by SHA-256
`a0dc992fb2fa4c9a7191ac01a81584d5bca32bc4b035a1cb14bb4cee4576de8c`.

The speed benchmark writes no checkpoint and does not access the official
FoundIR test. It compares, on H200, `2x2/w4`, `4x1/w4`, and `4x1/w8`; on H100,
`1x4/w4`, `2x2/w4`, and `2x2/w8`. Each runs 20 real optimizer steps, excludes
five warmup steps, records loss/throughput/data wait/peak HBM, and preserves
global batch 32 plus the full scientific schedule. Per-presentation noise and
timestep draws are now independent of microbatch grouping. Automatic selection
requires finite loss, at most 5% mean trajectory deviation, and at least 5%
speedup. A selected schedule is bound to the exact config, Git revision, and
GPU family before the full trainer accepts it.

The exact next task is to commit/push the validated fix and launch
`smoke/launch_foundir_flux2_speed_benchmark_8gpu.sh` on the user's independent
eight-H100/H200 instance. Do not restart full training until the benchmark
selection is inspected. The official FoundIR test remains protected.

## Continuation update — 2026-09-07 11:53 UTC

This section supersedes the older executive status, immediate blocker, and
exact-next-task sections later in this file. Preserve those sections as the
diagnostic history of the failure; do not treat them as current status.

The evaluation-contract blocker is fixed. Commit `f7df46b` implements a
deterministic, category-aware `source_group` fold assignment that balances
group diversity before record volume, then selects one record per available
group before deterministic group round-robin reuse. It preserves split
separation, keeps every group in exactly one evaluation fold, never lowers the
20-record quota, and leaves the frozen v3 config unchanged.

The evaluation contract is now frozen and tracked:

```text
manifests/foundir_five_type_evaluation_contract_v3.csv
records: 1,320
SHA-256: f650e7b3f83b2e6447f5b3ff1707c3fb13946fe78d48f87c0620a7b4e75b5e8b

manifests/foundir_five_type_evaluation_contract_v3.json
SHA-256: 076d239e6d8858a980b30302fe5d55844b2c1ce55b6eed6defb4436e756e10b2
schema_version: 2
selection_policy: balanced_group_fold_assignment_then_group_diversity_round_robin_v1
```

Verified contract properties:

- development: 660 records;
- sealed internal test: 660 records;
- all 66 role/category/fold buckets contain exactly 20 records;
- all 1,320 record IDs are unique;
- no `source_group` crosses evaluation folds;
- development and sealed-internal groups are disjoint;
- per-bucket record and unique-group counts are recorded in the JSON;
- protected official FoundIR test access remains `false`.

Validation completed after the fix:

- Python compilation and shell syntax: pass;
- focused unit tests: 38 passed;
- actual 685,495-row condition-contract freeze: pass;
- independent contract hash/cardinality/isolation verification: pass;
- complete launcher check marker:
  `FOUNDIR_FIVE_TYPE_MAIN_V3_LAUNCHER_CHECK_PASS`.

No FLUX full-curriculum process is running, and no training summary or
checkpoint selection exists. Do not reopen or regenerate Qwen training,
condition export, partition export, prompt caching, or the evaluation
contract.

The final CPU-side repository step is to synchronize the validated commits to
`origin/main`. Once local `HEAD` and `origin/main` match, the exact next task is
to use one homogeneous eight-GPU H100 or H200 instance to run the existing v3
launcher without `--check`. The launcher must verify and reuse the frozen
condition CSV, prompt cache, and evaluation contract before beginning FLUX
full-curriculum training. Monitor from the newest
`FOUNDIR_FIVE_TYPE_MAIN_V3_START` marker and fail closed on any error. The
official FoundIR test remains protected.

## Mandatory fresh-session startup

1. Work only in:
   `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`
2. Tell the user exactly:
   `I am working in zhenhaowang, which is the correct workspace.`
3. Read `agents.md` completely.
4. Read this file completely.
5. Use `SESSION_HANDOFF_2026-08-21.md` and older handoffs only for deeper
   historical context. Do not let their stale operational status override this
   file.
6. Before edits, run:

   ```bash
   git status --short --branch
   git rev-parse HEAD
   git rev-parse origin/main
   ```

7. Preserve all unrelated tracked modifications and untracked artifacts. The
   working tree is intentionally dirty. Never stage everything with `git add
   -A`.
8. Obey the current destructive-action rule in `agents.md`: ordinary edits do
   not require confirmation; genuinely destructive actions that risk
   difficult-to-recover loss require one exact-scope confirmation.

## Executive status at handoff

Handoff capture time:

- Beijing: `2026-09-07 18:57 CST`
- UTC: `2026-09-07 10:57 UTC`

Git:

- branch: `main`
- HEAD: `8a905f2d0ec07cf29929ade62188f51a38234f42`
- `origin/main`: same revision
- latest commit: `Handle undersized FoundIR Qwen inputs`

Current pipeline state:

- The frozen Qwen condition export is **complete** for all `685,495` approved
  FoundIR-training records.
- The FLUX-native Qwen3 prompt cache is **complete** for all `382` unique
  canonical restoration prompts.
- The v3 main pipeline is **not running**. It exited with status `1` at
  `2026-09-07T10:36:57Z` before FLUX full training began.
- Immediate blocker:

  ```text
  RuntimeError: insufficient validation groups for fold=0 category=02Blur_Noise: 5
  ```

- No full-curriculum training summary exists.
- No full-curriculum checkpoint selection exists.
- No evaluation-contract CSV/JSON exists, so there is no partial contract to
  delete or overwrite.
- The protected official FoundIR test has not been accessed.

Product goal state:

- The thread still shows a **paused, stale** goal about accelerating the
  685,495-record Qwen condition export.
- That export objective has already been achieved and must **not** be resumed
  or repeated. The operational next task is the evaluation-contract sampler
  correction described below.
- At handoff verification, no `foundir_five_type_flux2_main_v3` or FLUX
  full-training process was running.

Do **not** rerun Qwen training, Qwen condition export, partition export, or
prompt caching. Those expensive stages are complete and resume-safe.

## Project scope and authoritative decisions

The user owns only the `VLM Guided All-in-One` restoration workstream. The
separate `亮色大模型` workstream is out of scope.

Advisor-confirmed first-version FoundIR scope:

- blur;
- noise;
- rain;
- haze;
- low-light;
- their supported single, double, and triple combinations.

Hard exclusions and user decisions:

- JPEG is excluded from the first version. Never mix JPEG records into main
  training, validation, or acceptance gates without new explicit advisor
  authorization.
- Raindrop/NightRain and other noncanonical categories remain excluded unless
  explicitly mapped and authorized.
- Do not train or adapt Qwen using FoundIR images. The advisor explicitly said
  `不能用foundir训qwen`. FoundIR training images may be used for frozen-Qwen
  inference diagnostics and for training the restoration model, not for Qwen
  parameter updates.
- Qwen-condition oracle/generic/shuffled/wrong ablations are disabled by the
  user for the current mainline. Do not add them unless the user explicitly
  reauthorizes them.
- Do not add or report high-frequency-correlation gates. The SOW restoration
  metrics are PSNR and LPIPS.
- Do not access the official FoundIR test during development, tuning,
  checkpoint selection, or debugging. It is reserved for one frozen final
  evaluation after the complete system and protocol are locked.
- The released FoundIR checkpoint may be evaluated as a restoration baseline.
  That comparison is not a Qwen-condition ablation. Training-only distillation
  remains optional and is not part of the current first run.

## Frozen Qwen model — phase complete

Architecture and inference:

- base: `Qwen2.5-VL-7B-Instruct`;
- method: ask five independent questions per image, one each for blur, noise,
  rain, haze, and low-light;
- each answer is `L1`–`L5`, where `L1` means absent;
- the five decoder answers are aggregated into structured JSON and then fixed
  canonical restoration text;
- inference uses a deterministic 448×448 center view and greedy decoding;
- no five-head classifier was added;
- Qwen is frozen for the restoration phase.

Training setup:

- visual LoRA rank/alpha: `64/64`;
- language LoRA rank/alpha: `64/64`;
- trainable visual LoRA + language LoRA + visual merger;
- base weights, LM head, tokenizer, and visual patch embedding frozen;
- trainable parameters with merger: `129,550,080`;
- synthetic images: `100,000` train, `10,000` validation, `10,000` test;
- five dialogue presentations per image;
- global batch: `200`;
- total optimizer steps: `2,500`;
- synthetic degradations use the advisor-approved physical-v8 recipe and
  single/double/triple mixtures.

Frozen adapter:

```text
checkpoints/qwen_multidegradation_100k_rank64_per_type_physical_v8/
  merger_trainable/checkpoints/V8_physical_shard5/adapter.safetensors
SHA-256: 5416311d7cd0aa0308e206084c13c7c950f9dacfc96cd558466642804ccbee3c
```

Training config:

```text
configs/qwen_multidegradation_100k_rank64_per_type_physical_v8.json
SHA-256: eef9fe05e21a2fd5b92543f17473e96cb1723b1d4f800ef13a2f6e1622168aeb
```

Frozen synthetic validation result:

```text
checkpoints/qwen_multidegradation_100k_rank64_per_type_physical_v8/
  merger_trainable/validation_result.json
SHA-256: e1ed20381f8f9e57bd552c278999ec6d2e373fe43f9337bab6fa6af6151d75a8
status: passed
valid output: 100%
type micro-F1: 96.4748%
type macro-F1: 96.4381%
type-set exact: 87.83%
all five levels accuracy: 94.35%
active L2-L5 level accuracy: 87.4167%
single type-set exact: 92.9333%
double type-set exact: 85.5667%
triple type-set exact: 82.10%
combined type+level exact overall: 76.14%
combined type+level exact triple: 72.10%
```

Frozen FoundIR-training-only real-domain diagnostic:

```text
reports/foundir_five_type_qwen_real_gate_physical_v8.json
SHA-256: a07d31d659c009be22f555c8cd73beda21a6506a782c799561e266163e91ae09
records: 128
valid output: 100%
type micro-F1: 82.8431%
type macro-F1: 84.0531%
type-set exact: 52.3438%
blur recall: 67.3913%
noise recall: 54.3478%
haze recall: 91.6667%
rain recall: 100%
low-light recall: 100%
```

This diagnostic did not pass every originally proposed real-domain gate, but
the user explicitly authorized moving to the DiT phase after advisor review.
Do not misreport it as a pass and do not use it to retrain Qwen.

## Frozen FoundIR five-type development corpus

Training root:

```text
data/foundir/MillionIRData/Train
```

Manifest:

```text
manifests/foundir_five_type_pairs_v1.csv
records: 685,495
SHA-256: e8c018879029c0bcc2da8d074cc4999541158ca97d07f48aadbcdddcdccc3ca7
```

Splits:

- train: `654,881`;
- validation: `16,461`;
- sealed internal test: `14,153`.

The split policy groups exact/perceptually related GT sources before assigning
splits. Do not move records between splits or use sealed internal records for
iterative checkpoint tuning.

Included category counts:

| Category | Records |
|---|---:|
| 01Blur | 109,480 |
| 02Blur_Noise | 29,950 |
| 05Noise | 58,015 |
| 08Haze | 79,800 |
| 09Lowlight_Haze | 79,800 |
| 10Rain | 39,900 |
| 13Rain_Haze | 79,750 |
| 14Lowlight | 39,962 |
| 15Lowlight_Blur | 85,893 |
| 16Lowlight_Noise | 52,995 |
| 18Lowlight_Blur_Noise | 29,950 |

## Completed full Qwen condition export

All `1,339/1,339` immutable chunks were independently verified, covering
`685,495/685,495` records. Existing chunks must never be regenerated merely
because a later stage fails.

Final files:

```text
manifests/foundir_five_type_all_qwen_conditions_v3.csv
size: 774,491,758 bytes
SHA-256: e07e16234d4d1eb674b5d667732600628448b215e9fab67e6e6732f99c1ea2f7

manifests/foundir_five_type_all_qwen_conditions_v3.json
SHA-256: 8e39ce6c0a1078b0af8c510ab7acb06b4ed53392ae21fa3e722de5e01a78e96e

checkpoints/foundir_five_type_flux2_dit_v3/qwen_condition_chunks/
chunks: 1,339
records: 685,495
```

Export status marker:

```text
FOUNDIR_PHYSICAL_V8_FULL_EXPORT_PASS
```

The export uses the unchanged five-question legacy inference semantics. A
large-batch record-sharded optimization was benchmarked earlier, was slower,
failed exact-output equivalence, and was rejected. Never reactivate it.

### Undersized-image failure and committed fix

The first partition run stopped on 19 narrow Rain images whose shorter side
was below the required 448-pixel crop, for example `1534×406`. Commit
`8a905f2` added:

- deterministic aspect-ratio-preserving bicubic resize of only undersized
  inputs until the short side reaches 448;
- the unchanged exact center-crop path for every normal image;
- a complete 685,495-row geometry preflight;
- chunk-identity binding for the fallback policy and affected record IDs;
- regression tests and actual-file dimension validation.

Policy identifier:

```text
bicubic_short_side_to_448_then_center_crop_v1
```

All 19 affected images are in category `10Rain`. All prior valid chunks remain
unchanged.

Both partition supervisors later printed `rg: command not found` only while
checking already-written pass markers. This was a supervisor-shell bookkeeping
failure after generation; independent Python verification proved all 1,339
chunks complete. Do not rerun partition export. If that launcher is reused in
the future, replace its `rg -q` dependency with an available portable check,
with tests and the required edit confirmation.

## Completed FLUX-native Qwen3 prompt cache

The condition CSV maps to 382 unique canonical restoration prompts. FLUX's
frozen native Qwen3 text encoder encoded them successfully.

```text
checkpoints/foundir_five_type_flux2_dit_v3/qwen3_prompt_cache/index.json
prompts: 382
SHA-256: 6b5659feac3c9ff91039374228c521cd79b065248d0e2329d7e1547b89dd84ba
conditions SHA-256 bound by cache:
e07e16234d4d1eb674b5d667732600628448b215e9fab67e6e6732f99c1ea2f7
status: frozen_flux_qwen3_prompt_context_cache
```

Do not rebuild the prompt cache unless direct verification shows corruption.

## FLUX restoration architecture and training contract

Config:

```text
configs/foundir_five_type_flux2_dit_v3.json
SHA-256: 5145e97b2f170b617f34180783d8e7934a77b5ef82f2e6c5cae7756c89763810
```

Backbone and frozen components:

- FLUX.2 Klein 4B Base;
- FLUX VAE frozen;
- FLUX-native Qwen3 text encoder frozen;
- LQ image uses native FLUX.2 reference-image tokens;
- no separate restoration adapter before the DiT;
- deterministic degraded-anchored residual flow, not generation from pure
  random noise.

Trainable components:

- FLUX image/text-image interaction LoRA rank `32`, alpha `32`, dropout `0`;
- 50 target modules: image qkv/proj and single-stream linear layers;
- bounded pixel-space refiner after VAE decode;
- refiner: width 32, 6 blocks, 33,251 parameters, zero-output identity
  initialization, maximum residual 0.25.

The pixel refiner is retained because the corrected small overfit improved
final-output PSNR and LPIPS. It is not justified by high-frequency correlation,
which is prohibited as a gate.

Full curriculum:

- stage A: single degradations, 360,000 presentations;
- stage B: single + double, 720,000 presentations;
- stage C: single + double + triple, 720,000 presentations;
- global batch: 32;
- total expected optimizer steps: 56,250;
- 5% clean GT→GT identity regularization;
- zero generic-condition dropout;
- H200 schedule: microbatch 2/GPU, accumulation 2;
- H100 schedule: microbatch 1/GPU, accumulation 4;
- exactly eight homogeneous H100 or H200 GPUs.

Overfit gate already passed under the corrected SOW metric contract:

- input PSNR: `19.3229`;
- restored PSNR: `25.1260`;
- PSNR gain: `+5.8032 dB`;
- input LPIPS: `0.40043`;
- restored LPIPS: `0.15427`;
- LPIPS improvement: `0.24616`;
- automatic gate: pass;
- protected official test: untouched.

This proves the architecture can learn the paired mapping on a tiny subset; it
does not establish full generalization or SOW acceptance.

## SOW restoration metrics

Development evaluation and the eventual frozen official test must report both
PSNR and LPIPS for every task. Current Stage-2 targets are:

| Task | PSNR minimum | LPIPS maximum |
|---|---:|---:|
| Blur | 25.9 dB | 0.18 |
| Noise | 38.9 dB | 0.10 |
| Low-light | 18.9 dB | 0.28 |
| Haze | 22.1 dB | 0.19 |
| Rain | 34.2 dB | 0.08 |

Checkpoint selection is configured to maximize the worst normalized
PSNR/LPIPS margin across all five tasks so one strong task cannot hide one
failing task.

## Immediate blocker: infeasible evaluation-group quota

The latest main launch completed Stage 0, verified/published the full Qwen
conditions, and built the prompt cache. It then failed before training in:

```text
smoke/freeze_foundir_five_type_evaluation_contract.py
lowlight_restoration/foundir_flux2_protocol.py::balanced_evaluation_subset
```

Exact exception:

```text
RuntimeError: insufficient validation groups for fold=0 category=02Blur_Noise: 5
```

Why it fails:

1. Config requests 3 folds and 20 records per category per fold.
2. Each `source_group` is deterministically hashed to one fold.
3. The sampler then permits at most one record from each source group inside a
   `(fold, category)` bucket.
4. Several categories have far fewer than 20 unique groups per fold, even
   though they contain hundreds of valid records.
5. Some categories have fewer than 60 groups across all three folds, so no
   possible reassignment can provide 20 unique groups to each fold.

Observed unique-group counts by fold:

| Split/category | Fold 0 | Fold 1 | Fold 2 | Total unique |
|---|---:|---:|---:|---:|
| validation 01Blur | 64 | 57 | 44 | 165 |
| validation 02Blur_Noise | 5 | 8 | 8 | 21 |
| validation 05Noise | 14 | 11 | 9 | 34 |
| validation 08Haze | 49 | 55 | 54 | 158 |
| validation 09Lowlight_Haze | 49 | 55 | 54 | 158 |
| validation 10Rain | 41 | 35 | 41 | 117 |
| validation 13Rain_Haze | 49 | 55 | 54 | 158 |
| validation 14Lowlight | 9 | 5 | 7 | 21 |
| validation 15Lowlight_Blur | 40 | 35 | 28 | 103 |
| validation 16Lowlight_Noise | 14 | 10 | 8 | 32 |
| validation 18Lowlight_Blur_Noise | 5 | 8 | 8 | 21 |
| internal 01Blur | 38 | 51 | 47 | 136 |
| internal 02Blur_Noise | 3 | 8 | 5 | 16 |
| internal 05Noise | 6 | 14 | 12 | 32 |
| internal 08Haze | 42 | 60 | 55 | 157 |
| internal 09Lowlight_Haze | 42 | 60 | 55 | 157 |
| internal 10Rain | 42 | 38 | 37 | 117 |
| internal 13Rain_Haze | 42 | 60 | 55 | 157 |
| internal 14Lowlight | 1 | 8 | 7 | 16 |
| internal 15Lowlight_Blur | 21 | 31 | 24 | 76 |
| internal 16Lowlight_Noise | 6 | 14 | 11 | 31 |
| internal 18Lowlight_Blur_Noise | 3 | 8 | 5 | 16 |

The record counts themselves are sufficient; the extra one-record-per-group
constraint makes the requested quota impossible.

## Exact next task

Fix and freeze a scientifically honest, deterministic evaluation-contract
sampler **before FLUX full training**. Do not simply lower a number until the
code stops failing.

Recommended minimal route to investigate and implement:

1. Preserve train/validation/internal-test split disjointness exactly.
2. Preserve the rule that one `source_group` never crosses evaluation folds.
3. Within each `(fold, category)`, maximize group diversity first: select one
   deterministic record from every available group.
4. If fewer than 20 groups exist but at least 20 records exist, fill the
   remaining quota deterministically and round-robin across the groups already
   assigned to that fold. Avoid taking all extra records from one group.
5. Record both record count and unique-group count per role/category/fold in
   the frozen contract JSON so correlation is transparent.
6. Add tests for sparse-group categories, determinism, exact quota, no record
   duplication, no group crossing folds, split separation, and no protected
   test access.
7. Run the sampler against the actual 685,495-row condition CSV before any GPU
   launch and verify every role/category/fold has exactly 20 records.
8. Freeze the evaluation contract once, before training.

Do not change `configs/foundir_five_type_flux2_dit_v3.json` casually. Its
SHA-256 is embedded in every completed Qwen chunk and the final condition
summary. Changing an unrelated evaluation field in that config would make the
existing condition-export identity disagree and could force needless
regeneration. Prefer a code-level correction that retains the frozen config,
or design an explicit backward-compatible identity migration and prove all
existing conditions unchanged.

Reducing the quota to one record per fold would technically avoid the sparsest
internal category but would produce a weak, noisy evaluation and waste the
available records. Rehashing groups or reducing from three folds to two cannot
solve categories with only 16 total unique groups if one-record-per-group is
retained. These are not recommended shortcuts.

After the fix:

1. Run focused unit tests and an actual full-condition contract dry run.
2. Run the complete launcher check:

   ```bash
   cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && \
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   bash smoke/launch_foundir_five_type_flux2_main_v3_8gpu.sh --check
   ```

3. Commit only the intended source/tests/handoff updates and push. Preserve all
   unrelated dirty files.
4. On one eight-GPU H200 or H100 instance, resume with:

   ```bash
   cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && \
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   bash smoke/launch_foundir_five_type_flux2_main_v3_8gpu.sh
   ```

5. The launcher should verify/reuse the existing condition CSV and prompt
   cache, freeze the corrected evaluation contract, and only then begin FLUX
   full-curriculum training.

## Monitoring and required success markers

Main log:

```text
logs/foundir_five_type_flux2_main_v3.log
```

GPU telemetry:

```text
logs/foundir_five_type_flux2_main_v3_gpu.csv
```

Do not infer progress from old lines: logs are append-only. Anchor monitoring
to the newest `FOUNDIR_FIVE_TYPE_MAIN_V3_START` line.

Expected sequence after correction:

1. `FOUNDIR_FIVE_TYPE_STAGE0_V3_END status=0`
2. `FOUNDIR_PHYSICAL_V8_FULL_EXPORT_VERIFY_PASS` or equivalent verified reuse
3. `FOUNDIR_QWEN3_PROMPT_CACHE_VERIFY_PASS` or equivalent verified reuse
4. `FOUNDIR_FIVE_TYPE_EVALUATION_CONTRACT_PASS`
5. FLUX full-curriculum step/checkpoint markers
6. training-complete summary
7. development checkpoint selection using PSNR and LPIPS
8. sealed internal evaluation only after selection is frozen

Fail closed on any traceback, missing hash, altered frozen artifact, OOM,
non-finite loss, missing checkpoint, or protected-test access.

## Current artifact presence matrix

Present and complete:

- FoundIR five-type training data;
- frozen 685,495-record manifest;
- frozen physical-v8 Qwen adapter;
- all 1,339 Qwen-condition chunks;
- combined Qwen-condition CSV and summary;
- 382-entry FLUX Qwen3 prompt cache;
- corrected PSNR/LPIPS overfit result;
- FLUX.2 weights, VAE, Qwen3 encoder, AlexNet LPIPS weights;
- released FoundIR baseline checkpoints.

Absent because the pipeline stopped before training:

- `manifests/foundir_five_type_evaluation_contract_v3.csv`;
- `manifests/foundir_five_type_evaluation_contract_v3.json`;
- `checkpoints/foundir_five_type_flux2_dit_v3/full_curriculum/training_summary.json`;
- `checkpoints/foundir_five_type_flux2_dit_v3/checkpoint_selection.json`.

## Dirty-worktree warning

At handoff, these unrelated tracked files were already modified and must not be
restored, staged, or overwritten as part of the evaluation-sampler fix:

```text
SESSION_HANDOFF_2026-08-18.md
reports/foundir_lowlight_gate1_result_2026-08-22.md
reports/foundir_lowlight_gate1_result_v1.json
```

Many generated manifests, reports, acquisition scripts, and tests are
untracked. In particular, the new 774 MB full-condition CSV and its JSON are
untracked but authoritative generated artifacts; preserve them and do not add
the large CSV to Git. Inspect `git status --short` directly for the complete
current list.

## Source-of-truth files for the immediate continuation

- `agents.md`
- `SESSION_HANDOFF_2026-09-07.md`
- `configs/foundir_five_type_flux2_dit_v3.json`
- `lowlight_restoration/foundir_flux2_protocol.py`
- `smoke/freeze_foundir_five_type_evaluation_contract.py`
- `smoke/run_foundir_five_type_flux2_main_v3_8gpu.sh`
- `smoke/launch_foundir_five_type_flux2_main_v3_8gpu.sh`
- `tests/test_foundir_five_type_main_v3.py`
- `logs/foundir_five_type_flux2_main_v3.log`
- `manifests/foundir_five_type_all_qwen_conditions_v3.json`
- `checkpoints/foundir_five_type_flux2_dit_v3/qwen3_prompt_cache/index.json`

## Claims that are and are not currently justified

Justified:

- Qwen synthetic five-type validation passed its frozen thresholds.
- Frozen Qwen inference conditions exist for all approved FoundIR-training
  records.
- The prompt cache is complete.
- The small FLUX architecture overfit improves PSNR and LPIPS.
- The current pipeline is ready for a corrected evaluation-contract freeze and
  then full FLUX training.

Not justified:

- full FLUX training has started;
- any full development benchmark has passed;
- the model meets Stage-1 or Stage-2 SOW thresholds;
- the model is SOTA;
- the complete system generalizes to the official FoundIR test;
- official FoundIR-test results exist.

The next agent must continue from the corrected, frozen evaluation contract and
the eight-GPU FLUX launch/monitoring task. Do not reopen completed stages or
claim final success prematurely.
