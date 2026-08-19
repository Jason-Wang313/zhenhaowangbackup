# Frozen Qwen real-Lowlight validation readiness — 2026-08-19

## Status and scope

CPU readiness passed. GPU inference has not started at the time of this
report. This gate evaluates the already selected supervised-Qwen step-600
visual checkpoint on non-FoundIR real Lowlight data. It does not train or tune
the model, does not use FoundIR, and is not a restoration or SOW acceptance
result.

All project L1--L5 labels in this gate are
`reference_based_proxy_not_human_ground_truth`. They are derived from paired
reference-image luminance and are neither human annotations nor official SICE
or LOL-v2 severity labels. Results must retain that qualification.

## Frozen model and inference contract

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`, revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Selected checkpoint: step 600 from the passed 8k synthetic Lowlight pilot.
- Visual checkpoint SHA-256:
  `b0e65a8597077973a2a103dc3369e0cb1ea703de95e2e4841d359c3d34eff77f`.
- Frozen parameters: the complete 8,292,166,656-parameter model is loaded in
  bfloat16 for evaluation; the step-600 visual state is loaded strictly and
  all parameters are then frozen.
- Generation: deterministic greedy decoding, batch size 5, at most 64 new
  tokens, PyTorch deterministic algorithms, and the math SDPA backend.
- Config: `configs/qwen_lowlight_real_validation_step600.json`, SHA-256
  `748091129e4f6c59f9f9eb391749b139d3bcce3a4c306331e74fb4da1c528dbb`.

Thresholds, data selections, prompts, and checkpoint selection were frozen
before this real-domain inference. Model predictions were not consulted while
selecting records or setting labels.

## Real-domain datasets and proxy-label construction

### SICE balanced primary subset

- Source: held-out test data from the `okhater/SICE` mirror at revision
  `1a65a70ad5ae9feef5f8402dfa42347652ce1f2c`.
- Current workspace evidence: 47 held-out source groups and 223 readable source
  images.
- L1 uses a selected fusion reference as a proxy-clean image.
- Each lower exposure is assigned to the nearest fixed project luminance-ratio
  anchor in log2/EV space: L2 `0.50`, L3 `0.38`, L4 `0.30`, and L5 `0.14`.
- Only candidates within `0.75 EV` of their nearest anchor are eligible.
  Selection favors minimum anchor error, with seed `2026081906` only as a
  deterministic tie-breaker.
- The final balanced quota is 18 per level, or 90 records total. Eighteen is
  the largest defensible minimum because only 18 eligible L3 candidates exist
  under the frozen tolerance.

### LOL-v2 real-captured stress subset

- Official archive:
  `data/real_lowlight_validation/sources/LOL-v2.verified.zip`.
- Archive bytes: `1,046,491,030`.
- Archive SHA-256:
  `9820d8b112438d94d1f5d4d25817eee618a8cf4bc63a65cd6b19f5a98c3faefa`.
- Full ZIP CRC verification passed. The separate original assembled
  `LOL-v2.zip` has SHA-256
  `b3df5d641c317e7d7ff541b397983f313f440c8e44b2c485b7c19d87b7a709eb`,
  failed integrity, and remains quarantined as evidence; it is not used.
- All 100 official real-captured held-out Low inputs are retained. Their paired
  Normal images contain only 43 unique exact SHA-256 contents, so L1
  proxy-clean inputs are deduplicated by exact content rather than counted 100
  times.
- The resulting stress subset contains 143 records: L1 `43`, L3 `2`, L4 `29`,
  and L5 `69`. It is intentionally skewed and is evaluated separately from the
  balanced SICE primary subset.

The frozen manifest contains 233 total records:
`manifests/qwen_lowlight_real_validation.csv`, SHA-256
`52f6fc9c5cdc0347559b37a30f7f5a607ac9a3d5576c6b9e8dc22424c478106b`.
Its metadata is `manifests/qwen_lowlight_real_validation.json`, SHA-256
`3b9be6683208580f51ddb93779436270b28e97ab4b27a2df720bfc6238cdd1fb`.
The metadata inventories 423 source files. CPU preflight re-hashed and decoded
all of them and re-hashed the official LOL-v2 archive.

## Predeclared automatic gates

Both subsets must pass; one passing subset cannot compensate for the other.

| Metric | SICE balanced | LOL-v2 stress |
| --- | ---: | ---: |
| Valid structured output | >= 0.99 | >= 0.99 |
| Degradation-type accuracy | >= 0.85 | >= 0.90 |
| Degradation-type macro-F1 | >= 0.75 | >= 0.90 |
| Exact severity accuracy | >= 0.50 | >= 0.60 |
| Severity macro-F1 | >= 0.45 | >= 0.35 |
| Severity MAE | <= 0.75 | <= 0.65 |
| Within-one-level accuracy | >= 0.85 | >= 0.90 |
| Minimum required-level accuracy | >= 0.25 | >= 0.25 |
| Quadratic weighted kappa | >= 0.60 | >= 0.75 |

SICE requires evidence for L1--L5. The LOL-v2 stress gate requires per-level
accuracy for its supported L1, L4, and L5 classes; its two L3 records remain in
all aggregate metrics.

## GPU choice, persistence, and outputs

One visible H100 or H200 is intentional for this small, frozen 233-image
inference job. Eight distributed copies of the approximately 8.3B-parameter
model would add model-loading and process-communication overhead to a job whose
prior 1,000-image distributed evaluation took only about 35--37 seconds after
loading. This adaptive one-GPU choice does not create a general one-GPU rule;
substantial training continues to prefer multiple GPUs when they reduce total
wall-clock time without changing experiment semantics.

The launcher uses workspace-local tmux 3.7 with `remain-on-exit`, an exclusive
run lock, immutable resumable per-batch outputs, unbuffered persistent logs,
terminal exit markers, and two-second GPU utilization/memory/power telemetry.
Status 2 means a legitimate automatic gate failure, not necessarily a runtime
crash.

- Persistent log: `logs/qwen_lowlight_real_validation_step600.log`.
- Event log: `logs/qwen_lowlight_real_validation_step600_events.log`.
- GPU telemetry: `logs/qwen_lowlight_real_validation_step600_gpu.csv`.
- Results: `checkpoints/qwen_lowlight_real_validation_step600/`.
- Final machine-readable decision:
  `checkpoints/qwen_lowlight_real_validation_step600/result_summary.json`.

## CPU verification completed

- Python compilation passed for the contract, builder, preflight, evaluator,
  and focused tests.
- Shell syntax passed for acquisition/recovery, runner, and tmux launchers.
- All 15 supervised-Qwen, pilot, and real-validation unit tests passed.
- Rebuilding the manifest reproduced and byte-verified both frozen outputs.
- CPU preflight passed with 423/423 source hashes and image decodes, official
  archive hashing, manifest/subset bundle verification, strict checkpoint
  structure inspection, and a five-level processor dry run.
- The 366 unique real-source file hashes have zero exact overlap with either
  the 8,000 synthetic training-file hashes or all 10,000 files in the frozen
  synthetic pilot manifest (manifest SHA-256
  `e1ea9132bc0d5e55b30998c51c8e37a0363f29b2a23e7851ee804f3100d4a784`).
- The processor dry run produced 117 visual tokens for each selected L1--L5
  record.
- Workspace-local tmux reported version 3.7, and the complete launcher
  `--check` path passed.
- `git diff --check` passed.

The GPU runner performs a second full preflight, including all large base-model
SHA-256 checks and exact H100/H200/runtime validation, before inference begins.
