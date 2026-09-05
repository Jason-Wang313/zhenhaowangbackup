# FoundIR five-type Qwen right-padding incident — 2026-09-05

## Outcome

The fixed-commit Stage 0 rerun started at `2026-09-05T01:37:49Z` and ended
with status 1 at `2026-09-05T01:48:17Z`.  The prior distributed-control-flow
deadlock was fixed: all eight ranks atomically published their assigned
16-record raw shards, and rank zero broadcast the same parser failure to every
rank without an unmatched collective.

The failure was caused by right-padded batched decoder generation, not by an
NCCL failure and not by evidence of Qwen real-domain collapse.

## Exact evidence

- The frozen Qwen processor config uses right padding for supervised training.
- Synthetic validation inputs are uniform 448-pixel crops, so batched prompt
  lengths are uniform and the generation defect was latent.
- The 128 FoundIR training-pilot inputs have different aspect ratios.  After
  Qwen smart resizing, they produce between 228 and 252 visual tokens.
- All `72/72` records with the batch-maximum 252 visual tokens produced valid
  fixed-slot JSON.
- All `56/56` shorter records that received 5--24 right-padding visual-token
  positions produced invalid text (`addCriterion`, `atical\n addCriterion`, or
  `tolua`).
- Transformers emitted its explicit decoder-only right-padding warning once
  per rank and instructed callers to use `padding_side='left'`.
- The protected FoundIR official test was not accessed.  Only authorized
  FoundIR training records were involved.

Therefore the provisional 56.25% valid-output rate and the diagnostic 40.15%
type micro-F1 do not measure the frozen Qwen model's FoundIR transfer quality.
They are invalidated by this batching defect and must not be cited as model
performance.

## Correction

- Added one exception-safe `left_padded_generation` context manager.
- Applied it to joint, per-type, pilot-export, and full-sharded decoder
  generation while preserving right padding for supervised training.
- Versioned pilot raw-shard identity and location as `left_padding_v2`, so the
  eight immutable faulty shards remain preserved and cannot be reused by the
  corrected rerun.
- Added padding-side and decoding-version fields to published condition-cache
  identities.

## CPU verification

- `36/36` focused FoundIR/Qwen trainer tests passed.
- Python compilation passed for all five modified source/test files.
- Shell syntax passed for the Stage 0, full runner, and full launcher.
- `git diff --check` passed.

The corrected 128-record GPU Stage 0 must be rerun before any claim about
FoundIR-domain Qwen type recognition or before FLUX restoration training.
