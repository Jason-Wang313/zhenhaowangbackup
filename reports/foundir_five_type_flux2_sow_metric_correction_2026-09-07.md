# FoundIR five-type FLUX.2 SOW metric correction — 2026-09-07

## Decision

The active five-type FoundIR restoration route evaluates and gates restoration
quality with the two objective metrics named by the SOW: PSNR and LPIPS.
High-frequency correlation is removed from the active configuration, evaluator,
and admission logic. Historical v2/v3 artifacts remain unchanged and are not
relabelled.

## Historical v3 result

The first v3 overfit completed 200 optimizer steps and wrote an immutable
checkpoint and result. It improved mean output PSNR but did not calculate
LPIPS, then stopped because of the now-superseded internal metric. That result
is not a SOW pass or failure.

The corrected route reuses the exact step-200 checkpoint solely to evaluate the
same 32 FoundIR-training-only records with PSNR and offline AlexNet LPIPS. The
new result is written separately under `native_reference_overfit_sow/`. If the
historical checkpoint is unavailable in a fresh installation, the corrected
200-step overfit is run from scratch.

## Development protocol

The official FoundIR training pairs are grouped by exact and perceptual GT
identity before a 96/2/2 train/development/sealed-internal split. Development
selects a checkpoint with per-task PSNR and LPIPS; the sealed internal split is
consumed once after selection. These are generalization estimates, not official
FoundIR-test claims. The official test remains protected until the entire model
and inference protocol are frozen and the user explicitly authorizes the final
evaluation.
