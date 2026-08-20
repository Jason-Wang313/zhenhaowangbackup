# Qwen real-Lowlight constrained diagnostic readiness — 2026-08-20

## Status and interpretation

This is a post-failure diagnostic, not a replacement for the failed frozen
real-domain gate and not an independent validation result. The original gate
remains failed with immutable result SHA-256
`85542baf277c72a442e12d6da4f04e02051e45f7d586020472253437b6ebceda`.

The failed predictions were inspected before this method was selected. That
post-hoc fact is recorded in the config and every result. Even if this
diagnostic clears the original numeric thresholds, a fresh independently held
out real set is required before beginning diffusion on a validated-VLM claim.

## Failure being isolated

The SICE primary subset produced malformed free-form JSON for 31 of 90 images,
distributed across every true proxy level: L1 `3`, L2 `6`, L3 `8`, L4 `7`, and
L5 `7`. LOL-v2 produced canonical JSON for all 143 images but missed its exact
severity and clean-versus-low-light thresholds. The diagnostic asks whether a
substantial part of the failure is decoding instability rather than absent
conditional class evidence.

## Frozen recovery method

- Load the exact failed step-600 visual checkpoint, SHA-256
  `b0e65a8597077973a2a103dc3369e0cb1ea703de95e2e4841d359c3d34eff77f`.
- Freeze every model parameter. No training or weight update occurs.
- Reuse the exact 233 records, reference-based proxy labels, images, processor,
  prompt, and original predeclared thresholds.
- For each image, teacher-force each of the five existing canonical JSON
  targets for project L1--L5.
- Apply the normal causal language-model shift and calculate negative log
  likelihood only over unmasked assistant-target tokens.
- Normalize each candidate by its number of scored target tokens so the shorter
  L1 JSON is not automatically favored.
- Select the candidate with minimum mean target-token NLL; exact ties select the
  lower project level.
- Emit the selected canonical JSON. Free-form generation is not used, so
  structured validity is guaranteed by construction.

The diagnostic config is
`configs/qwen_lowlight_real_validation_constrained_step600.json`, SHA-256
`ab04a83f0a19669dbd2c827aa05b259d413c12dc73a6237894fe90b77ccc6219`.
It explicitly freezes that labels, thresholds, prompt, and model weights did
not change after failure.

## Evidence and execution safeguards

Preflight verifies the original failed config and result hashes, all eight
parent metric artifacts, all 47 parent generation batches, and all 233 parent
predictions before allowing this diagnostic. It then repeats the original
model, checkpoint, data, processor, source-file, archive, runtime, and GPU
checks. The H100/H200 runner performs all large model hashes before scoring.

Canonical candidate scores, token counts, selected levels, and runner-up
margins are stored in immutable per-batch JSON files. The job is resumable,
uses a workspace-local retained tmux session, writes unbuffered persistent
logs and terminal markers, and records H100/H200 utilization every two seconds.

One visible H100 or H200 is appropriate because this is a 233-image frozen
diagnostic. It performs five forward scoring passes per image batch and does
not train.

## CPU verification

- Canonical causal-shift masking, mean-NLL scoring, tie-breaking, and invalid
  score rejection have focused unit coverage.
- All 18 supervised-Qwen, pilot, real-validation, and constrained-scoring tests
  passed.
- Python compilation and shell syntax passed.
- Contract-only recovery preflight verified the immutable parent failure and
  complete parent artifact cardinality.
- Workspace-local tmux 3.7 and the complete launcher `--check` path passed.

The diagnostic must be reported as either `post-failure constrained diagnostic
cleared the original thresholds` or `post-failure constrained diagnostic did
not clear the original thresholds`. It must never be reported as retroactively
passing the original real-domain gate.
