# Qwen Lowlight real-adaptation failure follow-up — 2026-08-21

## Preserved scientific result

The source-disjoint real-adaptation experiment completed all 200 optimizer
steps and failed its automatic scientific gate. It was not a training crash.
The immutable result remains under
`checkpoints/qwen_lowlight_real_adaptation_step600/`; this follow-up does not
modify or overwrite that directory.

- training summary SHA-256:
  `924f3d552b5305783873f769c8e1c18053548ed24714511671892a5db7378f03`
- checkpoint-selection SHA-256:
  `c7618cd719f63b8f08f0426ef62356cda4595428a199216c18a42ab5d6640324`
- selected checkpoint: step 150
- selected visual SHA-256:
  `32c0496ee758b990d312d67a6614d2f85ef8e87804c97e9f6158bad51ace9388`
- real validation: passed all predeclared safeguards
- consumed fresh real test: failed the required per-level accuracy floor
- synthetic replay: failed structured-output validity, degradation-type
  accuracy, and severity-MAE safeguards
- official FoundIR used: no
- diffusion authorized: no

The consumed real test remains historical failed evidence. It must not be run
again, used for checkpoint selection, or used to tune a corrective experiment.

## Launcher-status defect and fix

The trainer intentionally returned status 2 after writing a complete verified
failed summary. `torch.distributed.run` wrapped that child exit as
`ChildFailedError` and returned shell status 1, so the original shell events
looked like a crash. Those historical logs are preserved unchanged.

The checked runner now classifies the completed result after torchrun returns.
Classification verifies the config hash, nonempty Boolean gate checks, selected
checkpoint path containment, exact checkpoint metadata, and the selected visual
SHA-256. A verified pass maps to 0, a verified automatic-gate failure maps to 2,
and missing, corrupt, or inconsistent evidence maps to 1. Focused CPU tests
cover all three outcomes plus config and checkpoint corruption.

## Validation-only checkpoint diagnostics

The next diagnostic evaluates existing visual checkpoints 25, 50, 100, 150,
and 200 by deterministic free generation on only:

- the frozen 143-record real validation split; and
- the frozen 200-record synthetic replay validation split.

The driver prohibits writes into the historical result directory and publishes
each launch to a new immutable attempt path below
`checkpoints/qwen_lowlight_real_adaptation_checkpoint_diagnostics_20260821/`.
It verifies every source checkpoint and records per-checkpoint metrics, gate
checks, hashes, and the validation-only steps that pass every real and synthetic
safeguard. It performs no checkpoint selection and explicitly records that the
consumed real test and official FoundIR were not inferred.

At report creation, this coding-agent terminal has no visible CUDA devices.
Therefore, the diagnostic implementation is CPU-verified but no new diagnostic
metrics exist yet. A checked workspace-local tmux launcher is provided for the
separate GPU-enabled interface. Corrective adaptation design and new-holdout
acquisition remain downstream of the diagnostic result; no corrective training
is authorized by this report.
