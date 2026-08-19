# Qwen scaling preflight JSON-compatibility incident — 2026-08-19

## Outcome

The first GPU launch attempt, `20260819T113717Z-3487254`, stopped safely in
preflight with terminal status `1`. No 1/2/4/8-GPU benchmark worker started,
no model was loaded for training, and no world-size summary was written.

The authoritative preflight itself remained valid and unchanged. The failure
was an equality-check representation bug: `selected_per_level` used integer
keys in the newly computed Python dictionary, while the immutable JSON reload
correctly represented JSON object keys as strings. The values and selected
record bundle were identical.

## Additive compatibility retry

To preserve the original tracked implementation and immutable runtime
evidence, the retry is additive:

- JSON-key compatibility entry point:
  `smoke/validate_qwen_lowlight_supervised_scaling_json_compat.py`
- persistent retry runner:
  `smoke/run_qwen_lowlight_supervised_scaling_1_2_4_8_json_retry.sh`
- persistent retry tmux launcher:
  `smoke/launch_qwen_lowlight_supervised_scaling_json_retry_tmux.sh`

The compatibility entry point canonicalizes only the
`selected_per_level` mapping keys before the existing fail-closed validator
compares current evidence with its JSON reload. All hashes, parameter counts,
selected records, world-size contracts, correctness checks, telemetry checks,
and selection thresholds remain unchanged.

The retry uses a distinct tmux session name because the failed launch's
persistent shell may still be inspectable in the GPU interface. It appends a
new attempt ID to the existing persistent main, event, and telemetry logs.
