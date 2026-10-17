# Complete candidate energy v1 formal pass 1 — 2026-10-17

## Outcome

`formal1` reached step `1,706` on eight H100 GPUs and wrote a complete
checkpoint and development-validation artifact. Training remains numerically
healthy and all protected-data flags are false, but the first scientific gate
fails badly.

| Arm | PSNR | LPIPS |
|---|---:|---:|
| Proposal | 43.124160178005695 | .050362067760033824 |
| Fidelity | 43.39715355634689 | .028955043449059303 |
| Candidate-energy selected | 43.18257042020559 | .03665485502278898 |

Relative to fidelity, selection loses `.21458313614130042 dB` and worsens
LPIPS by `.007699811573729676`. The selected result remains below the frozen
PSNR requirement `43.424160178005695`; the joint gate is false.

## Failure signature

The selector accepts an alternative on `96.69189453125%` of 32x32 regions,
despite fidelity being the intended safe fallback. Selection fractions are:

- proposal: `43.511962890625%`;
- perceptual: `25.885009765625%`;
- under-fidelity: `12.7685546875%`;
- over-fidelity: `9.844970703125%`;
- fidelity: only `3.30810546875%`;
- 50/50 fidelity/perceptual: `2.62451171875%`;
- 75/25 fidelity/perceptual: `2.056884765625%`.

Therefore the learned uncertainty/risk estimates are overconfident and do not
provide the promised fail-closed behavior at pass 1. This cannot be described
as a small miss or as evidence that the complete candidate energy already
works.

## Runtime and decision

Measured throughput through pass 1 is `1.1797516255411302` seconds per
optimizer step. The projected complete run is `2.7953559349627337`
wall-clock hours and `22.36284747970187` allocated GPU-hours. The frozen run
continued past step 1,900 because no pass-1 stop rule was predeclared. Do not
retune or restart it from this validation result. Inspect the next frozen
milestone while preserving this pass-1 failure as evidence.

Checkpoint:
`checkpoints/foundir_complete_candidate_energy_noise_v1/formal1/step_001706.pt`.
Validation:
`checkpoints/foundir_complete_candidate_energy_noise_v1/formal1/validation_step_001706.json`.
