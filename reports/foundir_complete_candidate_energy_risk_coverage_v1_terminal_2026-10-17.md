# Complete candidate energy risk/coverage audit terminal — 2026-10-17

## Outcome

The no-training audit completed successfully from UTC `20:30:28` to
`20:33:51` on eight H100 GPUs: `203` seconds, `.0564` wall-clock hours and
`.4511` allocated GPU-hours. All protected-data flags are false.

The current complete-candidate scorer has no deployable confidence signal.
On the 174-record calibration partition drawn from 17 source groups,
fidelity-only PSNR is `43.341621968938014`. Every threshold that accepts any
alternative is worse:

- `96.82%` acceptance: `43.115114025685976`;
- `24.21%` acceptance: `43.263815057688745`;
- `4.85%` acceptance: `43.32472296966903`;
- `.9698%` acceptance: `43.337493545707616`;
- `.4849%` acceptance: `43.339416964300746`.

Because no nonzero-coverage threshold improves calibration PSNR, the frozen
selection rule correctly falls back to zero acceptance. On the disjoint
82-record, 17-source-group evaluation partition, calibrated selection exactly
equals fidelity at `43.514986642977085/.03269933527846616`; PSNR and LPIPS
changes are both zero. The machine result is
`scorer_has_deployable_signal=false`.

## Decision

This rules out a simple threshold/calibration repair. The scorer's ordering is
wrong even among its highest-confidence proposed changes. Do not resume
`formal1`, tune another threshold on evaluation, or claim that fail-closed
selection worked.

The next model must formulate the deployment decision directly: for each
candidate versus fidelity, predict whether its local MSE advantage is
positive and whether it causes perceptual harm. Train these pairwise decisions
with balanced, margin-aware supervision, then calibrate acceptance on source-
disjoint calibration groups. Do not reuse the failed self-reported
heteroscedastic uncertainty as confidence.

Machine-readable result:
`reports/foundir_complete_candidate_energy_risk_coverage_v1.json`.
SHA-256:
`231d55a77f6ab216e7c805890bf48cb08983f30d12519b6255c9cda694cc202d`.
