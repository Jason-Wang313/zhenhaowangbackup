# FoundIR-Lowlight Gate 2 result — 2026-08-22

## Decision

Gate 2 passed its execution and integrity contract on 245 reusable paired-real
outer-development records without accessing protected FoundIR IDs 1101--1150.

- Provisional conservative base: `P1_FOUNDIR_CASCADE`.
- Input-only bounded photometric predictor: killed for no cross-family
  transfer.
- Teacher value gate versus the fixed photometric control: failed for both the
  generalist and cascade under the every-family promotion rule.
- Teacher-at-inference deployment: remains conditional on provenance, license,
  runtime, and incremental value in the later trained bridge.
- Learned proposal fusion: not admitted at Gate 2. Both target-accessing fusion
  ceilings have useful PSNR headroom, but neither satisfies the frozen LPIPS
  non-regression rule on every family, and the strategy forbids training fusion
  before compact/FLUX proposals exist.
- Proceed to Gate 3: yes. Train the mandatory compact raw-to-restored
  deterministic control and run the frozen Qwen ordinal/causal gate. The
  compact control is compared against both the fixed photometric control and
  the provisional cascade.

## Conservative saved-uint8 metrics

All values are scene/group-balanced estimates. PSNR is the worse of literal
RGB and frozen `pyiqa` paths; LPIPS is the worse of the frozen `pyiqa` and
direct `lpips==0.1.4` paths.

| Candidate | Equal-family PSNR | Worst-family PSNR | Equal-family LPIPS | Worst-family LPIPS |
|---|---:|---:|---:|---:|
| Raw input | 10.4902 | 7.6175 | 0.5236 | 0.6865 |
| Fixed train-mean exposure | 13.6246 | 9.4313 | 0.4327 | 0.5864 |
| Historical synthetic FLUX | 14.0795 | 10.2948 | 0.4694 | 0.6118 |
| FoundIR generalist | 12.2307 | 9.4681 | 0.3917 | 0.5284 |
| FoundIR cascade | 14.7152 | 11.5087 | 0.3410 | 0.4646 |
| Oracle convex blend | 17.5173 | 13.4370 | 0.3222 | 0.4293 |
| Oracle frequency blend | 16.9150 | 13.0164 | 0.3407 | 0.4920 |
| Oracle monotone tone/WB | 25.7408 | 22.8427 | 0.3256 | 0.4306 |
| Oracle tone/WB/illumination | 25.8192 | 22.8886 | 0.3249 | 0.4289 |

The cascade improves over raw input on PSNR and LPIPS in LOL-v2, SDSD, and
SICE, so it is the safest current proposal. It does not pass the stronger
teacher-promotion comparison against fixed exposure on SDSD and SICE. This is
why it is a provisional base rather than evidence that the released teacher is
already sufficient.

The target-accessing convex fusion PSNR gap over the cascade is `+0.7099 dB`
on LOL-v2, `+5.7681 dB` on SDSD, and `+1.9283 dB` on SICE. The frequency
fusion gap is `+0.8737`, `+4.2180`, and `+1.5077 dB`, respectively. These
ceilings prove complementary information exists, but their per-image target
selection is non-deployable and their LPIPS trade-offs prevent automatic
fusion admission.

## Integrity and runtime evidence

- Complete metric rows: 2,940 = 245 records x 12 candidates.
- Generalist, cascade, and historical manifests: 245 records each; every PNG
  hash and native dimension independently reverified.
- Maximum literal-versus-`pyiqa` PSNR difference: `0.00015742 dB`.
- Maximum frozen LPIPS-path difference: `0.0` in this environment.
- Cascade inference: 87.4502 seconds total measured model time, 0.35694 seconds
  per record, 648,053,248 peak allocated bytes.
- Historical FLUX: 2,596.2363 seconds total, 10.59688 seconds per record,
  12,885,828,608 peak allocated bytes.
- The resume ran from `09:41:40Z` to `11:35:49Z`. Most wall time was the
  target-accessing full-resolution oracle grid search, not deployable model
  inference. This implementation must not be reused as a runtime estimate for
  the final system.

## Frozen identities

- Result JSON: `reports/foundir_lowlight_gate2_proposal_lattice_result_v1b.json`,
  SHA-256 `18ed75980d574d223162478a23ef1c41c0dbc52762ffae8e4b10edc3dc13f711`.
- Metric rows: `reports/foundir_lowlight_gate2_metric_rows_v1b.csv`, SHA-256
  `42ecead00754f8aa0db9601abcecff824e8290703808c90e440e8da430dc6ccd`.
- Teacher result: `reports/foundir_lowlight_gate2_teacher_result_v1b.json`,
  SHA-256 `2308304c54f3b11e33e2050f799d65ec83af33a0204c5fc4bbc167d9926e99d4`.
- Cascade manifest: `manifests/foundir_lowlight_gate2_cascade_v1b.csv`, SHA-256
  `2d271161839fc56b7d5a87f85a624efcb0e4542d65d48d9b7b9ae80006c64434`.
- Generalist manifest: SHA-256
  `71c6fdb8199f3b7b952fbade457b4bb05bc496bbe9e307413caddba0f2ad81b7`.
- Historical FLUX manifest: SHA-256
  `10f7cd537723dc39e8d002a8d3a7bbff0171ab794a4f7a814436b613aee7f1a7`.

Protected-test status: `NOT_ACCESSED_DURING_GATE_2`.
