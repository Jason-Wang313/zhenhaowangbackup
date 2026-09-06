# Qwen post-codec v9 readiness — 2026-09-06

## Decision

Physical-v8 completed successfully on frozen synthetic validation but failed the
FoundIR-training-only type gate because degradation-set exact match was 52.34%,
below the frozen 60% threshold. This was a scientific failure, not a crash.

The failure is structured. On the 128-record FoundIR-Train pilot, the model
detected Noise in all 25 PNG Blur+Noise records, but missed all 21 JPEG
Noise-only or Lowlight+Noise records under the frozen second-highest multi-view
rule. Across the complete authorized training manifest, all 58,015 `05Noise`
and all 52,995 `16Lowlight_Noise` inputs have JPEG/JPG filenames, while all
29,950 `02Blur_Noise` and all 29,950 `18Lowlight_Blur_Noise` inputs are PNG.
The exact machine audit is
`reports/foundir_qwen_noise_codec_order_audit_2026-09-06.json`, SHA-256
`24195841f37ed383895bc0d026fe54867bf741349ff0799bc62515a575ef8c49`.

Physical-v8 applies its label-neutral JPEG nuisance before target degradations
and adds Noise last. It therefore never teaches the model how target Noise
looks after a lossy codec has attenuated and spatially correlated it. The
available evidence strongly supports this mismatch. FoundIR does not publish
its training-data generator, so the exact upstream operator order is not
claimed as proven.

## Minimal correction

Post-codec v9 keeps the Qwen architecture, LoRA ranks, prompts, decoder output,
and five-type labels unchanged. It starts from the passing physical-v8 adapter
with a fresh optimizer and trains one bounded 500-step continuation:

- 20,000 synthetic training images / 100,000 five-query dialogues;
- 10,400 original physical-v8 replay images to resist forgetting;
- 4,800 post-target-JPEG Noise-positive images;
- 4,800 post-target-JPEG Noise-negative images;
- JPEG quality 75, 85, 94, or 99, with balanced positive and negative use;
- no FoundIR image used for Qwen gradients;
- no protected synthetic test record present or accessed.

Validation contains the complete frozen 10,000-image physical-v8 validation
split plus 2,400 post-target-JPEG records, balanced 1,200/1,200 for Noise
present/absent. Separate domain gates prevent the larger original slice from
hiding a post-codec failure. The post-codec slice requires Noise recall at
least 85% and precision at least 90%; the original slice must retain Noise
recall and precision of at least 96%.

## Frozen identities and checks

- v9 config SHA-256:
  `1df3f3853ab7ee90bf6e2b22f497ea63756be1139d363d572e3c8084b5eb7f0f`
- replay manifest SHA-256:
  `872c01cb9ea3768fe7eb815e46a4e869eee7b0a7c047a46d1e48907b4457085c`
- replay metadata SHA-256:
  `09a58770351da367388b589660e650d23c53687f044b598669ab4b6df46baf72`
- physical-v8 initialization adapter SHA-256:
  `5416311d7cd0aa0308e206084c13c7c950f9dacfc96cd558466642804ccbee3c`
- full artifact/hash preflight: passed;
- Qwen trainable scope: 129,550,080 parameters;
- focused CPU tests: 50 passed;
- FoundIR official test pixels accessed: no;
- protected synthetic test inference: no.

Based on physical-v8 timing, 500 training steps should take about 48 minutes on
eight H200s. Full preflight and 12,400-image validation bring the expected
end-to-end wall time to roughly 65–80 minutes. Eight H100s are supported with
the same scientific global batch and may be somewhat slower.
