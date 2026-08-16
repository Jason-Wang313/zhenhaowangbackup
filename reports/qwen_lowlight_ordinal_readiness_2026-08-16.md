# Frozen-Qwen Lowlight ordinal-head readiness

Date: 2026-08-16 UTC
Scope: train-only head fitting/selection followed by fixed Lowlight validation

## Design

Qwen2.5-VL remains frozen. For each image, the frozen vision tower produces
3,584-dimensional merged visual tokens. Token-wise mean and standard deviation
are concatenated into a 7,168-dimensional feature. A CORAL-style linear head
predicts four ordered severity boundaries and contains exactly 28,676
trainable parameters.

The head uses 300 deterministic train sources with complete same-source L1-L5
sequences:

- head fitting: 240 sources / 1,200 records;
- train-source selection holdout: 60 sources / 300 records;
- source overlap: none;
- fixed validation is not opened unless the train-source selection gate passes;
- internal test and FoundIR remain protected.

The selection gate requires strict structured output, all five predicted
classes, at least 0.4 recall for every level, at least 0.6 accuracy, and
performance matching or beating its train-fitted luminance-centroid baseline:
69.33% accuracy and 0.3467 MAE. Only a frozen head that passes selection may be
evaluated on the already frozen 25-source validation set, where it must also
beat the 64% / 0.40 baseline.

Noise, Blur, Haze, and Rain remain `blocked_no_task_calibration`; this runner
does not generate, train, or evaluate those tasks.

## CPU preflight evidence

- config SHA-256:
  `6bb849efd4c97cb817d14b4d4c6eb4ae9abe24cc9d22554139d0b12baf29824d`
- Qwen revision:
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- exact model files verified: 16
- train bundle SHA-256:
  `55678a61a06ca337ced648aff4dd0090d54a064bf345e94c815525ce53c8d8b5`
- selection bundle SHA-256:
  `33b980831158fc33b4db4c21869e0aa1cf1b9c6ebd52e5bd78e81329c3c0f7b3`
- frozen feature dimension: 7,168
- trainable parameters: 28,676
- image-only processor batch: five images, 494 visual tokens each
- feature caches bind pair/image hash, config, model revision, pooling schema,
  and exact extractor-code hash and fail closed on any mismatch.
- extractor SHA-256:
  `12fb32fb7b090fc1181e5cb65fb60de1412cfafd5ebaad9be4949d474dfa6569`
- 41 repository unit tests, Python compilation, launcher shell syntax, and
  `git diff --check` passed.

## Offline H100 command

```bash
CUDA_VISIBLE_DEVICES=0 smoke/run_qwen_lowlight_ordinal_head.sh
```

Feature extraction is resumable. If the train-source selection gate fails,
the launcher exits nonzero without opening fixed-validation images.
