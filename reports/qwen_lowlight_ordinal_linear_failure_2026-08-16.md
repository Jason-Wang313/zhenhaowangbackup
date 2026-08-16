# Frozen-Qwen linear ordinal-head selection failure

Date: 2026-08-16 UTC
Scope: train-source selection only; fixed validation was not touched

The 28,676-parameter CORAL linear head completed feature extraction and
train-source holdout selection, then failed its baseline-relative gate and
stopped before fixed validation.

- code revision: `609caa99c4ae0a6bb67c199dc24eb1766e78ba54`
- config SHA-256:
  `6bb849efd4c97cb817d14b4d4c6eb4ae9abe24cc9d22554139d0b12baf29824d`
- selection artifact SHA-256:
  `1e6fdd301326846be74b7c208c02dfd8b47dd080bae0a7bcc91a85f952d43b17`
- checkpoint SHA-256:
  `c05a35ad9ff8ba81881adafaa5dc1a440049576b0c7d9f2ee1629e7019e1e03d`
- log SHA-256:
  `498b19061c4dd43e4a6fdf5bd0b52bc1bbe6c464a3b916b70d65cca6b9a7eb91`
- best epoch: 61
- selection accuracy / MAE: `0.6667 / 0.3967`
- comparison baseline accuracy / MAE: `0.6933 / 0.3467`
- per-level recall L1-L5:
  `0.8833, 0.5167, 0.5333, 0.4167, 0.9833`
- all five predicted levels represented: `true`
- minimum per-level recall gate: `pass`
- baseline-relative accuracy and MAE gates: `fail`
- fixed validation touched: `false`
- internal test used: `false`
- FoundIR used: `false`

The 1,500 frozen-Qwen feature files are retained with exact image, model,
configuration, pooling, and extractor provenance. They support bounded
train-holdout head selection without repeating H100 feature extraction.
