# Expanded frozen-Qwen ridge Lowlight readiness

Date: 2026-08-16 UTC
Scope: Lowlight train-only head selection and one new unexposed validation gate

The first ridge head is preserved as a failed result. This bounded successor
addresses its high-dimensional small-sample overfit by using every Lowlight
train source and a much larger, dataset-stratified train-only selection
holdout. It does not tune on the exposed 25-image validation result.

## Frozen design

- config: `configs/qwen_lowlight_ridge_expanded_head.json`
- config SHA-256:
  `b04bf893e073cf50423a384456398b12ce92625a979453c8f8c61e210f8bd33c`
- frozen Qwen revision:
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- pooling: visual-token mean and standard deviation, `7,168` dimensions
- train-fit: `1,280` sources / `6,400` records
- train-only selection: `320` sources / `1,600` records
- fit bundle SHA-256:
  `721eb7694fae0ba9222c4ef38729948358de63122ab9ce15cdb39dad802c4780`
- selection bundle SHA-256:
  `75fb05afe9a273c6d9d8b19c62760df58ca89438dbb34fb4ef39170a419ce944`
- candidate ridge alphas: `300, 1000, 3000, 10000, 30000`
- fixed decoder: nearest integer, clipped to L1-L5
- train-only luminance baseline: accuracy `0.629375`, MAE `0.465`

The train partition is stratified by source dataset. The fit source counts are
DIV2K `505`, LSDIR `707`, and MiO100 `68`; the selection counts are DIV2K
`126`, LSDIR `177`, and MiO100 `17`.

## New validation gate

The previous 25 validation source IDs are read from the exact failed metrics
artifact, whose SHA-256 is
`4baf465aab731a845249a3bdbbdbec7e642439bae892686a0a7f5b09f6c5b0fd`.
All 25 are explicitly excluded.

The replacement validation set is frozen before GPU execution:

- `50` previously unused validation sources
- all five levels per source, `250` records total
- DIV2K `22`, LSDIR `27`, MiO100 `1`
- bundle SHA-256:
  `c8e017fd19b9527610890e3976ceb536f0ae6ff5ebc9680094bc0538542e00ea`
- exact source IDs are stored in the config

No new validation image or Qwen validation feature is opened before a ridge
candidate passes every train-selection gate. The internal test and official
FoundIR remain protected.

## Execution behavior

Feature extraction uses a new provenance-bound cache under
`checkpoints/qwen_lowlight_ridge_expanded_head/features`. It does not overwrite
or relabel the prior 1,525 cached features. The H100 batch is increased from 5
to 20; the earlier run allocated about 16 GiB at batch 5, leaving ample room on
the 80 GiB H100. Every feature file is written atomically, verified by image
hash/config/model/extractor metadata, and reusable after interruption.

The first run extracts `8,000` train features. If and only if one predeclared
candidate passes the 1,600-record selection gate, it fits the final head on all
8,000 train records and extracts the `250` frozen validation features. A failed
selection exits without opening those validation images. A failed validation
is retained as failure evidence and must not be tuned against.

CPU readiness passed with zero pre-existing expanded-cache features, zero new
validation images opened, all 25 exposed sources excluded, all 16 offline model
files verified, all 48 repository unit tests passed, Python compilation passed,
and launcher shell syntax passed. A train-only 20-image processor check also
passed with exactly 20 image grids; it did not open the new validation subset.

## Offline H100 command

```bash
CUDA_VISIBLE_DEVICES=0 smoke/run_qwen_lowlight_ridge_expanded_head.sh
```

The live extraction counter is resumable. On a first run it advances toward
`8000`; after the train-selection gate, a second counter advances toward `250`.
Noise, Blur, Haze, and Rain remain `blocked_no_task_calibration`.
