# Frozen-Qwen ridge ordinal-head readiness

Date: 2026-08-16 UTC
Scope: cached train-source selection and gated fixed Lowlight validation

After the CORAL linear head failed narrowly, six predeclared scalar ridge
regularization values were compared using only the existing train-source
selection holdout: `0.1, 1, 10, 100, 1000, 10000`. Alpha `1000` was selected.
This is a scalar ordered-level regressor, not a prompt classifier, and has only
7,169 fitted parameters.

## Frozen selection result

- config SHA-256:
  `943b1d5a0fee6c55d27610064683a70fc1e7f17c54ffb5bc3ae97686e2ba2097`
- cached frozen-Qwen features verified: 1,500
- train records: 1,200
- train-source selection records: 300
- selection accuracy: `0.74`
- selection MAE: `0.2633333`
- selection recall L1-L5:
  `0.90, 0.65, 0.70, 0.5667, 0.8833`
- baseline accuracy / MAE: `0.6933333 / 0.3466667`
- train bundle SHA-256:
  `55678a61a06ca337ced648aff4dd0090d54a064bf345e94c815525ce53c8d8b5`
- selection bundle SHA-256:
  `33b980831158fc33b4db4c21869e0aa1cf1b9c6ebd52e5bd78e81329c3c0f7b3`
- scikit-learn: `1.5.2`
- fixed-validation images opened during selection/preflight: `0`
- 45 repository tests, Python compilation, launcher shell syntax, complete
  cached-feature preflight, and `git diff --check` passed.

The H100 runner refits and verifies this exact result. Only after the selection
gate passes does it load frozen Qwen and extract the 25 already-frozen
validation features. Internal test and FoundIR remain protected. Noise, Blur,
Haze, and Rain remain blocked pending their own task calibration.

## Offline H100 command

```bash
CUDA_VISIBLE_DEVICES=0 smoke/run_qwen_lowlight_ridge_head.sh
```

The 1,500 train/selection features are reused, so only 25 validation features
require GPU extraction after the selection gate.
