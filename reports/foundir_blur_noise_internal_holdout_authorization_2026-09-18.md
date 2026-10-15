# Blur+Noise training-only internal holdout authorization

- Date: 2026-09-18 UTC
- Authorization: after the frozen step-1000 specialist passed its 128-record validation, the user explicitly instructed `execute` for the proposed one-shot internal-holdout confirmation.
- Permitted data: exactly the 128 `internal_test` records (83 source groups) already frozen in `manifests/foundir_blur_noise_restormer_pilot_v1.csv`; all originate from the FoundIR training split.
- Frozen model: `checkpoints/foundir_blur_noise_restormer_pilot_v1/step_001000.pt`.
- Prohibited: model selection, retraining, threshold tuning, FoundIR development access, and all FoundIR official-test access.
