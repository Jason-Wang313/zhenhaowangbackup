# Four-task FLUX.2 fail-fast capacity sanity design

## Decision and scope

The advisor directed the project to train all degradation types jointly next. The user explicitly authorized the agreed 1→5 sequence: first run one bounded independent capacity sanity gate for Noise, Haze, Rain, and Low-light; then design the balanced five-single-task joint baseline; evaluate each task separately; restore Qwen type/severity conditioning only after that baseline works; and add mixed degradation last.

This document freezes phase 1 only. It does not authorize an automatic phase-2 joint launch.

## Frozen experiment

- Backbone: FLUX.2 Klein Base 4B, fresh independent rank-32 LoRA per task.
- Frozen components: VAE and native Qwen3 text encoder.
- Conditioning: one fixed task-specific restoration prompt; no Qwen degradation prediction.
- Data: isolated single-degradation FoundIR training archive only.
- Per task: 16 unique train source groups and 16 unique source-disjoint validation groups, one deterministic 512×512 crop each.
- Optimization: 400 steps, global batch 8, LR `1e-4`, checkpoints 50/100/200/400.
- Inference: official 50-step FLUX.2 shifted schedule, guidance 1.0.
- Evidence: fixed train and validation `Input | Output | GT` triptychs plus PSNR/LPIPS at every checkpoint.
- Validation is descriptive and cannot select task, prompt, checkpoint, loss, or threshold.

The four arms run concurrently on one homogeneous 8×H100/H200 host as four independent two-GPU jobs. Each arm uses microbatch 1 and accumulation 4, preserving global batch 8. There is no cross-task gradient sharing.

## Gate and interpretation

Each task receives one bounded capacity gate: probe-loss ratio at most 0.35, mean train PSNR gain at least 2 dB, mean train LPIPS improvement at least 0.05, and at least 75% of train records improving each metric. Task-specific Stage-2 SOW thresholds are reported separately as a tiny-train proxy, not claimed as development or official-test performance. Human visual review is mandatory.

All four terminal results are audited together before any five-task joint baseline is designed or launched. Failure does not trigger an automatic retry, LR sweep, Qwen conditioning, mixed degradation, sealed-internal evaluation, or official FoundIR test access.

## Why this is not the failed historical v3 rerun

Historical five-type trainers assumed precomputed Qwen conditions, admitted mixed categories, and combined multiple mechanisms. This gate uses only the verified generic FLUX loading, native reference-image path, LoRA, optimizer, official inference, and metric primitives. It removes Qwen degradation predictions, mixtures, pixel refinement, and protected splits so single-task capacity is identifiable.
