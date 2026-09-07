# FoundIR FLUX.2 full-run failure analysis and speed plan — 2026-09-07

## Outcome

The H200 run beginning at `2026-09-07T12:04:43Z` did not remain in slow
steady-state training. It completed optimizer step 1 and failed at
`2026-09-07T12:20:15Z`. No full-training checkpoint was written, so restarting
from fresh zero LoRA/refiner initialization loses no reusable optimizer state.
The protected official FoundIR test was not accessed.

## Root cause

Rank 6 selected `foundir_train_0395700` while preparing the next optimizer
step. Its aligned LQ/GT pair is `1920x494`, while the FLUX training crop is
`512x512`. `PresentationDataset` rejected every pair whose short side was below
512. The worker exception was:

```text
RuntimeError: unaligned or undersized FoundIR pair: foundir_train_0395700
```

The earlier geometry preflight covered the Qwen inference crop of 448, not the
FLUX training crop of 512. Therefore this pair passed Qwen preprocessing but
failed FLUX training. The apparent NCCL failure was secondary: rank 6 stopped
before the next DDP gradient collective, while the other ranks entered the
collective and waited for the 600-second watchdog timeout. GPU 6 remaining at
0% utilization was a real stalled-rank signal, not merely a telemetry anomaly.

The complete frozen 685,495-row manifest contains 37 pairs below 512 pixels on
their short side. All 37 are in `10Rain`: 35 train, one validation, and one
sealed internal record. Their real LQ and GT files were opened and verified to
match the frozen manifest geometry exactly. The exact sorted record-ID set is
bound by SHA-256
`a0dc992fb2fa4c9a7191ac01a81584d5bca32bc4b035a1cb14bb4cee4576de8c`.

## Prevention

The trainer now applies
`bicubic_short_side_to_512_then_aligned_crop_v1`: only an undersized pair is
upscaled, and its LQ and GT receive the same deterministic aspect-ratio-
preserving bicubic geometry before the existing deterministic aligned crop.
Normal-size pairs are byte-array unchanged before crop. No record is removed
from the frozen data contract.

A dedicated preflight now scans all manifest dimensions and opens all 37
undersized LQ/GT pairs before eight model replicas are loaded. In addition,
each rank loads all microbatches for one optimizer step before any rank enters
DDP forward/backward. Rank readiness is reduced collectively; a worker decode,
shape, or crop exception is reported by all ranks immediately rather than
being hidden behind a gradient-all-reduce timeout. DataLoader workers also
have a bounded 120-second timeout, and NCCL desynchronization tracing is
enabled in the launchers.

## Controlled speed benchmark

The independent benchmark performs real forward, backward, optimizer, and
data-loading work for 20 fresh optimizer steps and excludes the first five
steps from throughput measurement. It writes no checkpoint and never reads the
official FoundIR test. Every candidate keeps global batch 32, the full
presentation plan, optimizer-step/LR schedule, loss, data order, checkpoint
selection, and evaluation protocol. Random noise and timesteps are now keyed
per presentation, so changing microbatch grouping does not change a sample's
random draw.

On H200 it compares `microbatch=2, accumulation=2, workers=4` against
`4,1,4` and `4,1,8`. On H100 it compares `1,4,4` against `2,2,4` and
`2,2,8`. Selection requires finite execution, mean loss-trajectory deviation
no greater than 5%, and at least 5% measured speedup. The selected runtime
schedule is hardware-family-specific and is cryptographically bound to the
frozen config and exact Git revision before the full trainer will accept it.
