# FoundIR Blur Restormer v3 failure audit — 2026-09-09

## Outcome

The corrected v3 run was stopped after its step-1,000 development evaluation.
It was a scientific gate failure, not a runtime failure. The checkpoint and
development result are immutable and complete. No sealed internal or protected
official FoundIR test record was accessed.

| System | Blur PSNR | Blur LPIPS | Noise PSNR | Noise LPIPS |
|---|---:|---:|---:|---:|
| Frozen Qwen-routed v2 base | 25.3164 | 0.2881 | 39.4291 | 0.0659 |
| Restormer v2 step 1,000 | 25.0159 | 0.2759 | 39.4291 | 0.0659 |
| Restormer v3 step 1,000 | 25.2331 | 0.2349 | 39.4291 | 0.0659 |
| Stage-2 target | >=25.9 | <=0.18 | >=38.9 | <=0.10 |

Adding aligned raw LQ, using 512-pixel training crops, and restricting the
corpus to isolated Blur was effective: relative to v2 at the same step, v3
gained 0.2173 dB and reduced LPIPS by 0.04092. Relative to the frozen base,
LPIPS improved by 0.05314. However, PSNR was still 0.08324 dB below the base,
leaving gaps of 0.66689 dB PSNR and 0.05495 LPIPS to the target. The
predeclared continuation rule therefore rejected a blind run to step 3,000.

## Runtime and stop integrity

The successful run used eight H200 GPUs, reached step 1,075, and had no OOM,
rank stall, DataLoader failure, non-finite loss, or NCCL error. Across 3,360
telemetry rows, mean utilization including startup/evaluation was 96.71% and
maximum HBM use was 53,385 MiB per GPU. The process-group lock was available
after stopping; both the training log and GPU telemetry stopped updating.

An earlier v3 launch failed before step 1 because the newly constructed fusion
convolution was on CPU while the Restormer core was on CUDA. That implementation
bug was fixed by creating fusion parameters on the core device/dtype, moving
the complete wrapper to the selected device, asserting one parameter device
before DDP, and adding a regression test. The successful run proves that bug
did not recur.

## Deeper root cause

v3 exposed a real perception-distortion tradeoff rather than insufficient raw
information. Its learned residual strongly improved LPIPS but full-strength
application reduced pixel fidelity.

Three train/deployment mismatches remained:

1. v3 trained on all 105,057 isolated `01Blur` records, but the frozen router
   activates this refiner only for Qwen Blur L4/L5. 19,904 L1-L3 training
   records (18.9%) could never use the deployed gate.
2. Among deployment-active training rows, L5 represented 48,757/85,153
   (57.3%). The active development set contained 44/54 L5 records (81.5%). The
   hardest cases were therefore materially under-sampled.
3. At step 1,000 the weighted loss contributions were approximately 0.04029
   Charbonnier, 0.00662 RGB MSE, 0.00189 edge L1, and 0.01624 LPIPS. MSE—the
   quantity directly determining PSNR—was numerically much weaker than the
   L1-like and perceptual terms. In addition, evaluation averages per-record
   PSNR, while the training MSE was globally pixel-averaged.

The v3 architecture itself should be retained because it delivered a much
larger LPIPS gain than v2. The next gate is a short corrective phase initialized
from the immutable v3 step-1,000 checkpoint: train only routed L4/L5 records,
sample L5 with probability 0.75, reduce the learning rate to 2e-5, make MSE
dominant, and add a per-image log-MSE PSNR surrogate. The first checkpoint is
only 250 new optimizer steps. Development also evaluates residual scales
0/0.5/0.75/1/1.25; scale 0 provides an explicit non-regressing base candidate.

The corrective phase may continue only if the early Blur PSNR/LPIPS trajectory
supports it. This rule is enforced in the trainer: after each milestone it
automatically stops unless PSNR is at least the frozen base, LPIPS is no worse
than the v3 source, and the worst normalized gate margin strictly improves. It
also stops immediately when all gate checks pass. Internal data remains
unavailable until a winner is frozen, and the protected official FoundIR test
remains untouched.
