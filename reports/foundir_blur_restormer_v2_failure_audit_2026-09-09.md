# FoundIR Blur Restormer v2 failure audit — 2026-09-09

## Outcome

The exact-identity-initialized 26.13M Restormer gate was stopped after its
predeclared step-3,000 development evaluation. It was a scientifically failed
trajectory, not a runtime failure. No internal or official FoundIR test record
was accessed.

| System | Blur PSNR | Blur LPIPS | Noise PSNR | Noise LPIPS |
|---|---:|---:|---:|---:|
| Frozen Qwen-routed v2 base | 25.3164 | 0.2881 | 39.4291 | 0.0659 |
| Restormer step 1,000 | 25.0159 | 0.2759 | 39.4291 | 0.0659 |
| Restormer step 3,000 | 25.2088 | 0.2717 | 39.4291 | 0.0659 |
| Stage-2 target | >=25.9 | <=0.18 | >=38.9 | <=0.10 |

At step 3,000, PSNR was still 0.1075 dB below the frozen base. LPIPS improved
by only 0.00419 over the additional 2,000 steps and remained 0.09168 above the
target. That trajectory did not justify spending another 9,000 optimizer
steps.

## Runtime health

The run reached step 3,125 without traceback, OOM, rank stall, non-finite
loss, or DataLoader timeout. Across 3,192 telemetry samples, mean GPU
utilization including startup/evaluation was 89.25%; maximum memory was only
16,105 MiB per H200. This rules out hardware capacity and input-pipeline
throughput as the reason for scientific failure.

## Root-cause audit

Three design mismatches explain the observed PSNR/LPIPS behavior:

1. The refiner received only the frozen base output. Details suppressed by the
   base were no longer directly available, so additional capacity could polish
   perceptual appearance but could not reliably reconstruct the target pixels.
2. Training used 256-pixel crops while the frozen development protocol uses
   512-pixel crops. Restormer's global spatial attention therefore trained and
   evaluated with different context scales.
3. The training pool mixed isolated Blur with Blur+Noise and Lowlight+Blur
   categories. The deployed isolated-Blur gate required a focused deblurring
   residual, but a large fraction of training presentations also asked the
   same component to correct exposure/noise interactions. The loss was not
   task-factorized, which diluted the isolated-Blur objective.

The next bounded gate must therefore expose both the frozen base and aligned
raw LQ, train at the exact 512-pixel evaluation scale, and start with the
isolated `01Blur` corpus. It must retain an exact identity initialization and
the same frozen Qwen routing and development records. Step 1,000 remains the
first decision point; internal and official test data remain inaccessible.
