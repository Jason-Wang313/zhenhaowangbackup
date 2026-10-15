# FoundIR triple OneRestore-style R0 terminal audit

Date: 2026-09-26 UTC
Status: runtime success; scientific gate failure; output collapse

The 8×H200 R0 job started at 12:30:12 UTC and exited with `status=0` at
12:35:51 UTC. The trainer reported 240.26 seconds for three sequential
1,200-step arms. All three final checkpoints, 210 output PNGs, 210 provenance
sidecars, 210 metric sidecars, `training_result.json`, `result.json`, the
terminal report, and the end marker exist. All checkpoint SHA-256 values match
`training_result.json`; all output SHA-256 values match the 210 metric rows.
Each output is 448×448 RGB. No traceback, OOM, NCCL error, or nonfinite
metric appears in the run log. The official FoundIR test and the 60-image
FoundIR triple endpoint were not accessed.

| Arm | Development PSNR / LPIPS (30) | Held-out triple PSNR / LPIPS (40) |
|---|---:|---:|
| A unconditioned | 6.810193 / 0.967408 | 7.010534 / 0.987038 |
| B fixed categorical | 7.007366 / 0.984490 | 7.182890 / 1.004814 |
| C fixed categorical + CDRL | 6.140267 / 0.950947 | 6.636498 / 0.966714 |

The frozen primary gate failed 2/5 checks: B's development LPIPS is worse
than A by 0.017082, and C's development PSNR is worse than B by 0.867099 dB,
exceeding the allowed 0.2 dB drop. The other three checks passed.

An additional read-only pixel audit shows output collapse. Across the 30
development images, the mean input PSNR versus clean target is 17.545341 dB.
Every arm's output is worse than its input on all 30 images. Mean output pixel
value and fraction of pixels within one 8-bit level of either endpoint are:

| Arm | Mean output pixel | Endpoint saturation fraction | Output better than input |
|---|---:|---:|---:|
| A | 0.012591 | 0.998505 | 0/30 |
| B | 0.000005 | 0.999953 | 0/30 |
| C | 0.048034 | 0.986916 | 0/30 |

The checkpoint and metric artifacts are internally consistent, so this is a
scientific failure rather than an incomplete run. The model's residual output
uses a hard `[0,1]` clamp; saturation is a plausible mechanism for vanishing
gradients, but this audit does not establish a unique causal explanation.
These results cannot support a claim that the categorical descriptor or CDRL
improves restoration or triple generalization.

The authorized R0 has reached its terminal stop condition. Do not rerun it or
start R1, R2, continuous severity, spatial factorization, or CSR automatically.
Any next experiment needs a separately frozen design that addresses the
observed collapse and preserves the held-out triple boundary.
