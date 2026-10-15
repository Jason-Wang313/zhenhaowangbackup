# FoundIR triple OneRestore-style R0 terminal result

Status: R0 terminal; stop after this diagnostic.

| Arm | Development PSNR | Development LPIPS | Held-out triple PSNR | Held-out triple LPIPS |
|---|---:|---:|---:|---:|
| A_unconditioned | 6.810193 | 0.967408 | 7.010534 | 0.987038 |
| B_fixed_categorical | 7.007366 | 0.984490 | 7.182890 | 1.004814 |
| C_fixed_categorical_cdrl | 6.140267 | 0.950947 | 6.636498 | 0.966714 |

Primary gate: `FAIL`

- B_psnr_not_worse_than_A: `PASS`
- B_lpips_not_worse_than_A: `FAIL`
- C_psnr_within_0.2db_of_B: `FAIL`
- C_lpips_not_worse_than_B: `PASS`
- heldout_triple_metrics_reported_without_selection: `PASS`

The held-out triple set was accessed only after all three final checkpoints were frozen and was not used for checkpoint selection.
This is a synthetic diagnostic, not a FLUX.2 mainline result and not an official FoundIR/SOW result.
The implementation is OneRestore-style, not an exact CDD-11 reproduction: it uses frozen seeded Gaussian descriptors rather than GloVe semantic descriptors, and a stable `[0,1]` residual clamp rather than the public output normalization.
Neither the official FoundIR test nor the 60-image FoundIR triple endpoint was accessed.

Next action: **STOP**. Do not automatically enter R1, R2, continuous severity, spatial factorization, or CSR.
