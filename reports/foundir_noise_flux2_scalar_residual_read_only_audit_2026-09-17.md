# FLUX.2 Noise frozen scalar-residual read-only audit

## Scope

At the user's instruction, this audit applies the Blur breakthrough formula to
the already-generated isolated-Noise step-400 outputs:

`candidate = LQ + alpha * (FLUX output - LQ)`

It uses sixteen prior training triptychs as calibration and sixteen prior
source-group-disjoint validation triptychs once after alpha selection. It does
not rerun FLUX, update weights, access development/sealed/official test data,
or consume GPU time. Because the terminal advisor hold remains active, this is
post-hoc read-only feasibility evidence and not a new formal experiment pass.

Frozen evidence:

- checkpoint: `checkpoints/foundir_four_task_flux2_capacity_sanity_v1/noise/step_000400.safetensors`
- checkpoint SHA-256: `1b4ed05ad5ad403a3d7fcc633eca97cdb69d00454e357da0c3d3131ac2c3ff1a`
- training-triptych aggregate SHA-256: `340de52c4574f8084fd31de359b037f5aa215aeb6b561c5e89e11b1e48f377a2`
- validation-triptych aggregate SHA-256: `561b80a21f72ef6e69e4cb7a18bc42dcf3659acd7d4dada3d2ce941a0200bd07`

## Frozen selection and result

The alpha grid was `[0, .025, ..., .975, 1]`. Candidate pixels were rounded
and clipped to uint8 before PSNR/LPIPS. Calibration selection maximized mean
PSNR subject to mean LPIPS <= `.09`, providing `.01` margin to the Noise SOW
LPIPS limit `.10`. Calibration alone selected `alpha=.30`.

| Split | Records | PSNR | LPIPS |
|---|---:|---:|---:|
| Calibration | 16 | 41.37065 | .054805 |
| Source-disjoint validation | 16 | 39.83359 | .067606 |

The validation averages pass both Noise SOW thresholds (`PSNR >=38.9`, `LPIPS
<=.10`). This establishes that the same low-degree-of-freedom Pareto logic
that unlocked Blur also works for Noise averages.

## Tail limitation and conclusion

Against the unmodified validation input, candidate mean PSNR is `.51256 dB`
lower, PSNR improves on 50% of records, and the worst regression is `5.63777
dB` on `foundir_train_0187477`. A conservative identity/tail gate therefore
fails. The result must be stated as **SOW averages pass / tail-safety fails**,
not as an unconditional Noise pass.

The next defensible design is a GT-free, inference-time identity/strength-aware
alpha policy calibrated without validation GT. More fixed-alpha searching on
validation, more training steps, Qwen conditioning, mixed degradation, sealed
internal access, and official-test access are not authorized by this audit.

Runtime: CPU-only reuse of existing artifacts; GPU wall time `0`, GPU-hours
`0`; metric calculation completed in under one minute after artifact lookup.
