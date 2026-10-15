# FLUX.2 Haze frozen scalar-residual read-only audit

## Scope

Following the user's frozen task order, this audit evaluates Haze after
Low-light and does not touch Rain. It reuses the already-generated isolated
Haze step-400 `Input | FLUX.2 Output | Ground Truth` PNGs and tests the same
lowest-complexity correction that was useful for Noise and Low-light:

`candidate = LQ + alpha * (FLUX output - LQ)`

No FLUX inference, training, model change, full-development, sealed-internal,
or official-test access occurred. GPU wall time and GPU-hours were both zero.

Frozen evidence:

- checkpoint: `checkpoints/foundir_four_task_flux2_capacity_sanity_v1/haze/step_000400.safetensors`
- checkpoint SHA-256: `e96fed2f1e399edeb3ea172333a47b8363c72819bb3dff3637dd1a5d21c52439`
- calibration triptych aggregate SHA-256: `6b7cd292a3ebb150db47dc711bb72aa5e604beb3d02b6780f74c4c05afbf661e`
- validation triptych aggregate SHA-256: `2f139a976893b334fbf0e2cbc9ef069c24c027c9235ec14852197e973042e386`
- calibration: 16 prior training records
- validation: 16 prior source-group-disjoint records, used once after selection

## Frozen selection and result

The alpha grid was `[0, .025, ..., .975, 1]`. Pixels were rounded and clipped
to uint8 before PSNR and AlexNet LPIPS. Calibration selection maximized mean
PSNR subject to mean LPIPS <= `.17`, leaving `.02` margin to the Haze SOW
LPIPS maximum `.19`. Calibration alone selected `alpha=.95` and measured
`29.782782 PSNR / .048885 LPIPS`.

Applied once to the untouched source-disjoint validation set:

| Validation arm | Records | PSNR | LPIPS |
|---|---:|---:|---:|
| Input | 16 | 13.880965 | .324857 |
| Base FLUX | 16 | 19.956481 | .138165 |
| Selected alpha=.95 | 16 | **19.949140** | **.139469** |

Against Haze Stage-2 SOW (`PSNR >=22.1`, `LPIPS <=.19`), candidate PSNR
fails by `-2.150860 dB`, a **-9.7324%** margin relative to the 22.1 floor.
LPIPS passes with `.050531` safety margin, **26.5951%** relative to the .19
cap. Passing LPIPS alone is insufficient: the two-metric SOW gate fails.

Relative to base FLUX, candidate loses `.007341 dB` mean PSNR and worsens
LPIPS by `.001304`; it wins per-record PSNR on 68.75% and LPIPS on 56.25% of
records. The worst PSNR delta is `-.965925 dB` on
`foundir_train_0217755`. Relative to input, it gains `6.068175 dB` and
improves LPIPS by `.185388`, with 81.25%/93.75% PSNR/LPIPS win rates; the
worst input-relative PSNR delta is `-3.637457 dB` on
`foundir_train_0213312`.

## Comparison and decision

| Small validation | SOW PSNR margin | SOW LPIPS safety margin | Gain over base | Scale decision |
|---|---:|---:|---:|:---:|
| Noise, 16 images | `+.9336 dB` (`+2.40%`) | `.03239` (`32.39%`) | strong PSNR gain | proceed; later full-development passed |
| Low-light, 16 images | `+.1267 dB` (`+.6706%`) | `.09669` (`34.53%`) | `+1.5441 dB`, LPIPS better | proceed; later full-development passed |
| Haze, 16 images | `-2.1509 dB` (`-9.7324%`) | `.05053` (`26.60%`) | neither mean metric improves | **do not proceed** |

Haze is materially weaker evidence than either prior 16-image pass. The
selected `.95` is nearly unchanged FLUX, so scalar residual blending has not
found a scalable correction. The frozen 1,580-record Haze full-development
run must not be launched from this result.

Visual review of all sixteen existing Haze triptychs shows that FLUX generally
removes haze and preserves scene structure. The dominant residual is global
photometric/color mismatch: some outputs are too dark or over-contrasted,
vegetation can be too yellow/dark, and some targets are brighter, cooler, or
differently saturated. This points to at most one more simple,
calibration-only global photometric gate (for example, a per-channel affine or
low-degree luma/color transform), frozen on the sixteen calibration images
without validation tuning. It should be scaled only if the next untouched
gate reaches both SOW thresholds, improves over base, and remains visually
acceptable. If that simple gate fails, stop Haze method expansion rather than
add a complex module.

Runtime: existing artifacts only; GPU wall time `0`, GPU-hours `0`. No Haze
full-development or Rain job was started.
