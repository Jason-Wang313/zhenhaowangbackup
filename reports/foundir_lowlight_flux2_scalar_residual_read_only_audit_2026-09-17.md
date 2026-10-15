# FLUX.2 Low-light frozen scalar-residual read-only audit

## Scope

Following the user's frozen task order, this audit evaluates Low-light before
Haze and does not touch Rain. It reuses the already-generated isolated
Low-light step-400 `Input | FLUX.2 Output | Ground Truth` PNGs and applies the
lowest-complexity correction:

`candidate = LQ + alpha * (FLUX output - LQ)`

No FLUX inference, training, model change, development/sealed/official-test
access, or GPU work occurred. Under the active advisor hold this is read-only
feasibility evidence, not a new formal experiment pass.

Frozen evidence:

- checkpoint: `checkpoints/foundir_four_task_flux2_capacity_sanity_v1/lowlight/step_000400.safetensors`
- checkpoint SHA-256: `592e49ca2606565c5aeec8aadeb5466cf0c34b30024d5c02f0f2292ad28c90d8`
- calibration triptych aggregate SHA-256: `bbe04ff20dcb53ccc780b28212886fdfe61255167aa826cc3bdf7205bdfb67e7`
- validation triptych aggregate SHA-256: `8799aca28f7460126befe0cd2e8ec81d2db2b436fa1259fc068b6cd471b2d60c`
- calibration: 16 prior training records
- validation: 16 prior source-group-disjoint records, used once after selection

## Frozen selection and result

The alpha grid was `[0, .025, ..., .975, 1]`. Pixels were rounded and clipped
to uint8 before PSNR and AlexNet LPIPS. Calibration selection maximized mean
PSNR subject to mean LPIPS <= `.26`, leaving `.02` margin to the Low-light SOW
LPIPS maximum `.28`. Calibration alone selected `alpha=.925`.

| Split / arm | Records | PSNR | LPIPS |
|---|---:|---:|---:|
| Calibration, selected alpha | 16 | 29.37054 | .051762 |
| Validation input | 16 | 9.10589 | .515858 |
| Validation base FLUX | 16 | 17.48268 | .192500 |
| Validation, selected alpha | 16 | **19.02674** | **.183309** |

The untouched validation average passes both Low-light SOW thresholds (`PSNR
>=18.9`, `LPIPS <=.28`). Its margins are `+.12674 dB` PSNR and `.09669` LPIPS.
Relative to base FLUX, it gains `1.54406 dB` mean PSNR and improves LPIPS by
`.009207`; PSNR and LPIPS each improve on 12/16 records. Relative to input, it
gains `9.92085 dB`, PSNR improves on 15/16 records, and LPIPS improves on
16/16 records.

## Tail and visual interpretation

The worst PSNR delta versus input is `-4.91232 dB` on
`foundir_train_0494919`. The worst delta versus base FLUX is `-1.43432 dB` on
`foundir_train_0510167`. These are mandatory descriptive tail evidence, not
part of the user's stated mean SOW gate.

Inspection of representative best, worst-input, worst-base, and high-LPIPS
triptychs shows that alpha `.925` remains visually very close to the existing
FLUX output, as expected from only a 7.5% reduction in restoration strength.
It introduces no new visible structural failure in those examples and mildly
reduces over-correction. It cannot repair pre-existing color-temperature
outliers such as `foundir_train_0509643`; that record remains far from its
yellow ground truth and has selected-candidate LPIPS `.43829`.

The correct conclusion is **small source-disjoint feasibility pass, ready for
one frozen full-development check**, not “Low-light solved.” Because the PSNR
margin is only `.12674 dB`, scaling is necessary before proceeding to Haze.
The next candidate must keep checkpoint step 400, alpha `.925`, the exact
uint8 formula, and formal checks limited to the Low-light mean SOW metrics,
while retaining per-record tails and fixed visuals descriptively. No alpha
retuning on validation and no complex correction module are justified.

Runtime: CPU-only reuse of existing artifacts; GPU wall time `0`, GPU-hours
`0`. The final vectorized metric calculation took under one minute after
dependency import. Rain was not accessed.
