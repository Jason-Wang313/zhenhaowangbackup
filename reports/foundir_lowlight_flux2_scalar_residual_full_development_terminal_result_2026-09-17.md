# FLUX.2 Low-light frozen scalar-residual full-development terminal result

## Outcome

The user-authorized one-shot isolated Low-light evaluation completed normally
on all 1,400 source-group-disjoint development records. Frozen alpha `.925`
passes both formal Stage-2 SOW mean metrics and the fixed visual review.

| System | PSNR | LPIPS | PSNR >=18.9 | LPIPS <=.28 |
|---|---:|---:|:---:|:---:|
| Input | 9.702180 | .485682 | fail | fail |
| Paired alpha=1 FLUX | 18.295276 | .165303 | fail | pass |
| Frozen alpha=.925 candidate | **19.779135** | **.154058** | **pass** | **pass** |

Stage-2 SOW-relative margins use the frozen reporting convention requested by
the user:

- PSNR absolute margin: `+0.879135 dB`; percentage margin:
  `(19.779135 - 18.9) / 18.9 = +4.6515%`.
- LPIPS absolute safety margin: `.125942`; percentage safety margin:
  `(.28 - .154058) / .28 = 44.9792%`.

Relative to paired base FLUX, the candidate gains `+1.483859 dB`, improves
LPIPS by `.011244`, wins PSNR on `86.07%`, and wins LPIPS on `84.00%` of
records. Relative to input it gains `+10.076955 dB`; PSNR improves on `95.64%`
and LPIPS improves on `98.14%` of records.

## Descriptive tails

Tail evidence remains mandatory but is not part of the user's formal SOW mean
gate. Thirty-two records regress by more than 1 dB and sixteen regress by more
than 2 dB versus input. Worst versus input is `foundir_train_0510526`:
candidate `11.343157/.459535`, input `21.335590/.592838`, a `-9.992432 dB`
PSNR delta, although candidate LPIPS is better. Worst versus base is
`foundir_train_0512275`: candidate `27.300735/.070482`, base
`34.200887/.065829`, a `-6.900153 dB` delta; the candidate remains objectively
high-quality and far better than its input `7.967234/.498535`.

The lowest candidate PSNR is `6.463374` on `foundir_train_0508736`; it still
improves input by `1.164546 dB`. The highest candidate LPIPS is `.553613` on
`foundir_train_0509629`; it improves PSNR over input and base but exposes a
remaining difficult perceptual tail.

## Visual and artifact audit

All sixteen prospectively frozen `Input | Candidate | Ground Truth`
triptychs, spanning sixteen source groups, were reviewed. They pass for visible
illumination recovery, structure, object/text identity, and absence of gross
hallucination or blocking artifacts. Some examples retain imperfect exposure
or color-temperature matching, especially bright retail/package scenes; the
visual result is strong but not perfect per-record photometric recovery.

Hash-bound visual review:
`reports/foundir_lowlight_flux2_scalar_residual_full_development_visual_review_v1.json`,
SHA-256 `bf183ca66ed3628ecaedabc49cc9a3f9e7614d4d70f05608e9bd22e583924ed1`.

Independent read-only audit confirms exactly 1,400 unique sidecars, 21 source
groups, 16 visuals, frozen config/checkpoint/alpha identities, finite aggregate
agreement, and false protected-data flags. Result SHA-256:
`2673ace6eac831b06684826cd732fbd14c57974eef81eb86e0260cf931bd93ca`.
There was no traceback, OOM, NCCL, nonfinite, or runtime failure.

## Runtime and stop

The runner started `2026-09-17T05:08:13Z` and ended
`2026-09-17T05:29:49Z`: actual end-to-end wall time **21m36s** on homogeneous
8xH100, approximately **2.88 aggregate GPU-hours**. The evaluator's measured
distributed phase was `1240.355 s` (`20m40s`, about `2.76 GPU-hours`).

Sealed internal and official FoundIR test were not accessed. Active hold:
`flux2_lowlight_scalar_full_development_result-b8b6567c249a58b1`. Stop here;
Haze and Rain are not automatically authorized by this terminal result.

For every later validation report, include both absolute margins and the same
percentage Stage-2 SOW margins for PSNR and LPIPS.
