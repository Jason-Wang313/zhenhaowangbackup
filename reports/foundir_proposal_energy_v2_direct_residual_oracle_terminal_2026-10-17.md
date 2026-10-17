# FoundIR V2 direct-residual oracle — terminal result

Date: 2026-10-17 UTC

## Outcome

The GT-only feasibility diagnostic completed normally on all 256 records from
the fixed source-disjoint development subset. It performs no training and did
not access official, sealed, internal, or frozen-660 data. It is an upper-bound
diagnosis, not a deployable method result.

The oracle does **not** find any combination of the learned direct residual
that simultaneously clears the predeclared `+.30 dB PSNR` and `-.005 LPIPS`
gate.

| Output | PSNR | LPIPS | PSNR gain | LPIPS reduction |
|---|---:|---:|---:|---:|
| Frozen proposal | `43.124160178005695` | `.050362067760033824` | — | — |
| Direct residual at 100% | `42.97123210132122` | `.04185807168323663` | `-.15292807668447495` | `.008503996076797193` |
| GT-fitted 8x8 patch amplitude | `43.16661264002323` | `.04641580081442953` | `.04245246201753616` | `.003946266945604293` |
| Best Pareto point satisfying LPIPS gate | `43.16162772476673` | `.044729646269843215` | `.03746754676103592` | `.005632421490190609` |

The 8x8 patch oracle selected local residual amplitudes for 255 images and
kept the unchanged proposal for one. This proves that local selection is
directionally useful: it improves both metrics instead of trading one for the
other. However, its PSNR gain is only about 14% of the required `.30 dB`, and
its LPIPS reduction is only about 79% of the required `.005`. Even after
allowing a separate GT-selected candidate for every image, no Pareto point
reaches `.30 dB` PSNR gain. Therefore the frozen direct residual lacks enough
corrective information; learning a better gate over exactly these candidates
cannot pass the joint target.

## Protocol and identity

The rerun proposal differs from the checkpoint-recorded proposal by only
`.00046180933713912964 dB` PSNR and `.000002161842530767899` LPIPS, within the
frozen reproduction tolerance and negligible relative to the scientific gate.
The successful run lasted from `2026-10-17T09:19:43Z` to `09:23:10Z`: `207`
seconds (`.0575` wall hours) on eight GPUs, or about `.46` allocated GPU-hours.
Including the earlier run stopped by the overly strict reproduction assertion,
the audit used about `.94` allocated GPU-hours.

Machine-readable result:
`reports/foundir_proposal_energy_v2_direct_residual_oracle_step005118.json`.
Its SHA-256 is
`6400d78624dd2a9fe2cc2f67e622dc9371fd44243346df4f5977c6c17fc8a317`.
