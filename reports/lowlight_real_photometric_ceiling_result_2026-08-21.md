# Paired-real global-photometric ceiling result

Date: 2026-08-21 UTC

## Outcome

A per-image oracle containing only one shared RGB gamma, a 3 by 3 color
matrix, and RGB bias raises scene-balanced PSNR from `8.4356` dB to
`24.1209` dB on the 174-pair non-FoundIR development suite. The paired gain is
`+15.6853` dB, with a 95% scene-clustered, dataset-stratified bootstrap
interval of `+14.7717` to `+16.6036` dB.

This is strong evidence that global exposure/tone/color prediction must be the
first learned stage. It is not a deployable result: the oracle fits parameters
against each paired target and must never be used at inference.

## Metrics

| Weighting / dataset | Input PSNR | Oracle PSNR | Gain | Input SSIM | Oracle SSIM | Gain |
|---|---:|---:|---:|---:|---:|---:|
| Record weighted, all | 8.4126 | 23.8675 | +15.4549 | 0.2317 | 0.6247 | +0.3930 |
| Scene balanced, all | 8.4356 | 24.1209 | +15.6853 | 0.2475 | 0.6571 | +0.4096 |
| Scene balanced, LOL-v2 | 7.6175 | 24.7729 | +17.1554 | 0.1419 | 0.5832 | +0.4414 |
| Scene balanced, SICE | 9.2537 | 23.4688 | +14.2151 | 0.3532 | 0.7310 | +0.3778 |

The scene-balanced 95% intervals are:

- input PSNR: `8.0993` to `8.7987` dB;
- oracle PSNR: `23.3061` to `24.9770` dB;
- PSNR gain: `14.7717` to `16.6036` dB;
- SSIM gain: `0.3815` to `0.4355`.

## Alignment evidence

The evaluator estimates global translation from photometrically normalized
edge maps but never warps images used for PSNR or SSIM. Seven SICE records
exceeded the predeclared 8-pixel diagnostic threshold; none were removed or
corrected. All LOL-v2 records remained below the threshold. The strict metric
therefore includes the alignment penalty.

## Exact artifacts

- Config SHA-256:
  `256ac8ad204bc31339c7fc9e8f5b53efaa1efafea66e6dc9d3e451a25daaffd1`
- Per-record CSV SHA-256:
  `5a6b711a1e087ad0c734f63caed267c590b3eb691d5a066dd1671844c78bedad`
- Summary JSON SHA-256:
  `bad63bb7317a5dc955c8699c28906f83d765a13dbb86017d59a6c12d9957e5dd`
- Paired-development manifest SHA-256:
  `fee42cbf440a62c494bccc49e6f25b871ce8ea0044bcd825472e538dc3f5300d`

No project model inference and no official FoundIR test-image access occurred.

## Decision

Implement and cross-validate a bounded learned photometric head before adding
more DiT capacity. In parallel, measure the frozen VAE and released-teacher
ceilings. The learned head must predict parameters from the input alone and
must be evaluated leave-one-dataset-out; the oracle numbers are only an upper
bound.
