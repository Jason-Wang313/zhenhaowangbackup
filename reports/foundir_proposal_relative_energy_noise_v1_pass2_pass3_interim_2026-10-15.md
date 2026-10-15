# FoundIR proposal-relative energy Noise v1 — pass-2/pass-3 interim

Date: 2026-10-15 UTC

Run `noise_energy_v1_pass1` is continuing from its corrected pass-1 checkpoint
toward the frozen five-pass endpoint at 8,530 optimizer steps. This report
records the automatically produced pass-2 and pass-3 source-disjoint
development validations. It is not a terminal result.

## Fixed 256-record validation

Higher PSNR is better; lower LPIPS is better. The frozen proposal remains
`43.124160178005695/.050362067760033824` at every checkpoint.

| Checkpoint | Direct PSNR / LPIPS | Energy-1 PSNR / LPIPS | Energy-2 PSNR / LPIPS | Energy-3 PSNR / LPIPS |
|---|---|---|---|---|
| Pass 2, step 3,412 | `43.25875762850046/.06282198585995502` | `43.21375361829996/.056435911629705515` | `43.212642684578896/.060747250086023996` | `42.987051747739315/.054700696127838455` |
| Pass 3, step 5,118 | `43.27525299042463/.059980700971209444` | `43.19604617357254/.05484414789316361` | `43.23734363913536/.06065949178446317` | `43.11910688877106/.057597222285039607` |

At pass 3, Energy-1 is the best perceptual energy arm, but it is still only
approximately `+.07189 dB` PSNR and `+.00448` LPIPS relative to the frozen
proposal. LPIPS therefore remains worse, not better. No energy arm meets the
predeclared minimum `+.30 dB/-.005 LPIPS` gate. The direct refiner also retains
the same tradeoff: higher PSNR and worse LPIPS.

The run is numerically stable and remains authorized to finish the frozen
five-pass endpoint. These intermediate results do not justify changing loss,
step count, validation data, or checkpoint selection during the run.

## Artifact identity

- pass-2 checkpoint:
  `c00d90ba20a38b71d6688c80e71962686ffe5e228dc920f071906c8d66cae85a`;
- pass-2 validation:
  `134669a121bd56e70806c3480855f5c1de45032b00203ba0fe375328b5c5771e`;
- pass-3 checkpoint:
  `de246acbb18d4a42dbd45ca9d306594ab243495fd5e724c82eadac511af1b996`;
- pass-3 validation:
  `5965495c7a29f95531bc8e2e60fba3860e714388a06f93994fbe7ae989534dd6`.

Protected-data flags remain false in both validation artifacts.
