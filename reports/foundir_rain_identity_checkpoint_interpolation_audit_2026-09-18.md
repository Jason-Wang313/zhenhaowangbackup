# Rain identity-checkpoint interpolation read-only audit

This audit used no GPU, no training, and no new validation data. It read the
same frozen 48 train-split development outputs from the `1:1:1` identity gate
and the `3:2:1` ratio gate. Sealed internal and official FoundIR test data were
not accessed.

Fixed output-blend results (`beta` is the contribution from the `3:2:1`
checkpoint):

| beta | PSNR | LPIPS | Directional gate | SOW PSNR margin | SOW LPIPS safety |
|---:|---:|---:|:---:|---:|---:|
| 0.00 | 32.3041 | .06515 | fail | -1.8959 dB (-5.5435%) | .01485 (18.5667%) |
| 0.25 | 32.2758 | .06165 | pass | -1.9242 dB (-5.6263%) | .01835 (22.9342%) |
| 0.50 | 32.1713 | .06018 | fail | -2.0287 dB (-5.9319%) | .01982 (24.7781%) |
| 0.75 | 32.0128 | .06043 | fail | -2.1872 dB (-6.3954%) | .01957 (24.4666%) |
| 1.00 | 31.8111 | .06231 | fail | -2.3889 dB (-6.9851%) | .01769 (22.1129%) |

`beta=.25` is the only fixed blend that passes all five parent-relative
directional checks. It nevertheless remains 1.9242 dB below the Rain SOW PSNR
target of 34.2.

Ceilings:

- per-image choice between the two checkpoints: 32.5657 PSNR;
- per-image continuous output-blend oracle, using GT to select beta on a .01
  grid: 32.6335 PSNR;
- independent per-image LPIPS oracle: .05722.

Even the unavailable-at-deployment GT oracle is 1.5665 dB (4.5803%) below the
34.2 PSNR target. Therefore fixed output alpha/checkpoint blending cannot close
the Rain SOW gap. A LoRA weight merge is not mathematically identical to output
blending, but this audit provides no evidence that spending another GPU run on
that single knob can recover the missing ~1.6-1.9 dB.

Decision: do not launch checkpoint interpolation, another sampling-ratio
sweep, or more steps. The current FLUX flow-only family is closed for Rain
pixel fidelity. The next bounded approach should directly optimize a
deterministic pixel-space residual restorer against paired Rain GT, with one
small source-disjoint gate before any scale-up.
