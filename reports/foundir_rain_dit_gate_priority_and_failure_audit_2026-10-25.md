# Rain DiT gate before MoE: decision and failure audit (2026-10-25)

## User decision

Work in this order only: Rain FLUX/DiT must first pass the frozen SOW
development gate (PSNR >= 34.2 dB and LPIPS <= 0.08); next build an ordinary
working image-only DiT-MoE; finally add projected VLM tokens to its router.
Blur DiT has already passed; do not repeat it. The 1,000-step image-only MoE
failed badly and is not a working MoE baseline. Do not launch its VLM arm.

## Existing Rain experiment is terminal, not unfinished

Source: `checkpoints/foundir_rain_flux2_direct_endpoint_distillation_formal_v1/result.json`,
`logs/foundir_rain_flux2_direct_endpoint_distillation_formal_v1.log`, and
`configs/foundir_rain_flux2_direct_endpoint_distillation_formal_v1.json`.
The independent direct-endpoint formal run completed all 3,000 optimizer
steps on 8xH100 in 6.470578311735836 wall hours / 51.76462649388669
allocated GPU-hours. It used the 38,334-record, 5,780-source training
manifest and the frozen 100-record, source-disjoint development gate. The
reported protected-data access flags are false.

| Step | Development PSNR | Development LPIPS |
| ---: | ---: | ---: |
| 250 | 29.986104933238117 | 0.09817367502488196 |
| 500 | 30.719960865090016 | 0.0938842959003523 |
| 1,000 | 31.295812397616352 | 0.09385522781405597 |
| 2,000 | **32.085315339143946** | **0.08882607079111039** |
| 3,000 | 31.900863136522702 | 0.08921627756208181 |

No checkpoint passed either gate. Step 2,000 is 2.114684660856055 dB
below the PSNR gate and 0.008826070791110387 above the LPIPS ceiling.
More training with the unchanged objective is not supported by the observed
regression. At step 3,000, every one of the 24 development records whose
input PSNR was >=34 dB lost PSNR; 14 of the remaining 76 also lost PSNR.
Thus preservation of already-good regions is a major issue, but not the only
issue: many harder inputs also remain poorly restored.

## No-training amplitude feasibility audit

`smoke/audit_foundir_rain_direct_endpoint_blend_v1.py` reads only the saved
step-2,000 development triptychs. The images are PNG-quantized; its PSNR is
exploratory and must not replace the original evaluator or be called a model
result. The computed mean per-image PSNR for fixed input-to-FLUX output
amplitudes 0, 0.25, 0.5, 0.75, and 1 is respectively 29.543657998644395,
30.84030921988115, 31.896249031492342, 32.545104133834506, and
32.085315603138305 dB. Even a non-deployable GT-only choice of the best
amplitude **per image** among those five reaches only 33.44433927487587 dB.
Therefore a fixed blend or per-image strength selector cannot by itself meet
34.2 dB. No LPIPS is computed by this audit, and no protected data is read.

## Next Rain repair must change corrective information

The next experiment should be a distinct observation-anchored Rain action:
retain the working FLUX DiT features but predict a local RGB correction
*relative to the input*, so unchanged pixels have an exact identity path.
The current direct-endpoint model instead decodes an entire predicted clean
latent, which can damage unaltered pixels. The previously implemented
`BoundedPixelRefiner` is anchored to the decoded FLUX image rather than the
observation; blindly appending it is not this test. This hypothesis remains
unproven and must not be described as a solution yet.

Before a GPU launch, freeze (a) an observation-anchored DiT correction and
its image-only small-head control, so the DiT's contribution cannot be
misattributed to a CNN; (b) train/validation source identities and the same
100-record gate; (c) a short engineering qualification and measured formal
budget. Evaluate both PSNR and LPIPS without altering the gate or selecting
from protected test data. If the DiT branch does not beat its image-only
control and pass the Rain gate, do not advance to MoE or claim a DiT Rain
expert. No GPU run was launched by this audit because the current Codex
environment has no `nvidia-smi`/CUDA node.
