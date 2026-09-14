# Z-Image latent-first v2 terminal audit

DIAGNOSTIC ONLY / 非主线. This is a supervised one-step endpoint repair, not
a successful teacher or distillation result and not an official FoundIR/SOW
evaluation. The fixed 16+16 crop experiment completed normally but failed both
the automatic development gate and human visual review.

## Verified execution

- Launch: 2026-09-14 11:03:53Z; terminal status 0 at 11:17:51Z on homogeneous
  8x H100; launch revision `3902381ef2be0b63c09003967c2c4eb62ab10fdc`.
- Exact strict baseline replay and all eight distributed rehearsals passed.
- All 1,200 optimization rows, six FP32 checkpoints (476 tensors each), 14
  aggregate snapshots, 224 triptychs and 224 sidecars are complete and valid.
- Result SHA-256:
  `f2924d157f9fe9c2814122ec96f1e1cb5fa543016df0b61cdbfa84f303b8800f`.
- Final checkpoint SHA-256:
  `b90f5782e1ffa5db3c78ed53295472a5b3aac307d423ac03c067c3a00e23db92`.
- One finite preclip gradient spike occurs at joint step 891: 2543.1211; clip
  1 activates. Development had already degraded substantially by step 800, so
  this spike cannot alone explain the generalization failure.

## Fixed metrics

| Step | Train PSNR / LPIPS | Development PSNR / LPIPS | Train / dev latent MSE |
|---:|---:|---:|---:|
| Input | 24.7232 / .3085 | 23.9045 / .3264 | n/a |
| 0 | 25.4065 / .2551 | 22.8756 / .3519 | .186149 / .235011 |
| 200 | 26.5550 / .3472 | 22.1416 / .4575 | .144913 / .233491 |
| 400 | 27.3084 / .3088 | 21.6275 / .4644 | .128942 / .249622 |
| 600 | 27.6990 / .2821 | 21.5719 / .4651 | .114695 / .262173 |
| 800 | 27.9370 / .2669 | 21.4969 / .4703 | .104120 / .273831 |
| 1000 | 28.8981 / .1478 | 21.5454 / .4623 | .099838 / .279537 |
| 1200 | 29.2780 / .1247 | 21.4293 / .4635 | .094283 / .287624 |

The recipe clearly learns the 16 training crops: both image metrics and latent
MSE pass there. Development moves in the opposite direction: PSNR loses
2.4753 dB, LPIPS worsens by .1371, no record improves PSNR, and latent MSE rises
from .235011 to .287624. Joint RGB/LPIPS training after step 800 improves the
training images but does not repair development.

## Visual review

The strong-blur training statue 0042483 recovers its coarse pose, showing real
memorization capacity, but remains smeared and artificially textured compared
with GT. Near-clean training 0091486 is unnecessarily altered: 36.883 to
31.757 dB and .047 to .085 LPIPS.

Development review fails decisively. Leaves 0092666 acquire heavy grain and
ghosting (30.097 to 23.778 dB; .152 to .392 LPIPS). Text/logo example 0100870
has block texture and distorted characters. Bus 0055608 has severe geometric
warping and blocks. Forest 0060122 remains blurred and gains artificial web
texture rather than credible detail. These are not acceptable restoration
outputs and agree with the aggregate failure.

## Decision

Do not select an earlier checkpoint: none passes development and step 1200 was
predeclared. Do not start full training, call this teacher distillation, tune
the decoder from this model, or add another LR/step sweep. The agreed two-round
decision budget is complete. Evidence now supports a formulation/generalization
problem: the 79M-parameter LoRA can memorize 16 crops but the supervised
endpoint mapping does not generalize across the source-disjoint 16 crops.

The next task is a consequential scientific redesign and should be handled
with Astra medium per the recorded model contract. It must decide a genuinely
different restoration formulation/data regime before more GPU work; routine
Sol monitoring is no longer the bottleneck. This recommendation is not a new
advisor request or authorization to change protected scope. Qwen degradation
conditioning, mixed degradation, sealed internal, official FoundIR test and
development-target optimization remain prohibited.
