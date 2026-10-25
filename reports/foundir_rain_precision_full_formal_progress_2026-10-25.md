# Full-data Rain formal: completed, joint gate not yet met

Run: `formal_v1`, started 2026-10-25T21:19:26Z on eight H100 GPUs.
Frozen 3,000-step schedule completed without intervening recipe changes.
Ended 2026-10-25T22:55:59Z, launcher status 0. Engineering completion is
successful; the scientific joint gate remains false. Development gate: PSNR >=34.2 and
LPIPS <=.08. Higher PSNR / lower LPIPS are better.

| Step | Train-probe PSNR | Train-probe LPIPS | Development PSNR | Development LPIPS |
| --- | ---: | ---: | ---: | ---: |
| 0 | 32.10802291110691 | 0.07403585378779098 | 32.08470176034529 | 0.0888220098754391 |
| 250 | 33.165263671232395 | 0.06575017247814685 | 33.187940708396006 | 0.08063413599971682 |
| 500 | 33.12226387894517 | 0.07149094255873933 | 33.26594373552425 | 0.08544677892699838 |
| 1000 | 33.586227005480914 | 0.06417295400751755 | 33.51466280495532 | 0.08047093592118472 |
| 2000 | 33.79339515340005 | 0.06371757120359689 | 33.624228321542766 | 0.07938327748794109 |
| 3000 | 33.984372473834554 | 0.060009161999914795 | 34.00838778684664 | 0.07343229320365935 |

## Historical milestone observations (not active instructions)

Step250 improves both development metrics over initialization, but neither
threshold is met yet. No extrapolation to final success is justified.
Step250 LoRA, optimizer and metric artifacts were saved; training continued.
Keep the frozen schedule; do not retune or advance MoE.

Metric files:
`checkpoints/foundir_rain_flux2_precision_full_v1/formal_v1/metrics/`.
Log: `logs/foundir_rain_flux2_precision_full_v1_formal_v1.log`.
Launch/config source: 761a1f5. Qualification evidence: 2ad559c.

Step500: PSNR increases modestly from step250 but LPIPS regresses.
Both gates remain unmet; do not call this stable joint progress. Step500
checkpoint/optimizer and metrics were saved. Continue the frozen schedule.

Step1000: both development metrics improve over step500. PSNR remains
.68533719504468 dB below 34.2 and LPIPS remains above .08. Neither gate is
passed. Checkpoint/optimizer/metrics saved; continue unchanged to step2000.

Step2000: LPIPS now passes .08; PSNR remains .5757716784572371 dB below
34.2. Gain since step1000 is only about .110 dB, so final success must not
be promised. Both metrics are currently best among the formal milestones.
Checkpoint/optimizer/metrics saved; continue the fixed final 1,000 steps.

## Terminal audit

Final step3000 is best on both development metrics among the predeclared
formal checkpoints. Relative to matched step0, PSNR improves
1.9236860265013505 dB and LPIPS decreases .015389716671779755.
LPIPS passes .08; PSNR remains .19161221315336263 dB below 34.2.
The final 1,000 steps add .384159465303874 dB. This is encouraging measured
progress, not a guarantee that an extension would cross the gate.

All 3,000 steps completed normally. Both sampled LoRA factors updated;
all logged losses/gradients were finite. Final LoRA and optimizer exist.
The LoRA header independently confirms 100 F32 tensors. Current source and
config hashes match the run. The 100 final metric rows have unique IDs and
their independently recomputed means match the reported PSNR/LPIPS.
No protected data was accessed. Inference remains one FLUX LoRA forward;
the Restormer teacher was training-only and no pixel refiner was added.

The eligible pool remains 38,334 records from 5,780 sources. Actual exposure:
96,000 sample presentations, 36,156 unique records and all 5,780 sources.
This is 94.3183596807012% unique-record coverage, not a claim that every
eligible record was seen. Source-balanced sampling intentionally differs
from a uniform exhaustive record epoch.
A descriptive audit of saved final development rows finds 14 of the 24
inputs with input PSNR >=34 lose PSNR after restoration; local preservation
remains an unresolved issue. This audit does not tune an inference rule.

Full launcher duration: 5,793 seconds (1h36m33s), 1.6091666666666666 wall
hours and 12.873333333333333 allocated H100 GPU-hours. Narrower trainer
timer: 1.6022229538601822 wall hours and 12.817783630881458 GPU-hours.
Median training step: 1.923873278312385 seconds. Peak reported allocated
memory: 24.858047485351562 GiB.

Result:
`checkpoints/foundir_rain_flux2_precision_full_v1/formal_v1/result.json`.
Result SHA-256:
`59849092b5b4e609aaa5aef703537807b219a579142d86b18bdfc5c4e2875972`.
Final/best LoRA:
`checkpoints/foundir_rain_flux2_precision_full_v1/formal_v1/step_003000.safetensors`.
LoRA SHA-256:
`336d412299cd7cd9ff919d8983b2b4c435105161496510c0cec8748785c15904`.

The task is finished; no automatic continuation or new GPU run was launched.
Do not claim the Rain joint gate passed or advance MoE. Any extension or
recipe change needs a fresh user decision and a frozen budget/protocol.
