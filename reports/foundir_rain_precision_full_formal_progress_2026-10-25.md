# Full-data Rain formal: validated progress

Run: `formal_v1`, started 2026-10-25T21:19:26Z on eight H100 GPUs.
Frozen 3,000-step schedule; no intervening recipe changes. This is a live
progress record, not terminal success. Development gate: PSNR >=34.2 and
LPIPS <=.08. Higher PSNR / lower LPIPS are better.

| Step | Train-probe PSNR | Train-probe LPIPS | Development PSNR | Development LPIPS |
| --- | ---: | ---: | ---: | ---: |
| 0 | 32.10802291110691 | 0.07403585378779098 | 32.08470176034529 | 0.0888220098754391 |
| 250 | 33.165263671232395 | 0.06575017247814685 | 33.187940708396006 | 0.08063413599971682 |

Step250 improves both development metrics over initialization, but neither
threshold is met yet. No extrapolation to final success is justified.
Step250 LoRA, optimizer and metric artifacts were saved; training continued.
Keep the frozen schedule; do not retune or advance MoE.

Metric files:
`checkpoints/foundir_rain_flux2_precision_full_v1/formal_v1/metrics/`.
Log: `logs/foundir_rain_flux2_precision_full_v1_formal_v1.log`.
Launch/config source: 761a1f5. Qualification evidence: 2ad559c.
