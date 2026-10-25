# Rain precision qualification: completed

Run: `qualification_v1`, 25 steps, eight NVIDIA H100 80GB GPUs.
Launcher ended at 2026-10-25T20:39:00Z with status 0.
Engineering pass is true; the scientific development gate remains false.

| Split | Shared-path step-0 PSNR | Step-25 PSNR | Step-0 LPIPS | Step-25 LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Fixed 32 training records | 32.10802291110691 | 33.364734543215874 | 0.07403585378779098 | 0.06714100111275911 |
| Frozen 100 development records | 32.08470176034529 | 32.61941190970921 | 0.0888220098754391 | 0.08720199926523492 |

Higher PSNR and lower LPIPS are better. Neither split passes the joint
34.2/.08 target. Training-fit is not evidence of held-out generalization.
Legacy step-0 development was 32.08531687594775/.08882607079111039:
changing inference subtraction alone did not materially improve performance.
The training gain cannot be attributed solely to FP32 without a matched
BF16 training control; crop size and source exposure also differ from history.

Both A/B sampled parameter updates were observed. Losses and gradients were
finite; LoRA/optimizer artifacts and final metrics were written. Reported
peak allocated memory was 24.856319427490234 GiB. No protected-data access.
Median optimizer-step duration: 1.9891585316509008 seconds.
Launcher elapsed time: 295 seconds, including initial startup, corresponding
to 0.08194444444444444 wall hours and 0.6555555555555556 allocated GPU-hours.
The narrower trainer timer reports 0.05591961305717834 wall hours and
0.4473569044574267 allocated GPU-hours; do not confuse the timer boundaries.

Result: `checkpoints/foundir_rain_flux2_precision_fit_v1/qualification_v1/result.json`.
Result SHA-256: `7e963694308b3882c975f6b7188de45717a30fb9fd6364e12542d83944e1f195`.
Config SHA-256: `0b3609a8203b472d2c76b5952e0045affcbeb72d0191d80063f9819d9d08ed2a`.

Next: user-authorized independent 300-step fit diagnosis, from the original
step-2,000 LoRA and fresh optimizer. Qualification weights must not seed it.
Use the existing `fit fit_v1` tmux wrapper on one eight-GPU node. Estimated
15-20 wall minutes / 2.0-2.67 allocated GPU-hours including validation;
this is an estimate, not a completed-run measurement. No automatic full-data
training. No fit log existed when preparing this handoff.
