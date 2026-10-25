# Rain precision fit diagnostic: terminal result

Run `fit_v1` completed 300/300 steps on eight H100 80GB GPUs, ending at
2026-10-25T20:59:43Z with launcher status 0. It initialized independently
from the frozen direct-endpoint step-2,000 checkpoint and fresh optimizer,
not the qualification weights. No full-data training was launched.

## Exact matched metrics

PSNR higher is better; LPIPS lower is better. Rows use the shared FP32
endpoint path. Train-fit uses the same 32 fixed training inputs throughout;
development uses the unchanged source-disjoint 100 inputs.

| Split | Step | PSNR | LPIPS |
| --- | ---: | ---: | ---: |
| Train-fit | 0 | 32.10802291110691 | 0.07403585378779098 |
| Train-fit | 100 | 34.00089868743218 | 0.058390050719026476 |
| Train-fit | 300 | 35.09303906168583 | 0.04928288975497708 |
| Development | 0 | 32.08470176034529 | 0.0888220098754391 |
| Development | 100 | 32.54685884616055 | 0.08776275758864358 |
| Development | 300 | 32.44553214852097 | 0.08547242213506251 |

The training split passes the numeric 34.2/.08 thresholds, not the scientific
Rain gate. Development remains 1.7544678514790348 dB short and LPIPS exceeds
.08 by .005472422135062507. Against its own matched step 0, final development
improves by .36083038817567825 dB and lowers LPIPS by .0033495877403765945.
Development PSNR declines from step 100 to 300 while training PSNR improves.
In the separate 25-step qualification, development was
32.61941190970921/.08720199926523492; this is not a checkpoint in this run.

## What the diagnostic establishes

- The existing trainable FLUX LoRA can fit these 32 fixed Rain inputs to the
  numeric target under the revised precision/resolution recipe.
- It does not demonstrate equivalent generalization. Repeated exposure to
  these 32 images should not be extended as the next remedy.
- Saved development rows show 20 of 24 inputs with input PSNR >=34 lose
  PSNR after restoration at step 300. This is a descriptive failure audit,
  not a tuned inference rule or an independent evaluation.
- Shared inference arithmetic alone was essentially unchanged at step 0.
  FP32 optimization, matched 512 crops, and altered source exposure are not
  isolated by this experiment; do not attribute all gains solely to FP32.
- No protected pixels were accessed. No scientific success, MoE readiness,
  or novelty claim follows from training-set fitting.

## Engineering, cost and artifacts

Finite gradients and sampled updates in both LoRA matrices were logged.
Final LoRA, optimizer and metrics exist; engineering_pass is true.
Peak reported allocated memory: 24.854209899902344 GiB.
Median optimizer-step time: 1.5413306262344122 seconds.
Full launcher duration: 546 seconds / .15166666666666667 wall hours /
1.2133333333333334 allocated H100 GPU-hours.
The narrower trainer timer reports .1449106847973437 wall hours and
1.1592854783787496 allocated GPU-hours; timer boundaries differ.

Result: `checkpoints/foundir_rain_flux2_precision_fit_v1/fit_v1/result.json`.
Result SHA-256: `ab101bfb57719f7b09c264feb7fe3dcf86ee1056f5f52c9cc480607c74f3cbb0`.
Config SHA-256: `0b3609a8203b472d2c76b5952e0045affcbeb72d0191d80063f9819d9d08ed2a`.
Initial LoRA SHA-256: `eff05dccc0d9bcdf538305ac4095304348a876eb6cb3a4ded3ea27767901f36a`.
Log: `logs/foundir_rain_flux2_precision_fit_v1_fit_v1.log`.

Next decision: propose a frozen full-data matched-resolution generalization
protocol and measured budget; do not launch automatically, extend this
32-image fit, introduce a pixel refiner, or advance MoE before Rain passes.
