# FLUX.2 Noise GT-free adaptive-alpha read-only gate failure

## Design

The user authorized one CPU-only attempt to improve the fixed-alpha Noise tail
before full-development scaling. The policy was designed and frozen using only
the sixteen existing step-400 training triptychs as calibration. It uses no GT
at inference: a robust high-pass estimate of LQ noise determines alpha for
`LQ + alpha * (FLUX - LQ)`:

- sigma <= `.0005`: alpha `.075`
- sigma <= `.0015`: alpha `.125`
- sigma <= `.0025`: alpha `.325`
- otherwise: alpha `.525`

Pixels are rounded/clipped to uint8 before metrics. After freezing these bins,
the policy was applied once to the sixteen existing source-group-disjoint
validation triptychs. No FLUX inference, weight update, GPU, protected split,
or official FoundIR test was used.

## Result

| Split | PSNR | LPIPS | Mean PSNR delta vs LQ | PSNR wins | >2dB regressions |
|---|---:|---:|---:|---:|---:|
| Calibration (16) | 42.00849 | .058768 | +1.55058dB | 15/16 | 0/16 |
| Validation (16) | 39.51763 | .068551 | -.82852dB | 5/16 | 4/16 |

Validation averages pass Noise SOW (`>=38.9 / <=.10`), but the tail/identity
gate fails. Six records regress by more than 1dB and four by more than 2dB;
the worst is `foundir_train_0151268` at -4.00788dB.

The adaptive rule is rejected. Relative to fixed alpha `.30`, it reduces the
single worst loss from about 5.64 to 4.01dB but doubles the >2dB failure count
from 2/16 to 4/16, lowers PSNR win rate from 50% to 31.25%, and worsens mean
PSNR delta from -.51256 to -.82852dB. High-pass magnitude is confounded with
scene texture/content and cannot safely control restoration strength here.

Do not retune this policy on validation or scale it to full development. The
validation evidence is consumed for this policy family. A successor requires
a prospectively frozen design with a stronger GT-free identity signal and a
new independent validation protocol.

Runtime: CPU-only; GPU wall time `0`, GPU-hours `0`; final frozen-policy metric
run approximately nine seconds.
