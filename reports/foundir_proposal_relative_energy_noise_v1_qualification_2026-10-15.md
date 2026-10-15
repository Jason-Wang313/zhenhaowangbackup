# Proposal-relative energy Noise v1 — 500-step qualification

Date: 2026-10-15 UTC
Run ID: `noise_energy_v1_retry1`
Status: systems qualification complete; scientific gate not yet decided

## Outcome in plain language

The single-node `8 x H100 80GB` implementation completed all 500 optimizer
steps, validation, checkpointing, and terminal finalization without OOM, NCCL,
non-finite loss, or protected-data access. The mechanism is runnable and the
H100 memory profile is qualified.

At this intentionally short scale, energy refinement has not yet improved both
metrics. It changes PSNR by only about `+.001` to `+.002 dB` and makes LPIPS
slightly worse. This is not a method pass, but the predeclared policy also says
500 steps are not enough to reject the method. The next scientific checkpoint
is one complete effective pass at 1,706 total steps.

## Fixed 256-record source-disjoint development validation

| Arm | PSNR | LPIPS | PSNR vs proposal | LPIPS vs proposal |
|---|---:|---:|---:|---:|
| Frozen Restormer proposal | `43.124160178005695` | `.050362067760033824` | — | — |
| Matched direct refiner | `43.231114752590656` | `.05958490435659769` | `+.10695457458496094` | `+.009222836596563866` |
| Energy, 1 step | `43.124949902296066` | `.05044266952609178` | `+.0007897242903709412` | `+.00008060176605795277` |
| Energy, 2 steps | `43.125487580895424` | `.050498137210524874` | `+.0013274028897285461` | `+.0001360694504910498` |
| Energy, 3 steps | `43.126018427312374` | `.05055382074897352` | `+.001858249306678772` | `+.0001917529889396974` |

Higher PSNR is better; lower LPIPS is better. The direct refiner obtains a
small PSNR gain but materially worsens LPIPS. Additional energy steps produce
monotonic but negligible PSNR gains and monotonically worse LPIPS. No
paper-quality claim is supported by this qualification checkpoint.

## Runtime and hardware

- GPUs: `8 x NVIDIA H100 80GB HBM3`
- Micro-batch: `1/GPU`
- Gradient accumulation: `4`
- Global batch: `32`
- Completed optimizer steps: `500`
- Measured optimizer time: `1.421636643178761 s/step`
- Total wall-clock time: `.1976426635413534 hours`
- Allocated cost: `1.5811413112655281 GPU-hours`
- Projected 8,530-step primary run: `3.368489046198564 wall hours`,
  `26.94791236958851 allocated GPU-hours`, excluding startup variation

## Artifact identity

- Checkpoint SHA-256:
  `dffad058a9b9697fb76c580144b482e22dddbb5ae2bfb618e0b1134e1f107cdf`
- Validation JSON SHA-256:
  `42833d516b98afc104090607b8e1d0638317d9bd85620ce662359110a31ebe3b`
- Terminal JSON SHA-256:
  `c304780981131a9bd92dc354aee4bd5b49bbb1d8c6c210a13bbc49d7f37aa4e2`

No internal-test, official FoundIR test, sealed-internal, parent-test, or frozen
660 reporting pixels were accessed.

## Next frozen action

Post-run audit found that the qualification launcher originally tied cosine-LR
horizon to the requested stop step. The 500-step systems run therefore reached
minimum LR at step 500; directly resuming it to 1,706 would cause an invalid LR
jump. Preserve its systems/throughput evidence, but do not use its checkpoint
as the scientific trajectory initialization.

The corrected config freezes the LR horizon at the 17,060-step continuation
cap, independent of the requested milestone. Start a fresh run to `1,706`
total optimizer steps and evaluate the same fixed 256 records. This makes every
future 500/1,706/8,530/17,060 checkpoint a true prefix of one schedule. Do not
change the loss, learning objective, candidate-step set, validation records,
or checkpoint-selection rule. Decide whether to proceed toward 8,530 only from
the one-pass and later predeclared development trend.
