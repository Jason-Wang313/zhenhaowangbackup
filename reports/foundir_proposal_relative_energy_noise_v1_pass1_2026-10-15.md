# FoundIR proposal-relative energy Noise v1 — pass-1 result

Date: 2026-10-15 UTC

## Outcome

Run `noise_energy_v1_pass1` completed the frozen first scientific checkpoint:
`1,706` optimizer steps, equal to one effective presentation of the 54,585
Noise training records at global batch 32. The run completed successfully on
one homogeneous `8 x NVIDIA H100 80GB` node. It had no OOM, NCCL, non-finite,
or protected-data-access failure.

This checkpoint does **not** pass the method gate. Candidate-gradient energy
refinement improves PSNR slightly but makes LPIPS worse. The matched direct
refiner has the same tradeoff. This is scientific evidence after one pass, but
the frozen protocol requires the primary five-pass checkpoint before rejecting
the method unless a code or numerical failure occurs.

## Runtime

- completed optimizer steps: `1,706`;
- wall-clock time: `.6484166688554817 hours`;
- allocated GPU-hours: `5.187333353174229`;
- measured time: `1.36807257973601 seconds/optimizer step`;
- hardware: `8 x NVIDIA H100 80GB`;
- micro-batch: `1/GPU`;
- gradient accumulation: `4`;
- global batch: `32`.

## Frozen source-disjoint development validation

All arms use the same 256 records and fixed crops. Higher PSNR is better;
lower LPIPS is better.

| Arm | PSNR | Change from proposal | LPIPS | Change from proposal |
|---|---:|---:|---:|---:|
| Frozen proposal | `43.124160178005695` | — | `.050362067760033824` | — |
| Direct refiner | `43.24551533162594` | `+.12135515362024307` | `.06125720706040738` | `+.010895139300373557` |
| Energy, 1 step | `43.20002077519894` | `+.07586059719324112` | `.05506254433294089` | `+.004700476572907064` |
| Energy, 2 steps | `43.21786245703697` | `+.0937022790312767` | `.05692732973329839` | `+.006565261973264569` |
| Energy, 3 steps | `43.22869956493378` | `+.10453938692808151` | `.058651921212913294` | `+.00828985345287947` |

The predeclared minimum technical gate is at least `+.30 dB` and `-.005`
LPIPS versus the frozen proposal. No arm passes: the PSNR gains are too small,
and every learned arm increases LPIPS. More energy steps monotonically increase
PSNR but also monotonically worsen LPIPS at this checkpoint.

## Data-access boundary

- official FoundIR test accessed: `false`;
- sealed internal accessed: `false`;
- internal test accessed: `false`;
- frozen 660 accessed: `false`.

## Artifact identity

- checkpoint `step_001706.pt`:
  `d5f2b64a903cea880a5831bc2a9950c109b7b50efd017efc7636dfad7ab74709`;
- validation JSON:
  `bd97f1087c23ad51708d236884237dcc28313ac7aad4b1c9db5598c79493fb81`;
- terminal JSON:
  `139e94dd42d68305787d16e461c600bad39dd12242edcc578022400f2b0afe76`.

## Next frozen action

Resume the same `noise_energy_v1_pass1` trajectory from `step_001706.pt` to
the five-pass checkpoint at `8,530` total optimizer steps. Do not start a new
run ID and do not alter the loss, learning rate, candidate steps, validation
records, or checkpoint rule. At measured H100 throughput, the remaining 6,824
steps are approximately `2.59 wall-clock hours` and `20.75 allocated
GPU-hours`, plus small validation/checkpoint overhead. The projected total
through step 8,530 is approximately `3.24 wall-clock hours` and `25.93
allocated GPU-hours`.
