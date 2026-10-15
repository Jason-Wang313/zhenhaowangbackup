# FoundIR proposal-relative energy Noise V2 — 25-step GPU qualification

Date: 2026-10-15 UTC

## Outcome

Run `noise_energy_v2_qualification1` completed all `25/25` optimizer steps and
ended with launcher status `0` on one homogeneous `8 x NVIDIA H100 80GB` node.
The full exact-six-change V2 graph ran through CUDA, DDP, differentiable LPIPS,
three-step second-order energy refinement, the four frozen contrastive models,
optimizer update, fixed development validation, and checkpoint serialization.

There was no OOM, NCCL failure, non-finite objective/model tensor, or
protected-data access. All `488` saved model tensors are finite and all `488`
trainable entries have optimizer state.

This is engineering evidence only. Step 25 is still in the declared learning-
rate warm-up and its image metrics cannot establish success or failure. The
qualification checkpoint must not initialize the formal run.

## Hardware and time

- hardware: `8 x NVIDIA H100 80GB`;
- micro-batch: `1/GPU`;
- gradient accumulation: `4`;
- global batch: `32`;
- launcher interval: approximately `.10 wall-clock hours`;
- corresponding end-to-end allocation: approximately `.80 GPU-hours`;
- measured training/validation interval stored by the program:
  `.04358181471315523 hours`;
- program-reported allocation for that measured interval:
  `.34865451770524186 GPU-hours`;
- validation artifact's conservative average:
  `6.270415588766337 seconds/optimizer step`.

The 25-step measurement uses the first `128 x 128` crop stage. It must not be
treated as an exact duration for the full run, which later uses `192 x 192`
and `256 x 256` crops. If one incorrectly held the smallest crop speed fixed,
`8,530` steps would be `14.86` wall-clock hours and `118.84` GPU-hours; that is
a lower-bound-style projection, not the final expected duration.

## Step-25 development metrics

Higher PSNR is better; lower LPIPS is better. All arms use the same fixed 256
source-disjoint development records.

| Arm | PSNR | LPIPS |
|---|---:|---:|
| Frozen proposal | `43.124160178005695` | `.050362067760033824` |
| Matched direct refiner | `43.12250038981438` | `.0503378546636668` |
| Energy, 1 step | `43.12417149543762` | `.05036190883765812` |
| Energy, 2 steps | `43.12418105453253` | `.05036183708216413` |
| Energy, 3 steps | `43.124188274145126` | `.05036169298909954` |
| Adaptive proposal/1/2/3 selector | `43.124188274145126` | `.05036169298909954` |

These nearly identical numbers are expected at an engineering checkpoint and
must not be described as a meaningful image-quality improvement.

## Artifact identity

- config SHA-256:
  `498b38dbf30b1473c2cfd897bfef1b191458f43e273d19ebfb28fee2e1c7e04b`;
- checkpoint SHA-256:
  `f55575e89c44a9df20f35d6ffc412f0c0501757c368b926b6415e1bcab7d39c0`;
- validation SHA-256:
  `18008d90535ab000bbb704d8a3cd4f9fbd5944cfc3e78067ac6c89834949a15d`;
- terminal-result SHA-256:
  `ab4d7c5b7f12b5bbc6dfbd85b750451e5b8bb2f5690eead8ea64207ff5ad3218`;
- log SHA-256:
  `97bd5194326a0834595190ba8abc53ba4c3ede7c876d55166ccb9cf16a87ca43`.

All protected-data flags are false.

## Next frozen action

Start one fresh formal V2 run with a new run ID, total `8,530` steps, and no
resume from `noise_energy_v2_qualification1`. Validate at the frozen pass
boundaries and judge the method only from the existing `+.30 dB/-.005 LPIPS`
development gate.
