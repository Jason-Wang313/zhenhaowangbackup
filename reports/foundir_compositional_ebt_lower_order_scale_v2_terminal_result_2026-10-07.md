# FoundIR matched Compositional EBT lower-order scale v2 — terminal result

> **DIAGNOSTIC ONLY / 非主线。** This is a 10k steps-only lower-order
> learnability result, not an unseen-triple, official EBT reproduction, FLUX.2
> mainline, or SOW result.

## Terminal status

- Runtime: success; terminal marker status `0`.
- Scientific gate: `lower_order_learnability_pass` for both arms.
- Actual hardware: two `NVIDIA H100 80GB HBM3` GPUs, one independent arm per
  GPU; no DDP and no batch change.
- Start/end: `20:34:16Z` to `20:41:24Z` (`7m08s`).
- Training time: Plain `367.468582s`; Compositional `366.756288s`.
- Allocated cost estimate: approximately `.238 GPU-hours` for two GPUs over
  the end-to-end 428 seconds.
- Optimizer steps: 10,000 per arm; 20,000 total.
- Result SHA-256:
  `9a4e73d261d9bc03b7996133588f90f80e1c130a4c1e4f5f5ecf83a2e61dd3c0`.

No R0 held-out-triple image, FoundIR 60-image triple endpoint, sealed internal
data, or official FoundIR test was accessed.

## Source-disjoint lower-order development

| Arm | Input PSNR | Output PSNR | PSNR gain | Input Smooth L1 | Output Smooth L1 | Improvement |
|---|---:|---:|---:|---:|---:|---:|
| Plain | 13.881658 | **17.041216** | **+3.159558 dB** | .020455223 | **.009882082** | **.010573141** |
| Compositional | 13.881658 | 16.099879 | +2.218221 dB | .020455223 | .012273887 | .008181335 |

Plain exceeds Compositional by `0.941337 dB` and has lower Smooth L1 by
`.002391806` on the 30-parent aggregate.

| Arm | Singleton, n=10 | Gain | Double, n=20 | Gain |
|---|---:|---:|---:|---:|
| Plain | 16.195173 | +2.465943 dB | **17.535324** | **+3.575398 dB** |
| Compositional | 15.921655 | +2.192425 dB | 16.191811 | +2.231884 dB |

Train loss also decreased materially: last-25/first-25 ratio is `.474358`
for Plain and `.535821` for Compositional. This confirms that the 300-step v1
was insufficient and that the local EBT adaptation is trainable.

## Interpretation and stop

The result supports EBT viability, but it does not yet support the proposed
Compositional EBT advantage. On observed singleton/double compositions, the
dense Plain arm currently learns substantially better, especially on doubles.
That may reflect the intended compositional constraint trading in-distribution
capacity for unseen-composition generalization, but this explanation is only a
hypothesis until a separately authorized frozen triple evaluation tests it.

Do not automatically run 100k. With only 60 training parents and a 146k model,
100k would present each parent thousands of times and may primarily increase
overfitting. The advisor/user should choose between a one-time frozen held-out
triple diagnostic, an official-recipe/model/data scale redesign, or a 100k
steps-only overfit study. No option is authorized by this terminal pass.

Checkpoints:

- Plain:
  `checkpoints/foundir_compositional_ebt_lower_order_scale_v2/scale10k_v2/plain/step_010000.pt`,
  SHA-256
  `c95367c42f49d52e9b9fff0bb182a271cab3fff3bea512c040cf7f1889bebca1`.
- Compositional:
  `checkpoints/foundir_compositional_ebt_lower_order_scale_v2/scale10k_v2/compositional/step_010000.pt`,
  SHA-256
  `e5bc4f54458dbcd22df6bbc3d53f1b1a8c46411fbdaced50afeb53ce619b16e2`.
