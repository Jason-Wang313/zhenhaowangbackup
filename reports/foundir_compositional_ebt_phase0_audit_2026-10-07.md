# Compositional EBT Phase-0 audit — 2026-10-07

> **DIAGNOSTIC ONLY / 非主线。** This readiness work is not a trained model,
> restoration result, FoundIR result, or proof of novelty.

## Decision

The current novelty hypothesis is sufficiently specific to test without adding
the extra mechanisms collected in `ebt transformer gpt brainstorm`. The minimum
causal comparison is a matched Plain conditional EBT versus an atomic/pairwise
Compositional EBT, both trained only on singleton and double degradations and
selected without triple data. Extra test-time sampling, self-verification,
ResFlow-style search, or additional conditioning modules are excluded until
this comparison succeeds and exposes a specific remaining limitation.

## Official EBT source truth

- Paper: Gladstone et al., *Energy-Based Transformers are Scalable Learners
  and Thinkers*, arXiv:2507.02092.
- Official repository: `https://github.com/alexiglad/EBT`.
- Pinned revision: `19420cbeae655bbf11930219a675ade6897019e8`.
- Local pinned checkout: `third_party/ebt_official`.
- License: Apache-2.0.
- No released image-denoising checkpoint exists in the pinned repository. The
  official inference script contains the placeholder `your/model/ckpt`.

The published image experiment is narrower than FoundIR restoration:

- COCO 2014 RGB images resized/cropped to `128x128`;
- patch size `16`;
- training noise level `sigma=.1` under a diffusion noise schedule;
- OOD test noise level `sigma=.2`;
- `100,000` training steps;
- fixed square-image implementation;
- Gaussian/diffusion-schedule denoising, not coupled Blur/Noise/Rain/Haze/
  Low-light restoration.

Published Table 4 values are:

| Model | ID sigma=.1 PSNR / pixel MSE | OOD sigma=.2 PSNR / pixel MSE |
|---|---:|---:|
| DiT | 26.58 / 142.98 | 19.56 / 718.7 |
| EBT | 27.25 / 122.55 | 23.29 / 305.2 |

The paper's `99% fewer forward passes` statement corresponds to `300` DiT
forward passes versus `3` EBT forward passes in its recursively applied OOD
denoising comparison. It is valid evidence motivating this experiment, but it
must not be described as a demonstrated FLUX.2/FoundIR speedup.

## Faithful core and required restoration adaptation

The official EBT image code:

1. maps candidate-image patches to scalar patch energies;
2. averages patch energies to a per-image scalar;
3. differentiates the scalar with respect to the candidate image;
4. updates the candidate by gradient descent;
5. trains through that update using Smooth L1 reconstruction, requiring
   higher-order gradients;
6. forces math attention because optimized attention kernels do not reliably
   support the required higher-order derivatives.

The official denoising module does not separately condition its energy on an
observed degraded image. The project adaptation therefore defines compatibility
over concatenated observation/candidate patches, `E(y, x_hat | q)`. This is a
real method adaptation and must be disclosed; it is not an untouched
reproduction of the official denoising model.

## Phase-0 implementation

`lowlight_restoration/compositional_ebt.py` implements two exactly
parameter-matched arms:

- frozen config SHA-256:
  `b79d3cfec14f55a26f8563e9488fa8f6ed57c1f55ac15467ea5642caabaf0cea`;
- implementation SHA-256:
  `5a5540ab06f67a8f165ca126ed8852d4dc6f63fd2998d6f454163c40af0918b6`;

- **Plain EBT:** dense mixing between the full VLM descriptor basis and learned
  patch-energy components.
- **Compositional EBT:** the identical tensors and forward compute, with a fixed
  diagonal mask enforcing one base, five atomic, and ten pairwise terms.

The shared descriptor is

```text
[1, q_blur, q_noise, q_rain, q_haze, q_low_light,
 normalized q_i*q_j for all ten pairs]
```

The two arms each contain `143,360` trainable parameters in the Phase-0 tiny
configuration. This number is only a smoke-test identity, not the future pilot
model size.

CPU verification passed:

- Python compilation;
- three focused unit tests;
- exact Plain/Compositional trainable-parameter equality;
- finite, nonzero candidate-image gradients;
- finite second-order parameter gradients through a reconstruction loss;
- one gradient step lowered energy for every synthetic sample in both arms;
- optimizer steps: zero;
- triple endpoint, sealed internal, and official FoundIR access: false.

CPU result:
`checkpoints/foundir_compositional_ebt_phase0_v1/cpu/result.json`, SHA-256
`fe4484e21c15307e578f3559c4aac8e51c20a946d992c56af8a31985e0fdcc74`.

The official EBT output layer is zero-initialized. That makes the candidate
gradient exactly zero before the first parameter update. The Phase-0 gradient
smoke therefore uses a small deterministic nonzero head initialization to test
the inference/backpropagation machinery. The future training implementation
must separately decide and predeclare whether to retain official zero
initialization or use a justified stability initialization; Phase-0 does not
make that scientific decision.

## Remaining gate

Only a zero-optimizer-step CUDA smoke remains. It may use one H100 or H200 and
must not load protected data. Formal training, checkpoint selection, and triple
evaluation remain unauthorized until a separate matched pilot protocol is
frozen with a measured runtime and compute budget.
