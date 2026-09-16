# FLUX.2 Blur-only pixel-fidelity paired gate terminal result — 2026-09-16

## Conclusion

The exact paired gate completed normally on homogeneous 8x
`NVIDIA H100 80GB HBM3`. Both arms ran 400/400 optimizer steps from the same
immutable step-10k parent with independently fresh AdamW states. The apparent
`ChildFailedError` at the end is torchrun propagating the predeclared
scientific-failure exit code 2 from rank 0 after complete `result.json`
creation; there was no CUDA OOM, NCCL error, NaN, or missing artifact.

The gate fails. Direct frozen-VAE RGB MSE has a real causal PSNR effect on the
16 training crops, but it persistently worsens LPIPS relative to the paired
flow-only control and both arms overfit the tiny set rather than preserving the
step-10k parent on source-disjoint crops. Do not full-scale either arm and do
not rerun or extend this experiment.

## VAE ceiling

| Role | Records | Mean PSNR | Minimum PSNR | Mean LPIPS | Maximum LPIPS |
|---|---:|---:|---:|---:|---:|
| train | 16 | 42.220081 | 37.737290 | 0.007048 | 0.021170 |
| descriptive development | 16 | 41.716497 | 33.108700 | 0.006953 | 0.013350 |

All four ceiling checks pass by large margins. The frozen VAE representation
is not the remaining `25.9 dB` bottleneck.

## Paired step-400 result

Both arms share the same parent baseline on the 16 training crops:
`23.653284 dB / 0.167751 LPIPS`.

| Arm / role | PSNR | LPIPS | PSNR better than input | LPIPS better than input |
|---|---:|---:|---:|---:|
| flow control / train | 26.149324 | 0.126395 | 8/16 | 14/16 |
| RGB MSE / train | 26.549051 | 0.139953 | 9/16 | 14/16 |
| parent / descriptive development | 24.591062 | 0.150766 | 9/16 | 14/16 |
| flow control / descriptive development | 22.365686 | 0.181016 | 4/16 | 15/16 |
| RGB MSE / descriptive development | 22.749107 | 0.195493 | 5/16 | 14/16 |

On train, RGB MSE beats the flow control by `+0.399728 dB` and improves PSNR
on 14/16 paired records, proving that the differentiable decoder path works.
It fails the no-perceptual-regression check: LPIPS is `+0.013557` worse than
flow control, above the predeclared `0.003` limit. It remains better than its
parent in both train metrics, but that is insufficient.

The PSNR/LPIPS tradeoff is persistent rather than a final-step accident:

| Step | RGB minus flow PSNR | RGB minus flow LPIPS (positive is worse) |
|---:|---:|---:|
| 50 | +0.338906 | +0.010712 |
| 100 | +0.354368 | +0.011863 |
| 200 | +0.435123 | +0.013170 |
| 400 | +0.399728 | +0.013557 |

No sampled checkpoint supplies the desired PSNR gain without the LPIPS cost.
The descriptive source-disjoint result is a stronger stop signal: versus its
own parent, RGB MSE loses `1.841955 dB` and worsens LPIPS by `0.044726`.
Development GT did not select any arm, weight, or checkpoint.

## Visual review

The fixed images agree with the metrics:

- hard training examples such as `foundir_train_0042483` recover sharp,
  credible structure, demonstrating local tiny-set learnability;
- the near-clean training example `foundir_train_0091486` is largely
  preserved visually, although RGB MSE is slightly worse than control in both
  metrics;
- some source-disjoint examples, notably `0106209` and `0100870`, improve
  strongly and remain content-consistent;
- many source-disjoint failures produce plausible sharp content whose exact
  edges, positions, texture, or photometry do not match GT. The bus example
  `0055608` looks sharper but remains badly pixel-misaligned and scores
  `15.1085 dB` versus `18.9710 dB` input. Eleven of 16 development records do
  not improve PSNR.

This is unsuitable for a pixel-aligned restoration target despite occasional
visually appealing outputs.

Machine-bound visual review:
`reports/foundir_blur_only_flux2_pixel_fidelity_visual_review_v1.json`.

## Interpretation and stop rule

The experiment isolates two distinct facts:

1. VAE capacity is ample and RGB gradients do reach all 100 LoRA tensors.
2. Supervising the one-step random-time estimate
   `clean_pred = noise - predicted_velocity` does not sufficiently constrain
   the deployed 50-step rollout. It shifts the local PSNR objective but does
   not prevent tiny-set trajectory overfit or perceptual regression.

Therefore the next scientifically distinct proposal, if authorized, should
target the deployed trajectory (for example a bounded differentiable
multi-step endpoint/consistency objective with content-preserving
regularization), not repeat this weight, extend it, or launch full-scale.
This report does not authorize that new experiment.

## Immutable evidence

- config SHA-256:
  `e366396c9189dbd03dda45a501d41552cb80ccfb2ebc60748848dcd22284a9b1`;
- step-10k parent SHA-256:
  `e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`;
- root result SHA-256:
  `5e8d1aff388efb75ca4aecb29217a9e979e2561843720236254c5b868d68ec1b`;
- flow-control step-400 SHA-256:
  `52c87b7c225edbeeffc6469c894d005478c59d8031645cfcad01519d3c0f6b6f`;
- RGB-MSE step-400 SHA-256:
  `ddbc3ec9614f2c7c711c8a27546e75c8ccb273f6396586be99af52c78d59dc49`;
- training/development VAE ceiling SHAs:
  `c790d555b6d0cacaa0b3c0271c86a3f1c52a7b7f23bbd6e936d6d1edfbacad9b` /
  `bee88273cd168d00a91d1ab8bdb5673a986199cc6780ff8825b766ddc5dc377c`;
- runtime log SHA-256:
  `9b9a9967a59f6a4784eb58c03b938628807815fbaff29fafe264cbddbd251906`;
- terminal audit: 2 arms, 8 checkpoints, 8 optimizer states, 14 metrics
  files / 224 finite metric rows, and 224 fixed PNG triptychs all pass;
- wall time: `460.1632 s`;
- protected scope: Qwen degradation conditioning, mixed degradation, pixel
  refiner, decoder tuning, development-GT optimization, sealed internal, and
  official FoundIR test access are all false.

Active terminal hold:
`flux2_blur_pixel_fidelity_tiny_result-54e5b532a225ebd7`.
Governance report SHA-256:
`8b3e6c93bbaa80f93ed86e1d35680423a65dbff83b8b4ba502353fa13a6915d1`.
