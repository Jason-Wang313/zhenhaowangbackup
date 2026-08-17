# Expanded frozen-Qwen ridge Lowlight validation result

Date: 2026-08-17 UTC
Scope: Lowlight degradation-severity assessment only

## Result

The expanded scalar ridge head passed both its predeclared train-only
selection gate and the one-time, previously unexposed 50-source validation
gate. It is approved as the frozen predicted-severity signal for preparing the
fair FLUX.2 restoration-conditioning comparison.

This result validates only Lowlight type/severity assessment. It does not show
that Qwen conditioning improves restored images, it does not report an
official FoundIR metric, and it is not a Phase 1 acceptance result.

## Bound evidence

- training/evaluation Git revision:
  `c42600df0386119de5882d68b5c19ceecae8dd50`
- frozen config SHA-256:
  `b04bf893e073cf50423a384456398b12ce92625a979453c8f8c61e210f8bd33c`
- frozen Qwen revision:
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- selected ridge alpha: `3000`
- fitted parameters: `7,169`
- checkpoint:
  `checkpoints/qwen_lowlight_ridge_expanded_head/ridge_head.safetensors`
- checkpoint SHA-256:
  `27f58645d992c97f8b3e44af82a2a9ae11d93d0fa4bc5090b3d1bde39271b368`
- final metrics SHA-256:
  `18285e16fad062d8653e9468159f6679531a7f97ea1d5b794bc12fe4994cc29a`
- train-selection metrics SHA-256:
  `6b7fc4b99cc79fc4f6bb536884882f55d762edf04d6d5ff21b9ca9631f0550e9`
- execution log SHA-256:
  `5e174e9e7adcac426d57ffceb8a0cdfda6a03478628e224b8c8586eb0df6d5e5`
- train-fit bundle SHA-256:
  `721eb7694fae0ba9222c4ef38729948358de63122ab9ce15cdb39dad802c4780`
- train-selection bundle SHA-256:
  `75fb05afe9a273c6d9d8b19c62760df58ca89438dbb34fb4ef39170a419ce944`
- validation bundle SHA-256:
  `c8e017fd19b9527610890e3976ceb536f0ae6ff5ebc9680094bc0538542e00ea`

The H100 environment was Python 3.12.13, PyTorch 2.7.0+cu128, CUDA
12.8, scikit-learn 1.5.2, Transformers 4.56.1, and one NVIDIA H100 80GB
HBM3. Peak allocated GPU memory was about 17.51 GiB.

## Train-only selection

All `8,000` features from the `1,600` Lowlight train sources were extracted.
The frozen split used `1,280` sources / `6,400` records for fitting and `320`
sources / `1,600` records for candidate selection. All five predeclared ridge
alphas passed the minimum selection gate; the frozen selection key chose
`alpha=3000`.

| Metric | Qwen ridge | Luminance baseline |
|---|---:|---:|
| Severity accuracy | 0.715625 | 0.629375 |
| Severity MAE | 0.289375 | 0.465000 |

Selected-head recall L1-L5 was `0.884375, 0.631250, 0.712500, 0.521875,
0.828125`. All five output levels were represented and every per-level recall
exceeded the predeclared `0.40` floor.

No validation feature existed before train-only selection completed. The
validation set was not used to choose the alpha, decoder, pooling method, or
source partition.

## One-time unexposed validation

The final gate used exactly `50` previously unused validation sources with all
five levels per source (`250` records). It excluded all `25` source IDs exposed
by the earlier failed ridge experiment.

| Metric | Qwen ridge | Luminance baseline |
|---|---:|---:|
| Severity accuracy | 0.680 | 0.640 |
| Severity MAE | 0.332 | 0.432 |
| Valid structured-output rate | 1.000 | 1.000 |

Validation recall L1-L5 was `0.88, 0.60, 0.68, 0.44, 0.80`. All five output
levels were represented. Every frozen gate passed: absolute accuracy,
baseline-relative accuracy, baseline-relative MAE, complete output support,
minimum per-level recall, and valid structured output.

The gain over the simple luminance baseline is modest in accuracy (`+0.04`
absolute), although the MAE reduction is larger (`-0.10`). L4 remains the
weakest class at `0.44` recall, only `0.04` above the gate floor. These limits
must remain visible when interpreting the later restoration ablation.

## Leakage and authorization boundary

- internal test used: `false`
- official FoundIR used: `false`
- validation used for candidate selection: `false`
- exposed validation sources excluded: `25/25`
- cached feature files after completion: `8,250`
- per-record validation result files: `250`

The 50 validation sources used here are now exposed and must not be presented
again as a fresh unbiased assessor gate after post-hoc tuning. The internal
test and official FoundIR remain protected.

This pass freezes the current assessor checkpoint and authorizes implementation
of the fair restoration-conditioning experiment. The baseline and conditioned
systems must retain the same FLUX.2 backbone, restoration LoRA path, paired
data, source split, optimizer, learning rate, steps, batch size, precision,
seeds, GPU budget, and sampling settings. The only intended difference is the
presence of the frozen Qwen-derived Lowlight severity condition. Improvement
must be established from restored-image PSNR, SSIM, LPIPS, and visual identity;
assessor accuracy alone does not answer the project question.

Noise, Blur, Haze, and Rain remain `blocked_no_task_calibration`.
