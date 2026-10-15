# FoundIR Blur Restormer v4 failure audit — 2026-09-09

## Outcome

The final, provenance-correct v4 run started at `2026-09-09T11:40:40Z` on
eight NVIDIA H200 GPUs and automatically stopped after its step-250
development evaluation. This was a scientific gate failure, not a training
runtime failure. The intentional trainer exit code 2 was wrapped by
`torch.distributed.run` as shell status 1 after the complete checkpoint,
development result, continuation decision, and final result had been written.

The run used config SHA-256
`6705b1a58abd109d7f5520857786e48e817a210ff50d614d3c011ba0599d709f`,
trainer SHA-256
`d39108004011d74ea466226b83a1dd30f2aa8a78b789b0aebfcd9b5ebfb2e3e9`,
and fusion SHA-256
`2480080b90fe4ac43ce38c25caa243e866d1ebda2efc37dad9d0dbfdc65b9546`.
No disclosed reused internal record or protected official FoundIR test record
was accessed.

| Candidate | Blur PSNR | Blur LPIPS |
|---|---:|---:|
| Frozen Qwen-routed v2 base, scale 0 | 25.3164 | 0.2881 |
| v4 scale 0.50 | 25.2314 | 0.2570 |
| v4 scale 0.75 | 25.1338 | 0.2493 |
| v4 scale 1.00, legacy-score selection | 25.0724 | 0.2460 |
| v4 scale 1.25 | 24.9080 | 0.2467 |
| v3 step 1,000, prior best evidence | 25.2331 | 0.2349 |
| Stage-2 target | >=25.9 | <=0.18 |

At the selected scale, v4 was `-0.2440 dB / -0.04205 LPIPS` relative to the
frozen base: perceptual distance improved, but pixel distortion worsened.
Relative to v3 it was worse on both metrics: `-0.16072 dB / +0.01109 LPIPS`.
Noise remained exactly unchanged at `39.4291 / 0.06587` because the frozen
router did not produce a false Blur L4/L5 route on any Noise record.

## What the run ruled out

The v3 audit attributed part of the failure to a numerically weak PSNR
objective. v4 corrected that imbalance. At step 250, approximate weighted
loss contributions were:

- Charbonnier: `0.25 * 0.044233 = 0.01106`;
- RGB MSE: `10 * 0.006557 = 0.06557`;
- per-image log-MSE surrogate: `0.02 * 3.479407 = 0.06959`;
- edge L1: `0.05 * 0.035254 = 0.00176`;
- LPIPS: `0.1 * 0.218543 = 0.02185`.

Thus PSNR-aligned terms contributed about `0.1352` of the `0.1698` total
loss. Their weakness was no longer the limiting factor. Increasing those
weights further is not supported: the development PSNR still fell, and LPIPS
also regressed from v3.

The 250-step update changed the complete Restormer state by only `0.489%` in
relative L2 norm, but changed the output head by `24.09%` relative L2 norm.
This concentrated head drift was sufficient to erase v3's development
generalization. The result is consistent with destructive fine-tuning of a
mismatched prior, not insufficient optimizer time.

## Deeper bottleneck and selection correction

Every v2-v4 Restormer ultimately inherited
`third_party/foundir/specialist_model/ckpt/weather.pth`. The local FoundIR
package contains a Restormer architecture and weather/low-light specialist
weights, but no task-native motion-deblurring checkpoint. v3 learned useful
perceptual corrections from FoundIR training pairs, yet v4 showed that
reweighting the loss on the same weather-derived core cannot reliably recover
both distortion and perception.

The legacy selection score also mixed unnormalized PSNR dB margins with LPIPS
margins divided by `0.01`. This made one PSNR dB equal to one score unit while
`0.01` LPIPS also equaled one unit. With the predeclared balanced units of
`0.1 dB` and `0.01 LPIPS`, v4 scale 0.50 would beat scale 1.00, although no v4
scale comes close to passing the actual gate. This scoring defect did not
cause the gate failure, but it must not be carried into the final attempt.

## Final-attempt implication

The only remaining evidence-backed attempt is task-native initialization:
strictly load the upstream Restormer motion-deblurring weights, preserve their
raw-LQ input distribution, blend their full restored output against the frozen
Qwen-routed base, and use a ten-times-smaller learning rate on the pretrained
core than on the new base/LQ fusion layer. Evaluate before any optimizer step,
then continue only on balanced joint development progress. If that attempt
fails, stop experimentation and report the current system and the unresolved
Blur model-capacity/domain-transfer bottleneck to the advisor.
