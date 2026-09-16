# FLUX.2 Blur-only source-diverse spatial-anchor pilot terminal result — 2026-09-16

**DIAGNOSTIC ONLY / 非主线.** The bounded pilot completed its intended work,
but failed the frozen full-scale-admission gate. It is not an official FoundIR
test or SOW result.

## Runtime and identities

- homogeneous `8 x NVIDIA H100 80GB HBM3`, global batch 8;
- end `2026-09-16T09:22:55Z`, model wall time `3744.3806 s`;
- both arms completed `400/400`; exit 2 is the intentional scientific-failure
  status after result emission, not an OOM, NCCL, NaN, or runtime failure;
- config SHA-256: `ab5a41af64d6a8e39e228625a18b5b978ab616cc007308dcd2999242bb1a05be`;
- parent step-10k SHA-256: `e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`;
- immutable root result SHA-256: `47d72b0d4efb887f9e684d09e83854dcd3e9a00b233ee5c707292fefab674de7`;
- control/candidate step-400 SHA-256: `53521088154d4ac9798711e809bda0052f3e16c26e2b7f3deb01db6d40dff083` /
  `a1be1e983d33de71faa6ea0da3be1ff161a73c5bc791e602cb7be0a5fff3d29d`.

The mandatory CUDA rehearsal passed: all 100 LoRA tensors had finite nonzero
gradients, the frozen VAE had no gradients, and peak allocated memory was
`29,487,960,064` bytes.

## Metrics

| Split/state | PSNR | LPIPS |
|---|---:|---:|
| train matched parent replay | 22.619227 | 0.183367 |
| train flow control step 400 | 22.626140 | 0.184824 |
| train spatial candidate step 400 | 23.521008 | 0.185978 |
| development matched parent replay | 24.032013 | 0.158307 |
| development flow control step 400 | 23.426864 | 0.160281 |
| development spatial candidate step 400 | 24.105156 | 0.170990 |

Candidate versus matched control on development is `+0.678292 dB` PSNR but
`+0.010708` LPIPS worse. Versus the matched parent replay it is only
`+0.073143 dB` and `+0.012683` LPIPS worse. It wins per-record PSNR on 13/16,
but the worst record regresses by `8.129650 dB`. It fails the frozen absolute
PSNR `>=24.591062`, LPIPS `<=0.153766`, LPIPS-control, and worst-record gates.

## Protocol caveat and visual review

The pilot used inference seed `2026091607`, whereas the prior trajectory gate
used `2026091602`. Therefore `24.032013/.158307` is the internally matched
parent replay for this pilot but does not exactly reproduce the previously
quoted `24.591062/.150766`. This is a protocol/comparability mistake. It does
not rescue the candidate: its LPIPS is worse than both matched parent and
control, the worst matched-parent PSNR loss is 8.13 dB, and it also fails the
predeclared absolute thresholds.

All 16 final candidate train and all 16 final source-disjoint development
`Input | Output | Ground Truth` triptychs were reviewed. The candidate is often
visibly sharper and coherent, showing that source diversity plus spatial
anchors add useful PSNR generalization versus control. Exact fidelity remains
unreliable: text, small structure, moving objects, and edges may be restored at
slightly incorrect positions or shapes, while near-clean images can be altered
unnecessarily. Hash-bound review:
`reports/foundir_blur_only_flux2_source_diverse_spatial_anchor_visual_review_v1.json`,
SHA-256 `c806345148fd76ae61024f71349478c47ede6336eb5aa024b8aff9df2091ead8`.

## Artifact audit and conclusion

Artifact counts are complete: 4 checkpoints, 4 optimizer states, 10 metrics
JSON files, and 160 PNG triptychs. The VAE ceiling passes at
`41.664286/.007284` train and `41.716497/.006953` development. Qwen degradation
conditioning, mixed degradation, distillation, decoder tuning, development-GT
optimization, sealed internal, and official FoundIR test access are all false.

Final conclusion: the mechanism provides real matched-control PSNR evidence,
but the checkpoint is not safe to scale. Do not extend, full-scale, distill,
or decoder-tune it. Active terminal hold:
`flux2_blur_source_diverse_spatial_anchor_pilot_result-b2414002ee6a3cdc`.
