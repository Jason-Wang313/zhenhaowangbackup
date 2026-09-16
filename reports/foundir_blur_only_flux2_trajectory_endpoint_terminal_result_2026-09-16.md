# FLUX.2 Blur-only trajectory-endpoint gate terminal result — 2026-09-16

**DIAGNOSTIC ONLY / 非主线.** The bounded gate completed normally but failed
its required human visual/full-scale-admission gate. It is not a FoundIR
official-test or SOW result.

## Runtime and identities

- homogeneous 8x NVIDIA H100 80GB HBM3;
- start `2026-09-16T06:45:10Z`, end `2026-09-16T07:18:16Z`, model wall time
  `1941.4355 s`;
- no OOM, NCCL, NaN, traceback, or runtime failure;
- config SHA-256:
  `d1824e219a5f46629e6d8cc3e964e5b676512e2bb8037c41c5bf88a01f9f7b23`;
- immutable root result SHA-256:
  `d73470876cb5b63a55c964b501e31288fcc4cae7f7fe27b2cca49c0869811586`;
- candidate step-200 SHA-256:
  `67a0dc81fc7361de2b5f00a7abcfd0b97ab872e73e60a8662f637a4de5278abc`;
- control step-200 SHA-256:
  `4eda55c096a3735221a4958d140f00c0cbd9ced9e3c36961799d81a949e0ea79`.

The full-resolution CUDA rehearsal passed before optimizer step 1: peak
allocated memory was 29.49 GB, every rank had nonzero gradients for all 100
trainable LoRA tensors, and frozen VAE parameters had no gradients.

## Metrics

| Split/state | PSNR | LPIPS |
|---|---:|---:|
| train parent | 23.653284 | 0.167751 |
| train flow control step 200 | 24.943428 | 0.144580 |
| train trajectory candidate step 200 | 25.373643 | 0.144914 |
| development parent | 24.591062 | 0.150766 |
| development flow control step 200 | 22.234235 | 0.183982 |
| development trajectory candidate step 200 | 22.676701 | 0.186763 |

The predeclared train-only automatic checks all passed. Candidate versus
control is `+0.430215 dB / +0.000334 LPIPS`, with PSNR better on 13/16 train
records. Candidate versus train parent is `+1.720359 dB / -0.022837 LPIPS`.

This does not translate into source-disjoint fidelity. Candidate development
is `-1.914361 dB / +0.035996 LPIPS` versus its exact parent and beats parent
PSNR on only 8/16 records. It remains far below the Blur SOW PSNR target
`25.9 dB` and its development LPIPS `0.186763` also misses `0.18`.

## Visual gate

All 16 candidate step-200 train and all 16 source-disjoint development
`Input | Output | Ground Truth` triptychs were reviewed. Training outputs are
mostly coherent and substantially sharper, proving that the endpoint gradient
mechanism works. Development outputs often look plausibly sharp, but several
hard motion/text examples place edges, object boundaries, or lettering at
positions/shapes that do not match GT. This is precisely the failure mode that
PSNR exposes.

Representative development regressions include:

- `foundir_train_0055608`: `24.264 -> 15.411 dB`, LPIPS
  `0.1177 -> 0.3333`;
- `foundir_train_0001925`: `29.047 -> 19.879 dB`, LPIPS
  `0.0466 -> 0.1921`;
- `foundir_train_0021020`: `27.486 -> 21.026 dB`, LPIPS
  `0.1300 -> 0.2121`.

Hash-bound review:
`reports/foundir_blur_only_flux2_trajectory_endpoint_visual_review_v1.json`,
SHA-256
`17e48c633813b4138c61fe037a9aef2740683612b5cd8b3b3ae2bb9f9f6d4d62`.

## Artifact audit and conclusion

The independent audit passes: 6 checkpoints, 6 optimizer states, 12 complete
metrics files with 192 finite rows, and 192 fixed PNG triptychs. The VAE
ceiling remains `42.220081/.007048` train and `41.716497/.006953`
development. Protected flags are all false.

Final conclusion: metric-mechanism evidence is positive, but the required
visual/source-disjoint full-scale admission fails. Do not full-scale, extend,
distill, or decoder-tune this checkpoint. A next experiment requires a new
explicit decision and must address cross-image spatial correspondence rather
than repeat this tiny endpoint loss.
