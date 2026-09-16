# FLUX.2 Blur-only low-LR continuation terminal result

Date: 2026-09-16 UTC

## Verdict

The exact-optimizer step-10k→12k continuation completed successfully, but it
did not improve the verified step-10k model. The conditional step-14k gate is
closed. Do not train steps 12k→14k and do not perform another LR/step sweep.

## Comparable full-development result

Both rows use all 2,595 isolated-Blur source-group-disjoint development
records from 165 source groups, one identical deterministic aligned 512×512
crop and inference seed per record, official FLUX.2 shifted 50-step inference,
guidance 1.0, and PNG-equivalent uint8 quantization before PSNR/LPIPS.

| Checkpoint | PSNR | LPIPS | PSNR >=25.9 | LPIPS <=0.18 |
|---:|---:|---:|:---:|:---:|
| 10,000 | 25.734236 | 0.145406 | fail | pass |
| 12,000 | 25.709168 | 0.145691 | fail | pass |

Step-12k versus step-10k:

- PSNR: **-0.025068 dB**;
- LPIPS: **+0.000285** (worse);
- remaining PSNR gap to SOW: **0.190832 dB**.

At step 12k, output PSNR averaged 25.709168 versus input PSNR 25.849545,
giving a -0.140376 dB mean gain; only 45.048% of records improved PSNR. LPIPS
still improved strongly and passed its threshold, but that does not compensate
for regression on the only failing metric.

## Training and runtime integrity

- parent: exact step-10k LoRA and AdamW optimizer;
- parent optimizer hardware profile: H200; continuation hardware: 8×H100;
- global batch remained exactly 32 under both profiles;
- cumulative deterministic presentation start: 320,000;
- objective unchanged: pure noise-to-clean rectified flow with native LQ
  reference;
- training records/source groups: 105,057 / 7,339;
- LR: 5e-6 at 10k, 4e-6 at 11k, 3e-6 at 12k;
- step-11k LoRA SHA-256:
  `05e26346aa25163df6b278c0a00185ae9546ada29933315243482a56ab06dda7`;
- step-12k LoRA SHA-256:
  `741aaaf3b3178b53dfe772395f50cc25221d4cfa7e27fc1f65b87add34ee9b2d`;
- step-12k optimizer SHA-256:
  `3f66879cf1fafa726fdd31e0eb41ff8a9f0679b59ad8b44af8a16b4cf438f18e`;
- all 2,595 immutable per-record sidecars completed;
- runner terminal status: 0 at `2026-09-16T04:57:36Z`;
- no Qwen degradation conditioning, mixed degradation, pixel refiner,
  development-GT optimization, sealed internal, or official FoundIR test.

Authoritative result:

- path:
  `checkpoints/foundir_blur_only_flux2_continuation_v1/full_development_step_012000/result.json`;
- SHA-256:
  `aaf08bb78292884bb5cce682db190e8d7e894ccf87bb5a21048d2aad331db561`.

## Fixed visual subset warning

The fixed 16-record subset moved from 25.663550 / 0.122995 at step 10k to
25.236514 / 0.129536 at 11k and 25.251881 / 0.129344 at 12k. It exaggerated
the final full-set PSNR regression, but correctly predicted its direction.

## Gate decision and next strategy

The executable gate rejected step 14k because step-12k PSNR did not strictly
exceed 25.734236. No step-13k or step-14k artifact exists.

Retain step 10k as the best verified FLUX.2 checkpoint. Further optimizer
steps at lower LR are no longer evidence-backed. The next experiment should
change the fidelity mechanism while using step 10k as the parent—specifically,
target the remaining PSNR gap with an explicitly pixel-aligned correction or
fidelity objective and a small overfit gate before any full-scale run. This is
a new strategy, not part of the now-closed continuation experiment.
