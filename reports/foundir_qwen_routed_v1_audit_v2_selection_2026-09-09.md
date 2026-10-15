# FoundIR Qwen-routed v1 audit and v2 selection — 2026-09-09

## Scope and protocol

This audit uses only the frozen 660-record FoundIR-training development
contract. It does not use the reused internal records or the protected official
FoundIR test. Ground-truth category is used only after inference to aggregate
and diagnose metrics; it never controls a route.

The completed v1 result is bound by SHA-256
`d8f49018139f2a60006d6b73dcede80276327c4139bc99df3bfb666b0e0cd255`.
It completed all 660 records but passed only 5 of the 10 Stage-2 PSNR/LPIPS
checks.

## Root cause in isolated tasks

| True task | Actual v1 route | Records | Mean PSNR | Mean LPIPS |
|---|---|---:|---:|---:|
| Noise | generalist+none | 53 | 38.9515 | 0.0582 |
| Noise | generalist+low_light | 7 | 18.8160 | 0.1876 |
| Rain | generalist+none | 20 | 34.7888 | 0.0624 |
| Rain | generalist+weather | 36 | 30.5823 | 0.0784 |
| Rain | blur_specialization+weather | 3 | 26.2899 | 0.1981 |
| Rain | generalist+low_light | 1 | 10.5607 | 0.5368 |
| Low-light | generalist+low_light | 49 | 22.5671 | 0.1906 |
| Low-light | generalist+none | 11 | 11.4409 | 0.3665 |

The main v1 regression is therefore routing, not an evaluator arithmetic
error. Weak Low-light false positives severely damage Noise. Treating Rain as
a Weather-specialist trigger damages Rain relative to the released generalist.
The Low-light false negatives are also visibly harmful.

## Frozen v2 policy selected on development only

- Blur base requires Qwen Blur `L4+` instead of `L2+`.
- Low-light refinement requires Qwen Low-light `L3+` instead of `L2+`.
- Weather refinement is triggered only by Qwen Haze `L2+`; Rain alone uses the
  generalist.
- Weather is skipped when the Blur-specialized base is selected.
- Low-light retains precedence over Weather.
- No ground-truth category is an input to any decision.

Projected routes over all 660 development records are:

| Route | Records |
|---|---:|
| blur_specialization+low_light | 24 |
| blur_specialization+none | 93 |
| generalist+low_light | 204 |
| generalist+none | 231 |
| generalist+weather | 108 |

Existing immutable v1 and released-baseline candidates permit an exact
counterfactual for four isolated tasks:

| Task | Projected PSNR | Projected LPIPS | Stage-2 checks |
|---|---:|---:|---|
| Noise | 39.4330 | 0.0660 | 2/2 pass |
| Haze | 23.4737 | 0.1517 | 2/2 pass |
| Rain | 37.0992 | 0.0736 | 2/2 pass |
| Low-light | 20.1627 | 0.2270 | 2/2 pass |

Blur cannot be projected exactly because four records change from a
Blur-plus-refinement route to Blur-only and no immutable Blur-only candidate
exists for those records. The remaining 56 available Blur candidates average
`25.1982 / 0.2907`, so the full v2 GPU development run is required for the
exact Blur result, and Blur is still expected to miss its Stage-2 targets.

The v2 config preserves the v1 result and freeze hashes and uses the same
evaluation seed so routing is the intended experimental difference.

## Completed integrated v2 development result

The complete 660-record Qwen-routed v2 development run is finished and
frozen. It produced 660 PNG outputs and 660 JSON sidecars. The result SHA-256
is `f4a4bbda2af3c59b25cb478db8a364eb30109ee44ec9b74171d64a9042e961ff`,
and the freeze SHA-256 is
`8a99d64e00fbf1650448ae07d81678f012bfbc533ddbcb3f3f55869b7abc225b`.

| Task | PSNR | LPIPS | Stage-2 checks |
|---|---:|---:|---|
| Blur | 25.3164 | 0.2881 | 0/2 |
| Noise | 39.4291 | 0.0659 | 2/2 |
| Haze | 23.4020 | 0.1517 | 2/2 |
| Rain | 37.1069 | 0.0735 | 2/2 |
| Low-light | 20.2694 | 0.2258 | 2/2 |

The route correction therefore raises the complete system to 8/10 checks.
Only Blur remains: PSNR is 0.5836 dB below 25.9, and LPIPS is 0.1081 above
0.18. The Blur and Noise records selected for the next gate use only
`blur_specialization+none` or `generalist+none`; therefore their frozen v2
metrics are also an exact base-output reproduction check for the new
refinement evaluator.

## Blur Restormer refinement readiness

Before reused-internal inference, the approved development-only capacity gate
adds a 26,126,644-parameter Restormer after the frozen selected Blur base. It
is initialized from the released FoundIR Weather Restormer but is fully
fine-tuned on the four FoundIR-training categories containing Blur. After a
strict checkpoint load, only its residual output head is zero-initialized, so
the complete refiner is an exact identity at step 0 while retaining the
released encoder/decoder features. It does not update Qwen or the frozen
Blur/generalist bases and does not access the internal or official FoundIR
test.

The frozen schedule uses global batch 32 on eight homogeneous H100/H200 GPUs,
12,000 optimizer steps, and development evaluations at steps 1,000, 3,000,
6,000, and 12,000. Selection maximizes the worst normalized Blur/Noise
PSNR/LPIPS margin. The gate requires Blur `>=25.9 / <=0.18` and preserves the
already-passing Noise `>=38.9 / <=0.10` metrics.

CPU readiness completed with:

- strict Restormer checkpoint load and exact 26,126,644 parameter count;
- strict selected Blur checkpoint load and 36,261,187 base-network parameters;
- pinned released dependency versions (`transformers 4.47.1`, `tokenizers
  0.21.0`, `huggingface_hub 0.27.1`);
- exact step-0 identity with zero nonzero parameters in the residual output
  head;
- verified differentiable FP32 AlexNet LPIPS gradient;
- bounded DataLoader waits and synchronized cross-rank error propagation
  before DDP backward;
- 47 focused/regression tests passed;
- Python compilation, shell syntax, config/hash contract, immutable
  checkpoint/resume checks, and launcher preflight passed;
- protected official FoundIR test and reused internal records remained
  unaccessed.

The launch route is
`smoke/launch_foundir_blur_restormer_refinement_v1_8gpu.sh`. Its persistent log
is `logs/foundir_blur_restormer_refinement_v1.log`, and GPU telemetry is
`logs/foundir_blur_restormer_refinement_v1_gpu.csv`.
