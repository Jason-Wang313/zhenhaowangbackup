# FLUX.2 Haze step-3000 full-development evaluation readiness

Date: 2026-09-18

## Authorized scope

The user explicitly instructed completion of the missing Haze full-development
validation. This run performs no training, tuning, alpha selection, routing,
or checkpoint selection. It evaluates the immutable isolated-Haze FLUX.2
step-3000 LoRA once on all 1,580 validation records across 158 source groups.

Released FoundIR generalist and specialist weights are prohibited and absent
from the config. Sealed internal and official FoundIR test data remain
prohibited.

## Frozen identities

- config:
  `configs/foundir_haze_flux2_step3000_full_development_v1.json`
- config SHA-256:
  `a285c102fb63e1716aeb5ae7131b66037e188903c42aaa54caecc3bbae442b71`
- FLUX Haze checkpoint:
  `checkpoints/foundir_haze_flux2_source_diverse_continuation_v1/step_003000.safetensors`
- checkpoint SHA-256:
  `4d94ecf6dd902704345e28ae0bf3447d7e172235158c2b2177933794fd973e00`
- source manifest SHA-256:
  `e8c018879029c0bcc2da8d074cc4999541158ca97d07f48aadbcdddcdccc3ca7`

The validation split contains exactly 1,580 unique records / 158 source
groups. The isolated-Haze training split contains 76,650 records / 7,642
source groups, with zero train/validation source-group overlap. Every
validation image is at least 512 pixels on both axes.

## Frozen protocol

- one deterministic aligned 512 crop per record;
- official FLUX.2 shifted 50-step schedule, guidance 1.0;
- deterministic per-record inference seed;
- uint8 quantization before PSNR and LPIPS;
- record-weighted means plus 5,000-resample equal-source-group bootstrap;
- formal gate: mean PSNR >=22.1 and mean LPIPS <=.19;
- tail and bootstrap evidence are mandatory descriptive outputs;
- sixteen previously fixed, source-diverse Haze records are saved for visual
  review;
- no automatic follow-on.

## Verification

- focused tests: 5/5 passed;
- Python compilation: passed;
- runner and launcher shell syntax: passed;
- config, parent, checkpoint, pilot result, base weights, manifest, all 3,160
  LQ/GT image hashes, cardinality, source isolation, visual-set diversity, and
  protected-data flags: passed;
- complete launcher `--check`: passed;
- governance before launch: resolved;
- GPU job started from the Codex shell: no; this shell exposes no
  `nvidia-smi`.

## Runtime disclosure and launch

Use one homogeneous eight-GPU instance only:

- 8xH100: 23--31 minutes total wall time, center 27 minutes; 3.1--4.1
  aggregate GPU-hours, center about 3.6;
- 8xH200: 17--24 minutes total wall time, center 20 minutes; 2.3--3.2
  aggregate GPU-hours, center about 2.7.

The estimate includes preflight, model loading, all 1,580 50-step inferences,
PSNR/LPIPS, source-group bootstrap, visuals, aggregation, and governance
reporting. Primary uncertainty is shared-storage and inference throughput.

Exact launch:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
smoke/launch_foundir_haze_flux2_step3000_full_development_v1_8gpu.sh
```

Monitor:

```bash
tail -f logs/foundir_haze_flux2_step3000_full_development_v1.log
```

Do not start a duplicate. At completion, report Haze only, including absolute
and SOW-relative percentage margins, source-group bootstrap, tails, fixed
visual review, actual start/end/wall time, and no protected-data access.
