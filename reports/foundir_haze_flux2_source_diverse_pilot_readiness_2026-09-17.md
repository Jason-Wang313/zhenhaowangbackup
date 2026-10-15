# FLUX.2 Haze source-diverse pilot readiness

## Status

`READY_FOR_ONE_HOMOGENEOUS_8XH100_OR_8XH200_LAUNCH`

The user explicitly authorized the smallest source-diverse Haze training
pilot. This is the same simple FLUX.2 LoRA flow formulation as the prior
capacity gate, with more source coverage and proportionate training. It adds
no Qwen degradation condition, mixed degradation, refiner, new loss, or
automatic full-development continuation.

## Frozen experiment

- training: 1,580 isolated-Haze records from 1,580 unique train source groups;
- validation: 128 isolated-Haze records from 128 unique validation source
  groups, used as the formal final pilot gate;
- prior 16 calibration and 16 validation source groups are excluded;
- train/validation source-group overlap: zero;
- model: FLUX.2 Klein Base 4B, frozen VAE/text encoder, fresh zero-initialized
  DiT LoRA rank/alpha 32/32 on the existing 50 modules;
- objective: unchanged pure-noise-to-clean rectified flow with native aligned
  reference tokens;
- schedule: 2,000 optimizer steps, global batch 32, LR 5e-5, checkpoints at
  steps 1,000 and 2,000;
- inference: official 50-step shifted schedule, guidance 1.0;
- formal automatic pilot gate: validation PSNR >=22.1 and LPIPS <=.19;
- fixed visuals require human review after metrics;
- no automatic 1,580-record full-development evaluation and no Rain launch.

## Identity and verification

- config SHA-256:
  `1fde72e12149b8097fdd533bf73734b89a3c6ffb7278c5d483b2ced0f87d596d`;
- frozen manifest SHA-256:
  `3fa9f4333c375fb0a329dac68f67344d430984a9ff9d5bd60577e105b209c531`;
- manifest metadata SHA-256:
  `4a804fd69366c6b7279f1da351929ae8ccabd9d257d6e5aac2d83c37adf11f37`;
- preflight receipt SHA-256:
  `7f7fad6ecff330b88913d9abb372fadfcf4d4546506c475de4180e669b6091f3`.

Five focused tests, Python compilation, shell syntax, deterministic manifest
regeneration, complete 3,416 LQ/GT file-hash checks, complete crop decoding,
both model hashes, source isolation, governance, and launcher `--check` pass.
The first preflight exposed and rejected a typo in the newly copied expected
FLUX SHA; independent hashing proved the model file unchanged, the config was
corrected to the established model identity, and the complete preflight then
passed. No GPU job started in the CPU-only coding shell.

## End-to-end GPU time disclosure

For one homogeneous 8xH100 instance, expected complete wall time is **42--58
minutes**, central estimate **50 minutes**. Aggregate compute is **5.6--7.7
GPU-hours**, central estimate about **6.7 GPU-hours**. This includes launcher
checks, model loading, 2,000 training steps, fixed step-0/1,000/2,000
evaluations, final 128-image validation, aggregation, and terminal governance
reporting. The main uncertainty is microbatch-1/accumulation-4 step throughput
and shared-storage image decoding.

Use one eight-GPU instance. Multiple 8xH100 instances would duplicate the same
frozen seed and cannot parallelize the sequential optimizer trajectory; the
small final validation does not justify a second instance.

## Launch

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
smoke/launch_foundir_haze_flux2_source_diverse_pilot_v1_8gpu.sh
```

Monitor:
`tail -f logs/foundir_haze_flux2_source_diverse_pilot_v1.log`.
