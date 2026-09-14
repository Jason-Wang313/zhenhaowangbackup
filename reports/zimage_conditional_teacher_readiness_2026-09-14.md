# Conditional Z-Image teacher implementation and verification

DIAGNOSTIC ONLY / 非主线. CPU readiness is not restoration quality or an
official FoundIR / SOW result. No new GPU job has run in this session.
Final CPU readiness verified at 2026-09-14T12:11:47Z.

Design: `reports/zimage_conditional_teacher_design_2026-09-14.md`.
Config: `configs/foundir_blur_only_zimage_conditional_teacher_v1.json`, SHA256
`c8bbf5d7dc85c2436f554a7e01e38b9d44d3f554686e001566c248d278da94db`.

Implementation is in `lowlight_restoration/zimage_conditional_teacher.py` and
`smoke/train_zimage_conditional_teacher.py`; launcher:
`smoke/launch_zimage_conditional_teacher_8gpu.sh`. It loads the original pinned
Base/VAE/text weights, creates fresh LoRA, promotes all trainables to FP32,
and adds the zero-initialized 64-to-3840 LQ patch projection. There are 477
trainable tensors / 79,626,240 parameters. The native prediction head remains
pretrained. State, targets and Euler updates use FP32; native forward uses
BF16. LQ is supplied on every call through an explicit argument, including
checkpoint recomputation; no mutable forward hook/context is used.

## Verified CPU evidence

- 27 focused tests pass: `logs/zimage_conditional_teacher_tests_v1.log`.
  They include analytic noise/clean endpoints and oracle Euler integration,
  Base schedule agreement, native forward/input-gradient equivalence,
  condition gradients and output sensitivity, source selection, protected
  scope, pilot review admission, and two-process Gloo accumulated updates
  against a single-process equal-global-batch reference.
- Final nine conditional-teacher tests also pass, including the added RGB
  crop-cache equivalence test across all eight paired D4 transforms:
  `logs/zimage_conditional_teacher_final_tests_v1.log`. Combined coverage is
  28 distinct tests (the other eight conditional tests overlap the 27 above).
- Real 6B, 64x64 native-versus-conditional zero-projection forward and all
  LoRA gradients match exactly. Two synchronized-update sequences repeat
  exactly; AdamW moments and trainables are FP32. Checkpoint save/reload and
  reset are exact. After two updates, replacing LQ with zeros changes velocity
  by maximum absolute .642578125. This establishes an active input path, not
  useful restoration conditioning. Evidence:
  `logs/zimage_conditional_cpu_8912hbw1/rehearsal_rank_0.json`, SHA256
  `af6cd929be01902f3a9f482c0220d77e3f2121e670fc9a494ab16a1cc9001cba`;
  log `logs/zimage_conditional_teacher_cpu_smoke_v1.log`.
- A separate real 6B CPU two-step solver run repeats exactly, decodes finite
  64x64 RGB, writes an Input|Output|GT image and computes finite PSNR/LPIPS.
  These untrained reduced-resolution outputs are not quality evidence.
  `logs/zimage_conditional_cpu_rrvf3ago/inference_smoke.json`, SHA256
  `47c9aaa8a7475085d0a276f154859c41842b12620019a422b39406cc9531bc4f`;
  log `logs/zimage_conditional_teacher_cpu_inference_v1.log`.
- Final trainer stores only one shared rehearsal checkpoint after cross-rank
  weight agreement. The first CPU rehearsal used the earlier per-rank filename;
  the numerical function is unchanged. Every GPU rank will check loading the
  shared file before restoring fresh initialization for formal training.

Pilot selection is frozen before inference: 2048 training groups and 128
development groups, one record/crop per group, all from the existing approved
training-archive train/validation splits. The original 16+16 exact fixed crops
are included. Manifest SHA256:
`19309079bd2a64e253527ff1b637fc3bd2a17522cde8c47fbd3a60283315e11f`.
The trainer pins this hash; metadata and CSV are in
`manifests/foundir_blur_only_zimage_conditional_pilot_v1.{json,csv}`.
Pilot flips/rotations are applied jointly to RGB pairs BEFORE VAE encoding;
the implementation does not assume that latent rotations equal RGB rotations.

Full six-model-artifact hashes, both hardware inventory profiles, and all
2176 pilot image pairs are checked by the launcher preflight. The initial
serial image scan was intentionally interrupted after roughly 960 pairs to
remove high-resolution PNG decode serialization; its log is preserved as
`logs/zimage_conditional_teacher_preflight_v1.log`, not a passing preflight.
The complete replacement uses eight bounded CPU workers and the identical
per-pair checks, with progress every 128 pairs. Its terminal status must be
verified in `logs/zimage_conditional_teacher_preflight_parallel_v1.log`:
**PASS, status 0**, all 2176/2176 pairs, full model hashes and both hardware
profiles. Log SHA256:
`64c193d13d77a06ce70d5e7ccdec66da96049a77cc9eee69d9ba6d29c9454a46`.
This is CPU verification only, not a failed or restarted GPU experiment.

## Runtime and decision boundaries

Tiny: 1200 steps, global batch 8 (microbatch 1, accumulation 1), fixed
0/400/1200 evaluation. Expected 1200 training rows, 2 formal checkpoints,
6 aggregates, 96 triptychs with hash-bound sidecars, 8 native-equivalence
artifacts, 8 rehearsal artifacts, and terminal result. GPU rehearsal requires
exact native forward/backward equality, repeated updates, synchronized weights,
condition sensitivity and checkpoint reload before formal optimizer step 1.
H100 and H200 share the same scientific schedule and batch profile.

Tiny admission uses final TRAIN learnability plus visual review. Development
on 16 examples is reported descriptively. Never relabel prior failures using
these new prospective criteria. Pilot admission additionally requires a
review artifact bound to config, passing tiny result and checkpoint hashes;
no review artifact is fabricated at readiness. Pilot uses global batch 32
(microbatch 1, accumulation 4 on both hardware families) for 2000 steps.
Its displayed train metrics concern the same 16 fixed training examples,
not all 2048 training records; development covers all 128 selected groups.

Do not launch pilot or full training before inspecting the tiny outputs.
The launcher never automatically starts a dependent stage. Tiny launch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_zimage_conditional_teacher_8gpu.sh
```

Persistent monitor:

```bash
tail -f logs/foundir_blur_only_zimage_conditional_teacher_v1_tiny.log
```

Tmux session is `zimage-conditional-tiny`; the executable is resolved at
runtime and verified. If unavailable, the launcher uses nohup plus persistent
logs and PID. It refuses existing evidence and duplicate jobs, records hardware
telemetry and UTC terminal status, and emits the governance result.

The existing user authorization covers this redesign and subsequent staged
work without another advisor round. All prohibited degradation/test/GT
optimization flags remain false. Full teacher quality, distillation and any
conditional decoder intervention remain incomplete.
