# Z-Image conditional teacher maximum single-instance scale readiness

DIAGNOSTIC ONLY / 非 official FoundIR test or SOW acceptance result.

## Authorized run

The user explicitly authorized the maximum single-instance continuation after
reviewing the completed 2048-source pilot. One homogeneous 8xH100/H200 node
continues from the exact pilot step2000 FP32 trainable state with a fresh FP32
AdamW optimizer and100-step warmup. Tiny and pilot failures remain failures.

- Full training: all105,057 isolated-Blur training records /7,339 source groups.
- Increase over pilot:+103,009 records /+5,291 source groups.
- 20,000 additional optimizer steps,global batch32,microbatch1/rank,accumulation4.
- 640,000 presentations,approximately6.0919 passes over full records.
- Same model/objective:Z-Image Base6B,frozen original VAE/native Qwen3 text
  encoder,rank32 FP32 LoRA,persistent aligned LQ patch projection at every DiT
  call,Gaussian-noise-to-GT flow,LR5e-5/condition LR1e-4,clip1.
- Same 50-step shift6/CFG1 inference. No failed-adapter substitution.
- Save/evaluate a fixed165-source monitor at0 and every2,000 steps through20,000.
- After training, freeze the union of monitor-best PSNR,monitor-best LPIPS and
  final (maximum three candidates), then evaluate each on all2,595 Blur
  development records/165 source groups. Candidate selection occurs before
  those complete metrics are read.
- Full development gate requires output improvement over input in both metrics
  AND PSNR>=25.9/LPIPS<=.18, followed by actual visual review. It is a training-
  archive proxy and cannot establish official FoundIR-test acceptance.
- No automatic distillation/decoder training. No Qwen degradation conditioning,
  mixed degradation,sealed internal,official FoundIR test or development-GT
  training.

## Runtime and recovery

Config:`configs/foundir_blur_only_zimage_conditional_teacher_full_v1.json`,
SHA256 `d3fd35437abd6239c450f42bbb9674133aeb84f8d929dce00ced000f5b9ec8d7`.
Authorization SHA256:
`56257204db0239315301e0b45db1469a7dcb7e3aea8d13eeb1805847f3ca4eb8`.
Parent result/checkpoint SHAs:
`45b74d23629f6fe7a5f3d73d6f28891f29f044e86461ee9df220f5a49a984097` /
`cf4a9b0f710f4026816df2898e10d299624139aa5472eec0610297540b465c54`.

The launcher retains a tmux shell when available and has a nohup fallback,
unique log/output guards,one shared training lock,terminal markers and5-second
eight-GPU telemetry. The training and evaluation phases are separate torchrun
processes in one launcher, so completed training checkpoints survive a later
evaluation failure. Runtime outputs are immutable; recovery code must inspect
them and must never relaunch training blindly.

Host cache estimate is conservatively below roughly350GiB after six augmented
passes:each rank caches its aligned RGB crops and encountered D4 latent pairs.
The known 8-GPU environment has ample host memory; GPU memory is expected near
the prior approximately24GiB/rank because microbatch/model geometry are unchanged.
Measure actual host/GPU use rather than treating estimates as evidence.

At the measured pilot rate2.18s/step, optimization alone is approximately12.1h.
Monitor and maximum-three full-development evaluations bring the conservative
single-node total to approximately14--18h. Actual first2,000-step telemetry is
authoritative and should update this estimate.

## Verification

`smoke/launch_zimage_conditional_teacher_full_8gpu.sh --check` passed exit0:

- 40 focused tests;
- Python compilation and shell syntax;
- immutable authorization/config/parent identities;
- all105,057 training records,7,339 groups,2,595 development records and165
  groups,with source-disjoint split and isolated Blur only;
- all107,652 pair paths inside the approved training archive;
- all six large model blobs and both H100/H200 inventory profiles;
- bounded candidate selection and SOW+input-gain gate tests.

Log:`logs/zimage_conditional_teacher_full_preflight_v1.log`.
CPU/preflight readiness is not GPU execution or quality evidence. No full output
or log existed at verification time and no GPU job was started by Codex.

Exact GPU-shell launch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_zimage_conditional_teacher_full_8gpu.sh
```

Monitor:

```bash
tail -f logs/foundir_blur_only_zimage_conditional_teacher_full_v1.log
```
