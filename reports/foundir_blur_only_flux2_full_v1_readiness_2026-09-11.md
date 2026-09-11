# FLUX.2 Blur-only full-training readiness — 2026-09-11

## Status

`READY_FOR_ONE_8XH100_OR_8XH200_LAUNCH`

Advisor authorization is recorded in
`reports/advisor_governance/advisor_decision.json`: after the implementation
was described as the original FLUX.2 training core with Qwen degradation
prediction, mixed degradation, and pixel refiner disabled, and official
50-step inference enabled, the advisor replied `可以试试`.

## Frozen experiment

- Model: FLUX.2 Klein Base 4B; frozen VAE and native frozen Qwen3 text encoder;
  fresh trainable DiT LoRA rank/alpha `32/32` on the same 50 target modules.
- Condition: one fixed deblurring prompt. No Qwen degradation prediction.
- Data: all `105,057` isolated Blur records in the FoundIR training archive,
  from `7,339` source groups. No mixed degradation and no pixel refiner.
- Training: homogeneous 8 H100 or 8 H200, global batch `32`, `10,000`
  optimizer steps / `320,000` presentations / approximately `3.046` dataset
  passes, LR `5e-5` with 200-step warmup and cosine decay. H100 uses microbatch
  `2` per rank with accumulation `2`; H200 uses microbatch `4` with accumulation
  `1`. Both profiles preserve the same scientific schedule.
- Flow: pure-noise-to-clean reference-conditioned rectified flow.
- Candidate checkpoints: steps `2,000`, `4,000`, `6,000`, `8,000`, and `10,000`.
- Development protocol: the full source-group-disjoint Blur development split
  has `2,595` records from `165` source groups. Before training, 16 fixed
  development records from 16 groups were frozen for visual checkpoint review.
  The same `Input | Output | Ground Truth` triptychs are generated at step 0
  and every candidate checkpoint.
- Selection: automatic checkpoint selection is disabled. Training stops after
  triptych generation. Human visual screening must occur before full
  development PSNR/LPIPS comparison.
- Protected data: sealed internal and official FoundIR test are not accessed.

## Identity and verification

- Config: `configs/foundir_blur_only_flux2_full_v1.json`
  (`ba5e2975ba36c9300d454c573882312ddda7415a3de370efcef707899512b61c`)
- Trainer: `smoke/train_foundir_blur_only_flux2_full_ddp.py`
  (`b7bd5ea7b96bc0e371f82b539f355ec31e9d21e76bc5abd61d4f6a27cfee01d7`)
- Runner: `smoke/run_foundir_blur_only_flux2_full_v1_8gpu.sh`
  (`758091a9f0a0b5776853251aa19d89aa98918c95b2df7b7ef1344fde78d35993`)
- Launcher: `smoke/launch_foundir_blur_only_flux2_full_v1_8gpu.sh`
  (`e3811809accc65a5b4767e6a353ab5af1127bc9e4159e127d2e5d47cda8e8f15`)
- Implementation code revision: `1ca05787498def612114eef697f382562580aafc`;
  this exact commit is preserved in the current `origin/main` history.
- CPU verification: Python compilation passed; shell syntax passed; `9/9`
  focused tests passed, including both hardware profiles.
- Full preflight: config/model/manifest/image identities passed; exact training
  and development cardinalities passed; train/development source-group overlap
  is `0`; all protected-data flags remain false.
- Resume: the runner resumes only from the newest complete candidate checkpoint
  whose config, optimizer, checkpoint hash, and step identity match. Partial or
  differing artifacts fail closed. Runtime GPU detection uses CUDA properties,
  never the instance hostname; a shared `flock`, not cross-node PID identity,
  is the duplicate-run authority.

## Launch

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/launch_foundir_blur_only_flux2_full_v1_8gpu.sh
```

The detached run writes persistent training logs and five-second per-GPU
telemetry. Completion emits an advisor report and a new mandatory hold, then
stops for visual review.
