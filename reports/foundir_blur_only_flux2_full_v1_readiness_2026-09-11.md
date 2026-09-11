# FLUX.2 Blur-only full-training readiness — 2026-09-11

## Status

`READY_FOR_ONE_8XH200_LAUNCH`

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
- Training: 8 H200, global batch `32` (`4` per rank, no accumulation), `10,000`
  optimizer steps / `320,000` presentations / approximately `3.046` dataset
  passes, LR `5e-5` with 200-step warmup and cosine decay.
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
  (`01b00df153f3dd9279f63c039857d0989ca88f331c32b1e02e9c734d895bd0ae`)
- Trainer: `smoke/train_foundir_blur_only_flux2_full_ddp.py`
  (`841e1ffc08289b2c39384d6011ab04022ffcc61354df12cd8975002dca952298`)
- Runner: `smoke/run_foundir_blur_only_flux2_full_v1_8gpu.sh`
  (`ce0813d13bac13077c21fd785047640add4fb30f693057ff5d75c31420d866b8`)
- Launcher: `smoke/launch_foundir_blur_only_flux2_full_v1_8gpu.sh`
  (`eaf82f1d31c552905dba93847b106383167e7f65e6c3cd841a0276f281acc90a`)
- Code revision: `1ca05787498def612114eef697f382562580aafc`, equal to `origin/main`.
- CPU verification: Python compilation passed; shell syntax passed; `9/9`
  focused tests passed; integrated launcher `--check` passed.
- Full preflight: config/model/manifest/image identities passed; exact training
  and development cardinalities passed; train/development source-group overlap
  is `0`; all protected-data flags remain false.
- Resume: the runner resumes only from the newest complete candidate checkpoint
  whose config, optimizer, checkpoint hash, and step identity match. Partial or
  differing artifacts fail closed.

## Launch

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/launch_foundir_blur_only_flux2_full_v1_8gpu.sh
```

The detached run writes persistent training logs and five-second per-GPU
telemetry. Completion emits an advisor report and a new mandatory hold, then
stops for visual review.
