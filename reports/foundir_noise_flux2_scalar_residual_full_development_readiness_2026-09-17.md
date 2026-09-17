# FLUX.2 Noise frozen alpha=.30 full-development readiness

## Frozen experiment

- Model: FLUX.2 Klein Base 4B isolated-Noise step-400 LoRA.
- Checkpoint SHA-256: `1b4ed05ad5ad403a3d7fcc633eca97cdb69d00454e357da0c3d3131ac2c3ff1a`.
- Config SHA-256: `5e7a356ea89d21f7b60b4b736f851ef26a0216d8555d73846f527b958b5b7dd6`.
- Candidate: `LQ + .30 * (FLUX - LQ)`, uint8 quantized before metrics.
- Data: all 1,859 isolated-Noise development records across 34 source groups.
- Formal gate: mean PSNR >=38.9 and mean LPIPS <=.10.
- Tail statistics are descriptive and cannot veto the SOW result, per explicit
  user acceptance-scope correction.
- Sixteen fixed `Input | Candidate | GT` visuals require human review.
- No weight update, development tuning, Qwen degradation conditioning, mixed
  degradation, sealed internal access, or official-test access.

## Readiness evidence

Python compilation and shell syntax pass. Seven combined scalar/full-development
focused tests pass. Preflight verified config/evidence/checkpoint/manifest
identities, exactly 1,859 records and 34 source groups, train-development source
isolation, all 3,718 LQ/GT image hashes, valid 512 crops, all sixteen fixed visual
IDs, and H100/H200 profiles. Governance and launcher `--check` pass.

Resolved design hold:
`flux2_noise_scalar_full_development_design_v1-57099e444d0cb19b`.

## Runtime and launch

For one homogeneous 8xH100 instance, complete end-to-end wall time is estimated
at **25--35 minutes (central 30)**, equal to 3.3--4.7 aggregate GPU-hours
(central 4.0). For one homogeneous 8xH200 instance, estimate **18--27 minutes
(central 22)**, equal to 2.4--3.6 GPU-hours (central 2.9). This includes
preflight, loading, all inference, metrics, visuals, aggregation, and reporting.
The principal uncertainty is shared-storage I/O and inference/LPIPS throughput.

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
smoke/launch_foundir_noise_flux2_scalar_residual_full_development_v1_8gpu.sh
```

Monitor:
`tail -f logs/foundir_noise_flux2_scalar_residual_full_development_v1.log`.
The run stops after its result and governance report; no automatic follow-on.
