# FLUX.2 Low-light frozen scalar-residual full-development readiness

## Decision and exact scope

After reviewing the 16-record source-disjoint feasibility result and its thin
`+.12674 dB` PSNR margin, the user explicitly instructed: `let's proceed to
low light now`. This authorizes one frozen confirmation only:

- isolated `14Lowlight` validation records only;
- all 1,400 records from 21 source groups, disjoint from training groups;
- exact step-400 Low-light checkpoint;
- official 50-step FLUX.2 shifted inference schedule;
- `candidate = LQ + .925 * (FLUX - LQ)`, quantized to uint8;
- formal checks: mean PSNR >=18.9 and mean LPIPS <=.28;
- tails and sixteen fixed source-diverse visuals are descriptive;
- no alpha tuning, weight update, Haze, Rain, mixed degradation, protected
  sealed data, or official FoundIR test access.

Frozen config:
`configs/foundir_lowlight_flux2_scalar_residual_full_development_v1.json`, SHA
`fea0227d13b6d572897d683a8833e2312047cf28be2d4cf12462ddf95092856e`.
Checkpoint SHA:
`592e49ca2606565c5aeec8aadeb5466cf0c34b30024d5c02f0f2292ad28c90d8`.

## Verification

- governance resolves the hash-bound design hold
  `flux2_lowlight_scalar_full_development_design_v1-b07dc4830b5426b4`;
- focused tests: 5/5 pass;
- Python compilation and both shell syntax checks pass;
- complete launcher `--check` passes;
- all 1,400 records, 21 source groups, and train/development group isolation
  are verified;
- all 2,800 LQ/GT paths and SHA-256 identities pass;
- the sixteen fixed visual records span sixteen different source groups;
- H100 and H200 runtime profiles preserve identical evaluation semantics;
- all forbidden/protected access flags are false.

This Codex shell has no `nvidia-smi` or visible GPU, so no GPU job was started
here. That is an environment limitation, not a preflight or scientific
failure.

## Mandatory total-time disclosure and launch

On one homogeneous 8xH100 instance, estimated total end-to-end wall time is
**20--27 minutes, central estimate 23 minutes**, or **2.7--3.6 aggregate
GPU-hours, central about 3.1**. On one homogeneous 8xH200 instance, estimated
wall time is **15--21 minutes, central estimate 17 minutes**, or **2.0--2.8
GPU-hours, central about 2.3**. This includes governance/preflight, model
loading, all 1,400 50-step inferences, three-arm metrics, sixteen visuals,
aggregation, and advisor reporting. The main uncertainty is shared-storage I/O
plus actual inference/LPIPS throughput.

Run exactly once in the homogeneous eight-GPU shell:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
smoke/launch_foundir_lowlight_flux2_scalar_residual_full_development_v1_8gpu.sh
```

Monitor
`tail -f logs/foundir_lowlight_flux2_scalar_residual_full_development_v1.log`.
The launcher is detached, resume-safe at immutable per-record sidecars, and
rejects duplicate terminal output. On completion, stop for metric and fixed
visual audit; there is no automatic Haze or Rain action.
