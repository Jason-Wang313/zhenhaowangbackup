# Z-Image Layer-1 corrective bridge v2 readiness — 2026-09-16

## Scope and rationale

The user explicitly authorized execution after reviewing the v1 tiny failure.
This remains a pure Z-Image Base 6B Layer-1 teacher, not an independent
restoration branch. It uses the same fixed 16 Blur train crops and 16
source-disjoint descriptive development crops. It does not use Qwen degradation
conditioning, mixed degradation, sealed internal data, the official FoundIR
test, development-GT optimization, distillation, or decoder tuning.

The v1 teacher learned on ideal linear bridge states but inferred on its own
states. Prediction error moved inference off that line, where the fixed
`LQ-GT` target no longer corrected accumulated error. Fifty solver evaluations
then amplified false texture. V2 addresses that specific failure rather than
adding capacity or merely extending training.

## Corrective formulation

- Mix 50% analytic bridge states with 50% detached on-policy rollout states.
- For any state at nonzero sigma, supervise the corrective velocity
  `v=(state-GT)/sigma`; consequently `state-sigma*v=GT` even off trajectory.
- Align training and inference to the same eight-step, shift-6 solver.
- Retain three-scale LQ latent conditioning at input and DiT depths
  0/5/10/15/20/25 and FP32 rank-64 DiT LoRA.
- Add a zero-initialized 1x1 spatial residual gate to the existing global
  per-channel gate. Their sum passes through a bounded `tanh`, preserving exact
  identity and unit gradient at initialization while limiting overshoot.
- Make endpoint fidelity dominant: flow .5, latent Charbonnier 1, RGB
  Charbonnier 1, RGB-edge Charbonnier .15, LPIPS .08, and a GT-weighted
  near-clean identity regularizer .1. GT controls loss weighting only and is
  never a model condition.

## Frozen tiny protocol

- 1,000 optimizer steps, global batch 8, one homogeneous 8xH100/H200 host.
- Fixed snapshots at 0/250/500/1000 with per-image Input|Output|GT,
  PSNR/LPIPS, latent error, checkpoint hashes, and full train/development
  coverage.
- Gate: train PSNR >=26.4 dB, LPIPS <=.16, both better than input, then visual
  review. No automatic scaling or downstream stage.

Config:
`configs/foundir_blur_only_zimage_corrective_bridge_tiny_v2.json`

Config SHA-256:
`64c1b53518e3b429e11c33f62d84097375a1220b0ea0cf90703cee461d184ddf`

Validation completed before GPU launch:

- seven focused tests pass, including arbitrary off-trajectory correction,
  native token padding, exact zero global+spatial gate, gradient routing, and
  protected-scope locks;
- all fixed image identities and six pinned model-weight hashes pass;
- both homogeneous H100 and H200 profiles pass;
- a real Base-6B CPU smoke loaded DiT/text/VAE, completed two FP32 optimizer
  updates, produced nonzero projection (`2.154e-2`) and LoRA (`2.532e-4`)
  gradients, exercised a detached on-policy rollout, and produced a finite
  corrective target. The final bounded-gate smoke produced projection
  (`2.150e-2`) and LoRA (`2.527e-4`) gradients. Evidence:
  `logs/zimage_bridge_cpu__d5__0v3/`.

## Launch

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
smoke/launch_zimage_corrective_bridge_tiny_v2_8gpu.sh
```
