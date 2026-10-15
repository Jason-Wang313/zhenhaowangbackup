# FoundIR matched Compositional EBT lower-order pilot v1 — terminal result

> **DIAGNOSTIC ONLY / 非主线。** This is not a FLUX.2 mainline, SOW, or
> held-out-triple result.

## Terminal status

- Runtime status: success; terminal marker status `0`.
- Frozen scientific gate: `lower_order_learnability_fail`.
- Hardware actually used: one `NVIDIA H100 80GB HBM3`.
- Wall time: `24.270043` seconds.
- Optimizer steps: `600` total (`300` Plain + `300` Compositional).
- GPU cost: approximately `0.006742 GPU-hours`.
- Result SHA-256:
  `0bfcc30805387e22b8ede15e91e1aa13b6c3d8335d924695d0d99d746ce4b78f`.

No held-out-triple image, FoundIR 60-image triple endpoint, sealed internal
data, or official FoundIR test was accessed.

## Frozen development result

| Arm | Input PSNR | Output PSNR | Gain | Input Smooth L1 | Output Smooth L1 | Improvement |
|---|---:|---:|---:|---:|---:|---:|
| Plain | 13.881658 | 14.141973 | +0.260315 dB | .020455223 | .019265166 | .001190057 |
| Compositional | 13.881658 | 14.175251 | +0.293593 dB | .020455223 | .019118107 | .001337115 |

Both arms passed the predeclared development PSNR, Smooth L1, finite-output,
and non-collapse conditions. Compositional exceeded Plain by only
`0.033279 dB` and `.000147058` Smooth L1 on this small lower-order development
set; this is not evidence of unseen-triple generalization.

## Why the frozen gate failed

The predeclared train-loss condition required last-25/first-25 mean loss
`<=0.9`. Plain produced `1.361700`; Compositional produced `1.377185`.
Therefore both arms formally fail even though both improve the development
metrics.

A read-only aligned-epoch audit does not reverse the frozen decision. Mean of
the last five complete 15-step epochs versus the first five is `0.952996` for
Plain and `0.966471` for Compositional: a small decline, but not the required
10%. The raw loss is also affected by changing deterministic crops and widely
varying degradation difficulty. Consequently the honest interpretation is:
the 300-step pilot learned a small positive lower-order correction, but did not
meet the frozen convergence criterion and does not justify triple promotion.

## Stop condition

The terminal failure created governance hold
`foundir_compositional_ebt_lower_order_pilot_v1-c31644dbf2370957`.
Do not rerun, scale, change the gate, or access held-out triples until a new
advisor/user decision resolves that hold.

The user has recorded that one full `8×H200` node is available for later
experiments. The present tiny 146k-parameter pilot did not need eight GPUs;
future substantial work should be designed to use the available homogeneous
8×H200 node where scaling is technically meaningful.
