# FoundIR-Lowlight Gate 3 causal Qwen result

Date: 2026-08-22 UTC

## Decision

The initial four-way causal Qwen matrix **failed**. Gate 4 remains blocked.
This is not an ordinal-Qwen failure and not a restoration-interface capacity
failure: the target-derived oracle condition improved every family, while the
actual Qwen condition did not beat measured-only and shuffled-Qwen controls.

No protected FoundIR-Lowlight ID 1101--1150 was accessed.

## Frozen aggregate result

| Variant | Equal-family PSNR | Equal-family LPIPS |
|---|---:|---:|
| measured-only | 17.3927 | 0.3865 |
| oracle + measured | 18.9932 | 0.3883 |
| Qwen + measured | 17.2685 | 0.3934 |
| shuffled Qwen + measured | 17.3443 | 0.3942 |

| Family | Oracle PSNR gain | Qwen gain over measured | Qwen gain over shuffled | Qwen LPIPS change over measured | Oracle gain recovered |
|---|---:|---:|---:|---:|---:|
| LOL-v2 | +2.1001 | +0.0751 | -0.0298 | +0.00780 | 3.58% |
| SDSD | +1.9823 | -0.5719 | -0.2140 | +0.00812 | -28.85% |
| SICE | +0.7193 | +0.1241 | +0.0164 | +0.00479 | 17.26% |

Every family failed the predeclared causal threshold. Qwen therefore has not
earned benchmark-critical restoration complexity in this interface.

## Attribution before correction

The oracle arm proves that the frozen compact model plus nine-parameter global
tone interface can exploit a useful condition. The failure is specifically how
the generic Qwen representation is decoded:

- the Qwen expected severity correlates with the target-derived first-channel
  log-gain by 0.519 on training SDSD and 0.681 on training SICE;
- on outer development the same correlation is 0.303 on unseen LOL-v2, 0.789
  on SDSD, and 0.761 on SICE;
- grouped five-fold ridge attribution finds only a 4--7% oracle-summary MSE
  reduction from heavily regularized Qwen features;
- the 32-D projected visual summary is not stable under leave-one-family-out
  transfer, while the low-dimensional posterior is more stable;
- a free-form jointly trained Qwen branch therefore amplifies domain-specific
  rendering bias, most visibly on SDSD, instead of extracting the narrower
  monotone exposure signal.

## Authorized correction 1 rationale

One basic-interface correction is scientifically justified because measurable
Qwen exposure information exists but the decoder did not isolate it. The
correction must not add generic restoration capacity. It will:

1. freeze the measured-only base and compact restorer;
2. use the five soft severity probabilities, expected severity, entropy, and
   margin as the primary Qwen signal;
3. predict only a bounded monotone exposure/tone residual after residualizing
   against measured statistics;
4. shrink corrections by Qwen uncertainty and training-support distance;
5. train correct and within-family group-shuffled Qwen pairs together with a
   paired contrastive ranking constraint;
6. retain identical restoration weights for measured, oracle, Qwen, and
   shuffled-Qwen evaluation.

This is correction index 1 of at most 2. It is not permission to change the
development suite, lower thresholds, start FLUX, or access FoundIR pixels.

## Evidence identities

- result JSON SHA-256:
  `5503ebbe533c3a944c563f07404c3ebdef8c0c92686d79a50b688f0f3559d9a0`;
- metric rows SHA-256:
  `7ed72b536a65df908f8d215c94020e4ad544966636e4313c63d058ec73a79805`;
- prediction manifest SHA-256:
  `edb5257e58c1b94c9caa3b28d327b5b4caedcd42f582c78dcc1bb30864a66846`;
- Qwen feature bundle SHA-256:
  `a9e006f03949ecc348506f78a94b722079cb3a517c53dede50cdd8b8fb75e506`;
- adapter checkpoint SHA-256:
  `3927884f2f812ea34710f797463c4cae3d14ee923b8cf3fd0332ac0d6c046e79`.
