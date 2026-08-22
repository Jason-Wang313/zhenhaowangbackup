# FoundIR-Lowlight Gate 3 initial bundle result — 2026-08-22

## Terminal outcome

The four-H100 bundle completed with terminal status 0. It did not access
protected FoundIR-Lowlight IDs 1101--1150.

- Qwen same-scene real exposure ordering: passed.
- Compact deterministic training/inference/metrics: completed.
- Compact deployment admission: failed because of an SDSD LPIPS regression.
- Gate 3 overall: still in progress; the already-frozen four-way downstream
  causal Qwen comparison remains mandatory.

## Qwen ordinal result

The frozen step-600 checkpoint passed all predeclared real relative-exposure
thresholds:

- continuous concordance: `0.980132`;
- hard non-inversion: `1.000000`;
- median stack Spearman: `1.000000`;
- positive-stack Spearman fraction: `1.000000`;
- pooled expected-level span: `1.070585`.

This proves relative exposure sensitivity only. It is not an absolute L1--L5
accuracy claim and not evidence of downstream restoration gain.

## Compact result

The 1,684,172-parameter model completed 1,200 steps / 38,400 presentations and
245 exact-native-dimension outer outputs. Training plus inference took
`1056.612 s`. Final compact metrics were:

| Family | PSNR | LPIPS | PSNR vs fixed | PSNR vs cascade |
|---|---:|---:|---:|---:|
| LOL-v2 | 16.2655 | 0.4874 | +6.8343 | +1.6557 |
| SDSD | 19.4742 | 0.2787 | +0.5152 | +1.4471 |
| SICE | 16.6800 | 0.3601 | +4.1964 | +5.1713 |

Equal-family PSNR was `17.4733 dB`, versus `14.7152 dB` for the FoundIR
cascade. The model therefore demonstrated substantial cross-family PSNR
capacity. It was not admitted because SDSD LPIPS regressed by `+0.02658`
versus fixed exposure, exceeding the frozen `+0.005` limit. The dominant
bottleneck is perceptual/detail quality under the MSE-led compact objective,
not exposure correction or insufficient PSNR capacity.

The target-only compact/cascade blend ceiling reached equal-family PSNR
`19.9098 dB` and LPIPS `0.3261`. It showed large complementarity on LOL-v2 and
SDSD but only `+0.0372 dB` over the already-dominant compact output on SICE, so
it correctly failed the every-family complementarity rule. It remains a
non-deployable ceiling.

## Next decision

The compact model remains a scientific control, not a deployment candidate.
Because Qwen passed the ordinal gate, run the frozen measured-only,
oracle-plus-measured, measured-plus-Qwen, and measured-plus-shuffled-Qwen
comparison with identical restoration weights. Qwen is retained only if it
passes the paired-real causal thresholds. Gate 4 remains blocked until that
decision is complete.

Protected-test status: `NOT_ACCESSED_DURING_GATE3_INITIAL_BUNDLE`.
