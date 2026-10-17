# FoundIR local action energy v1 formal pass 1 — 2026-10-17

`formal1` reached step `1,706` successfully on eight NVIDIA H100 80GB GPUs.
The checkpoint and matched fixed 256-record source-disjoint development
validation were written normally. All protected-data access flags are false.
Measured elapsed time through validation was about `1,829.36` seconds, or
`.5082` wall-clock hours and `4.0652` allocated GPU-hours. Average time through
validation was `1.0720749893171138` seconds per optimizer step.

## Development metrics

| Output | PSNR | LPIPS | PSNR gain vs proposal | LPIPS reduction vs proposal |
|---|---:|---:|---:|---:|
| Proposal | 43.124160178005695 | .050362067760033824 | 0 | 0 |
| Frozen fidelity | 43.39715355634689 | .028955043449059303 | +.27299337834119797 | .02140702431097452 |
| Local action | 43.39715355634689 | .028955043449059303 | +.27299337834119797 | .02140702431097452 |

The selector accepted `0%` of alternatives and selected alpha `1.0` for
`100%` of regions. It is still `.02700662165880202 dB` below the predeclared
`+.30 dB` PSNR gate. LPIPS passes the `-.005` requirement widely.

This result proves the hard fail-closed path does not harm validation at pass
1, but it does not demonstrate learned action selection. Recent
best-candidate cross-entropy is approximately `1.98`; the uniform seven-class
reference is `ln(7)=1.94591`. Thus the ranking objective has not yet shown
better-than-uniform generalization. Continue the frozen later passes without
changing the live loss or decision margin.

Checkpoint SHA-256:
`a865f585558e2262659920925f1ec0d8ba066c5f9d565c38a1df40e3fba26439`.
Validation JSON SHA-256:
`dc0800012d58cf37ecfb9c6beae36709d5e393f5ce4d37cb72df039bbf7a0101`.
