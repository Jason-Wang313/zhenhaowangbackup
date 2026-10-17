# FoundIR local action energy v1 formal pass 2 — 2026-10-17

`formal1` reached step `3,412` successfully on eight NVIDIA H100 80GB GPUs.
The checkpoint and fixed 256-record source-disjoint development validation
were written normally. All protected-data access flags are false. Elapsed
time through validation was about `3,449` seconds, or `.958` wall-clock hours
and `7.665` allocated GPU-hours. Average time through validation was
`1.0107961409268869` seconds per optimizer step.

## Development metrics

| Output | PSNR | LPIPS | PSNR gain vs proposal | LPIPS reduction vs proposal |
|---|---:|---:|---:|---:|
| Proposal | 43.124160178005695 | .050362067760033824 | 0 | 0 |
| Frozen fidelity | 43.39715355634689 | .028955043449059303 | +.27299337834119797 | .02140702431097452 |
| Local action | 43.39715355634689 | .028955043449059303 | +.27299337834119797 | .02140702431097452 |

The selector still accepts `0%` of alternatives and selects alpha `1.0` for
every region. PSNR remains `.02700662165880202 dB` below the joint gate; LPIPS
passes. Recent best-action cross-entropy fluctuates around `1.95–1.98`, close
to or worse than the uniform seven-class value `ln(7)=1.94591`.

After two full passes there is no evidence of deployable action selection.
This substantially lowers the probability of later success. The run is
numerically and operationally healthy and has already entered pass 3; it
continues under the frozen contract unless the user explicitly stops it.

Checkpoint SHA-256:
`1a5108bfb0ac0bca7e92c08ea936de0ea0fbc4af3af0b47e209d4d1f95dded48`.
Validation JSON SHA-256:
`e29b5ab55ee9869f7d055ee15e4d209092c968a93c75485df2d26e64a33e3186`.
