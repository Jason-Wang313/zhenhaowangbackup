# Full-data Rain precision qualification: PASS

Completed 25/25 steps on eight NVIDIA H100 80GB GPUs.
Launcher started 2026-10-25T21:14:06Z and ended 2026-10-25T21:16:59Z,
status 0. `engineering_pass=true`, `development_gate_pass=false`.
Actual exposure was 800 records / 800 unique source groups / 800 presentations
from the full 38,334-record, 5,780-source eligible pool.

## Matched metrics

PSNR higher is better; LPIPS lower is better.

| Split | Step-0 PSNR | Step-25 PSNR | Step-0 LPIPS | Step-25 LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Fixed 32 training probes | 32.10802291110691 | 32.66665987243047 | 0.07403585378779098 | 0.0708855870179832 |
| Frozen 100 development records | 32.08470176034529 | 32.64423398615503 | 0.0888220098754391 | 0.08657967696897685 |

Development improves .5595322258097397 dB and LPIPS decreases
.002242332906462252 from its matched initialization. Neither final split
passes both 34.2/.08 thresholds. A short qualification does not establish
formal success, precision-only causality, or publishable novelty.

## Engineering checks and cost

- All logged losses/gradient norms finite; A/B parameter updates observed.
- No OOM, NCCL or runtime failure; terminal launcher status 0.
- Saved LoRA header independently checked: 100 tensors, all F32.
- Optimizer, metrics, result and receipt exist. Current config/source hashes
  still match the completed qualification and permit independent formal.
- No protected-data access; original development/probe identities unchanged.
- Median training step: 1.5106424172408879 seconds.
- Peak reported allocated memory: 24.854560375213623 GiB.
- Full launcher: 173 seconds / .04805555555555555 wall hours /
  .3844444444444444 allocated H100 GPU-hours.
- Narrower trainer timer: .032021771482605904 wall hours /
  .25617417186084723 allocated GPU-hours; excludes part of startup.

Result: `checkpoints/foundir_rain_flux2_precision_full_v1/qualification_v1/result.json`.
Result SHA-256: `f6f640c7af829d8918acdf61b67806fdaa511bebda023e959d72136c58f5df59`.
Config SHA-256: `382bb767dfc6dae743f8c70d690f2307b640acc1eb1ff06d25ce557aef981237`.
Log: `logs/foundir_rain_flux2_precision_full_v1_qualification_v1.log`.

## Next permitted step

The user previously authorized full-data training. Engineering qualification
is now satisfied. Independent 3,000-step formal remains unstarted and must
initialize from the original step-2,000 checkpoint, not this qualification.
Frozen evaluations: 0/250/500/1000/2000/3000. Revised budget: 1.5-2.0 wall
hours / 12-16 allocated GPU-hours on eight H100s, allowing startup/evaluation
and IO variation over the approximately 75.5-minute median-step projection.
This is an estimate, not a guarantee of constant long-run throughput.

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_flux2_precision_full_v1_tmux.sh formal formal_v1
```

Do not modify the qualified source/config or claim Rain has passed. No
automatic launch from the CPU-only Codex environment.
