# Session handoff — 2026-09-12

This is the latest operational continuation guide. It supersedes older
handoffs and stale operational-status sections in `agents.md` for the live job
and exact next task only. The advisor-authoritative mainline, governance,
workspace boundary, H100/H200 portability, destructive-action, and protected
FoundIR-test rules in `agents.md` remain higher authority.

## Mandatory fresh-session startup

1. Work only in:
   `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`
2. Tell the user exactly:
   `I am working in zhenhaowang, which is the correct workspace.`
3. Read `agents.md` completely, then read this file completely.
4. Do not restart completed training or begin another model/task branch.
5. Preserve the intentionally dirty worktree. Never stage everything with
   `git add -A`.

## Active objective and hard scope

Execute the advisor-authorized FLUX.2 Blur-only full development evaluation:

- evaluate all 2,595 source-group-disjoint isolated-Blur validation records;
- evaluate checkpoints at steps 2,000, 4,000, 6,000, 8,000, and 10,000;
- compare the complete PSNR/LPIPS curve and determine whether late training
  overfit;
- generate the advisor-ready report and then stop;
- do not access sealed internal or the official FoundIR test;
- do not train, switch to Z-Image, reintroduce Qwen degradation conditioning,
  add mixed degradations, add a pixel refiner, or start another diagnostic.

Latest advisor statement, recorded in
`reports/advisor_governance/advisor_decision.json`:

> 全跑了看。肉眼还行，但是没用。现在应该过拟合了。

This explicitly authorizes the current full development evaluation. No further
question is needed before finishing it.

## Live job — authoritative snapshot

Snapshot time: `2026-09-12T02:44:29Z`.

- Start marker: `2026-09-12T02:25:15Z`.
- Git revision: `25ff5216d700dab4960571df0570adf3c9ffd223`.
- `HEAD == origin/main` at launch.
- Hardware: homogeneous `8 x NVIDIA H200`.
- Runtime profile: batch `4/GPU`, 32 images across eight GPUs per global
  inference batch.
- Observed utilization: normally 100% on all eight GPUs during inference.
- Observed memory: about 19,944 MiB/GPU, with no OOM.
- Current checkpoint: step 2,000.
- Current completed immutable sidecars: `1,364 / 2,595` for step 2,000.
- Later checkpoints: zero sidecars; no aggregate is complete yet.
- No traceback, error, or terminal end marker was present at snapshot time.
- Initial measured total throughput was about 1.2--1.4 images/second. The
  expected total wall time from launch is roughly 2.6--3.0 hours, subject to
  checkpoint-boundary aggregation and storage variability.

The PID file contains a PID from the remote GPU instance. The Codex shell is a
different PID namespace, so `ps`/`pgrep` here cannot prove remote liveness.
Advancing GPU telemetry timestamps, log markers, and sidecar counts are the
authoritative liveness evidence. Do not kill or restart a healthy job because
the remote PID is locally invisible.

## Exact artifacts

Launcher:

```text
smoke/launch_foundir_blur_only_flux2_full_development_v1_8gpu.sh
```

Config:

```text
configs/foundir_blur_only_flux2_full_development_v1.json
SHA-256: 42556934cfed99660916c11de14fb09508b5d628bef5dec7c8a070d8cc39bac9
```

Evaluator:

```text
smoke/evaluate_foundir_blur_only_flux2_full_development_ddp.py
SHA-256: 8e1b2508b79626a15e02d12cec9064db000ec04d98f4dab7ba5907ea670d10cb
```

Persistent runtime evidence:

```text
logs/foundir_blur_only_flux2_full_development_v1.log
logs/foundir_blur_only_flux2_full_development_v1_gpu.csv
logs/foundir_blur_only_flux2_full_development_v1.pid
```

Resume-safe output root:

```text
checkpoints/foundir_blur_only_flux2_full_v1/full_development_evaluation_v1/
```

Expected final artifacts:

```text
checkpoints/foundir_blur_only_flux2_full_v1/full_development_evaluation_v1/result.json
checkpoints/foundir_blur_only_flux2_full_v1/full_development_evaluation_v1/advisor_report.md
reports/advisor_governance/reports/blur_only_full_development_evaluation-*.md
reports/advisor_governance/advisor_hold.json
```

## Frozen scientific identities

Model and training setup:

- FLUX.2 Klein Base 4B;
- isolated Blur only;
- frozen VAE and native Qwen3 text encoder;
- fixed deblurring prompt; no Qwen degradation predictions;
- DiT LoRA rank/alpha 32/32, 50 target modules;
- no mixed degradation and no pixel refiner;
- 10,000 optimizer steps, global batch 32, AdamW, LR 5e-5;
- training records: 105,057 from 7,339 source groups;
- development records: 2,595 from 165 source groups;
- train/development source-group overlap: zero.

Candidate checkpoint hashes are frozen in
`reports/foundir_blur_only_flux2_candidate_bundle_v1.json` with SHA-256
`243ea85048b8c7f8bf3ad29212e497f97fbbb3043ac6b36974fce047991ebf19`.

Evaluation protocol:

- one deterministic aligned 512x512 crop per record;
- the exact same crop and per-record seed across all checkpoints;
- official FLUX.2 shifted schedule, 50 inference steps, guidance 1.0;
- PNG-equivalent uint8 quantization before PSNR/LPIPS;
- record-mean metrics plus source-group bootstrap confidence intervals;
- SOW Stage 2 Blur thresholds: PSNR >=25.9 dB and LPIPS <=0.18.

The earlier fixed 16-image visual snapshots exist for every checkpoint. On
that subset, step-10,000 was 25.6123 dB / 0.1237 LPIPS versus input 23.9045 dB
/ 0.3263 LPIPS. The advisor judged the images visually acceptable enough to
run all development data, while noting that 16 images had no statistical
value and suspecting overfitting.

## Exact monitoring task

Continue monitoring the same live job. A concise state check is:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ
rg 'FOUNDIR_BLUR_FULL_DEVELOPMENT|Traceback|ERROR:|_END status=|ADVISOR_REPORT' \
  logs/foundir_blur_only_flux2_full_development_v1.log | tail -40
tail -16 logs/foundir_blur_only_flux2_full_development_v1_gpu.csv
for step in 002000 004000 006000 008000 010000; do
  count=$(find "checkpoints/foundir_blur_only_flux2_full_v1/full_development_evaluation_v1/step_$step/per_record" \
    -type f -name '*.json' 2>/dev/null | wc -l)
  echo "step_$step=$count"
done
```

At each checkpoint completion, verify:

- exactly 2,595 immutable per-record sidecars;
- `metrics.json` exists and reports 2,595 records / 165 source groups;
- its checkpoint/config hashes match the frozen config;
- PSNR/LPIPS are finite;
- protected-test flags are false;
- log contains `FOUNDIR_BLUR_FULL_DEVELOPMENT_CHECKPOINT_COMPLETE`.

Update this file after every checkpoint completion and any terminal event.

## Terminal handling

Success requires all five checkpoint aggregates, `result.json`,
`advisor_report.md`, the generic immutable governance report, a new advisor
hold, and terminal status 0. Independently inspect the complete curve:

- report the best PSNR step and best LPIPS step;
- report whether either metric worsens before step 10,000;
- call clear late overfitting only when an earlier checkpoint Pareto-dominates
  step 10,000 (PSNR no lower, LPIPS no higher, at least one strict);
- compare every checkpoint to PSNR >=25.9 and LPIPS <=0.18;
- do not select a checkpoint by metrics if its saved fixed visual outputs are
  invalid;
- send the concise advisor-ready report and stop.

On any runtime/scientific failure, do not improvise another run. Preserve the
sidecars and logs, inspect the exact root cause, update this handoff, and obey
the emitted advisor hold. The evaluator is resume-safe, but a restart is not
authorized until the matching advisor/user decision is recorded.

## Git and worktree safety

The evaluation implementation was committed and pushed as `25ff521`.
The worktree contains many unrelated historical tracked modifications and
untracked experiment artifacts owned by the user. Preserve them. If this
handoff or `agents.md` is committed, stage only those exact files and any
verified checkpoint-result/governance artifacts required by the current gate.
