# Session handoff — 2026-09-12

This is the latest operational continuation guide. It supersedes older
handoffs and stale operational-status sections in `agents.md` for the live job
and exact next task only. The advisor-authoritative mainline, governance,
workspace boundary, H100/H200 portability, destructive-action, and protected
FoundIR-test rules in `agents.md` remain higher authority.

## Z-Image paired-flow teacher v2 CPU-ready — 2026-09-14T04:16:52Z (LATEST AUTHORITY)

The user explicitly resolved hold
`zimage_blur_teacher_tiny_overfit-a6becc1eff058fec` and directed execution of
the same Blur-only Z-Image route without another advisor round. The matching
decision SHA-256 is
`48f62ed9fc5efb05063b86be8d15615c2b7d43cd848f92e2daee73005b7c036f`;
`advisor_governance.py check` passes.

The failed v1 artifacts remain immutable. Corrected v2 removes the
Turbo-trained ControlNet and implements a Base-native paired latent flow:

- `x_sigma=(1-sigma)*GT_latent+sigma*LQ_latent`;
- velocity target `LQ_latent-GT_latent`;
- exact LQ VAE latent initialization at sigma 1;
- 50-step descending-sigma Euler inference to sigma 0;
- published Base static shift 6 for training and inference;
- CFG 1; frozen VAE/text encoder; DiT LoRA rank 32;
- unchanged fixed 16 train plus 16 source-disjoint visual-development crops.

Config:
`configs/foundir_blur_only_zimage_teacher_paired_flow_overfit_v2.json`,
SHA-256 `d6f5946fcd75050a17d96c093f7590552f89d14caaf7dce0c659fe35fcb2fbb8`.
Readiness report:
`reports/foundir_blur_only_zimage_teacher_paired_flow_v2_readiness_2026-09-14.md`.

Verification completed: 14 focused tests passed, including numerical agreement
within float32 linspace rounding tolerance between the DiffSynth and official
Base 50-step shift-6 schedules; full hashes
passed for the six Base/VAE/text artifacts with no ControlNet; Python compile, imports, shell
syntax, launcher `--check`, governance, and `git diff --check` passed. A real
6B 64x64 CPU forward/backward returned finite loss `3.0448391437530518`, all
476 trainable tensors had finite gradients, and the expected 238 LoRA-B
tensors had nonzero gradients. Trainable count remains 79,380,480.

This Codex shell has PyTorch 2.7.0+cu128 but no visible CUDA device, so v2 has
not trained and has no quality result yet. Exact next action: on one
homogeneous 8xH100/H200 GPU shell, set literal
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and run
`smoke/launch_foundir_blur_only_zimage_teacher_paired_flow_v2_8gpu.sh` inside
a retained tmux session. Then inspect the fixed step-0/step-400 triptychs and
PSNR/LPIPS. Do not begin full teacher training, distillation, or decoder tuning
until this teacher gate passes. Sealed internal and official FoundIR test
remain untouched.

## Advisor-authorized Z-Image continuation — 2026-09-13 (LATEST AUTHORITY)

The advisor has reviewed the completed FLUX.2 Blur-only development result and
explicitly authorized a model-family change. Exact statement:

> 换换Z Image吧。先训teacher，然后蒸馏一下，少步/单步。如果点还是不行，微调一下decoder。

The response is recorded in
`reports/advisor_governance/advisor_decision.json` and resolves hold
`blur_only_full_development_evaluation-d2f43558352e4365` as
`approve_documented_change`.

Current route and ordering are locked as follows:

1. Continue with isolated Blur only. Audit and implement an undistilled
   multi-step Z-Image restoration teacher first.
2. Prove teacher learnability with a bounded tiny-overfit and fixed
   `Input | Output | Ground Truth` visual evidence before full teacher
   training/evaluation.
3. Only after the teacher is valid, distill it to few-step and then evaluate a
   single-step candidate. Preserve the teacher as the quality reference.
4. Decoder fine-tuning is conditional: perform it only if the distilled
   output remains inadequate and after diagnosing a decoder/latent ceiling.

Do not reintroduce Qwen degradation conditioning, mixed degradation, sealed
internal, or official FoundIR test. Do not report Z-Image diagnostic results
as the five-type FLUX.2 mainline result. The exact next task is a read-only
architecture/checkpoint/license/interface audit of the released Z-Image
foundation model and available training/distillation implementations, followed
by the smallest reproducible teacher gate. No Z-Image weights were present in
the workspace at the start of this audit.

## Z-Image teacher tiny-overfit terminal failure — 2026-09-14 (LATEST AUTHORITY)

The authorized tiny-overfit job is complete and must not be restarted. It ran
400/400 optimizer steps on homogeneous 8x `NVIDIA H100 80GB HBM3` with
`flash_attention_2`, then generated all 64 fixed triptychs and all PSNR/LPIPS
aggregates. Training wall time was `433.1167 s`. The launcher ended with status
3 only because the old advisor emitter did not accept the evaluator's explicit
`official_foundir_test_accessed: false`; training and evaluation had already
completed successfully.

- config SHA-256:
  `faba42a614a033b1b53a675555baec7962760ca8cce182083a219a71f59bf94c`;
- step-200 checkpoint SHA-256:
  `8412dfd3da2dec67d56bbec94556edb4fa38276c007c8a3b8aaebf27e6a18a38`;
- step-400 checkpoint SHA-256:
  `2505c79fd554e76297eaaaa6be0d6e5aabbd75a702a2c11666c19df87697177b`;
- immutable evaluation result SHA-256:
  `5e9b8b4a715382bb4d8d5a3e233314da9c34b7b3d7e4c4031663160dbea0e651`.

| Role / state | PSNR input -> output | LPIPS input -> output | Improved |
|---|---:|---:|---:|
| train / step 0 | 24.7232 -> 7.6775 | 0.3085 -> 0.8538 | 0/16, 0/16 |
| train / step 400 | 24.7232 -> 9.3654 | 0.3085 -> 0.8408 | 0/16, 0/16 |
| visual development / step 0 | 23.9045 -> 8.1023 | 0.3263 -> 0.8460 | 0/16, 0/16 |
| visual development / step 400 | 23.9045 -> 9.0932 | 0.3263 -> 0.8949 | 0/16, 0/16 |

The fixed visual gate fails: step 0 is blue-grey noise; step 400 recovers some
coarse layout but has severe green/yellow cast, incorrect content,
low-frequency blocks, and collapsed texture. This is not restoration. The
teacher gate therefore fails and full teacher training, distillation, and
decoder tuning are prohibited.

Read-only code comparison found two strong but unproven incompatibility risks:
the ControlNet is Turbo-trained but attached to undistilled Base, and the
published Base scheduler uses static shift 6 while the current DiffSynth path
used static shift 3. Native and DiffSynth prediction sign/timestep scaling
match; their CFG formulas differ at scale 4. Do not claim a proven root cause
until a controlled diagnostic isolates these factors.

The governance emitter now fail-closed accepts either official-test protection
field (and requires both to be false if both exist), infers failure from
`automatic_metric_gate_pass: false`, and the runner uses
`--outcome auto --diagnostic-only --hold-on-failure`. Focused tests pass.
The immutable result was not changed.

Active hold:
`zimage_blur_teacher_tiny_overfit-a6becc1eff058fec`. Advisor-ready report:
`reports/foundir_blur_only_zimage_teacher_tiny_overfit_failure_2026-09-14.md`.
Exact next action: send that report and request authorization for one bounded
corrected tiny diagnostic that isolates ControlNet/Base compatibility,
scheduler trajectory, CFG, and prediction sign. Before a matching recorded
advisor decision, only read-only audit and report preparation are allowed; do
not run any new inference or GPU job.

## Z-Image teacher preflight checkpoint — 2026-09-13T16:16:24Z

The Blur-only undistilled Z-Image teacher tiny-overfit implementation and all
required runtime weights are now present and verified. The complete artifact
preflight (without `--skip-model-artifacts` or `--skip-large-hashes`) passed:

- config SHA-256:
  `faba42a614a033b1b53a675555baec7962760ca8cce182083a219a71f59bf94c`;
- two DiT shards:
  `ecd5df7768856215812af84d5cc785ba9d84c6d8c84e6939c7ccf30a2e0d2425`
  and
  `e2d5fb75ca504b2d669a33af9380b68e0c1632ecddbfca85f82631958a4b81b1`;
- three Qwen3 text-encoder shards:
  `328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223`,
  `6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5`,
  and
  `7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52`;
- VAE:
  `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`;
- frozen ControlNet:
  `7f611e6d52b133f64b84bef2549fcb84589a766b8255954f96ea34684f52b633`;
- 16 tiny-overfit crops and 16 source-disjoint visual-development crops;
  train/visual source-group overlap is zero;
- official Z-Image source revision
  `26f23eda626ffadda020b04ff79488e1d72004cd` and DiffSynth revision
  `32ef37e4fc38400f9fe14b1affd335dad9cdf50d`;
- sealed internal and official FoundIR test access flags remain false.

Runtime imports, offline LPIPS initialization, Python compilation, shell
syntax, focused tests (`7 passed`), and the launcher `--check` path all pass.
The current Codex shell has no `nvidia-smi` and no exposed CUDA devices, so no
training was started here. The exact next action is to launch
`smoke/launch_foundir_blur_only_zimage_teacher_overfit_v1_8gpu.sh` on one
homogeneous 8xH100/H200 node with
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. After training/evaluation completes,
stop and inspect the fixed step-0/step-400 triptychs plus PSNR/LPIPS before
authorizing full teacher training. Do not proceed directly to full training,
distillation, or decoder tuning.

## Z-Image real-constructor checkpoint — 2026-09-13T17:00:41Z

After the artifact preflight, the pinned weights were loaded through the real
trainer and evaluator constructors on CPU. The runtime identified
`ZImageDiT`, `ZImageTextEncoder`, both Flux VAE halves, and
`ZImageControlNet`; the training unit chain included
`ZImageUnit_PAIControlNet`. Rank-32 DiT LoRA injection produced 476 trainable
tensors / 79,380,480 parameters, all LoRA-only.

A 64x64 structural forward/backward smoke used
`DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch` only because this shell has no CUDA.
With fixed seed 0 it produced finite flow loss `2.9709219932556152`, finite
gradients for all 476 trainable tensors, and the expected 238 nonzero
first-step `lora_B` gradients. The trainer's exact exported state naming then
hot-loaded into all 238 evaluator LoRA targets with zero mismatches and cleared
successfully. A separate two-step evaluator smoke exercised CFG 4.0,
ControlNet, VAE decode, RGB conversion, PSNR, and offline LPIPS end to end.

These reduced CPU runs are interface evidence only, not image-quality evidence
and not a substitute for the frozen 512x512 / 50-step gate. The remaining next
action is unchanged: run the tiny-overfit on a homogeneous 8xH100/H200 node,
then stop for fixed-triptych and PSNR/LPIPS review before any full teacher,
distillation, or decoder work.

## Z-Image launch-hardening checkpoint — 2026-09-13T17:05:59Z

The tiny-overfit startup path was hardened after the real-constructor audit.
The CUDA preflight now runs a finite BF16 attention operation on every visible
GPU using DiffSynth's selected attention backend and records that backend in
`preflight.json`. This catches a missing or ABI-incompatible FlashAttention
kernel before model construction or optimizer step 1.

The trainer and evaluator no longer make all eight ranks independently hash
the complete roughly 29 GB model set. Each distributed phase still performs
one complete SHA-256 verification on rank 0; the other ranks verify the file
set and wait at a barrier. This preserves fail-closed artifact identity while
avoiding approximately 16 redundant full-model reads across training and
evaluation. Python compilation, focused tests (`7 passed`), and the launcher
`--check` path passed after the change. No GPU job was present or started in
this shell, so the exact next action remains the authorized 8-GPU tiny-overfit.

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

## Terminal completion — 2026-09-12 (LATEST AUTHORITY)

The advisor-authorized FLUX.2 Blur-only full development evaluation completed
successfully at `2026-09-12T05:12:58Z` with terminal status `0`. The independent
completion audit passed (`FULL_COMPLETION_AUDIT_PASS`): all five checkpoints
contain exactly 2,595 immutable sidecars, all 12,975 sidecars match the frozen
config/checkpoint identities and protected-test constraints, all metrics are
finite, and independently recomputed aggregates exactly match each
`metrics.json`.

Complete record-mean curve:

| Step | PSNR (dB) | LPIPS | PSNR >=25.9 | LPIPS <=0.18 |
|---:|---:|---:|:---:|:---:|
| 2,000 | 24.962659 | 0.156851 | fail | pass |
| 4,000 | 25.477848 | 0.148589 | fail | pass |
| 6,000 | 25.349482 | 0.147846 | fail | pass |
| 8,000 | 25.635764 | **0.144837** | fail | pass |
| 10,000 | **25.734236** | 0.145406 | fail | pass |

- Best PSNR is step 10,000; it misses the SOW Stage-2 Blur threshold by
  `0.165764 dB`.
- Best LPIPS is step 8,000; every checkpoint passes the `<=0.18` threshold.
- No earlier checkpoint Pareto-dominates step 10,000, so there is no clear
  Pareto overfitting. LPIPS worsens only `0.000570` from step 8,000 to 10,000;
  the step-6,000 PSNR dip later recovers.
- Sealed internal and official FoundIR test were not accessed; all protected
  flags are false.

Terminal artifact SHA-256 values:

- `result.json`: `e08491f2def20de010231fa59b138efd965675bc00706362db61e55ddcaec464`
- local `advisor_report.md`: `dc8b9588f4f2fbc849599ddb1b83820cc1d981d56e27754f40afea093b1ea260`
- immutable governance report JSON: `299faa172bd8aaeec3d59cd8f81888c3ee05a7a30b05d19816d7ecada8d3cf06`
- immutable governance report Markdown: `92e82d042d533db2cdcf70230dec3fe6df25a6eb0b07408d7d91a1e91258bbaa`
- hold event and active `advisor_hold.json`:
  `44a38ff4a3332eab3c902e2e50509bd64c20c73f3ea1f87b907a61a0071b6bb9`
- step 2,000/4,000/6,000/8,000/10,000 `metrics.json` respectively:
  `a8d340f4e769946e0292dff50752f8761cc42f689c0949fad84c9577a1e30e8a`,
  `8ea06743f851b162c10f4829160b6380d25969deac675aba9d2ee5526b7ee3d0`,
  `d81711b3604b90294fece4b93d48aea73da0a874fa5a7e0d7e21efcdadd0f08b`,
  `050ec21c229193bb258f37af4f618b4a54aab8ea1f68d5d0b6a60ffdf24b98a1`,
  and `26009847cb26d6c0a60a2a03369e798a14526dc9a861f2baf20c07d63e561646`.

Exact next action: send the advisor-ready report and stop. A new advisor hold
is active; do not launch training, evaluation, Z-Image, Qwen conditioning,
mixed degradation, or any other experiment until a matching explicit advisor
decision is recorded.

## Live job — authoritative snapshot

Snapshot time: `2026-09-12T04:40:14Z`.

- Start marker: `2026-09-12T02:25:15Z`.
- Git revision: `25ff5216d700dab4960571df0570adf3c9ffd223`.
- `HEAD == origin/main` at launch.
- Hardware: homogeneous `8 x NVIDIA H200`.
- Runtime profile: batch `4/GPU`, 32 images across eight GPUs per global
  inference batch.
- Observed utilization: normally 100% on all eight GPUs during inference.
- Observed memory: about 19,944 MiB/GPU, with no OOM.
- Completed checkpoints: steps 2,000, 4,000, 6,000, and 8,000. Each has exactly
  `2,595 / 2,595` immutable sidecars and a verified aggregate.
- Step-2,000 record-mean metrics: PSNR `24.962659 dB` (fails SOW 25.9),
  LPIPS `0.156851` (passes SOW 0.18). Input baseline on the same crops is
  PSNR `25.849545 dB` / LPIPS `0.291106`.
- Step-2,000 improved fractions: PSNR `0.405780`, LPIPS `0.896724`.
- Step-2,000 source-group-balanced mean: PSNR `21.409985 dB`, LPIPS
  `0.240523`; 95% source-group bootstrap intervals are PSNR
  `[20.792630, 22.065142]` and LPIPS `[0.225900, 0.255222]`.
- Step-4,000 record-mean metrics: PSNR `25.477848 dB` (fails SOW 25.9),
  LPIPS `0.148589` (passes SOW 0.18). PSNR improved `+0.515189 dB` and LPIPS
  improved `0.008262` versus step 2,000, so there is no overfitting signal
  through step 4,000.
- Step-4,000 improved fractions versus input: PSNR `0.423121`, LPIPS
  `0.914066`. Its source-group-balanced mean is PSNR `21.873010 dB` / LPIPS
  `0.228955`, with 95% intervals PSNR `[21.230370, 22.546758]` and LPIPS
  `[0.214107, 0.243807]`.
- Step-6,000 record-mean metrics: PSNR `25.349482 dB` (fails SOW 25.9),
  LPIPS `0.147846` (passes SOW 0.18). Relative to step 4,000, PSNR worsened
  `0.128366 dB` while LPIPS improved only `0.000742`, which is a
  PSNR-specific overfitting signal but not Pareto overfitting.
- Step-6,000 improved fractions versus input: PSNR `0.413873`, LPIPS
  `0.907514`. Its source-group-balanced mean is PSNR `22.004194 dB` / LPIPS
  `0.225940`, with 95% intervals PSNR `[21.378480, 22.605865]` and LPIPS
  `[0.211827, 0.240397]`.
- Step-8,000 record-mean metrics: PSNR `25.635764 dB` (fails SOW 25.9 by
  `0.264236 dB`), LPIPS `0.144837` (passes SOW 0.18). It improves both metrics
  over steps 4,000 and 6,000; the step-6,000 PSNR dip was therefore an
  intermediate fluctuation rather than sustained overfitting through step
  8,000.
- Step-8,000 improved fractions versus input: PSNR `0.443545`, LPIPS
  `0.910597`. Its source-group-balanced mean is PSNR `22.171110 dB` / LPIPS
  `0.222588`, with 95% intervals PSNR `[21.505189, 22.809813]` and LPIPS
  `[0.208230, 0.237342]`.
- Current and final checkpoint: step 10,000, with 20 sidecars at snapshot
  time.
- No traceback, error, or terminal end marker was present at snapshot time.
- Each completed checkpoint required about 33 minutes. The expected remaining
  time at this snapshot is roughly 0.5--0.6 hours, subject to
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
