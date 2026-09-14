# Session handoff — 2026-09-12

This is the latest operational continuation guide. It supersedes older
handoffs and stale operational-status sections in `agents.md` for the live job
and exact next task only. The advisor-authoritative mainline, governance,
workspace boundary, H100/H200 portability, destructive-action, and protected
FoundIR-test rules in `agents.md` remain higher authority.

## User-authorized scaling after failed tiny — 2026-09-14T12:56:33Z (LATEST AUTHORITY)

User explicitly said: “最新训练曲线仍在改善(if so, why don't we just proceed,
and don't ovethink it, scale”, AFTER the failed metrics/visual report. Execute
the already-prepared diverse pilot. This supersedes the prior no-pilot directive;
it does NOT relabel tiny as passing or authorize protected data. No additional
advisor question/model-switch ceremony is needed. Do not keep redesigning tiny.

Design: `reports/zimage_conditional_teacher_scale_pilot_2026-09-14.md`.
Pinned explicit admission exception:
`reports/zimage_conditional_teacher_scale_authorization_v1.json`, SHA256
`21c6cc9b8f20e37fde16fad5799e4fa701118c4730ba5defc4fca6d1e6365460`.
Governance hold `zimage_conditional_teacher_scale_pilot_authorization-260d83120dd1a2d4`
is resolved using this exact user instruction, NOT a fabricated advisor reply.

Same architecture/config/objective/LRs/50-step inference. Initialize exact tiny
step1200 (52a61cd9...), reset optimizer/warmup; 2048 training sources,128 source-
disjoint development sources,2000 ADDITIONAL steps,global batch32, paired D4.
Original16+16 fixed crops retained. Pilot metric AND visual criteria unchanged.
Tiny remains failed. No automatic full training/distillation/decoder continuation.
No development-GT optimization, Qwen degradation condition, mixed degradation,
sealed internal or official FoundIR test. This is not an official SOW result.

Implementation adds explicit `--scale-pilot` launcher / hash-bound trainer
admission. Default pilot launch still requires passing tiny review. Runtime and
result embed admission provenance, source initialization and optimizer reset.
36 focused tests pass including exception scope and identity/budget tampering.
Full preflight PASSED (`CONDITIONAL_PREFLIGHT_PASS`, exit0): exact six model
artifact hashes, tiny checkpoint/authorization identity, all2176 pilot pairs,
both H100/H200 hardware profiles. Log:
`logs/zimage_conditional_teacher_scale_preflight_v1.log`.
No GPU is exposed here. No pilot log/output exists; no GPU job started.

After preflight, single GPU-shell launch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_zimage_conditional_teacher_8gpu.sh --scale-pilot
```

Monitor: `tail -f logs/foundir_blur_only_zimage_conditional_teacher_v1_pilot.log`.
Output: `checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_pilot/`.
Tmux session `zimage-conditional-pilot` if available; verified nohup fallback.
Expected:2000 training rows,2 checkpoints,6 aggregates,432 triptychs/sidecars,
eight CUDA rehearsals and terminal result. Step0/1000/2000 snapshots contain
16 training +128 development images. Review final metrics/images and fixed16
development subset separately; never compare differing sample means as a curve.

## Conditional teacher tiny terminal failure — 2026-09-14T12:42:25Z (historical)

The user launched the new conditional teacher; it finished 1200/1200 normally
on 8xH100 at 12:36:45Z, status0, launch revision e2d2c3c. This is a scientific
learnability/fidelity failure, not a runtime failure. No job remains to monitor.
Independent audit passed 1200 ordered finite rows, two exact FP32 checkpoints
(477 finite tensors each), six recomputed aggregates, 96 hash-matching
triptych/sidecar pairs and all eight native/reproducibility CUDA rehearsals.
Config SHA remains c8bbf5d7dc85c2436f554a7e01e38b9d44d3f554686e001566c248d278da94db.
Result SHA: ed945a012bd39ac6d08cc3d08f5f9ce0ce4e662f229405ecc21fe06d4281359d.
Step1200 SHA: 52a61cd9412f494c2141d5f34ffd9cbb2dc66911755e008f2828a63071ea93a7.

Train input24.7232/.3085 -> output21.6206/.2783: PSNR loses3.1026dB;
LPIPS improvement .030242 just passes its >.03 criterion. Fixed training
probe .464905 ->.399082, ratio .858417, fails <.8. Thus 2/3 TRAIN criteria
fail; descriptive development is not the reason for refusal. Development
input23.9045/.3264 ->19.9942/.4420, zero of16 improves PSNR.

Visual review FAILS: statue0042483 recovers rough form but repaints pattern;
near-clean train0091486 falls36.8832 ->22.1782dB and changes grass/rock/stem
content; dev0055608 bus structure is painted/distorted; dev0092666 rearranges
leaves. There is no passing visual-review artifact. DO NOT launch the prepared
pilot, full training, distillation, decoder tuning, or rerun this tiny job.

Important interpretation: both train AND development improve from step400
to1200 (train18.1619/.5004 ->21.6206/.2783; dev17.1693/.5604 ->19.9942/.4420).
This sampled curve does NOT show late overfitting. Native semantics plus a
functioning persistent LQ path were insufficient within this fixed budget;
the unique cause remains unresolved. Do not call the route proven impossible,
promise more steps will fix it, or silently change gates. The next decision
requires evidence about conditional reconstruction/content preservation and
sampling error, informed by healthy VAE/oracle results, before another run.

Detailed audit: `reports/zimage_conditional_teacher_tiny_terminal_audit_2026-09-14.md`;
machine audit: `reports/zimage_conditional_teacher_tiny_terminal_audit_v1.json`.
Launcher governance report: `zimage_conditional_teacher_tiny-00140bbcd7f61c87`.
No new advisor round or model-switch ceremony is requested. This check turn
only audited and documented; no training code, threshold or job changed.
All protected degradation/test/development-GT optimization flags remain false.
The overall teacher -> distillation -> conditional decoder goal is unfinished.

## Conditional teacher CPU-ready — 2026-09-14T12:11:47Z (historical)

The user-authorized first-principles redesign is implemented and verified.
No new GPU job has started; CUDA is unavailable in the coding shell. Do not
resume any old teacher/endpoint/decoder/LR experiment. No model-switch pause
or new advisor question is required; the user switched and authorized this.

New model: original Z-Image Base 6B, native Gaussian-noise-to-GT flow, persistent
aligned LQ projection at EVERY model call, fresh FP32 LoRA/AdamW, original
frozen VAE and native fixed-prompt text encoder. This repairs identified
pretraining/conditioning mismatches but is NOT a proven quality improvement.
477 trainable tensors / 79,626,240 parameters. Do not inherit failed adapters.

Config: `configs/foundir_blur_only_zimage_conditional_teacher_v1.json`, SHA256
`c8bbf5d7dc85c2436f554a7e01e38b9d44d3f554686e001566c248d278da94db`.
Design and verified readiness:
`reports/zimage_conditional_teacher_design_2026-09-14.md` and
`reports/zimage_conditional_teacher_readiness_2026-09-14.md`.
28 distinct focused tests pass (27 prior-suite plus one added crop-cache test).
Real 6B CPU native forward/LoRA-gradient equality, repeated FP32 updates,
LQ sensitivity, checkpoint reload/reset, repeated two-step inference, VAE
decode and PSNR/LPIPS pass. These were 64px CPU checks, not CUDA quality gates.
Final full preflight PASSED status0: six model hashes, both H100/H200 profiles,
and all 2176 pilot image pairs. Log:
`logs/zimage_conditional_teacher_preflight_parallel_v1.log`. The old serial
preflight was intentionally interrupted; it is preserved and is NOT a pass.

Exact next action in the user's GPU shell, from this workspace:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_zimage_conditional_teacher_8gpu.sh
```

Monitor:
`tail -f logs/foundir_blur_only_zimage_conditional_teacher_v1_tiny.log`.
Tmux session: `zimage-conditional-tiny`, with verified nohup fallback.
Output: `checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_tiny/`.
One command runs all 1200 tiny steps and evaluations at 0/400/1200. Before
formal training, all eight CUDA ranks must pass native equality and repeated
synchronized updates. Expected 1200 training rows, 2 formal checkpoints,
6 aggregates, 96 triptychs/sidecars and terminal result. Inspect metrics AND
fixed images after completion, then update this handoff.

Tiny tests TRAIN learnability; its 16-development results are descriptive.
Only after prospective train metric + visual pass may the prepared pilot run:
2048 source groups for training, 128 for development, one crop/group,
2000 steps/global batch32. Tiny global batch8. Both hardware families use
microbatch1, accumulation1 (tiny) /4 (pilot). RGB crops are cached per rank;
paired D4 transforms occur BEFORE VAE encoding. Pilot manifest is pinned at
`19309079bd2a64e253527ff1b637fc3bd2a17522cde8c47fbd3a60283315e11f`.
All original 16+16 exact crops are retained and parent split membership stays
fixed. Pilot requires `reports/zimage_conditional_teacher_tiny_visual_review_v1.json`
bound to the tiny config/result/checkpoint and actual reviewed images; this
review does NOT exist yet. Never fabricate it or chain pilot automatically.
Pilot final development must improve both PSNR and LPIPS and pass visuals;
its reported train snapshots concern only 16 fixed training examples.

Full training/all-2595 development evaluation, successful teacher quality,
distillation and any conditional decoder tuning remain unfinished dependent
stages. Old failures retain their original labels and criteria. Qwen
degradation conditioning, mixed degradation, sealed internal, official FoundIR
test and development-GT optimization remain prohibited.

## First-principles redesign authorized — 2026-09-14T11:49:13Z (historical)

User switched model and explicitly said execute / first principles. The prior
model-switch pause is satisfied; do not ask for another switch or advisor round.
Design: `reports/zimage_conditional_teacher_design_2026-09-14.md`; config:
`configs/foundir_blur_only_zimage_conditional_teacher_v1.json`.
The new architecture restores native Gaussian-noise-to-GT flow and adds a
zero-initialized aligned LQ patch projection at every DiT call. Fresh FP32
LoRA/AdamW; original frozen VAE/text encoder; no inherited failed adapter.
The design is registered as `zimage_conditional_teacher_design-09530a53c2f3fd6e`
and its hold is resolved with the explicit user instruction, not a fabricated
advisor reply. Implementation is complete; final image preflight is still
running, so this paragraph alone is not GPU-readiness evidence.

Scientific correction: poor development from 16 training crops is not enough
to reject full-data generalization. Separate tiny training learnability from
a prospective 2048-source training / 128-source development pilot, with
fixed metrics and visual review at each boundary. All historical failures
remain failures under their original criteria. Do not rerun those jobs.
No GPU is exposed here. No new GPU training has started. Protected scope is
unchanged. 27 tests passed, including native equality and two-process gradient
accumulation. Real 6B CPU native forward/LoRA-gradient equality, two repeated
FP32 updates, active LQ dependence and checkpoint reload/reset all passed;
separate repeated two-step inference/VAE/PSNR/LPIPS also passed. Details:
`reports/zimage_conditional_teacher_readiness_2026-09-14.md`.
Pilot CSV is frozen at SHA256
`19309079bd2a64e253527ff1b637fc3bd2a17522cde8c47fbd3a60283315e11f`.
The initial serial CPU image scan was intentionally interrupted for bounded
eight-worker parallel decoding; no GPU job was affected. Preserve that partial
log. Next: verify `logs/zimage_conditional_teacher_preflight_parallel_v1.log` terminal pass,
commit/push exact scoped source, then provide the single tiny GPU launch.

## Round-2 v2 terminal scientific failure — 2026-09-14T11:21:43Z (historical)

V2 completed all 1200/1200 steps normally at 11:17:51Z, terminal status0,
8xH100. This is a SCIENTIFIC QUALITY FAILURE, not a runtime failure. Independent
audit passed 1200 ordered finite training rows, six exact FP32 checkpoints
(476 tensors each), 14 complete aggregates, 224 triptychs, 224 hash-matching
sidecars, eight exact rehearsals, config/result identities and false protected
flags. Result SHA:
`f2924d157f9fe9c2814122ec96f1e1cb5fa543016df0b61cdbfa84f303b8800f`.
Final checkpoint SHA:
`b90f5782e1ffa5db3c78ed53295472a5b3aac307d423ac03c067c3a00e23db92`.

Final train succeeds locally: input 24.7232/.3085 -> 29.2780/.1247, latent
MSE .186149 -> .094283. Development decisively fails: input 23.9045/.3264 ->
21.4293/.4635; PSNR gain -2.4753dB, LPIPS worsens .1371, zero of16 improves
PSNR, latent MSE rises .235011 -> .287624. The joint phase improves training
but not development. One finite gradient spike at joint step891 reaches
2543.121 preclip and is clipped; development had already worsened by step800,
so it cannot alone explain failure.

Human fixed-image review is complete and FAILS. Training statue 0042483 shows
memorized sharpening but retains artificial smear/web texture; near-clean
0091486 is unnecessarily degraded (36.883 ->31.757dB, .047 ->.085 LPIPS).
Development 0092666 gains heavy grain/ghosting; 0100870 text is distorted;
0055608 bus geometry has severe blocks/warping; 0060122 retains blur plus web
texture. These outputs are not credible restoration.

Do not select an earlier checkpoint: none passes development and step1200 was
predeclared. Do not launch full training, label this teacher/distillation,
tune decoder from this failed model, or start another LR/step sweep. The agreed
two-round decision budget is complete. The evidence supports a formulation/
generalization problem: 79M LoRA parameters memorize 16 crops but do not
generalize to source-disjoint crops.

Detailed audit:
`reports/zimage_latent_first_v2_terminal_audit_2026-09-14.md`, SHA
`3983d70be38dd6ca1e0c7e479f3b9c3fcb7828ad367ad7b25ce9f9b98b2adb76`.
Machine audit JSON SHA:
`5898bafc056174826c0da93180241082c0585f7b30e460a778db3dcb9180ea00`.
Governance report: `zimage_latent_first_v2_terminal_audit-2c4419999557a6e2`,
JSON/Markdown SHAs `9bf8ea827dbfa6babdc051c3fc5d2e1c779b07bbd6bd343ffa02b198288e903a` /
`d17d1d4a804d720d5e4ba064b69d0fbe2596784eaa1db00c846e90db5a9f3d1b`.
No new hold/advisor round is requested.

MANDATORY next action under the recorded model contract: stop GPU execution
and tell the user to switch to Astra medium before choosing a materially new
scientific formulation/data regime. Sol monitoring/implementation is no longer
the bottleneck. Astra must reason from all teacher/endpoint/decoder/oracle/
repro evidence and avoid another cosmetic objective sweep. Do not implement
or launch the redesign during this status turn. Qwen degradation conditioning,
mixed degradation, sealed internal, official FoundIR test and development-GT
optimization remain prohibited.

## Round-2 v2 GPU live — 2026-09-14T11:11:17Z (historical)

User launched v2 on homogeneous 8x H100 at 11:04:50Z, revision 3902381.
The shared log and telemetry prove it is live; do not restart or duplicate.
All eight CUDA rehearsals pass. Complete 16+16 strict baseline replay passes
all five checks: current repeats are exact, initial weights have the common
expected digest, historical maximum per-record drift is .036730dB/.001476
LPIPS, and maximum role-aggregate drift is .001960dB/.0001103 LPIPS. Thus v1's
missing rows reflect bounded historical/current execution drift, not a wrong
checkpoint/input. Baseline report is preserved in v2 output.

At 11:11:17Z training is live at 616/1200, latent-only phase, no error or
terminal result. Checkpoints/snapshots 200/400/600 are complete. Step600 train
improves versus input to 27.6990/.2821 and latent MSE .114695, but development
worsens to 21.5719/.4651 and latent MSE .262173. This shows substantial early
overfit during latent-only training. Do NOT stop/cherry-pick: the frozen design
requires finishing step800 then evaluating the joint RGB/LPIPS phase through
predeclared final step1200. This is intermediate evidence, not a final gate.
Eight GPUs show advancing telemetry around 26.3GiB each. No OOM/traceback.

Exact monitor: `tail -f logs/foundir_blur_only_zimage_latent_first_v2.log`.
Wait for step800/1000/1200, result/report and terminal marker. After completion,
audit hashes/counts/metrics and visually inspect fixed train/development images
before judging quality or taking another action. No protected data accessed.

## Round-2 v2 recovery verification — 2026-09-14T10:56:00Z (historical)

User said `execute`. The current v1 runtime hold is explicitly resolved as
`resume_same_mainline` by the existing advisor authorization plus this user
instruction; decision SHA-256
`5f5edd222ebb56bf86f5bc2656d2f6647930d6725153d0cb4a67f4a995acfb44`.
This records the user's instruction and does NOT fabricate a new advisor reply.

V2 preserves all v1 failure evidence and uses a new output root. Scientific
training recipe/schedule/data/model are unchanged. Before step1, it saves all
32 baseline rows and repeats the complete baseline twice. It requires exact
current repeat latent/RGB/image/metric equality and exact restored checkpoint
weights across ranks. It then fully reports historical endpoint deltas and
requires predeclared maxima: per-record .05dB/.005 LPIPS and role-aggregate
.02dB/.002 LPIPS. These limits were frozen before the two missing v1 rows were
observed and do not substitute for exact current replay. Any failure writes
complete evidence and stops before training.

Config: `configs/foundir_blur_only_zimage_latent_first_v2.json`, SHA-256
`6c6cfc24febcaae8d94b2ffbac188d6f09abff67421fd38a4b08d3723edc884d`.
Design: `reports/zimage_latent_first_v2_recovery_design_2026-09-14.md`.
Governance design report: `zimage_latent_first_v2_recovery_design-e775e8893c6f6c02`.
Sixty-one focused tests pass. Full six-weight/parent/failure-evidence hashes,
both H100/H200 inventory profiles, Python compilation and shell syntax pass.
Real 6B 64x64 CPU rehearsal passes both stages, exact repeats and initial reset;
the initial/gradient/update digests exactly match prior CPU evidence. Evidence:
`logs/zimage_latent_first_v2_tests.log`,
`logs/zimage_latent_first_v2_preflight.log`,
`logs/zimage_latent_first_v2_cpu_smoke.log`, and
`logs/zimage_latent_first_cpu_i1_du9il/rehearsal_rank_0.json` (SHA-256
`c315b5feb4e33d078d0bcc67f6ce254cff37733c0a721a47c87a93f0a0b5bda9`).
This CPU evidence does not establish CUDA baseline agreement or quality.
V2 recovery implementation and scoped governance evidence were committed and
pushed as `c7f6be0ef10bc8507fa6a1637b44d18dd5ec834c`; remote main was verified at
the same revision.

No v2 GPU job/log/output exists. After final commit/push, exact next GPU-shell
action is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_foundir_blur_only_zimage_latent_first_v2_8gpu.sh
```

Monitor: `tail -f logs/foundir_blur_only_zimage_latent_first_v2.log`.
Output root: `checkpoints/foundir_blur_only_zimage_latent_first_v2/`.
Tmux session: `foundir-zimage-latent-first-v2`, with nohup fallback if tmux is
unavailable. Do not run v1 again. No protected data or development training.

## Round-2 startup terminal failure — 2026-09-14T10:37:26Z (historical)

User asked `check`. The job launched at 10:13:31Z on homogeneous 8x H100,
revision 7e2e074, and exited at 10:15:24Z with status1. Formal training is
0/1200 steps: no training.jsonl, checkpoint or result.json exists. Do NOT
restart v1 or describe this as training divergence/quality failure.

All eight full-resolution CUDA two-stage rehearsals passed. Independently
verified every rehearsal artifact: exact same-rank repeats, one common initial
parameter digest, and one common updated-parameter AND post-clipping averaged-
gradient digest across all eight ranks for each of the two updates. Initial
FP32 weights were restored before evaluation. This supports the tested
distributed repeatability but does not establish agreement with historical
ordinary execution or restoration quality.

The failure is the inherited endpoint baseline check in evaluate(0), on
development ranks2 and7: `inherited endpoint reproduction mismatch`.
Thresholds are abs PSNR delta <=.02dB and abs LPIPS delta <=.002 versus the
immutable endpoint step400. All16 train records passed; aggregate step0 is
25.40651dB/.2551265. Only12 development records were saved, all within those
tolerances. No full development aggregate exists.

The failing rows/values were NOT saved because the assertion precedes image
and sidecar writing. From exact frozen manifest order and missing sidecars,
the first development item on rank2 is 0092666, and on rank7 is 0055120;
their second items 0021020 and0055608 were never reached. Exact failing
metric deltas and whether PSNR or LPIPS triggered are unavailable. Do not
invent their magnitude or declare the threshold too strict. A deterministic
versus historical ordinary-path numerical difference is a hypothesis only;
the current evidence does not isolate the source.

Runtime report: `zimage_latent_first_runtime-36be52b1efd75072`, JSON SHA
`b99e84b92eed68cac7498faca743fb6fd8c13365d47caffd152fcacd4f60e971`.
The launcher generated a new unresolved machine hold of that ID. Existing
user authority still says no additional advisor round; record its binding
when authorized recovery is executed, never fabricate a new advisor reply.
This check turn changed no training code, thresholds, jobs or authorization.
All prior output remains preserved; protected-data restrictions are unchanged.

Next repair task: persist complete baseline failure evidence before asserting,
then compare inherited endpoint inference under matched execution, exact
weights, prompts, encoded inputs and decoder settings. Determine the source
of baseline drift before deciding recovery; do not simply relax tolerances
or repeat the entire scientific gate. Reuse passed rehearsal evidence where
scientifically valid. The single round-2 quality recipe has NOT yet been tested.

## Round-2 latent-first CPU-ready — 2026-09-14T10:04:08Z (historical)

User said `execute`. Round 1 is complete; do not rerun it. The single bounded
round-2 recipe is now in `configs/foundir_blur_only_zimage_latent_first_v1.json`
and `reports/zimage_latent_first_design_2026-09-14.md`: exact endpoint step-400
FP32 LoRA initialization, original frozen VAE, same 16 train/16 source-disjoint
development 512x512 crops, LR 5e-5/global batch 8/clip 1. Steps 1--800 use GT
latent MSE only; 801--1200 use latent MSE 1 + RGB MSE 1 + LPIPS .05. No weak
teacher imitation and no decoder tuning. This is supervised endpoint repair,
NOT successful teacher distillation. All protected-data restrictions remain.

New trainer/helper/tests/launcher are implemented and CPU-verified. Sixty
focused tests passed (including two-process Gloo synchronization versus a
single-process global-batch reference for both objectives), compilation and
shell syntax passed, and the full six-model-artifact/parent/image hash
preflight passed with both H100/H200 profiles. Real 6B 64x64 CPU two-stage
rehearsal completed with exact repeats, nonzero FP32 updates, FP32 AdamW,
frozen-gradient exclusion and exact initial-state reset. These CPU tests do
NOT establish full-resolution CUDA memory, reproducibility or image quality.

Config SHA-256:
`cc4cb5e26803fa20a4ea92fc91ae2b6185619685b1e04e0a071e908aa89da904`.
Design/governance report: `zimage_latent_first_design-e0bd267718808756`.
It records existing user `execute`; no additional advisor response or hold.
Verification logs: `logs/zimage_latent_first_tests_v1.log` (60 passed),
`logs/zimage_latent_first_preflight_v1.log`,
`logs/zimage_latent_first_cpu_smoke_v1.log`.
CPU evidence: `logs/zimage_latent_first_cpu_9f9hciau/rehearsal_rank_0.json`, SHA
`58da013637862fb29272f68aa42fcb57922a684b686f270aca2129b29557d076`.
Rehearsal latent/joint losses 5.0682168/.2708406 come from DIFFERENT training
records; do not call that a learning curve. Preclip norms 10.3766603/.8073046.

One real launch will include deterministic padding/Flash controls and a
two-repeat, two-stage synchronized rehearsal before resetting all state for
actual training. Fixed-layout FP32 allreduce, NCCL Ring/Simple. Rehearsal hashes
POST-clipping averaged gradients, outputs and updated parameters; all ranks
must retain identical weights. Fixed snapshots 0/200/400/600/800/1000/1200;
candidate is predeclared final step1200, no checkpoint cherry-picking. Gate
requires lower training latent MSE and both PSNR/LPIPS improvements over input
on train AND development, followed by visual review; no automatic full launch.
If this recipe fails, end the agreed two-round decision budget and reconsider
formulation, not another open-ended sweep.

CUDA rechecked here: unavailable, zero devices. No round-2 GPU log/output
exists and no GPU work has started. Exact next action, on the user's shared
homogeneous 8xH100/H200 GPU shell in this workspace:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_foundir_blur_only_zimage_latent_first_v1_8gpu.sh
```

Monitor: `tail -f logs/foundir_blur_only_zimage_latent_first_v1.log`.
Output: `checkpoints/foundir_blur_only_zimage_latent_first_v1/`.
Retained tmux session: `foundir-zimage-latent-first-v1`; launcher resolves the
executable or uses verified nohup fallback, never requires /usr/bin/tmux.
It rejects prior evidence/concurrent launches and logs UTC start/end, status,
runtime environment, eight-GPU telemetry and governance terminal results.
One job runs all stages; no manual launch at step800. Eight rehearsal-pass
markers and inherited step0 per-record metric agreement precede training.
Expect 6 new FP32 checkpoints, 14 aggregates, 224 triptychs/sidecars,
1,200 training rows, result.json and report.md. Final checkpoint hashes are
roundtrip-verified; ranks must have identical weights at every saved snapshot.
Review both training and development plus fixed visuals before further work.
Monitoring remains routine; follow existing escalation contract only when a
new consequential scientific design is actually needed, not for model-tier
ceremony. No quality-valid overall teacher/student result exists yet.

## Round-1 reproducibility terminal pass — 2026-09-14T09:42:45Z (historical)

Supplement finished normally at `2026-09-14T09:38:26Z`, status 0. All 12 new
strict-deterministic fits completed, and all 12 immutable ordinary fits were
reused: combined 24/24. Root result SHA-256 independently verified:
`9d85b92bc4cfc25d7843379b17e7221875e6b87ee9303a85e976609ff11438d3`.
Root: `checkpoints/foundir_blur_only_zimage_repro_repair_v1/`.
Governance report: `zimage_repro_repair_v1-5c5e584b89148a99`.

ALL 8 strict same-GPU repeat/instrumentation comparisons pass EXACT first
gradient, first-update, final-weight and every snapshot-latent equality. Both
strict cross-device replicate comparisons also pass all these exact checks.
This establishes a reproducible execution route for the tested controls.
The repaired CUDA padding path passes its operator and full-model equivalence
checks. Ordinary execution was non-repeatable; this controlled execution
bundle resolves that measured issue, but does not isolate a unique responsible
kernel or prove the cause of every historical training collapse.

Final off_b matches off_a/on exactly: strong-blur RGB 26.6786/.4183,
near-clean mixed 34.7077/.0381. Quality still FAILS; near-clean PSNR is below
36.883 dB input, strong-blur perceptual fidelity remains poor, and near-clean
preclip gradient peaks at 1.342907. Determinism is not a cure for all optimization
instability and is not evidence of full-data generalization. No decoder tuning,
development-GT optimization or protected-test access occurred in this batch.

Exact next task: round 1 is complete; do NOT run more reproducibility/LR sweeps
or rerun this supplement. Review the fixed visuals and collected layer/loss
evidence, then freeze ONE evidence-supported round-2 correction/learnability
gate on the existing 16 train/16 source-disjoint development crops, using the
verified deterministic execution and padding repair. No full teacher training
or successful restoration model is established yet. Overall teacher/student
goal remains open; existing user authority permits continuation without a new
advisor round. This `finished?` turn only audited results and updated handoff.

## Padding repair GPU live evidence — 2026-09-14T09:37:09Z (historical)

Supplement launched 09:28:24Z on H100, revision 842d528. Workers 2/3/6/7
are alive per advancing shared logs/telemetry; GPUs 0/1/4/5 are deliberately
idle because ordinary controls are reused. No terminal marker or root result
yet. Do not restart the supplement or rerun completed ordinary controls.

All four workers passed actual CUDA padding-operator and full 512x512 model
forward/backward repair gates. Within each worker, original/patched/strict_a/
strict_b outputs and ALL trainable-gradient hashes match exactly. The original
indexing error is absent and strict determinism remains enabled. This now
establishes GPU compatibility for the tested path, not just CPU readiness.

Eight of 12 new fits completed: off_a and on at 200 steps on all four workers.
For EVERY off_a/on pair, first gradients, first updated weights, final weights
and all 14 snapshot rows are exactly identical. Both device replicates also
give identical reported final results. Final off_b repetition is running at
step >=100/200 on all four workers; wait for it and unified artifact validation
before declaring the entire same-GPU repeat gate complete.

| Objective / record | off_a = on PSNR / LPIPS | Maximum preclip gradient |
|---|---:|---:|
| Strong blur RGB / 0042483 (ranks 2,6) | 26.6786/.4183 | .174144 |
| Near-clean mixed / 0091486 (ranks 3,7) | 34.7077/.0381 | 1.342907 |

Reproducibility is improving, but image-quality failure is not repaired by
this evidence: near-clean PSNR remains below its 36.883 dB input, strong-blur
RGB LPIPS remains poor, and near-clean gradient clipping still activates.
These are two training-image fits, not generalization. Keep the runtime
repair conclusion separate from quality, stability and unique-kernel causality.

Exact next task: finish monitoring remaining four off_b phases, audit merged
12-reused + 12-new result/report and identities, then interpret the round-1
controls before designing the single agreed round-2 correction/learnability
gate. User asked only `check` this turn; no code or new job was launched.

## Deterministic padding repair CPU-ready — 2026-09-14T09:26:26Z (historical)

User explicitly said `execute` after the failed CUDA controls. Implemented a
bounded round-1 runtime repair, NOT a new experimental sweep or round-2 gate.
The two pinned Base-pipeline boolean-index padding assignments are replaced
process-locally with torch.where; strict determinism stays enabled. Upstream
source is hash-guarded and the ignored checkout is untouched. Only frozen pad
tokens and the exact existing tensor shapes/dtypes are supported.

Design: `reports/zimage_padding_repair_design_2026-09-14.md`.
Compatibility code: `lowlight_restoration/zimage_padding_compat.py`.
Supplement runner: `smoke/repair_foundir_blur_only_zimage_repro_batch.py`.
Config: `configs/foundir_blur_only_zimage_repro_repair_v1.json`, SHA-256
`806f60d6d8a92363c5c9a5a29039e21b9cb895c70e8a1e1804c5219bd674b840`.
Transformed function SHA-256:
`87849924eb0b6da807c6b4f4540d5c6ba113e7b2670665bf4d4ca02d3640cf3f`.
Design report: `zimage_padding_repair_design-8619468fbbaa03c7`; records existing
user authority, not a fabricated advisor response or new approval request.

Only failed original worker/rank IDs 2/3/6/7 are launched (12 new fits).
Ordinary workers 0/1/4/5 (12 fits) are reused after full parent config/result,
checkpoint, log and image checks; merged results retain their original config
identities and mark them reused. No copying, overwrite or full-batch rerun.
Same-GPU off/on/off sequence remains intact. Eight-GPU H100/H200 inventory
is required, but only four GPUs perform training in this supplement.

New safeguards before ANY optimizer step on each CUDA worker:

1. Exact padding output/input-gradient equivalence at width 3840, FP32/BF16,
   no/partial/all masks, ordinary original vs strict replacement.
2. Full actual 512x512 model/loss/backward: original, patched, patched-strict
   twice; require exact output and all trainable-gradient hashes. FlashAttention
   deterministic backward and deterministic cuDNN remain enabled throughout.
3. Only passing workers enter training. Aggregate additionally requires the
   saved CUDA operator/model gate artifacts and strict-deterministic flags;
   CPU evidence cannot substitute for CUDA validation.

CPU verification complete: 31 focused tests passed, including refusal to
accept supplemental workers without saved CUDA repair gates. Full artifact
and H100/H200 launcher preflight passed. BOTH real 6B 64x64 RGB and mixed-loss
equivalence gates passed: original/patched/strict_a/strict_b outputs and all
trainable-gradient hashes are exact within each record/objective. Mixed probe
loss is .0318890028. Both records then completed off_a/on/off_b two-update
CPU smokes with identical first-gradient, first-update and final-weight hashes
across their three phases. These are CPU interface/equivalence results only;
the mandatory full-resolution CUDA gates remain pending the GPU launch.
Logs: `logs/zimage_padding_repair_cpu_mixed_v1.log`,
`logs/zimage_padding_repair_cpu_rgb_v1.log`,
`logs/zimage_padding_repair_preflight_v1.log`.

No GPU is exposed here and no supplement GPU job has started. After completed
CPU checks, the exact single GPU-shell command is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_foundir_blur_only_zimage_repro_repair_v1_8gpu.sh
```

Master log: `logs/foundir_blur_only_zimage_repro_repair_v1.log`.
Output: `checkpoints/foundir_blur_only_zimage_repro_repair_v1/` with separate
worker_2/3/6/7.log. Retained tmux session: `foundir-zimage-repro-repair-v1`;
launcher uses verified nohup fallback if tmux is absent. Existing evidence and
concurrent launches fail closed. Never call this repair GPU-validated before
its runtime gates pass. Round-2/full training remain dependent on the combined
control/quality interpretation; all protected-data restrictions remain.

## Consolidated batch terminal partial failure — 2026-09-14T09:12:52Z (historical)

The batch ended at `2026-09-14T09:04:59Z`, status 2, with a complete partial-
failure aggregate and governance report `zimage_repro_batch_v1-ea2c28ad8b5e6f56`.
All ordinary workers 0/1/4/5 finished off_a/on/off_b: 12 successful 200-step
fits, 168 PNGs and 12 final checkpoints. Deterministic workers 2/3/6/7 failed
all 12 phases before training on the same text-padding CUDA indexing assertion
documented below. No worker remains to monitor in this batch. Do not rerun it.
Independently verified result SHA-256:
`97c3d855f447d3e24667401466cc1bff985efe1ad4304f954ac171e78474ca23`.
Root: `checkpoints/foundir_blur_only_zimage_repro_batch_v1/`.

Important new evidence: ordinary same-GPU off_a/off_b repeats ALL differ from
the first gradient onward, despite restored initial weights/seed and identical
evaluation schedule. Maximum common-snapshot PSNR differences by rank 0/1/4/5
are .202929/.701574/.175568/2.739633 dB. These are trajectory maxima, NOT
all final-step differences. Thus cross-device assignment or enabling the
optional tracer is not necessary for divergence. The evidence establishes
ordinary-path non-repeatability under these controls, not a unique responsible
kernel or proof that every historical collapse shares that cause.

Final ordinary strong-blur RGB outputs range 26.5660--26.8700 dB /
.4062--.4232 LPIPS; near-clean mixed outputs 33.8293--34.9161 dB /
.0305--.0438 LPIPS. Near-clean input remains better in PSNR (36.883 dB).
Ranks 1 on/off_b still show preclip gradient peaks 1.8540/1.2397; all strong-
blur gradients stay below .176. None of this is a quality/generalization pass.

Exact next task: repair/validate the deterministic text-padding assignment
compatibility, retain strict deterministic execution, then supplement missing
controls while preserving/reusing the 12 successful ordinary fits. Verify
replacement forward AND backward equivalence and actual CUDA behavior before
calling that repair successful. Do not switch off determinism to manufacture
a pass, repeat completed ordinary fits without need, or start round-2/full
training from an incomplete control matrix. This remains round-1 runtime
repair under existing user authorization, not a new scientific sweep.
This turn only checked logs and updated handoff; no code or GPU job changed.

## Consolidated batch live partial failure — 2026-09-14T09:01:35Z (historical)

User launched the batch at 08:56:58Z on 8x H100, revision d85896a, and asked
`check the log`. Master log has no terminal marker yet. DO NOT restart or
interrupt the surviving workers; this coding shell cannot infer remote
liveness from local PIDs. Advancing worker logs and GPU telemetry confirm
ordinary workers 0/1/4/5 remain active.

All four strict-deterministic workers (2/3/6/7) exited status 2. Their off_a,
on and off_b phases ALL fail at evaluate(0), before any optimizer step, with
the same PyTorch CUDA Indexing.cu:699 internal assertion:
`number of flattened indices did not match number of elements in the value
tensor: 115200 vs 14745600`. Exact failing source is the broadcasted masked
text-padding assignment in
`third_party/diffsynth-studio/diffsynth/pipelines/z_image.py:633`:
`cap_feats[torch.cat(patch_metadata.get("cap_pad_mask"))] = dit.cap_pad_token.to(...)`.
This is a deterministic-execution runtime incompatibility, NOT an observed
training collapse or proof that FlashAttention backward is at fault. Both
real-model CPU smokes passed but did not exercise this CUDA-only indexing path.
The 12 failure artifacts/tracebacks are retained under rank_2/3/6/7. Do not
silently disable deterministic mode or relabel these failed controls.

Ordinary workers 0/1/4/5 completed off_a (200 steps each) and were running
instrumented on at step >=20 at this snapshot. Their first-phase metrics:

| Rank / record | PSNR / LPIPS | Maximum preclip gradient |
|---|---:|---:|
| 0 / strong blur RGB | 26.8700/.4110 | .173525 |
| 4 / strong blur RGB replicate | 26.7858/.4147 | .174331 |
| 1 / near-clean mixed | 34.9161/.0325 | .356129 |
| 5 / near-clean mixed replicate | 34.0046/.0305 | .097114 |

Thus current count is 4 completed fits, 12 failed before training, 4 running
and 4 off_b phases pending. No complete root result/report exists yet; the
batch cannot pass the full deterministic-control check even if all ordinary
phases finish. Ordinary cross-device replicates already differ, but same-GPU
off_a/off_b comparison is still pending. Do not infer a unique numerical cause
or generalization from these intermediate two-image results.

Next action: allow all ordinary phases to finish and audit their unified
partial-failure report. Then isolate the exact masked-assignment CUDA behavior
and verify a mathematically equivalent deterministic implementation before
repairing missing controls. This is a runtime repair within round 1, not a
new hyperparameter sweep or the round-2 quality gate. No training code was
changed and no new GPU job was launched during this log-check turn. Existing
user continuation authority and all protected-data restrictions remain.

## Consolidated reproducibility batch CPU-ready — 2026-09-14T08:13:33Z (historical)

The user challenged the inefficient serial diagnostic cadence, accepted a
consolidated diagnostic plus one corrected learnability gate, then explicitly
said: "sure, aiminng for the best quality, don't afraid of extra work, execute
now". This is a TWO-ROUND DECISION BUDGET, not a promise to finish restoration
in two rounds. Do not ask the advisor again. No open-ended extra LR sweeps.

Round 1 is implemented in
`smoke/audit_foundir_blur_only_zimage_repro_batch.py`, with pure control and
comparison helpers in `lowlight_restoration/zimage_repro_batch.py`.
Design/interpretation: `reports/zimage_repro_batch_design_2026-09-14.md`.
Config: `configs/foundir_blur_only_zimage_repro_batch_v1.json`, SHA-256
`9aef9ebef30487e8657a6df506e0914ef55cd77765c37e39ff420211a95c7046`.
Informational authorization/design report:
`zimage_repro_batch_design-53bd9122ccbf5eed`. No model-family/architecture
change or newly invented advisor response is involved.

Eight independent workers cover two prior failed record/loss combinations x
ordinary/deterministic execution x two device replicates. EACH worker runs
off_a/on/off_b sequentially on the SAME GPU; exact initial FP32 LoRA, fresh
FP32 AdamW and RNG seed are restored every phase. No DDP gradient averaging.
Every phase uses LR 5e-5, clip 1, batch 1, 200 steps, identical 14-snapshot
schedule and original frozen BF16 base/VAE. Thus one user launch runs 24 fits,
4,800 independent steps, 336 triptychs and 24 final checkpoints. Instrumented
phases record 49 module traces each. All phases record exact initial weights,
first gradient, first-updated weights, final weights and snapshot latent hashes.

Deterministic mode enables explicit FlashAttention-2 deterministic backward,
strict PyTorch deterministic algorithms and deterministic cuDNN. Both modes
disable TF32/cuDNN benchmarking and use CUBLAS_WORKSPACE_CONFIG=:4096:8.
This compares execution BUNDLES, not FlashAttention alone. Record actual
Flash/SDPA calls, GPU UUID, package versions and source hashes. Unsupported
deterministic operators produce preserved failed controls, not silent fallback.
Common hash/evaluation instrumentation changes timing versus historical runs;
these are matched NEW references, not guaranteed exact historical replays.

Each worker has a separate persistent log; independent failures are retained
while other workers finish. The supervisor validates artifact hashes, images,
snapshot sidecars, optimizer-step counts and traces before comparisons. It
emits both result.json and report.md even for missing/failed workers. Exact
same-GPU repeat/instrumentation controls and metric-tolerance comparisons are
reported separately. There is NEVER an automatic quality/full-training pass.

Verification: 21 focused tests pass, including the complete 24-fit aggregator,
missing-worker handling, artifact tampering, control assignments and exactness
versus metric-tolerance logic. Real 6B CPU 64x64 RGB and mixed-loss off/on/off controls both
completed all three two-update runs. Within each record/objective, initial,
first-gradient, first-update and final hashes are identical across all phases;
all snapshot latent hashes also match. Mixed CPU control enables strict
PyTorch deterministic algorithms. Full model/parent hashes and both H100/H200
profiles passed the launcher check. FlashAttention's installed function
signature supports explicit deterministic backward; GPU execution remains
unverified until the real batch. A complete-batch aggregation test is also
included alongside missing-worker and artifact-tampering rejection tests.
Logs: `logs/zimage_repro_batch_cpu_smoke_v1.log`,
`logs/zimage_repro_batch_cpu_mixed_smoke_v1.log`,
`logs/zimage_repro_batch_preflight_v1.log`.

No GPU is exposed in this coding shell; no GPU batch has started. Once CPU
readiness is verified, exact single GPU-shell launch is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_foundir_blur_only_zimage_repro_batch_v1_8gpu.sh
```

Master log: `logs/foundir_blur_only_zimage_repro_batch_v1.log`.
Output and per-worker logs:
`checkpoints/foundir_blur_only_zimage_repro_batch_v1/worker_0.log` through
`worker_7.log`. Launcher retains tmux session `foundir-zimage-repro-v1` when
available; otherwise verified nohup/log fallback. It rejects prior evidence
and concurrent launches. Do not rerun completed stability or optimization v1.

Round 2 is NOT implemented/launched: after round-1 controls and visual/quality
review, freeze one evidence-supported correction and its 16-train/16-dev gate.
If reproducibility remains unresolved, isolate the recorded first divergence;
do not select a LR or pretend full-data training is justified. If round 2 still
cannot learn reliably, revisit the training formulation rather than indefinitely
extending tiny probes. Full teacher training/evaluation, quality-valid student
distillation and conditional decoder work remain unfinished dependent stages.
All protected-test/degradation restrictions and model-escalation rules remain.

## LoRA stability terminal audit — 2026-09-14T07:50:02Z (historical)

User asked `check the log`. The already-launched stability audit completed
normally at `2026-09-14T07:41:42Z`, status 0, homogeneous 8x H100, 200 steps
per independent optimizer. No new experiment was launched during this audit.
Verified result SHA-256:
`defa6076114ba0b21b545af83c6bf66980be21779dfdbe621e63ac51b510d4b3`.
Result: `checkpoints/foundir_blur_only_zimage_lora_stability_v1/result.json`.
Log: `logs/foundir_blur_only_zimage_lora_stability_v1.log`.
Governance report: `zimage_lora_stability_v1-1331d9a8148a5aca`.

Audit verified 8 unique record/objective/LR combinations, 1,600 training log
rows, 112 finite metric snapshots and 112 PNGs/sidecars, 16 checkpoint hashes
(8 final plus 8 reference snapshots), and all 8 optimizer-trace hashes.
Each trace has 49 finite rows. Protected-test and development-GT-optimization
flags are false; original decoder stays frozen. These are two training-image
fits, not development evaluation or a deployable restoration success.

| LR | Strong blur RGB PSNR/LPIPS | Near-clean mixed PSNR/LPIPS |
|---:|---:|---:|
| 5e-5 | 26.8247/.4114 | 33.3981/.0393 |
| 2e-5 | 25.1227/.4819 | 33.8761/.0316 |
| 1e-5 | 24.1468/.5082 | 34.3268/.0377 |
| 5e-6 | 23.3246/.5233 | 34.3602/.0500 |

CRITICAL: both reference-reproduction checks are false. The previous 5e-5
strong-blur collapse (13.8618 dB at 200) did NOT recur (now 26.8247 dB).
Near-clean mixed ends at 33.3981 rather than 29.2787. Do not claim lower LR
fixed the original failure: reference LR itself changed trajectory and gives
the best strong-blur PSNR in this sweep. All near-clean candidates still lose
PSNR against their 36.883 dB input; strong-blur RGB LPIPS remains poor.

The old/new step-0 PSNR/LPIPS match exactly, but actual first updates already
differ: strong RGB update L2 .3722290871 vs .3722108480; near-clean mixed
.3375445214 vs .3375542356. New-minus-old PSNR at steps 1/20/100/200:
strong RGB -.000126/+.001325/+.032361/+12.962954 dB; near-clean mixed
+.008828/+.000977/+.323802/+4.119382 dB. The later extra evaluation schedule
cannot alone explain this first-update difference. GPU backward numerical
nondeterminism, instrumentation/allocation/timing effects and changed rank/device
assignment are hypotheses, not established causes. No deterministic GPU control
was enforced; CPU hook equality does not establish GPU trajectory equality.

Strong RGB reference gradient norm never exceeds .172463 this run; near-clean
mixed still spikes to 2.406561 at step 160 (clip 1 active), despite recovering
to .007469 at step 200. Therefore do not call all trajectories fully stable.
Across recorded normalization inputs, denominator minimum is .110420, while
noise_refiner.0.ffn_norm2 input absmax reaches 1,073,741,824 for near-clean
reference at 200. Large pre-normalization activations deserve comparison to
initial/base traces but do not by themselves prove overflow or a causal bug.

Exact next task: isolate reproducibility before LR-based design/scaling:
compare identical initialization and first-step gradients/updates with and
without instrumentation on the same GPU/backend, then control deterministic
backward and repeat the reference trajectory with matched evaluation schedule.
This is a proposed next diagnostic, not an implemented or launched job. Preserve
all completed artifacts; do not rerun v1 or extend the failed decoder blindly.
Existing user authorization allows continuation without another advisor round;
the overall teacher/student restoration objective remains incomplete. Follow
the existing task/model escalation contract for consequential next design.

## LoRA stability audit CPU-ready — 2026-09-14 (historical)

The user said `next` after the eight optimization probes identified late LoRA
instability. The next diagnostic freezes each previously failing record/loss
combination and changes only LR: [5e-5,2e-5,1e-5,5e-6]. For every rate, one
rank fits strong-blur 0042483 with RGB loss and one rank fits near-clean 0091486
with the existing mixed loss. This gives eight independent optimizers. Exact
FP32 endpoint initialization, frozen BF16 base/VAE, AdamW settings, clipping 1,
seed, single-image batch and 200 steps are unchanged within each comparison.

Design: `reports/zimage_lora_stability_design_2026-09-14.md`.
Config: `configs/foundir_blur_only_zimage_lora_stability_v1.json`, SHA-256
`73fa6baf6a081103bbd23268480e39618238acfc96a9e835777c8af0955a52e2`.
Implementation extends `smoke/audit_foundir_blur_only_zimage_optimization.py`
with explicit `--stability`; instrumentation is in
`lowlight_restoration/zimage_optimizer_trace.py`. Original diagnostic artifacts
remain immutable; the old config still selects its original four-arm behavior.

Every step logs loss and pre-clipping gradient norm. At step 1, multiples of 20,
and every step >=160 it records per-module gradient norms, normalization input
denominator ranges/absmax, and actual LoRA update/parameter norms. Hooks capture
original forward only, not backward checkpoint recomputation. Fixed snapshots
are 0/1/20/40/60/80/100/120/140/160/180/190/195/200 (112 triptychs total).
Reference-rate ranks save 180/190/195/200 checkpoints; all ranks save final
weights. Trace and checkpoint hashes enter each result. Lower-rate fits may
progress more slowly in 200 steps; stability alone is not a quality win.
The result compares reference-rate PSNR/LPIPS at common snapshots to the prior
immutable audit (tolerances .05 dB/.005), reporting `reference_reproduction`.
If false, investigate trajectory differences before causal LR conclusions.

Six focused tests passed, including exact forward/gradient equality with hooks
enabled. Real 6B CPU 64x64 two-step instrumented smoke passed:
`logs/zimage_lora_stability_cpu_smoke_v1.log`. It captures 235 normalization
layers and 238 gradient modules. First-step relative update L2 is .00558038;
77,364,194 parameters change, and loss/gradient/output match the prior
uninstrumented CPU smoke exactly. This is not full-resolution GPU evidence.
Final launcher `--check` passed against the config hash above, including six
focused tests, full artifact hashes, and both H100/H200 inventory profiles.

Design hold `zimage_lora_stability_design-013d68aa7d35a2a6` is resolved by the
existing user authorization plus `next`; no new advisor round. This shell has
no exposed GPU and no stability GPU job has started. Exact GPU command is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_foundir_blur_only_zimage_lora_stability_v1_8gpu.sh
```

Log: `logs/foundir_blur_only_zimage_lora_stability_v1.log`.
Output: `checkpoints/foundir_blur_only_zimage_lora_stability_v1/`.
Inspect existing evidence before launching. Monitor with Sol medium; interpret
stability and layer traces with Astra high. No decoder changes or protected
test access; do not claim a successful deployable restoration model from fits
to these two training images.

## Optimization audit terminal evidence — 2026-09-14T07:17:44Z (historical)

All eight independent 200-step probes completed on homogeneous 8x H100, status
0. This is diagnostic completion, not restoration success. The audit verified
8 unique arm/record results, all 8 saved checkpoint hashes, 40 finite snapshots,
40 triptychs, nonzero actual first-step updates, and false protected-test and
development-GT-optimization flags. Immutable result SHA-256:
`7d9eac4a76675190fb1202e30e937fb3762167c81e1a8fb4d03fa6b91eb96ddf`.
Report ID: `zimage_optimization_audit_v1-124563377f0b169d`.

| Arm | Strong-blur 0042483 final PSNR/LPIPS | Near-clean 0091486 final PSNR/LPIPS |
|---|---:|---:|
| frozen endpoint start | 21.3415/.4797 | 32.6514/.0981 |
| latent RGB oracle | 38.2553/.0240 | 45.5702/.0064 |
| latent mixed oracle | 37.2272/.0038 | 43.6378/.0013 |
| LoRA RGB | 13.8618/.6822 | 36.7289/.0500 |
| LoRA mixed | 25.4717/.1975 | 29.2787/.0762 |

Direct latent optimization from the SAME initial predictions converges well
through the original frozen decoder for both objectives. Visual review of the
strong-blur latent-mixed oracle is close to GT. This supports decoder/latent
optimization feasibility for these examples, not a deployable restoration
method: the oracle optimizes against each training GT. A free latent can even
beat encoder-produced GT roundtrip PSNR; the roundtrip is not a strict bound on
all possible optimized latents.

LoRA behavior reveals important instability. Strong-blur LoRA RGB loss decreases
to .0094127 at step 180 but reaches .1507502 at step 200, with pre-clipping
gradient norm 134,864.390625; the post-update step-200 image collapses to
13.8618 dB with severe artifacts. Gradient clipping at norm 1 was active.
Near-clean LoRA mixed reaches 34.2864/.0349 at step 100, then degrades to
29.2787/.0762 at step 200 (pre-clipping gradient norm 1.88874). Strong-blur
LoRA mixed continues learning but remains substantially worse than the oracle.
No NaN/OOM/NCCL occurred; these are finite optimization/scientific failures.
The sparse log cannot establish the exact first unstable step between 180 and
200 for strong-blur LoRA RGB.

Initial 512x512 weighted LPIPS gradient is 2.38x RGB for strong blur and 6.79x
for near clean; pixel/LPIPS cosines are .0032 and .1664. Thus LPIPS gradients
are large but not strongly opposed at initialization. Both latent objectives
succeed, while one LoRA RGB and one LoRA mixed arm deteriorate. The hypothesis
that simply deleting LPIPS fixes the system is NOT supported.

Exact next task: investigate DiT/LoRA optimization stability at LR 5e-5,
including module-wise activation/gradient behavior and actual update sizes,
with a lower-step-size controlled comparison and dense pre-failure checkpoints
before scaling data or training duration. Distinguish optimizer instability,
BF16 activation/normalization sensitivity, and inherited adapter conditioning;
the current evidence does not isolate a unique cause. Preserve all eight probes;
do not rerun or adopt the fitted oracle as a model. Keep VAE frozen and retain
both metrics. User already authorized continuation without another advisor
round; scientific interpretation/design merits Astra high, execution Sol high,
monitoring Sol medium. Overall teacher/student restoration goal remains open.

## Optimization root-cause audit CPU-ready — 2026-09-14T07:08:32Z (historical)

After the decoder failure, the user requested deeper reasoning, received an
Astra-high recommendation, and said `execute`. The current task is a controlled
optimization audit, not another decoder continuation. Findings and frozen
interpretation rules are in `reports/zimage_optimization_audit_design_2026-09-14.md`.

Code audit verified paired crop coordinates, derivative/integration signs,
endpoint train/deploy mapping, native VAE scaling, LPIPS normalization, and
nonreentrant checkpointing. BF16 teacher update quantization remains a proven
hazard but not the complete established cause. Base model dispatch to the local
function named `model_fn_z_image_turbo` is structural (`siglip_embedder is None`),
not evidence of Turbo weights. Poor tiny-development metrics do not independently
prove failed learnability; train artifacts still justify an optimization probe.

The next frozen experiment compares four arms on the SAME two existing training
records (0042483 strong blur, 0091486 near clean): `latent_rgb`, `latent_mixed`,
`lora_rgb`, `lora_mixed`. Eight ranks each own one independent arm/record; there
is no DDP gradient averaging. Every arm initializes from exact FP32 endpoint
step 400 with the original frozen VAE. Latent LR .01, LoRA LR 5e-5, AdamW FP32,
200 steps, snapshots 0/1/20/100/200, one image per optimizer. The two records
have rejected teacher targets, so mixed loss is exactly RGB + .1 latent MSE +
.05 LPIPS. Teacher-target rejection and all parent hashes are validated.

It records initial weighted latent-gradient norms/cosines, actual first-step
parameter updates, quantized PSNR/LPIPS, raw RGB MSE, latent MSE, clipping,
GT roundtrip ceiling, fixed triptychs, and hashed final trainable tensors.
Direct latent optimization is a training-GT oracle, NEVER a deployable result.
Single-image LoRA learning also does not establish generalization. This probe
distinguishes loss conflict from latent optimization versus DiT parameter
optimization; two examples and fixed learning rates cannot prove a unique cause.

Config: `configs/foundir_blur_only_zimage_optimization_audit_v1.json`, SHA-256
`a28ca0c20214f649a0ef2e22f7772fceb737154e80050e1c63214823e2d81c31`.
Implementation: `smoke/audit_foundir_blur_only_zimage_optimization.py`.
Three focused tests, exact model-artifact hashes, and H100/H200 preflights pass.
Two real 6B CPU 64x64 two-update smokes passed (LoRA RGB and latent mixed):
`logs/zimage_optimization_audit_cpu_lora_rgb_v1.log` and
`logs/zimage_optimization_audit_cpu_latent_mixed_v1.log`.
LoRA first update changes 77,364,194/79,380,480 parameters (L2 .416281);
free latent changes 1024/1024 values. Decoder remains frozen and FP32 optimizer
state/finite gradients are verified. On this reduced crop, initial weighted
gradient norms are RGB .005552, latent .002975, LPIPS .015705; pixel/LPIPS
cosine is -.01389 (approximately orthogonal, not strong antagonism). Pure RGB
LoRA step 1 increases crop PSNR 23.4144 -> 25.0655. This is a local smoke clue,
not a 512x512 convergence result or proof of the dominant failure cause.
Design hold `zimage_optimization_audit_design-32f359c3991e638b` is resolved by
the user's existing authorization plus `execute`; no new advisor round.

Exact next action on the GPU shell (this agent has no exposed GPU):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_foundir_blur_only_zimage_optimization_audit_v1_8gpu.sh
```

Log: `logs/foundir_blur_only_zimage_optimization_audit_v1.log`.
Output: `checkpoints/foundir_blur_only_zimage_optimization_audit_v1/`.
Before launch inspect existing evidence; never duplicate. Sol medium can monitor
the frozen probes; interpretation and the next design warrant Astra high under
the user's latest model preference. Do not infer that a model tier guarantees
metric gains. Protected-data/degradation restrictions remain unchanged.

## Decoder-tail terminal failure — 2026-09-14T06:55:09Z (historical)

The user launched the decoder gate; it completed 400/400 steps normally on
8x H100 with status 0, FlashAttention 2, global batch 8 and 45.0741 seconds
for optimization plus in-loop snapshots. Observed training GPU utilization
was 95--97%, memory 25,916 MiB/GPU. No runtime failure occurred. The frozen
student baseline exactly reproduced the previous endpoint result.

| Split | Input PSNR/LPIPS | Frozen student | Decoder step 200 | Decoder step 400 |
|---|---:|---:|---:|---:|
| train | 24.7232/.3085 | 25.4045/.2550 | 25.6140/.2136 | 25.9623/.1930 |
| development | 23.9045/.3263 | 22.8776/.3520 | 22.8318/.3170 | 22.6640/.3094 |

Automatic gate is false. The decoder improves perceptual metrics but loses
development PSNR, and training-GT reconstruction anchoring at weight .1 does
not preserve the original decoder. Development GT roundtrip degrades from
40.7184/.0076 to 37.6282/.0323 at step 200 and 34.8313/.0487 at step 400;
both trained snapshots violate the .02 LPIPS retention threshold. Step 400
also violates the 35 dB retention threshold. Do not adopt either decoder as
a passing candidate or launch full training from this result.

Review of two fixed development triptychs (0055608 bus and 0092666 leaves)
shows reduced fine grain but remaining smeared/artificial texture, ringing,
and insufficient recovery of blurred geometry/detail. This intervention
establishes that the decoder can alter perceptual artifacts, while this
bounded tail/anchor recipe does not solve restoration fidelity. It does not
prove that a different decoder intervention could never work. The small
16-training-crop experiment also cannot establish full-data generalization.

Audit verified 8 snapshots of 16 unique records, 128 PNG triptychs, independently
recomputed output PSNR/LPIPS aggregates, terminal marker, checkpoint/result
hashes, and false sealed/official-test flags. Immutable identities:

- result `checkpoints/foundir_blur_only_zimage_decoder_tail_v1/result.json`:
  `cf6d7f0e6db7b72432fed03b76e1446f6d9bbbd771604a8b80a2cd65682ede46`;
- step 200: `04fb194b6a4af94f9017c5841e2611dba245e9aa0988fa53bf754e773f5bf673`;
- step 400: `b755b0834b802c85f8f2c6bb9719158c68ce1be0107a5dc8cda280cd26574ce2`;
- report JSON: `ab3c3e0329e710be3941c8d5cf7b0cf8e714f1a77b128cdf33af9a7613afccc5`;
- report Markdown: `f11b50a309ccbe19222c7084fdfa4c9c5663cb4ec29c38edededc1849303caf8`.

Report ID: `zimage_decoder_tail_v1-779bf2f253c8387e`. No new advisor round is
required under the user's existing authorization. The current agent already
is Astra; do not create a redundant model-switch pause. Exact next scientific
task is to address the upstream restoration mapping and the inadequacy of
learning from 16 fixed crops before committing substantial GPU training.
Keep the original frozen VAE as the reference; do not blindly extend this
failed decoder run. Freeze a concrete fidelity-focused next design and its
learnability/development checks before launch. Teacher/full-training quality
remains unresolved; the overall restoration objective is not complete.

## Decoder-tail CPU-ready — 2026-09-14T06:47:54Z (historical)

The user switched to Astra and said `execute`; the previous model-switch pause
is resolved. The next authorized experiment is documented in
`reports/zimage_decoder_tail_design_2026-09-14.md`: freeze endpoint step 400,
train only decoder blocks 15--17 and the output norm/convolution, with FP32
trainable parameters/AdamW, RGB MSE + .05 LPIPS and .1 training-GT latent
reconstruction anchoring. Same fixed 16 train/16 development crops, 400 steps,
global batch 8, LR 1e-5, snapshots at frozen baseline and 0/200/400.

The source is `smoke/train_foundir_blur_only_zimage_decoder_tail_ddp.py`;
config is `configs/foundir_blur_only_zimage_decoder_tail_v1.json`, SHA-256
`d6f29a5ee41d60a02651a1e8a775551e5b913ebdd0f943fadaf033876df01984`.
This tests off-manifold decoder sensitivity; existing evidence does not prove
that decoder tuning will solve the learned restoration mapping. A 16-pair
experiment cannot establish full-data generalization.

The loader explicitly restores exact FP32 student LoRA after the inherited
constructor's BF16 conversion. Frozen baseline must reproduce the previous
endpoint step-400 per-record metrics within .02 dB/.002 LPIPS before training.
Separate frozen and promoted-tail step-0 images expose precision effects.
The new automatic gate requires train gains over input, development gains over
both input and step 0, and development GT roundtrip >=35 dB and <=.02 LPIPS.
Visual review remains required. The deployed bundle needs both the immutable
endpoint LoRA and the new decoder-tail weights, with pinned base weights.

Design hold `zimage_decoder_tail_design-aca9906895c90c6d` is resolved by the
existing explicit conditional-decoder authorization and the user's `execute`.
No new advisor response was requested or invented. Three focused tests passed;
full model hash and both H100/H200 inventory preflights passed. Real 6B CPU
64x64 forward/backward/AdamW smoke passed: loss .0394539163, gradient norm
.507023752, 1,071,363 trainable decoder-tail parameters. All trainable gradients
are finite, FP32 AdamW updates weights, and DiT/decoder-prefix parameters have
no gradients. Log: `logs/zimage_decoder_tail_cpu_smoke_v1.log`. These reduced
CPU checks establish interface correctness, not 512x512 GPU quality or memory
results. The final launcher `--check` passed against the config hash above.
This shell has no exposed GPU; no decoder GPU job has been launched.

Exact next action on the GPU shell is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_foundir_blur_only_zimage_decoder_tail_v1_8gpu.sh
```

Log: `logs/foundir_blur_only_zimage_decoder_tail_v1.log`.
Output: `checkpoints/foundir_blur_only_zimage_decoder_tail_v1/`.
Inspect existing evidence before any launch; never duplicate a job. Sol medium
can monitor this frozen experiment; Sol high can fix implementation issues.
Use Astra medium again if the result demands a new scientific design. All
protected-data/degradation restrictions remain in force.

## Model-selection and escalation contract — 2026-09-14T06:35:15Z (LATEST AUTHORITY)

The user reports that the active agent is currently **GPT-5.6 Sol medium**.
That is the appropriate model for monitoring this already-designed endpoint
run, auditing logs/artifacts, maintaining this handoff, and handling routine,
localized implementation or runtime fixes. Do not interrupt healthy execution
merely to change models.

Use the following task-to-model policy for future sessions:

- **Sol medium**: live monitoring, telemetry/log parsing, artifact counts and
  hashes, status reporting, documentation, rerunning established tests, and
  small obvious fixes with no scientific-design consequence.
- **Sol high**: substantial but well-scoped coding, DDP/runtime debugging,
  test construction, launcher hardening, performance profiling, and
  optimization whose intended algorithm and acceptance criteria are already
  fixed. This should be the default for most implementation work in this
  repository.
- **Astra medium**: pivotal scientific interpretation or design, including
  explaining unexpected metric/visual disagreement, deciding the next
  distillation objective or architecture after a failed gate, deciding whether
  and how to fine-tune the decoder, or making a choice likely to materially
  affect restoration quality, code architecture, training stability, GPU
  efficiency, or optimization performance.
- **Astra high**: reserve for a genuinely high-stakes redesign with several
  interacting unresolved causes where Astra medium has not produced a
  defensible design. It is not warranted for ordinary monitoring or routine
  coding. Astra low is not preferred here: use Sol high for execution work or
  Astra medium for consequential scientific reasoning.

Mandatory communication rule: before beginning a task where using the current
model instead of the recommended model is reasonably likely to materially
reduce code quality or optimization/scientific performance, explicitly tell
the user to switch, name the exact recommended model/reasoning level, and state
the concrete decision that requires it. Keep this handoff current before the
switch so Sol and Astra can be alternated safely. Model switching does not
authorize a new experiment, change scope, or relax protected-data rules.

Official OpenAI documentation was queried for this update but returned
`Forbidden` in the current environment. The quality-first Astra role comes
from the bundled OpenAI-docs fallback and may be outdated; the Sol-specific
division above additionally uses the model descriptions exposed in this Codex
session and project experience, not a claim about current public pricing or
availability.

## Endpoint student terminal result — 2026-09-14T06:35:28Z (LATEST AUTHORITY)

The endpoint GPU job completed normally with terminal status 0 on homogeneous
8x `NVIDIA H100 80GB HBM3`. It ran all 400 optimizer steps in 179.4340 seconds
with global batch 8, FP32 trainable LoRA/AdamW state, and frozen BF16 base
weights. There was no OOM, NCCL, traceback, non-finite, or protected-data
failure. `ENDPOINT_TARGETS_READY teacher_usable=3/16` exactly matches the prior
sweep; this is correctly classified as `gt_anchored_filtered_distillation`.

| Split | Input PSNR / LPIPS | Step-200 output | Step-400 output |
|---|---:|---:|---:|
| tiny-overfit train | 24.7232 / 0.3085 | 25.3647 / 0.2542 | 25.4045 / 0.2550 |
| visual development | 23.9045 / 0.3263 | 23.0691 / 0.3584 | 22.8776 / 0.3520 |

The machine `automatic_metric_gate_pass: true` means only that the predeclared
bounded train learnability gate passed: at step 400 train gains are +0.6814 dB
PSNR and +0.0535 LPIPS improvement. It is not evidence of development
generalization. Development remains worse than input by 1.0270 dB PSNR and
0.0257 LPIPS, only 3/16 records improve PSNR, and step-400 development PSNR is
0.1915 dB worse than step 200. Do not scale this endpoint student or call it a
quality-successful distilled model.

Fixed-triptych review agrees with the metrics. The endpoint mapping can recover
coarse sharp structure on some hard training examples, but retains prominent
web/grain texture and imperfect detail. Near-clean training inputs are
unnecessarily altered, and development examples show artificial sharpening,
grain/texture changes, color or contrast shifts, and inadequate recovery of
strong motion blur. The output remains materially below GT quality.

Immutable terminal identities:

- config SHA-256:
  `1d98854ec7dc40e3d678879d849955180d652cbfcd86ec77596912214c956f50`;
- step-400 checkpoint SHA-256:
  `d979af50f3013cc5e02d4c36943f32c4b512a75ee26c0aab20a37728b9c15d7e`;
- result SHA-256:
  `b5934dc34bd4ac67ba030de67504af3fc0e517f66e650dcfc4e6ad9677adc680`;
- governance report JSON SHA-256:
  `f083f4da3f61061291dcaf3499ca4e16c932c2390586474c2f5d80babe13cb0f`;
- governance report Markdown SHA-256:
  `3b95e5488ba8f3628a1f57e0581381d86878b426fb811d6a15b73ba95a094ed7`.

Both `sealed_internal_accessed` and `official_foundir_test_accessed` are false;
the decoder was not fine-tuned. The advisor-authorized route still permits a
decoder intervention without another advisor request, but the evidence does
not yet identify whether the best next move is endpoint regularization/data
scaling, a better teacher/student objective, or conditional decoder tuning.

This is now a mandatory **GPT-6 Astra medium** decision point under the model
contract above. Stop Sol medium after completing the terminal audit and tell
the user to switch before selecting or implementing the next scientific
design. Astra should first decide whether the next bounded gate targets
generalization in the restoration mapping or off-manifold decoder sensitivity;
only after that design is frozen should execution return to Sol high. No new
advisor round is required. Continue to prohibit Qwen degradation conditioning,
mixed degradation, sealed internal, and official FoundIR test.

## Endpoint student CPU-ready — 2026-09-14T06:22:00Z (LATEST AUTHORITY)

The user switched to Astra and said `continue`; the earlier Sol-to-Astra pause
is resolved. The next bounded experiment is implemented in
`configs/foundir_blur_only_zimage_endpoint_distill_v1.json` and
`smoke/train_foundir_blur_only_zimage_endpoint_distill_ddp.py`.
It initializes from immutable v3 step 1200 and trains a single-step endpoint
with RGB MSE (weight 1), GT latent MSE (0.1), LPIPS (0.05), and filtered
teacher latent MSE (0.01). Teacher imitation is enabled only for training
pairs where the frozen 50-step teacher beats input on BOTH PSNR and LPIPS.
No development GT enters training or teacher filtering. If no training pair
qualifies, label the run supervised single-step recovery, not distillation.
The fixed 16 train/16 development crops remain unchanged; 400 optimizer
steps, global batch 8, checkpoints/evaluations at 0/200/400. Trainable LoRA
and AdamW state are FP32; frozen DiT/VAE/text weights remain BF16. The frozen
decoder is differentiated through, with activation checkpointing.

Code inspection verified v3 stores LoRA parameters and AdamW moments in BF16.
A scalar AdamW reproduction at weight 0.0300293 and LR 5e-5 made no parameter
change in 100 constant-gradient steps. Actual v3 checkpoints DO change, so
this is a proven numerical hazard, not a proven complete explanation of the
quality failure. V3 also trains on GT-interpolated states without endpoint
RGB supervision or a persistent LQ condition. The endpoint experiment
directly trains its deployed input/output mapping. Healthy GT roundtrips
establish VAE reconstruction capacity but do not rule out decoder sensitivity
to off-manifold predicted latents. Decoder tuning remains authorized later.

Design report/hold `zimage_endpoint_distillation_design-8fb09bd13e721f1a`
is resolved by recording the user's existing explicit authorization; no new
advisor response is invented or requested. Four focused tests, compilation,
shell syntax, launcher preflight, full artifact hashes, and H100/H200 inventory
checks pass. Real 6B CPU forward/backward/optimizer smoke passed at 64x64:
loss 0.5259563; all 476 trainable tensors have finite gradients; decoder
parameters have no gradients; an actual FP32 optimizer update changes weights
and all AdamW first moments are FP32. Log:
`logs/zimage_endpoint_cpu_smoke_v1.log`. This verifies interfaces, not quality.
Config SHA-256:
`1d98854ec7dc40e3d678879d849955180d652cbfcd86ec77596912214c956f50`.
Prior sweep identifies 3/16 training pairs with both metrics better than LQ
(`0073349`, `0080250`, `0102666`); the new run recomputes teacher targets and
filter decisions before training, without assuming bitwise equivalence between
PEFT and hotloaded LoRA arithmetic. Frozen teacher references and student
0/200/400 outputs will all be saved.

No endpoint GPU training has started; this shell is CPU-only. Exact next action
on the GPU shell (same workspace):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_foundir_blur_only_zimage_endpoint_distill_v1_8gpu.sh
```

Persistent log: `logs/foundir_blur_only_zimage_endpoint_distill_v1.log`.
Output: `checkpoints/foundir_blur_only_zimage_endpoint_distill_v1/`.
Inspect advancing telemetry and step-0/200/400 fixed images, then review both
training learnability and development generalization. This experiment changes
precision and objective together; a gain would not isolate their individual
causal contributions. Do not rerun the completed sweep. No full teacher
training or decoder fine-tuning is claimed complete.

## Z-Image VAE ceiling and solver sweep terminal result — 2026-09-14T06:09:50Z (historical terminal evidence)

The frozen-v3 diagnostic started at `2026-09-14T06:04:18Z` and completed all
scientific computation on homogeneous 8x `NVIDIA H100 80GB HBM3` with
`flash_attention_2`. Each rank used approximately 22.9GB. It produced 14
aggregate snapshots, 224 finite per-image metric rows, and all 224 fixed
triptychs. An independent terminal audit passed exact config/checkpoint,
record-count, record-identity, aggregate, artifact-count, and protected-data
checks.

The runner ended at `2026-09-14T06:07:41Z` with status 2 only after the
complete immutable `result.json` had been written and verified: its final
advisor-report command used unsupported outcome spelling `success` instead of
`information`. This was a post-result reporting bug, not a model, CUDA, NCCL,
or evaluation failure. The runner is corrected, and the information report was
emitted manually without rerunning any inference.

Immutable result identities:

- config SHA-256:
  `062d697b9a0e29728affd983212c27fcff2325a6764099bfca7b4f36b9953292`;
- frozen v3 step-1200 teacher SHA-256:
  `e583e4871d8c498da655e4350e4931662f9310bc197d3d089f9c4241e6625323`;
- result:
  `checkpoints/foundir_blur_only_zimage_vae_ceiling_v1/result.json`, SHA-256
  `b33eac791bd03a61783e0ad5ca288de75ed7e273fd832865de0c423b8a3cc431`;
- governance report JSON SHA-256:
  `9b3d52bb8f7b7ce6a1ae8cd1aa025a817d9e4c0d7c5232a8d1239fd2d329fbe4`;
- governance report Markdown SHA-256:
  `062dedf4bae46e5c7afa9939590f7d7699dc5790cb0bebab224f5f963bb02522`.

VAE ceiling:

| Split | Input PSNR / LPIPS | GT VAE roundtrip PSNR / LPIPS |
|---|---:|---:|
| train | 24.7232 / 0.3085 | 41.3357 / 0.0073 |
| development | 23.9045 / 0.3263 | 40.7184 / 0.0076 |

All four VAE ceiling checks pass with approximately 16.6--16.8dB PSNR
headroom. The frozen VAE can represent the GT crops accurately; it is not the
current dominant quality bottleneck. LQ VAE roundtrip stays close to input,
confirming that ordinary encode/decode distortion also does not explain the
teacher failure.

Frozen v3 solver curve:

| Steps | Train PSNR / LPIPS | Development PSNR / LPIPS |
|---:|---:|---:|
| input | 24.7232 / 0.3085 | 23.9045 / 0.3263 |
| 1 | 23.1627 / 0.3718 | 22.5707 / 0.4108 |
| 2 | **24.0993** / 0.3139 | **22.7699** / 0.3607 |
| 4 | 23.8649 / 0.2650 | 22.1956 / 0.3140 |
| 8 | 23.8768 / 0.2194 | 22.0329 / 0.2732 |
| 16 | 23.7494 / 0.2050 | 21.9650 / 0.2591 |
| 50 | 23.5666 / **0.2007** | 21.9002 / **0.2536** |

Two steps maximize PSNR and 50 steps minimize LPIPS on both splits, but no
step simultaneously improves aggregate PSNR and LPIPS over input;
`teacher_metric_candidate_steps` is empty and `solver_candidate_exists` is
false. Fixed visual review confirms the tradeoff: 2-step outputs preserve more
of the input but still add grain/web texture and recover too little blur;
50-step outputs sharpen more aggressively but visibly overprocess edges and
textures, while strong-blur examples remain far from GT. Distilling this v3
teacher naively would compress a known failure rather than repair it.

This is the explicit model-strategy decision point requested by the user.
Stop Sol here and ask the user to resume with **GPT-6 Astra medium** before
selecting and implementing the authorized few-step/single-step distillation
and/or decoder/restoration-interface correction. Distillation and decoder
tuning remain authorized without another advisor request, but the next design
must account for the evidence that the VAE ceiling is healthy and the learned
flow/teacher is the bottleneck. Continue to prohibit Qwen degradation
conditioning, mixed degradation, sealed internal, and official FoundIR test.

## Z-Image v3 terminal result and authorized post-v3 route — 2026-09-14T05:51:36Z (LATEST AUTHORITY)

The solver-aligned Blur-only Z-Image paired-flow v3 continuation completed
normally at `2026-09-14T05:37:31Z` on homogeneous 8x
`NVIDIA H100 80GB HBM3` with `flash_attention_2`. It initialized exactly from
the immutable v2 step-400 LoRA, reset the optimizer, and completed all 800
additional optimizer steps (`1200` cumulative) in `798.6471 s`. It trained
only on the deployed 50-step static-shift-6 grid with plain unweighted
velocity MSE. Terminal status was 0; there was no OOM, NCCL, runtime, or
protected-data failure.

Immutable identities:

- config SHA-256:
  `6909e7fcf2397fb4b911870076a74544b1f111edb3e0a895309490e804eaf438`;
- step-800 checkpoint SHA-256:
  `c37cc2211b7c80a610af6f2b1547f1aaa2055d852ea71f10aae77c753a5bbcb5`;
- step-1200 checkpoint SHA-256:
  `e583e4871d8c498da655e4350e4931662f9310bc197d3d089f9c4241e6625323`;
- training result SHA-256:
  `17def3137a126e57b3acd79c6d2767d250a2b35d15ceecaf65ca59787be3fd34`;
- evaluation result SHA-256:
  `393cc6427aa0b3d8aaf70db8dc90ca750e67388e6e9a2de475751dfebd414d12`.

| Split | Step | Input PSNR / LPIPS | Output PSNR / LPIPS |
|---|---:|---:|---:|
| train | 400 | 24.7232 / 0.3085 | 23.2828 / 0.2415 |
| train | 800 | 24.7232 / 0.3085 | 23.4878 / 0.2223 |
| train | 1200 | 24.7232 / 0.3085 | 23.5666 / 0.2007 |
| development | 400 | 23.9045 / 0.3263 | 22.6414 / 0.2692 |
| development | 800 | 23.9045 / 0.3263 | 22.5367 / 0.2640 |
| development | 1200 | 23.9045 / 0.3263 | 21.9002 / 0.2536 |

The v3 automatic gate is false. Additional optimization monotonically improved
train LPIPS and slightly improved train PSNR, but development PSNR worsened by
step 1200. Final output versus input is `-1.1566 dB / +0.1078 LPIPS
improvement` on train and `-2.0043 dB / +0.0727 LPIPS improvement` on
development. Fixed-triptych review shows content and geometry remain intact,
but outputs add fine grain/web texture, overprocess near-clean inputs, and do
not recover strongly blurred detail to GT quality. This is a scientific
teacher failure, not a runtime failure.

The emitted hold is
`zimage_blur_teacher_paired_flow_tiny_overfit_v3-b6567614820dbc4e`, SHA-256
`616244388727264984cb45e3abd287d1946e546c1a9be07f7513985f97a67898`.
The matching recorded user-after-advisor decision at
`2026-09-14T05:40:33Z` resolves it as `resume_same_mainline` and explicitly
states that no additional advisor question is required. Governance check
passes. Distillation and conditional decoder fine-tuning are authorized; old
text below saying they are categorically prohibited after a failed teacher is
superseded. The intended ordering remains teacher evidence, then
few-step/single-step work, then decoder tuning if quality remains inadequate,
but a failed teacher gate must not create a permanent deadlock.

Current exact task: finish and validate the already-partially-implemented
read-only VAE ceiling plus frozen v3 step-1200 solver sweep on the same fixed 16
train and 16 source-disjoint development crops. It measures LQ and GT VAE
roundtrips and evaluates `[1, 2, 4, 8, 16, 50]` steps, saving fixed
`Input | Output | Ground Truth` triptychs and PSNR/LPIPS. After that evidence,
proceed directly to few-step/single-step distillation; if results remain
inadequate or the VAE/latent ceiling is limiting, proceed directly to decoder
tuning without another advisor request. Continue to prohibit Qwen degradation
conditioning, mixed degradation, sealed internal, and official FoundIR test.

CPU readiness was completed at `2026-09-14T05:59:30Z`. The diagnostic config
SHA-256 is
`062d697b9a0e29728affd983212c27fcff2325a6764099bfca7b4f36b9953292`.
The launcher and runner executable bits are set. Python compilation/import,
shell syntax, 12 focused tests, the launcher `--check`, governance resolution,
`git diff --check`, all six roughly 29GB model-artifact hashes, and the v3
step-1200 checkpoint hash passed. The evaluator additionally rejects empty,
non-finite, duplicated, missing, or unexpected distributed records and reports
the best PSNR and LPIPS step for each split. This CPU-only shell exposes no
`nvidia-smi` or CUDA device, and no diagnostic runtime log/output exists yet;
therefore the sweep has not started. Exact next GPU action on a homogeneous
8xH100/H200 node is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_foundir_blur_only_zimage_vae_ceiling_solver_sweep_8gpu.sh
```

## Z-Image paired-flow teacher v2 terminal result and authorized v3 continuation — 2026-09-14T04:43:55Z (LATEST AUTHORITY)

The corrected v2 job completed successfully at `2026-09-14T04:39:37Z` on
homogeneous 8x `NVIDIA H100 80GB HBM3` with `flash_attention_2`, 400/400
optimizer steps, and terminal status 0. Training took `384.0424 s`; no OOM,
NCCL, protected-data, or runtime failure occurred. Immutable identities:

- config SHA-256
  `d6f5946fcd75050a17d96c093f7590552f89d14caaf7dce0c659fe35fcb2fbb8`;
- step-200 SHA-256
  `95844e51228046664b2647f2a3f34800f9455f1b171bbd79f0627c9e2e7c73c5`;
- step-400 SHA-256
  `2a3e9b3392dfea20f9e6e869febce1ff99bcaa751e317e247c908f683a4ecd39`;
- training result SHA-256
  `b941e313b15f233ad7bc938ab9f558f83da2ed35c8e784c142c8abf4cc39d2c2`;
- evaluation result SHA-256
  `1c1bdf59c03dfd29773fd122cfb1a101bc6b45542816de2b9c7b6b053d9bd890`.

| Role / state | PSNR input -> output | LPIPS input -> output | Improved fractions |
|---|---:|---:|---:|
| train / step 0 | 24.7232 -> 14.5875 | 0.3085 -> 0.6186 | 0.0000 / 0.0625 |
| development / step 0 | 23.9045 -> 13.0060 | 0.3263 -> 0.6930 | 0.0000 / 0.0625 |
| train / step 400 | 24.7232 -> 23.2828 | 0.3085 -> 0.2415 | 0.1250 / 0.8125 |
| development / step 400 | 23.9045 -> 22.6414 | 0.3263 -> 0.2692 | 0.0625 / 0.8125 |

Visual review of fixed train/development triptychs found that v2 repaired the
v1 noise/content collapse: scene content, geometry, and text are substantially
preserved and many outputs are sharper. It still fails the teacher gate because
train PSNR loses `1.4404 dB` and development PSNR loses `1.2632 dB`; outputs
remain below GT sharpness and sometimes add grain, over-sharpening, or luminance
changes. Near-clean records are unnecessarily altered, including train
`foundir_train_0091486` (`36.883 -> 26.250 dB`) and development
`foundir_train_0062371` (`27.230 -> 21.971 dB`). Full teacher training,
distillation, and decoder tuning therefore have not started.

The emitted hold is
`zimage_blur_teacher_paired_flow_tiny_overfit_v2-ea895f0c24c75257`, SHA-256
`c749f28a843c718fd474d714fa293a1ab77639a718d2d970116f51a1152abcd7`.
The user explicitly directed best-effort continuation of the same Blur-only
Z-Image route without another advisor round. The matching recorded decision is
`resume_same_mainline`, SHA-256
`515e869861d305acb0a747c9b15c26dd1fb2abd4a5c1934ada035fdc0758809a`;
`advisor_governance.py check` passes.

Exact next action: implement an isolated v3 continuation initialized from the
immutable v2 step-400 LoRA. Reset the optimizer explicitly, train only on the
actual 50-step shift-6 inference grid with plain unweighted velocity MSE, save
cumulative steps 800 and 1200 under a new output root, and reevaluate the same
fixed 16 train plus 16 source-disjoint development crops. Do not start full
teacher training until both PSNR and LPIPS improve over input and the fixed
visual gate passes. Decoder tuning remains prohibited until after a valid
teacher and inadequate distilled model establish a decoder/latent ceiling.
Qwen degradation conditioning, mixed degradation, sealed internal, and the
official FoundIR test remain prohibited.

V3 is now CPU-ready. Config
`configs/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3.json`
has SHA-256
`6909e7fcf2397fb4b911870076a74544b1f111edb3e0a895309490e804eaf438`.
Twenty-six focused tests, compilation/imports, shell syntax, full six-artifact
hashes, launcher `--check`, and governance passed. A real 6B CPU constructor
loaded all 476 parent LoRA tensors exactly; a 64x64 sigma-1 forward/backward
returned finite loss `0.2126598209142685`, finite gradients for all 476
trainable tensors, and nonzero gradients for all 238 LoRA-B tensors. No v3 GPU
job has been launched from this CPU-only Codex shell. The exact GPU action is
`smoke/launch_foundir_blur_only_zimage_teacher_paired_flow_v3_8gpu.sh` with
literal `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` on homogeneous H100/H200.

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
`smoke/launch_foundir_blur_only_zimage_teacher_paired_flow_v2_8gpu.sh`. The
launcher uses a retained tmux session when available and a verified
`nohup + PID + persistent log` fallback on GPU shells without tmux. Then inspect the fixed step-0/step-400 triptychs and
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
