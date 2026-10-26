# FoundIR current handoff — 2026-09-23

## ACTIVE: Rain-initialized image-only Rain/Blur MoE — 2026-10-26

The user explicitly approved executing the plain image-only MoE plan after
the Rain gate passed. This supersedes the prior Rain-before-MoE hold and the
failed raw-FLUX one-MLP pilot. The only new task is Rain+Blur ordinary DiT-MoE;
no VLM branch, LoRA expert bank, EBT/energy, pixel refiner, extra task or mixed
degradation experiment. Preserve the successful Rain and existing Blur experts.

Contract: `configs/foundir_rain_blur_image_moe_v1.json`, SHA-256
`8988e2374ad736f3e406945f1d9db83ae14587d0aba28a04dbf9cdc900ab5f12`.
Start from the hash-bound successful Rain step-4,000 LoRA on FLUX.2 Klein 4B.
Keep that single Rain LoRA and shared backbone frozen. Replace image MLPs in
all five double-stream blocks with two ordinary full FFN clones per layer;
learn only those FFNs and image routers (853,313,290 FP32 parameters).
There is no second external MoE and no bank of LoRA experts. The dense
control trains one FFN per same block (424,673,280 parameters); disclose the
capacity/compute difference, never call it a parameter-matched causal test.

Both tasks use the SAME fixed cached Rain text context and SAME one-step,
t=1, zero-noise FP32 endpoint. The fixed prompt preserves Rain initialization;
it does not expose each image's task label. No runtime Qwen/VLM.
Do not load the old 50-step Blur LoRA into this one-step path or retrain that
old expert. The NEW MoE must learn Blur from allowed paired GT. Keep the
existing eight Rain loss terms/coefficients, with the auxiliary Restormer
teacher enabled only on Rain; its weight is zero on Blur.

Use 38,334 Rain / 105,057 Blur training records, balanced 16+16 per global
batch32, source-balanced within each task, aligned random 512 crops and D4.
Cross-task development source groups and GT hashes are excluded; the audited
pools have zero such overlaps. Keep the exact 100 Rain development records
and all 2,595 original Blur development crops. A deterministic 100-source
Blur preview is for intermediate diagnostics only, never the full Blur gate.
No official/sealed/internal/parent-test pixels.

Run ONLY a fresh 25-step `moe/qualification_v1` first on one homogeneous
8xH100/H200 node. Before updates, replay all 100 Rain records and require
per-record agreement with the successful checkpoint (PSNR error <=1e-4,
LPIPS error <=1e-5) AND preserve its 34.2/.08 gate. Then exercise hard training
task routes, transition, and image-only routing (qualification hard6/end12).
Task labels never enter validation/inference; fixed-expert diagnostics do
not use labels either. Require finite gradients, actual expert/router
updates, memory safety, F32 model/Adam files and matched evaluation artifacts.

Formal is configured at 3,000 fresh optimizer steps, 96,000 presentations
(48,000 per task, NOT full traversal), with hard warmup250/transition-end500.
Validate at 0/1000/2000/3000, using full Blur only at formal final. Formal
requires a passing SAME-ARM qualification and never seeds from it. The dense
arm needs its own qualification. No automatic formal/other-arm/VLM launch.
Do not claim a working MoE merely from numeric gates: inspect expert usage
and constant-expert ablations, and compare against the matched-data dense arm.
Rain gate is >=34.2/<=.08; full Blur average gate >=25.9/<=.18, with tail
regressions reported separately as in the historical Blur result.

24 CPU tests, a separate two-rank CPU DDP test and real-file preflight passed.
The user then launched `moe/qualification_v1` on 8xH100. It TERMINALLY
COMPLETED at 2026-10-26T10:41:54Z, launcher exit 0, 25/25 steps, engineering
pass. Initial Rain per-record PSNR/LPIPS errors are both exactly zero.
Both expert branches and routers updated; finite training, FP32 checkpoint
and optimizer artifacts, no OOM/runtime failure. Recorded rank-0 peak memory
37.39800262451172 GiB; median step 1.7897918989765458 seconds.
Launcher: 484 seconds, .13444444444444445 wall hours,
1.0755555555555556 allocated H100 GPU-hours. Trainer-only timer:
.10672626883945971 hours / .8538101507156777 GPU-hours.

Do NOT describe this engineering pass as a working two-task MoE or gate pass.
Rain fell from 34.22743253549469/.07367917290423065 to
33.77674230661362/.07843095208052546 (PSNR gate FAIL, LPIPS pass).
Blur preview improved from 23.788320945242468/.42077609404921534 to
24.169359410868655/.4072999455779791; this is only 100 records, NOT the full
Blur gate. Fixed-expert diagnostics do not establish successful task
specialization, and task-average router weights remain close to half/half.
Preserve the successful original Rain checkpoint and the qualification.
Never resume qualification as formal or repeat its used run ID.

The user explicitly AUTHORIZED the fresh 3,000-step MoE formal on 2026-10-26
after reviewing qualification. Use run ID `formal_v1`, from the ORIGINAL
successful Rain, with unchanged 250/500 warmup and frozen protocol. Never
initialize from qualification weights. Qualification receipt/source/config
identities match and formal_v1 output/log were absent at launch handoff.
Codex is on a CPU-only node and has NOT launched it; the user must execute on
the available single 8xH100/H200 node:
`bash smoke/start_foundir_rain_blur_image_moe_v1_tmux.sh formal moe formal_v1`.
Session: `rain_blur_moe_formal_v1`; log:
`logs/foundir_rain_blur_image_moe_v1_moe_formal_v1.log`.
Do not claim it is running until shared launch/training evidence appears.
No dense/VLM run is authorized by this launch. Qualification's 6/12 warmup
is only an engineering stress test, not a reason to retune formal.
Measured provisional formal estimate: 1.94–3.24 wall hours / 15.53–25.90
allocated H100 GPU-hours; full Blur evaluation and filesystem IO add uncertainty.
Result SHA-256: `5c2d2b8cdcfc8177c58d938bb87e0e1c76eb4cc81a7ec4c6cae66a927215c071`.
Terminal report: `reports/foundir_rain_blur_image_moe_qualification_2026-10-26.md`.
Readiness: `reports/foundir_rain_blur_image_moe_readiness_2026-10-26.md`.

## TERMINAL PASS: Rain FLUX continuation completed — 2026-10-25

This supersedes every earlier active/live/failed/unstarted Rain status below.
`continue_v1` completed the authorized 3,000→4,000 continuation on 8xH100:
2026-10-25T23:20:16Z to 23:51:52Z, launcher exit 0, no runtime/OOM/nonfinite/
NCCL failure. Final fixed 100-record development PSNR/LPIPS is
34.22743253549469/.07367917290423065: BOTH >=34.2 / <=.08 gates PASS.
PSNR margin is .027432535494689603 dB, LPIPS margin .006320827095769352.
Compared with step 3,000, PSNR gains .21904474864805223 dB; LPIPS worsens
slightly by .0002468797005713008. Do not call this improvement in both means.
The train probe is 34.171933008547796/.05955105775501579 and is NOT the gate.
This passes the frozen development protocol, not an independent official test,
a novelty claim or a SOTA claim. Near-clean over-restoration remains: 14/24
records with input PSNR >=34 lose PSNR versus the input.

The successful mergeable Rain expert is:
`checkpoints/foundir_rain_flux2_precision_resume_v1/continue_v1/step_004000.safetensors`
SHA-256 `020113ce209728452917d49859da7be220f638f0d27f6f87f182ffdab0a8fc2e`.
Inference remains FLUX LoRA only; Restormer was a frozen training-only teacher.
Keep the expert and all historical checkpoints; do not continue training or
change the successful recipe automatically. The Rain prerequisite for the next
image-only DiT-MoE stage is now cleared. Only after a working image-only MoE,
add VLM token routing; no MoE job has been launched by this completion.

Final LoRA/Adam tensors, Adam step=4000, next_sample_position=128000,
configuration/source identities, metric means and matched record/input
identities were independently verified. No protected pixels. Additional
exposure is 32,000 presentations / 17,181 unique records / 5,780 sources;
cumulative presentations 128,000. Total launcher cost is 31m36s,
.5266666666666666 wall hours / 4.213333333333333 allocated H100 GPU-hours.
Trainer-only clock is .5022755164518539 hours / 4.018204131614831 GPU-hours.
Terminal evidence: `reports/foundir_rain_precision_resume_progress_2026-10-25.md`
and `reports/foundir_rain_precision_resume_terminal_2026-10-25.json`.
No Rain training remains active; do not mistake the older LIVE sections for
current instructions.

## MILESTONE: Rain continuation step 3,500 — 2026-10-25

Development is 34.0641582441431/.07644298265688121 on the identical 100
records. Relative to step 3,000: PSNR +.05577045729646102 dB, LPIPS worsens
by .0030106894532218598. PSNR still misses 34.2 by .1358417558569016 dB;
LPIPS remains below .08. This is a modest PSNR improvement, NOT a joint gate
pass or evidence of uniformly improving perception. Per-image wins are
57/100 PSNR, 35/100 LPIPS, 28/100 both. Train probe is
33.95975334657288/.06225986051140353 (not generalization evidence).
Saved LoRA is 100 F32 tensors; all 100 Adam states remain F32/finite with
step=3500 and next_sample_position=112000. Source hashes remain unchanged,
and aggregate means/record identities match the frozen evaluation files.
Training resumed normally beyond 3,500. Continue only to the authorized
4,000-step terminal milestone without retuning or early termination.

## LIVE: Rain step-3,000 continuation running — 2026-10-25

The user launched `continue_v1` on 8xH100 at 2026-10-25T23:20:16Z and
requested monitoring through terminal completion. GPU preflight passed.
The log explicitly confirms LoRA step=3000, Adam step=3000, exact optimizer
state restoration, first sample position=96000 and LR=1e-5. Step 3,001 and
subsequent steps have finite loss/gradients and real A/B updates. Do not
restart, duplicate, interrupt, retune or advance to MoE. Continue unchanged
through the frozen 3,500/4,000 validations; scientific success still requires
development PSNR >=34.2 and LPIPS <=.08, not merely normal process exit.
Early steady steps are ~1.5 s with ~24.86 GiB peak allocated memory per rank.
The earlier "not launched" wording below is now historical.
Progress: `reports/foundir_rain_precision_resume_progress_2026-10-25.md`.

## ACTIVE: true Rain continuation 3,000 → 4,000 — 2026-10-25

The user explicitly authorized exactly 1,000 more optimizer steps from the
completed full-data `formal_v1` step 3,000. This supersedes the older
"no continuation authorized" status, but not the frozen Rain-before-MoE gate.
Use `configs/foundir_rain_flux2_precision_resume_v1.json` and the new
`smoke/train_foundir_rain_flux2_precision_resume_v1_ddp.py`; do not edit or
restart the completed parent trainer/config, and never start with fresh Adam.

Restore the hash-bound step-3,000 FP32 LoRA AND all 100 Adam moment states /
step counters. Preserve LR 1e-5, the eight loss terms/coefficients, batch32,
512 crops, source-balanced schedule, augmentation, teacher and evaluation.
Resume absolute sample positions 96,000 through 127,999: exactly 32,000 new
presentations, cumulative 128,000. Validate/save only at global steps 3,500
and 4,000 on the same 100 development records and 32 training-probe records.
No protected data, new architecture, pixel refiner, EBT or energy experiment.
Do not automatically extend beyond 4,000 or advance to MoE.

The resume code has passed real LoRA/Adam/source/data preflight, CPU exact
state restoration and interrupted-versus-uninterrupted update tests. Old
checkpoints do not contain global RNG states: data sampling/crop/D4 are
stateless by absolute position, AE encoding uses its mean, anchor noise and
dropout are zero. No claim of bitwise uninterrupted GPU equivalence.
The original files/artifacts remain unchanged. New output namespace:
`checkpoints/foundir_rain_flux2_precision_resume_v1/continue_v1/`.
Log: `logs/foundir_rain_flux2_precision_resume_v1_continue_v1.log`.

Launch on ONE homogeneous 8xH100/H200 node:
`bash smoke/start_foundir_rain_flux2_precision_resume_v1_tmux.sh continue_v1`.
Codex has no GPU; no GPU continuation has been launched here. Session creation
alone is not success: verify `RAIN_RESUME_STATE_RESTORED` and step 3,001.
Estimate 40–50 normal wall minutes / 5.3–6.7 allocated GPU-hours on 8xH100;
replace with observed throughput after launch. Current baseline remains
34.00838778684664/.07343229320365935, PSNR gap .19161221315336263 dB.
Do not promise passing 34.2/.08. Readiness:
`reports/foundir_rain_precision_resume_readiness_2026-10-25.md`.

## TERMINAL: full-data Rain formal completed — 2026-10-25

This supersedes every earlier LIVE/unstarted status for this run.
`formal_v1` completed 3,000/3,000 steps on 8xH100 at
2026-10-25T22:55:59Z, launcher status 0. Final/best development is
34.00838778684664/.07343229320365935; training probe is
33.984372473834554/.060009161999914795. LPIPS passes .08; PSNR remains
.19161221315336263 dB below 34.2. Thus engineering completion succeeds but
the joint Rain gate is NOT passed. Do not advance image-only MoE/VLM routing.
Relative to matched step0, gains are +1.9236860265013505 dB and
-.015389716671779755 LPIPS. The final 1,000 steps add .384159465303874 dB;
do not promise an unrun extension will pass. No automatic continuation.
Actual exposure: 96,000 presentations / 36,156 unique records / 5,780 sources.
Total launcher cost: 1h36m33s, 1.6091666666666666 wall hours and
12.873333333333333 allocated H100 GPU-hours. All final artifacts, F32 tensor
headers, source/config identities and aggregate metric means were verified.
Final LoRA: `checkpoints/foundir_rain_flux2_precision_full_v1/formal_v1/step_003000.safetensors`,
SHA-256 `336d412299cd7cd9ff919d8983b2b4c435105161496510c0cec8748785c15904`.
No next run is active. Any extension/recipe change requires a fresh user
decision and frozen protocol; preserve all prior artifacts.
Complete report: `reports/foundir_rain_precision_full_formal_progress_2026-10-25.md`.

## LIVE: full-data Rain formal started — 2026-10-25

`formal_v1` started at 2026-10-25T21:19:26Z on 8xH100 and logged finite
training steps after matching the qualification contract. User requests
monitoring until terminal completion. Do not restart, duplicate, retune,
interrupt, or advance to MoE. Frozen schedule: 3,000 steps, evaluations at
0/250/500/1000/2000/3000; keep engineering success separate from 34.2/.08.
Log: `logs/foundir_rain_flux2_precision_full_v1_formal_v1.log`.
Output: `checkpoints/foundir_rain_flux2_precision_full_v1/formal_v1/`.
Launch/config source snapshot is backed up at commit 761a1f5 and the
qualification evidence at 2ad559c. The previous unstarted status is historical.

## Latest: full-data Rain qualification PASSED — 2026-10-25

The 25-step full-data qualification completed on eight H100s at
2026-10-25T21:16:59Z, launcher status 0. Actual exposure: 800 records from
800 unique training sources. Development PSNR/LPIPS improved from
32.08470176034529/.0888220098754391 to
32.64423398615503/.08657967696897685; the 34.2/.08 gate remains false.
Train-probe result: 32.66665987243047/.0708855870179832. This is engineering
qualification, not evidence that the final scientific target will pass.
Finite gradients, actual A/B updates, saved F32 LoRA/optimizer, matched
metrics and unchanged source/config hashes were verified. No protected data.
Total launcher elapsed 173 seconds (.04805555555555555 wall hours,
.3844444444444444 allocated H100 GPU-hours). Median step 1.5106424172408879
seconds, reported peak allocated memory 24.854560375213623 GiB.
Update the formal estimate to 1.5-2.0 wall hours / 12-16 allocated GPU-hours,
subject to sustained full-data IO. Formal has NOT been launched.
The already-authorized independent 3,000-step formal can now be started on
the user's GPU node with
`bash smoke/start_foundir_rain_flux2_precision_full_v1_tmux.sh formal formal_v1`.
Initialize from original step-2,000 LoRA/fresh Adam, never qualification.
Do not change source/config after qualification or auto-advance to MoE.
Full evidence: `reports/foundir_rain_precision_full_qualification_2026-10-25.md`.

## ACTIVE: full-data FP32 Rain FLUX — 2026-10-25

The user explicitly authorized full-data matched-resolution training after
the 300-step fitting diagnostic. This supersedes the earlier instruction to
wait for a new full-data decision. Do not resume the 32-image fit run.
Keep Rain development gate >=34.2 PSNR / <=.08 LPIPS before image-only MoE
and then VLM-token routing. No pixel refiner, EBT or action-energy branch.

Use `configs/foundir_rain_flux2_precision_full_v1.json` and its new trainer.
Independent initialization remains the original hash-bound direct-endpoint
step-2,000 LoRA with fresh Adam; neither diagnostic nor qualification weights
may seed formal. Preserve FP32 LoRA/Adam, BF16 compute, shared FP32 endpoint,
teacher training-only, rank32 and all eight original loss terms/coefficients.
The pool is all 38,334 Rain records / 5,780 sources. Cycle uniformly through
source groups then each group's records, random aligned 512 crops and D4
augmentation. Reflect-pad only the 35 training records with short side <512;
never resize or change development images. No decoded training-image cache.
Two spawned workers per rank prefetch two batches each; verify image SHA-256
before first decode and again if file stat changes. Metadata-path checks run
once on rank0, not redundantly on all ranks.

First run a fresh 25-step full-data engineering qualification on one homogeneous
8xH100/H200 node. Then independently train 3,000 steps, batch32, evaluating
at 0/250/500/1000/2000/3000. Formal requires a passing matching qualification
receipt and cannot auto-start. Keep the same 100-record development gate and
32 training probe; probe scores are not SOW results. No protected data.
Qualification estimate .10-.50 wall hours / .8-4 allocated GPU-hours.
Provisional formal estimate 1.5-3.0 wall hours / 12-24 allocated GPU-hours;
replace with full-data qualification throughput (cached 32-image speed is not
a guarantee). Report actual unique records/sources and sample presentations.

Eight CPU tests and full artifact/data preflight passed. No GPU full-data
qualification has been launched from this GPU-less Codex environment.
Start on the user's node:
`bash smoke/start_foundir_rain_flux2_precision_full_v1_tmux.sh`.
Readiness: `reports/foundir_rain_precision_full_readiness_2026-10-25.md`.

## Latest terminal update: Rain precision fit complete — 2026-10-25

The independent `fit_v1` completed 300/300 steps on 8xH100 at
2026-10-25T20:59:43Z, launcher status 0. Training-fit reaches
35.09303906168583/.04928288975497708 (numeric target passes); development
reaches 32.44553214852097/.08547242213506251 (Rain SOW gate fails).
Step-100 development was 32.54685884616055/.08776275758864358.
Do not describe training-fit success as Rain gate success or evidence that
FP32 alone caused the gain. This proves fitting ability on these 32 fixed
training inputs, not unseen-source generalization. Of 24 development inputs
with input PSNR >=34, 20 still lose PSNR after restoration at step 300.
The run is complete: do not resume/extend it on the same 32 images or
auto-launch full-data training. Next proposed work should address
generalization with a frozen full-data, matched-resolution protocol and
measured budget, requiring a new user decision before GPU launch.
Preserve the Rain gate -> image-only MoE -> VLM router order. No pixel refiner.
Total launcher time: 546 seconds (9m06s), .15166666666666667 wall hours,
1.2133333333333334 allocated H100 GPU-hours. This is faster than the prior
provisional estimate. Full result and limits:
`reports/foundir_rain_precision_fit_terminal_2026-10-25.md`.

## Latest update: precision qualification complete; fit authorized

25-step qualification ended normally at 2026-10-25T20:39:00Z on eight H100s.
Development: 32.61941190970921/.08720199926523492, improving both metrics
but still below the 34.2/.08 joint gate. Training-fit:
33.364734543215874/.06714100111275911. Parameter updates and finite gradients
were verified. No protected data used. Full report:
`reports/foundir_rain_precision_qualification_2026-10-25.md`.
User authorized independent 300-step fitting diagnosis from original
step-2,000 LoRA and fresh optimizer, not qualification weights or full-data
training. On the user's GPU node run:
`bash smoke/start_foundir_rain_flux2_precision_fit_v1_tmux.sh fit fit_v1`.
Expected 15-20 minutes / 2.0-2.67 allocated GPU-hours. Fit launch is not yet
confirmed; no log existed at this update, and Codex has no local GPU.

## 2026-10-25 current: Rain numerical repair, no pixel refiner

The latest user instruction is Rain DiT SOW pass first, ordinary image-only
MoE second, VLM-token router third. Blur already passes. The user rejected
the small pixel refiner proposal. Its code is untrained historical work;
do not launch it. The former direct-endpoint formal already completed and
failed: best step 2,000 was 32.085315339143946/.08882607079111039.

The authorized next run fixes BF16 trainable/Adam state and unequal latent
precision in training versus inference. Both now use one shared endpoint
function with FP32 observation-minus-prediction; model compute stays BF16.
Use the new `foundir_rain_flux2_precision_fit_v1` config/trainer/launchers.
The exact best step-2,000 LoRA initializes each independent run with a fresh
optimizer. First 25 engineering steps, then an independent 300-step fit
diagnostic on 32 deterministic training sources at 512 resolution. The same
100-source development set is evaluated separately at the unchanged gate.
The original loss coefficients are retained. No extra refiner or MoE is
trained, and no long full-data run is auto-launched. Four CPU tests and the
32+100 data/artifact preflight pass; no GPU result exists yet.

Launch on the user's single 8xH100/H200 node:
`bash smoke/start_foundir_rain_flux2_precision_fit_v1_tmux.sh`

Log: `logs/foundir_rain_flux2_precision_fit_v1_qualification_v1.log`.
Readiness, limits, budgets and later fit command:
`reports/foundir_rain_flux2_precision_fit_v1_readiness_2026-10-25.md`.

## 2026-10-17 active: explicit pairwise safe gate CPU-ready

Following the terminal confidence-ranking failure, the active replacement is
a fresh `1,199,043`-parameter pairwise safe gate. It directly predicts gain,
perceptual harm and relative log-MSE for each deployable alternative versus
fidelity. It does not reuse the failed scorer or train on obvious external
negatives. Structured VLM severity and candidate identity remain explicit
inputs.

Probability thresholds are selected only on source-disjoint calibration
groups and frozen before evaluation. No valid calibration policy means exact
fidelity fallback. Compilation, configuration/Bash checks, checkpoint hashes
and `7/7` focused tests pass. The next action is a fresh 25-step qualification
on 8xH100/H200, estimated at `.08–.25` wall-clock hours and `.64–2.0`
GPU-hours. Full contract:
`reports/foundir_pairwise_safe_candidate_gate_noise_v1_readiness_2026-10-17.md`.

## 2026-10-17 risk/coverage audit rejects current scorer

The source-disjoint no-training audit completed in `203` seconds (`.0564`
wall-clock hours, `.4511` H100 GPU-hours). On 174 calibration records from 17
source groups, every nonzero acceptance threshold reduces PSNR below the
fidelity-only `43.341621968938014`; even `.4849%` acceptance reaches only
`43.339416964300746`. Calibration therefore selects zero coverage.

On the disjoint 82-record/17-group evaluation partition, the output equals
fidelity exactly at `43.514986642977085/.03269933527846616`.
`scorer_has_deployable_signal=false`. This terminally rejects threshold repair
and the present heteroscedastic scorer. The next route is explicit pairwise
safe-versus-candidate advantage/perceptual-harm learning with source-disjoint
confidence calibration. Full record:
`reports/foundir_complete_candidate_energy_risk_coverage_v1_terminal_2026-10-17.md`.

## 2026-10-17 formal1 stopped; risk/coverage audit CPU-ready

The user stopped complete-candidate energy `formal1` after pass 1 failed.
Shared telemetry stopped after step `2,285` at UTC
`2026-10-17T20:23:20Z`; direct tmux termination produced no normal END marker.
Never resume it or use its post-pass-1 partial state.

The next experiment is a short, no-training source-disjoint risk/coverage
audit of the immutable step-1,706 checkpoint. The fixed 256 development rows
are split by source group into calibration and evaluation halves. Acceptance
threshold selection uses calibration only; evaluation is consumed once. This
determines whether the scorer contains any genuine high-confidence ordering
signal. If it fails, reject the current scorer and move to explicit pairwise
safe-versus-candidate learning rather than adjusting another threshold.

Static checks and `6/6` focused tests pass. Expected cost is `.08–.20`
wall-clock hours and `.64–1.60` allocated GPU-hours on one 8xH100/H200 node.
Full contract:
`reports/foundir_complete_candidate_energy_risk_coverage_v1_readiness_2026-10-17.md`.

## 2026-10-17 complete candidate energy pass 1 fails

`formal1` reached step `1,706` with numerically healthy training and no
protected-data access, but inference-only selection is substantially worse
than fidelity. Selected is `43.18257042020559/.03665485502278898`; fidelity is
`43.39715355634689/.028955043449059303`. This is
`-.21458313614130042 dB/+.007699811573729676 LPIPS` relative damage.

The core failure is over-acceptance: `96.69189453125%` of regions abandon the
safe fidelity candidate, and unchanged proposal alone is selected on
`43.511962890625%`. The intended uncertainty gate is therefore not calibrated
at pass 1. The joint gate is false. The frozen run continued beyond step
1,900 because no pass-1 stop rule was declared; do not restart, retune, or
interrupt it without a new user decision. Full record:
`reports/foundir_complete_candidate_energy_noise_v1_pass1_2026-10-17.md`.

## 2026-10-17 active: complete candidate-conditioned energy CPU-ready

The user explicitly paused the EBT Generalist gate and authorized the complete
candidate-conditioned Noise energy after the 32x32/64x64 GT oracle passed.
The previous seven-alpha and continuous-alpha selectors remain terminal and
must not resume.

The new implementation freezes the existing proposal and hash-bound
step-8,530 residual basis. A `1,193,907`-parameter shared scorer sees the
observation, proposal, actual candidate, two candidate-relative residuals,
and structured VLM severity. It predicts separate relative local MSE and
spatial-LPIPS risks plus uncertainty on 32x32 regions. Wrong-expert,
under-restored, detail-damaged and repeated-proposal outputs are training
negatives only and are never deployable actions. Fidelity is the fail-closed
baseline.

Static/CPU qualification is complete: compilation, JSON/Bash checks, all six
focused tests, spatial-LPIPS forward/backward, and every manifest/checkpoint
hash check passed. No GPU job has been launched from the Codex container,
which has no visible GPUs. The exact next action is one fresh 25-step
engineering qualification on a homogeneous 8xH100/H200 node. Its checkpoint
must never seed formal training. Estimated cost is `.2–.6` wall-clock hours
and `1.6–4.8` allocated GPU-hours; replace this estimate with measured data.
Full contract:
`reports/foundir_complete_candidate_energy_noise_v1_readiness_2026-10-17.md`.

## 2026-10-17 complete-candidate oracle passes both protocols

The no-training audit completed successfully in `235` seconds (`.0653`
wall-clock hours and `.5222` allocated H100 GPU-hours). Both 32x32 and 64x64
protocols pass all predeclared feasibility conditions.

- Fidelity: `43.39715355634689/.028955043449059303`.
- 32x32 oracle: `43.52247426658869/.03426853639393812`, or
  `+.1253207102417946 dB` beyond fidelity; five candidates exceed 1% wins and
  flip-aligned winner stability is `.77020263671875`.
- 64x64 oracle: `43.512206099927425/.03413928690861212`, or
  `+.11505254358053207 dB` beyond fidelity; four candidates exceed 1% wins and
  stability is `.818603515625`.

Both oracle LPIPS values remain safely below `.045362067760033824`. Candidate
wins are meaningfully distributed among over-fidelity, fidelity,
under-fidelity and unchanged proposal, while wrong-expert/detail-damaged
negatives are much worse than fidelity. This resolves the feasibility gate in
favor of implementing the complete candidate-conditioned energy. These are
GT-selected upper bounds, not deployable performance. Result SHA-256:
`7f7bbfa757580efeaeeb457ed32cb764e948c369393c907f6361cff7a8d0111d`.
Full record:
`reports/foundir_complete_energy_candidate_oracle_noise_v1_terminal_2026-10-17.md`.

## 2026-10-17 active: complete-candidate oracle after seven-alpha stop

The user stopped local-action `formal1`. Its final log record is step `3,910`;
the last complete checkpoint and validation are step `3,412`. Direct tmux
termination produced no normal END marker. Never resume this run.

The seven-alpha selector is now recognized as a narrow diagnostic, not the
complete advisor-PPT method. The next authorized action is no-training only:
audit the strong step-8,530 residual basis with genuinely different deployable
candidates (proposal, fidelity, perceptual, under/over fidelity, and
fidelity/perceptual blends), while treating wrong-expert and detail-damaged
outputs only as contrastive negatives. On the fixed 256-record development
set, measure 32/64-patch GT oracle metrics, candidate win shares, and aligned
horizontal-flip winner stability. Do not start energy training until this
candidate pool shows at least `+.08 dB` oracle headroom beyond fidelity,
LPIPS remains within the frozen gate, and winner labels are not random.

## 2026-10-17 local candidate-action formal pass 2: still no action

At step `3,412`, local action remains exactly equal to frozen fidelity:
`43.39715355634689/.028955043449059303`. The selector again accepts `0%` of
alternative regions and chooses alpha `1` everywhere. Relative to proposal it
is unchanged at `+.27299337834119797 dB/-.02140702431097452 LPIPS`, leaving
`.02700662165880202 dB` to the PSNR gate.

Two full passes now show no deployable change. Recent ranking cross-entropy is
roughly `1.95–1.98`, essentially the seven-class uniform reference `1.94591`.
This is a strong failure signal for the current formulation, not a promising
trend. The healthy run has already entered pass 3 and continues unchanged
unless the user explicitly stops it. Full record:
`reports/foundir_local_action_energy_noise_v1_pass2_2026-10-17.md`.

## 2026-10-17 local candidate-action formal pass 1: safe no-change

At step `1,706`, the local-action output is
`43.39715355634689/.028955043449059303`, exactly equal to frozen fidelity.
The fail-closed selector accepted `0%` of alternative regions and selected
alpha `1` everywhere. Relative to proposal this remains
`+.27299337834119797 dB/-.02140702431097452 LPIPS`; PSNR is still
`.02700662165880202 dB` short of the joint gate, while LPIPS passes widely.

This confirms no-harm behavior but is not evidence that the candidate-energy
mechanism works. Recent candidate-ranking cross-entropy remains about `1.98`,
which is not better than the uniform seven-class reference `ln(7)=1.94591`.
Continue the unchanged frozen schedule so later passes can determine whether
ranking begins to generalize; do not alter the live decision margin or loss.
Full record:
`reports/foundir_local_action_energy_noise_v1_pass1_2026-10-17.md`.

## 2026-10-17 local candidate-action energy qualification passed

Fresh run `qualification_v1` completed on 8xH100 from `15:08:02Z` to
`15:11:53Z` with launcher status `0`, finite training, checkpoint and matched
256-record development validation. Cost was `.0642` wall-clock hours and
`.5133` allocated GPU-hours. All protected-data access flags are false.

At step 25, the fail-closed selector accepts zero alternative regions and
chooses alpha `1` everywhere. Therefore local action output exactly reproduces
fidelity: `43.39715355634689/.028955043449059303`, equal to
`+.27299337834119797 dB/-.02140702431097452 LPIPS` versus proposal. This is
the intended safe engineering baseline, not evidence that action selection
improves quality. The qualification checkpoint must not seed formal training.
Next action is a fresh-ID, from-zero 8,530-step formal run.

Checkpoint SHA-256:
`ec98308fefd05fffa92a2dd6d05e76dbc2c38363b7ad251ce220d23c2e8be5f4`.
Validation JSON SHA-256:
`36512bcf48f63aea88e161d611f3bf7888056aba6f737a99a0a37db69794d48f`.

## 2026-10-17 active decision: local candidate-action energy

The user stopped `foundir_spatial_residual_amplitude_noise_v1/formal1` and
rejected further continuous-alpha regression. The launcher ended with status
`120` at `2026-10-17T14:53:05Z`; the last logged step was `5,525`, while the
last complete checkpoint and development validation remain step `5,118`.
Never resume or seed from this stopped run.

The active replacement freezes the Noise proposal and step-8,530 residual
basis, then learns to compare seven local restoration actions:
`alpha={0,.5,.75,1,1.25,1.5,2}`. The model sees only inference-available
observation/proposal/fidelity/residual tensors and predicts one relative cost
per candidate per 8x8 region. Training GT labels which action is locally
better. Validation and inference never use GT for action selection. Alpha
`1` is the safe baseline; initialization and uncertain decisions fall back
exactly to the frozen fidelity output.

This addresses the measured failure directly: continuous regression learned
an average alpha and recovered only `+.0031436309 dB` beyond fidelity, while
the GT-local oracle proves that the frozen direction contains about
`+.2626 dB` additional PSNR headroom beyond fidelity. The new test asks whether
explicit action comparison can recover the missing `.0270066217 dB` needed
for the original `+.30 dB` proposal-relative gate. It is not yet evidence of
an image-quality gain.

## 2026-10-17 spatial-amplitude formal pass 3: improvement flattening

At step `5,118`, spatial is `43.400297187268734/.0317960577121994`.
Compared with frozen fidelity, PSNR is `+.0031436309218444425 dB` better and
LPIPS is `.0028410142631400966` worse. Relative to proposal, it remains
`+.27613700926303864 dB/-.018566010047834425 LPIPS`, so LPIPS passes but PSNR
is `.023862990736961353 dB` short. Pass-2 to pass-3 PSNR gain grew by only
`.0007849782705344696 dB`; the deployable transfer is flattening far below the
GT oracle. Continue the last two frozen passes unchanged. Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass3_2026-10-17.md`.

## 2026-10-17 spatial-amplitude formal pass 2: tiny PSNR gain

At step `3,412`, spatial output is `43.3995122089982/.033109975053776`.
Compared with frozen fidelity, PSNR is now `+.00235865265130997 dB` better,
while LPIPS is `.004154931604716697` worse. Compared with proposal, the result
is `+.27535203099250793 dB/-.017252092706257827 LPIPS`: LPIPS still clears the
target comfortably, while PSNR remains `.02464796900749206 dB` short. The
predictor has begun to recover a small deployable PSNR benefit but remains far
below the GT-local oracle. Continue the frozen five-pass run unchanged. Full
record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass2_2026-10-17.md`.

## 2026-10-17 spatial-amplitude formal pass 1: slight regression

`formal1` is live and reached step `1,706`. Spatial output is
`43.395900413393974/.029359601011492487`, versus frozen fidelity
`43.39715355634689/.028955043449059303`: PSNR is lower by
`.00125314295291574 dB` and LPIPS is worse by `.000404557562433184`. The map
is learning nontrivial variation (`mean 1.0358318537`, standard deviation
`.1107148176`) but has not yet generalized to a metric improvement. Continue
the frozen run unchanged; pass 1 alone is not a justified stop or retune.
Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass1_2026-10-17.md`.

## 2026-10-17 spatial-amplitude GPU qualification passed

`qualification_v1` completed successfully on `8 x H100 80GB` from
`13:19:04Z` to `13:20:14Z`, with status `0`, finite training and matched
checkpoint/validation output. At step 25 the amplitude map remains near its
identity initialization (`mean 1.0000218451`, standard deviation
`.0004146491`); spatial output is therefore effectively identical to frozen
fidelity (`43.39715509861708/.028954966111086833` versus
`43.39715355634689/.028955043449059303`). This proves engineering safety but
is not enough optimization for a scientific conclusion.

The next action is a fresh, from-zero `formal1` run for 8,530 steps. Never seed
it from qualification. Estimated H100 cost is `2.5–3.5` wall-clock hours and
`20–28` allocated GPU-hours; update the estimate at step 1,706. Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_qualification_2026-10-17.md`.

## 2026-10-17 deployable spatial-amplitude predictor CPU-ready

The user authorized the narrow follow-up supported by the oracle. Proposal and
the step-8,530 residual basis remain frozen; only a `223,393`-parameter CNN is
trained to predict one `[0,2]` amplitude per 8x8 patch from observation,
proposal, fidelity and their residual. Its zero-initialized output starts at
exact alpha `1`, reproducing fidelity. Training GT creates analytic local
amplitude targets, but inference and development validation use no GT-time
selection.

JSON/Python/Bash checks, two focused unit tests, strict frozen-checkpoint
loading and diff checks pass. The next action is a fresh 25-step eight-GPU
qualification; its checkpoint must not seed the formal run. Readiness:
`reports/foundir_spatial_residual_amplitude_noise_v1_readiness_2026-10-17.md`.

## 2026-10-17 final fidelity-residual oracle: local headroom confirmed

The GT-only audit completed successfully in `72` seconds (`.020` wall-clock
hours, `.160` allocated GPU-hours), exactly reproducing the checkpoint's
proposal and fidelity metrics. No global amplitude improves enough to cross
the joint gate; maximum global PSNR remains at alpha `1.000`. However, the
GT-derived local 8x8 hard-amplitude upper bound is
`43.65973751991987/.03243445468615391`, or
`+.5355773419141769 dB/-.01792761307387991 LPIPS`, and all 8/16/32/64 local
patch variants pass.

This proves the fidelity direction contains enough local information, while a
single image-wide scale cannot recover it. These GT-dependent patch results
are not deployable model metrics. The next scientific problem is narrowly
defined: learn a spatial amplitude map from inference-available inputs without
GT, rather than simply extending training or changing a global residual
scale. Result SHA-256:
`ce6ab400c5003a630f3e05fed60a5e1fe9e6d6514af86cddc1fed6b9e65e2f66`.
Full report:
`reports/foundir_proposal_residual_basis_noise_v1_oracle_terminal_2026-10-17.md`.

## 2026-10-17 final fidelity-residual oracle launch-ready

The user authorized the no-training GT-only feasibility audit of the frozen
step-8,530 fidelity residual. The implementation evaluates global amplitudes
`0–2` in `.025` increments and analytic local 8/16/32/64-patch amplitudes on
the same fixed 256-record source-disjoint development subset. It verifies both
proposal and alpha-1 fidelity protocol reproduction before writing output and
reports fixed-candidate plus per-image Pareto upper bounds. This cannot be
reported as deployable inference quality.

Python compilation, Bash syntax, focused unit tests, and diff checks pass. The
Codex container exposes no GPU, so launch the prepared script on one
homogeneous eight-GPU H100/H200 node. Estimated cost is `.10–.17` wall-clock
hours and `.8–1.36` allocated GPU-hours. Do not begin continuation, energy, or
gate training before reviewing this result. Readiness details:
`reports/foundir_proposal_residual_basis_noise_v1_oracle_readiness_2026-10-17.md`.

## 2026-10-17 residual-basis formal terminal: near miss

`formal1` completed normally at step `8,530` with status `0`. Final fidelity
is `43.39715355634689/.028955043449059303`, or
`+.27299337834119797 dB/-.02140702431097452 LPIPS` versus proposal. It passes
the LPIPS target widely but misses the `+.30 dB` PSNR requirement by only
`.02700662165880202 dB`; the joint gate is therefore not passed. The final
pass added `+.03130790591239929 dB`, so optimization was not completely
stationary. The remaining PSNR gap corresponds to only about `.620%` further
MSE reduction relative to the final fidelity output.

The successful run used `2.5869` wall-clock hours and `20.6956` allocated
H100 GPU-hours. Do not resume or extend it. Next, run only a GT-only
amplitude/local feasibility audit of the frozen final fidelity residual to
determine whether its direction already contains enough headroom for the
gate. Do not start an energy/gate or continuation before reviewing that audit.
Full record:
`reports/foundir_proposal_residual_basis_noise_v1_terminal_2026-10-17.md`.

## 2026-10-17 residual-basis formal pass 4: PSNR plateau

At step `6,824`, fidelity is `43.365845650434494/.02899720575715037`, equal
to `+.24168547242879868 dB` PSNR and `-.021364862002883456` LPIPS versus the
proposal. It still improves both metrics strongly, but PSNR gained only
`+.011031992733478546 dB` over pass 3 and remains `.05831452757120131 dB`
short of the joint gate. This is a clear plateau signal, so final gate success
is now unlikely rather than assured. Finish the unchanged fifth pass to obtain
the frozen endpoint; do not retune the active job. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass4_2026-10-17.md`.

## 2026-10-17 residual-basis formal pass 3: near joint gate

At step `5,118`, fidelity reached `43.354813657701015/.030396450317311974`,
or `+.23065347969532013 dB` PSNR and `-.01996561744272185` LPIPS versus the
proposal. It now needs only `.06934652030467986 dB` more PSNR to clear the
joint `+.30 dB/-.005 LPIPS` gate. The PSNR gain increased by
`+.1251758486032486 dB` from pass 2 rather than saturating. This makes gate
success likely, but it is still interim development evidence. Continue passes
4 and 5 unchanged. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass3_2026-10-17.md`.

## 2026-10-17 residual-basis formal pass 2: hopeful trend confirmed

At step `3,412`, fidelity reached `43.22963780909777/.03339703518213355`,
improving over proposal by `+.10547763109207153 dB` and
`-.016965032577900274` LPIPS. PSNR gain therefore increased from pass 1's
`.04423 dB` into the predeclared `.09–.12 dB` hopeful pass-2 range, while the
LPIPS improvement also grew. The joint direction is real, but the PSNR gain
still needs another `.194522 dB` to clear the `+.30 dB` gate. Continue the
unchanged run; do not infer a linear endpoint or retune from pass 2. Full
record:
`reports/foundir_proposal_residual_basis_noise_v1_pass2_2026-10-17.md`.

## 2026-10-17 residual-basis formal pass 1: joint improvement

`formal1` is live on `8 x H100 80GB` and reached step `1,706` normally. The
fidelity head is `43.16839203238487/.0400125195938017`, versus frozen proposal
`43.124160178005695/.050362067760033824`: `+.044231854379177094 dB` PSNR and
`-.010349548166232125` LPIPS. It therefore improves both metrics and already
beats the old direct residual's GT-only oracle (`+.042452 dB/-.003946 LPIPS`)
in both dimensions. The new restoration-native basis is producing useful
correction information rather than merely learning a better gate.

The perceptual head trades more PSNR for LPIPS
(`43.0077431127429/.03262401805659465`), showing that the two heads have now
separated as intended. The fidelity head already clears the LPIPS component
of the `+.30 dB/-.005` gate by more than twofold, but its `.0442 dB` PSNR gain
is only about 15% of the PSNR requirement. This is promising pass-1 evidence,
not final success. Continue the unchanged five-pass run; do not retune or
restart it. Pass 1 took about `.458` wall-clock hours and `3.66` allocated
GPU-hours. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass1_2026-10-17.md`.

## 2026-10-17 residual-basis GPU qualification passed; formal run next

The fresh `qualification_v1` run completed successfully on `8 x H100 80GB`
from `2026-10-17T09:57:50Z` to `09:59:08Z`, with launcher status `0`, finite
training, and a saved step-25 checkpoint plus matched 256-record development
validation. The run took about `.0217` wall-clock hours and `.173` allocated
GPU-hours. All protected-data access flags are false.

Proposal is `43.124160178005695/.050362067760033824` PSNR/LPIPS. At step 25,
the fidelity head is `43.13571598380804/.05056087289904099` (`+.011555805802345276`
dB and LPIPS worse by `.00019880513900716323`), while the perceptual head is
`43.13552286475897/.050557308552015456` (`+.01136268675327301` dB and LPIPS
worse by `.00019524079198163236`). The near-identical heads and small LPIPS
regression are not a scientific failure after only 25 warm-up steps; this run
only establishes that the engineering path works.

The next action is a fresh, from-zero `formal1` run for the frozen five-pass,
`8,530`-step schedule. Never resume or seed it from `qualification_v1`. A
conservative runtime estimate is `3.5–6.0` wall-clock hours on one eight-GPU
node (`28–48` allocated GPU-hours), with pass 1 used to update the estimate.
No energy/gate training is authorized before the residual basis demonstrates
adequate headroom. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_qualification_2026-10-17.md`.

## 2026-10-17 residual-basis causal test authorized and CPU-ready

The user explicitly authorized fixing the oracle-proven lack of information in
the direct residual. The active experiment freezes the existing proposal and
uses a `26,116,852`-parameter restoration-native Restormer to learn two
independent proposal corrections from degraded image, proposal, and their
difference. The fidelity head directly optimizes MSE; the perceptual head uses
a stronger LPIPS term while both retain MSE and 8x8 local no-harm supervision.
Energy and gate training remain off until this basis proves sufficient.

JSON, Python, Bash, a focused unit test, and real step-250 checkpoint
initialization pass. The next action is a fresh 25-step engineering
qualification on one homogeneous 8xH100 or 8xH200 node. It must not seed the
formal run. Details:
`reports/foundir_proposal_residual_basis_noise_v1_readiness_2026-10-17.md`.

## 2026-10-17 direct-residual oracle terminal: joint gate fails

The step-`5,118` GT-only feasibility audit completed normally on all 256 fixed
source-disjoint development records. The best maximum-PSNR oracle output uses
GT-fitted 8x8 patch amplitudes on 255/256 images and keeps the proposal on one:
`43.16661264002323/.04641580081442953`, versus proposal
`43.124160178005695/.050362067760033824`. This simultaneously improves both
metrics, but by only `+.04245246201753616 dB` and `-.003946266945604293`
LPIPS, far below the `+.30/-.005` gate. Among target-selecting Pareto points
that do clear the LPIPS reduction, the best PSNR gain is only
`+.03746754676103592 dB`. No Pareto point gains `+.30 dB` PSNR, so the frozen
direct residual does not contain a combination capable of passing the joint
gate. A learned gate over the same candidates cannot fix that ceiling.

The successful rerun took `207` seconds (`.0575` wall hours) on 8 GPUs, about
`.46` allocated GPU-hours. Including the earlier false-failure run, the audit
used about `.94` allocated GPU-hours. Result SHA-256:
`6400d78624dd2a9fe2cc2f67e622dc9371fd44243346df4f5977c6c17fc8a317`.
Full report:
`reports/foundir_proposal_energy_v2_direct_residual_oracle_terminal_2026-10-17.md`.

## 2026-10-17 oracle first launch: protocol-tolerance false failure

The first direct-residual oracle launch completed all 256 records but exited
before writing its result because the protocol-reproduction assertion was too
strict. The rerun proposal was `43.124160178005695/.050362067760033824`
versus checkpoint validation `43.123698368668556/.05036422960256459`: only
`.00046180933713935146 dB` and `.000002161842530766122` LPIPS apart. This is
far below the scientific `+.30/-.005` gate and is consistent with a tiny
GPU-kernel evaluation difference, not a changed dataset or model. The
checkpoint/config hashes and 256-record protocol all matched. The assertion
now permits at most `.001 dB/.00001 LPIPS` and records the exact reproduction
delta in the output JSON. Re-run the same launcher; no training or recipe
change occurred.

## 2026-10-16 V2 stopped; direct-residual oracle is next

The user stopped `noise_energy_v2_formal1` externally after the last logged
step `6,655`. Do not resume it. Since no later milestone checkpoint was
written, the diagnostic input remains the hash-bound step-`5,118` checkpoint.
The user authorized a GT-only feasibility audit on the same fixed 256-record
source-disjoint development subset. It tests global residual amplitudes,
GT-fitted local patch amplitudes, and a per-image Pareto selection over those
candidates. It performs no training and accesses no protected set. The result
can establish whether the learned direct residual contains enough information
to clear `+.30 dB/-.005 LPIPS`; it cannot be reported as deployable quality.
Implementation and single-node launcher:
`smoke/evaluate_foundir_proposal_energy_v2_oracle_feasibility_ddp.py` and
`smoke/run_foundir_proposal_energy_v2_oracle_feasibility_v1_8gpu.sh`.

## 2026-10-16 V2 pass-3 interim

`noise_energy_v2_formal1` reached step `5,118` (three complete effective
passes) and continued to step `6,635/8,530` as of `19:09 UTC` on `8 x H200`.
On the fixed 256-record source-disjoint development set, frozen proposal is
`43.123698368668556/.05036422960256459` PSNR/LPIPS; the adaptive energy arm
is `43.12445606291294/.05040028668190644`, and the direct refiner is
`42.9707455933094/.04185937133115658`. No arm simultaneously meets the
predeclared `+.30 dB` and `-.005 LPIPS` improvement gate. This is interim
evidence, not a reason to alter or stop the frozen five-pass run. Full metrics,
artifact hashes, and protected-data flags:
`reports/foundir_proposal_relative_energy_noise_v2_pass3_2026-10-16.md`.

## 2026-10-16 V2 pass-2 interim

`noise_energy_v2_formal1` resumed from step `1,706` and passed step `3,412`
on `8 x H200`; it was still advancing beyond step `4,100` at the last check.
The fixed development validation shows adaptive energy
`43.12455575913191/.05040373227166128` versus its frozen proposal
`43.123698368668556/.05036422960256459`. The change is only
`+.0008573904633522034 dB` and `+.000039502669096691534 LPIPS` (worse LPIPS), so
the method has not met the gate. The direct refiner improves LPIPS but lowers
PSNR. Continue the frozen five-pass run without retuning. Details and artifact
hashes: `reports/foundir_proposal_relative_energy_noise_v2_pass2_2026-10-16.md`.

## STOP: 2026-10-15 MAX-COMPLETE PPT ENERGY V2 ONLY

The user explicitly chose one final, complete implementation of the PPT
proposal-relative energy idea before any DiT-MoE fallback. V1 must finish its
already-running 8,530-step endpoint and then stop. Do not extend V1.

V2 is restricted to exactly six corrections supported by the V1 failure:
LPIPS-aware training; real frozen-model wrong/under/over/detail-damaged
candidates; trained 1/2/3-step unrolling; per-step PSNR/LPIPS no-harm plus the
option to retain the proposal; per-image frozen-Qwen severity; and spatial
local energy supervision. Do not add a new backbone, dataset, degradation,
specialist, MoE, router, or compositional mechanism. Noise data, frozen
Restormer proposal, source-disjoint validation, protected-data boundaries, and
the `+.30 dB/-.005 LPIPS` gate remain unchanged. This section supersedes the
older instruction to switch directly from V1 to DiT-MoE.

### 2026-10-15 V1 terminal and V2 launch-readiness update

V1 has now completed its frozen five-pass endpoint at `8,530` steps and is no
longer running. Its final best energy arm was
`43.227015025913715/.05713279500741919` versus the unchanged proposal
`43.124160178005695/.050362067760033824`: PSNR improved only
`+.10285484790802033 dB` while LPIPS worsened `+.006770727247385367`.
Therefore V1 failed the `+.30 dB/-.005 LPIPS` gate and must not be resumed.
It consumed `3.1357351445996513` wall-clock hours and
`25.08588115926418` allocated H100 GPU-hours. Full record:
`reports/foundir_proposal_relative_energy_noise_v1_terminal_2026-10-15.md`.

The exact-six-change V2 implementation now exists and passes compilation,
JSON/Bash/diff checks and five direct CPU tests. A PyTorch naming collision
(`self.half` versus `nn.Module.half()`) was corrected before GPU launch.
No V2 GPU job has been launched yet. The next action is exactly one fresh
25-step, single-node eight-GPU engineering qualification; it is not scientific
evidence and must not be reused as the formal five-pass run. Readiness record:
`reports/foundir_proposal_relative_energy_noise_v2_readiness_2026-10-15.md`.

The 25-step qualification subsequently completed successfully as
`noise_energy_v2_qualification1` on `8 x H100 80GB`: launcher status `0`, all
`488` model tensors finite, all `488` trainable entries represented in the
optimizer state, checkpoint and 256-record validation written, and every
protected-data flag false. Step-25 metrics are intentionally non-scientific
and essentially unchanged from the proposal. The qualification checkpoint
must not seed the formal run. The next action is a new-ID, from-zero `8,530`-
step V2 run. Full record:
`reports/foundir_proposal_relative_energy_noise_v2_qualification_2026-10-15.md`.

## MANDATORY: keep coherent progress pushed to GitHub

Do not let model code, experiment contracts, fixes, research decisions, or
terminal results accumulate only in this dirty workspace. After every coherent
material milestone, make a focused commit and push it to `origin/main`.

Required push points are: launch-ready code/config; a substantive corrected
failure; an advisor/user priority change; each meaningful qualification or
pass-level validation; checkpoint selection; and every terminal report. Do not
commit per-step logs while a job is running, but ensure the exact code/config
that launched it is remotely recorded, then push the validated result later.

Because this workspace contains more than one thousand historical modified or
untracked paths, never use `git add -A`, `git add .`, `git clean`, or
`git reset --hard`. Fetch first, stage only an explicit file allowlist, exclude
all data/checkpoints/weights/images/caches/telemetry/live logs/large manifests,
scan staged content for secrets, run syntax and diff checks, then push. When
the dirty workspace conflicts with remote history, construct the commit in an
isolated temporary worktree based on `origin/main`; leave the live experiment
workspace untouched. A push is complete only after verifying the remote commit
hash and reporting it to the user.

The Git remote is `origin` (`Jason-Wang313/zhenhaowangbackup`) and the intended
tracking branch is `main`. The canonical remote base recovered on 2026-10-15
is `5003e1e51bf9373fbdb352cc0420d712d396f114`. Credentials must remain outside
the repository; never record the previously exposed token or any new token.

### 2026-10-15 historical workspace backfill

The accumulated reproducible text backlog was audited against current
`origin/main` and prepared as two explicit Git snapshots: executable
source/config/tests first, then reports/context. This is archival preservation,
not a reactivation of older research branches. The topmost STOP decision in
this handoff and `AGENTS.md` remains authoritative.

Large generated manifests, checkpoints, weights, datasets, images, caches,
telemetry, and live logs are intentionally excluded and remain untouched
locally. The stale lowercase `agents.md` and old top-level session handoffs are
also excluded so they cannot compete with the authoritative context files.
Full scope and validation record:
`reports/github_historical_workspace_snapshot_2026-10-15.md`.

## STOP: 2026-10-15 ADVISOR-APPROVED NOVELTY SEQUENCE

This section is the current terminal decision and supersedes every older active
research priority below. Existing experiment records remain historical
evidence. Do not resume the from-scratch EBT Generalist or start specialist
work.

### What was decided

The advisor reviewed the ideas shown in the PPT and said they are reasonable,
but correctly asked whether any of them already work. They do not yet have
metric evidence. The approved sequence is therefore:

1. **First test the PPT energy idea.** Keep the existing FLUX.2/Restormer
   restoration models frozen. Learn a small restoration-energy model that sees
   the degraded input, the current restored candidate, and the VLM's explicit
   degradation description. Train it to recognize whether that candidate is
   correctly restored, still under-restored, over-restored, produced by the
   wrong expert, or has lost details. Use its gradient for only a few bounded
   correction steps.
2. **Require a real metric improvement at an adequate scale.** Begin with the
   frozen Noise development protocol already specified in `AGENTS.md`. Compare
   with the unchanged FLUX/Restormer output and a matched feed-forward refiner.
   A lower learned energy is not success by itself. The first 500 steps are
   only a code, memory, throughput, and numerical-stability check; missing the
   final metric target there does not reject the method. Once qualification
   passes, use the full 54,585-record Noise training split and run at least one
   complete effective data pass. Continue for predeclared additional passes if
   the source-disjoint validation curve is still improving. Decide failure only
   after adequate exposure shows saturation/no gain, not after an intentionally
   tiny pilot.
3. **If the energy idea does not work, switch to the advisor's DiT-MoE
   fallback.** Let a VLM first state the degradation type, severity, and
   composition, then use this diagnosis to control experts inside a FLUX/DiT
   restoration network. This fallback is sequential, not a parallel active
   experiment.

In one sentence: **try the project's own restoration-energy idea first; if it
fails, test explicit VLM-guided expert selection inside DiT.**

### Frozen experiment scale and available hardware — 2026-10-15

The experiment may run on one homogeneous node containing either
`8 x NVIDIA H200` or `8 x NVIDIA H100 80GB`. There is no cross-node execution,
and a mixed H100/H200 node is forbidden. Both profiles preserve global batch
`32`: H200 uses micro-batch `4/GPU` with accumulation `1`; H100 uses
micro-batch `1/GPU` with accumulation `4`. The H100 adjustment only reduces
per-forward memory and does not change steps, data order, or data passes.

The first energy experiment uses the complete `54,585`-record Noise training
split. The planned global batch is `32`, so one complete effective data pass is
`ceil(54,585 / 32) = 1,706` optimizer steps. The scale is:

| Stage | Total optimizer steps | Effective passes | Meaning |
|---|---:|---:|---|
| Systems qualification | `500` | `0.293` | Code, finite gradients, memory, throughput, validation, checkpoint/resume only |
| First scientific checkpoint | `1,706` | `1` | First result allowed to inform the learning curve |
| Primary formal run | `8,530` | `5` | Minimum adequately scaled main judgment |
| Conditional continuation cap | `17,060` | `10` | Continue only if pass 4→5 validation still improves without overfitting |

Validation and checkpointing occur at every completed pass. The matched table
must include the unchanged frozen FLUX/Restormer proposal, a parameter-matched
feed-forward refiner, and energy refinement with 1/2/3 candidate-gradient
steps. The official/sealed/internal/parent-test and frozen 660 pixels remain
forbidden for training, tuning, checkpoint selection, or deciding the step
count.

`100,000` steps are not currently planned because they equal approximately
`58.6` passes over this Noise split. Increasing beyond the 10-pass cap requires
a later explicit decision supported by the pass-by-pass development curve.

The exact duration is pending a real single-node throughput measurement on the
GPU family actually used. After the 500-step qualification, calculate normal
wall-clock hours from that family's measured seconds/step and calculate
allocated cost as `wall hours x 8 GPUs`. Never transfer an H100 timing estimate
to H200 or vice versa, and do not use a historical cross-node measurement.

The current single-node implementation is:

- `configs/foundir_proposal_relative_energy_noise_v1.json`
- `lowlight_restoration/proposal_relative_energy.py`
- `smoke/train_foundir_proposal_relative_energy_noise_v1_ddp.py`
- `smoke/run_foundir_proposal_relative_energy_noise_v1_8gpu.sh`

The launcher auto-detects one homogeneous 8-GPU H100/H200 node. Start with the
500-step systems qualification:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && \
PROPOSAL_ENERGY_RUN_ID=noise_energy_v1 \
PROPOSAL_ENERGY_TOTAL_STEPS=500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash smoke/run_foundir_proposal_relative_energy_noise_v1_8gpu.sh
```

Do not launch the 8,530-step continuation until qualification has produced
finite objectives, a checkpoint, development validation, exact-resume
evidence, and a measured hardware-specific time projection.

#### 2026-10-15 first-launch import failure and fix

Run ID `noise_energy_v1` terminated before model/data/optimizer access with
status 1. All eight workers failed while importing a reused legacy evaluator:
that evaluator imported OpenCV, whose native module required unavailable
`libGL.so.1`. This was a code dependency error, not an H200/H100, CUDA, NCCL,
memory, data, or model failure. Completed optimizer steps: `0`; allocated GPU
training work was effectively zero apart from process startup.

The new training script now implements its required manifest validation,
source-group-disjoint split checks, deterministic sampling, pair loading,
alignment, augmentation, and validation crops locally using PIL/NumPy. It no
longer imports the OpenCV-based evaluator or the legacy Restormer training
script. Python compilation, Bash syntax, JSON parsing, and diff checks pass.
Preserve the failed log and retry with a new run ID:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && \
PROPOSAL_ENERGY_RUN_ID=noise_energy_v1_retry1 \
PROPOSAL_ENERGY_TOTAL_STEPS=500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash smoke/run_foundir_proposal_relative_energy_noise_v1_8gpu.sh
```

#### 2026-10-15 retry1 qualification terminal result

`noise_energy_v1_retry1` completed all 500 optimizer steps on one node with
`8 x H100 80GB`, global batch 32 (`1/GPU x accumulation 4`). Wall time was
`.1976426635413534 hours`; allocated cost was `1.5811413112655281 GPU-hours`.
No OOM, NCCL error, non-finite objective, or protected-data access occurred.

On the fixed 256-record source-disjoint validation set, the frozen proposal was
`43.124160178005695/.050362067760033824`. Energy refinement produced:

- 1 step: `43.124949902296066/.05044266952609178`;
- 2 steps: `43.125487580895424/.050498137210524874`;
- 3 steps: `43.126018427312374/.05055382074897352`.

Thus the short run changes PSNR by only `+.00079` to `+.00186 dB` and worsens
LPIPS by `.000081` to `.000192`. It is not a method pass. The matched direct
refiner reached `43.231114752590656/.05958490435659769`: `+.106955 dB` but a
much worse `+.009223` LPIPS. Per the frozen scale policy, 500 steps are a
systems qualification rather than a scientific rejection.

A post-run schedule audit found that v1 tied cosine-LR horizon to the requested
stop step, so this 500-step run reached minimum LR at step 500. Resuming the
checkpoint to 1,706 would make LR jump upward and would not be one continuous
scientific trajectory. The fix freezes LR horizon at the 17,060-step cap for
all milestones. Preserve retry1 as systems/throughput evidence, but start a
fresh corrected run to the first scientific checkpoint at 1,706 total steps.
Do not resume `step_000500.pt` and do not jump directly to 8,530.

Full report:
`reports/foundir_proposal_relative_energy_noise_v1_qualification_2026-10-15.md`.
Checkpoint SHA-256:
`dffad058a9b9697fb76c580144b482e22dddbb5ae2bfb618e0b1134e1f107cdf`.

#### 2026-10-15 corrected pass-1 scientific result

Fresh run `noise_energy_v1_pass1` completed `1,706` optimizer steps on
`8 x H100 80GB` in `.6484166688554817` wall hours, or
`5.187333353174229` allocated GPU-hours. It completed normally with no OOM,
NCCL, non-finite, or protected-data-access error.

On the same frozen 256-record source-disjoint development validation, the
proposal remains `43.124160178005695/.050362067760033824`. Energy refinement
at 1/2/3 steps reaches respectively
`43.20002077519894/.05506254433294089`,
`43.21786245703697/.05692732973329839`, and
`43.22869956493378/.058651921212913294`. Thus more steps add only
`+.07586` to `+.10454 dB` while worsening LPIPS by `.00470` to `.00829`.
The direct refiner reaches `43.24551533162594/.06125720706040738`, likewise a
small PSNR gain with materially worse LPIPS. No arm passes the predeclared
`+.30 dB/-.005 LPIPS` technical gate.

This is a weak one-pass result, not a code failure and not yet the frozen
five-pass method decision. Resume the exact same run ID and checkpoint to
`8,530` total steps. Do not restart, change hyperparameters, or access the
official/sealed/internal/frozen-660 endpoints. Estimated remaining H100 time is
about `2.59 wall hours` and `20.75 allocated GPU-hours`, plus small validation
overhead. Full report:
`reports/foundir_proposal_relative_energy_noise_v1_pass1_2026-10-15.md`.

Pass-2 and pass-3 validations were subsequently produced at steps `3,412` and
`5,118` while the same run continued normally. At pass 3, the best perceptual
energy arm is Energy-1 at `43.19604617357254/.05484414789316361`, versus the
unchanged proposal `43.124160178005695/.050362067760033824`. This is only about
`+.07189 dB`, while LPIPS is still worse by about `.00448`. No energy arm has
passed the frozen `+.30 dB/-.005 LPIPS` gate. Continue unchanged through the
already-running five-pass endpoint; do not tune from these intermediate
records. Interim report:
`reports/foundir_proposal_relative_energy_noise_v1_pass2_pass3_interim_2026-10-15.md`.

### Why the MoE fallback is not a clean first claim

- FoundIR-v2 already uses MoE, but uses an SDXL backbone rather than DiT.
- MiM-DiT already combines MoE and DiT for All-in-One image restoration.
- MiM-DiT directly looks at degraded image features to choose experts. It does
  not first use a VLM to explicitly report degradation type, severity, and
  composition.
- Therefore the broad claim "MoE + DiT" is already occupied. The only current
  differentiation hypothesis is whether an explicit VLM diagnosis makes
  expert selection and composition better than ordinary image-feature routing.
- Merely adding VLM input to a router is not enough. A matched comparison must
  show better mixed/OOD restoration or expert composition using the same DiT
  backbone and training protocol.

### Evidence status

- Advisor approval: confirmed.
- PPT/action-contrastive energy mechanism: implementable hypothesis, not yet a
  working image-quality result. It must receive an adequately scaled test;
  500-step qualification alone cannot reject it.
- VLM-guided DiT-MoE: fallback hypothesis, not yet implemented or validated.
- No GPU experiment was launched by this documentation update: `0` wall-clock
  GPU hours and `0` allocated GPU-hours.

### Required communication style for future handoffs

Lead with the answer in simple Mandarin, then give technical detail. Explain
every necessary English term in ordinary words, explicitly separate facts from
hypotheses, and always identify both methods and the evaluation protocol behind
a comparison. Never turn advisor approval or code execution into a performance
claim. Report duration in normal wall-clock hours plus allocated GPU-hours.

## STOP: 2026-10-12 FORMAL REAL-DATA GENERALIST ACTIVE

The Generalist-only real-data continuation is active on the frozen
`654,881`-pair train split. It implements hierarchical composition/category
balancing, EMA, raw/EMA development validation, exact checkpoint provenance,
and resumable 8/24/32/40-GPU execution. CPU preflight, epoch-18 hash-bound GPU
qualification, every selected node's I/O gate, and the final multi-node
benchmark all passed before launch. The full readiness record is
`reports/foundir_vlm_energy_real_generalist_v1_readiness_2026-10-11.md`.

### 2026-10-12 formal topology selection

The user selected four nodes / 32 GPUs for the three-pass real-data Generalist
continuation. The frozen profile uses batch 2 per GPU, global batch 64, 10,233
optimizer steps per pass, and 30,699 total steps. Initial estimate: 25–29 wall
hours and 800–928 allocated GPU-hours. The selected initialization is the
epoch-18 joint checkpoint, SHA-256
`666fb152b50024357b6c44ea96469012e5405acb67a4ed08ad5618da1aca182e`.
The epoch-18/config-hash qualification and all four node-local I/O checks were
mandatory launch gates and passed before the active run started.

Those gates have now passed. Epoch-18 qualification completed on 8 x H200 with
status `foundir_vlm_energy_real_generalist_qualification_pass`, config SHA-256
`3c6c8f50092f35331e39c498332f538ba0d5e9ef942f7bbacd028a0c6e79e73e`,
and initialization SHA-256
`666fb152b50024357b6c44ea96469012e5405acb67a4ed08ad5618da1aca182e`.
The four selected nodes each passed the 4,096-pair, 32-worker I/O gate at more
than 120 pairs/s. Hardware identity proved that the allocation is
heterogeneous: node 0 has 8 x H200 and nodes 1--3 each have 8 x H100 80GB.
All 32 devices are Hopper compute capability 9.0. The corrected mixed-Hopper
benchmark passed all ten optimizer steps at `3.171203` mean measured schedule
seconds/step. Formal run `real_generalist_4node_mixed_hopper_v1` is active with
batch/GPU 2 and global batch 64. Do not restart or duplicate it.

## 2026-10-12 pass-1 real-data diagnostic — WIP, not the 660 gate

Pass 1 completed all 10,233 steps and wrote a resumable checkpoint, SHA-256
`7a19231ba772f7dbfc5927ca80b325338574573749bdc2c871104fcc02fe4d88`.
The 16,461-record 256x256-crop validation produced:

| Weights | Overall PSNR | Overall LPIPS |
|---|---:|---:|
| Raw | `26.706398` | `.199866` |
| EMA | `26.293138` | `.202940` |

Raw tier metrics were:

| Tier | Records | PSNR | LPIPS |
|---|---:|---:|---:|
| Singleton | 8,269 | `28.829671` | `.172490` |
| Double | 7,613 | `23.916114` | `.242836` |
| Triple | 579 | `33.070977` | `.025842` |

These crop metrics are not protocol-compatible with the frozen 660
full-resolution reference and must not be presented as a pass. They only show
that the model has not collapsed and isolate Double LPIPS as the current risk.

Triple contains only `blur+low_light+noise`, while Double contains five
heterogeneous categories:

| Double category | Records | Raw PSNR | Raw LPIPS |
|---|---:|---:|---:|
| `blur+noise` | 579 | `42.135151` | `.017810` |
| `haze+low_light` | 1,580 | `22.383274` | `.196602` |
| `low_light+noise` | 1,743 | `24.340798` | `.242655` |
| `haze+rain` | 1,580 | `20.962397` | `.277549` |
| `blur+low_light` | 2,131 | `21.945075` | `.312665` |

The sampler assigns one third of presentations to each composition tier.
Because Triple has one category and Double has five, the Triple category gets
approximately five times the exposure of each Double category. LPIPS also has
weight `.05` and enters the objective only every four optimizer steps, so PSNR
can improve before hard Double perceptual artifacts are resolved. Do not alter
the active frozen run; inspect pass 2/3, freeze one checkpoint, then perform one
matched 660 evaluation.

## Frozen explanation: what the old FLUX/Restormer system lacked vs FoundIR

This is the exact pre-real-EBT conclusion, not a new experiment proposal.

1. **Data scale/domain.** Most FLUX work used synthetic/task-specific data from
   roughly 1,600 clean sources. FoundIR metadata covers 1,010,064 pairs,
   including 378,440 Low-light-containing records; the current safe real split
   provides 654,881 training pairs. The referenced full archive is about 4.85
   TB.
2. **Training exposure.** FLUX experts were selected after hundreds to roughly
   10,000 task-specific steps. Released FoundIR Generalist is checkpointed at
   step 1,300,000 and its Low-light specialist at iteration 60,000 / epoch 413.
   Steps are not batch-normalized, but the exposure gap remains substantial.
3. **Joint composition learning.** The project system was VLM routing plus
   heterogeneous FLUX.2/Restormer experts, not one jointly trained Generalist
   over all eleven admitted singleton/mixed categories.
4. **Sequential interface failure.** On 53 correctly routed Triple records,
   Noise added only `+.069859 dB`; Low-light added `+6.807510 dB` but overshot
   GT luma by `+29.503`; Blur worsened 49/53 and lost `.272308 dB`.
5. **Simulator/domain mismatch.** Synthetic strength and operator order do not
   fully match FoundIR; the Noise/JPEG audit found a codec-order and
   noise-visibility mismatch.
6. **EMA maturity.** Short FLUX/Restormer selections lacked a mature EMA path.
   Real EBT adds EMA `.9999` and matched raw/EMA validation after every pass.
7. **Task alignment, not parameter count.** FLUX.2 is a much larger pretrained
   generative DiT, while FoundIR's approximately 36.3M-parameter UNetRes is
   restoration-native. Generative capacity did not substitute for paired
   pixel-restoration alignment.
8. **Full-system specialist refinement.** Full FoundIR can apply Low-light or
   Weather specialists after its Generalist. This is separate from the current
   Generalist-only gate; specialist research remains paused until every
   Generalist tier passes.

## STOP: 2026-10-11 GENERALIST-FIRST DECISION — SPECIALIST WORK PAUSED

The user has frozen the research priority: focus exclusively on improving the
EBT Generalist. Do not design, implement, train, tune, benchmark, or launch a
Low-light, Weather, trajectory-energy, residual-energy, or other specialist.
The 2026-10-11 specialist prior-art audit remains useful deferred research, but
specialist work may resume only after the EBT Generalist has exceeded the
released FoundIR Generalist under the matched gate below.

### Decisive Generalist gate

Use the existing 660 full-resolution whole-image composition set and the exact
frozen metric protocol: `full_rgb_uint8_image` PSNR and
`area_weighted_nonoverlap_512_rgb_uint8` LPIPS. Compare Generalist against
Generalist; neither side may run a specialist. The immutable released baseline
is in
`checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json`
(SHA-256
`ab4a1b0330c732432072917dca4ba74012bf7efe166e347e43aee4461a1e09e6`).

An unambiguous pass requires the EBT Generalist to beat the released FoundIR
Generalist in both metrics in every tier:

| Tier | Records | FoundIR Generalist PSNR | FoundIR Generalist LPIPS |
|---|---:|---:|---:|
| Singleton | 300 | `27.2323939564508` | `.1744119563086327` |
| Double | 300 | `22.245453630403492` | `.2066620594635684` |
| Triple | 60 | `24.9455711880497` | `.03775338436922618` |
| Overall, reporting only | 660 | `24.757709920211013` | `.17664758756638468` |

The overall row is descriptive and cannot hide a failed Singleton, Double, or
Triple tier. Full FoundIR Generalist-to-specialist values are not the target of
this gate and must not be substituted for the released Generalist values.

### Current run and exact next sequence

The 40,197,726-parameter synthetic `formal_3node_v1` run completed and its
epoch-18 checkpoint is the hash-bound initialization of the active real-data
run `real_generalist_4node_mixed_hopper_v1`. Preserve both runs. Do not restart,
duplicate, reconfigure, or interrupt the active four-node run.

After the real-data run reaches a terminal state:

1. audit all expected checkpoints, validation files, hashes, logs, and terminal
   markers;
2. select a checkpoint using development/validation evidence only;
3. stop or extend only from the frozen development trend, never from official,
   sealed, parent-test, or internal-test feedback;
4. freeze one Generalist checkpoint and run one matched 660 whole-image
   Generalist-versus-Generalist evaluation;
5. reopen specialist research only if every tier's PSNR and LPIPS gate passes.

The passed mixed-Hopper benchmark projected approximately `27.04` pure-training
wall hours and `865.36` allocated GPU-hours on 32 GPUs; validation and
checkpoint overhead make the operational estimate approximately 28--32 wall
hours / 896--1,024 GPU-hours.

## STOP: 2026-10-08 tensor-shard I/O fix READY — GPU QUALIFICATION NOT RUNNING

The one-shot tensor bank is complete: 60,000 train records x eight deterministic
views = 480,000 paired 128x128 samples in 469 contiguous shards (about 44 GiB).
All shard hashes pass. CPU build took `0.096557 hours`, with zero GPU use and no
triple/official/sealed access. Manifest SHA-256:
`561037670c88ead7a115687ef19da51eed061c329fd53b8e9b383b598fb5f3ff`.

The v3 51,200-sample schedule is duplicate-free and exactly balanced at 25,600
singleton / 25,600 double. Shard-local sequential CPU throughput is about
`6,002 img/s`, approximately 37x cached-PNG v2 and 2.34x above measured GPU
feed demand. Terminal bank report:
`reports/foundir_compositional_ebt_tensor_shards_v1_terminal_2026-10-08.md`.
GPU readiness report:
`reports/foundir_compositional_ebt_tensor_shard_smoke_v3_readiness_2026-10-08.md`.

Exact command:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_TENSOR_SHARD_RUN_ID=shard400_v3 CUDA_VISIBLE_DEVICES=0,1 bash smoke/run_foundir_compositional_ebt_tensor_shard_smoke_v3_2gpu.sh
```

Stop after v3 qualification; do not automatically launch 100k.

## STOP: 2026-10-08 target cache READY; cached v2 rerun READY — NOT RUNNING

The immutable 66,000-image clean+nuisance target cache completed in
`1183.632873 seconds` (`0.328787 hours`) using CPU only. All 60,000 train and
6,000 validation PNGs passed a second full SHA-256 pass; triple count is zero.
Cache size is approximately 24 GB. Target manifest SHA-256:
`64bcea7cdb95ae1f399d469dfbb76b4078dbc98e501081084e54a9d086b65178`.
Terminal report:
`reports/foundir_compositional_ebt_scaled_targets_v1_terminal_2026-10-08.md`.

Cached learning v2 is CPU-preflighted. It is bit-identical to v1 for the same
record schedule, crop/augmentation seeds, all three nuisance families, and
fixed validation tensors. The only intervention is cached-target read versus
on-the-fly regeneration. Readiness report:
`reports/foundir_compositional_ebt_scaled_learning_cached_v2_readiness_2026-10-08.md`.

The first `cached400_v2` launch failed at import before data/model/optimizer
access because the direct script path did not yet expose the workspace root to
the sibling `smoke` import. Both arms exited status 1 and no result exists. The
import ordering is fixed; direct-script import and tensor-identity regression
pass. Preserve the failed run and use `cached400_v2_retry1`.

Exact launch command:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_SCALED_CACHED_RUN_ID=cached400_v2_retry1 CUDA_VISIBLE_DEVICES=0,1 bash smoke/run_foundir_compositional_ebt_scaled_learning_cached_v2_2gpu.sh
```

Estimated wall time `0.03–0.25 hours`, allocated cost `0.06–0.50 GPU-hours`.
Stop after v2 and compare against real400_v1; do not auto-launch 100k.

## STOP: 2026-10-08 scaled EBT real-data learning smoke READY — NOT RUNNING

The next bounded diagnostic is CPU-preflighted: matched Plain and
Compositional 26,043,648-parameter EBT arms, one-step descent, batch 128, 400
updates, one homogeneous H100/H200 GPU per arm. Each arm sees the identical
51,200-record unique deterministic train schedule and the same 512-record
source-disjoint lower-order validation subset before and after training.

Target reconstruction for `none/jpeg/resample` nuisance families passes, and
the OpenCV BGR-to-RGB target conversion has an exact source-crop identity
test. No triple, sealed, or official data is accessible. No checkpoint is
saved. Readiness report:
`reports/foundir_compositional_ebt_scaled_learning_smoke_v1_readiness_2026-10-08.md`.

Exact launch command:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_SCALED_LEARNING_SMOKE_RUN_ID=real400_v1 CUDA_VISIBLE_DEVICES=0,1 bash smoke/run_foundir_compositional_ebt_scaled_learning_smoke_v1_2gpu.sh
```

Estimated wall time is `0.10–0.75 hours`, allocated cost `0.20–1.50
GPU-hours`. Stop after the terminal result; do not automatically launch 100k.

## STOP: 2026-10-08 scaled EBT H100/H200 kernel benchmark READY — NOT RUNNING

The first launch at `09:50:21Z` correctly stopped in hardware preflight: the
node exposed four H100 80GB GPUs but the initial gate required H200. No worker,
optimizer update, data access, checkpoint, or terminal result occurred. Preserve
run ID `h200_benchmark_v1` and its log as failure evidence; do not reuse it.
The corrected gate accepts four homogeneous H100 or H200 GPUs.

A bounded synthetic benchmark is CPU-preflighted for two credible larger EBT
sizes: 11,266,048 parameters (384x6) and 26,043,648 parameters (512x8). Each
is tested with official-style one-step and current-project three-step energy
descent. Four independent profiles run concurrently on four homogeneous
H100/H200 GPUs and scan local batches 8/16/32/64/128. Additional GPUs are not
wasted on duplicate profiles.

Config, Python, shell, exact parameter counts, a miniature higher-order full
update, and no-CUDA fail-closed behavior pass. The current Codex shell exposes
no CUDA device, so the benchmark has not started and has consumed zero
GPU-hours. Readiness report:
`reports/foundir_compositional_ebt_scaled_kernel_benchmark_v1_readiness_2026-10-08.md`.

Exact launch command:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_SCALED_BENCHMARK_RUN_ID=h100_benchmark_v1 CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/run_foundir_compositional_ebt_scaled_kernel_benchmark_v1_4gpu.sh
```

Estimated wall time is `0.08–0.75 hours`; allocated cost is `0.32–3.00
GPU-hours`. This benchmark uses synthetic tensors, reads no train/triple/
sealed/official data, and saves no checkpoint. At completion, inspect
`checkpoints/foundir_compositional_ebt_scaled_kernel_benchmark_v1/h100_benchmark_v1/result.json`.
Do not automatically launch full training from this benchmark.

## STOP: 2026-10-08 lower-order EBT scaled-data v1 READY — NO GPU TRAINING

The source-disjoint scaled lower-order data contract is frozen and verified.
It contains 60,000 train records (30,000 singleton + 30,000 double) and 6,000
validation records (3,000 + 3,000), drawn from 1,600 train and 200 validation
source IDs with zero source overlap. All five singleton degradations and all
ten unordered pairs are balanced. Triple records and parent test records are
both exactly zero.

Every selected degraded PNG and every unique source passed SHA-256
verification. A separate `--check` pass and independent manifest aggregation
also passed. No held-out triple, FoundIR 60-image triple endpoint, sealed
internal, or official FoundIR test was accessed. No GPU job was launched and
the cost is `0.0 GPU-hours`.

Frozen output manifest:
`manifests/foundir_compositional_ebt_scaled_data_v1.jsonl`, SHA-256
`d7add2be0ec4edff95034caee1287b4746acffcec76d43ab3f288503bbf3b449`.
Machine-readable metadata:
`manifests/foundir_compositional_ebt_scaled_data_v1.json`, SHA-256
`44f4546a941fa7e150f51eb6e148b7bfb64d4346cbb8a342d73a82c970e51c7a`.
Readiness report:
`reports/foundir_compositional_ebt_scaled_data_v1_readiness_2026-10-08.md`.

The exact next task is a bounded throughput/memory benchmark of a credible
larger EBT configuration on the available 8xH200 node, followed by a frozen
wall-time and GPU-hour estimate. Do not automatically launch 100k training
with the 146,432-parameter toy model. Do not tune on or inspect triple data.
Full scaled training requires a recorded model/data-scale governance decision.

## STOP: 2026-10-07 Compositional EBT 10k steps-only v2 TERMINAL PASS

The matched 10k run completed normally on two H100 80GB GPUs from `20:34:16Z`
to `20:41:24Z` (`7m08s`, approximately `.238 allocated GPU-hours`). Each
independent arm completed 10,000 optimizer steps. Both checkpoints, arm
results, terminal result, and advisor report exist; no process remains.

Both arms pass lower-order learnability on the 30-parent source-disjoint
singleton/double development set:

- Plain: `13.881658 -> 17.041216`, gain `+3.159558 dB`; Smooth L1
  `.020455223 -> .009882082`.
- Compositional: `13.881658 -> 16.099879`, gain `+2.218221 dB`; Smooth L1
  `.020455223 -> .012273887`.
- Plain minus Compositional: `+.941337 dB` and `-.002391806` Smooth L1.

Plain also leads on singleton and double subsets; its double gain is
`+3.575398 dB` versus Compositional `+2.231884 dB`. Train loss materially
decreases in both arms (last-25/first-25 ratios `.474358` and `.535821`). The
correct conclusion is that EBT is viable and 300 steps were insufficient, but
the current lower-order evidence does not support a Compositional advantage.

Result:
`checkpoints/foundir_compositional_ebt_lower_order_scale_v2/scale10k_v2/result.json`,
SHA-256
`9a4e73d261d9bc03b7996133588f90f80e1c130a4c1e4f5f5ecf83a2e61dd3c0`.
Terminal report:
`reports/foundir_compositional_ebt_lower_order_scale_v2_terminal_result_2026-10-07.md`.

No held-out triple, FoundIR 60-image triple endpoint, sealed internal, or
official FoundIR test was accessed. Stop before 100k or triple. The next step
requires an explicit choice: one-time frozen held-out-triple diagnostic,
official-recipe/model/data scale redesign, or 100k steps-only overfit study.
Do not infer that more steps alone will close the `0.941337 dB` lower-order gap.

## 2026-10-07 Compositional EBT 10k steps-only v2 READY — NOT YET RUNNING

The advisor/user directed `maybe after 10k then we consider scale to 100k or
not, execute`. Governance hold
`foundir_compositional_ebt_lower_order_scale_v2_design-3cae9b0bf329cfa0`
is resolved and bound to that exact instruction.

v2 restarts matched Plain and Compositional arms from the identical seed and
changes exactly one scientific variable from v1: optimizer steps increase
from 300 to 10,000 per arm. Model (146,432 parameters), data, crop/presentation
schedule, batch 4, AdamW, LR, weight decay, fixed candidate step size 10,
three-step differentiable descent, final-step checkpoint rule, and access
boundaries are unchanged. The two independent arms run concurrently on two
homogeneous H200 GPUs without DDP, preserving per-arm batch semantics.

CPU preflight passed with identical initial loss and finite higher-order
gradients. Python compilation, JSON parsing, shell syntax, governance, hashes,
and whitespace checks pass. Config SHA-256:
`7d307614e00fdaa9ec6cb0f26bcc3ea2124543708ed03327eb2f409a201c44bd`.
Trainer SHA-256:
`70fc49a45bda1e908b5a1e8a561952494067e494363a6def46cf66041182a3e7`.

The Codex shell has no visible GPU, so v2 has not started and has consumed zero
GPU-hours. Exact external 8×H200-node command (uses GPUs 0 and 1 only):

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_LOWER_ORDER_SCALE_RUN_ID=scale10k_v2 CUDA_VISIBLE_DEVICES=0,1 bash smoke/launch_foundir_compositional_ebt_lower_order_scale_v2_2gpu.sh
```

Estimated total wall time is 8–20 minutes and total cost `.27–.67 GPU-hours`.
Monitor the main log plus the Plain/Compositional arm logs named by the
launcher. The workflow freezes both final checkpoints before one lower-order
development evaluation, then stops. It does not access held-out triples and
does not automatically enter 100k steps.

## 2026-10-07 interpretation lock: EBT remains a viable path

The advisor/user explicitly rejects interpreting the 300-step toy pilot as
evidence that EBT itself is non-viable. Preserve this interpretation in future
work. The official ICLR EBT image recipe trains for 100,000 steps and publishes
successful ID/OOD denoising results; the local v1 pilot used only 300 steps per
arm and a 146,432-parameter restoration adaptation. Its small positive
development gains establish that the implementation can learn a correction,
while its frozen convergence-gate failure establishes only that this bounded
pilot was insufficient.

Brutal qualification: the local toy recipe is not otherwise identical to the
official image recipe. Official S1 uses a large 24-block, width-1536 model,
batch 128, one candidate-gradient step, learnable alpha initialized to 3000,
alpha LR multiplier 9000, peak LR `1e-4`, weight decay `.01`, 10,000-step
warmup, and 100,000 optimizer steps. Local v1 uses depth 2/width 64, batch 4,
three candidate-gradient steps, fixed alpha 10, LR `2e-4`, weight decay
`.0001`, no warmup, 60 train parents, and 300 steps. Therefore a steps-only
scale-up is the correct next controlled test, but not an official-recipe
reproduction and not sufficient by itself to claim paper-level EBT training.

Recommended next bounded decision: restart matched Plain/Compositional arms
from the identical seed for 10,000 fixed steps each while changing no training
or model hyperparameter. This is a 33.3x step-only intervention. Use two H200s
to run the two independent arms concurrently without changing per-arm batch or
optimizer semantics; using all eight GPUs for this 146k-parameter, batch-4 toy
would waste compute and likely increase overhead. Only after this clean test
should official stability/scale ingredients be introduced in a separate,
explicitly labeled experiment.

## STOP: 2026-10-07 matched Compositional EBT lower-order pilot TERMINAL FAIL

The authorized matched lower-order pilot ran to terminal completion on one
NVIDIA H100 80GB HBM3 from `18:40:56Z` to `18:41:41Z`. The measured training
and evaluation body took `24.270043` seconds; 600 optimizer steps cost
approximately `.006742 GPU-hours`. Both final checkpoints and the result are
complete. No process remains.

The frozen scientific status is `lower_order_learnability_fail`. Both arms did
improve the 30-parent source-disjoint development set over degraded input:
Plain `13.881658 -> 14.141973` (`+.260315 dB`) and Compositional
`13.881658 -> 14.175251` (`+.293593 dB`); Smooth L1 also improved by
`.001190057` and `.001337115`, respectively. Compositional minus Plain is only
`+.033279 dB` on this lower-order diagnostic.

The formal failure comes from the frozen convergence gate. Last-25/first-25
train-loss ratios are `1.361700` Plain and `1.377185` Compositional, above the
required `.9`. A read-only complete-epoch audit gives last-five/first-five
ratios `.952996` and `.966471`: modest improvement, still below the required
10% reduction. Changing crop difficulty makes the raw trend noisy, but the
frozen gate must not be rewritten after seeing the result.

Result:
`checkpoints/foundir_compositional_ebt_lower_order_pilot_v1/pilot_v1/result.json`,
SHA-256
`0bfcc30805387e22b8ede15e91e1aa13b6c3d8335d924695d0d99d746ce4b78f`.
Terminal report:
`reports/foundir_compositional_ebt_lower_order_pilot_v1_terminal_result_2026-10-07.md`.

Held-out R0 triple images, FoundIR 60-image triple endpoint, sealed internal,
and official FoundIR test access are all false. The fail created unresolved
hold `foundir_compositional_ebt_lower_order_pilot_v1-c31644dbf2370957`.
Stop: do not rerun, scale, change the gate, or access triple data without a new
advisor/user decision.

User hardware note for future experiments: one homogeneous `8×H200` node is
available. The current 146k-parameter pilot used one H100 because eight-way DDP
would add overhead rather than reduce its 24-second runtime. Design later
substantial experiments for the available 8×H200 node when data/model scale
can benefit from it.

## STOP: 2026-10-07 matched Compositional EBT lower-order pilot CPU-ready and authorized

The advisor/user explicitly authorized the next EBT step with `sure, execute
new step now`. The new governance hold
`foundir_compositional_ebt_lower_order_pilot_v1_design-5d467aa6f7d278cd`
is bound to that exact bounded scope and resolved. The older Phase-0 decision
was preserved under `reports/advisor_governance/decisions/`.

The frozen fail-fast pilot compares Plain dense descriptor mixing with
atomic/pairwise diagonal Compositional mixing. Both arms contain 146,432
parameters and use identical initialization, 60 singleton/double train
parents, 30 source-disjoint singleton/double development parents, data order,
crops, three-step differentiable energy descent, optimizer, and 300 optimizer
steps. Final step is fixed; no metric checkpoint selection occurs.

CPU preflight passed with identical initial Plain/Compositional loss
`0.0009887457126751542`, finite outputs, and finite higher-order gradients.
Three focused EBT unit tests, Python compilation, shell syntax, governance,
and whitespace checks pass. Config SHA-256 is
`a69e48b35fd1e58b03a864e8d1da15bfc5fdd1e3a8f5401c43b5ae67f6844d87`.

The current Codex shell has no `nvidia-smi` or `/dev/nvidia0`, so training has
not started and GPU-hours remain zero. Exact one-GPU H100/H200 command:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_LOWER_ORDER_RUN_ID=pilot_v1 CUDA_VISIBLE_DEVICES=0 bash smoke/launch_foundir_compositional_ebt_lower_order_pilot_v1_gpu.sh
```

Estimated wall time is 10–30 minutes; cost is `.17–.50 GPU-hours`. The run must
stop after the lower-order gate. If either arm fails to improve source-disjoint
development PSNR and Smooth L1 over degraded input, do not access triples. If
both pass, both checkpoints remain frozen and a separate governance decision
is required before any one-time 40-image held-out-triple evaluation. FoundIR
60-image triple, sealed internal, and official test access remain prohibited.

Readiness report:
`reports/foundir_compositional_ebt_lower_order_pilot_v1_readiness_2026-10-07.md`.

## STOP: 2026-10-07 Compositional EBT Phase-0 CUDA PASS

The one-GPU zero-update smoke completed normally on an NVIDIA H100 80GB HBM3
from `18:14:47Z` to `18:15:05Z` (18 seconds, approximately `.005 GPU-hours`).
It ended with `FOUNDIR_COMPOSITIONAL_EBT_PHASE0_END status=0` and no residual
experiment process.

Both exactly matched 143,360-parameter Plain and Compositional EBT arms had
finite nonzero candidate gradients, finite second-order parameter gradients,
and decreasing energy after one synthetic candidate update. Optimizer steps
were zero. Triple endpoint, sealed internal, and official FoundIR test access
were all false.

Result:
`checkpoints/foundir_compositional_ebt_phase0_v1/gpu_smoke_v1/result.json`,
SHA-256
`74d19448da5a498e0e5cc6e18fa2fcf78ddb19f87a88c3fb211c265b01ac22ff`.
Terminal report:
`reports/foundir_compositional_ebt_phase0_terminal_result_2026-10-07.md`.

This proves only CUDA/mechanical readiness, not restoration performance or
novelty. Stop before training. Exact next action is to design and freeze a
matched singleton/double-only Plain-versus-Compositional EBT pilot with no
triple checkpoint-selection leakage and a predeclared one-time triple endpoint.

## Historical Phase-0 CPU readiness — superseded by CUDA PASS above

The advisor/user explicitly chose the lean EBT first step and said `just
execute`. A new hash-bound governance decision authorizes only the pinned
official EBT paper/source audit, minimal matched Plain/Compositional energy
implementation, CPU tests, and a zero-optimizer-step CUDA smoke. It does not
authorize training, checkpoint selection, triple endpoint access, sealed
internal access, or official FoundIR test access.

The official EBT repository is pinned at
`third_party/ebt_official` revision
`19420cbeae655bbf11930219a675ade6897019e8` under Apache-2.0. No released
image checkpoint exists in that revision. The published image comparison is
COCO 2014 `128x128` denoising (`sigma=.1` train, `sigma=.2` OOD test), not
coupled FoundIR restoration. Its 99% statement is specifically 300 DiT versus
3 EBT forward passes.

Phase-0 implements exactly parameter-matched Plain and atomic/pairwise
Compositional EBT arms. Python compilation, three focused unit tests, finite
nonzero candidate gradients, finite second-order parameter gradients, and
one-step energy descent all pass on CPU. Both tiny arms contain 143,360
trainable parameters; optimizer steps are zero. CPU result:
`checkpoints/foundir_compositional_ebt_phase0_v1/cpu/result.json`, SHA-256
`fe4484e21c15307e578f3559c4aac8e51c20a946d992c56af8a31985e0fdcc74`.

At this historical checkpoint the coding shell exposed no CUDA device and no
executable system tmux, so the GPU smoke had not yet run. The workspace-local
tmux was subsequently repaired and the CUDA smoke completed as recorded in the
terminal section above.

Audit report:
`reports/foundir_compositional_ebt_phase0_audit_2026-10-07.md`.

## STOP: 2026-09-27 stage-trajectory first launch failed before step 1; retry ready

The first external 8×H200 launch started at `09:18:29Z` and failed during
`dist.init_process_group()` around `09:19:55Z`. All ranks reported NCCL/CUDA
401 (`operation cannot be performed in the present state`). It never reached
cache construction, model forward/backward, optimizer step 1, checkpoint
creation, double validation, or triple endpoint access. No process remains and
no terminal marker exists. This is a runtime initialization failure, not a
scientific result.

The failure matches the previously verified H200 NVLS issue. The launcher now
exports `NCCL_NVLS_ENABLE=0`, keeps P2P enabled, and first runs an eight-rank
scalar all-reduce plus full adapter DDP forward/backward rehearsal with zero
optimizer steps. Only a passing immutable rehearsal permits formal training.
The correction changes no scientific protocol and passes Python compilation,
shell syntax, unit tests, CPU preflight, governance, and `git diff --check`.

Exact retry:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/run_foundir_stage_trajectory_compatibility_v1_8gpu.sh
```

Do not launch a duplicate. The current Codex shell has no `nvidia-smi`, so it
cannot perform the retry itself.

## Historical: 2026-09-27 stage-trajectory compatibility v1 initial CPU readiness

The advisor explicitly selected the correctly-routed triple restoration
failure instead of fail-closed routing as the paper direction and then said
`execute`. A bounded `StageTrajectoryCompatibilityAdapter` tiny gate is now
fully implemented and CPU-ready. The current shell has no `nvidia-smi`, so no
GPU process has started.

The 257,235-parameter adapter is exact identity at initialization and fuses
the frozen LQ/Noise/Low-light/Blur stage trajectory with a bounded residual.
Router and expert weights remain frozen. Training and selection contain only
source-disjoint `15Lowlight_Blur` and `16Lowlight_Noise` records; no triple is
used. All 221 source groups from the 660-image endpoint are excluded.

- train: 49 Low-light+Blur + 40 Low-light+Noise, 51 groups total;
- validation: 13 Low-light+Blur + 6 Low-light+Noise, 13 groups total;
- optimizer steps: 1,000 on exactly 8 homogeneous H100/H200 GPUs;
- promotion: both double validation categories must improve PSNR;
- only after promotion: one frozen 60-image full-resolution triple endpoint;
- official FoundIR sealed test access: false.

CPU preflight, exact-identity check, finite-gradient rehearsal, two focused
tests, Python compilation, shell syntax, hashes, source isolation, governance,
and `git diff --check` pass. Governance hold
`foundir_stage_trajectory_compatibility_v1_design-e435439b0673fefc` is bound to
the advisor's explicit scope and resolved.

Exact external launch:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/run_foundir_stage_trajectory_compatibility_v1_8gpu.sh
```

Estimated total wall time: H200 `0.20--0.40 hours`; H100 `0.25--0.50 hours`.
Do not launch a duplicate. Readiness report:
`reports/foundir_stage_trajectory_compatibility_v1_readiness_2026-09-27.md`.

## STOP: 2026-09-27 triple sequential-interaction diagnosis COMPLETE

The user rejected risk-controlled fail-closed routing as the main paper
novelty and explicitly selected the correctly-routed triple restoration
failure for closest-method reproduction and targeted improvement. Do not
resume the aborted CURE two-track training and do not reopen routing or
official singleton evaluation.

A CPU-only audit measured all existing full-resolution stage images for the
53 correctly routed `18Lowlight_Blur_Noise` records. It ran no inference or
training and accessed no official FoundIR test image. Frozen sequence:
`Noise -> Low-light -> Blur`.

- degraded input: `10.110227 dB`;
- after Noise: `10.180086 dB` (`+0.069859`, nearly inert);
- after Low-light: `16.987595 dB` (`+6.807510`) but mean luma overshoots GT
  by `+29.503`;
- after Blur/final: `16.715288/.181321`; Blur worsens 49/53 records and loses
  `0.272308 dB` while luma, gradient, and Laplacian errors all increase;
- Released FoundIR Generalist: `24.986867/.038079`;
- Full FoundIR Generalist -> Low-light specialist: `33.238410/.032058`.

The same-transition double-composition control strengthens the diagnosis:

- `Noise -> Low-light` is `+9.899631 dB` on 43/43 correct
  Low-light+Noise records, but only `+6.807510 dB` on triple: a `-3.092122 dB`
  transfer deterioration. Its post-stage luma bias changes from `+0.057` on
  double to `+29.503` on triple.
- `Low-light -> Blur` is `+0.125191 dB` on Low-light+Blur (32/52 improve),
  but `-0.272308 dB` on triple (only 4/53 improve).

Thus routing error is ruled out on this subset and ordering is secondary:
skipping Blur leaves only `16.987595 dB`, still `7.999272 dB` below the
Generalist. The evidence supports a composition-domain/inter-expert
compatibility failure. The next bounded hypothesis is whether compatibility
learned only from singleton/double expert interfaces can transfer to an
entirely held-out triple. A direct triple-trained specialist may improve
metrics but is not, by itself, credible novelty.

A GT-derived scalar-exposure oracle further separates the failure. On the 53
correctly routed records, per-image least-squares scaling raises post-Low-light
from `16.987595` to `20.631794 dB` (`+3.644199`) and post-Blur from
`16.715288` to `20.477888 dB` (`+3.762601`). This is a non-deployable upper
bound, not a method result. Exposure overshoot is therefore substantial, but
scalar correction alone still leaves roughly 4.36--4.51 dB to the Released
Generalist, proving that color/structure/detail interaction also remains.

New evidence:

- `smoke/diagnose_foundir_triple_sequential_stages.py`;
- `reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.md`;
- `reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.json`;
- `reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.csv`;
- `smoke/diagnose_foundir_expert_interaction_transfer.py`;
- `reports/foundir_expert_interaction_transfer_diagnosis_2026-09-27.md`;
- `reports/foundir_expert_interaction_transfer_diagnosis_2026-09-27.json`;
- `smoke/diagnose_foundir_triple_exposure_oracle.py`;
- `reports/foundir_triple_exposure_oracle_diagnosis_2026-09-27.md`;
- `reports/foundir_triple_exposure_oracle_diagnosis_2026-09-27.json`.

## STOP: 2026-09-27 routing paper ablation COMPLETE; CURE tracks remain stopped

The user stopped both CURE two-track training jobs externally before baseline
training. They are user-aborted, have no terminal scientific result, and must
not be resumed. The older `LIVE` section below is preserved as historical
launch-time evidence and is superseded by this status.

The requested CPU-only paper reaggregation is complete. It reused the frozen
16,461-record development predictions and existing restoration metric
sidecars; it performed no model update, GPU work, new inference, or
official/sealed-image access. At exactly 13,603 accepted records (82.6377%
coverage), the ordinary eleven-class head ranked only by confidence has 92
wrong accepted routes, while the final multi-head router has 17: 75 fewer
errors, or an 81.52% reduction at matched coverage. The final accepted-error
rate is .1250%, versus .6763% for the ordinary head.

The initial categorical consensus was not automatically beneficial: at its
87.6435% coverage it had 201 wrong routes, versus 153 for the pure eleven-head
confidence ranking at the same accepted count. The defensible contribution is
therefore the final compositional/ambiguity/physical fail-closed design, not a
generic claim that more heads always help.

The 60-image Blur+Noise+Low-light subset independently confirms the remaining
failure is restoration composition: 53 correctly routed records still reach
only `16.715288/.181321`, while the seven Low-light fallbacks reach
`18.651010/.147128`. Do not reopen routing or singleton metric optimization.

Canonical new artifacts:

- `reports/foundir_routing_paper_ablation_2026-09-27.md`;
- `reports/foundir_routing_paper_ablation_2026-09-27.json`;
- `reports/foundir_routing_paper_ablation_2026-09-27.csv`;
- `reports/foundir_routing_risk_coverage_curve_2026-09-27.csv`;
- `smoke/reaggregate_foundir_routing_paper_ablation.py`.

Exact next paper task: use this frozen evidence to write the method/related-work
claim around risk-controlled fail-closed restoration routing. Any new model
experiment should target the separately identified triple expert-interaction
failure and requires a new, specific hypothesis; do not repeat the routing
ablation or restart CURE.

## 2026-09-27 CURE two-track formal training LIVE on two nodes

The user launched the two approved tracks on separate homogeneous 8×H200
nodes. `published_style` started at 01:21:23 UTC and `strict_holdout` at
01:24:23 UTC, with the frozen config SHA-256
`01a428b7eced044cc26e5a6a939010d23eade49627d6e776d19938acef823119`.
Both acquired distinct track-local launch locks, passed governance, full-data
preflight, H200/NCCL/DDP, and the no-update full-loss rehearsal, then entered
formal embedder training. At 01:34:24 UTC, published-style was at epoch 15/200
and strict-holdout had completed epoch 14/200. Both produced their immutable
epoch-10 embedder checkpoint (144,974,255 bytes), validation was finite, and
there was no traceback, OOM, NCCL error, or terminal marker. Latest observed
validation included published epoch 14 `loss=0.2894 accuracy=0.909` and strict
epoch 13 `loss=0.1876 accuracy=0.943`. Remote PIDs are not visible in this
shell; advancing shared logs and checkpoints are authoritative liveness
evidence. Do not relaunch either track.

This is only stage 1 of 3. Each track must finish embedder 200 epochs, baseline
300 epochs, and CURE 300 epochs before any frozen CCDD-11 triple evaluation.
Continue monitoring both existing logs. Do not access test metrics early, and
do not infer the long CURE-stage wall time from embedder throughput.

## 2026-09-27 CURE GPU rehearsal TERMINAL SUCCESS — data preflight PASS

The corrected `v1_retry2` no-update rehearsal on homogeneous 8×H200 ran
01:17:32–01:18:01 UTC and ended with both
`no_update_cuda_rehearsal_pass` and `CURE_GPU_REHEARSAL_END status=0` in
`logs/cure_two_track_gpu_rehearsal_v1_retry2.log`. The eight-rank NCCL/DDP
gate passed. The original CURE batch-three full-loss forward/backward had
finite loss `0.2047935277`, finite trainable gradients, and peak GPU memory
`37,857,186,304` bytes (~35.25 GiB); measured first-use forward/backward
was 1.4942 seconds. No optimizer update occurred. This single-batch timing
is **not** a defensible end-to-end training rate, because it excludes
DataLoader/HDF5, optimizer, DDP all-reduce, checkpointing, and the other two
stages. Preserve the earlier failed `v1` and `v1_retry1` logs as provenance.

Both full-data gates also passed: `smoke/build_cure_h5_v1.py --check` verified
three HDF5 databases with 25,704 rows each, and
`smoke/preflight_cure_two_track_v1.py` verified the two archive SHA-256
digests, source-disjoint views, zero strict-track triple exposure, class order,
and matched optimizer-update budgets (10,200 embedder + 60,300 baseline +
321,300 CURE = 391,800 updates per track). No CCDD-11 test evaluation or
FoundIR official sealed test has run. The one-line frozen-unused-parameter
runtime patch in the pinned CURE working tree is documented immediately below;
retain and disclose it in all reproduction reporting.

The approved two-track training is now technically gate-ready, **but not
started**. A genuine full-training time/cost estimate must be measured from
real stage throughput, not inferred from the 1.4942-second rehearsal sample.
Each track needs a homogeneous eight-H100/H200 node; independent nodes can run
the tracks in parallel, while one node requires sequential execution. The
Codex shell does not see the GPU node and cannot start that training itself.

## 2026-09-27 CURE GPU rehearsal retry1 failure — legacy unused DDP parameter

`logs/cure_two_track_gpu_rehearsal_v1_retry1.log` ended at 01:11:24 UTC with
`status=1`. It passed H200 profile, eight-rank NCCL/DDP, released checkpoint
loading, and the full-loss forward/backward; the extra gradient gate then
reported `missing or nonfinite restorer gradients`. Source audit identified
the exact static unused parameter `encoder.conv_esstesss.weight`, explicitly
retained in the released OneRestore architecture for checkpoint compatibility
but never used in `forward`. The pinned official CURE working tree now has a
**one-line local runtime patch** in `cure/models/onerestore.py` to set only
that legacy layer's `requires_grad_(False)`. The weight, state-dict key and
forward numerics remain unchanged. This patch is required for the original
default DDP wrapper to avoid an unused-parameter reduction failure on later
iterations; the clone's HEAD remains `a294116...` but its working tree is
intentionally dirty and must **not** be described as an unmodified checkout.
File SHA-256 after patch:
`63e78fb5c73fbe71aba69b5107235649c66766109b9874f30af478d981182920`.
Single-process CPU forward/backward confirmed every remaining trainable
parameter has a finite gradient. No optimizer update or CCDD-11 test
evaluation has run.

Next GPU-node command, preserving both failed logs:

```bash
CURE_REHEARSAL_RUN_ID=v1_retry2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang/smoke/run_cure_gpu_rehearsal_v1_8gpu.sh
```

Do not start training until this attempt ends with both
`no_update_cuda_rehearsal_pass` and `CURE_GPU_REHEARSAL_END status=0`, and
the full source/archive preflight passes. This is a runtime compatibility
patch, not a new scientific method or a changed loss/data protocol.

## 2026-09-27 CURE GPU rehearsal retry needed — checkpoint reader fixed

The first GPU rehearsal ran on homogeneous 8×H200 from 00:59:14 to 01:00:18
UTC. Profile and eight-rank NCCL/DDP checks passed. The single-GPU full-loss
rehearsal then failed **before forward/backward or any optimizer update** with
`KeyError: 'state_dict'`: its ad-hoc checkpoint reader expected a wrapped
training checkpoint, whereas the SHA-verified released OneRestore restorer is
a plain `OrderedDict`. The failure is preserved at
`logs/cure_two_track_gpu_rehearsal_v1.log` (terminal `status=1`).

`smoke/cure_gpu_rehearsal_v1.py` now uses the pinned CURE repository's
`cure.checkpoint.load_model`, which supports both formats. CPU loading of the
actual released restorer passed with all 189 state entries; Python compile
and launcher `bash -n` passed. The launcher now accepts a safe unique
`CURE_REHEARSAL_RUN_ID` to retain the failed log and refuse duplicate retries.
The exact next GPU-node command is:

```bash
CURE_REHEARSAL_RUN_ID=v1_retry1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang/smoke/run_cure_gpu_rehearsal_v1_8gpu.sh
```

Monitor `logs/cure_two_track_gpu_rehearsal_v1_retry1.log` for
`no_update_cuda_rehearsal_pass` and `CURE_GPU_REHEARSAL_END status=0`.
The three full HDF5 files are present, but do not launch training until their
`--check`/preflight and the corrected GPU rehearsal pass. No CCDD-11 test
metrics or FoundIR official sealed test were accessed.

HDF5 `--check` subsequently passed all three databases: each has exactly
25,704 aligned patch rows and the frozen class order; the strict baseline has
no triple classes. Full source/archive preflight is running separately. GPU
retry had not started at the latest log check.

## 2026-09-26 CURE two-track execution checkpoint — HDF5 build active

The user confirmed that the CURE two-track authorization represents the
advisor's explicit decision. The exact hash-bound governance decision remains
resolved (`smoke/advisor_governance.py check` passes). The three full HDF5
databases are being built serially by the pinned original CURE builder via
`smoke/build_cure_h5_v1.py`; the live process writes
`logs/cure_two_track_h5_build_v1_retry1.log`. At the latest observed check,
`published_baseline.partial.h5` had grown to about 60 GB and the first source
pass was about 23% complete. Do not start another builder or delete/overwrite
the partial file. No CURE training or test evaluation has run.

The independent no-update GPU rehearsal launcher
`smoke/run_cure_gpu_rehearsal_v1_8gpu.sh` passed `bash -n`; the three focused
CURE test modules passed four tests, and the rehearsal Python modules passed
bytecode compilation. This Codex shell has no visible GPU or `tmux`, so the
GPU rehearsal has **not** run. It first checks a homogeneous eight-H100/H200
profile, then eight-rank NCCL/DDP, then a batch-three original CURE full-loss
forward/backward without optimizer update on one GPU. Its log is fail-closed
against duplicate launch. Do not infer full-training wall time from the R0
pilot or call a CPU-only check a CUDA pass. Run the rehearsal on a real GPU
node and use its measured time/memory to estimate the 391,800 optimizer
updates per track before launching training.

Next: monitor the existing HDF5 builder; when it ends, run its `--check` and
`smoke/preflight_cure_two_track_v1.py`, then complete the GPU rehearsal and
only afterward launch the two approved original-code training tracks. FoundIR
official sealed test and the 60-image FoundIR triple endpoint remain excluded.

## 2026-09-26 CURE two-track reproduction: authorized preparation, no GPU result yet

The user explicitly approved the CURE original-code two-track reproduction and
confirmed they are the advisor or are relaying the advisor's explicit approval.
The new diagnostic-only governance hold
`cure_two_track_reproduction_v1_design-e847b2e4404be017` is resolved by an
exact hash-bound decision. The prior R0 decision was preserved verbatim at
`reports/advisor_governance/decisions/foundir_triple_onerestore_r0_v1_design_frozen-4409a69aba5bf9fe.json`.

The official CURE repository is pinned under `third_party/cure_official` at
`a294116ad3d7dce58723a21bdd64f74b085e7b13`. CCDD-11 revision
`4784b88382f8616008ac9ae1e9be36c75f871f4e` is being downloaded into
`data/third_party/ccdd11` (74.06 GB train and 16.39 GB test archives). The
released CURE/OneRestore checkpoints were independently SHA-verified against
their Hugging Face LFS OIDs. A CPU inference on the official low+haze example
completed with non-collapsed output; this is only a runtime preflight, not a
two-track result.

The approved protocol is frozen in `configs/cure_two_track_reproduction_v1.json`.
`smoke/prepare_cure_two_track_v1.py` creates source-disjoint symlink views and
keeps strict-track triple image exposure at zero while matching embedder
presentations per epoch. Its synthetic source-disjointness/exposure unit test
passes. `smoke/estimate_cure_h5_v1.py` will estimate HDF5 storage before any
large database build. No CURE training or CCDD-11 test evaluation has run. The
Codex shell has no visible GPU. Do not claim completion or issue a GPU launcher
until archive integrity, extracted source layout, HDF5 budget, and actual-image
output-collapse preflight pass. FoundIR sealed test and the 60-image FoundIR
triple endpoint remain excluded.

Original-code causal qualification: `EmbeddingClassifier` freezes its text
branch by default; `PromptEncoder` subsequently loads only that text branch,
not the trained visual classifier. Both tracks retain the same 12-row GloVe
prompt vocabulary. Thus seeing triple images in published-style embedder
training need not affect downstream restoration; the baseline's triple-image
exposure is the likely consequential difference. Confirm this by comparing
text-tensor hashes after both embedder runs, not by assumption.

Data acquisition update: both CCDD-11 ZIPs have now passed the Hugging Face
LFS SHA-256 checks (`half_train` `fd957ac7...`, `half_test` `2ed9fd82...`).
`half_test` was safely extracted (12,400 members); the frozen evaluation
manifest has 200 sources per triple class, 400 full-restoration records, and
2,400 selective records with same-source `1deg`/`2deg` targets. Manifest SHA:
`80b6233a87ef80dc650351f96881fd5b3488816231082c9821d1c11e6bdc7106`.
`half_train` safe extraction is in progress. The parallel downloader had one
short HTTP range; it resumed exactly from the recorded offset and the complete
train archive subsequently passed the published whole-file hash. The official
code repository and dataset card do not declare a license in the inspected
metadata; retain code/data for internal reproduction and do not redistribute
them without separate license review.

## 2026-09-26 closest-work claim gate: CURE pretraining exposure qualified

The latest literature/code audit is
`reports/foundir_triple_novelty_claim_gate_2026-09-26.md`. CURE already
covers selective subset supervision and order consistency. Its *CURE
fine-tuning* excludes triples, but the released protocol initializes from a
OneRestore baseline and prompt embedder trained with triple classes. This is
an end-to-end exposure distinction, not a claim of test-image leakage.
DAME-Net already evaluates unseen combinations up to four factors, and a 2023
paper already models degradation operation histories. No new method novelty
has been established.

Next research action: reproduce CURE published-style versus strict
end-to-end triple-free training under matched source split and update budget.
Only if a real gap appears should an order-conditioned intervention method
be designed and checked against degradation-history prior art. The terminal
R0 v1 result remains frozen; no new GPU job was launched by this audit.

## STOP: 2026-09-26 CSR novelty claim withdrawn after CURE full-text audit

The earlier provisional claim that matched counterfactual subset supervision
plus selective removal and held-out triple generalization might be a novel
combination is superseded. CURE (arXiv:2607.03044, July 2026) already trains
with matched selective-restoration pairs and per-factor intermediate losses,
adds order consistency, excludes triples from its fine-tuning stage, and
tests on triple composites. The earlier baseline and embedder stages do see
triple classes. Full evidence and citations are in
`reports/foundir_triple_novelty_cure_prior_art_correction_2026-09-26.md`.

R0 v1 remains terminal and scientifically failed due output collapse. A
read-only checkpoint probe confirms strongly negative pre-clamp residuals and
nearly all-black clamped outputs; it did not prove why residuals drifted.
No new GPU job has been launched. The next paper step is a bounded,
CURE-aware novelty formulation and closest-work comparison. The old CSR
mechanism cannot be presented as a new contribution.

## STOP: 2026-09-26 OneRestore-style R0 TERMINAL — RUNTIME SUCCESS, SCIENTIFIC FAILURE

The user-authorized R0 ran on one homogeneous 8×H200 node from 12:30:12 to
12:35:51 UTC and exited successfully (`status=0`). All three 1,200-step final
checkpoints, 210 full-image outputs, provenance and metric sidecars, terminal
`result.json`, report, and end marker are present and hash-verified. No
official FoundIR test or 60-image FoundIR triple endpoint was accessed.

The frozen primary gate **failed**. Development PSNR/LPIPS: A
`6.810193/0.967408`, B `7.007366/0.984490`, C `6.140267/0.950947`.
Held-out triple PSNR/LPIPS: A `7.010534/0.987038`, B
`7.182890/1.004814`, C `6.636498/0.966714`. B's LPIPS is worse than A by
0.017082; C's PSNR is worse than B by 0.867099 dB. Outputs have collapsed
near black: B mean pixel value is ~0.000005, with 99.9953% endpoint-saturated
pixels. All three arms are worse than the degraded input on every development
image. This is a valid completed runtime but not usable restoration evidence.

Canonical R0 terminal artifacts:

- `checkpoints/foundir_triple_onerestore_r0_v1/result.json`
- `reports/foundir_triple_onerestore_r0_v1_terminal_result_2026-09-26.md`
- `reports/foundir_triple_onerestore_r0_v1_terminal_audit_2026-09-26.md`
- `logs/foundir_triple_onerestore_r0_v1.log`

The R0 stop condition is met. Do not rerun the frozen R0 or automatically
start R1, R2, continuous severity, spatial factorization, or CSR. A future
design must address output collapse and preserve the held-out triple boundary.

## 2026-09-26 OneRestore-style R0 GPU PACKAGE READY — USER AUTHORIZED, NOT YET RUNNING

The user explicitly authorized this bounded diagnostic with the exact statement
`execute`. Do not ask for another approval to fix an in-scope implementation or
runtime defect. Authorization is limited to the frozen three-arm R0:
`A_unconditioned`, `B_fixed_categorical`, and
`C_fixed_categorical_cdrl`. Stop after the R0 terminal result. Do not
automatically enter continuous severity, spatial factorization, R1, R2, or CSR.

The complete implementation, evaluator, and launch path now exist and pass CPU
preflight. A protocol defect found during review was fixed before GPU use: the
trainer now freezes all three final checkpoints before the first held-out-triple
export/access. The official FoundIR test and the 60-image FoundIR triple endpoint
remain prohibited and have not been accessed.

Core launch files:

- `configs/foundir_triple_onerestore_r0_v1.json` — frozen SHA-256
  `415ffaa126393e0125ca070830596c413645919836e290b7a16970b2f27c2c64`
- `lowlight_restoration/onerestore_r0.py`
- `smoke/train_foundir_triple_onerestore_r0_ddp.py`
- `smoke/evaluate_foundir_triple_onerestore_r0_pyiqa_ddp.py`
- `smoke/run_foundir_triple_onerestore_r0_v1_8gpu.sh`
- `smoke/launch_foundir_triple_onerestore_r0_v1_8gpu.sh`

Verification completed on 2026-09-26 UTC: governance resolved; trainer preflight
including real frozen-VGG16 CDRL forward/backward passed; offline pyiqa 0.1.16
LPIPS-Alex preflight passed; 12/12 focused tests passed; Python compilation,
shell syntax, and whitespace checks passed. The rendered cache remains complete
at 620 PNGs. The Codex shell has no `nvidia-smi`, so it could not start the GPU
job. No R0 process, checkpoint, result, or GPU-hours exist yet.

Exact next command on one homogeneous eight-H100 or eight-H200 node:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash smoke/launch_foundir_triple_onerestore_r0_v1_8gpu.sh
```

Estimated total normal wall time, including three sequential arms, all exports,
LPIPS/PSNR, aggregation, sidecars, and terminal report: H200 0.75–2.0 hours
(center 1.25), H100 1.0–2.5 hours (center 1.5). Aggregate cost is H200 6–16
GPU-hours (center 10) or H100 8–20 GPU-hours (center 12), using exactly one
eight-GPU node. The dominant uncertainty is the absence of an external-node
benchmark, especially frozen VGG16 CDRL backward and DDP/shared-storage
throughput.

Expected terminal artifacts are
`checkpoints/foundir_triple_onerestore_r0_v1/result.json` and
`reports/foundir_triple_onerestore_r0_v1_terminal_result_2026-09-26.md`. At
terminal completion, report the R0 result and stop.

## 2026-09-26 triple counterfactual R0 CPU READINESS COMPLETE

The CPU-only reproducibility package prescribed by the literature memo is
complete. It freezes 130 source-disjoint AgenticIR synthetic parents: train
20 singleton + 40 double; development 10 singleton + 20 double; held-out 40
triple. The held-out set covers all ten triple combinations. The builder
expands 620 ordered-subset nodes and 1,710 counterfactual pairs.

Full build and independent `--check` pass. Six focused tests pass. Every source
hash is verified; triple gradient leakage and triple checkpoint-selection
leakage are both zero. Neither official FoundIR test nor the 60-image FoundIR
triple endpoint was accessed. No model was implemented or trained, no GPU ran,
and GPU-hours are zero.

Frozen identities:

- config SHA: `f099113d41867bf42f4143c70ebc7cbbc31d393368d81c7cc0fea3f9ca1b0cd8`;
- manifest SHA: `29fafafec8929d57136068433f081c8998053df0ec638b8f654a5f69e0c4ac85`;
- metadata SHA: `82302da80acde9de29a8964f4125c69b8f893686db81bc31bd5d140afa97f948`;
- readiness report:
  `reports/foundir_triple_counterfactual_reproduction_r0_cpu_readiness_2026-09-26.md`.

Two implementation defects were caught and fixed without changing protocol:
nested lattice operator objects are now deep-copied so mutation cannot evade
validation, and combination names now use the project canonical order instead
of alphabetical sorting.

This is not GPU-ready. A OneRestore-style diagnostic restorer/conditioning path
is a model-family/architecture experiment and requires a governance design hold
plus explicit user/advisor decision before neural implementation or launch.
There is no GPU command or defensible end-to-end runtime estimate yet. Exact
next task after authorization: implement only the R0 fixed categorical
descriptor + CDRL arm, add no-update/forward-backward/DDP checks, benchmark it,
publish normal wall-time and GPU-hour estimates, run the smallest R0 gate, and
stop before R1/R2.

## 2026-09-26 triple-composition literature/reproduction memo COMPLETE

The user directed a return to the literature-driven paper-novelty task after
freezing the successful double-composition aggregate. A read-only literature,
public-code, and local-evidence audit is complete. No GPU training or new
evaluation was launched; GPU-hours are zero.

The prior working hypothesis, factorized continuous type/severity conditioning,
cannot by itself be claimed as novel. CDG-IR (TMM 2026,
DOI `10.1109/TMM.2026.3702504`) already predicts multi-dimensional continuous
composite degradation intensity over five dimensions. F2D-Net (Knowledge-Based
Systems 2026, DOI `10.1016/j.knosys.2026.116411`) already factorizes a shared
transport backbone into degradation-specific increments with pixel-wise spatial
gating. DPPD, CDIR, UniCoRN, and MDIR further occupy dynamic prompting,
local/context-aware processing, controllable multi-head diffusion/MoE, and
continuous compact conditioning.

The revised, still-provisional research question is held-out composition
generalization through counterfactual subset supervision: train only on
clean/single/double synthetic compositions, explicitly supervise selective
removal against rendered subset targets, and evaluate on triples excluded from
training and checkpoint selection. Continuous severity and factorized control
are baselines, not the claimed contribution.

Local data truth supports this protocol: AgenticIR manifests preserve
per-component levels, operator parameters, application order, source hashes,
and source-disjoint splits. The 60 FoundIR Low-light+Blur+Noise validation
records have composition labels but no authoritative per-component severity;
such labels must not be invented. Existing Qwen evidence also shows
composition-complexity degradation: triple type-set exact `.75` and combined
type+severity exact `.4667`.

Full memo:
`reports/foundir_triple_composition_literature_reproduction_memo_2026-09-26.md`.

Exact next task: prepare a CPU-only R0 reproducibility package: hash-pin the
public OneRestore revision, build a source-disjoint matched synthetic
clean/single/double mini-manifest with held-out triples, validate
counterfactual-lattice parameter/order consistency and triple exclusion, add
unit tests, and emit a governance-bound design/readiness report. Stop before
any GPU launch for user/advisor review.

## Operating instruction — implementation fixes do not require repeated approval

The user explicitly directed that once an exact implementation fix is known,
apply it directly without asking for approval again. This applies to scoped
code/runtime compatibility fixes that do not change the scientific protocol,
data, model, optimizer, checkpoint-selection rule, or evaluation target. Record
the fix and verification evidence in this handoff. New scientific experiments
or protocol changes still require their normal authorization.

## 2026-09-26 route-aware 660-image integrated Restormer v2 TERMINAL COMPLETE

The user authorized the formal 660-image paper-table integration after the
Low-light+Haze Restormer experiment. A new route-aware, hash-bound artifact
replay completed normally from `2026-09-26T10:40:24Z` to `10:41:02Z` and
emitted `FOUNDIR_COMPOSITION_INTEGRATED_RESTORMER_V2_END status=0`.

The frozen router has 53 accepted `haze+low_light` routes, five
`low_light_fallback` routes, and two abstentions among the 60 Low-light+Haze
records. Only the 53 accepted routes use the new direct Restormer; fallback and
abstention behavior remains fail-closed. Consequently the deployed category
result is `21.168830/.199704`, not the capacity-only all-60 result
`22.215445/.182860`.

Terminal 660-image metrics:

- route-aware project: `25.647200 PSNR / .168062 LPIPS`;
- Released FoundIR Generalist: `24.757710/.176648`;
- Full FoundIR Generalist -> Specialist: `27.169232/.167938`;
- project minus Generalist: `+.889490 dB / -.008586 LPIPS`;
- project minus Full FoundIR: `-1.522032 dB / +.000124 LPIPS`.

The 300-image double aggregate is now `24.321537/.192313`, beating Full
FoundIR `24.184117/.194418` by `+.137420 dB / -.002105 LPIPS`. Freeze the
double aggregate. Triple Low-light+Blur+Noise remains unchanged at
`16.941122/.177332` versus Full FoundIR `33.677913/.031967` and is the next
paper-novelty diagnostic target.

This run performed no fresh neural inference for the 607 unchanged records.
It generated 660 new per-record audit sidecars by verifying and binding every
source sidecar and project output SHA-256, then replaying their immutable
metrics; this provenance must be disclosed in the paper. It avoids about 10.75
hours of redundant 8-GPU inference without changing the deterministic result.
All 660 sidecars are complete, there are zero temporary files, and official
FoundIR test access is false. Wall time was 38 seconds; GPU-hours were zero.

- config SHA-256:
  `c30581a701f6b08b6c66503b7b681bcc1f386ca43799a74d387367ab280ad6ad`;
- result SHA-256:
  `511940015fb3e3898f7da4ce7f31958dfb7f0bf9175062002de801cd2fcf7c1f`;
- terminal report:
  `reports/foundir_composition_integrated_restormer_v2_terminal_result_2026-09-26.md`.

Exact next task: return to the literature-driven paper novelty work. Analyze
the triple failure and reproduce the most relevant compositional-restoration
mechanism before proposing or launching a new scientific method. Do not
automatically add another category-specific specialist.

## 2026-09-26 Low-light+Haze direct Restormer v1 TERMINAL COMPLETE

The frozen `09Lowlight_Haze -> clean` Restormer experiment completed normally
after exact-resume from the immutable step-3000 checkpoint. The resumed run
started at `2026-09-26T07:08:48Z`, all eight ranks reported
`resume_step=3000`, and the runner emitted
`FOUNDIR_LOWLIGHT_HAZE_RESTORMER_V1_END status=0` at
`2026-09-26T08:37:16Z`. There was no new traceback, OOM, CUDA failure, or NCCL
warning after the resumed-run start marker.

Frozen development selection:

- step 6000: `21.732965 PSNR / .200751 LPIPS` on 128 source-disjoint,
  full-resolution development records;
- step 3000: `20.767505/.221818`, so step 6000 improves by
  `+.965460 dB / -.021067 LPIPS`;
- the frozen rule
  `maximize_worst_normalized_development_gate_margin_then_earlier_step`
  correctly selected step 6000;
- selected checkpoint:
  `checkpoints/foundir_lowlight_haze_restormer_v1/step_006000.pt`, SHA-256
  `a6f9523a6b9b44ab95d693c29038bb12378ba050b575104e62b6d9b5d1ef0cc3`;
- development result SHA-256:
  `7f6289c661c9cd9ecfc8ec4cdb6ed3a6ef329d915dde9bbe25691562c00d83f3`.

Frozen final 60-image comparison, which was not used for checkpoint selection:

| System | PSNR | LPIPS |
|---|---:|---:|
| Project Restormer step 6000 | 22.215445 | .182860 |
| Full FoundIR | 20.939309 | .159126 |
| Project minus Full FoundIR | +1.276136 dB | +.023734 |

The experiment substantially improves the previous project result
`13.112670/.251747` by `+9.102775 dB / -.068887 LPIPS`. It beats Full FoundIR
on PSNR but not LPIPS, so `beats_full_foundir_both_metrics=false`; the double
composition objective is not yet achieved on both metrics. This is a mixed
development/comparison experiment, not the official sealed singleton test, and
`official_foundir_test_accessed=false`.

Terminal artifacts and audit:

- final result:
  `checkpoints/foundir_lowlight_haze_restormer_v1/final_comparison/result.json`,
  SHA-256 `26266d6ff414919fab77b6c01126afb56c8dba6f43e154483fd75e796d9d78ae`;
- exactly 60 final PNGs and 60 per-record JSON sidecars, with no temporary
  files; record IDs match one-to-one;
- all 60 sidecars have `status=complete`; recomputed means match the terminal
  result, and all 60 recorded output SHA-256 values match their PNGs;
- resumed terminal wall time: 1 hour 28 minutes 28 seconds on 8xH200,
  approximately 11.80 GPU-hours;
- the earlier corrected run used 2 hours 27 minutes 49 seconds before external
  node/session termination; both full-run segments total 3 hours 56 minutes
  17 seconds of active 8-GPU time, approximately 31.50 GPU-hours, excluding
  short diagnostics and downtime;
- do not rerun this frozen v1 automatically. The exact next scientific task is
  to analyze and improve the remaining `.023734` LPIPS gap without using the
  final 60-image comparison for iterative checkpoint selection; any new
  experiment/protocol requires its normal authorization.

## 2026-09-26 Low-light+Haze NCCL recovery correction

The initial run and the first `NCCL_P2P_DISABLE=1` diagnostic both failed before
optimizer step 1. Full NCCL evidence identifies `transport/nvls.cc` with CUDA
error 401 while NVLS multicast remained enabled. P2P disabling does not disable
NVLS. The exact code-only correction is therefore `NCCL_NVLS_ENABLE=0` with P2P
retained, explicit CUDA `device_id` process-group binding, current-device CUDA
seeding, and a two-stage diagnostic (scalar all-reduce followed by Restormer DDP
forward/backward with no optimizer step). If that bounded diagnostic passes, the
same frozen run resumes automatically. No scientific protocol changes.

Implementation and CPU-side verification completed at 2026-09-26 UTC:

- `NCCL_NVLS_ENABLE=0`, `NCCL_P2P_DISABLE=0` are enforced by recovery;
- trainer, evaluator, and diagnostic explicitly bind `device_id=device`;
- trainer seeds the CPU default generator and current CUDA device only;
- diagnostic performs scalar all-reduce before constructing Restormer DDP;
- the tmux starter now enters recovery rather than bypassing it;
- verbose `NCCL_DEBUG=INFO,COLL` is scoped to the bounded diagnostic and is not
  intentionally inherited by subsequent full training;
- Python compilation, shell syntax, governance binding, frozen preflight, and
  launcher `--check` all pass;
- the current Codex shell has no `nvidia-smi`, so GPU execution cannot be
  initiated from this shell. On the existing 8xH200 shell, run exactly:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/recover_foundir_lowlight_haze_restormer_v1_8gpu.sh`.

Runtime update at 2026-09-26T03:38:07Z: the corrected diagnostic passed all
stages on 8xH200 and wrote
`checkpoints/foundir_lowlight_haze_restormer_v1/nccl_recovery/result.json`.
The unchanged frozen run then started automatically at 03:32:27Z and reached
step 150/6000 with finite loss, no new traceback, and 8-GPU utilization. The
old CUDA 401 lines earlier in the append-only log belong to the 03:00 failed
attempt. Observed 128-crop throughput is about 1.55 seconds/step. Because crops
increase to 192 at step 1500 and 256 at step 4000, the original 1.7--2.7 hour
estimate is not yet supported by runtime evidence; a cautious first projection
is roughly 6.5 hours for training plus milestone/final evaluations, to be
refined after the first 192-crop interval. The first recovered run inherited
diagnostic NCCL verbosity; the recovery script has been corrected so future
full runs will not inherit it. The current run remains scientifically valid.

Runtime update at 2026-09-26T07:05:20Z: that GPU node/session stopped abruptly
at 06:00:16Z after reporting step 5450/6000. The GPU telemetry stopped at the
same time, the runner lock is free, and there is no new traceback, OOM, NCCL
warning, or normal END marker; this is consistent with external node/session
termination rather than a model/code failure. The latest immutable resumable
milestone is step 3000 (`2d8baa0a...`); step 5450 was not checkpointed. Step
3000 development is `20.767505 PSNR / .221818 LPIPS`, improved from step 1000
`18.356467/.301158` but still short of the frozen `20.939309/.159126` gate.
Launcher/preflight checks pass and exact-resume is automatic. On a fresh
8xH200 shell run:
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/launch_foundir_lowlight_haze_restormer_v1_8gpu.sh`.
It resumes at step 3000; expected remaining wall time is about 1.4--1.8 hours
plus startup variance, approximately 11.2--14.4 GPU-hours.

## Historical readiness record — superseded by Restormer v1 terminal completion

The user explicitly authorized `yes, restormer, execute`. A bounded direct
`09Lowlight_Haze -> clean` Restormer workflow is frozen and CPU-ready. It uses
the public task-native Restormer deraining checkpoint as initialization and no
released FoundIR weights. Every source group represented in the frozen
660-image comparison is excluded from training and development selection.

- training: 75,370 records / 7,514 source groups, 6,000 steps, global batch 32;
- development selection: 128 source-disjoint full-resolution records at steps
  1,000, 3,000, and 6,000;
- final gate: one frozen 60-image `09Lowlight_Haze` comparison against Full
  FoundIR `20.939309/.159126`, not used for checkpoint selection;
- config SHA-256:
  `d60d67ba2a285fa0cb0691ac04f9630e2dc3129f1bf95d7563934ce777235cd3`;
- manifest SHA-256:
  `afa0e22b27c9c9773be67454a36c80bfcc7fd3e26f97b91be7577d17bd8b8a0e`;
- readiness:
  `reports/foundir_lowlight_haze_restormer_v1_readiness_2026-09-26.md`;
- governance hold `lowlight_haze_direct_restormer_v1_design-e3c715f7298fa7c2`
  is resolved by the user's scope-limited decision;
- CPU preflight, Python compilation, shell syntax, source isolation, immutable
  hashes, H100/H200 guards, and launcher `--check` pass.

No GPU is visible in the current Codex shell, so the job has not launched and
no training result exists. Exact next action is one external launch on a
homogeneous 8xH100 or 8xH200 node. Total estimate: H200 100--160 minutes
(center 125; 13.3--21.3 GPU-hours), or H100 130--200 minutes (center 160;
17.3--26.7 GPU-hours). The job includes training, all milestone evaluations,
selection, and final comparison. Do not launch a duplicate or automatically
start Low-light+Blur afterward.

## 2026-09-25 official SOW600 Released FoundIR comparison TERMINAL COMPLETE

The user-authorized direct singleton comparison completed normally on the same
600 official SOW pairs. The frozen terminal project v8 beats Full FoundIR on
both PSNR and LPIPS in every category:

| Category | Project PSNR / LPIPS | Full FoundIR PSNR / LPIPS | Project margin |
|---|---:|---:|---:|
| Blur | 28.092635 / .144089 | 25.577556 / .180644 | +2.515080 dB / -.036555 |
| Noise | 40.048323 / .056601 | 39.807950 / .064671 | +.240373 dB / -.008070 |
| Haze | 22.677420 / .110437 | 19.084373 / .224938 | +3.593046 dB / -.114500 |
| Rain | 37.809175 / .052901 | 34.293876 / .079429 | +3.515298 dB / -.026529 |
| Low-light | 19.322459 / .148799 | 18.240826 / .238644 | +1.081634 dB / -.089846 |

The 600-image weighted means are Project `29.168753/.103485`, Released
Generalist `25.821272/.169251`, and Full FoundIR `26.626220/.164044`.
Project minus Full FoundIR is `+2.542533 dB/-.060559`.

- result:
  `checkpoints/foundir_official_sow600_released_full_pipeline_v1/result.json`,
  SHA-256 `afdef408521989bc449497c637403642612a9687e5004878f4687a765532b9d6`;
- terminal report:
  `reports/foundir_official_sow600_released_full_pipeline_terminal_result_2026-09-25.md`;
- artifacts: 600 Generalist PNGs/metadata, 250 Specialist PNGs/metadata, 600
  metric sidecars;
- independent audit: pass;
- runtime: `2026-09-25T01:19:59Z` to `01:38:23Z`, 18m24s on 8xH200,
  approximately 2.45 GPU-hours, status 0;
- training/model updates: none.

The singleton objective is already achieved; do not reopen singleton training
merely to beat FoundIR. Per the user's ordered plan, the next research stage is
double compositions, followed by triple composition only after the double stage
is frozen. Keep official sealed singleton data metric-only and outside all
training and iterative checkpoint selection.

## Historical readiness record — superseded by terminal completion above

The user explicitly authorized a direct singleton comparison on the same 600
official SOW pairs: terminal project v8 versus Released FoundIR Generalist and
the documented Generalist -> Specialist pipeline. This is evaluation only;
official pixels/GT remain forbidden for training or gradient updates.

The implementation and full CPU preflight are complete. Preflight verified all
600 manifest rows, 1,200 LQ/GT hashes and dimensions, released checkpoint
hashes, canonical commit `5003e1e` project metrics, `pyiqa` 0.1.16, LPIPS
weights, governance, and syntax. No GPU is visible in the current Codex shell,
so the run has not launched and the output directory is empty.

- config:
  `configs/foundir_official_sow600_released_full_pipeline_v1.json`, SHA-256
  `3dff5bb4341d3f97c15470fe2d8bfaab7c9b22af051c13171d38fa986224802d`;
- readiness:
  `reports/foundir_official_sow600_released_full_pipeline_readiness_2026-09-25.md`;
- expected scope: 600 Generalist outputs, 200 Weather Specialist outputs, 50
  Low-light Specialist outputs, and 600 metric sidecars;
- estimate: 24--38 minutes on 8xH200 (center 30), or 30--48 minutes on 8xH100
  (center 38).

Exact next action is one launch on a homogeneous 8-GPU node, then monitor to
terminal completion, independently audit artifacts, write the terminal report,
and freeze the per-category singleton gaps. Per user direction, subsequent
quality work must address singleton categories first, double compositions only
after singleton success, and triple composition last.

This is the current FoundIR operational handoff. It supersedes
`reports/foundir_current_handoff_2026-09-21.md` for project status and the
next action. Historical results remain immutable evidence and must not be
rewritten as the current state.

## 2026-09-24 full FoundIR composition comparison TERMINAL COMPLETE

The follow-up 660-image full-resolution development comparison is now complete
for the complete released FoundIR inference pipeline. The earlier 2026-09-23
result ran the released Generalist only and must be labeled accordingly. The
new run reused those immutable Generalist PNGs, then applied the documented
Weather specialist to 120 records and Low-light specialist to 240 records;
the other 300 records retained Generalist output.

Overall 660-image means are:

- project: `24.914822 PSNR / .172793 LPIPS`;
- Released FoundIR Generalist: `24.757710 / .176648`;
- Full FoundIR Generalist -> Specialist: `27.169232 / .167938`;
- project minus Full FoundIR: `-2.254410 dB / +.004855 LPIPS`.

Thus the project slightly beats the released Generalist-only aggregate but
does not beat the complete FoundIR pipeline overall. Triple
Low-light+Blur+Noise is the dominant gap: project `16.941122/.177332` versus
Full FoundIR `33.677913/.031967`.

Terminal identities:

- result:
  `checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json`;
- result SHA-256:
  `ab4a1b0330c732432072917dca4ba74012bf7efe166e347e43aee4461a1e09e6`;
- config SHA-256:
  `c1dc876671622fd4d0541b52ca28a6cf95c036f181249d275ba370c98d835760`;
- report:
  `reports/foundir_composition_full_pipeline_terminal_result_2026-09-24.md`;
- artifacts: 360 specialist PNGs, 360 specialist metadata sidecars, 660 metric
  sidecars;
- runtime: `2026-09-24T23:11:09Z` to `23:16:22Z`, 5m13s on 8xH200,
  approximately .70 GPU-hours;
- model updates: none; official FoundIR test accessed: false;
- independent terminal audit: pass.

This remains `DIAGNOSTIC ONLY / 非主线`. It does not alter the already-terminal
official SOW600 v8 pass. Exact next action is paper/architecture analysis of
the observed mixed-degradation gap; do not restart this comparison or reopen
official singleton training.

## Current status in one paragraph

The five project-trained single-type systems already pass their complete
source-disjoint development mean SOW gates. In particular, Blur is not an
unfinished development task: the frozen Blur FLUX.2 step-10000 scalar-residual
system reaches `26.559141 PSNR / .151993 LPIPS`, and the fail-closed final
eleven-class router integrated over all 2,595 Blur development records reaches
`26.614726 / .156974`; both are above the Blur SOW thresholds `25.9 / .18`.
The user confirms that the final official sealed evaluation is also complete
and all five SOW types are well above threshold. The final all-five official
result artifact, exact metrics, and SHA-256 are not discoverable in the current
workspace snapshot as of `2026-09-23T05:28Z`; the latest conventional
`official_result.json` files visible here are historical v1--v6 intermediate
rounds and must not be presented as the current final outcome. Bind the final
official artifact here as soon as its path is supplied or synchronized.

## Scope and system identity

- Workstream: VLM Guided All-in-One restoration only; the separate bright/color
  model workstream remains out of scope.
- Categories: Blur, Noise, Haze, Rain, and Low-light plus the six admitted
  mixed categories; JPEG remains excluded.
- Deployed structure: Qwen degradation inference, fail-closed routing/fallback,
  and frozen project-trained FLUX.2 or Restormer experts. It is a VLM-routed
  multi-expert system, not one monolithic restoration checkpoint.
- The official 600-pair scope was authorized on 2026-09-21. Authorization:
  `reports/advisor_governance/foundir_official_sow600_qwen_authorization_2026-09-21.json`,
  SHA-256 `001c70b975e6c7b3cc1683e1b9ac82e0a62d73f06020290c415ad308132fefc0`.
- The authorization permits documented retraining, tuning, and retesting, but
  official images/GT remain forbidden as training examples or gradient targets.

## Frozen source-disjoint development evidence

Registry:
`reports/foundir_five_type_source_disjoint_development_registry_v1.json`.

| Type | Frozen project system | Records / groups | PSNR | LPIPS | Mean SOW |
|---|---|---:|---:|---:|---:|
| Blur | FLUX.2 step-10000, `LQ + .75*(FLUX-LQ)` | 2,595 / 165 | 26.559141 | .151993 | pass |
| Noise | FLUX.2 step-400, `LQ + .30*(FLUX-LQ)` | 1,859 / 34 | 40.150553 | .068880 | pass |
| Haze | FLUX.2 step-3000 | 1,580 / 158 | 23.311888 | .101155 | pass |
| Rain | project-trained Restormer step-5000 | 835 / 117 | 42.284347 | .047452 | pass |
| Low-light | FLUX.2 step-400, `LQ + .925*(FLUX-LQ)` | 1,400 / 21 | 19.779135 | .154058 | pass |

Blur's final-router record-weighted development evidence is
`checkpoints/foundir_blur_record_weighted_final_eleven_router_v2/result.json`:

- 2,223 correct routes, 372 safe abstentions, zero wrong routes;
- `26.614726 PSNR / .156974 LPIPS`;
- PSNR margin `+.714726 dB / +2.7596%`;
- LPIPS headroom `.023026 / 12.7920%`.

Do not call Blur unresolved based on the older 128-record source-group-balanced
rehearsal. That rehearsal intentionally overweighted rare hard groups and is a
tail/distribution diagnostic, not the complete record-weighted Blur result.

## Work completed after the stale 2026-09-21 handoff snapshot

### Official authorization and intermediate rounds

The 2026-09-21 handoff stopped while the first Noise Restormer experiment was
still described as pending. The following work subsequently completed:

1. Official SOW600 access was explicitly authorized for the Qwen-routed five
   SOW categories, 600 pairs total.
2. Historical official v1 established the initial full-resolution baseline.
3. Historical v2 froze improved Rain and Haze paths; both official categories
   passed.
4. The first Noise Restormer attempt (historical v3) failed official Noise.
5. A project-trained Noise MSE Restormer reached `42.719396/.066883` on all
   1,859 development records. A 250-step MSE+LPIPS continuation reached
   `42.629757/.056313` on the same development set.
6. Historical official v4 passed Noise PSNR but narrowly failed LPIPS:
   `39.511948/.105248`.
7. Historical official v5 passed Noise at `39.467855/.094125`. At that point
   Noise, Haze, Rain, and Low-light all passed that intermediate official round.
8. Historical official v6 was a Blur-motion audit. It returned
   `21.831774/.389771` for that candidate and failed. It is preserved as a
   rejected intermediate candidate, not the current final sealed result.

Historical official result SHAs:

| Round | Result SHA-256 |
|---|---|
| v1 | `9a17808e6982f363e92e0423681715b5d879847f0d1b1e941666a918bdb2fdb0` |
| v2 | `b238430b24b77cc6cae2ff3eab3a26fd07c5e61c39085a960285a70485f58ebe` |
| v3 | `cea5206525414eab43c71f503960eb8ad6a6b5643fdf480656a0ceec1940eddd` |
| v4 | `8a0221d6a8257b9e6f00a3699432c3c97304ec649d61fa73bc514c9a5f4d8fb6` |
| v5 | `61948d5df57e85cd029a638779323a72e9ce1e45e5548a0cac22a0e4ee14ca72` |
| v6 | `22256877a8fa0e0fd4b8582c6a4c3447d66cb8e432cfef80f7dc46689f15cd61` |

### Blur diagnostics completed after v6

- The training-only severity audit covered 71,393 Blur records represented by
  2,782 source groups. Representative mean input PSNR was `22.373173`, and
  2,021 groups fell inside the disclosed official input-PSNR range. Evidence:
  `reports/foundir_blur_training_severity_audit_v1.json`, SHA-256
  `4478f834bcadde8afcf23babf6b82f05246c9487f90e81d3a6319442e3956c61`.
- The 2,000-step heavy motion MSE+LPIPS Restormer gate failed at
  `22.200263/.229873` on 44 source-disjoint development records.
- The external FFTFormer diagnostic failed qualification at
  `19.949758/.188968` on the 150 official Blur records and is non-deployable.
- These rejected diagnostics do not invalidate the already-passing complete
  source-disjoint Blur development system or the user-confirmed final sealed
  five-type pass.

## Live 660-image whole-image composition comparison

On 2026-09-22/23 a separate read-only development comparison was prepared and
launched. This is a composition/whole-image characterization run, not a missing
single-type SOW gate and not another official sealed evaluation.

Frozen identities:

- config: `configs/foundir_composition_whole_image_v1.json`, SHA-256
  `b9b04fc4242524cfa44a2f2ac038bd73266d4bf14809dad20ce2b5929a6de0cb`;
- manifest: `manifests/foundir_composition_whole_image_v1.csv`, SHA-256
  `a94128195ffd219f7e94b44c1ec98291c75e727f17e2c3af88cf1c2f9b6282a9`;
- 660 validation images, 60 per each of eleven categories;
- compare the frozen routed project system against released FoundIR on whole
  images with tiled inference;
- no model update and no official-test access.

Verified live snapshot at `2026-09-23T05:28Z`:

- one homogeneous 8xH200 instance;
- all eight GPUs at 100% utilization, about 19,046 MiB per GPU;
- complete routed stage counts: Rain 111, Blur+Noise 53, Noise 137;
- Haze in progress with 105 immutable stage PNGs written;
- no traceback, OOM, NCCL error, terminal result, or end marker;
- final `checkpoints/foundir_composition_whole_image_v1/result.json` does not
  exist yet.

Monitor only:

```bash
tail -f logs/foundir_composition_whole_image_v1.log
```

Do not restart or duplicate the job merely because its remote PID is invisible
from the Codex shell. Advancing telemetry, log records, and immutable PNGs are
authoritative liveness evidence.

## Final official sealed result binding still required

User-authoritative current status: **all five official sealed SOW types pass
well above threshold**. Do not regress the handoff to v6 or describe Blur as
unfinished.

The following provenance fields still need to be copied from the final result
artifact once it is visible in this workspace:

- exact result path and SHA-256;
- exact five PSNR/LPIPS pairs and percentage margins;
- frozen router/config/checkpoint identities;
- run start/end, hardware, and total wall time;
- output/sidecar counts and terminal marker;
- whether this was one run or a later version after historical v6.

Until these are bound, cite the user's confirmation for the terminal sealed
status and cite the verified development artifacts for exact Blur numbers.
Never substitute the failed historical v6 JSON as the final result.

## Exact next action

1. Continue monitoring the already-running whole-image composition comparison;
   do not launch another GPU job.
2. At terminal completion, verify all 660 project outputs, all 660 released
   FoundIR outputs, all metric sidecars, `result.json`, hashes, and end marker.
3. Write its terminal report and update this handoff with actual timing and
   metrics. Treat it as composition/robustness evidence, not as reopening the
   already-passed five-type sealed result.
4. Locate or synchronize the final all-five official sealed result, then bind
   its exact metrics/path/SHA in this handoff.
5. After both records are complete, freeze the deliverable package and prepare
   the reproducibility/technical report. Do not start another Blur/Noise model
   search merely to enlarge already-passing margins.

## Compute-sizing rule retained

Choose GPU count from measured workload and end-to-end wall time, not available
inventory. Prefer one tested 8-GPU node for jobs under roughly one to two hours;
use embarrassingly parallel multi-node inference only when it materially saves
wall time. Preserve global batch, optimizer steps, LR semantics, data ordering,
and checkpoint comparability. Every future GPU command requires a total
end-to-end wall-time estimate before launch.

## Potential SOTA follow-up recorded on 2026-10-12

This is deferred planning, not a new active launch. It does not supersede the
Generalist-first gate: finish `real_generalist_4node_mixed_hopper_v1`, select
raw versus EMA only from frozen validation evidence, and run exactly one
matched 660-image Generalist-versus-Generalist evaluation. Do not tune from the
660 result. The project is not yet demonstrably SOTA because pass-1 metrics are
crop diagnostics and Double LPIPS remains the main unresolved gate risk.

Only after EBT beats released FoundIR Generalist in both PSNR and LPIPS for
Singleton, Double, and Triple should this public-benchmark plan be activated:

| Experiment block | Purpose | New training? |
|---|---|---:|
| Frozen 660 whole-image gate | Establish matched FoundIR Generalist result | No |
| CDD-11 matched train/eval | Public composite Singleton/Double/Triple evidence | Yes |
| Standard 5D matched train/eval | SOTS/Rain100L/BSD68/GoPro/LOL comparability | Yes |
| No-VLM ablation | Isolate VLM conditioning | Yes |
| No-energy/direct-restoration ablation | Isolate learned iterative energy | Yes |
| Fixed steps versus adaptive TTS | Quality/compute curve | No |
| Self-verification on/off | Isolate verification benefit and cost | No |
| Efficiency evaluation | Params/FLOPs/memory/latency/descent steps | No |

The minimum package is eight blocks, four new training runs, and roughly six
to eight evaluation jobs. A broader AiOIR SOTA claim additionally requires a
separately trained standard 3D setting, real-world benchmarks, and
robustness/OOD evidence, for roughly 12--15 blocks and six to eight training
runs.

Main CDD-11 targets (PSNR/SSIM) are AllRestorer `33.72/.9436`, DAR-Net
`29.73/.888`, ClearAIR `29.34/.886`, MoCE-IR-S `29.05/.881`, and OneRestore
`28.47/.878`. AllRestorer is the strongest published claim even though a
reproducible public implementation was not verified; unrestricted SOTA wording
cannot omit it. Main standard-5D average targets include D3Net `32.36/.901`,
DAR-Net `31.02/.921`, DPC-Net `31.00/.923`, DFPIR `30.64/.913`, MoCE-IR
`30.58/.919`, and CyclicPrompt `30.52/.935`. FoundIR-v2, VLU-Net, InstructIR,
Perceive-IR, ClearAIR, DFPIR, and EBT are also mandatory related-work
comparisons for the claimed VLM-conditioned learned-energy/adaptive-TTS/self-
verification intersection. Do not compare unmatched metrics or imply that one
method leads every task.

Measured planning baseline: the current 30,699-step schedule is approximately
8.97 wall hours per pass, hence about 27 wall hours and 864 allocated
heterogeneous GPU-hours for three passes on 32 GPUs. Four mechanical repeats
would be an unapproved ceiling of about 108 serial wall hours and 3,456
GPU-hours. Before any CDD-11, 5D, or ablation run, perform a 500-step protocol-
specific throughput qualification, targeted at no more than 0.5 wall hour and
4 allocated GPU-hours on eight GPUs, and derive the real budget from matched
epochs and unique-sample exposure.

The intended defensible claim is narrowly scoped to composite/mixed Generalist
SOTA or best matched average. Winning the project-specific 660 comparison is a
necessary first gate but is not, by itself, universal All-in-One Image
Restoration SOTA. Specialist work remains paused.
