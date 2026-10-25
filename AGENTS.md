# Project agent instructions

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

## Latest Rain qualification update — 2026-10-25

The 25-step FP32 precision qualification completed successfully at
2026-10-25T20:39:00Z on 8xH100, status 0. Development improved from
32.08470176034529/.0888220098754391 to
32.61941190970921/.08720199926523492, still failing the 34.2/.08 gate.
The user authorized the independent 300-step fit diagnostic, not full-data
training. Initialize from the original step-2,000 LoRA with fresh optimizer,
never the qualification checkpoint. Command on the GPU node:
`bash smoke/start_foundir_rain_flux2_precision_fit_v1_tmux.sh fit fit_v1`.
Estimate 15-20 wall minutes / 2.0-2.67 allocated GPU-hours. Fit has no launch
log at this update; this Codex environment has no GPU. Keep Rain gate first,
image-only MoE second, VLM router third; no pixel refiner.
See `reports/foundir_rain_precision_qualification_2026-10-25.md` for evidence.

## Active workspace and runtime

- The sole active project root is `/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang`.
- Use `/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3` for all Python work.
- Preserve all dirty and untracked project files. Do not reset, clean, or overwrite unrelated work.
- Keep the old HDD workspace untouched.
- Never use official, sealed, parent-test, or internal-test pixels for training, hyperparameter selection, checkpoint selection, or early stopping.

## ACTIVE: Rain FLUX precision repair and training-fit diagnosis — 2026-10-25

The user rejected the proposed small pixel refiner and authorized the
precision/fit diagnosis. This supersedes the refiner suggestion below.
Keep the sequence Rain DiT >=34.2 PSNR / <=.08 LPIPS, then working image-only
MoE, then VLM tokens. Do not train the observation-anchored pixel head.

The prior run stored trainable LoRA weights and both Adam moments in BF16
at learning rate 1e-5. At step 3,000, 64.2164% of LoRA A entries were exactly
unchanged from its parent. This is evidence to investigate update rounding,
not proof that it explains the entire failure. Training kept the encoded
observation FP32 for subtraction while inference cast it to BF16 first.

The new isolated implementation is configured by
`configs/foundir_rain_flux2_precision_fit_v1.json`: FP32 LoRA parameters and
Adam states, BF16 CUDA compute, and one shared FP32 endpoint subtraction in
training/evaluation. Start independently from the hash-bound best Rain
direct-endpoint step-2,000 LoRA with a fresh optimizer. Retain the eight
parent loss terms and weights. Use 32 deterministic training source groups
with fixed 512 crops; evaluate the exact same training inputs and the
unchanged 100-source development gate at 512. The initial evaluation also
reports the legacy subtraction path on the same checkpoint.

Run a 25-step qualification first on one homogeneous 8xH100/H200 node. Its
checkpoint cannot seed the independent 300-step fit diagnostic. Fit requires
a passing qualification with matching config, source and sample hashes.
Qualification/fit milestones are [0,25]/[0,100,300]. No automatic full-data
training follows. Passing the 32-image numeric target is evidence of fitting
ability only; development PSNR/LPIPS remain separate. This is not a matched
long-training BF16-versus-FP32 ablation and cannot attribute all gains to
precision alone. Both launchers preserve existing run directories/logs.
Read `reports/foundir_rain_flux2_precision_fit_v1_readiness_2026-10-25.md`.

## ACTIVE: Rain DiT gate, then image-only MoE, then VLM router — 2026-10-25

This user decision supersedes the older MoE-first scheduling text below. The
order is strict: (1) repair the Rain FLUX/DiT expert until it passes the frozen
Rain development gate (`PSNR >=34.2`, `LPIPS <=0.08`); (2) build and validate
an ordinary working image-only restoration DiT-MoE using successful experts;
(3) only then add pooled VLM output tokens through a small projector to
adjust that same router. Blur DiT has already passed its SOW gate; do not
repeat Blur training. The failed 1,000-step image-only MoE is not a usable
baseline, and its VLM-assisted arm must not be launched.

The full-data direct-endpoint Rain FLUX formal run completed on 2026-10-21 at
3,000/3,000 steps (8xH100, `6.470578311735836` wall hours,
`51.76462649388669` allocated GPU-hours). Its frozen 100-record development
metrics were `31.900863136522702/.08921627756208181`, failing both gates.
The best predeclared checkpoint was step 2,000 at
`32.085315339143946/.08882607079111039`; 3,000 regressed. Do not repeat or
extend the same direct-endpoint recipe. On the 3,000-step result, all 24
records with input PSNR >=34 dB lost PSNR after restoration; 14 of the other
76 records also lost PSNR. A no-training, fixed 0.75 blend of input and the
step-2,000 output reaches only about 32.545 dB PSNR on saved development
triptychs. Even a non-deployable GT-only best blend per image reaches only
33.444 dB. These are quantized-export exploratory calculations, not official
model results. Simple strength control cannot repair the gap. Diagnose the
Rain action/representation and verify a distinct fix before another GPU run.
See `reports/foundir_rain_dit_gate_priority_and_failure_audit_2026-10-25.md`.
An observation-anchored correction head now exists and passes three CPU unit
tests, but it has no Rain trainer or GPU result. Do not call the gate passed.

## ACTIVE: advisor-approved minimal VLM-token-to-MoE-router pilot — 2026-10-25

This section supersedes every older Rain, EBT, energy, specialist, and
explicit-VLM scheduling priority below. The user reported that the advisor
approved the minimal architecture and explicitly authorized execution. Do not
restart an older branch unless the user explicitly returns to it.

Implement and test only the minimal advisor proposal:

1. start from one ordinary restoration DiT-MoE with one expert bank;
2. retain its image-feature router and restoration experts;
3. pool frozen VLM output tokens/features, pass them through one small MLP,
   and use the result to adjust the original router's expert logits;
4. select/weight the same MoE experts and combine their outputs normally.

There must not be a second MoE, a bank of LoRA experts, EBT/energy refinement,
atomic-token composition, counterfactual routing, hard diagnosis priors, or
block/timestep scheduling in this first experiment. The matched comparison is
image-only routing versus the identical MoE with the VLM score branch enabled.
Use the same backbone, expert bank, data, optimization, and inference budget.

CPU/static implementation and a training-only data preflight are complete.
The frozen 25-step eight-GPU qualification is defined by
`configs/foundir_vlm_token_moe_minimal_qualification_v1.json`. Run the
`image_only` and `vlm_assisted` arms separately; neither qualification may seed
formal training or auto-launch later work. Keep official, sealed, parent-test,
and internal-test pixels out of development. Report wall-clock hours and
allocated GPU-hours for every GPU run.

The `image_only` qualification completed successfully on 8xH100 at
2026-10-25T15:38:24Z. It ran 25/25 steps in `.0563902` wall-clock hours and
`.451122` allocated GPU-hours, with finite training, no OOM/NCCL/runtime
failure, and `12,653,083,136` bytes peak allocated memory per rank. The VLM
logit adjustment was exactly zero, expert weights summed to one within
`1.1921e-7`, and the checkpoint hash was verified. This is engineering
qualification only, not a restoration-quality result, and its checkpoint may
not seed formal training. The next active action is the fresh matched
`vlm_assisted` 25-step qualification; do not launch formal training yet.

The independent `vlm_assisted` qualification also completed successfully on
8xH100 at 2026-10-25T15:53:30Z. It ran 25/25 steps in `.0460070` wall-clock
hours and `.368056` allocated GPU-hours, with no OOM/NCCL/runtime failure and
`12,669,009,408` bytes peak allocated memory per rank. Its VLM branch was
active: maximum logit adjustment was `.75` and the output-head L1 update was
`.78411865234375`. Expert weights remained normalized within `1.1921e-7`.
Both engineering arms have therefore passed. Their last-five training losses
(`1.1534720` image-only versus `1.1534493` VLM-assisted) are not restoration
metrics and must not be presented as scientific improvement. Neither
qualification checkpoint may seed formal work. The next step is to freeze a
matched scientific train/evaluation protocol before any longer GPU run.

That matched protocol is now frozen and CPU-qualified in
`configs/foundir_vlm_token_moe_matched_formal_v1.json`. Each arm starts
independently from the same pretrained FLUX.2 weights, trains for 1,000
optimizer steps (32,000 fit presentations, about 14.2 fit-set exposures), and
uses the same data/order/objective/compute. Calibration is evaluated at steps
500 and 1,000; the source-disjoint 450-record holdout is read exactly once at
the fixed final step. Metrics are PSNR, AlexNet LPIPS, per-degradation results,
and routing traces. Four protocol unit tests and both arm-specific CPU/data
preflights pass. No protected data was accessed. Run the arms separately with
`smoke/run_foundir_vlm_token_moe_matched_formal_v1_8gpu.sh`; start with
`image_only`. The launcher is resume-safe within the same arm but forbids all
qualification checkpoints and never launches the other arm automatically.
Estimated cost per arm is `1.3-2.5` wall hours and `10.4-20.0` allocated
GPU-hours. Do not claim scientific success until both final holdouts have been
paired by `smoke/compare_foundir_vlm_token_moe_matched_formal_v1.py`.

The `image_only` formal arm completed normally at 2026-10-25T16:53:00Z:
1,000/1,000 steps, `.643642` wall hours, `5.14914` allocated H100 GPU-hours,
finite training, and no OOM/NCCL/runtime failure. It failed the scientific
prerequisite that the ordinary MoE be usable. On the fixed 450-record holdout
it reached only `15.634318159133873` PSNR and `.34884399509264363` LPIPS,
versus the unprocessed input at `23.71294072670594/.3023991875624698`.
Routing collapsed almost completely onto expert 1: mean weights are about
`5.8873e-7/0.99999941`, with entropy `8.6261e-6`. Step 500 showed the same
failure and step 1,000 did not repair it. This is a valid terminal result for
the current image-only construction, not a compute/runtime failure.

Do not call this a working restoration MoE or use it as paper evidence. Do not
automatically launch `vlm_assisted`: an additive VLM score cannot by itself
repair a restoration backbone/expert bank whose image-only output is already
catastrophically below the input. The next design must begin from an actually
working restoration model/MoE and explicitly prevent router collapse; obtain
the user's decision before replacing the frozen matched protocol.

## MANDATORY: regular focused GitHub progress tracking

GitHub is the durable project history. Do not allow material work to remain
only in the dirty local workspace for long periods. The following is a hard
operating rule for all later agents, not an optional cleanup suggestion.

Create a focused commit and push it to `origin/main` after each coherent,
material milestone, including:

- a new or materially changed model, training script, evaluation script,
  configuration, launcher, or reproducibility contract;
- a code failure that required a substantive fix once the corrected code has
  passed the available static/CPU/GPU qualification;
- an advisor/user research-priority decision that changes the active plan;
- a completed qualification, pass-level validation, selected checkpoint, or
  terminal experiment result after its report/handoff has been updated;
- before leaving a long-running experiment unattended, provided its launch
  code/config and current handoff form a coherent reproducible snapshot.

For an active run, do not commit every log line or optimizer step. Commit the
launch-ready code before or immediately after launch, then commit again at
meaningful validated milestones and terminal completion. A running job is not
an excuse to leave its source/config untracked.

Every push must follow this safety procedure:

1. Run `git fetch origin` and inspect local/remote divergence.
2. Preserve the dirty workspace. Never use `git reset --hard`, `git clean`, or
   broad checkout commands to prepare a push.
3. Stage an explicit allowlist of relevant source/config/report files. Never
   use `git add -A`, `git add .`, or a directory-wide add in this repository.
4. Exclude checkpoints, model weights, raw/generated images, datasets, caches,
   telemetry, live logs, temporary files, and large generated manifests. Keep
   data identity through small metadata summaries and SHA-256 bindings.
5. Scan the staged files for credentials, access tokens, private keys, and
   unintended machine-specific secrets. Never commit the previously exposed
   GitHub token or any replacement credential.
6. Run syntax/JSON checks and `git diff --cached --check`; inspect the staged
   file list and size before committing.
7. If the main workspace is heavily dirty or an untracked file conflicts with
   an incoming remote path, create an isolated temporary Git worktree from
   `origin/main`, copy only the explicit allowlist, commit there, and push from
   there. Do not disturb the live training workspace.
8. Push the focused commit to `origin/main`, verify the remote commit hash, and
   report the commit hash and included scope to the user.

If authentication, networking, a remote conflict, or a required secret scan
blocks the push, report that immediately and preserve the ready commit. Do not
silently defer Git tracking across later milestones. Do not claim the work is
backed up until the remote hash has been verified.

## TERMINAL: residual-basis formal run complete — 2026-10-17

The user explicitly authorized addressing the oracle-proven lack of corrective
information in the V2 direct residual. The next experiment is not another
energy/gate run. Freeze the existing Noise proposal and train exactly one
restoration-native residual-basis model on the same 54,585-record training
split. It receives the degraded image, proposal, and their difference and
produces two independent directions: fidelity trained explicitly with MSE and
perception trained with stronger LPIPS while retaining MSE/local no-harm.

This is a causal prerequisite for later energy work. The fresh 25-step
eight-GPU engineering qualification completed successfully on `8 x H100
80GB` with status `0`, finite training, checkpoint and matched 256-record
development validation. It consumed about `.0217` wall-clock hours and `.173`
allocated GPU-hours. At step 25 the fidelity/perceptual heads improved PSNR by
about `+.0116/+.0114 dB` but worsened LPIPS by about `.00020`; this is only an
engineering check, not scientific evidence or a reason to alter the frozen
recipe.

The next action is a fresh formal run ID for the predeclared five-pass/8,530-
step schedule, validating after every effective pass. Never initialize it from
`qualification_v1`. Do not restart V2, train an energy/gate, use protected
data, or claim novelty success until the new residual basis itself
demonstrates adequate headroom on the fixed 256-record development protocol.
Formal runtime is conservatively estimated at `3.5–6.0` wall-clock hours on
one eight-GPU node (`28–48` allocated GPU-hours); update this estimate from
the measured pass-1 runtime.
Implementation contract:
`configs/foundir_proposal_residual_basis_noise_v1.json` and
`reports/foundir_proposal_residual_basis_noise_v1_readiness_2026-10-17.md`.

`formal1` is now live on `8 x H100 80GB` and passed its first effective pass
at step `1,706`. Do not restart, duplicate, retune, or interrupt it. The
fidelity head reached `43.16839203238487/.0400125195938017`, simultaneously
improving over the frozen proposal by `+.044231854379177094 dB` and
`-.010349548166232125` LPIPS. This already exceeds the old direct residual's
GT-only oracle in both metrics, so the restoration-native basis has resolved
the earlier lack-of-information bottleneck at least for LPIPS/headroom. It has
not passed the full `+.30 dB/-.005 LPIPS` gate because PSNR gain is still only
`.0442 dB`. Continue the frozen five-pass run and judge the next validations
without changing the recipe. Full pass-1 record:
`reports/foundir_proposal_residual_basis_noise_v1_pass1_2026-10-17.md`.

Pass 2 at step `3,412` improved further. Fidelity is now
`43.22963780909777/.03339703518213355`, or `+.10547763109207153 dB` and
`-.016965032577900274` LPIPS versus proposal. This lands inside the
predeclared `+.09–.12 dB` pass-2 hopeful trend range. It has not passed the
`+.30 dB` PSNR gate, but the evidence now supports completing the unchanged
five-pass schedule. Do not extrapolate linearly or retune from this interim
result. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass2_2026-10-17.md`.

Pass 3 at step `5,118` strengthened the result materially. Fidelity reached
`43.354813657701015/.030396450317311974`, equal to
`+.23065347969532013 dB` and `-.01996561744272185` LPIPS versus proposal. It
now needs only `.06934652030467986 dB` more PSNR to pass the joint target, and
the pass-2 to pass-3 PSNR increment was `+.1251758486032486 dB`. Continue the
unchanged run through passes 4 and 5; do not stop early or alter the schedule.
Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass3_2026-10-17.md`.

Pass 4 at step `6,824` did not cross the gate. Fidelity is
`43.365845650434494/.02899720575715037`, or `+.24168547242879868 dB` and
`-.021364862002883456` LPIPS. The PSNR gain increased only
`+.011031992733478546 dB` from pass 3 and remains `.05831452757120131 dB`
short, showing clear late-stage saturation after the 256-crop transition and
with the learning rate already near `6e-6`. Complete the frozen fifth pass,
but do not claim likely gate success or modify the live recipe. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass4_2026-10-17.md`.

`formal1` completed successfully at step `8,530` with launcher status `0`.
Final fidelity is `43.39715355634689/.028955043449059303`, equal to
`+.27299337834119797 dB` and `-.02140702431097452` LPIPS versus proposal. It
therefore misses the PSNR gate by only `.02700662165880202 dB` while passing
the LPIPS gate widely. The joint gate is terminally **not passed** for this
run. Do not resume or silently extend it. The next authorized reasoning step
is a no-training GT-only amplitude/local feasibility audit of the frozen
step-8,530 fidelity residual; no energy/gate or new training may start before
that evidence is reviewed. Terminal report:
`reports/foundir_proposal_residual_basis_noise_v1_terminal_2026-10-17.md`.

## TERMINAL: final fidelity-residual oracle passes local headroom — 2026-10-17

The user authorized the next feasibility audit. Run exactly one no-training,
GT-only diagnostic on the fixed 256-record source-disjoint development subset
using the hash-bound step-8,530 checkpoint. It evaluates global fidelity-
residual amplitudes from `0` to `2` in `.025` increments plus analytic local
8/16/32/64-patch amplitudes, and reports fixed-candidate and per-image Pareto
upper bounds. It must reproduce both proposal and alpha-1 fidelity metrics
within the frozen tolerance before writing a result. This is not deployable
evidence and may not access protected data. Code and CPU/unit qualification
are complete; the Codex container exposes no GPU, so the next action is the
single-node eight-GPU launcher:
`smoke/run_foundir_proposal_residual_basis_noise_v1_oracle_8gpu.sh`.
The audit completed successfully in `72` seconds (`.020` wall-clock hours,
`.160` allocated GPU-hours), with exact proposal and fidelity metric
reproduction and no protected-data access. No global amplitude crosses the
joint gate; alpha `1.000` remains the maximum-PSNR global setting. In contrast,
the GT-derived local 8x8 amplitude upper bound reaches
`43.65973751991987/.03243445468615391`, or `+.5355773419141769 dB` and
`-.01792761307387991` LPIPS, and all tested local patch families cross the
gate. Therefore the frozen residual direction contains ample local corrective
information; the remaining problem is learning a deployable spatial amplitude
map without GT. Do not report any patch-oracle metric as model performance.
Result SHA-256:
`ce6ab400c5003a630f3e05fed60a5e1fe9e6d6514af86cddc1fed6b9e65e2f66`.
Full report:
`reports/foundir_proposal_residual_basis_noise_v1_oracle_terminal_2026-10-17.md`.

## LIVE: deployable 8x8 spatial amplitude predictor ready — 2026-10-17

The user authorized the minimal causal follow-up to the local oracle. Freeze
both the proposal and step-8,530 residual basis and train only the `223,393`-
parameter spatial amplitude predictor. It receives observation, proposal,
fidelity and fidelity-minus-proposal, outputs one `[0,2]` amplitude per 8x8
patch, and starts at exact alpha `1` so step zero reproduces fidelity. Training
GT provides analytic patch-amplitude supervision; inference and validation
must never use GT to select amplitudes. The loss contains restored-image MSE,
residual-energy-weighted teacher regression, weak LPIPS and local no-harm.

JSON, Python, Bash, focused unit tests, strict frozen-checkpoint loading and
diff checks pass. The next action is a fresh 25-step engineering qualification
on one homogeneous 8xH100/H200 node. It must not seed a formal run. Do not
train energy, use protected data, or report the GT oracle as model quality.
Contract:
`configs/foundir_spatial_residual_amplitude_noise_v1.json` and
`reports/foundir_spatial_residual_amplitude_noise_v1_readiness_2026-10-17.md`.

The fresh `qualification_v1` run completed successfully on `8 x H100 80GB`
with status `0`, finite training, checkpoint/validation output and no protected
data access. At step 25, predicted amplitude is still near its exact-identity
initialization (`mean 1.0000218451`, standard deviation `.0004146491`), so
spatial output reproduces fidelity within `+.0000015423 dB` and
`-.0000000773` LPIPS. This is an engineering pass, not a scientific result.
The next action is a fresh-ID, from-zero five-pass/8,530-step formal run; never
seed it from qualification. Estimated H100 cost is `2.5–3.5` wall-clock hours
and `20–28` allocated GPU-hours, to be updated from pass-1 throughput.

`formal1` is live and completed pass 1 at step `1,706`. Spatial output is
`43.395900413393974/.029359601011492487`, which is worse than frozen fidelity
by `.00125314295291574 dB` PSNR and `.000404557562433184` LPIPS. The amplitude
map has moved materially (`mean 1.0358318537`, standard deviation `.1107148176`),
so this is an early generalization miss rather than a no-learning failure.
Continue the unchanged five-pass schedule; do not retune or stop from pass 1.
Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass1_2026-10-17.md`.

Pass 2 at step `3,412` is mixed but slightly more encouraging. Spatial output
is `43.3995122089982/.033109975053776`, which improves PSNR over frozen
fidelity by `+.00235865265130997 dB` but worsens LPIPS by
`.004154931604716697`. Relative to proposal it remains
`+.27535203099250793 dB/-.017252092706257827 LPIPS`, so LPIPS still passes the
joint target while PSNR is `.02464796900749206 dB` short. Continue unchanged;
do not claim oracle transfer or retune from this interim result. Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass2_2026-10-17.md`.

Pass 3 at step `5,118` remains a small positive PSNR result but is flattening.
Spatial is `43.400297187268734/.0317960577121994`, or
`+.0031436309218444425 dB` PSNR and `.0028410142631400966` worse LPIPS versus
frozen fidelity. Relative to proposal it is
`+.27613700926303864 dB/-.018566010047834425 LPIPS`, leaving
`.023862990736961353 dB` to the PSNR gate. Continue passes 4 and 5 unchanged;
current evidence does not support likely gate success or early retuning. Full
record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass3_2026-10-17.md`.

## STOP: V2 and direct-residual oracle are terminal — 2026-10-17

The user stopped `noise_energy_v2_formal1` externally after its last logged
step `6,655`; do not resume it. The last complete checkpoint with a matched
validation is step `5,118`. Its GT-only direct-residual feasibility diagnostic
finished on the same fixed 256-record source-disjoint development set. Even
the target-selecting upper bound failed the `+.30 dB/-.005 LPIPS` joint gate.
The maximum-PSNR oracle point was only `+.042452 dB/-.003946 LPIPS`; when the
LPIPS requirement was enforced, the best PSNR gain was only `+.037468 dB`.
Therefore the learned direct residual does not contain enough corrective
information, and a learned gate over these candidates cannot meet the target.
Do not resume V2, train a gate over this frozen residual, or claim the oracle
as deployable evidence. No V3 or fallback is authorized until the user makes
the next research decision. Full result:
`reports/foundir_proposal_energy_v2_direct_residual_oracle_terminal_2026-10-17.md`.

## STOP: max-complete PPT energy V2 only — 2026-10-15

This section supersedes the V1/fallback sequence below wherever they differ.
The proposal-relative energy V1 must finish its already-running frozen
five-pass endpoint and then stop; do not extend it to ten passes. Its observed
failure mode is higher PSNR with worse LPIPS, poor gradient-direction
alignment, and multi-step drift. Preserve V1 as causal evidence.

Before switching to the DiT-MoE fallback, implement and test exactly one
max-complete V2 of the advisor-approved PPT energy idea. V2 must contain all
and only these six missing parts:

1. LPIPS/perceptual-aware training in addition to pixel/edge fidelity;
2. actual frozen-model outputs for wrong-expert, under-restored,
   over-restored, and detail-damaged contrastive candidates;
3. directly unrolled and trained 1/2/3-step energy trajectories;
4. per-step no-harm constraints for both pixel fidelity and LPIPS, with the
   learned energy also allowed to select the unchanged proposal;
5. per-image VLM degradation severity from the existing frozen Qwen manifest,
   not a constant Noise one-hot vector;
6. spatially preserved/local energy supervision rather than reducing all
   quality evidence to one early global scalar.

Do not add another backbone, dataset, degradation task, specialist, routing
mechanism, MoE, compositional branch, or unrelated loss/idea to V2. Keep the
frozen Noise proposal, the 54,585-record source-disjoint training split, the
fixed 256-record development validation, 1/2/3-step reporting, protected-data
prohibitions, and the existing minimum `+.30 dB` and `-.005 LPIPS` gate.
Engineering qualification and the adequately scaled scientific run are
allowed only to validate this exact V2; they are not extra research branches.
Do not claim the PPT approach works until both metrics improve on the frozen
development protocol.

Current status: V1 completed `8,530` steps and failed the gate because its best
energy arm improved PSNR only `+.10285484790802033 dB` while worsening LPIPS
by `+.006770727247385367`. V1 is terminal and must not resume. The exact-six-
change V2 code passed its fresh 25-step qualification on `8 x H100 80GB` with
status `0`, finite saved weights, optimizer state for every trainable entry,
successful validation/checkpointing, and no protected-data access. The next
action is a new-run-ID, from-zero `8,530`-step formal V2 run. The 25-step
checkpoint must never seed the formal run.

## DEFERRED: advisor-approved two-stage novelty plan — 2026-10-15

This section supersedes the 2026-10-14 energy-TTS decision below wherever the
two differ. The phrase "PPT direction" means the restoration-energy method
already presented in the advisor slides; it does not mean making more slides.

The advisor's decision is:

1. **Try the PPT restoration-energy idea first.** Test whether a lightweight
   action-contrastive / proposal-relative energy can improve the already
   frozen FLUX.2/Restormer output. The energy is conditioned on the degraded
   input and explicit VLM degradation information. It should learn to
   distinguish a correct restoration from under-restoration, over-restoration,
   a wrong expert action, or newly damaged details, and then use that signal
   for a small number of candidate-image correction steps. This is currently a
   hypothesis, not an established working result.
2. **Use a staged but adequately scaled causal test.** Start from Noise on
   source-disjoint development data and compare against both the frozen
   proposal and a matched feed-forward refiner. Preserve the numerical gates
   and data restrictions in the 2026-10-14 section below. The first 500 steps
   are an engineering qualification only: they test correctness, memory,
   throughput, checkpoint/resume, and whether gradients and validation are
   finite. They are not enough to declare the research idea dead. After that
   qualification, train on the complete 54,585-record Noise training split.
   The first scientific decision point must include at least one complete
   effective pass over those records; if the validation trend is still
   improving, continue through a predeclared multi-pass schedule rather than
   stopping because the model is initially undertrained. Derive the exact
   optimizer-step schedule and GPU-hours from measured throughput before the
   formal launch. Do not claim success merely because learned energy decreases;
   PSNR/LPIPS and candidate ranking/refinement must improve.
3. **Only if the PPT energy idea does not work, switch to the advisor's DiT-MoE
   fallback.** The fallback is a FLUX/DiT restoration model whose internal
   experts are controlled using explicit VLM predictions of degradation type,
   severity, and composition. Do not begin this fallback in parallel with the
   first energy test unless the user explicitly changes the sequence.

Novelty boundaries for the fallback must be stated honestly:

- FoundIR-v2 already uses MoE, although its backbone is SDXL rather than DiT.
- MiM-DiT already uses MoE inside a diffusion transformer for All-in-One image
  restoration. Therefore **"MoE + DiT" is not a new contribution**.
- The remaining hypothesis is narrower: MiM-DiT directly looks at degraded
  image features to choose experts, while the proposed fallback first asks a
  VLM to state the degradation type, severity, and composition and then uses
  that explicit diagnosis to control DiT experts.
- "We added a VLM to the router" is not automatically enough novelty. It only
  becomes defensible if a matched comparison shows that explicit VLM guidance
  improves mixed/OOD degradation restoration or expert composition over an
  otherwise identical image-feature router.

Advisor-chat provenance, in plain language: the advisor said the PPT ideas are
reasonable but may not work and asked whether any already work. The advisor
then suggested MoE, initially noting FoundIR-v2 and its lack of this DiT
combination. After learning that MiM-DiT already covers the broad MoE+DiT idea
but does not explicitly output degradation type/severity through a VLM, the
advisor approved the sequence: **try the project's own PPT energy idea first;
if it fails, the VLM-guided DiT-MoE fallback is still acceptable because it is
not a complete overlap.** No method-success claim follows from that approval.

### Frozen scale and hardware plan for the first energy experiment

Only one homogeneous eight-GPU node may be used for this experiment. It may be
**8 x NVIDIA H200** or **8 x NVIDIA H100 80GB**, but never a mixed node and
never multiple nodes. Do not create or propose a cross-node launch. All
training, validation, checkpointing, and resume logic must work on either
supported single-node profile while preserving global batch 32.

- H200 profile: micro-batch 4/GPU, gradient accumulation 1.
- H100 profile: micro-batch 1/GPU, gradient accumulation 4.

These profiles have the same eight GPUs, global batch, optimizer-step schedule,
and training-record sampler. The smaller H100 micro-batch is only a memory
compatibility change; it does not authorize fewer data passes or different
method settings.

The Noise training split contains 54,585 paired records. Use global batch 32
unless the measured H200 memory/throughput qualification proves a different
batch is materially better. At global batch 32, one complete effective data
pass is `ceil(54,585 / 32) = 1,706` optimizer steps.

Use this frozen staged scale:

1. **Systems qualification:** 500 optimizer steps. This checks correctness,
   finite second-order gradients, memory, throughput, validation, checkpoint,
   and exact resume. It is not a scientific failure decision.
2. **First scientific checkpoint:** 1 complete effective pass, 1,706 total
   optimizer steps. Evaluate the frozen proposal, matched feed-forward refiner,
   and 1/2/3-step energy refinement on the fixed source-disjoint development
   validation set.
3. **Primary adequately scaled run:** 5 complete effective passes, 8,530 total
   optimizer steps. Save and validate at the end of every pass. This is the
   minimum scale for the main method judgment unless an earlier run has a
   numerical/code failure rather than merely weak metrics.
4. **Pre-authorized continuation:** extend the same run to at most 10 complete
   effective passes, 17,060 total optimizer steps, only when pass-4 to pass-5
   source-disjoint validation still shows a meaningful improving trend and no
   overfitting. Do not change the validation set, loss, candidate steps, or
   checkpoint rule based on protected reporting data.

Do not launch 100,000 steps by default: that would be approximately 58.6
passes over this split and is not currently justified. The 10-pass cap may be
changed only by a later explicit user decision backed by the pass-by-pass
development curve.

Exact wall-clock and allocated GPU-hours cannot be claimed before the H200
qualification measures seconds per optimizer step. After the 500-step run,
compute and report:

- `wall hours = remaining_steps * measured_seconds_per_step / 3600`;
- `allocated GPU-hours = wall hours * 8`.

Never transfer a timing estimate between H100 and H200, and never substitute
an old multi-node EBT benchmark. Measure seconds/step on the exact single-node
GPU family used for the run.

## Mandatory reporting style

All future user-facing reports must be easy to understand on the first read:

- Start with the practical answer or current outcome, then explain why.
- Use short, direct Mandarin sentences; retain English technical terms only
  where they prevent ambiguity.
- Translate each necessary term immediately into plain language. For example,
  say "模型直接看图选专家" before or alongside "latent-feature routing".
- Clearly separate **confirmed fact**, **our hypothesis**, and **next test**.
- For every comparison, name both sides, the dataset/protocol, and what each
  number means. Never use an unexplained phrase such as "coverage", "gate", or
  "better".
- Do not present an idea, advisor approval, lower energy, or a successful code
  run as evidence that image quality improved. State "not tested yet" when no
  metric result exists.
- Give time estimates in normal wall-clock hours and also report allocated
  GPU-hours for every proposed or completed GPU experiment.
- Prefer one concrete example over dense jargon. Add equations or internal
  implementation details only after the plain-language explanation.

## STOP: frozen FLUX/Restormer proposal + energy-TTS decision — 2026-10-14

This section supersedes the older Generalist-first and live-run sections below.
The from-scratch EBT Generalist is paused and must not be resumed, duplicated,
reconfigured, evaluated on the protected 660 endpoint, or used to select a
checkpoint. Its existing artifacts and failure logs remain historical evidence.

The only active restoration-model research objective is a lean test of
EBT-style test-time optimization applied to the already successful frozen
FLUX.2/Restormer routed system. FLUX.2/Restormer produces an immutable initial
proposal. A lightweight observation- and VLM-conditioned scalar energy
`E(y; x, c)` may then refine candidate `y` by a small number of gradients with
respect to the candidate image. Do not retrain or alter the frozen proposal
models during the first causal test.

Work source-disjoint development data only. Never use the official, sealed,
parent-test, internal-test, or frozen 660 reporting pixels for training,
hyperparameter selection, checkpoint selection, early stopping, or deciding
the number of test-time steps.

Proceed in this order:

1. Start with the Noise singleton, because the frozen routed result
   `40.133486/.096746` is behind Full FoundIR `42.655305/.067910` in both PSNR
   and LPIPS.
2. Compare the frozen proposal against a matched-parameter feed-forward bounded
   refiner and fixed 1/2/3-step candidate-gradient refinement. Require extra
   steps to improve restoration rather than merely lower an uncalibrated
   energy.
3. A minimum technical pass is at least `+.30 dB` and `-.005` LPIPS relative
   to the frozen proposal on protected source-disjoint development validation.
   Paper-worthy evidence is at least `+.50--.75 dB` and `-.010` LPIPS, a useful
   compute-quality curve, and superiority to the matched feed-forward refiner.
4. Only after the Noise gate passes may the same frozen mechanism be tested on
   the remaining singleton categories. A convincing singleton aggregate target
   is at least `29.2 dB` with LPIPS no worse than `.135`, with gains across
   multiple categories rather than one category dominating the mean.
5. Only after multi-category singleton evidence may the unchanged mechanism be
   evaluated on Double compositions. The frozen Double baseline is
   `24.321537/.192313`; an indicative paper-strength target is at least
   `24.8/.180`, subject to per-category no-harm evidence.
6. Do not reopen Triple, specialists, compositional EBT, or another from-scratch
   Generalist until this lean mechanism succeeds or fails under its frozen
   causal comparison.

The working novelty hypothesis is not simply "EBM", "more inference steps",
or "VLM + EBT". It is candidate-dependent, differentiable test-time refinement
of frozen heterogeneous AiOIR proposals under an observation- and structured
VLM-conditioned scalar energy, with the same energy used for iterative
restoration and stopping/verification. Treat "first" as unverified until the
targeted prior-art audit is complete. Current audit and exact claim boundary:
`reports/foundir_flux_restormer_energy_tts_novelty_audit_2026-10-14.md`.

The first GPU qualification, if the implementation and protocol pass CPU
review, is capped at 500 optimizer steps on one 8-GPU node. Its duration is
measured rather than constrained by the obsolete `.5` wall-hour estimate,
because second-order candidate gradients may take longer on H100 than H200.
**This is a systems check, not a method-failure gate.** After it passes, use
its measured throughput to freeze
an adequately scaled schedule over all 54,585 Noise training records, with at
least one complete effective data pass and further predeclared passes whenever
the source-disjoint validation trend is still improving. Report both
wall-clock hours and allocated GPU-hours. Stop only for a code/numerical
failure, a clearly saturated or worsening development curve after adequate
exposure, or completion of the frozen scale schedule—not merely because a
small run misses the final metric target.

## DEFERRED: Generalist-first research decision — 2026-10-11

This was the active objective on 2026-10-11 but is superseded by the
2026-10-14 decision above. Preserve it as historical protocol context only.

Do not design, implement, train, tune, benchmark, or launch any low-light,
weather, trajectory-energy, residual-energy, or other specialist. Preserve the
specialist prior-art/design notes as deferred research only. Do not use a
specialist result to claim that the Generalist gate passed.

The decisive comparison is Generalist versus Generalist on the existing 660
full-resolution whole-image composition set, using identical records and the
frozen metric protocol (`full_rgb_uint8_image` PSNR and
`area_weighted_nonoverlap_512_rgb_uint8` LPIPS). Neither side may run a
specialist. The released FoundIR Generalist reference is hash-bound in
`checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json`.

For an unambiguous pass, the EBT Generalist must beat the released FoundIR
Generalist in both PSNR (higher) and LPIPS (lower) for every composition tier,
not merely in a weighted aggregate:

| Tier | Released FoundIR Generalist PSNR | Released FoundIR Generalist LPIPS |
|---|---:|---:|
| Singleton, 300 images | 27.2323939564508 | 0.1744119563086327 |
| Double, 300 images | 22.245453630403492 | 0.2066620594635684 |
| Triple, 60 images | 24.9455711880497 | 0.03775338436922618 |
| Overall, 660 images (reporting only; not a substitute for tier gates) | 24.757709920211013 | 0.17664758756638468 |

## HISTORICAL FAILED RUN: real-data Generalist and pass-1 diagnostic — 2026-10-12

`real_generalist_4node_mixed_hopper_v1` ran on 8 x H200 plus 24 x H100. It used
global batch 64, 10,233 steps/pass, three planned passes, EMA `.9999`, and the
frozen 654,881-pair real training split. It failed during pass 2 at pass step
1,300 / global step 11,533 after a 600-second NCCL ALLREDUCE watchdog timeout.
Only pass 1 is complete. Do not resume or evaluate this run while the
2026-10-14 decision is active. Its 16,461-record 256x256-crop validation is
diagnostic only and is not comparable to the decisive 660-image protocol.

Pass-1 raw tier metrics were Singleton `28.829671/.172490`, Double
`23.916114/.242836`, and Triple `33.070977/.025842`. The apparent Double/Triple
inversion is not contradictory:

- Triple contains only `blur+low_light+noise` (579 validation records), whereas
  Double averages five heterogeneous categories.
- Double raw per-category LPIPS is `blur+noise .017810`,
  `haze+low_light .196602`, `low_light+noise .242655`, `haze+rain .277549`,
  and `blur+low_light .312665`. The last three drive the Double weakness.
- The sampler assigns one third of presentations to each composition tier. The
  sole Triple category therefore receives approximately five times the
  category-level exposure of each of the five Double categories.
- LPIPS has weight `.05` and is included only every four optimizer steps, so
  PSNR can improve before hard Double perceptual artifacts are resolved.

The former instruction to complete all three passes and run a matched 660
evaluation is deferred. Do not access the 660 endpoint for this failed run
while the 2026-10-14 decision is active.

## Historical FLUX/Restormer gap record for the monolithic/generalist route

The old routed FLUX.2/Restormer system's Triple gap to FoundIR was not a simple
parameter-capacity problem. Preserve these established causes. They motivate
the bounded energy-refinement test above but do not authorize specialist or
Triple work:

1. Most FLUX work used synthetic/task-specific data derived from roughly 1,600
   clean sources, versus FoundIR metadata covering 1,010,064 pairs and the
   frozen 654,881-pair real training split now used by EBT.
2. FLUX experts received hundreds to roughly 10,000 task-specific steps,
   whereas the released FoundIR Generalist is at step 1,300,000. Raw steps are
   not batch-normalized, but the exposure gap is material.
3. FLUX/Restormer was a heterogeneous routed expert system, not one jointly
   trained Generalist over all singleton/double/triple categories.
4. Sequential experts have measured inter-expert domain shift: on 53 Triple
   records, Noise was nearly inert (`+.069859 dB`), Low-light overshot GT luma
   by `+29.503`, and Blur worsened 49/53 records and lost `.272308 dB`.
5. Synthetic operator order/strength did not fully match FoundIR, including the
   documented Noise/JPEG visibility mismatch.
6. Short FLUX/Restormer selections lacked a mature EMA selection path; current
   EBT uses EMA `.9999` and validates raw and EMA after every pass.
7. FLUX is a large pretrained generative DiT, while FoundIR's approximately
   36.3M-parameter UNetRes is restoration-native. More FLUX parameters did not
   substitute for paired restoration alignment.
8. Full FoundIR also has Low-light and Weather specialist refinement. That is a
   separate full-system advantage and must not be mixed into the frozen
   Generalist-versus-Generalist gate. Specialist work remains paused.

After the active real-data run completes, the remaining path is:

1. Audit all pass/checkpoint/validation artifacts and hashes.
2. Select and freeze raw or EMA weights without protected-test access.
3. Run one matched 660 whole-image Generalist-versus-Generalist evaluation.
4. Only after all tier gates pass may specialist research be reopened.

Report wall-clock hours and allocated GPU-hours for every proposed or completed
GPU experiment.

## DEFERRED: potential SOTA follow-up after the Generalist gate — 2026-10-12

This records a candidate paper-evaluation program. It does not change the
active Generalist-first decision and does not authorize another training or
evaluation while `real_generalist_4node_mixed_hopper_v1` is running. First
finish all three frozen passes, select raw versus EMA using only the frozen
16,461-record validation evidence, and run exactly one matched 660-image gate.
Do not tune from that result. The project is not yet demonstrably SOTA:
pass-1 crop validation is encouraging, but Double LPIPS remains worse than the
released FoundIR Generalist and is not directly comparable to whole-image
metrics.

If and only if EBT beats the released FoundIR Generalist in both PSNR and LPIPS
for Singleton, Double, and Triple on the frozen 660 protocol, the lean
potential-SOTA program is:

1. Train/evaluate one matched CDD-11 model and report all eleven categories
   plus Singleton/Double/Triple aggregates with PSNR/SSIM. Main composite
   targets are AllRestorer `33.72/.9436`, DAR-Net `29.73/.888`, ClearAIR
   `29.34/.886`, MoCE-IR-S `29.05/.881`, and OneRestore `28.47/.878`.
   AllRestorer lacks a verified reproducible public implementation, but its
   published number cannot be omitted from an unrestricted SOTA claim.
2. Train/evaluate one matched standard 5D model on SOTS, Rain100L, BSD68 sigma
   25, GoPro, and LOL. Important average PSNR/SSIM targets include D3Net
   `32.36/.901`, DAR-Net `31.02/.921`, DPC-Net `31.00/.923`, DFPIR
   `30.64/.913`, MoCE-IR `30.58/.919`, and CyclicPrompt `30.52/.935`.
   Disclose metric/protocol tradeoffs and never compare unmatched numbers.
3. Run two matched retraining ablations: no VLM conditioning and no learned
   energy/direct restoration. Use the frozen full checkpoint for evaluation-
   only adaptive-TTS fixed-step sweeps and self-verification on/off tests.
4. Report parameters, FLOPs, peak memory, latency, and actual adaptive descent
   steps. Cite and discuss FoundIR-v2, VLU-Net, DFPIR, InstructIR, Perceive-IR,
   ClearAIR, and EBT even where benchmarks are not directly matched.

The minimum defensible follow-up has eight blocks: frozen 660 evaluation;
matched CDD-11 train/eval; matched 5D train/eval; two retraining ablations;
adaptive-TTS evaluation; self-verification evaluation; and efficiency
evaluation. This means four new training runs and roughly six to eight
evaluation jobs after the current gate. A broad AiOIR claim additionally needs
separately trained 3D, real-world, and robustness/OOD evidence, raising the
program to roughly 12--15 blocks and six to eight training runs.

The current 30,699-step 32-GPU schedule measures about 8.97 wall hours/pass,
or about 27 wall hours and 864 allocated heterogeneous GPU-hours for three
passes. Four mechanical repeats would be an unapproved conservative ceiling of
108 serial wall hours and 3,456 GPU-hours. Before any CDD-11, 5D, or ablation
launch, run a protocol-specific 500-step throughput qualification (target at
most 0.5 wall hour and 4 allocated GPU-hours on eight GPUs), then size training
by matched epochs and unique-sample exposure. Narrow the intended claim to
composite/mixed Generalist SOTA or best matched average; winning the
project-specific 660 comparison alone is not universal AiOIR SOTA. Specialist
work remains paused.
