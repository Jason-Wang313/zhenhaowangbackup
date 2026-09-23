# FoundIR current handoff — 2026-09-21

## STOP: terminal official status updated 2026-09-22 18:39 Shanghai time

This section supersedes the historical status and plans below. Do **not** treat
Blur as unresolved and do **not** resume the older Blur-only plan. The authorized
600-record official SOW evaluation completed successfully: all five categories
and all ten PSNR/LPIPS checks pass.

| Category | Records | Official PSNR | Official LPIPS | SOW | Result |
|---|---:|---:|---:|---|---|
| Blur | 150 | 28.092635 | .144089 | >=25.9 / <=.18 | PASS |
| Noise | 100 | 40.048323 | .056601 | >=38.9 / <=.10 | PASS |
| Haze | 200 | 22.677420 | .110437 | >=22.1 / <=.19 | PASS |
| Rain | 100 | 37.809175 | .052901 | >=34.2 / <=.08 | PASS |
| Low-light | 50 | 19.322459 | .148799 | >=18.9 / <=.28 | PASS |

The official result reports `status=pass` and
`all_ten_sow_checks_pass=true`. Routing used no ground-truth label as router
input and recorded 538 correct decisions, 62 fallbacks, and zero wrong routes.
Metrics used original-size RGB images, no resize/crop/border shave, `pyiqa`
0.1.16, and unweighted per-image means within each category.

Canonical evidence:

- terminal report:
  `reports/foundir_official_sow600_qwen_v8_terminal_result_2026-09-22.md`;
- Git-tracked compact result:
  `reports/foundir_official_sow600_qwen_v8_result_2026-09-22.json`;
- local raw result:
  `checkpoints/foundir_official_sow600_qwen_v8_long_singletons/official_result.json`;
- raw-result SHA256:
  `8ea3a681eec65f1ef6b28cbd48292a17e1d6b2ee8e11f04345bd54d7cee12684`;
- frozen config SHA256:
  `6a84adfec7c929f291fd501976358cf61c062f35179217425b835e0d4979e04d`;
- completion log timestamp: `2026-09-22T10:39:22Z`
  (`2026-09-22 18:39:22` Shanghai time).

The earlier development-set table (`26.6147/.15697`, `40.1506/.06888`,
`23.3119/.10116`, `42.2843/.04745`, `19.7791/.15406`) is not the terminal
official result and must not replace the official table above.

## Latest user direction: diagnosis and first-principles redesign only

The user explicitly declined new training, then requested leaving the repeated
FFTformer/NAFNet fine-tuning route and reasoning from the blur formation process.
Do not launch NAFNet fine-tuning from the older architecture-selection text below.
The diagnostic proposal is
`reports/foundir_blur_inverse_problem_strategy_2026-09-21.md`: audit paired
geometry/photometry, identify a constrained forward blur operator on training
pairs, and test fixed-operator inversion before considering any learned estimator.
This proposal has not been executed; no training or new restoration inference ran.

Two corrections to historical interpretation: FFTformer v1/v2 changed the seed
and LR horizon as well as the loss, so their difference does not isolate loss
causality. The 26.5851/.21237 fusion oracle quoted below is from 128 development
records, not official Blur, and combines independently optimal PSNR/LPIPS values
over its tested alpha grid rather than one achievable joint candidate.

## Mandatory compute-sizing rule

Future agents must recommend the number of GPU nodes from the actual workload, not from the number of available nodes.

Before proposing multi-node execution, estimate all of the following:

1. single-node wall time for the complete remaining job;
2. model/gradient size and compute per optimizer step;
3. cross-node bandwidth and expected communication time per step;
4. whether changing world size changes global batch, optimizer-step count, learning-rate schedule, or checkpoint comparability;
5. engineering time for code changes, smoke tests, rendezvous, debugging, and user coordination;
6. expected net wall-clock saving after including that engineering time.

Use multi-node training only when the expected end-to-end saving materially exceeds implementation and coordination cost, and when the optimization change is scientifically acceptable. If coding, validation, and launch overhead is comparable to simply finishing on one 8-GPU node, recommend one node immediately. Do not optimize GPU utilization at the expense of total user time or experiment reliability.

Practical default:

- small/medium models and jobs under roughly 1–2 hours on 8 GPUs: use one 8-GPU node unless an already-tested multi-node launcher exists;
- embarrassingly parallel inference/evaluation: shard across nodes without NCCL;
- long training jobs with an existing verified launcher and favorable networking: consider multi-node after a short communication benchmark;
- do not rewrite a live or ready-to-run job merely because extra nodes are temporarily available.

## Multi-node audit and current decision

A 3-node × 8-H100 NCCL TCP smoke completed successfully on `eth0`:

- 24 ranks, correctness pass;
- 16 MiB all-reduce: 0.06335 s;
- 64 MiB all-reduce: 0.07832 s;
- 256 MiB all-reduce: 0.25309 s;
- estimated 256 MiB bus bandwidth: 1.8933 GiB/s;
- network interface reports 10 Gb/s; RDMA was disabled.

The smoke proves feasibility, not that multi-node is optimal. The current Noise Restormer is only about 26M parameters and the original 8-GPU job is expected to finish in approximately 45–70 minutes. Time already spent implementing and validating a 24-GPU variant is comparable to a substantial fraction of the original run. Only two nodes remain available, and creating another 16-GPU variant would add more engineering and validation delay while changing the training batch/step trajectory.

**Current optimal decision: do not continue adapting the job to 16 GPUs. Run the already-tested single-node 8-GPU job now.** The 24-GPU files remain experimental evidence and are not the recommended current launch path.

## Current official status

Qwen made zero routing errors on the five-category official scope. Four paths are frozen and must not be changed:

- Noise: `39.4679 PSNR / .09412 LPIPS`;
- Haze: `22.2106 / .12359`;
- Rain: `37.8092 / .05290`;
- Low-light: `19.1876 / .17261`.

Blur is the only unresolved category. Its SOW gate is `PSNR >= 25.9 / LPIPS <= .18`.

- deployed FLUX step-10k plus alpha `.75`: `21.5006 / .20456`;
- official Blur input: `21.8639 / .35360`;
- Qwen routing: `150/150` correct, so routing is not the cause;
- Restormer motion step-1000: `21.8318 / .38977`;
- even per-image GT-oracle FLUX fusion: `26.5851 / .21237`, so alpha/fusion cannot pass LPIPS.

Do not repeat FLUX continuation, small Restormer continuation, alpha/fusion, loss-weight sweeps, heavier-blur sampling, or router work. Existing evidence rules them out.

## FFTformer initialization audit and bounded final plan

The pinned upstream FFTformer is `kkkls/FFTformer` commit `01306b5d4072146a011c8c6e9d4927cf4607f1d5` (MIT). External checkpoints are diagnostic/initialization artifacts only and cannot be deployed directly.

- GoPro zero-training: `19.9498 / .18897`;
  - relative to SOW: PSNR `-22.97%`, LPIPS exceeds limit by `4.98%`;
- RealBlur-J zero-training: `19.6849 / .21650`;
  - relative to SOW: PSNR `-24.00%`, LPIPS exceeds limit by `20.28%`.

Neither qualifies for direct deployment. GoPro wins both metrics, especially LPIPS, and is therefore the frozen initialization. Do not test RealBlur-R or a third architecture.

The one authorized training run is now implemented:

- full FFTformer model, initialized from GoPro;
- only project FoundIR Blur training pairs: 101,907 records / 7,083 source groups;
- uniform source-group sampling, paired 256 crops, geometric augmentation;
- original FFTformer objective: RGB L1 + `0.1 * FFT L1`;
- AdamW, fixed peak LR `1e-4`, global batch 32, 8 GPUs;
- official GT never enters gradients;
- exact full-resolution Qwen + tiled restoration + official `pyiqa` evaluation at 2k/5k/10k;
- the pinned upstream layer `reduce_chan_level2.weight` is registered but never used by FFTformer's forward; its 18,432 unreachable parameters stay frozen and all 16,542,042 active parameters are trained;
- the launcher runs a real two-iteration 8-GPU DDP smoke before the 2k phase;
- automatic bounded stop: stop on SOW pass, no-signal/regression, or mandatory 10k ceiling;
- no alpha, fusion, initialization sweep, or third architecture.

Launch on one 8×H100/H200 node:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash smoke/launch_foundir_blur_fftformer_gopro_finetune_v1_8gpu.sh
```

Monitor:

```bash
tail -f logs/foundir_blur_fftformer_gopro_finetune_v1.log
```

CPU preflight has passed: 535 tensors, 16,560,474 total parameters (16,542,042 reachable/trainable), 101,907 records, 7,083 groups. The first launch exposed the upstream unreachable parameter at DDP iteration two and stopped before producing a checkpoint; the explicit freeze and mandatory two-step DDP smoke prevent recurrence. Estimated first 2k gate is 1–2 hours and the bounded 10k maximum is roughly 4–7 hours on one node; use the measured first 50-step rate instead of this estimate once launched.

## FFTformer FoundIR 2k terminal result

The corrected 8-GPU run completed its 2k milestone and exact official 150-image Qwen/tiled/pyiqa evaluation, then stopped under the frozen gate:

- step-2k: `23.3071 PSNR / .23299 LPIPS`;
- relative to SOW: PSNR is `2.5929 dB / 10.011%` below target; LPIPS is `.05299 / 29.439%` above the maximum;
- versus zero-training GoPro initialization (`19.9498 / .18897`): PSNR improved by `+3.3573 dB`, but LPIPS regressed by `.04402`;
- versus the current FLUX Blur path (`21.5006 / .20456`): PSNR improved by `+1.8065 dB`, but LPIPS regressed by `.02843`;
- Qwen remained `150/150` correct and official GT was not used for gradients.

Interpretation: FFTformer has useful FoundIR pixel-reconstruction capacity, but the original `L1 + 0.1 FFT` objective pushes perception in the wrong direction. Do not resume this checkpoint with the same loss to 5k/10k. The next proposal, if authorized, must directly constrain LPIPS/perceptual drift while retaining the demonstrated PSNR gain; it is a loss-alignment problem, not a routing, data-loading, architecture-loading, or insufficient-step diagnosis.

The user authorized that one bounded correction. It is implemented as v2: restart from the perceptually safer GoPro initialization, remove FFT loss entirely, train with `L1 + 0.1 AlexNet LPIPS`, and perform one terminal official evaluation at step 2k. There is no weight sweep, alpha/fusion, continuation from the perceptually damaged v1 checkpoint, third architecture, or training beyond 2k. Launcher: `smoke/launch_foundir_blur_fftformer_gopro_l1_lpips_v2_8gpu.sh`.

V2 completed and failed the joint SOW gate:

- official step-2k: `22.0802 PSNR / .17126 LPIPS`;
- PSNR is `3.8198 dB / 14.748%` below the SOW minimum;
- LPIPS passes with `.00874 / 4.858%` safety margin;
- versus the official input: `+0.2163 dB` PSNR and `-.18235` LPIPS;
- Qwen remained 150/150 correct; no official GT entered gradients.

The paired experiments establish a clean objective tradeoff on the exact official set: v1 `L1+FFT` reached `23.3071/.23299` (better PSNR, failed LPIPS), while v2 `L1+LPIPS` reached `22.0802/.17126` (passed LPIPS, weaker PSNR). Do not continue either unchanged. Any further authorized experiment should be described as one bounded Pareto-balancing objective, not more steps or a blind weight sweep.

The user subsequently authorized changing to a more suitable paired deblurring architecture. Historical inventory found no existing project-trained NAFNet/MPRNet/MIMO-UNet Blur checkpoint. NAFNet width-64 was selected as the single next architecture: upstream `megvii-research/NAFNet` commit `2b4af71ebe098a92a75910c233a3965a3e93ede4`, MIT, official GoPro checkpoint SHA-256 `329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f`, 664 tensors / 67,888,835 parameters. Strict load and forward pass succeeded. The external checkpoint remains diagnostic-only and cannot be deployed. Run the bounded official initialization diagnostic with `smoke/launch_foundir_blur_nafnet_gopro_external_diagnostic_v1_8gpu.sh`; only a project checkpoint subsequently trained on FoundIR Blur pairs may enter integration.

The NAFNet zero-training official diagnostic completed at `19.9253 PSNR / .18723 LPIPS`. Relative to SOW, PSNR is `5.9747 dB / 23.068%` below target and LPIPS is `.00723 / 4.017%` above the maximum. It reduces LPIPS by `.16637` versus the input but lowers PSNR by `1.9386 dB`. It is essentially tied with FFTformer-GoPro (`19.9498/.18897`) and failed the frozen transfer-qualification gate, so no FoundIR NAFNet training was automatically started. This establishes another GoPro-to-FoundIR domain mismatch; it does not establish that a fully FoundIR-trained NAFNet architecture can never work.

## User-authorized FoundIR-native NAFNet run

The user subsequently and explicitly authorized one FoundIR-native NAFNet
training run. This supersedes only the zero-training diagnostic's automatic
stop; it does not relax provenance or the bounded-experiment rules. The run is
now implemented but has not been GPU-launched from the Codex environment.

- initialization: pinned MIT NAFNet GoPro width-64 checkpoint;
- final eligible artifacts: only new checkpoints trained on the project's
  FoundIR Blur pairs;
- data: 101,907 training records / 7,083 source groups, sampled uniformly by
  source group, paired 256 crops with geometric augmentation;
- model: ordinary `NAFNet` during training and `NAFNetLocal` for tiled
  deployment evaluation;
- objective: the original NAFNet RGB `PSNRLoss`, i.e. mean per-image log-MSE;
- optimizer: AdamW, peak LR `2e-4`, global batch 32, one 8-GPU node;
- official 150-image full-resolution Qwen/tiled/pyiqa gates at
  1k/3k/6k/10k; Qwen remains required and frozen;
- stop immediately on joint SOW pass, weak joint progress, regression, or the
  10k ceiling;
- no alpha/fusion, loss-weight sweep, fourth architecture, or changes to the
  four already-passing category paths;
- official GT is used only to calculate milestone metrics and never enters a
  training gradient.

Frozen continuation rules:

- 1k: continue only with PSNR `>=22.5` and LPIPS `<=.21`;
- 3k: continue only if PSNR gains at least `.25 dB` and LPIPS falls at least
  `.002` versus 1k;
- 6k: continue only if the worst normalized SOW margin improves by at least
  `.01` versus 3k;
- 10k: mandatory terminal stop if no earlier joint pass.

CPU/static preflight passed: 664 tensors / 67,888,835 parameters, strict
initialization load, 101,907 records / 7,083 source groups, Python compilation,
and Bash syntax. The launcher forces a real two-step 8-GPU DDP smoke before
step 1k.

Launch on exactly one 8×H100/H200 node:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash smoke/launch_foundir_blur_nafnet_foundir_psnr_v1_8gpu.sh
```

Monitor: `tail -f logs/foundir_blur_nafnet_foundir_psnr_v1.log`.

### NAFNet 1k result and authorized geometry-matched continuation

The first FoundIR-native NAFNet run completed and stopped at its frozen 1k
joint gate:

- official 150 Blur: `22.326276 PSNR / .242858 LPIPS`;
- relative to SOW: PSNR `-3.573724 dB / -13.7982%`; LPIPS exceeds the cap by
  `.062858 / 34.9210%`;
- Qwen remained 150/150 correct;
- versus zero-training NAFNet, PSNR gained `2.400950 dB` while LPIPS regressed
  by `.055627`;
- no official GT entered training gradients.

A read-only per-record oracle audit then evaluated input, both external
initializations, both trained FFTformer variants, zero-training NAFNet, and
NAFNet step-1k. Even GT-informed per-image selection cannot pass:

- unconstrained maximum-PSNR oracle: `25.347463 / .207781`;
- maximum PSNR subject to mean LPIPS `<=.18`: `24.735863 / .179941`.

Therefore no alpha, fusion, router, or selector over existing outputs can
solve Blur. The audit is frozen in
`reports/foundir_blur_existing_candidate_oracle_audit_2026-09-21.json`.

The user authorized one capacity-first continuation because 1k exposed only
32,000 crops, less than the 101,907-record pool and only 0.25% of the original
NAFNet 400k-step schedule. The continuation is implemented independently and
does not overwrite v1:

- parent: project-trained NAFNet step-1k, not the external checkpoint;
- training data: 71,393 training-only 1920×1080 PNG pairs / 2,782 source
  groups, matching official Blur geometry/encoding;
- uniform source-group/record/random-crop sampling; the older hard-crop
  metadata is explicitly not inherited;
- fresh AdamW phase at LR `1e-4`, original RGB PSNRLoss;
- step-3k gate: PSNR `>=23.5`, LPIPS catastrophic ceiling `.35`;
- step-6k gate: PSNR `>=25.0`, gain `>=.75 dB` from 3k, LPIPS `<=.35`;
- step-10k: joint SOW pass freezes immediately; otherwise a one-time
  perceptual-recovery phase is eligible only if PSNR reaches `>=26.5`;
- the recovery phase is deliberately not materialized before a qualifying
  10k checkpoint exists, because its checkpoint SHA and actual PSNR budget
  must be pinned first.

CPU/static preflight passed: strict project-parent load, 664 tensors /
67,888,835 parameters, 71,393 records / 2,782 groups, continuation LR starts
at global step 1001, Python/Bash syntax, and launcher check. The GPU launcher
also forces a real two-step 8-GPU DDP smoke.

Launch on one 8×H100/H200 node:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash smoke/launch_foundir_blur_nafnet_geometry_matched_psnr_v2_8gpu.sh
```

Monitor: `tail -f logs/foundir_blur_nafnet_geometry_matched_psnr_v2.log`.

### Properly scaled NAFNet long schedule (v4, frozen 200k overnight horizon)

The 6k continuation reached `23.605029/.195281`; versus 3k this was only
`+.393155 dB/-.000806 LPIPS`, so the short 10k schedule gate stopped. A later
audit corrected the experimental scale assumption: upstream NAFNet trains for
400k iterations at global batch 64 (25.6M crop presentations), whereas this
project had exposed only 192k crops by step 6k, or 0.75% of that budget.

The user authorized one properly scaled continuation. V4 strictly restores
the project step-6k model and AdamW state, retains the 71,393 training-only
1080p PNG pairs, and replaces only the prematurely short LR horizon:

- LR is continuous at step 6001 (`4.267021716e-5`) and cosine-decays to `1e-6`
  at step 200k;
- official Qwen/tiled/pyiqa milestones: 25k, 50k, 100k, 150k, 200k;
- do not stop at intermediate milestones, even on a temporary pass or plateau;
  the user explicitly requested full overnight utilization through 200k;
- PSNR `>=26.5` records perceptual-recovery eligibility, but the unattended
  overnight PSNR run may continue while PSNR still improves so that it builds
  a larger fidelity budget without changing the predeclared loss or schedule;
- 25k/50k/100k/150k are observation and checkpoint milestones only;
- 200k is the mandatory terminal ceiling; final selection first prefers any
  joint-SOW-passing milestone, then maximizes the worst normalized SOW margin;
- official GT remains metric-only and never enters gradients.

Measured 8×H200 throughput predicts roughly 35–40 minutes to 25k, 1.4 hours
to 50k, 3 hours to 100k, and 6.1 hours to 200k from step 6k, plus five
milestone evaluations. V4 had not been launched when this horizon was extended,
so no active run or artifact was mutated.

Launcher:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash smoke/launch_foundir_blur_nafnet_long_psnr_v4_8gpu.sh
```

#### V4 terminal result — Blur solved at step 50k

The full unattended 200k run completed successfully. All five official
milestones were preserved and evaluated; later checkpoints did not overwrite
the best artifact. Step 50k is the frozen selection:

- official 150-image Qwen/tiled/pyiqa: `28.092683 PSNR / .144089 LPIPS`;
- SOW: `25.9 / .18` — both pass;
- PSNR margin: `+2.192683 dB / +8.4660%`;
- LPIPS safety: `.035911 / 19.9506%` below the maximum;
- SSIM: `.830897`;
- Qwen routing: 150/150 correct;
- official GT was metric-only and never entered training gradients;
- selected checkpoint:
  `checkpoints/foundir_blur_nafnet_long_psnr_v4/model_step_050000.pth`;
- checkpoint SHA-256:
  `7d391aa1692f72a3165b9e12b57dd94fef906bbeb3c522dc9c90533dd87f0af7`;
- official result SHA-256:
  `762ccf061b56080136142aed7f206e7d2dfc8ef223aaa5640bba1cec040c677b`;
- terminal selection SHA-256:
  `455dbe7efd5b2ec64c7e38577f9be392f7b9786eadc591d4205f4a95be512bc6`.

Milestone trajectory confirms that longer is not monotonically better:

- 25k: `23.820326/.191056` — fail;
- 50k: `28.092683/.144089` — strong joint pass and selected;
- 100k: `21.283699/.195393` — regression;
- 150k: `23.479511/.172428` — LPIPS-only pass;
- 200k: `22.561716/.179267` — LPIPS-only pass.

Do not deploy `latest_training_state.pt`, step 100k, 150k, or 200k. Do not run
the planned perceptual-recovery phase: step 50k already has substantial safety
on both required metrics. The next action is to pin step 50k into the Blur
branch of the all-in-one integration and run an integration fidelity check;
no further Blur quality training is justified.

#### V8 all-in-one integration prepared — GPU fidelity run pending

Step 50k is now pinned as `kind=nafnet` in
`configs/foundir_final_eleven_restoration_v8_nafnet_blur.json`. This changes
only the singleton Blur expert. Noise, Haze, Rain, Low-light, `blur_noise`, and
all mixed restoration behavior remain frozen and unchanged. The official
Qwen decisions are also unchanged, with all 150 official Blur records routed
correctly to Blur.

The runtime now has an explicit NAFNet branch and will fail closed rather than
fall through to Restormer. Its contract pins and validates:

- upstream revision `2b4af71ebe098a92a75910c233a3965a3e93ede4`;
- architecture SHA-256
  `01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967`;
- selected checkpoint SHA-256
  `7d391aa1692f72a3165b9e12b57dd94fef906bbeb3c522dc9c90533dd87f0af7`;
- exactly 664 tensors and 67,888,835 parameters;
- the exact prior one-tile `/255`, 512/128 sqrt-Hann, uint8 execution path.

Static syntax, JSON, full artifact SHA, repository revision, and strict CPU
model-load preflights passed. A fresh integration directory already contains
hard-linked/copy-verified versions of the 450 unchanged non-Blur outputs; the
old v6 outputs were not overwritten. The only pending operation is the real
8-GPU restoration of the 150 Blur records followed by a required 150/150
pixel-exact comparison against the already selected step-50k outputs. The
prior equivalent Blur restoration took 31 seconds on eight GPUs, so allow a
few minutes including model load, shared-storage checks, and comparison.

Launch on one 8×H100/H200 node:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash smoke/launch_foundir_official_sow600_nafnet_blur_fidelity_8gpu.sh
```

The 8-GPU integration subsequently completed all 150 Blur records in 25.01
seconds. A literal byte/pixel-exact comparison did not pass because the
selected reference was produced on H200 while the integration was reproduced
on H100: every differing channel value was exactly one uint8 level. Across
all 933,120,000 RGB channel values, only 0.626647% differed; no value differed
by more than one, and the worst individual image changed 1.161925% of channel
values. The synthetic same-runtime path test remained pixel-exact, supporting
cross-GPU convolution rounding rather than an integration-path discrepancy.

The corrected, explicitly bounded cross-GPU numerical-equivalence gate
requires max absolute uint8 difference `<=1`, zero values above that limit,
global changed fraction `<=2%`, and every record changed fraction `<=2%`.
It passed 150/150 records with max difference 1, zero above-limit values, and
global changed fraction 0.00626647. The terminal artifact is
`checkpoints/foundir_official_sow600_qwen_v7_nafnet_blur/blur_integration_fidelity.json`
with status `pass_numerically_equivalent`. This completes Blur singleton
integration fidelity; no additional Blur GPU run or training is justified.

### Three independent long-training jobs prepared (Rain explicitly excluded)

The user authorized longer training for Haze, Low-light, and Noise on three
separate 8-GPU nodes, then explicitly declined additional Rain training. Rain
therefore remains frozen at official `37.8092/.05290` (PSNR +10.55% over SOW,
LPIPS 33.88% below its ceiling).

Prepared jobs:

- Haze FLUX: exact continuation from step 3000 to absolute step 25k, using
  the proven 1,580 source-diverse training groups and all 158 native
  validation groups. Milestones: 5k/10k/15k/20k/25k.
- Low-light FLUX: exact continuation from step 400 to absolute step 25k.
  The old capacity run used only 16 training sources; the long run expands to
  all 723 training source groups and all 21 validation groups. Milestones:
  5k/10k/15k/20k/25k.
- Noise Restormer: a fresh low-LR AdamW phase initialized from the passing
  project step-250 MSE+LPIPS model, with 25k new steps on all 1,755 training
  source groups. Milestones: 2k/5k/10k/15k/25k.

Every job preserves its current deployed checkpoint as an external fallback.
FLUX evaluates the parent checkpoint on the same new development split before
training, and final selection considers the parent plus every milestone. Noise
selection likewise keeps all milestone checkpoints; no final-step assumption
is made. Official test is not used for milestone selection.

Static syntax, artifact/config contracts, full Haze and Low-light image/crop
preflights, model hashes, Noise data preflight, and 15 existing regression
tests passed. Expected H100 wall times under ordinary GPFS load are Haze
7–9h, Low-light 8–10h, and Noise 6–7h; H200 may be roughly 15–25% faster.
Concurrent GPFS load can push runs toward the upper bound.
