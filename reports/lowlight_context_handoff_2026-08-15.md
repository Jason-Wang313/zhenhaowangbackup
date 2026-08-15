# Lowlight Phase 1 fresh-session handoff

Date: 2026-08-15 UTC
Workspace: `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`
Branch: `main`

## Mandatory startup behavior

Before every tool call session, tell the user exactly:

> I am working in zhenhaowang, which is the correct workspace.

Read `agents.md` completely before editing. Work only inside this workspace.
The current user has completed the two confirmations required by the
destructive-change rule for the Lowlight baseline files described below. Do
not ask for those same confirmations again unless the proposed write scope
expands.

The previous Goal attachment path
`/root/.codex/attachments/14d61612-279c-4f11-b8f2-588c3914ef2a/pasted-text-1.txt`
was available in the prior environment but is not mounted in the new CPU
environment. This handoff preserves its operative objective and restrictions.

## Active objective

Complete the **Lowlight-only Phase 1 gate**:

1. Audit and document the local FLUX.2 Klein Base 4B restoration/training
   architecture.
2. Implement a paired Lowlight restoration baseline with frozen FLUX.2, VAE,
   and FLUX text encoder; train only the minimum LoRA/restoration adapter.
3. Pass a fail-closed 16-pair overfit gate before authorizing full training.
4. Train/evaluate the full 8,000-pair baseline with the fixed 1,000 validation
   and 1,000 internal-test records.
5. Select a real vision-capable Qwen-VL checkpoint. The existing
   `models/qwen3-4b-fp8` is only FLUX's text encoder and is **not Qwen-VL**.
6. Freeze Qwen-VL initially and make it output only:
   `{"degradations":["low_light"],"levels":{"low_light":1-5}}`.
7. Compare condition injection in this order: learned context token, gated
   AdaLN/modulation, then residual injection only if the first two fail.
   `condition=None` must reproduce the baseline path.
8. Run a fair three-way diagnostic with identical backbone, data, split,
   optimizer, learning rate, steps, batch size, precision, seeds, GPU budget,
   and sampling settings:
   - baseline without level condition;
   - Qwen-predicted level;
   - manifest ground-truth level, internal diagnostic only.
9. Acquire and validate the official FoundIR-Lowlight evaluation split and
   report PSNR, LPIPS, and SSIM with per-image evidence.
10. Lowlight Phase 1 threshold: PSNR `>= 15.1 dB` and LPIPS `<= 0.34`.
11. Commit/push each major verified code milestone and verify local/remote SHA.
12. After the Lowlight report, stop.

Do **not** begin Blur, Noise, Haze, Rain, the 100k dataset, or the separate
`亮色大模型` workstream. Do not claim full five-task Phase 1 acceptance from a
Lowlight pass.

## Data and verified existing state

Authoritative paired manifest:
`manifests/low_light_pilot_10000.csv`.

- 10,000 paired records from 2,000 unique clean sources.
- Sources: MiO100 100, DIV2K train 800, LSDIR train 1,100.
- Train: 8,000 pairs / 1,600 sources.
- Validation: 1,000 pairs / 200 sources.
- Internal test: 1,000 pairs / 200 sources.
- L1-L5: exactly 2,000 records per level.
- Split is source-disjoint.
- Final generator/validator previously passed.
- FoundIR overlap hashing was not run because no FoundIR index was available;
  this is recommended before final reporting but is not a current blocker.

FLUX.2 state:

- Source: `third_party/flux2`.
- Required revision: `50fe5162777813d869182b139e83b10743caef15`.
- Local patch: `patches/flux2-offline-smoke.patch`.
- Base weights:
  `models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors`.
- VAE: `models/flux2-klein-base-4b/ae.safetensors`.
- FLUX text encoder: `models/qwen3-4b-fp8/`.
- Earlier offline H100 inference smoke passed and generated
  `output/sample_0.png`. This proves model loading/basic inference only, not
  restoration training or FoundIR quality.

Last verified top-level Git state before current uncommitted work:

- local `HEAD`: `5ab77d1b129cb56db3cf6607ec48f07eca67b866`
- `origin/main`: `5ab77d1b129cb56db3cf6607ec48f07eca67b866`

Recheck both before committing; do not assume they remain unchanged.

## Current environment

The coding-agent terminal is CPU-only with internet access. The separate H100
terminal has no internet. The filesystem is shared, but processes, shell
variables, GPU visibility, and packages outside workspace-local environments
are not shared.

New CPU allocation verified on 2026-08-15:

- 128 logical CPUs (Intel Xeon Gold 6530);
- cgroup memory limit: 300 GiB;
- approximately 1.5 GiB in use at inspection time;
- shared filesystem had approximately 965 TB available.

Use `.venv/flux2-smoke/bin/python`. Never infer GPU success from CPU checks.
When H100 execution is genuinely ready, give the user exactly one offline
launcher command, then inspect logs/checkpoints/metrics from the shared
workspace.

## Baseline architecture decision already audited

The local official FLUX.2 repository is inference-only; it does not include an
official BFL training loop or LoRA recipe. The following training objective is
therefore a project-derived, explicitly documented formulation:

- clean latent is the rectified-flow data endpoint;
- noise is the start endpoint;
- `x_t = (1-t) * clean + t * noise`;
- velocity target is `noise - clean`;
- degraded latent is appended as FLUX reference-image tokens;
- reference token time coordinate is `10`;
- inference uses the same native reference-token layout and 50-step CFG path.

Frozen components: FLUX.2 base, VAE, and FLUX Qwen3 text encoder.

LoRA targets:

- `double_blocks.*.img_attn.qkv`
- `double_blocks.*.img_attn.proj`
- `single_blocks.*.linear1`
- `single_blocks.*.linear2`

Expected result: 50 matched Linear modules and 8,110,080 trainable parameters
at rank 8.

## Current uncommitted Lowlight implementation

Intended files:

- `.gitignore`
- `train_flux2_lowlight.py`
- `lowlight_restoration/__init__.py`
- `lowlight_restoration/config.py`
- `lowlight_restoration/data.py`
- `lowlight_restoration/flow.py`
- `lowlight_restoration/lora.py`
- `configs/lowlight_flux2_overfit16.json`
- `configs/lowlight_flux2_full.json`
- `smoke/build_lowlight_overfit_manifest.py`
- `smoke/lowlight_baseline_preflight.py`
- `smoke/evaluate_lowlight_overfit.py`
- `smoke/run_lowlight_baseline_overfit.sh`
- `tests/test_lowlight_restoration.py`
- `manifests/low_light_overfit16.csv`
- `manifests/low_light_overfit16.json`
- this handoff and the pending architecture report.

The 16-pair manifest uses 16 unique train sources with quotas L1=3, L2=3,
L3=3, L4=3, L5=4.

Do not stage unrelated untracked files. In particular, several old download,
calibration, and non-Lowlight scripts are visible in `git status`; preserve
them and leave them uncommitted unless separately justified.

## Fixes applied immediately before this handoff

These changes are written but not yet committed:

- `.gitignore` now permits top-level `reports/*.md`.
- Config validation now checks all runtime/training counts, non-negative
  reference time and weight decay, LoRA dropout, inference seed, and VRAM.
- Manifest loading validates non-empty source IDs, SHA-256 syntax, and seeds.
- Flow construction rejects non-finite/out-of-range timesteps.
- CPU preflight no longer constructs a complete 4B meta model to count LoRA
  modules. It derives the exact 50 modules and 8,110,080 parameters directly
  from SafeTensors shapes, avoiding the former 4 GiB OOM path.
- Preflight now validates exact local Qwen3 shard sizes/hashes unless large
  hashes are explicitly skipped, and imports `skimage` because evaluation
  requires it.
- Evaluation reuse now checks config hash, checkpoint hash, all output
  artifacts, finite metrics, and refuses to treat an existing failed gate as
  success.
- The launcher uses `flock`, so a killed process cannot leave a stale logical
  lock.

Important: the evaluator now requires `checkpoint_sha256` in the training
summary, but `train_flux2_lowlight.py` has **not yet been patched to emit it**.
That is the immediate code blocker and must be fixed before H100 handoff.

## Verification evidence and interrupted check

After the current patches:

- Python compilation passed.
- `bash -n smoke/run_lowlight_baseline_overfit.sh` passed.
- `git diff --check` passed.
- Four unit tests passed:
  - deterministic aligned crops;
  - rectified-flow formula/reference layout;
  - zero-init LoRA preserves baseline and scopes gradients;
  - LoRA checkpoint round trip.

The subsequent CPU preflight rerun was interrupted after unit tests when this
handoff was requested. Do not report preflight PASS yet; rerun it.

## Immediate next actions

1. Patch `train_flux2_lowlight.py` to:
   - emit and verify final checkpoint SHA-256;
   - record Git revision, FLUX revision, Python/PyTorch/CUDA/GPU environment,
     exact config, and output paths;
   - recover summary generation safely if interruption occurs after a
     checkpoint but before `metrics.jsonl` append;
   - use a fixed deterministic probe loss for the loss-decrease gate instead
     of comparing two unrelated stochastic minibatch losses;
   - fail closed rather than overwrite partial/differing checkpoint artifacts.
2. Add CPU tests for config rejection, deterministic index ordering,
   interrupted-summary recovery, non-finite metric rejection, and failed
   evaluation reuse.
3. Rerun compilation, shell syntax, full unit tests, CPU preflight with
   `--skip-large-hashes`, then full hash preflight if practical.
4. Write `reports/lowlight_stage1_architecture_audit.md`, explicitly labeling
   the flow-training recipe as project-derived because the BFL checkout is
   inference-only.
5. Update `agents.md` with verified CPU facts only.
6. Review/stage only intended files, commit, push, and verify remote SHA.
7. Only then ask the user to run exactly:

   ```bash
   CUDA_VISIBLE_DEVICES=0 smoke/run_lowlight_baseline_overfit.sh
   ```

8. Inspect:
   - `logs/lowlight_baseline_overfit16.log`
   - `checkpoints/lowlight_baseline_overfit16/training_summary.json`
   - `checkpoints/lowlight_baseline_overfit16/evaluation/metrics.json`
   - restored samples and contact sheets.
9. Do not authorize the full 8,000-pair run until automatic metrics and visual
   identity review both pass.

## Remaining major work after overfit

- Full 8,000-pair baseline training and fixed validation/internal-test
  evaluation.
- True Qwen-VL checkpoint acquisition/validation.
- Frozen Qwen-VL structured severity prediction and internal-test accuracy.
- Condition-token implementation and bounded 16-pair comparison; AdaLN and
  residual methods only as documented fallbacks.
- Fair baseline/Qwen/ground-truth-label diagnostic.
- Official FoundIR-Lowlight acquisition and protocol validation.
- PSNR/LPIPS/SSIM report with per-image metrics, outputs, failure cases,
  checkpoint hashes, configs, commands, revisions, and verified Git SHA.

The Goal is not complete until all of those items are proven. Do not mark it
complete based on CPU tests or the 16-pair smoke alone.
