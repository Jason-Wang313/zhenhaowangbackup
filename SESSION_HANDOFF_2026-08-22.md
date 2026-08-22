# Live execution handoff — 2026-08-22

This is the live operational continuation file for the autonomous
FoundIR-Lowlight execution. The scientific authority is
`reports/foundir_lowlight_adversarial_final_strategy_2026-08-22.md`; the
workspace and advisor authority remain `agents.md`.

## Immutable scope

- Workspace: `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`.
- Execute strategy Gates 1--6 autonomously.
- Do not open, infer on, calibrate from, or otherwise access the protected
  FoundIR-Lowlight IDs `1101--1150` during Gates 1--6.
- Pause before Gate 7 for explicit user authorization.
- Preserve the user-modified `SESSION_HANDOFF_2026-08-18.md` and every
  unrelated untracked file.
- Lowlight only; no Blur, Noise, Haze, Rain, mixed-degradation, or separate
  bright/color-model execution.

## Current repository identity

- Branch: `main`.
- Gate 1 HEAD and `origin/main`:
  `fb5c44af412593e9bdd9b2c21a033cfb639cc496`.
- Strategy status: frozen after 29 adversarial rounds.
- Active side-goal thread:
  `01a021ed-dd9c-7ef2-9659-d6dc7ac51376`.

## Current gate

Gate 1 — protocol and development freeze: **passed**.

Gate 2 — proposal/ceiling lattice: **in progress; CPU photometric lattice
passed and the one-shot GPU remainder is being frozen**.

Verified at start:

- the coding-agent terminal has no `nvidia-smi` executable and no visible GPU;
- the current frozen metric environments contain neither `pyiqa` nor the
  direct `lpips` package;
- the official FoundIR environment pins `lpips==0.1.4` but leaves `pyiqa`
  unpinned;
- the official SDSD repository was cloned for provenance/terms audit at
  revision `07330bb0ac124d2d56f6e92908d650dde3f7f7c2`;
- its README explicitly provides indoor/outdoor research download links and
  describes 70 indoor plus 80 outdoor paired sequences, but exposes no dataset
  redistribution license;
- SDSD was acquired from Hugging Face mirror `Guanyanan/SDSD` at pinned
  revision `fb6d4310fa07eb0c135dba01be6ca086ecd31009`; three mirror byte ranges
  were identical to previously acquired official Google Drive bytes. The
  frozen selective suite contains 500 pairs from 150 sequence groups;
- official FoundIR teacher weights remain hash-audited and no teacher inference
  has started. The Lowlight specialist's earlier 61-character transcription
  was corrected from the immutable checkpoint to SHA-256
  `f834248e145d020d5a02f257b7ceb383b579add2522932938b3e1e0bd460abe6`.

Operational interpretation:

- SDSD may be used internally for the documented research experiment, but it
  must not be redistributed or packaged;
- metric versions and weights must be installed into a workspace-local,
  reproducible environment and frozen before proposal inference;
- GPU-required work will use checked launchers in the shared GPU terminal
  unless GPU visibility changes.

Gate 1 froze 674 records / 232 groups: LOL-v2 outer 102/41, SICE
training/outer 54/18 records across 31/10 scenes, and SDSD training/outer
375/125 records across 125/25 sequences. No scene, sequence, input hash, or
target hash crosses training and outer roles. The 245 outer records are
reusable development, not a new hidden test. Teacher-at-inference remains
conditional because development overlap is unverifiable, checkpoint-specific
redistribution terms are not separately stated, and runtime/value await Gate 2.

## Protected-test status

`NOT_ACCESSED_DURING_AUTONOMOUS_EXECUTION`

The existing official manifest may be used only as a hash/ID exclusion index.
No protected image pixels or public restored outputs may be opened.

## Gate 2 live evidence

The CPU photometric stage completed on 429 frozen training records and all 245
outer-development records. It cached seven proposals / 1,715 exact PNG outputs.

- config: `configs/foundir_lowlight_gate2_photometric_v1.json`, SHA-256
  `a1fed1aadfaf69db7a02b72bf9523cd85ada7f2b65a54f989a562ece2f38fdc2`;
- manifest: `manifests/foundir_lowlight_gate2_photometric_v1.csv`, SHA-256
  `1176ad4c481b0a1a0dc5fd1974bda0b9dcd25abfb39be89cd322c211b63c2680`;
- result: `reports/foundir_lowlight_gate2_photometric_result_v1.json`, SHA-256
  `a097d0267256fcbc2a4363457dd98a84d9b90befa21b83dfc3c1857fcb296909`.

The bounded input-only photometric predictor failed transfer: equal-family
PSNR `7.6557 dB`, including `5.5939 dB` on SICE, versus the raw-input
equal-family `10.4902 dB`. It is killed without a correction. This is not a
lack-of-headroom result: target-accessing O2 monotone tone/WB reached
`25.7409 dB` equal-family / `22.8427 dB` worst-family, and O3 smooth
illumination reached `25.8192 / 22.8886 dB`. The bottleneck is inference-time
identifiability and target-render transfer, not the existence of a useful
photometric correction.

The remaining Gate 2 bundle is frozen to run on exactly one visible H100/H200:
official FoundIR generalist, documented generalist-to-Lowlight-specialist
cascade, historical FLUX, both metric paths, and the convex/frequency fusion
ceilings. It does not access protected FoundIR pixels.

The currently available GPU host was reported as four NVIDIA H100 80 GB GPUs,
55 CPU cores, and 900 GB RAM/shared memory. This is compatible with the
architecture-9.0 CUDA extensions. Gate 2 deliberately selects one physical GPU
with `CUDA_VISIBLE_DEVICES=0`; the process therefore sees exactly one device
while the other three remain available. The frozen GPU runner SHA-256 values
are `db1218d442949f2ceb20657ebe8bc9e34bb63fff1f1b73b04379214349bf1038`
(teachers), `ee57ab3717e5c889f4cc3377e12ae1bc0546137b96aa06df82c5b10a0e47467e`
(historical FLUX), and
`da384a37dc7a36b895a55d0c60d2c042b594704b89a69bc543bb4c524d57c043`
(metrics/blends).

## Exact next actions

1. Commit and push the Gate 2 partial checkpoint before GPU inference. The
   workspace-local WaveMamba CUDA dependencies are compiled for architecture
   9.0 and the generalist checkpoint passed strict CPU loading; the launcher
   performs the remaining GPU imports before starting inference.
2. Launch `smoke/launch_foundir_lowlight_gate2_gpu.sh` once on one visible
   H100/H200. The launcher performs three environment/GPU preflights before
   detaching and uses append-only attempt logs, so a failed attempt does not
   overwrite evidence.
3. On Gate 2 completion, apply the frozen promotion/kill decision and proceed
   to Gate 3 only if allowed by the strategy stop rules.

## Gate progression

| Gate | Status | Result identity |
|---|---|---|
| 1. Protocol/development freeze | passed | `reports/foundir_lowlight_gate1_result_v1.json` |
| 2. Proposal/ceiling lattice | in progress; CPU stage passed | `reports/foundir_lowlight_gate2_photometric_result_v1.json` |
| 3. Compact control/causal Qwen | pending | pending |
| 4. FLUX endpoint decision | pending | pending |
| 5. Optional budget/full-scale winner | pending | pending |
| 6. Replication/deployment freeze | pending | pending |
| 7. Protected FoundIR event | blocked pending user authorization | none |
