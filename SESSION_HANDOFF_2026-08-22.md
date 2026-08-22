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
passed, the dimension-invalid GPU v1 contract was killed before inference,
and the first dimension-safe v1b GPU attempt preserved its completed caches
but failed before specialist inference because its preflight was not fail
closed**.

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

The remaining Gate 2 bundle is frozen to run one single-device process on each
of two visible H100/H200 GPUs:
official FoundIR generalist, documented generalist-to-Lowlight-specialist
cascade, historical FLUX, both metric paths, and the convex/frequency fusion
ceilings. It does not access protected FoundIR pixels.

The currently available GPU host was reported as four NVIDIA H100 80 GB GPUs,
55 CPU cores, and 900 GB RAM/shared memory. This is compatible with the
architecture-9.0 CUDA extensions. Gate 2 exposes GPUs 0 and 1 but preserves
the frozen one-device semantics inside each model process: the teacher pass
runs on GPU 0 while historical FLUX runs concurrently on GPU 1, then metrics
run on GPU 0 after both proposal caches pass. GPUs 2 and 3 remain available.
The frozen v1b Python runner SHA-256 values
are `6c0a608c97459a29694b2a4483748959b3e0bccebbdb1a23a00a730fe355212e`
(teachers), `67ed3d8365a811df9c08a61df8090cf60550923546e0d2e87531d02121f43233`
(historical FLUX), and
`fc9a13208d74bb9a8b0f0d8d19646687216ac19d6fe12e0cff7612a5ef7a0c11`
(metrics/blends).
The two-GPU launcher, concurrent bundle, and durable logging wrapper hashes are
`2235fbfc2dd12c2c4afe99589c22eb1e67b699e85ce62b5c752ec0276f9fca95`,
`733b67c73d4f95595ebb3289a1dda63e7c19f077f9199fd30a9cb89548436a1b`,
and `3cf889c3b3d1b918d203621d62a9dfc226f8b6c15c36428fead74ee9b8a09e3d`.

A pre-launch dimension audit found that all 18 outer SICE records are 8.30 to
24.00 megapixels, contradicting v1's assertion that none exceeded 1.048576
megapixels. No GPU attempt had started. The v1 entry is therefore preserved as
`killed_pre_inference_dimension_contract_mismatch`; v1b applies the released
FoundIR 1024/512 overlap policy to the generalist, bounded 512-pixel specialist
tiling, and a disclosed Lanczos downsample/restore policy only to the historical
synthetic FLUX baseline. Every saved proposal still has exact native dimensions.
Nineteen focused CPU tests cover the dimension-safe policies and prior gates.

The first v1b GPU attempt started at `2026-08-22T07:57:30Z`; its durable log is
`logs/foundir_lowlight_gate2_gpu_v1b_aZ2F9Z.log`. The generalist completed all
245 records and its fully rehashed manifest is
`manifests/foundir_lowlight_gate2_generalist_v1b.csv`, SHA-256
`71c6fdb8199f3b7b952fbade457b4bb05bc496bbe9e307413caddba0f2ad81b7`.
The specialist failed before producing any output because the mutable training
environment contained `transformers==5.14.1`, while the released FoundIR
environment freezes `transformers==4.47.1` and `tokenizers==0.21.0`. The same
error occurred during preflight, but the original multi-command preflight
block returned the final echo's status and incorrectly printed PASS. This is
an execution-contract defect, not a scientific model failure and not a
different-user GPU-instance issue.

The correction isolates the official FoundIR versions in workspace-local
dependency directories, asserts exact versions, makes both preflights fail
closed, and adds a cache-preserving resume. The resume rehashes all 245
generalist outputs, runs only the missing cascade, waits for the already
running historical FLUX result if necessary, and then runs metrics. It never
reruns the generalist or historical FLUX. The historical branch from the first
attempt is still advancing independently and will retain its immutable result.

## Exact next actions

1. Commit and push the fail-closed dependency correction, resume scripts, and
   completed generalist manifest.
2. Launch `CUDA_VISIBLE_DEVICES=0,1 bash
   smoke/launch_foundir_lowlight_gate2_gpu_resume.sh` once. It uses GPU 0 for
   the missing specialist and metrics and preserves the historical FLUX work
   already completed on GPU 1.
3. On Gate 2 completion, apply the frozen promotion/kill decision and proceed
   to Gate 3 only if allowed by the strategy stop rules.

## Gate progression

| Gate | Status | Result identity |
|---|---|---|
| 1. Protocol/development freeze | passed | `reports/foundir_lowlight_gate1_result_v1.json` |
| 2. Proposal/ceiling lattice | in progress; first GPU attempt cached generalist/historical, specialist resume prepared | `reports/foundir_lowlight_gate2_proposal_lattice_result_v1b.json` |
| 3. Compact control/causal Qwen | pending | pending |
| 4. FLUX endpoint decision | pending | pending |
| 5. Optional budget/full-scale winner | pending | pending |
| 6. Replication/deployment freeze | pending | pending |
| 7. Protected FoundIR event | blocked pending user authorization | none |
