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
- Starting HEAD and `origin/main`:
  `94932f480ccedd3a992d881c3a93beb6dcce9114`.
- Strategy status: frozen after 29 adversarial rounds.
- Active side-goal thread:
  `01a021ed-dd9c-7ef2-9659-d6dc7ac51376`.

## Current gate

Gate 1 — protocol and development freeze: **passed**.

Gate 2 — proposal/ceiling lattice: **ready after the isolated Gate 1 commit is
pushed**.

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

## Exact next actions

1. Rerun the Gate 1 freezer after the ledger completion transition, then run
   focused tests, compilation, JSON validation, and `git diff --check`.
2. Commit and push only the isolated Gate 1 files while preserving the
   user-modified 2026-08-18 handoff and unrelated untracked files.
3. Launch Gate 2 from the frozen proposal-lattice contract. No teacher
   inference starts before the Gate 1 push is verified.

## Gate progression

| Gate | Status | Result identity |
|---|---|---|
| 1. Protocol/development freeze | passed | `reports/foundir_lowlight_gate1_result_v1.json` |
| 2. Proposal/ceiling lattice | ready after Gate 1 push | `configs/foundir_lowlight_proposal_lattice_v1.json` |
| 3. Compact control/causal Qwen | pending | pending |
| 4. FLUX endpoint decision | pending | pending |
| 5. Optional budget/full-scale winner | pending | pending |
| 6. Replication/deployment freeze | pending | pending |
| 7. Protected FoundIR event | blocked pending user authorization | none |
