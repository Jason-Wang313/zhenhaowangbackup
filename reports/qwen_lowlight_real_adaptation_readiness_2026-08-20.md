# Qwen Lowlight real-adaptation readiness — 2026-08-20

## Status

**CPU readiness passed; GPU adaptation has not started in this coding-agent
terminal.** This gate is the corrective experiment after both frozen step-600
real-domain evaluations failed. It is not a FoundIR evaluation, a diffusion
result, or authorization to claim real-domain VLM success before its automatic
gate passes.

The two failures remain immutable evidence:

- free generation result SHA-256:
  `85542baf277c72a442e12d6da4f04e02051e45f7d586020472253437b6ebceda`;
- constrained-scoring diagnostic result SHA-256:
  `8cb62790a423438d2d62e096d0985075a3c4c2e179e950d2beb5a0e61ab44552`;
- both reports retain `status=failed` and `automatic_gate_pass=false`.

## Frozen experiment

- Config: `configs/qwen_lowlight_real_adaptation_step600.json`
- Config SHA-256:
  `0a817d70d88f172d6d16d37853ce9c3cd77308446b3eed3d5e5cdab0c3b3b541`
- Initial visual checkpoint: frozen synthetic-pilot step 600
- Initial visual SHA-256:
  `b0e65a8597077973a2a103dc3369e0cb1ea703de95e2e4841d359c3d34eff77f`
- Trainable scope: all 676,550,144 parameters in `model.visual`
- Frozen scope: the 7,615,616,512-parameter language decoder and LM head
- Optimizer: fresh AdamW state
- Training: 200 optimizer steps and a fixed global batch of 40
- Every step: 20 balanced real records plus 20 balanced synthetic-replay
  records; every microbatch contains L1, L2, L3, L4, and L5
- Schedule SHA-256:
  `8ae8e4e06a306c3828423374e39cecac4e3be2cf90b2b08564005b928a62c71f`
- Two GPUs: microbatch 5 per rank with four accumulation steps
- Eight GPUs: microbatch 5 per rank with one accumulation step
- The 2- and 8-GPU paths realize the same global record schedule, sample
  count, optimizer-step meaning, global batch, learning-rate schedule, and
  checkpoint-selection protocol.

Checkpoint selection uses minimum macro-per-level teacher-forced loss on the
real validation split, with an earlier-step tiebreaker. Only after selection
does the run generate predictions on real validation, synthetic replay, and
the fresh real test. The real test is evaluated once and was not used to form
the split, train the model, set thresholds, or select a checkpoint.

## Frozen real data

- Manifest: `manifests/qwen_lowlight_real_adaptation.csv`
- Manifest SHA-256:
  `91f95644f852f628891513f116c0a3607b8de980bbe5252c360ff41d9bcc9ab7`
- Metadata SHA-256:
  `2b9f35ffdc2543a4a99bf95fbb4eb61938a022c54cba4569a14863c5b536f06f`
- Annotation kind: `reference_based_proxy_not_human_ground_truth`
- Train: 673 records / 190 perceptual scene components;
  L1–L5 = `190/14/18/52/399`
- Validation: 143 records / 41 components;
  L1–L5 = `41/3/3/11/85`
- Fresh test: 145 records / 41 components;
  L1–L5 = `41/3/4/11/86`
- Exact overlap with previously evaluated real source files: zero
- LOL-v2 Train-versus-Test Normal pHash edges at Hamming distance at most 4:
  zero
- Official FoundIR used: false

The preflight verified all 1,378 extracted LOL-v2 training files and the
official 1,046,491,030-byte archive with SHA-256
`9820d8b112438d94d1f5d4d25817eee618a8cf4bc63a65cd6b19f5a98c3faefa`.

## Synthetic replay

- Frozen 8,000-record synthetic training pool bundle SHA-256:
  `dff705df26a863994cdc42105f9e4a9b2074556173fa72ce8c47266688a495f0`
- The world-size-invariant schedule presents 4,000 distinct replay records
  during adaptation.
- Frozen replay validation: 40 complete sources / 200 records
- Replay-validation bundle SHA-256:
  `1981939ac92de5412df990f39f9d7c2ee8776ec537e76dec059a81378407cfc3`
- The CPU preflight hashed all 4,200 scheduled/validation replay images.

## Verification completed

- Python compilation passed for the module, manifest builder, preflight,
  trainer, and tests.
- Shell syntax passed for the checked runner, tmux launcher, and short
  launcher.
- Workspace-local tmux 3.7 was verified at `.tools/tmux-env/bin/tmux`.
- Thirty focused Qwen tests passed.
- Contract-only launcher preflight passed for the two-GPU path.
- CPU preflight with only large base-model hashes skipped passed.
- CPU preflight with every base-model hash enabled passed.
- Both CPU preflights used `--no-write-artifact`; this deliberately reserves
  the immutable `preflight_full.json` for the GPU launcher, where CUDA must be
  visible.

The trainer validates every numeric field and internal weighted/macro
relationship in reused validation-loss artifacts. It also validates every
numeric training-step artifact, refuses differing immutable outputs, audits
gradient scope and frozen parameters, and preserves status 2 as a legitimate
automatic gate failure rather than a crash.

## Checked launch behavior

The short launcher requires one explicit device set:

- `CUDA_VISIBLE_DEVICES=0,1` for two H100s or H200s;
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` for eight H100s or H200s.

It verifies H100/H200 identity and at least 75 GiB per selected GPU, uses the
workspace Python and workspace-local tmux, checks that all scientific source
files are committed and unchanged, writes persistent attempt/event logs,
locks against duplicate jobs, samples every selected GPU at two-second
intervals, and leaves the tmux pane inspectable after exit.

For the currently available small adaptation gate, two H100s are a reasonable
immediate choice rather than waiting indefinitely for eight H200s. Eight H200s
remain supported and should reduce wall-clock time if they become available
before launch. No maximum-utilization or minimum-runtime claim is made before
telemetry from the actual run exists.

## Gate consequence

If the automatic adaptation gate passes, the validated VLM may advance to the
Lowlight diffusion-restoration conditioning stage after user review. If it
exits with status 2, preserve and diagnose the failed adaptation; do not tune
on the fresh real test, lower thresholds, start diffusion, or run FoundIR.
