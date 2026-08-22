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

Gate 2 — proposal/ceiling lattice: **passed after cache-preserving recovery**.

Gate 3 — compact control and credible VLM gate: **initial four-H100 bundle
complete; Qwen ordinal passed, compact output was not admitted because of an
SDSD LPIPS regression, and the frozen downstream causal matrix is pending**.

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
closed, and adds a cache-preserving resume. The historical branch from the
first attempt completed all 245 records and passed at `2026-08-22T08:47:07Z`;
its result is `reports/foundir_lowlight_gate2_historical_flux_result_v1b.json`.
The result SHA-256 is
`361cb232c2bddf1ce28fd4496cf49f29597e3656cbd73c427b83e09cd5f1c7d2`;
the manifest SHA-256 is
`10f7cd537723dc39e8d002a8d3a7bbff0171ab794a4f7a814436b613aee7f1a7`.

Resume preflight 1 then correctly failed before detaching because BasicSR's
top-level package eagerly imported its training-only losses and therefore
requested `pyiqa`, which specialist inference does not use. The exact log is
`logs/foundir_lowlight_gate2_gpu_resume_v1b_3s0vYe.log`. The corrected loader
now imports only the released WaveMamba architecture and registry, bypassing
BasicSR data/loss/model/train packages. This is supported by the first GPU
trace: every architecture module, including WaveMamba and its CUDA imports,
had loaded before the later losses import failed. The resume still rehashes
all 245 generalist outputs, runs only the missing cascade, consumes the
completed historical result, and then runs metrics. It never reruns the
generalist or historical FLUX.

The corrected resume passed at `2026-08-22T11:35:49Z`. The complete result is
`reports/foundir_lowlight_gate2_proposal_lattice_result_v1b.json`, SHA-256
`18ed75980d574d223162478a23ef1c41c0dbc52762ffae8e4b10edc3dc13f711`.
It contains 2,940 metric rows for 245 records x 12 candidates. Independent
rehashing verified all 245 generalist, cascade, and historical outputs and
their native dimensions. The maximum literal/`pyiqa` PSNR disagreement was
`0.00015742 dB`; the two LPIPS paths were identical in the frozen environment.

The provisional conservative base is `P1_FOUNDIR_CASCADE`: equal-family PSNR
`14.7152 dB`, worst-family PSNR `11.5087 dB`, equal-family LPIPS `0.3410`, and
worst-family LPIPS `0.4646`. It improves over raw input on every family but
does not pass the stricter teacher-value comparison against fixed exposure on
SDSD and SICE, so teacher-at-inference remains conditional rather than
promoted. The input-only photometric predictor is killed. Target-accessing
convex/frequency fusion has more than `+0.5 dB` PSNR headroom on every family
but fails the all-family LPIPS non-regression rule, so learned fusion is not
admitted at Gate 2. Full interpretation is frozen in
`reports/foundir_lowlight_gate2_result_2026-08-22.md`.

## Exact next actions

1. Preserve, audit, commit, and push the initial Gate 3 evidence.
2. Implement and run the already-frozen measured-only,
   oracle-plus-measured, measured-plus-Qwen, and
   measured-plus-shuffled-Qwen causal matrix with identical restoration
   weights. Reuse the complete compact checkpoint as the fixed base.
3. Stop for review before
   Gate 4 if Qwen misses the two-correction causal gate.

## Gate progression

| Gate | Status | Result identity |
|---|---|---|
| 1. Protocol/development freeze | passed | `reports/foundir_lowlight_gate1_result_v1.json` |
| 2. Proposal/ceiling lattice | passed; cascade is provisional base, photometric predictor killed | `reports/foundir_lowlight_gate2_proposal_lattice_result_v1b.json` |
| 3. Compact control/causal Qwen | Qwen ordinal passed; compact not admitted; causal matrix pending | `reports/foundir_lowlight_gate3_compact_result_v1.json` |
| 4. FLUX endpoint decision | pending | pending |
| 5. Optional budget/full-scale winner | pending | pending |
| 6. Replication/deployment freeze | pending | pending |
| 7. Protected FoundIR event | blocked pending user authorization | none |

## Continuation update — Gate 3 causal matrix frozen for launch

The initial Gate 3 bundle and evidence were committed and pushed at
`d937d443b9d448ca56a3359c71010faa5d8c73fb`. The next mandatory experiment is
now implemented and CPU-validated under
`configs/foundir_lowlight_gate3_causal_v1.json`.

It keeps the step-1200 compact restorer frozen and trains only one
zero-initialized global-tone condition adapter. One checkpoint is evaluated as
measured-only, oracle-plus-measured, measured-plus-Qwen, and
measured-plus-shuffled-Qwen. Qwen exports five-level soft canonical
posteriors, uncertainty, and a fixed projection of pooled frozen visual tokens
for all 674 development inputs. The fixed within-family shuffled arm crosses
scene/sequence groups. No protected FoundIR pixels are used.

CPU evidence:

- eight focused causal/compact tests pass;
- causal contract covers 429 training and 245 outer records;
- deterministic measured/oracle features cover all 674 records;
- frozen image-feature SHA-256:
  `8bb8059a077f8b7eb6e243a03696c030eaf7d88d878000cc0027fbf9f7580db4`;
- causal contract SHA-256:
  `ca30e194aca381bf58ffefc4e488e53f4313e76231eeca32732ade37cb84b539`.

After the causal implementation commit is pushed, launch from the workspace
on four H100/H200 GPUs with the following one-line command:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/launch_foundir_lowlight_gate3_causal.sh
```

The durable log is `logs/foundir_lowlight_gate3_causal_v1.log`. The bundle uses
all four GPUs for Qwen export, adapter training/inference, and metrics. Expected
wall time is approximately 20--35 minutes. Do not start Gate 4 unless
`reports/foundir_lowlight_gate3_causal_result_v1.json` passes every frozen
family-level causal and LPIPS check. A failure uses correction index 0 and is
reviewed before deciding whether either of the two allowed interface
corrections is scientifically justified.

The first causal GPU launch at `2026-08-22T13:03:39Z` failed before Qwen model
load because the causal runtime section omitted the export seed. No
development inference or scientific training began. The recovery adds only
the missing frozen seed plus a fail-closed CPU test for every execution field;
it does not consume either scientific Qwen/interface correction. Nine focused
tests now pass. The corrected contract SHA-256 is
`d70ef493834c0da6e4c2f519d96e5cfd300f23c0eaed20994979f29aa1b7a111` and
the corrected frozen image-feature SHA-256 is
`d0eb331fb57536f3bdd38b4ec5990faa41d807d28c6469490d37f49909f9684e`.
Relaunch with the same launcher after the recovery commit is pushed.

The corrected causal run then completed all scientific work: Qwen features
for 674 records, the step-400 adapter checkpoint, and 980 native-resolution
outer outputs. Metrics failed before evaluating the first record because the
causal runner selected the training environment, which does not contain the
already-frozen `lpips` package. The metric-only recovery selects the same
`.venv/foundir-metrics` environment used by Gates 2--3, preflights exact
`lpips==0.1.4` and `pyiqa==0.1.16`, validates all cached identities and 980
files, skips Qwen/training/inference, and runs only distributed metrics. This
also consumes no scientific correction. Relaunch the same launcher after the
metric-resume commit is pushed.

The metric-only recovery completed at `2026-08-22T13:41:27Z`. The causal gate
failed: oracle conditioning gained +2.1001/+1.9823/+0.7193 dB on
LOL-v2/SDSD/SICE, proving interface capacity, but Qwen gained only
`+0.0751/-0.5719/+0.1241` dB over measured-only and did not beat shuffled Qwen.
Qwen LPIPS also regressed by 0.00780 on LOL-v2 and 0.00812 on SDSD. Gate 4
remains blocked. Cheap attribution shows expected Qwen severity still
correlates with oracle log-gain by 0.303/0.789/0.761 on outer
LOL-v2/SDSD/SICE, so a structured monotone, uncertainty-shrunk,
correct-versus-shuffled residual decoder is justified as scientific interface
correction 1 of 2. Full interpretation is in
`reports/foundir_lowlight_gate3_causal_result_2026-08-22.md`.

## Continuation update — structured Qwen correction 1 frozen

The cheap attribution was converted into the first of at most two scientific
Qwen/interface corrections. It freezes the compact restorer and completed
measured adapter, reuses the previous measured/oracle outputs byte-for-byte,
and trains only a 12-parameter structured monotone tone calibration. The input
is limited to the five Qwen posteriors, expected severity, entropy, and margin;
the unstable generic 32-D visual projection is excluded. Uncertainty and
distance from training support can only shrink the correction. Correct and
within-family group-shuffled conditions are trained together with a shuffled
tether and correct-versus-shuffled/measured ranking losses.

The frozen contract is
`configs/foundir_lowlight_gate3_correction1_v1.json`, SHA-256
`24023ae3fcdc9be8d784656967b5cc5c138dccd42019a948699d44237d6325df`.
Fourteen focused tests and the 674-record CPU contract pass. No Qwen inference
is needed because the complete cache is reused. No protected FoundIR pixel was
accessed. Full rationale is in
`reports/foundir_lowlight_gate3_correction1_contract_2026-08-22.md`.

After the implementation commit is pushed, launch on the four-H100 host:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/launch_foundir_lowlight_gate3_correction1.sh
```

Expected wall time is approximately 10--20 minutes. If correction 1 fails,
run residual attribution before deciding whether correction 2 is scientifically
justified. Gate 4 remains blocked until the frozen causal threshold passes.
