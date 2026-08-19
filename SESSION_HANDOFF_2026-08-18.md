# Session handoff — 2026-08-18

This file is the current continuation guide for a fresh agent. Read it after
`agents.md`; the `READ FIRST` section in `agents.md` is authoritative if any
older experiment note conflicts with this document.

## Workspace and scope

- Correct workspace: `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`.
- Work only inside this workspace.
- The active responsibility is the **VLM Guided All-in-One image-enhancement
  workstream**. The separate `亮色大模型` workstream is out of scope.
- Branch is `main`. Latest verified local/remote commit at handoff:
  `8d81b30a13741abc493857b15d6ed13c7c2811e1`.
- Preserve unrelated historical/untracked files unless the user gives an
  exact deletion scope and confirms it through the repository's two-round
  destructive-change procedure.

## Advisor-authoritative architecture

The tracked advisor diagram is
`advisor_vlm_guided_all_in_one_main_route.jpg` with SHA-256
`02e32805142e64a7c191469fa4fbe4d4169914b25e3fa9df75853d4cde266194`.
It is the main-route authority.

The intended route is:

1. Train a degradation-aware VLM with supervised token-level
   cross-entropy. Input is a degraded image plus a degradation-assessment
   question. Targets identify degradation type(s) and independent severity
   level(s). Train the vision encoder and multimodal connector/path; keep the
   tokenizer and LLM decoder frozen, following the advisor diagram.
2. Validate the trained VLM on the downstream/real image-enhancement domain
   that it will actually analyze. Synthetic validation alone is insufficient.
3. Freeze the validated VLM. In the restoration stage, provide the degraded
   image to both the frozen VLM and the trainable diffusion restoration model.
   Use the VLM's **full structured degradation description** (types and
   severities) through a condition adapter. A single learned scalar/token is
   only a prototype, not the intended architecture.
4. Freeze the complete system and run the official FoundIR evaluation.

In this handoff, the two halves of the advisor diagram are called the
**VLM stage** and the **diffusion restoration stage**. They must not be called
“Phase 1” and “Phase 2.” The terms **SOW Stage 1** and **SOW Stage 2** are
reserved for the contractual timeline below.

The full degradation description belongs to the diffusion restoration stage.
It does not mean that diffusion training must start now. The immediate work
is the SOW Stage 1 supervised-VLM pilot and real-domain validation; the
diffusion interface may be scaffolded but should not be used for the main
claim before the VLM passes its pilot.

## SOW Stage 1 versus SOW Stage 2

These are contractual project stages, not the two halves of the advisor
diagram:

- **SOW Stage 1 (`T–T+6`)**: research and architecture selection, synthetic
  data construction, preliminary VLM and image-enhancement training, and
  initial validation. The current execution is targeting this stage first,
  with Lowlight as the independent gate.
- **SOW Stage 2 (`T+6–T+12`)**: expand and improve the data and training
  pipeline, optimize the unified model across Lowlight, Blur, Noise, Haze, and
  Rain, complete subjective/objective evaluation, and deliver the mature
  reproducible algorithm packages and reports. It is not merely “run the same
  test for better numbers,” although improved FoundIR metrics are part of its
  acceptance criteria.

The project ultimately targets both SOW stages. The immediate fresh-thread
milestone is only the SOW Stage 1 supervised-VLM pilot; do not claim SOW Stage
2 completion from it.

## Current execution scope and expansion order

The current execution window is **Lowlight only**. Do not start the all-five
task corpus or all-five-task diffusion training yet. Use Lowlight as a
vertical slice to validate the data format, supervised VLM, real-domain
generalization procedure, diffusion conditioning interface, and evaluation
bookkeeping before spending GPU time on broader training.

The planned order is:

1. Lowlight synthetic supervised-VLM pilot (8k train / 1k validation / 1k
   internal test).
2. Lowlight VLM validation on a separately labeled real/downstream set that
   is not FoundIR.
3. Lowlight diffusion development and conditioning comparison.
4. Freeze the Lowlight candidate and run the official FoundIR-Lowlight
   evaluation.
5. If the Lowlight slice is technically sound, expand the same validated
   pipeline to Blur, Noise, Haze, and Rain, then to mixed degradations and the
   larger corpus. Validate each domain and finally run the complete official
   FoundIR task suite.

Shared schemas, interfaces, and balanced multi-degradation data code may be
designed during the Lowlight slice, but the 100k corpus and broad five-task
training should wait until the Lowlight pilot clears its gates.

## Mandatory gate-stop and CPU/GPU handoff rule

After **each** numbered gate above, the fresh agent must stop, summarize what
was actually completed, report pass/fail evidence and remaining risks, and
wait for the user's review before starting the next gate. It must not chain
all five gates from one prompt or silently continue after a failed gate.

The coding-agent terminal may be CPU-only. The newly provisioned user GPU
environment is reported as **169 CPU cores, 1,800 GB system RAM, and 8 NVIDIA
H200 GPUs** (expected 141 GB HBM per GPU; verify with `lscpu`, `free -h`, and
`nvidia-smi` before the first run). Both environments share this same project
workspace, so code, manifests, checkpoints, logs, and reports written here
are visible to the GPU environment. For every gate requiring a GPU, the agent
must provide the user with one exact copy-paste **tmux command**. The command
must create or attach to a named tmux session before launching training, so
closing a VS Code/browser tab cannot terminate the GPU job. It must include the
working directory, `CUDA_VISIBLE_DEVICES` setting when relevant, launcher path,
and log/output location, plus the matching tmux attach/log-monitor command. The
agent must not claim that GPU training is running unless it has directly
verified a live process or log.

## Future GPU-scaling policy and current provision

The current future-capacity target is one flexible **8×H200** instance with
approximately **169 CPU cores and 1,800 GB RAM**. This is sufficient for the
planned 100k-total-image, five-degradation experiments under the intended
parameter-efficient VLM/diffusion training route. The extra H200 memory should
be treated as capacity and throughput headroom, not as permission to change
the scientific experiment silently.

GPU time should be minimized by using the provisioned hardware efficiently.
For each substantial experiment, measure and improve data-loader throughput,
GPU utilization, batch/gradient-accumulation settings, precision, checkpoint
frequency, and distributed communication overhead. Avoid leaving H200s idle
when the job can safely use them; at the same time, do not increase global
batch size, learning rate, resolution, or sample count without documenting the
change and preserving a fair comparison.

Use all eight H200s for the later large-corpus and all-five-task jobs when the
distributed implementation and effective-batch semantics have been validated.
Use a subset for small smoke tests when that is more efficient. Before scaling
from a subset to all eight, record samples/second, wall-clock time, per-GPU
memory, utilization, communication overhead, seed, global/effective batch,
gradient accumulation, optimizer/scheduler, precision, and steps/samples
seen. H200s are expected to provide substantially more memory headroom than
the earlier H100 plan; they will not automatically produce proportionally
better model quality.

Every GPU launcher must make the intended device set explicit (for example,
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` for all eight GPUs), state whether the
run is a smoke, pilot, or full experiment, and write a unique log/output
directory. The launcher must run inside tmux, and the handoff must include a
reconnection command such as `tmux attach -t <session>`. Never claim that all
eight GPUs were utilized without checking `nvidia-smi` or equivalent live
evidence.

## What has already been completed

### Synthetic Lowlight data and labels

- AgenticIR is the approved synthetic-degradation basis.
- GenDeg is abandoned and removed from active paths; do not restore or restart
  it.
- Project labels are L1–L5. L1 is clean after shared preprocessing; active
  degradation operators use only L2–L5.
- Clean-only records use `{"degradations": [], "levels": {}}`.
- Never encode `low_light L1`, `noise L1`, etc.; L1 is not an active operator
  level.
- The proposed future 100k corpus policy is a project choice, not an
  AgenticIR fact: 10,000 clean-only, 30,000 single, 30,000 double, and 30,000
  triple records. AgenticIR's relevant precedent is the 1:1:1 balance among
  single/double/triple degraded records. Do not generate the 100k corpus until
  the small supervised VLM pilot succeeds.
- Current pilot split is intended as 8k train, 1k validation, and 1k internal
  test from source-disjoint synthetic Lowlight data. Existing manifests are
  evidence; inspect their hashes before use.

### FLUX.2 restoration baseline

The internal synthetic Lowlight baseline is complete:

- test PSNR: `24.7722845`
- SSIM: `0.8858240`
- LPIPS: `0.0440393`

The visual identity gate passed, although extremely dark/uniform crops remain
known failure cases. This is an internal synthetic result, not FoundIR.

### Frozen-Qwen experiments (historical negative evidence)

These were not genuine Qwen fine-tuning:

- Frozen `Qwen/Qwen2.5-VL-7B-Instruct` produced 3,584-dimensional image-token
  features.
- Mean/std pooling yielded 7,168 features.
- A 7,169-parameter scalar ridge head (`alpha=3000`) predicted rounded/clipped
  L1–L5 severity.
- Independent synthetic validation: 68% severity accuracy, MAE 0.332;
  simple luminance baseline: 64%, MAE 0.432; L4 recall 44%; structured-output
  validity 100%.

The one-token Qwen-conditioned FLUX.2 ablation used the frozen ridge
predictions and the same restoration backbone:

- PSNR: `24.7723 -> 24.8091` (`+0.03683 dB`)
- SSIM: `0.885824 -> 0.885987` (`+0.000163`)
- LPIPS: `0.0440393 -> 0.0436826` (`-0.000357`)
- No convincing visual improvement.

Treat this as a near-null internal ablation. It is not a FoundIR result, not
the advisor-directed VLM route, and not evidence that Qwen as a technology is
useless. Do not spend more GPU time optimizing it unless the user explicitly
re-authorizes it as a diagnostic.

### FoundIR

- Official FoundIR was acquired and validated.
- Isolated category: `14Lowlight`.
- Exactly 50 official pairs, IDs `1101`–`1150`.
- No exact overlap with the 2,000 synthetic training sources.
- Official manifests/protocol are frozen.
- No restoration model has yet been run on FoundIR at this handoff.

FoundIR provides paired restoration targets, but it does **not** provide
official project L1–L5 degradation labels. Therefore:

- Use FoundIR for final restoration PSNR/SSIM/LPIPS and qualitative review.
- Do not report project-level L1–L5 accuracy on FoundIR without a separately
  declared annotation protocol.
- For iterative VLM generalization testing, create a small separate real
  Lowlight validation set with human labels (preferably two raters and an
  agreement statistic). If that is impractical, use a predeclared
  reference-based rubric and explicitly call it a proxy severity annotation,
  not official ground truth. Keep FoundIR protected from tuning.

## Current truth and interpretation

The baseline restoration model is a useful completed foundation. The current
Qwen artifact is a weak, frozen-feature linear probe and its restoration
conditioning effect is practically negligible. The project is not blocked;
it is at the point where the architecture must be corrected to match the
advisor's supervised-VLM route.

The old handoff's instruction to run an expanded frozen-feature ridge attempt
is superseded by the advisor route and this file. Do not launch that old
experiment merely because an old log or script exists.

## Exact next implementation milestone

### 1. Audit the local Qwen module

Inspect the installed `transformers` Qwen2.5-VL implementation and record:

- image processor and chat/template format;
- vision encoder module names;
- multimodal projector/connector module names;
- language-model decoder module names;
- parameter counts and which modules can receive gradients;
- dtype/device behavior on the H100 environment.

Do not modify external environments or download a second model copy without
checking available disk and model paths.

### 2. Implement a genuine supervised VLM trainer

Required behavior:

- image plus a fixed degradation question;
- deterministic structured target text, for example JSON containing
  `degradations` and `levels`;
- teacher-forced cross-entropy over target tokens;
- trainable vision encoder and multimodal connector/path;
- frozen tokenizer and LLM decoder, as specified by the advisor route;
- checkpoint, config, manifest hash, model revision, seed, trainable/frozen
  parameter counts, and per-step loss logging.

Do not describe a linear head or frozen feature extractor as VLM fine-tuning.

### 3. Run a tiny H100 overfit smoke first

Use 32–100 examples. The smoke must demonstrate:

- labels and JSON serialization are correct;
- image preprocessing is correct;
- gradients reach the intended trainable modules;
- frozen modules have no gradients/updates;
- loss decreases and the model memorizes the tiny set;
- resume/checkpoint behavior is real and documented.

This smoke is a code/gradient gate, not a scientific result.

### 4. Run the source-disjoint supervised pilot

Use the existing approximately 10k synthetic Lowlight pilot split:

- 8,000 train;
- 1,000 validation;
- 1,000 internal test.

Report type accuracy, per-severity accuracy/MAE, confusion matrix, exact
structured-output validity, and failure examples. Keep the split fixed once
the run begins. Validation is useful here even though the advisor said it may
be omitted in some step-based training code; this pilot needs it for model
selection and debugging.

### 5. Validate on real/downstream data

Before diffusion claims, test the frozen pilot VLM on a separately labeled
real Lowlight/enhancement-domain set. Record the annotation rubric, rater
agreement (if human-labeled), source hashes, and whether labels are exact or
proxy. Do not tune on FoundIR.

### 6. Only after the VLM passes, start diffusion restoration conditioning

Freeze the validated VLM and condition diffusion on the structured type and
severity description. Compare against the already completed FLUX.2 baseline
under the same data, split, optimizer, steps, precision, seed, and sampling
protocol. Preserve a ground-truth condition run only as an explicitly labeled
oracle diagnostic.

### 7. Run FoundIR last

After the complete system is frozen, run the official FoundIR-Lowlight
protocol. The SOW Stage 1 Lowlight sub-target is PSNR `>=15.1 dB` and LPIPS
`<=0.34`. A Lowlight pass is not complete five-task All-in-One SOW
acceptance; Blur, Noise, Haze, and Rain remain separate tasks.

## Rules for reporting and fresh-agent behavior

- Lead with pass/fail and distinguish synthetic internal metrics from official
  FoundIR evidence.
- Never claim the ridge head was Qwen fine-tuning.
- Never claim a tiny baseline-vs-Qwen metric delta is a meaningful improvement.
- Give one exact copy-paste GPU command at a time and state whether a process
  is actually running; do not claim tmux/checkpoint resume without checking.
- Preserve manifests, logs, checkpoints, hashes, and failure reports.
- Ignore the known `pkg_resources is deprecated` warning unless a separate
  traceback identifies a real failure.
- Before every work session tell the user: “I am working in zhenhaowang,
  which is the correct workspace.”

## Repository state at creation

Tracked commits:

```text
8d81b30 Track advisor VLM restoration route diagram
d7f5fc8 Set advisor VLM route and retire GenDeg
be699c9 Clarify 100k Qwen corpus policy
```

Known unrelated untracked evidence/scripts include old manifests, audit
reports, and historical smoke/download helpers. Leave them untouched unless
the user explicitly requests a scoped cleanup.

The next agent should begin with the Qwen module audit and then implement the
tiny supervised-VLM overfit smoke. It should not start the 100k data build,
the old ridge experiment, diffusion training, or FoundIR tuning first.
