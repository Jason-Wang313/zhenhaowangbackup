# Qwen Lowlight supervised 8k DDP8 pilot readiness — 2026-08-19

## Result

**CPU/static readiness passed. GPU training has not started.** This artifact
prepares only the source-disjoint synthetic Lowlight supervised-VLM pilot. It
is not a real-domain result, a FoundIR result, or an SOW acceptance claim.

The user explicitly instructed `skip scaling, prepare 8k`. The skipped formal
1/2/4/8 scaling gate is recorded by SHA-256
`9e19be42fc58f99f30ed8c6f31f3996a03a0b91a36ef4fa60149a40b4f8e59d1`.
The earlier scaling attempt stopped safely in preflight and is not used as a
benchmark result. The selected eight-GPU execution is instead bound to the
already passing balanced DDP8 overfit gate.

## Frozen experiment

- model: `Qwen/Qwen2.5-VL-7B-Instruct`, revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- objective: teacher-forced token-level cross-entropy on deterministic
  structured degradation JSON
- trainable: complete `model.visual` path, `676,550,144` parameters
- frozen: language model and LM head, `7,615,616,512` parameters
- hardware: all eight H200 GPUs, explicit
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- per-rank microbatch: one complete L1–L5 source group, five records
- global batch: 40; gradient accumulation: 1
- train: 8,000 records / 1,600 sources
- validation: 1,000 records / 200 sources
- untouched internal test: 1,000 records / 200 sources
- schedule: 200 steps per epoch, five epochs, 1,000 optimizer steps, 40,000
  sample presentations
- learning rate: 50-step warmup to `2e-5`, cosine decay to `2e-6`
- immutable checkpoints: steps 50, 200, 400, 600, 800, and 1,000
- validation teacher-forced loss: steps 0, 200, 400, 600, 800, and 1,000
- model selection: minimum validation loss only, with earlier-step tiebreak
- generation: full validation once and full test once, only after checkpoint
  selection

The internal test split cannot affect training or model selection. FoundIR is
not read by this pilot.

## Fixed hashes and passing prerequisite

- pilot config:
  `28b243cf3a98aee12aa271334baa50e1a2bc5937467f9f3781acbd6e93d35dff`
- manifest:
  `e1ea9132bc0d5e55b30998c51c8e37a0363f29b2a23e7851ee804f3100d4a784`
- train bundle:
  `dff705df26a863994cdc42105f9e4a9b2074556173fa72ce8c47266688a495f0`
- validation bundle:
  `440066c2e6db631028d44f7856b4d053d5be3450548ea4a8e403bfcc5eb0ed86`
- test bundle:
  `1e66ab2825b5f60b50c27cd673d8d716e5f1b2f4fa375a661fd22010bb4b4e73`
- passing DDP8 overfit config:
  `a19ead5a89975292ebc83604b42989d37553d90db707838b1db6ca8f6f5bec2b`
- passing DDP8 overfit summary:
  `edfb655d613deae3af8bd2ab5047201060c9eaa363485f4f280c32fef9387c94`
- passing DDP8 final visual checkpoint:
  `ad42468115b42bc1bf076cdff4821a5ca82dd3a20f77bc3d09cbbb2d54b80e6f`
- observed prerequisite DDP8 throughput: `15.195928848911935` global
  samples/second; this is operational evidence, not a formal scaling result

## Preflight evidence

Both immutable preflight modes passed with Python `3.12.13`, PyTorch
`2.7.0+cu128`, and Transformers `4.56.1`.

- all 10,000 unique selected image files passed exact SHA-256 checks
- verified image bytes: `23,254,448,351`
- verified image-file bundle SHA-256:
  `46930464b2313c9176c26d38afdae261bf31b71a00f0ff943b8dffb2e3e0f4a2`
- all 16 frozen Qwen inventory files passed size and full SHA-256 checks
- all 729 checkpoint tensors matched the audited parameter groups
- each train rank receives 200 complete source groups; each validation and
  test rank receives 25 complete source groups
- processor, fixed prompts, assistant-only label masking, structured targets,
  and visual-token counts passed
- skip-model-hash preflight artifact SHA-256:
  `f4dda5bdc722025061070732a637d3a8dd9dd50dea4d7b2fc7cdbd08ae88c722`
- full preflight artifact SHA-256:
  `cca4842c6e402dc5ed838bb431d90d313e60787fbc3556eac4a7932acd40cc78`

Runtime preflight artifacts are under the ignored output directory
`checkpoints/qwen_lowlight_supervised_pilot_8k_ddp8/` and are revalidated by
the GPU runner.

## Implementation and failure behavior

The trainer provides deterministic fixed rank shards with local epoch
reshuffling, first-step full gradient-scope audit, immutable resume
checkpoints, a frozen-decoder audit at every checkpoint, epoch-level
distributed validation loss, validation-only checkpoint selection, batched
distributed generation, predictions/confusions/failure examples, and a final
machine-readable gate summary.

The persistent runner:

- refuses to run unless all relevant files are tracked and equal to `HEAD`
- requires exactly eight H200 GPUs and the explicit device set
- uses the workspace-local validated Python and offline model files
- holds an exclusive lock to prevent duplicate launches
- writes canonical and per-attempt logs, UTC/event markers, real exit status,
  and two-second telemetry for all eight GPUs
- resumes only from verified immutable scheduled checkpoints
- returns status `2` for a legitimate automatic metric-gate failure and
  records that status distinctly from an infrastructure crash

The tmux launcher uses `.tools/tmux-env/bin/tmux`, retains the session after
the command returns, avoids duplicate launches, and attaches automatically so
no separate monitoring command is required. Closing the browser or VS Code
terminal does not terminate the job.

## CPU validation performed

```text
16 pilot/supervised/distributed unit tests: PASS
Python compilation: PASS
Bash syntax: PASS
pilot runner --check: PASS
tmux launcher --check: PASS (tmux 3.7)
skip-model-hash preflight: PASS
full-model-hash preflight: PASS
git diff --check: PASS
```

The next and only authorized action is the single 8k synthetic pilot launch.
After it reaches a terminal result, stop and review its evidence before any
real-domain validation gate.
