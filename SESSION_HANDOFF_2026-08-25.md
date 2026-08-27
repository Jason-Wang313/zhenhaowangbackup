# Session handoff — 2026-08-25

## Mandatory startup

Work only in `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang` and begin
by telling the user exactly: `I am working in zhenhaowang, which is the correct
workspace.` Read `agents.md` and this handoff before acting. Preserve all
unrelated tracked and untracked user files. Verify `HEAD`, `origin/main`, and
the machine-readable artifacts directly rather than relying on chat history.

## Protected-data rules

- Do not access the FoundIR official test.
- Do not use FoundIR training data for this Qwen stage.
- The synthetic fixed test has not been accessed by Correction 4 and remains
  protected for exactly one evaluation only after all frozen validation gates
  pass.
- Do not lower gates or tune from a fixed-test result.

## Advisor-approved objective

Qwen must identify every present degradation and each independent strength:
blur, noise, rain, haze, and low-light, with L1 absent and L2--L5 increasing
severity. The downstream interface remains:

`Qwen JSON -> deterministic canonical restoration text -> FLUX text encoder -> FLUX attention`

The eventual restoration model is one FoundIR-trained all-in-one model. The
synthetic AgenticIR corpus is for Qwen supervision, not a substitute FoundIR
restoration dataset.

## Correction history

All four full validation variants failed at least one frozen gate and stopped
before fixed-test access:

| Variant | Macro-F1 | Set exact | Combined exact | Typed severity accuracy | Typed severity MAE | Triple set exact |
|---|---:|---:|---:|---:|---:|---:|
| Correction 1 | 0.9285 | 0.7842 | 0.6258 | 0.7074 | 0.5398 | 0.5500 |
| Correction 2 | 0.9314 | 0.7958 | 0.6333 | 0.7130 | 0.5296 | 0.5750 |
| Correction 3 | 0.9282 | 0.7900 | 0.6333 | 0.7162 | 0.5278 | 0.5806 |
| Correction 4 | 0.9287 | 0.7875 | 0.6075 | 0.6801 | 0.5694 | 0.5750 |

Correction 4 replaced variable-length JSON with five fixed slots:

`{"blur":1,"noise":1,"rain":1,"haze":1,"low_light":1}`

The full implementation was committed and pushed at
`9e1a838aba33b34b594fd8e5c4f666e04a4c3918`. Its frozen config is
`configs/qwen_multidegradation_lora_correction4_v1.json`, SHA-256
`b954bd3822b1250ee0571bec91d1a4c7d1e4503bb72017e952db28ebfc82768a`.

## Correction-4 GPU result

The user launched the run on two NVIDIA H100 80 GB GPUs. It ran from
`2026-08-25T04:14:58Z` to `2026-08-25T04:49:52Z` and completed all 600 steps.
Training took `1910.61` seconds; the final reported loss was `0.0582643`.
Gradient scope passed on both ranks. The runner exited with intentional status
2 after frozen validation failed; this was not a crash.

Validation passed structure (`0.9967`), macro-F1 (`0.9287`), micro-F1
(`0.9291`), and gradient scope. It failed:

- degradation-set exact: `0.7875 < 0.80`;
- combined exact: `0.6075 < 0.65`;
- typed severity accuracy: `0.6801 < 0.75`;
- typed severity MAE: `0.5694 > 0.50`;
- triple set exact: `0.5750 < 0.65`.

Micro precision was high (`0.9623`) but recall was only `0.8981`. Across the
five slots, 220 active types were missed versus 76 false positives. Fixed slots
therefore did not solve compound omission and regressed most decision metrics
relative to Correction 2.

Authoritative artifacts:

- result:
  `checkpoints/qwen_multidegradation_lora_correction4_v1/merger_trainable/validation_result.json`,
  SHA-256 `8972c5d9d4dd201b123286bd7eff4a9ca16e6ad5d3eb5b7aab635dd826e5029d`;
- predictions:
  `checkpoints/qwen_multidegradation_lora_correction4_v1/merger_trainable/validation_predictions.json`,
  SHA-256 `2b09e832d8fdf0c2f5e13a94f2968600c5c8c89fa4b6c3080329bf88a3dccd80`;
- adapter:
  `checkpoints/qwen_multidegradation_lora_correction4_v1/merger_trainable/checkpoints/G_fixed_slot_independent_decisions/adapter.safetensors`,
  SHA-256 `f5b2fd7d303142e8655c11457f943efb4c3bc01e2205eda8279b61f00b84afcf`;
- report: `reports/qwen_multidegradation_validation_correction4_2026-08-25.md`;
- log: `logs/qwen_multidegradation_lora_correction4_v1.log`.

No Correction-4 `fixed_test_result.json` or `fixed_test_predictions.json`
exists. Selection, real-sanity inference, and rendering did not run. FoundIR
was not accessed.

## Advisor-reviewed next route — ready for GPU launch

The prior stop condition has now been resolved by direct advisor review. The
advisor rejected independent classification heads because they would reduce
the dialogue VLM to an ordinary classifier. The advisor explicitly approved
the second dialogue route: ask the same image five separate questions, one for
each degradation, let the Qwen decoder answer each L1--L5 level, then aggregate
the answers. The advisor also required Qwen LoRA rank 32, a much larger batch,
and longer training with evidence of whether loss is still decreasing.

The new advisor framework image is
`f95684090e025c3de176065a2dfdf043.png`, SHA-256
`5b50d6e6d28a55949d2dda2217b152cdb8ae25af95db4400e5a23c037c2cfc12`.
The approved implementation is frozen in
`configs/qwen_multidegradation_per_type_rank32_v1.json`, SHA-256
`43af6977452d4d62a5845d18c3d55cbcd30c56d9e9c1e9f8019b76c3983d5f58`.

It starts fresh from the Qwen base, uses rank-32 visual and language-attention
LoRA plus the trainable visual merger, and has no classification head. Each of
9,600 training images produces five type-specific dialogues per epoch. Three
epochs give 144,000 presentations and 1,440 steps. Microbatch is 25 per GPU,
with fixed global batch 100 on both two and four H100s. The 50-step overfit
gate uses that same microbatch before full training, so an H100 memory issue or
dialogue-code defect fails early.

Readiness evidence is in
`reports/qwen_multidegradation_per_type_rank32_readiness_2026-08-25.md`.
Full hashes, Python/shell syntax, the rank-32 meta-architecture audit, and 47
focused tests pass. No new GPU output exists. The fixed synthetic test and
FoundIR remain unaccessed. After the readiness commit is pushed, the next
action is one launch of
`smoke/launch_qwen_multidegradation_per_type_rank32.sh` on either two or four
H100s, followed by autonomous monitoring.

## Per-type rank-32 GPU training and evaluation-recovery status

The user launched commit `b8c2630872f820cf94d2cde745c83466792357f6` on
two NVIDIA H100 80 GB GPUs on `2026-08-27`. Full hashes and the overfit gate
passed. Peak training memory was approximately 55 GB per GPU. The overfit
decoder achieved 1.0 exact type/level metrics on all 32 selected images,
including triples.

All 1,440 optimizer steps and 144,000 dialogue presentations completed. The
three epoch mean losses were `0.0774334`, `0.0405060`, and `0.0297576`; every
epoch logged `loss_still_descending=true`. The final H3 checkpoint was safely
published at `2026-08-27T06:15:38Z`.

All 6,000 validation decoder queries then completed, but evaluation crashed
before metrics were written. The cause was an evaluation-only cardinality
bug: five individually valid type answers could aggregate to four or five
active predictions, while `make_label` correctly restricts corpus ground truth
to at most three. The aggregator incorrectly reused that ground-truth
validator and raised instead of retaining extra predicted types as false
positives. Training and checkpoints were not affected; no protected fixed test
was accessed.

The recovery patch keeps ground-truth/corpus validation at clean through
triple, permits four/five-type *predictions* only, and counts their extra types
as false positives. It also adds a hash-checked `--validation-only` path. The
existing runner detects the completed H3 checkpoint plus absent validation
result and uses evaluation-only recovery, so no retraining occurs. After the
patch is pushed, relaunch the same two/four-H100 launcher once.

## Per-type rank-32 frozen validation result

Validation-only recovery eventually completed at `2026-08-27T10:00:36Z`.
The gate failed scientifically and the runner stopped with status 2 before
selection or protected fixed-test access. Overall per-query five-class
accuracy was 86.65%, valid-response rate 100%, type micro-F1 93.33%, precision
97.19%, and recall 89.77%. However, active L2--L5 exact accuracy was 65.51%,
combined type/level exact was 50.83%, and triple set exact was 55.0%. Thus the
advisor's stated 80%+ overall accuracy expectation was met, but the project's
stronger active-severity and compound exact gates were not.

The authoritative report is
`reports/qwen_multidegradation_per_type_rank32_validation_2026-08-27.md`.
Do not access the fixed test, weaken the frozen validation gates, or authorize
this Qwen checkpoint for FLUX until the advisor resolves which success
definition is authoritative.
