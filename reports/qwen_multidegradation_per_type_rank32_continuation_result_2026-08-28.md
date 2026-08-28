# Qwen per-type rank-32 continuation result — 2026-08-28

## Decision

The advisor-directed continuation completed through total epoch 8. Longer
training did not solve compound-degradation omission. None of the frozen epoch
5, 6, or 8 candidates passed validation, so no checkpoint was selected and
the protected synthetic fixed test and real-image sanity stage were not
accessed.

This result rejects the specific hypothesis that the prior 55% strict triple
recognition was mainly caused by training for only three epochs. The training
objective continued improving, but the validation metrics plateaued.

## Training completion

The run resumed the exact epoch-3 rank-32 LoRA and AdamW state. It added five
complete epochs, 1,200 optimizer steps, and 240,000 per-type dialogue
presentations. The cumulative experiment reached 2,640 optimizer steps and
384,000 dialogue presentations.

Added-epoch mean losses were:

| Total epoch | Mean loss | Within-epoch descending |
|---:|---:|:---:|
| 4 | 0.02635093 | yes |
| 5 | 0.02573352 | yes |
| 6 | 0.02535615 | no |
| 7 | 0.02517450 | no |
| 8 | 0.02513574 | yes |

The mean loss fell only 4.6% from epoch 4 to epoch 8 and flattened near 0.025.
All trainable visual LoRA, visual merger, and language LoRA parameters received
gradients; frozen parameters remained frozen. Training took 24,541 seconds
(6 h 49 min) on the original 2-H100 run.

## Frozen validation comparison

Each checkpoint answered five separate questions for each of 1,200
source-disjoint synthetic validation images, producing 6,000 decoder answers.

| Metric | Epoch 5 | Epoch 6 | Epoch 8 | Gate |
|---|---:|---:|---:|---:|
| Valid answer rate | 100.00% | 100.00% | 100.00% | 100% |
| Per-query L1–L5 accuracy | 86.68% | 86.68% | **86.72%** | >=80% |
| Type micro-F1 | 93.21% | 93.22% | **93.29%** | >=92% |
| Active L2–L5 accuracy | 66.02% | **66.16%** | **66.16%** | >=75% |
| Degradation-set exact | 78.83% | 78.92% | **79.08%** | >=80% |
| Combined type-level exact | 51.00% | **51.17%** | 51.08% | >=65% |
| Triple degradation-set exact | 55.83% | **56.11%** | 55.83% | >=80% |

Epoch 6 ranked first because the frozen selector prioritizes strict triple-set
exact, but 56.11% remains 23.89 percentage points below the required 80%.
Training from epoch 3 to epoch 8 changed strict triple exact only from 55.00%
to 55.83%; the best intermediate result was 56.11%. The gains are too small
to justify more identical epochs.

The high overall per-query accuracy and micro-F1 remain partly driven by easy
L1 absent decisions. The scientifically important active-severity and exact
compound metrics still fail. Therefore this checkpoint is not authorized as
the Qwen condition source for FLUX.

## Recovery incident and resolution

Training and epoch-8 validation had completed before the first pipeline
attempt crashed during epoch-5 evaluation. Epoch 5 incorrectly tried to reuse
epoch 8's immutable `validation_predictions.json`. Artifact protection refused
the overwrite; rank 0 exited while rank 1 waited for a collective and later
hit the NCCL watchdog.

Commit `59511887c7e4d55e9a854e091ec57f78a5a9904a` assigns stage-specific H5 and
H6 prediction paths and broadcasts rank-zero publication failures immediately.
Twenty-four focused tests, Python compilation, shell syntax, and
`git diff --check` passed. Recovery reused the existing checkpoints and did
not retrain. It completed in 4 min 29 sec on eight H100s. All eight GPUs reached
100% sampled utilization and approximately 20,183 MiB peak memory during
inference; lower whole-run average utilization reflects model loading,
synchronization, and short autoregressive answers.

## Protected-data status and immutable evidence

- selection status: `no_variant_passed`;
- selected stage and adapter: none;
- fixed test accessed: false;
- real-image sanity accessed: false;
- config SHA-256:
  `ec46b18c72ec9d2df18b725f0fd2dde39d42136a6008a677b45ba3559bf6cd4b`;
- epoch-8 adapter SHA-256:
  `9b3b24f507c371bd5e85a327927a092383de966249ec1cafa87d3ff2e147f30e`;
- checkpoint-selection SHA-256:
  `101b4aa466967f392bebdb08b2bfb917cde74cb7d8e490b94c37b51f4c639d4e`.

## Advisor-ready summary (Mandarin)

按您建议，我们没有加分类 head，而是继续采用“一张图分别询问五次”的 Qwen
decoder 方案；visual encoder 和语言侧 LoRA rank 都为 32，并从第 3 轮的完整
LoRA 与 optimizer 状态继续训练到第 8 轮。训练 loss 从第 4 轮的 0.02635 降到
第 8 轮的 0.02514，但冻结合成 validation 上三退化严格全识别率没有明显提升：
第 5/6/8 轮分别为 55.83%/56.11%/55.83%。最好第 6 轮的总体五分类准确率
86.68%、type micro-F1 93.22%，但真实存在退化的 L2–L5 强度准确率只有
66.16%，三退化严格全识别仍远低于目标 80%。因此问题已经不能解释为“只训练
3 轮、尚未拟合”；继续相同训练预计没有收益。三个 checkpoint 都未通过预先
冻结的 validation gate，因此没有查看 fixed test，也没有进入真实图检查。下一步
建议先和您确认：是否回到“一张图一次回答五种退化”的方式，重点排查 per-type
五次问答是否丢失了联合退化上下文；或者保持五次问答，但修改数据/目标以专门
惩罚复合图漏检，再做一个小规模、可证伪的诊断，而不是继续堆训练轮数。
