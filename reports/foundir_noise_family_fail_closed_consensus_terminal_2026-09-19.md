# Noise-family fail-closed consensus：终结报告

日期：2026-09-19

## 结论

最终双共识决策层通过冻结的 development gate。它不修改恢复模型，也不训练新模型：

- `Noise` 继续交给已冻结的 Noise expert；
- `Noise+Low-light` 继续交给已冻结的 Noise→Low-light sequential experts；
- dual nonlinear selector 与 physical guard 给出相同的 Noise-family 路线时才接受；
- 任一 selector 拒绝、越界或两者冲突时，一律 `abstain`，不得静默猜测。

## 完整 development 结果

两份冻结预测先对齐并核验了全部 16,461 条 development record；最终只评估范围内的 3,602 条 Noise-family record。

| 类别 | Records | 正确接受 | 错误接受 | 安全拒绝 | Coverage | 接受样本准确率 |
|---|---:|---:|---:|---:|---:|---:|
| Noise | 1,859 | 674 | 0 | 1,185 | 36.2561% | 100% |
| Noise+Low-light | 1,743 | 1,168 | 0 | 575 | 67.0109% | 100% |

冻结 gate 要求每类错误数为 0、每类 coverage 至少 30%；两类均通过。

## 计算与数据边界

- GPU：0 张
- GPU-hours：0
- 本轮只有 CPU CSV 校验、路由单元测试和报告生成。
- 只使用 training-only/source-disjoint development 工件。
- 未读取、未运行、未修改 FoundIR official/internal test。

## 冻结工件

- Dual development predictions SHA-256：`0107c8ca432257bc5bfb529d6e2e2f4cdf8df3941292c471873541a1a0eb1b6f`
- Dual noise-side head SHA-256：`7b91d71650d2bd0fc1c07cc92e5b44159d6ea71397c44a6eee026389555f8647`
- Physical development predictions SHA-256：`fb3ff1eef8b5acaf9cadc6cc973735c28b430144fc3da69d73430270d4d6633e`
- Physical rule SHA-256：`029098ef90228f7b6a55a6166b2d776a71481d7bac246097570ee1a893b5f7d8`

配置、可复现评估结果和逐条决策分别位于：

- `configs/foundir_noise_family_fail_closed_consensus_v1.json`
- `checkpoints/foundir_noise_family_fail_closed_consensus_v1/result.json`
- `checkpoints/foundir_noise_family_fail_closed_consensus_v1/development_decisions.csv`

## 不能夸大的地方

这是安全优先的决策层，不是 100% coverage 的完整自动路由。development 的零错误说明在当前完整开发集上没有错误接受，但不是未见分布上的数学保证。任何 `abstain` 必须交给明确的安全 fallback 或人工处理，不能随后被另一个不受约束的规则强行分流。

本轮冻结的是 decision layer。两个 selector 的在线推理和两个既有 restoration backend 仍需由 integrated pipeline 按此规则连接；不能把离线 CSV 本身误称为完整在线部署。
