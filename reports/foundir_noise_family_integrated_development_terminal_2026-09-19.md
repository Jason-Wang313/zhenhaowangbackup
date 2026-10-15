# Noise-family fail-closed integration development：终结报告

日期：2026-09-19

## 结论

端到端接线完整且忠实，但预先配置的严格 restoration safety gate 未通过。不能把本轮整体写成 PASS，也不能把它误写为路由错误或 SOW 失败。

- **Routing/integration fidelity：PASS**
- **Noise SOW mean metrics：PASS**
- **Noise+Low-light restoration improvement：PASS**
- **新增的 Noise mean-PSNR-over-input safety check：FAIL**
- **Overall configured automatic gate：FAIL**

## 路由与执行完整性

- 接受样本：1,842
- 实际执行接受样本：1,842
- 实际执行拒绝样本：0
- `Noise → noise`：674
- `Noise+Low-light → noise → low_light`：1,168
- Noise stage 输出：1,842/1,842
- Low-light stage 输出：1,168/1,168
- 缺失输出、错误专家计划和运行异常：0

因此 fail-closed decision layer 已被 integrated pipeline 忠实执行。

## Restoration 指标

| 路线 | Records | Input PSNR / LPIPS | Restored PSNR / LPIPS | PSNR delta | LPIPS delta | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Noise | 674 | 40.04556 / .12944 | 39.97339 / .07951 | −0.07217 dB | −.04993 | SOW 通过；严格 input-PSNR gate 失败 |
| Noise+Low-light | 1,168 | 9.94886 / .52665 | 19.73496 / .21161 | +9.78610 dB | −.31504 | 通过 |

Noise SOW 为 PSNR ≥38.9、LPIPS ≤.10。本轮 Noise 相对 SOW 的余量为：

- PSNR：+1.07339 dB，即 +2.7594%
- LPIPS：低于上限 .02049，即 20.4868% 安全余量

Noise 的历史冻结选择没有接错：仍为 step-400 加固定 α=.30。完整 1,859-record development 的既有结果为 40.15055/.06888，同时早已记录相对输入平均 PSNR −0.32356 dB。本轮接受子集上的 −0.07217 dB 与该已知取舍一致，而不是 integration 漂移。

## 资源与数据边界

- 硬件：8×NVIDIA H100 80GB HBM3
- 开始：2026-09-19 11:43:56 UTC
- 结束：2026-09-19 12:29:41 UTC
- Wall time：约 45 分 45 秒
- GPU-hours：约 6.10
- 权重更新：无
- FoundIR official/internal test：未访问
- 数据：training corpus 内冻结的 source-disjoint development partition

## 决策边界

本轮结果不授权放宽已经预注册的 gate，也不授权访问 official/internal test。若继续改善严格 safety，必须作为一个新的、bounded、training-only calibration 任务；不得在这 674 条 development 结果上直接选择新 α 并把同一批数据宣称为独立验证。

结果文件：`checkpoints/foundir_noise_family_integrated_development_v1/result.json`
