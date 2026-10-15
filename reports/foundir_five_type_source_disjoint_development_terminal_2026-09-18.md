# 五类 source-disjoint development 终局冻结

日期：2026-09-18

## 决定

停止继续优化任何单一退化类型。五个当前选中的项目自产专家已经分别在完整、
source-disjoint development split 上完成评估，因此不重复运行相同 checkpoint
和推理规则。重复运行不会增加独立证据，只会消耗 GPU。

本次只进行了现有证据、checkpoint 和结果 SHA 的只读核验，并建立机器可读
registry；没有启动 GPU、没有访问 sealed internal 或 official FoundIR test。
Released FoundIR generalist/specialist 没有被选为任何一类的项目结果。

机器可读 registry：

`reports/foundir_five_type_source_disjoint_development_registry_v1.json`

Registry SHA-256：

`225004da30a29854423ef1e921787830344d3af824d1b213900c0481c5328e47`

## 五类冻结结果

| 类型 | Development records / source groups | PSNR / LPIPS | PSNR 相对 SOW 余量 | LPIPS 相对 SOW 安全余量 | 决定 |
|---|---:|---:|---:|---:|---|
| Blur | 2,595 / 165 | 26.559141 / .151993 | +2.5449% | 15.5594% | 冻结 |
| Noise | 1,859 / 34 | 40.150553 / .068880 | +3.2148% | 31.1196% | 冻结 |
| Haze | 1,580 / 158 | 23.311888 / .101155 | +5.4837% | 46.7605% | 冻结 |
| Rain | 835 / 117 | 42.284347 / .047452 | +23.6384% | 40.6848% | 冻结 |
| Low-light | 1,400 / 21 | 19.779135 / .154058 | +4.6515% | 44.9792% | 冻结 |

五类平均 PSNR 和 LPIPS 均通过各自 Stage-2 SOW。

## 不隐藏的限制

- Blur 平均指标通过，但固定 alpha=.75 的严格 per-record regression gate 失败。
- Noise 平均指标安全，但部分 near-clean 输入会被过处理。
- Haze 的均值和 source-group bootstrap 均通过，但 16 张固定视觉中有 1 张
  near-clean 图出现明显橙色色偏。
- Rain 安全通过；无需继续质量优化。
- Low-light 完整 development 与 16 张固定视觉通过，但 development 只有
  21 个 source groups。

这些限制继续作为 routing/identity-safety 风险记录，不再通过新增单类型训练或
后处理 sweep 解决。

## 下一未完成阶段

单类型质量阶段到此结束。下一未完成阶段是五专家 fail-closed routing/integration：

1. 精确绑定上述五个 checkpoint 和各自后处理规则；
2. 确保 Rain Restormer 及其他专家不能误用于别的类型；
3. 对 unknown/mixed/ambiguous route fail closed；
4. 只评估 integration 是否忠实复现上述已冻结专家，不重新选择 checkpoint，
   不重新优化单类型指标。
