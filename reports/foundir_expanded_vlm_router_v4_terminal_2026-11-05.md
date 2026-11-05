# V4 完整实验终态 — 2026-11-05

run_id: `expanded_conditional_v4`。prepare、两节点features、CPU训练、两节点render及finish均status=0；最终出现`EXPANDED_V4_RESULT_VERIFIED`。没有自动后续任务。

完成11457份新特征、6400步CPU训练；按校准CE预先选step4800，独立100图分类99/100。四组共400行评价，去重117次恢复生成，四worker完整。

| 固定组 | PSNR ↑ | LPIPS ↓ |
|---|---:|---:|
| image_only | 26.44992962 | 0.14313587 |
| previous_v3 | 27.00370093 | 0.13509792 |
| conditional_v4 | 27.30831943 | 0.12209467 |
| known_class_reference | 27.31971725 | 0.12162699 |

V4相对两个对照均改善：
- 相对image_only：PSNR +0.85838982 dB，LPIPS下降14.7002%。
- 相对previous_v3：PSNR +0.30461850 dB，LPIPS下降9.6251%。

Lowlight相对V3的PSNR下降0.00232547 dB，LPIPS改善；平均改善不代表所有类别不退步。V4比已知类别参考低0.01139782 dB、LPIPS高0.00046769，仅Blur剩余一张错选；其它四类复用相同路由输出。类别参考不是可部署选路。

| 类别 | V4 PSNR | V4 LPIPS | 新100数值目标 |
|---|---:|---:|---|
| blur | 21.75404641 | 0.21621393 | 未通过 |
| noise | 40.30307644 | 0.09138287 | 通过 |
| haze | 21.90291628 | 0.08778361 | 未通过 |
| lowlight | 18.30808811 | 0.16801234 | 未通过 |
| rain | 34.27346993 | 0.04708061 | 通过 |

**整体双指标优于两个对照，但五类数值目标没有全部通过。** Blur两项未达标；Haze、Lowlight的PSNR未达标。新100裁图不是官方SoW完整测试，不宣称官方通过、混合退化成功或单因素VLM消融。来源对路由训练分离，但冻结专家/Qwen可能见过。

## 耗时、硬件与预算

- prepare：22秒；CPU训练：2334秒（38分54秒）；CPU finish：8秒，均退出码0。
- 特征：2节点×2张H100，节点1651/1613秒，共1.813333 GPU-hours；峰值allocated约15.734956 GiB/卡。
- 恢复：2节点×2张H200，节点610/579秒，共0.660556 GPU-hours；峰值allocated约10.922726 GiB/卡。
- 总实际使用2.473888889 GPU-hours，低于原18 GPU-hours预算；各阶段一次attempt，无失败。并行节点耗时不相加当作用户等待时间，整机独占成本未知。
- H100特征实测通过；恢复只实测H200，不能宣称H100全流程已验收。VAE限定范围采用FP32 native conv、query chunk256；旧2960特征保留，不保证与旧cuDNN路径逐像素等价。

## 结果绑定

- 终态JSON SHA256：`0014bbd99efe90a61776188702a26fd1585b1e2dcf01afe9d4e64b5d2df2fd42`
- render receipt SHA256：`31f73ba4e89b47d5620bd99278ec85b410c8a9824aee3abc28fc65b80ea79fb1`
- render plan SHA256：`ccd252d058788cc6a254b491f12f1ab3b6f6fa12940eeb01f67aa59d3d3fbdc4`
- 本次仅发布终态JSON和本简报；大资产、历史结果和无关修改保留。未改配方、扩大预算或启动新GPU任务。
