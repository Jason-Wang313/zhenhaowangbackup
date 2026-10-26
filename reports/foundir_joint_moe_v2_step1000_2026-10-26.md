# 联合 MoE 第1,000步：两臂中期结果

日期：2026-10-26。两条正式训练均在继续，用户要求监控至结束。
本记录不是最终选择结果，也不是完整 Blur 测试。没有改参数、重启或读取受保护像素。

## 指标（PSNR越高越好，LPIPS越低越好）

| 同一步数／协议 | 普通版 PSNR | 普通版 LPIPS | VLM版 PSNR | VLM版 LPIPS |
|---|---:|---:|---:|---:|
| Rain100，第1000步 | 33.9101104544897 | .07389099790481851 | 33.92674730344711 | .07357312987558544 |
| Blur100预览，第1000步 | 24.841265019393163 | .35753815472591666 | 24.776332465206465 | .35622713952558116 |

VLM相对普通版：Rain约+.01664 dB／LPIPS降低.000318；Blur约-.06493 dB／LPIPS降低.001311。
Rain小幅更好，Blur有好有坏，没有稳定的两指标优势。
两条相对自身step0都改善Blur、损害Rain。Rain PSNR仍低于34.2，LPIPS低于.08；
Blur预览PSNR低于25.9、LPIPS高于.18。不能用100张预览替代最终2595张评估。
这些是中期描述，没有运行仅接受最终10000步结果的正式比较器，也没有按中期结果挑checkpoint。

两条路由已把Rain主要分给专家0、Blur主要分给专家1。VLM版部分层更接近单专家选择；
这只是选择行为，并不证明图像质量更好。GT_used_for_routing=false。

## 工程状态与后续

两条日志推进、损失和梯度有限，未观察到OOM／NCCL fatal／运行时错误。
完整step1000权重、Adam和续训记录已生成；world_size8、next_sample_position32000。
普通版用8×H100，VLM版用8×H200；配置和数据一致，但保留step0微小数值差异说明。
不把速度差或微小指标差自动归因给VLM，不宣布显著性、创新或SOTA。

按既定计划继续至10000步，下一质量检查3000／6000。
最终还需完整Blur2595、固定专家对照和VLM关闭对照，以及两条launcher正常退出。
每臂原预算5.45–9.06正常小时／43.63–72.45分配GPU-hours，
两臂总预算87.26–144.89 GPU-hours；这不是最终测得耗时。

## 小文件独立SHA-256

文件位于 `checkpoints/foundir_rain_blur_joint_vlm_moe_v2/<arm>/formal_v1/`。

| arm／文件 | SHA-256 |
|---|---|
| image_only／metrics/rain_001000_learned.json | cacb51da5286f2cd5f236208c212105eb20a700702637271daeee094e98ed637 |
| image_only／metrics/blur_preview_001000_learned.json | 8a8cfcfe41ea499c789fe013bb533281eba9acdf5855ca90f754889aacdab199 |
| image_only／step_001000.resume.json | 1dcb34a80a408add35add18e636e83862bd327498e2c7b15487652f3a473edc6 |
| vlm_assisted／metrics/rain_001000_learned.json | 940f7d1a919570a20b55ce1fb012fef2225000281d68076cb20992b2de955507 |
| vlm_assisted／metrics/blur_preview_001000_learned.json | 6607761c8545c895e4d641e4354d087d8b7d389f4f33033e2a6c027d0761913f |
| vlm_assisted／step_001000.resume.json | 9989ce1731f1210203be8537bb600842f864da5282f7c3dc44322f0cd4313f4b |

以下大文件hash来自训练器保存时的receipt，本轮未重新读取全部大文件：
image-only权重 `9197996c43079af71f5fb88fa0ebf8562a770fedc299b53b587d41473a8f8015`；
Adam `01e69a570644376cf49025adf2f9e81f074622a8b29d5d2d6761b048e2049753`。
VLM权重 `1cea768ea0d5aaec33470a36041eff4a38751cd39ff37433a1e6cb79d6762c4a`；
Adam `b11cf024f542e454f2d5be2b81998ea8355ada2c407a0682d74b7ecbc0cedbb5`。

只提交文档，不提交权重、原始日志、图片或数据。当前无需用户操作。
