# 匹配top-1路由恢复评价终态 — 2026-11-04

matched_top1_four_missing_v1已完成，MATCHED_RESTORATION_RESULT_VERIFIED和END status=0。仅新增4个Haze输出，按固定input+预测专家身份复用76/80条预测记录；两组各40图，合计47种预测路由。CPU summarize重复核验通过，逐图均值独立核对通过，没有重算旧指标或生成旧40图。

实测单节点2×H200，76秒（1分16秒），152 GPU-seconds = 0.042222222使用GPU-hours；一次attempt，无失败成本。若独占更多卡，整机占用另计（范围未确认）。峰值allocated137.932561GiB/卡，未解决H10080GB兼容。

| 方案 | 选对 | PSNR | LPIPS |
|---|---|---|---|
| image_only top-1 | 33/40 | 26.919075 | .130795 |
| image_plus_vlm top-1 | 38/40 | 28.199716 | .126656 |
| 旧step408 soft | 36/40 | 27.465518 | .124592 |
| 已知类别单选参考 | 40/40 | 28.633983 | .108469 |

同样top-1政策下，VLM组比image_only平均PSNR高1.280642dB、LPIPS低.004140（约3.165%）；Noise/Lowlight/Rain两项改善，Haze相同，Blur两项退步（PSNR低.379681dB、LPIPS高.047268）。不得只报告总体改善而隐藏Blur损害。

VLM组相对旧soft，PSNR高.734199dB，但LPIPS高.002063（变差）；仍未超过已知类别单选参考，两张错选图在两项均退步。不能宣称所有指标超过旧方案。旧soft对比同时改变训练和部署策略，不把整段差异都归于VLM。

固定单seed/40开发图结果，不是官方SoW/Rain100门槛或混合退化成功。GT仅在生成完成后评分，未用GT改选或按指标挑缓存。没有重训、扩图、扩预算或启动后续GPU。

完整逐图、五类、配对差值、运行身份和输出SHA见同名JSON；只上传小报告。tensor/图片/模型/缓存/原日志均留共享目录。

日志：logs/foundir_supervised_router_restoration_v1.log
日志 SHA256：b7196f6e04e9a32f830d1616b65c4bc505b8d67a8f8615fa7166be3e1b1a9cf4
