# 官方V4修复后真实GPU启动 — 2026-11-05

截至2026-11-05T19:56:15.566569+00:00，四节点8worker均输出`OFFICIAL_VLM_V4_METRIC_DEVICE_VERIFIED device=cuda:0/1 full_state=true`，并均产生真实OFFICIAL_VLM_V4_TILE。评分身份修复已通过实际H200/H100 CUDA迁移检查；没有删除权重校验。四节点新日志未见错误，尚无新attempt结束记录。

共享目录恢复块receipt 88/21828（0.4032%），完成整图指标0/600。计数是已提交receipt数量，不是重新扫描全部tensor哈希。尚无完整原图指标，不宣称画质或原图metric显存验收通过。当前早期吞吐不用于承诺最终ETA。

复用原19,460块特征与600路由，不重训、不重跑route、不改变科学配方。原首次失败0.918888889 GPU-hours与route 1.74 GPU-hours继续记账；当前运行尝试尚未收尾，最终GPU-hours后续从attempt账累计。不限时授权保持有效，无自动重启循环。

新日志`logs/foundir_official_vlm_router_v4_4x2_render_node<N>_metricfix.log`。只推本小报告，不推PNG/tensor/权重/live logs。修复binding SHA256：`ef906054dde065a314fe8b6da1a457569c14138f16c7d10b6ee9c61e5b5d6e18`。
