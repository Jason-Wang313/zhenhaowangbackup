# VLM保留正确路由V3终态 — 2026-11-04

按4f3b1f4安全同步并执行指定tmux入口，prepare→train→verify已完成，1600/1600步、1601份完整step marker。VLM_PRESERVE_V3_RESULT_VERIFIED，attempt结束记录status=0，实测345秒（5分45秒），CPU8线程/0GPU-hours；一次attempt，无失败成本。未改配方，未启动GPU。

预设校准筛选未通过：四个检查点均不合格。最低CE的step800仅供失败诊断，calibration_screen_pass=false，不是获准部署/恢复评价的模型。固定校准400图正确数349→366，CE .459019→.262142；相对V2校准365/.272625略好，但纠正28仍相同，误改12→11。

| 校准类别 | 冻结图像 | V3 step800 |
|---|---:|---:|
| Blur | 74/80 | 76/80 |
| Noise | 63/80 | 74/80 |
| Haze | 73/80 | 66/80 |
| Lowlight | 77/80 | 76/80 |
| Rain | 62/80 | 74/80 |

Haze和Lowlight下降，不满足每类不降；不能只报告总体提高。训练保护不保证新图no-harm。本轮只用fit/calibration特征，旧holdout特征未读取，预留100图仅核对元数据未读取像素/特征。没有新的独立确认、恢复图、PSNR/LPIPS或官方门槛结果。保留全部状态与大资产，不自动扩大预算或启动下一阶段。

只上传同名结果JSON和本耗时/退出简报，不上传权重、特征、图片或live logs。

日志：logs/foundir_vlm_preserve_router_v3_cpu_train.log
日志SHA256快照：337b2c69d9086deb044ecf37e9f635ce9b2010b3ec498a1de2db573d27125c99
