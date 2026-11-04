# CPU路由监督修复终态 — 2026-11-04

fresh_ce400_matched_v1 已完成，两组各400步，802份逐步checkpoint均保留共享目录。入口prepare/train/verify全部通过，SUPERVISED_ROUTER_FINAL_VERIFIED及CPU_END status=0。单次attempt实测212秒（3分32秒），CPU8线程，0 GPU-hours，无失败尝试。

| 组别 | 训练正确数 | 开发正确数 | Blur | Noise | Haze | Lowlight | Rain |
|---|---|---|---|---|---|---|---|
| image_only | 606/640 | 33/40 | 7/8 | 6/8 | 8/8 | 6/8 | 6/8 |
| image_plus_vlm | 640/640 | 38/40 | 7/8 | 8/8 | 8/8 | 7/8 | 8/8 |

两组fresh初始化/样本顺序/400步/global32/lr.001匹配；VLM组从第1步启用分类监督，日志记录有限非零梯度，图像组VLM梯度为0。两组训练完成后才读取固定40图开发特征，未挑步或早停。

VLM组比匹配image_only多正确5张；相对旧step408的36/40，修正旧4个错例中的3个（0144583、0189233、0508782），0479048仍错误，同时新增0007534错例，最终38/40。image_only旧4错例均未修正，新增3个错误。完整逐图分数、混淆与转变见同名JSON。

这是单seed、40开发图的分类比较，不能据此声称统计稳健VLM收益、恢复质量或官方门槛通过。没有读GT/LQ像素、没有新PSNR/LPIPS或恢复图，未改变专家/FLUX/VAE/Qwen，也未启动后续GPU或扩大预算。本轮CPU结果不代表H100显存问题已解决。

仅上传小报告与耗时；模型/Adam/特征/实时日志保留原目录。

日志：logs/foundir_router_supervised_repair_v1.log
日志 SHA256：31e694af9886287f05f3010f7f000d700648eeecdc2855a6ffb08383588a5adf
