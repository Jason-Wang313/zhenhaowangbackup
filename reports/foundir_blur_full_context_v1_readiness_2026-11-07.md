# Blur同源全图上下文诊断：固定前8训练来源、冻结1280

前一结果：同64训练裁图26.805872/.143962，而官方150仅20.936988/.196516，未过25.9/.18。当前先区分相同像素在独立裁图与全图拼接时的变化，不能直接把差距归因训练来源或继续加步。

固定原1024训练来源前8项、原1280导出权重、原冻结参考LoRA、FP32 VAE和原推理公式。仅node3双H100。对这8张1920×1080完整训练LQ执行与官方相同的512/128分块、padding、双uint8和sqrt-Hann拼接，共120块；另外用相同无缓存推理重新生成8个原训练裁图作路径对照。总256次模型调用，无训练/optimizer更新。

保留旧训练64中的8个原裁图PNG；按同一GT算：全图LQ/恢复、同512像素的LQ/旧裁图/无缓存裁图/全图截取、固定中心384的相同四项，共10项。所有输出先保存再读训练GT评分，不拟合、不逐图选结果。另报原裁图外的MSE和同一位置输出差，不把masked LPIPS当有效指标。

若无缓存原裁图与旧PNG就不同，先查数值/缓存路径；若两者接近而全图中相同像素明显变化，则存在上下文/拼接敏感性。全图同源结果若保持改善，才更支持官方来源差异。结果只能定位贡献，不能单独宣判唯一根因，也不是部署修复或新质量gate。

3项CPU测试通过：真实1080p坐标/官方恒等拼接、原裁图外像素统计、无效裁图边界；Python/shell语法通过。目的地CPU预检5.244058秒：前8来源、16原图文件SHA、8旧输出和最终权重绑定、120块几何通过，未读取官方像素、未解码训练GT。GPU实际路径仍待本job。

源码发布后由既有node3 worker派发round16_blur_full_context_v1，入口smoke/run_foundir_fullstack_blur_full_context_v1.sh。预估2–4分钟/.067–.133使用GPU-hours，非实测承诺。固定8图；故障停止、资产保留、无自动重试，无下一训练自动派发。

```bash
.conda/gpu-train-stable/bin/python -S smoke/foundir_fullstack_node3_extension_v1.py submit --job-id round16_blur_full_context_v1 --entry smoke/run_foundir_fullstack_blur_full_context_v1.sh --commit <verified_release_commit> --purpose 'Frozen1280 same training8 full-frame context diagnostic; no training or new gate' --sources smoke/diagnose_foundir_blur_full_context_v1.py smoke/blur_context_geometry_v1.py smoke/evaluate_foundir_blur_exposure_official_2x2_v2.py lowlight_restoration/blur_delta_velocity_v1.py
```

新输出checkpoints/foundir_blur_full_context_v1。两rank最终/参考权重前后hash均不变，launcher退出0后CPU运行diagnose_foundir_blur_full_context_v1.py finish，核验全部8图/120块/8个无缓存裁图及10项指标后推小报告/耗时。无独立持久控制器，接手查本job真实状态，勿重复提交。仍仅官方150的25.9/.18为Blur验收；Blur未过不动其它类别。
