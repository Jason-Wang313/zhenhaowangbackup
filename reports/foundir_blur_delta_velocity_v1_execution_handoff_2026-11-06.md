# Blur200步小试验：真实启动与网络恢复

当前仍只推进Blur，完整官方150张25.9/.18双目标通过前不动其它类别。原生LoRA预测差两步资格已真实通过：8图初始化浮点及uint8均完全等于原LQ、MSE/LPIPS梯度非零、两rank权重一致、step1真实恢复后step2完成；退出0，138.20684秒、.07678158 GPU-hours，峰值16.96299GiB/卡。资格报告58411a7已推送。
参考缓存160/160完成，退出0，68.10502秒、.03783612 GPU-hours。node3两H100上fresh200步pilot正在执行，首次训练报告2a5a0d6已推送。训练/开发结果均不得当官方通过；旧资产、energy及其它类别均保留。
GitHub连接曾超时，现采用独立CPU调度和上传器，避免上传阻塞已发布的实验；GPU绑定源码4182f8c未改。v1观察器已停止，当前smoke/watch_foundir_blur_delta_network_v2_cpu.py：monitor会话blur_delta_v2_monitor，upload会话blur_delta_v2_upload，日志logs/foundir_blur_delta_velocity_v2_{monitor,upload}.log。持久请求publication_requests及已发布published在checkpoints/foundir_blur_delta_velocity_v1；接手先查这些，不重复提交/启动。网络失败只延迟小报告，GPU失败仍停链。
固定顺序为200步→全128训练/32开发评价→CPU finish→小报告；无自动其它类别或官方评价。step1/50/100/200报告先原子落盘，再独立上传。所有GPU任务仍使用同一已发布commit及固定配方；轻任务只用node3双卡，不占其它三台。

CUDA资格、缓存与首次训练已发布；当前实验仍运行。预计以实际完整step检查点时间更新ETA，不能仅用不含保存耗时的计算秒数。
