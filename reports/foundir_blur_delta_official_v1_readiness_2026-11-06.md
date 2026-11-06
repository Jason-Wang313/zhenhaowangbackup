# Blur最终200步直接评价官方150

用户最新明确要求仅使用官方测试集，不再使用其它测试集作为前置门槛。本轮覆盖此前32开发放行要求；保留已经完成的训练来源选择、CPU/CUDA资格与参考缓存，不中断当前固定200步训练，不重跑资格，不再提交旧160图评价。

固定step200是训练前指定终点，不按官方反馈从中间检查点挑最佳。仅使用官方SOW600中的全部150张Blur，固定冻结conditional_v4无标签路由；150张均必须由已保存的logits自身选择Blur，不能拿类别标签路由。复用已核验的路由决定，不重做训练或路由。

每张原尺寸，沿用512块、128重叠、原padding、sqrt Hann拼接和uint8量化。每块运行原Blur参考LoRA与最终200步LoRA两次FLUX前向，使用已CUDA验收的LQ + VAE解码差；无噪声采样、无步幅搜索、无候选/GT回退。不再是旧50步推理。最终完整PNG先固定，再读官方GT，以同一冻结pyiqa0.1.16全尺寸RGB默认PSNR/LPIPS/SSIM评分，不能改指标器、缩图或截边。

最终唯一成功条件为全150张同一输出的平均PSNR≥25.9且LPIPS≤.18；完整结果、相对旧50步和LQ、退步及耗时都报告。其它类别等待Blur通过。官方已历史曝光且用于反复开发决策，因此明确非独立盲测；官方GT不参与参数训练、阈值拟合或逐图选图。

目的地预检分两段：plan仅绑定官方清单、无标签路由、原指标器和源码；训练退出0后prepare核验最终200模型/Adam/两rank RNG/采样位置/哈希、201份step receipt连续性，并导出同一FP32 LoRA，核对全部300输入/GT文件SHA及指标身份。无需重复扫描78GiB历史checkpoint；不能据此声称已重新核验全部历史大文件。prepare不解码官方GT像素。

轻量两调用推理只用node3双H100；各rank75张，逐块和逐图保存SHA，可复用已核验输出，无自动失败重试。数据、权重、PNG和所有旧记录保留共享盘，只推小报告。准备后提交入口`smoke/run_foundir_fullstack_blur_delta_official_v1.sh 3`；代理使用常驻worker，不需用户重复终端命令。

CPU接续器`smoke/watch_foundir_blur_official_v1_cpu.py monitor --commit <已发布commit>`只等待现有训练→prepare→官方150→finish；独立upload模式上传报告。旧`blur_delta_v2_monitor`已停止，旧上传器处理step200及明确“仅训练结束、开发评价未运行”的终态后退出。新监视器不调用旧evaluate或旧finish。

无累计GPU-hours上限，本轮仅固定官方150张，不扩到其它类别。科学配方、训练来源和运行绑定文件均不改；本轮改变的是用户授权的验收范围。
