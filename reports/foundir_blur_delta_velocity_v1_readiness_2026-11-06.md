# Blur：保留LQ的原生FLUX预测差小试验

用户当前唯一优先级为Blur。完整官方150张PSNR≥25.9且LPIPS≤.18之前，不推进其它类别。无累计GPU-hours上限，但每轮固定范围，记录失败与全部费用。

前一六频段修正开发PSNR23.89174→24.66047，LPIPS.21738→.33055，未采用。固定8训练图的LQ局部对齐也未联合改善。证据支持原生成存在内容/位置误差；不能通过缩放一个不准的残差保证修复。

本轮可证伪假设：让网络从原LQ保留的空间位置学习去模糊修正，避免重新生成整幅图。冻结Blur step10000作为参考θ0，新θ从同权重复制，更新原生50处LoRA的32,440,320参数；不增加CNN、RGB refiner或energy。固定原Blur文本、native CFG1、reference time10、t=1。推理：

```
delta_v = v_theta(z_LQ, LQ_reference) - v_theta0(z_LQ, LQ_reference)
output = clamp(LQ + (decode(z_LQ - delta_v) - decode(z_LQ)), -1, 1)
```

这是两次FLUX前向的新推理契约，不是旧50步或已试过的LQ-start尾段rollout；也不声称首次提出LQ锚定。参考velocity和原LQ解码可按图缓存，全部只读LQ。初始化应为原LQ，必须真实CUDA检查，不能用代数替代验收。GT只用于训练loss和输出落盘后的评价。

固定128训练/32来源分离开发，复用已绑定512裁图、来源排除和旧50步缓存。历史权重曝光不被抹去；开发不是全历史独立盲测。不读官方像素，不以官方拟合/选阈值。qualification仅前8训练图、2步，pilot独立从原LoRA/fresh Adam开始200步，不用qualification权重初始化。

固定global batch8，每张微批1，单节点node3两张H100，每rank4张；逐样本loss/8、跨rank梯度SUM，clip1，AdamW lr1e-5、weight_decay0、无warmup。目标10×RGB[0,1] MSE+.1×冻结pyiqa默认LPIPS连续值，保持两种质量信号；系数是本轮预先固定的试验选择，不是已证实最优。FP32可训练LoRA/Adam、BF16 FLUX、FP32 latent/VAE；VAE只做query分块与禁用cuDNN，不改变空间范围。

每完整步保存模型、Adam、每rank RNG、next_global_slot、SHA和所有历史；损坏拒绝，不静默回退，world变化拒绝。qualification实测step1恢复后继续step2，核对Adam逐元素一致；CPU测试验证恢复后下一步与不中断完全一致。每rank初始4图，共8图uint8严格等于LQ，训练/推理模式均检查；每步须有限非零梯度、rank权重一致。

固定最终step200评价全部128训练/32开发，同一完整pyiqa权重身份。与原LQ和匹配旧50步输出比较；输出PNG先固定再读GT，无逐图fallback。不按中途开发选检查点。开发放行要求25.9/.18、两项优于旧50步且最大单图PSNR退步≤1dB；不通过保留失败，依据结果决策。开发通过也不等于官方通过，官方150完整入口需另行实现/核验。

入口：`smoke/run_foundir_fullstack_blur_delta_v1.py`，阶段prepare→qualify→verify-qualification→cache→train→evaluate→finish。队列只发node3，其他三台不启动。持久监视器`smoke/watch_foundir_blur_delta_v1_cpu.py`负责顺序、工程门槛、step1/50/100/200及终态小报告推送。GPU失败/非有限/零梯度/OOM/hash错误停链，不自动重试。不设累计GPU-hours上限，固定200步不自动扩展；实际速度与费用由真实节点日志更新，目前无实测ETA。

输出`checkpoints/foundir_blur_delta_velocity_v1`，全步checkpoint约78GiB，加缓存/输出约数GiB；仅小源码、测试和JSON/Markdown上传。源码先发布、receipt绑定资产及本地third_party哈希，再提交worker；不修改旧运行绑定文件。原工作区修改、energy、所有资产和失败记录保留。
