# Noise FLUX：LPIPS/PSNR冲突诊断V2

## 为什么做

V1训练来源GT诊断已正常完成，最强PSNR提升约.824dB，但LPIPS目标未满足。
这证明有保真修正空间，没有证明两项指标本来就不可兼得。
用户授权下一步，聚焦三个可区分原因：两个梯度的大小/方向、更新幅度、量化代理。
不是再训模型，不是论文创新成功，也不再同时扫窗口。

## 固定边界

- 相同8 training sources、512裁图、逐记录seed、Noise step400、50步。
- 原alpha=.30和两次uint8 round保持不变，基线必须逐像素等于原缓存。
- 只修改index48处的latent，再真实运行余下两步并解码；完整FLUX输入梯度。
- 冻结FLUX/VAE/LPIPS，不加载Restormer、旧energy或在线Qwen。
- 不访问任何official/sealed/internal/parent-test图像，不动1859条最终开发集。
- 相对原FLUX目标不变：+.30dB，LPIPS相对下降.09928107050377874。
- V1终态SHA256绑定为
  `4985f367cca8bc1fb730dee173d111eb4929fc6ebda4b9d0db6da223d523f7b1`，
  不修改其源码/结果或假装V2延续了V1学习权重。

## 具体检查

### A. 梯度究竟有没有冲突

对同一个实际两步终图，分别计算按初始无引导结果归一化后的MSE和LPIPS梯度。
记录RMS、大小比值、cosine，以及各候选方向预测的两个局部变化率。
loss数值都约1，不代表梯度同样大；cosine<0表示有局部方向冲突，但不自动证明
不存在共同下降方向。只有实际量化终图指标能决定本次修正有没有用。

四个固定方向：MSE-only、LPIPS-only、原联合目标、两个梯度先各除自身RMS再相加。
每个最终方向再次归一化为单位RMS，避免把方向和步幅混为一谈。
原联合目标通过其自身一次backward获得，避免两个BF16反传梯度相加的舍入差异。
若平衡方向退化成零，明确记录并保持起点，不伪造有效方向。
这只是标准多目标优化诊断，不把梯度平衡包装为新架构贡献。

### B. 连续目标和量化目标是否错位

分别使用continuous blend，以及double-round straight-through（STE）代理。
STE的前向必须与真实deployed_proposal逐像素一致，反向把round近似为identity；
它不是取整函数的真实梯度，也不接受“有限差分必须等于STE梯度”的错误判据。
每个arm同时记录continuous图像和真实量化图像的PSNR/LPIPS。

固定幅度为.0005/.001/.002/.005/.01。
2种代理×4个方向×5个幅度，共40个single-update arms，全部报告，不隐藏负收益。
加历史BF16和FP32 latent无引导对照，共42行。

### C. 多次小改是否优于一次大改

每种代理另做一条最多8次的balanced方向轨迹。每次重新计算当前梯度、尝试全部
5个幅度、完成真实rollout及量化，并只在GT指标同时不损害当前PSNR/LPIPS且至少
一项严格改善时接受。多个安全候选时选归一化MSE+LPIPS代价最低者，同分选小幅度。
这不是用单次loss下降代替真实双指标检查。

无安全候选或达到双目标时停止，保留当前结果；在1/2/4/8步均报告，注明实际
接受次数/停止原因，不能把停住后重复结果当成持续改善。每次所有trial及拒绝保留。
两条轨迹×4个milestones，共8行；总共50个报告arms。

特别重要：上述接受/停止直接读取训练GT，只是oracle诊断，不是可部署selector、
energy自验证或正式TTS比较。严格逐步no-harm的有限搜索失败，不证明不存在
暂时牺牲一项指标、后来再恢复的有效路径。8图成功也不证明energy能学会或泛化。

## 预算与启动

单机4/8张同型号H100/H200；4卡每rank顺序2张，8卡各1张；不减数据或增加单卡batch。
每张模型生命周期结束后释放显存。fresh ID、节点锁、硬件空闲/型号、身份哈希和超时
保护继续使用。上限4卡1小时、8卡.5小时，均4 GPU-hours，另最多30秒清理。
这不是实测ETA；V1实测509秒只能作为有限参考，不能直接当V2预测。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/start_foundir_flux_tail_tradeoff_v2_tmux.sh tradeoff_4gpu_v1 4
```

八卡改可见列表为0–7、第二参数8，并使用fresh ID。
日志`logs/foundir_flux_tail_tradeoff_v2_tradeoff_4gpu_v1.log`；
输出`checkpoints/foundir_flux_tail_tradeoff_v2/tradeoff_4gpu_v1/`。
需要完整8来源×50arms、基线与STE前向验证通过、terminal result及verify标志；
退出码0本身不等于双目标通过。新energy训练不会自动启动。

## 本地验证

Python/shell静态检查通过。四卡/八卡参数的真实权重/缓存/训练来源预检均通过，未解码原图。
新单元测试涵盖STE前向精确性/近round边界、代理梯度与clamp、梯度尺度与冲突、
共同下降方向/完全相反退化、GT安全选择与停止、配置预算、完整汇总与禁止虚假声明。
还检查小尺寸真实FLUX模型经checkpoint后的独立/联合三次反传。
18项CPU回归测试全部通过（原10项+新8项，4.793秒，不含Python冷启动）。
GPU诊断尚未启动，当前没有V2质量结果。
