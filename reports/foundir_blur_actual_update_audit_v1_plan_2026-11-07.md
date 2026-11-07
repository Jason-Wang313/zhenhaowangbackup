# 下一项计划：只读核验实际已保存的Blur优化器更新

当前只有计划，尚无实现/prepare/新GPU提交。用户完整自主修复授权持续，仍Blur优先、官方150为唯一质量门槛。

已有证据：64已训练图20张方向错误、28张PSNR退步；实际LoRA参数梯度中MSE范数更大，两个global8的MSE/LPIPS梯度cosine为负。原始混合SGD的一阶判断与实际Adam行为不能等同。因此下一步读取已经发生的优化器更新，不先改loss系数或追加训练。

固定两条转移：当前1024来源试验的step0→1和step199→200。只用各步原本的global8训练样本，由原`core.sample_indices(before_step,rank,count=1024)`还原每rank四条；不能拿前8图代替实际训练批次。原来源、512裁图、参考缓存、prompt、数值路径保持。两个转移可能共享来源，不把它当新独立评价集。

CPU prepare逐份核验四个不可变checkpoint实际文件SHA、step/world/next_global_slot/模型hash/Adam步数/RNG，导出其原FP32 LoRA，保存原rank RNG；不改旧checkpoint/receipt。读取after checkpoint history对应的训练前loss/MSE/LPIPS/gradient_norm作为重放对照。无需扫描另外197份历史大文件。

单节点node3双H100，模型和loss完全沿用现有实现，无optimizer.step、不产生训练新权重。每条转移：

1. 加载before权重、原rank RNG；读取after权重求实际参数delta=theta_after−theta_before，导数用before模型。
2. 对原每rank四图分别计算10*MSE/8和.1*LPIPS/8的参数导数；保留图，再用原combined loss/8的backward累积真实combined梯度。三者分别NCCL SUM。
3. 不假定BF16下两分项梯度之和逐值等于一次combined backward；记录相对差异。核对combined梯度范数及batch loss与原日志一致（预先用loss绝对1e-7、norm按max(1,原norm)归一误差3e-4；失败时报告差异，不据未复现运行解释原训练）。
4. 计算分项/combined梯度与**已保存实际delta**的内积。负值是沿真实更新方向的一阶局部改善预测，仍不能替代有限步结果。
5. 加载已保存after权重，在同一global8上重新前向，记录真实连续MSE/LPIPS、原weighted loss及量化PSNR的前后变化。权重切换明确记录，每个检查点在求导/前向期间SHA不变。

保存每图及两rank完整汇总、输出范围和参数hash；GT只为这些已训练图的loss/诊断。新的候选选择、官方GT拟合、阈值扫描、超参选择均不发生。只比较已经存在的实际步骤，不把无GT部署安全性或泛化结论从训练侧推出。

判读边界：若实际训练批次两loss改善而许多已训练图/官方退步，优先研究跨图片干扰、覆盖与训练程度；若同批次有限更新损害一个/两个loss，结合实际delta内积及combined重放误差区分Adam方向、动量或有限步/精度效应。任何方向均需真实结果支持，不预定根因，也不直接保证后续配方能通过官方。

预计短诊断，仅node3两卡；实现与目的地预检完成、源码发布后才能提交，需记录实际耗时/退出码/GPU-hours并推小报告。全部旧资产/energy工作保持，不能重跑两个200步训练。
