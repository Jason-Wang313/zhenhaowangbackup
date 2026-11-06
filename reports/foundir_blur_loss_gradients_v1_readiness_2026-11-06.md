# Blur：两检查点、固定8训练图的真实LoRA梯度诊断

64已训练图中20张修正方向非正相关、28张PSNR差于LQ，说明不能只归因官方迁移。旧资格仅证明加权LPIPS的RGB梯度范数是加权MSE约5–12倍；网络Jacobian及逐图抵消可能改变实际参数方向，不能把RGB比值当参数更新证据。

固定当前1024来源的前8条及已有512裁图，两个明确检查点：原Blur step10000初始化（参数hash1643c425...）和当前最终delta step200（3c2d0d76...）。只加载/求导，不创建optimizer，不执行参数更新。使用真实10*MSE_RGB01和.1*默认pyiqa LPIPS；模型train模式及原checkpoint路径，初始化输出必须逐值等于LQ，step200量化输出必须复现之前64训练图诊断的对应PNG。

分别计算每图两个loss的RGB梯度和全部32,440,320个LoRA参数梯度，报告范数、LPIPS/MSE比、夹角、混合梯度与各分项梯度内积，并按double/single block及A/B分组。两rank各4图，先把每图梯度除以8，再真实NCCL SUM，另报完整global8梯度几何；不能用平均逐图norm冒充global梯度norm。每检查点求导前后参数SHA不变，.grad不累积，零梯度明确报告null夹角/比值。

混合梯度与MSE梯度负内积只表示沿原始SGD混合下降方向的一阶局部MSE增加，**不是实际Adam更新、有限步收益或官方失败根因的证明**。本轮不执行假设更新，不生成多个候选、不拟合官方GT/阈值、不挑最佳输出。

单节点node3双H100，8图×2检查点，共16组逐图诊断，外加两个8图汇总。无其它训练集/开发集/测试集质量门槛；唯一质量验收仍完整官方Blur150的25.9/.18。全部GT只来自这8个已有训练来源。

4项CPU数值测试、Python/shell语法检查、真实权重/缓存/父结果prepare通过（2.037856秒/GPU0）；新GPU诊断待发布后提交round11_blur_loss_gradients_v1。入口smoke/diagnose_foundir_blur_loss_gradients_v1.py prepare|run|finish，launcher smoke/run_foundir_fullstack_blur_loss_gradients_v1.sh 3。输出checkpoints/foundir_blur_loss_gradients_v1。短任务无独立持久监视器，当前代理须检查实际两rank/launcher终态，再CPU finish并推小报告。预估1–2分钟，仅估计；以实测费用记录，不自动重试/追加训练。

保留旧所有源码、模型、缓存、日志和energy工作。完整自主研究授权持续有效；根据诊断固定下一受控训练方案，不因本次求导工程通过就宣布修复。
