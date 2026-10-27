# MoE梯度诊断V2：执行完成

diagnostic_v2在8×H200于2026-10-27T11:14:22Z以status=0结束，
耗时331秒=.09194444444444444普通小时、
.7355555555555555 allocated H200 GPU-hours。
索引修复已通过实际GPU执行；不是仅有CPU测试。

结果文件：checkpoints/foundir_dual_retry_v1/moe/diagnostic_v2/result.json。
SHA-256：c8de85120eaea8b5478f53017000d5353126d0f821be3ade77900094ae9cc54e。
独立检查全部51项来源hash一致；64条完整记录，
image_only/vlm_assisted各含16 Rain及16 Blur训练来源。
零optimizer steps，不修改模型权重；未读开发或保护像素。
V1失败仍保留，不用V2成功覆盖历史错误。

本报告确认诊断产物完整与工程完成，尚不对哪项训练损失导致Blur失败作因果结论。
结果仅包含代表性张量梯度范数和固定坐标方向抽样，
不能把它当全网络精确梯度、泛化成绩、最佳损失权重或VLM贡献证明。
后续解释需要结合各项损失大小、任务、共享/专家参数分别比较，
不能看到一个负余弦就宣布找到了唯一瓶颈。
