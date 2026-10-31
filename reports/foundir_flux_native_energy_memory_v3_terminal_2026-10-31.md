# Head分块诊断：梯度数值门槛仍未通过

2026-10-31，memory_heads_v3，源码c281061，单H200；18:31:37 UTC开始，
39秒后status1，0.010833分配GPU-hours；内部29.681秒，无optimizer更新。
一阶梯度区间峰值：原实现13.2679GiB，head分块9.2911GiB。
同状态energy分数绝对差0.0005463064；latent梯度relative-L2差0.0812790319
（8.1279%），超过预定1%，因此未执行完整三步/最终反传。阈值未放宽。

实际数值设置allow_tf32=true、float32_matmul_precision=high、math-SDPA低精度
reduction=false。此设置已记录，不足以单独锁定失败原因。没有更改这些全局开关。
完整日志和原始result在同名JSON；旧资产保留。
下一候选保留原attention函数、矩阵形状与原生autograd，仅逐attention重计算VJP，
去掉分块算术和手写导数。CPU通过后仍须同一真实数值门槛和GPU反传验收。
