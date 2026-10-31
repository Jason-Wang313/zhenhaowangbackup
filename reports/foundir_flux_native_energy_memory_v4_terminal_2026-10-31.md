# 原生attention VJP v4：数值完全匹配，首个高阶更新OOM

2026-10-31，memory_native_vjp_v4，源码399d3cf，单H200；18:44:07 UTC启动，
50秒后status1，0.013889分配GPU-hours；内部39.930秒，无optimizer更新。
真实4B同状态energy分数绝对差0，latent一阶梯度relative-L2差0，数值门槛通过。
首个unroll更新的create_graph输入梯度计算仍OOM，allocated峰值136.8115GiB；
追加申请1.15GiB失败。未完成三步或最终loss反传。OOM后退出清理另有SavedTensorHooks
内部assert，完整保留，不把清理报错当作已锁定的初始OOM原因。

CPU随后复现：Q/K/V共享投影/view祖先时，内层autograd.grad会访问调用方上游图；
分别clone后内层回溯不再访问这些祖先，外层高阶反传仍可通过clone传回。
这揭示了实际边界问题，尚未由GPU证明它完全解释或解决显存失败。
完整result/日志/allocator摘要在同名JSON。全部旧资产保留，无自动重启。
