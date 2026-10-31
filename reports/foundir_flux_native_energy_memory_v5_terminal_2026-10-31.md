# VJP输入隔离v5：数值相同，GPU显存仍未解决

2026-10-31，memory_isolated_vjp_v5，源码8e78eb9，单H200；18:49:48 UTC启动，
42秒后status1，0.011667分配GPU-hours。内部33.732秒，无optimizer更新。
真实4B同状态分数差0，一阶latent梯度差0；仍在第一个高阶更新OOM，
allocated峰值137.6805GiB，追加1.15GiB失败。未完成三步或最终反传。
OOM后清理仍有SavedTensorHooks内部assert，原样保留。

CPU已复现/修复的共享祖先回溯问题，没有解决真实GPU峰值，不能宣称修复完成。
当前停止增加未经分配证据支持的新计算实现；先采集allocator的实际存活分配栈，
保持v5运算不变。完整原result、日志及allocator摘要保存在同名JSON。
五个单卡诊断累计110+38+39+50+42=279秒，即0.0775分配GPU-hours；
加初始6卡资格1678 GPU秒，合计1957 GPU秒/0.543611 GPU-hours。
所有历史产物保留；没有长训或自动GPU重试。
