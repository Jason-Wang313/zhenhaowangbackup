# 单H200导数图重计算诊断：最终反传仍OOM

2026-10-31，memory_recompute_v1，代码cbc1975；18:12:47 UTC启动，
110秒后status1退出，0.030556分配GPU-hours。Python诊断内部74.113秒。
单个训练来源index17，512图，完整三步，未执行optimizer更新。

三次energy更新全部完成，其区间allocated峰值最高13.3024GiB；三个真实尾段解码、
MSE/LPIPS与排序loss均完成，最终反传前allocated14.1977GiB，之前区间峰值17.2524GiB。
随后loss.backward()仍OOM，allocated峰值135.3005GiB；再申请1.15GiB失败。
因此第一版外层导数checkpoint只解决前半段保存量，未解决最终反传重计算的显存峰值。
不能宣称H200/H100已通过、恢复训练或调整科学结论。

完整原始result、阶段记录、报错、allocator摘要及完整日志保存在同名JSON。
所有旧资产保留。下一步针对energy attention导数分块，先检查数值和高阶梯度等价；
无自动GPU重试，不改三步/512/global batch8/损失，不恢复CNN。
