# v6显存分配跟踪已采集；原计算仍OOM

2026-10-31，`memory_alloc_trace_v6`，代码531e413；单H200，计算与v5一致。
19:07:45 UTC启动（日志文件名），launcher用时54秒，以status1退出；
结束时间据此推算19:08:39 UTC。内部诊断计时42.738秒；使用0.015 GPU-hours。
同状态分数及一阶latent梯度误差均0，通过原1%门槛。
第一次高阶更新即OOM，峰值allocated **137.680503 GiB**，追加1.15 GiB失败。
没有完成第一次更新、三步展开或最终反传；optimizer更新0，资格未通过。
异常清理时的SavedTensorHooks内部assert也完整保留，未将其当作已确定根因。

## 新获得的证据

`NATIVE_ENERGY_MEMORY_SNAPSHOT_SAVED`已出现，快照8,521,833字节，SHA256核验通过：
`227d8197b94110dd64d0da42b5ad6ac402934afb1ae5392dc5cfeb8cd093d32a`。
26项运行源码哈希与共享目录全部一致；allocation_summary与原快照独立计数一致。
快照只留共享目录，不上传GitHub；同名JSON保存完整result、内部日志、bootstrap日志、
memory.jsonl、CUDA摘要，以及所有原文件SHA和CPU分析。

异常后捕获时仍有133.905050 GiB存活分配，等待释放为0；这不是137.680503 GiB峰值快照。
按分配栈第一个`at::_ops::`归类：mul 50.594128 GiB、add 35.839154 GiB、
empty_strided 21.672340 GiB、无可识别operator栈7.760705 GiB；这是分配来源，不能直接
将全部归为attention矩阵、clone或某个张量。125.355734 GiB（93.6154%）存活分配的栈
同时含`_native_vjp`、`value_and_gradient`及checkpoint的`recompute_fn`。
OOM栈也显示局部VJP调用中触发外层energy梯度重计算，再进入局部VJP。
这支持继续检查嵌套求导/重计算的张量保留关系，尚未证明具体根因或验证修复。

历史记录达到30,000条上限且首条已是释放事件，开头已截断；异常展开也可能释放峰值张量。
因此不能将此快照包装为完整峰值生命周期重建。已保存一次OOM事件：
请求1,233,814,016字节，当时device_free为943,718,400字节。

本轮没有更改计算、数值门槛或配方，没有自动重试/启动训练。
下一工程问题是CPU隔离嵌套checkpoint/VJP的重计算和保留行为；当前没有新GPU启动命令。
六次单卡诊断累计333秒/0.0925使用GPU-hours；加初次六卡资格1678 GPU秒，
本FLUX-native阶段累计2011 GPU秒/0.558611使用GPU-hours。更早历史实验成本仍保留。
