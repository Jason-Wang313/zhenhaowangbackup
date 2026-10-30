# V3 direct 三节点启动日志回报

采集UTC：2026-10-30T19:59:55.199625+00:00。run ID：`qualification_3x2_direct_v1`。
同步代码版本：`7e3c8fbc0323239e91487b8bdb5cba1894542b97`。
运行preflight SHA：`dfb0b22a505541f2446362777a4f4f17dffa464c59928009e8d74176eaec87f4`；
运行绑定源码/配置67项。

三台于19:57:07/08/10 UTC启动。六rank、三节点、六张实际H200，
共享存储及标量/4MiB NCCL检查通过，日志有`CONNECTIVITY_PASS`和`PREPARED`。
基础缓存24/24完成，日志有`BASE_COMPLETE`。当前处于初始反例准备阶段，
尚未出现optimizer step/训练checkpoint/probe/result；不能称训练步骤已开始或资格完成。

| node rank | 最后日志更新时间UTC | Traceback/RuntimeError/NCCL error/END标记数 |
| --- | --- | ---: |
| 0 | 2026-10-30T19:59:41.836389+00:00 | 0 |
| 1 | 2026-10-30T19:59:42.273382+00:00 | 0 |
| 2 | 2026-10-30T19:59:53.507668+00:00 | 0 |

现有警告为`pkg_resources is deprecated`，未见异常终止。日志位置：
`logs/foundir_flux_tail_relative_energy_v3_qualification_3x2_direct_v1_node{0,1,2}.log`。
以上为采集时刻共享日志快照；未直接访问远端进程，未修改运行代码或重启任务。
