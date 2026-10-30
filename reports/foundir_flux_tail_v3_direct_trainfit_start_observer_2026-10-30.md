# 独立200步trainfit启动日志回报

UTC：2026-10-30T21:20:51.562468+00:00；run：`trainfit_3x2_direct_v1`；protocol `c47620eb566c525d69f3546dd42e5d04b01ed0db355fa291e02d9d7343addda1`。
三节点六H200已通过CONNECTIVITY_PASS，mode=trainfit、world6、steps200正确。
preflight明确`qualification_checkpoint_loaded=false`，资格result SHA与交接匹配。
初始反例24/24完成，有MINING_COMPLETE generation0和TRAIN_READY next_step1。
fresh step0已保存，文件SHA已核对；正在第1步microbatch，无RESUME标记。
最新完整checkpoint：step0，`steps/energy_000000_1793395188569585650.pt`，SHA `053610cf3613d2c376cc725713916cd8db2ceef7c77d5b3df894db4af67a2f02`。

| 节点 | 最后日志UTC | 错误标记 | END标记 |
| --- | --- | ---: | ---: |
| 0 | 2026-10-30T21:20:50.565240+00:00 | 0 | 0 |
| 1 | 2026-10-30T21:18:24.275931+00:00 | 0 | 0 |
| 2 | 2026-10-30T21:18:23.935495+00:00 | 0 | 0 |

尚无完整训练步/真实probe指标。预计11–13小时来自资格吞吐外推，本次不能据此宣称已完成。
原进度标记、节点启动/异常检查和checkpoint身份见[同名JSON](foundir_flux_tail_v3_direct_trainfit_start_observer_2026-10-30.json)。
只读共享日志和保存记录；未直接访问GPU进程、未修改运行代码或重启任务。
