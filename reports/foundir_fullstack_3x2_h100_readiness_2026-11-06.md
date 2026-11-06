# 三节点六张H100：执行器及官方输入基线

当前三台为用户重新编号的新实例，旧node映射不可复用。IP和notebook详见同日preflight JSON及AGENTS.md顶部。
旧四节点任务未启动、无任何GPU结果，已写cancel.json；全部旧代码、预检和队列保留。

新入口 `bash smoke/start_foundir_fullstack_worker_3x2_v2_tmux.sh <0|1|2>`。
分别在10.244.211.17、10.244.47.55、10.245.39.185上使用0、1、2。
使用共享Python和 `.tools/tmux-env/bin/tmux`，无需系统tmux。启动前核验IP与两张同型号H100>=75GiB；登记实际UUID，防止同卡冒充不同节点。
新tmux session `foundir_fullstack_worker_3x2_v2_node<N>`，日志 `logs/foundir_fullstack_worker_3x2_v2_node<N>.log`。
心跳 `checkpoints/foundir_fullstack_executor_3x2_h100_v2/workers/node<N>/heartbeat.json`。
常驻程序空闲不加载CUDA；任务失败保留日志与退出码，不自动重试。原四节点任务不进入新队列。

本轮只计算官方600张未处理输入的PSNR/LPIPS/SSIM；6worker各100张，无遗漏或重复。
三个节点各自使用standalone torchrun启动两个独立评分worker，无跨节点NCCL或MASTER_ADDR依赖。
新输出 `checkpoints/foundir_fullstack_official_input_baseline_3x2_v2`。
`FOUNDIR_OFFICIAL_NNODES=4`只用于读取已完成父实验的冻结身份，不代表本轮启动4节点；本轮分片与汇总显式使用6worker。
不重跑训练、路由、恢复生成，不更改评分口径或官方600图集合。

CPU预检通过，12项工程测试通过，包括新IP/型号验证、错误节点及重复GPU拒绝、600图六分片完整性、六worker汇总及缺记录拒绝、任务失败不重试、参数字面传递、源码修改拒绝。
旧四个入口与已发布6e0f985逐字节一致。真实GPU任务尚待用户启动常驻执行器，CPU通过不代表GPU已执行。

全部6worker完成且3个任务退出码0后运行：
```bash
.conda/gpu-train-stable/bin/python -S smoke/audit_foundir_fullstack_official_baseline_3x2_v2.py finish
```
完成后推送小结果JSON、各节点退出码与实际GPU-hours，不上传缓存/图片/权重或live logs。
