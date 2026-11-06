# 全流程执行器与首轮诊断 — 2026-11-06

全流程改进授权及职责已记录于 AGENTS.md 顶部，提交 f285a8a。官方五类600张为反复使用的自适应基准；官方GT只评分和分析，不进训练loss、梯度、逐图选路或选输出。

## 已完成分析

600张已保存输出的只读CPU分析完成，退出码0，255.740630秒，0 GPU-hours。结果为 foundir_fullstack_round1_cpu_audit_2026-11-06.json。

|类别|输入PSNR（NumPy诊断）|正确专家PSNR|比输入退步图数|
|---|---:|---:|---:|
|Blur|21.863902|21.546762|91/150|
|Noise|37.417881|37.693592|38/100|
|Haze|13.466771|21.963813|0/200|
|Rain|23.448853|32.034930|0/100|
|Lowlight|6.985977|19.231371|0/50|

NumPy与原pyiqa PSNR最大差小于0.000006 dB，仅用于诊断；正式验收仍用原pyiqa。Blur路由已正确而专家整体退步，不能靠继续训练router解决；Noise同时存在路由错误与专家恢复差距。当前没有新的训练结果，没有达到全部官方阈值。

## 常驻执行入口

共享目录不变，每个节点执行一次：`bash smoke/start_foundir_fullstack_worker_tmux.sh <node>`。
node0/1为两张H200，node2/3为两张H100；执行时核验真实型号、>=75GiB显存、物理GPU UUID和节点锁。
使用 `.tools/tmux-env/bin/tmux -L zhenhaowang_jobs`，无需系统tmux或fakebin。
空闲时仅标准库轮询共享队列，不加载CUDA。日志 `logs/foundir_fullstack_worker_node<N>.log`；心跳 `checkpoints/foundir_fullstack_executor_v1/workers/node<N>/heartbeat.json`。

仅接受新 `smoke/run_foundir_fullstack_*.sh` 入口；提交前源码必须与已发布origin/main祖先commit匹配，执行前再校验哈希。参数使用列表传递；保留开始、结束、退出码和两卡占用耗时。失败不会自动重试；孤立未结束attempt需要核对；取消只终止本执行器创建的进程组。后续科学决策与任务提交由当前代理完成，常驻程序本身不会生成新配方。

## 首项GPU任务

官方600张未处理输入的PSNR/LPIPS/SSIM基线，用于分离路由和专家的问题；不加载FLUX、不恢复图像、不训练、不读取其它测试集。四节点各两个独立worker，无跨节点NCCL依赖，无需MASTER_ADDR。复用已验证的pyiqa CPU构造/整模型迁移身份修复。

输出 `checkpoints/foundir_fullstack_official_input_baseline_v1`。CPU prepare已通过；仅预检receipt在首次GPU启动前因完成worker复用修复而保留到 local_sync_backups，重新绑定当前代码。没有覆盖GPU资产。

六项执行器CPU测试及shell语法通过。真实GPU执行器启动、评分显存和吞吐待现场运行；不把tmux创建当成任务已开始。GPU完成后执行本轮脚本finish，推送小JSON及实际耗时/退出码；官方旧结果与累计106.544444444 GPU-hours保持记录。
