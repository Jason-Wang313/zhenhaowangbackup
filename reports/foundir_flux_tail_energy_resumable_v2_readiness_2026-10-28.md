# 八卡与逐步保存修复：同一 FLUX-tail energy 25步资格

## 决策与边界

用户明确要求用满八卡并保留进度。本次只修执行方式、缓存复用、保存/恢复与预算，
不改energy架构、损失、初始化seed、global batch、来源划分、512裁图、lr或25步规模。
不自动正式训练，不读取任何protected或最终1859开发集像素。

旧四卡已生成全部32图缓存，训练步速246–250秒，原30–60分钟全程估计错误。
旧程序只在step25落盘，当前显存中的中间训练权重/Adam无法靠修改文件补保存。
因此这次切换是复用缓存、重算旧未保存步骤，不是无损恢复旧内存状态。
旧缓存、日志完整保留；切换程序还保存日志快照及最终已记录步数。
若发现旧step25权重已经存在，程序拒绝丢弃它，要求先审查迁移。

## 保存和恢复

- 初始化step0及每个完成的optimizer step保存不可变checkpoint和SHA receipt。
- 保存模型、Adam、各rank Python/torch CPU/CUDA RNG、warmup位置、下一global slot、
  完整训练日志、累计时间和显存峰值；保持同global batch32候选对。
- 同目录临时文件写完并fsync后才发布，latest单独原子更新；保留所有旧步。
- 恢复取最高完整receipt，不依赖可能滞后的latest指针；损坏校验失败时明确退出。
- 每个完整步恢复后不重算；突发中断时尚未提交的一步可能需重算。
- 评价每张图完成后保存；重复运行校验并跳过已完成图，不以GT挑检查点。
- 默认8卡，每rank累积4对；4卡兼容，每rank累积8对。当前恢复只允许同world_size，
  不能把重新分组求和冒充逐位一致的4↔8恢复。
- 用户可创建run目录中的STOP_AFTER_STEP请求保存完当前步后退出；再次启动不会
  自动忽略这个标记。不自动重启或无限扩预算。

## 安全切换

新启动器先核验并复用32个原缓存，再仅针对旧train脚本/指定run-ID/4卡参数/项目cwd
匹配进程。检查GPU没有无关任务，SIGTERM旧torchrun，最多等待45秒，不广泛杀Python。
保留旧日志快照后获取节点锁、检查全部卡空闲，再启动新任务。
使用同型号单节点4/8×H100/H200，每卡>=75GiB，不混卡、不跨节点。

默认命令（也用于今后恢复同一任务）：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_flux_tail_energy_resumable_v2_tmux.sh
```

默认八卡0–7，不继承旧tmux server的四卡可见环境。
日志追加至logs/foundir_flux_tail_energy_resumable_v2_qualification_8gpu_v2.log。
输出为checkpoints/foundir_flux_tail_energy_resumable_v2/qualification_8gpu_v2。
完整通过须同时有TAIL_ENERGY_V2_RESULT_VERIFIED与TAIL_ENERGY_V2_LAUNCH_END status=0。

## 实测与预算

52项CPU回归通过；两进程真实二阶DDP恢复测试通过，下一步参数与不中断路径逐元素相同。
实际32图缓存导入/哈希核验通过，未重新生成图片。Python/JSON/shell静态检查通过。
新八卡GPU训练尚未开始，Codex节点无GPU，也未停止远端旧任务；需要用户执行切换命令。
八卡CUDA保存/恢复吞吐尚未实测，不能用CPU通过冒充GPU通过。

八卡粗估55–75分钟、7.3–10使用GPU-hours，不保证两倍加速；旧四卡成本另计。
每次尝试硬cap八卡90分钟/四卡180分钟，均12使用GPU-hours，另最多30秒清理。
超时仍保留所有已提交训练步及完成的评价样本；不自动重新启动。
若使用四卡却独占八卡整机，另报整机占用成本。

真正效果目标仍是energy相对匹配FLUX独立增加+.30dB、LPIPS下降9.928107050377874%。
工程完成、缓存GT收益、低训练loss都不能替代无GT推理与正式泛化证据。
