# V3 direct：8步资格之后的独立200步训练拟合

## 本次决定与实际状态

用户已在看完8步终态后明确要求continue。下一任务沿用已通过的工程实现，
不新增模型/损失/数据/诊断，不重跑资格。主代理已准备此交接；当前新共享工作根
未挂载到主代理，所以这里没有启动GPU或确认远端进程。本文件不是完成记录。
目的地CPU代理仍只同步/核对/回报；用户在GPU节点启动已批准的任务。

唯一执行根：`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
执行Python：该根下 `.conda/gpu-train-stable/bin/python`；不修改环境。

## 冻结运行身份（使用现有启动器，无代码修改）

| 项目 | 值 |
| --- | --- |
| 现有入口 | `smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh` |
| mode | `trainfit` |
| run ID | `trainfit_3x2_direct_v1` |
| world/layout | 6 / 三节点各2GPU，沿用刚成功的H200节点 |
| qualification run ID | `qualification_3x2_direct_v1` |
| qualification result SHA | `e35f33204d4496fb095e28f41f2cd11ef770c2c4b60cf0a1f88a0bd0300d5a92` |
| 已验收代码提交 | `7e3c8fbc0323239e91487b8bdb5cba1894542b97` |
| 科学配置 SHA | `9335e8cbe9469bbfaec468159c1c7899a4f6768fc23bfeb562558ceb4e4856c5` |
| 输出 | `checkpoints/foundir_flux_tail_relative_energy_v3_direct/trainfit_3x2_direct_v1` |
| 总optimizer steps | 200 |
| 反例刷新 | step50 / 100 / 150 |
| 实际图片probe | step200结束后，原8个训练来源，一次修改 |
| global batch | 32对，六rank为6/6/5/5/5/5，仍按world/32加权 |

200步是独立fresh模型和Adam；资格result只证明工程通过，绝不加载资格权重。
它不是训练step8接到step208，也不是正式泛化/五类全量训练。
训练过程中逐step保存模型、Adam、每rank RNG和采样进度；中断后保持同一run/拓扑/
设备映射/运行环境才允许真正恢复，不重新初始化冒充续训，不删除任何旧产物。
三节点需同一run/qualification ID、Python、共享根和当前可达MASTER_ADDR。
复用刚成功任务的实际rank映射；若节点已更换，先报告，不从旧聊天IP猜测。
现有锁、身份/缓存校验和NCCL检查是正常入口的一部分，不另起诊断任务。

## 已完成的证据，不重复消耗GPU

终态报告 `foundir_flux_tail_v3_direct_terminal_2026-10-30.{md,json}`，Git db03d94。
三节点均于20:36:05 UTC status0结束，8步全部saved，完整8图probe、RESULT_VERIFIED。
每图数值重新平均与报告一致，训练记录的loss/梯度均有限。59项Git可见源码SHA一致；
8项third_party代码和远端checkpoint实物仅有目的地报告，主代理不声称独立重读。
结果：current original 38.176074505/.080517740；bounded 38.784466743/.078427173。
PSNR +.608392239dB达均值目标，LPIPS降2.596406%未达9.928107%；3/8图PSNR变差，
3/8图LPIPS变差，LPIPS改善集中在一图。工程通过，不是学习/泛化通过。

本交接不修改任何绑定源码/配置，故不触发重新资格。200步正常入口仍核对qualification
源码身份和result，通过才开始；不取消断言，不预先生成缓存，不改历史reference记录。

## 预算与观察点

六H200资格实测187.331965秒/optimizer step；200步纯训练外推10.4073小时。
计入bootstrap、三次反例刷新、最后probe及启动开销，估计11–13普通小时，
66–78使用GPU-hours。不是保证；吞吐/共享I/O变化会影响耗时。
既有硬上限48000秒=13小时20分，80使用GPU-hours，另30秒清理宽限。
若实际独占节点有额外闲置GPU，分配成本应按真实分配数另报，不能拿使用成本替代。

CPU观察代理确认开始时应看到mode=trainfit、world=6、steps=200，以及fresh step0/1；
正常恢复才应出现RESUME。之后回报step50/100/150已保存及对应MINING_COMPLETE。
这些不是质量验证点；真实8图指标只在step200跑完后产生，不承诺中途PSNR趋势。
终态需要TRAIN_COMPLETE、PROBE_COMPLETE、RESULT_VERIFIED及三节点status0。
若到硬上限或外部停止，保存事实/最新完整step，不自动重启或加预算。
报告按原流程推GitHub，不提交权重、缓存、图片或整份活日志。

## 距离三关还剩几轮

1. **能不能做到：已有有限范围的肯定证据。**
   `foundir_flux_tail_confirmation_v1_terminal_2026-10-28.md` 的额外24来源，
   固定GT多轮搜索+.986179dB、LPIPS降31.562025%，22/24单图双目标通过。
   这些来源是开发用训练来源，不是代表整个数据集的随机样本，更不是learned泛化。
   此结果用了多轮搜索；同报告第一轮LPIPS仅降4.284375%，不能拿16轮GT成绩
   保证当前单步energy必过。不再重复该GT确认或修改本轮单步协议。
2. **能不能学会：下一轮200步回答训练侧问题。**
   报相同图片上的raw/bounded、当前原FLUX/FP32/历史三种对照，保留所有图。
   同时看PSNR+.30、平均LPIPS相对下降.09928107050377874以及误接受/受损图。
   不用均值好看掩盖局部失败，不把梯度cosine当像素正确率，不自动延长到100k。
3. **能不能泛化：尚未实现/冻结，必须独立定义。**
   如果200步能学会，冻结模型与推理策略，在训练和选参未使用过的source groups
   上评估。不能把当前8张probe、旧反复查看的8张检查图或GT确认来源重标为unseen。
   若要校准阈值，用单独来源的校准集，与最终报告集分开；不能在评估结果上反复调。
   继续禁止official/sealed/internal/parent-test及最终1859像素作为训练或调参资源。

**最顺利再2轮得到首次独立来源结果**：200步拟合，然后冻结评估。
更现实以3–4轮规划：中间可能需要一次扩大训练、一次独立校准；不是强制增加实验。
若单步目标学习明显不足，可能要另行设计多步或修复训练，轮数不能保证。
后续阶段要根据实际结果单独决定，不因本文件规划就自动运行。
首次Noise泛化成功也不是完整论文；其他退化、双退化、VLM及等算力消融另计。

## 本次同步范围

仅更新以下五份文档，不重新拷贝/改动已运行成功的代码或旧18文件同步清单：
`AGENTS.md`、`agents.md`、`reports/foundir_current_handoff_2026-09-23.md`、
`reports/foundir_energy_design_history_and_failure_playbook_2026-10-30.md`、本文件。
旧清单记录的是其提交时的文档SHA，不是禁止后来更新交接；代码/配置SHA仍必须匹配。
保留本地无关修改，有冲突先备份并报告。同步报告记录实际共享根/提交/SHA及未启动GPU。
本地静态检查仅验证现有launcher的3x2 trainfit参数展开，不做CUDA或缓存/数据操作。
