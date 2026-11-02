# 已有FLUX专家重放：四节点×两卡纠正

## 纠正了什么

用户要求4个节点，每节点2张H100/H200，总8卡。此前dbd8b29按单节点实现错误；
只交付过代码，没有启动GPU，也没有需要迁移/停止的本轮GPU任务。
本次只纠正资源分工、跨节点等待、保存与汇总，不改研究方案。

当前新根：`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
四节点须能同时访问这个共享目录。Python默认新根`.conda/gpu-train-stable/bin/python`，
该路径曾在目的地CPU节点核验；仍须确认每个GPU节点均可用，不假设复制项目等于复制环境。
旧HDD3仅只读资产参考；本地隔离worktree用于实现/测试，不在那里跑GPU。

## 资源与执行

- 固定`nnodes=4`、`nproc_per_node=2`、world size 8；每节点`CUDA_VISIBLE_DEVICES=0,1`。
- global rank = node rank × 2 + local rank。原任务分片不变，每rank16图，共128图不重不漏。
- 节点0处理Blur/Noise第一半，节点1处理Haze/Lowlight第一半；节点2/3处理对应第二半。
- 同节点两卡必须同系列且≥75GiB。节点间H100/H200可不同：同图两条路径始终在相同rank、
  同GPU系列/软件环境比较，不把混合集群结果称作统一硬件复现。
- 每节点只有一个FLUX和VAE副本/卡；不增加训练网络，没有Restormer/CNN恢复器。
- node0准备共同收据，并短时用本节点GPU0离线编码四条prompt；其余节点等待。
  四节点使用同一份prompt tensor，随后通过torchrun静态rendezvous启动共8个worker。
- 需要当前node0可达地址和端口（默认29681），不要拿历史IP填入。不改防火墙/自动杀其他任务。

入口仍是`smoke/start_foundir_existing_flux_replay_v1_tmux.sh`，但参数已改为四个必填参数：
`<node_rank 0..3> <当前node0地址> <run_id> <本次共用fresh_launch_id>`。
不再接受旧`replay8_v1 8`两参数命令，不提供隐式单节点默认。
目的地代理拿到实际当前地址后给用户四条命令；主代理不编造地址或假称已经启动。

## 完成与恢复

同一run的四个节点各有共享锁`nodeN.lock`和本机锁；防止同rank或同节点重复启动。
每次尝试的launch ID必须四节点一致且从未使用，先各自独占claim，再等待四个不同hostname
的boot标记；master地址/端口必须一致。共享路径不通、节点未到齐会等待并到预算超时，
不会擅自降级成单节点。中途失败留下failed标记，各节点每2秒检查并停止自己启动的
timeout子进程组，GNU timeout转发TERM及最多30秒清理；没有标记的断电/SIGKILL仍由
torchrun及外层硬时限处理，不承诺跨节点零延迟停止，不杀无关任务。

node0生成receipt；各节点检查相同源码/资产身份，保存不可变`runtime_nodeN.json`；
全部ready后进入采样。每节点两条本地worker都成功才发布done。四个done齐全且身份一致，
node0才读全部保存tensor重新核验；仅`RESULT_VERIFIED pass=True records=128`后发布verified。
其他节点等verified，四份`EXPERT_REPLAY_NODE_END status=0`才是完整成功，不是tmux创建即成功。
result记录四个runtime及其SHA。GPU类别/软件版本变动会拒绝复用旧节点runtime。

恢复保留run ID和原四节点rank角色，用新的共用launch ID；重新核验源码、资产、
runtime和逐图SHA后跳过已完成图。旧单节点receipt即使world也为8仍被拒绝，不能冒充兼容。
source/config变化不支持在同run静默恢复。张量和不可变收据逐图原子落盘，未提交孤立文件保留。
不覆盖历史图、不把重算说成无损续训；本轮没有optimizer或训练进度。

每节点日志：`logs/foundir_existing_flux_replay_v1_<run_id>_node<N>.log`。
成本：`checkpoints/foundir_existing_flux_replay_v1/<run_id>/attempts/<launch_id>_node<N>.json`。
总GPU-hours = 四份allocated_gpu_seconds之和/3600，包含准备和等待。断电/SIGKILL可能无终态
成本文件，须按实际日志时间补报，不把缺文件当零成本。
粗估集群0.5–1.5小时、4–12GPU-hours未经跨节点实测；每节点硬cap2小时，四节点总cap
16GPU-hours，另最多30秒进程清理和最多20秒失败标记写入。不会自动重启、扩预算或训练Rain。

## 科学协议没有变化

Blur/Noise/Haze/Lowlight各32训练来源、512 LQ；不打开GT，不访问开发/封存/官方测试像素。
原权重、任务prompt、crop/seed、50步、CFG1、reference time10、原alpha和量化顺序全部保留。
每图原LoRA与四分支bank固定选择同一expert：decoded RGB最大绝对差≤1e-6且uint8逐元素相同。
没有PSNR/LPIPS新成绩，没有动态soft/router验收，不是Rain适配或MoE科学成功。
此前资产审计配置中的single_node GPU policy只是当时建议；为了不伪造审计哈希，原文件不改。
新replay config和本报告才是执行拓扑。其余审计的权重/数据/源码限制继续严格校验。

## CPU验收与目的地交接

最终34项CPU测试全部通过，测试体79.849秒（不含共享盘Python/torch导入）。
真实资产只读预检通过，32项源码绑定复核一致，选图摘要不变；shell语法及Python编译通过。
首轮硬件mock测试误把依赖包元数据也模拟了，已修复测试隔离后完整重跑；没有跳过生产依赖检查。

本次CPU测试覆盖总rank分片、local/global rank映射、旧单节点收据拒绝、两卡型号检查、
四节点模拟ready/done/恢复、陈旧标记/失败/地址冲突、node0独占共同标记，以及原LoRA算子等价。
完整命令为测试模块`test_existing_flux_experts`、`test_existing_flux_replay_v1`、
`test_existing_flux_replay_cluster_v1`，低层4-rank分片测试保留，但实际入口只支持4节点×2卡。
测试结果和真实资产预检另存`foundir_existing_flux_replay_4node_cpu_preflight_2026-11-02.json`。
这不是跨主机网络或CUDA实测；128图全采样与显存/速度必须到GPU节点验证。

目的地代理：安全同步本次focused提交（保留脏文件和所有运行资产，禁止reset/clean），
检查四节点共享新根、Python和当前node0地址；运行CPU测试与`preflight --gpus 8`。
选择摘要应仍为`2682b41f653d2e3bc69d1a5964895da8e1cf20f0d7cec0d08762b5dc52c03b36`。
随后给用户四条分节点命令；不替用户启动、不顺带恢复旧retention/energy。
完成后只回传小报告/收据身份/成本，绝不上传模型、数据、缓存、tensor或实时日志。
