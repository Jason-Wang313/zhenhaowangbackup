# 当前FLUX-only实验：默认3节点×每节点2卡

## 用户决定与适用范围

所有当前及后续非轻量GPU工作默认3×2，总6张H100/H200；允许节点间混用。
当前每节点两卡需同型号、每卡至少75GiB；三节点软件/源码/共享项目目录保持一致。
真正轻量检查可用CPU或单节点1/2卡并明确说明。25步的4B三步二阶训练并非默认轻量。
保留显式单机2/4/8兼容入口，但禁止自动回退；其它重任务拓扑须用户另行决定。

本次没有启动GPU。主会话无GPU访问；本地CPU测试不能证明三节点NCCL/显存/速度。
不重复历史GT/灰度诊断，不恢复CNN/Restormer，不自动200步/正式训练。

## 改动与不变量

- 一个同步任务，world_size=6，nnodes=3，每节点nproc_per_node=2。
- global batch=8、microbatch=1、512图、完整三次latent更新、25步、数据/损失/学习率/
  FLUX-only架构及+.30dB/LPIPS相对降9.92810705%目标不变。
- 8不能整除6：每步两rank各2样本、四rank各1样本；每步轮换，多出的工作不固定压在
  某节点。每个样本乘6/8，all-reduce求和后除6，等价8个样本梯度平均。
  不同rank在局部反传中不通信，最后按相同参数顺序汇总，避免不等长循环的DDP死锁。
- 这并非完美负载均衡；同步速度受慢节点/额外样本/网络影响。不得承诺更快，也不
  擅自改global batch=12。六卡数据并行不降低每卡模型/二阶图显存要求。
- 每节点仅做一次CPU资产preflight。node0创建共享receipt；node1/2核对它，不抢写。
  所有rank在载入4B前完成身份核对、collectives和实际共享目录读写检查。
- H100/H200可跨节点不同；软件、源码、通信设定须一致。默认TCP兼容设置IB禁用、
  NVLS禁用；不猜网卡接口。如需接口设定，应三节点一致设置并显式传入tmux。
- 日志/锁/tmux按node rank分开，避免共享锁把node1/2误拦。每完整step仍保存全部
  6个rank RNG、Adam、全局采样位置及权重。主要checkpoint/result只由global rank0写。
- 新ID=`qualification_3x2_v1`，fresh开始。任何旧4/8卡或改动前源码断点保留，不直接
  改world_size来恢复。新3×2中断后只能同GPU身份/环境/代码/config三节点一起resume。

## 同步与启动边界

执行根仍为：
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
先安全同步本次Git提交的改动文件，以及先前966688d的FLUX-native实现。保留本地脏
文件、数据、权重、缓存、日志；不要reset/clean。三节点须看到同一共享工作根。

本次必须同步的运行改动：

- `configs/foundir_flux_native_energy_v1.json`
- `smoke/flux_native_energy_contract_v1.py`
- `smoke/flux_native_energy_launcher_lib.sh`（新增，不能漏）
- `smoke/train_foundir_flux_native_energy_v1.py`
- `smoke/run_foundir_flux_native_energy_v1_gpu.sh`
- `smoke/start_foundir_flux_native_energy_v1_tmux.sh`

同时同步AGENTS.md、agents.md、handoff、本报告、CPU测试及PPT资源更新。
CPU观察代理只同步/核对/回报，不自行启动GPU。用户要求启动时，须使用**当前**node0
可被另外两节点访问的地址，不能复用旧聊天IP。三个节点应依次尽快启动、不要等node0
完成后才启动node1。入口会等其他节点；tmux创建不代表训练开始。

为交接保留命令结构（占位地址必须替换；不是已执行命令）：

```bash
NODE_RANK=0 MASTER_ADDR=<当前node0地址> MASTER_PORT=29731 CUDA_VISIBLE_DEVICES=0,1 bash smoke/start_foundir_flux_native_energy_v1_tmux.sh qualification_3x2_v1 6 fresh
NODE_RANK=1 MASTER_ADDR=<相同node0地址> MASTER_PORT=29731 CUDA_VISIBLE_DEVICES=0,1 bash smoke/start_foundir_flux_native_energy_v1_tmux.sh qualification_3x2_v1 6 fresh
NODE_RANK=2 MASTER_ADDR=<相同node0地址> MASTER_PORT=29731 CUDA_VISIBLE_DEVICES=0,1 bash smoke/start_foundir_flux_native_energy_v1_tmux.sh qualification_3x2_v1 6 fresh
```

每行在对应节点工作根执行。各节点只暴露两卡，参数6是**全局总卡数**。
不得三行都在一个节点运行；不得缺一个节点却声称训练已开始。
新3×2中断恢复须三行都改为resume；完成的run拒绝重复启动。

日志：`logs/flux_native_energy_qualification_3x2_v1_node{0,1,2}_<UTC>.log`。
会话：`flux_native_energy_qualification_3x2_v1_node{0,1,2}`。
结果：`checkpoints/foundir_flux_native_energy_v1/qualification_3x2_v1/result.json`。
实际开始标记：`NATIVE_ENERGY_SHARED_STORAGE_AND_COLLECTIVES_PASS ranks=6`，随后缓存
生成；必须出现step=1的checkpoint与TRAIN记录，才可说已完成首个训练步。

## 成本与验证边界

每节点GPU阶段上限7200秒，包含torchrun会合、候选生成、训练和probe；3×2合计
上限12 GPU-hours，CPU资产核对和最多60秒退出清理另计。它是限额，不是速度预测。
各节点END记录local_gpu_seconds；把三节点的值相加才是实际总分配成本，不能再乘6。
真实吞吐/剩余时间由实际step记录计算。失败保留日志和已完成step，不自动重试。

本次16项CPU测试全部通过（unittest报告33.294秒，不含环境导入耗时）。
其中包括原生小尺寸FLUX的9项回归，以及新增7项拓扑/六进程测试。
CPU验证包括六rank Gloo样本加权对串行batch8、六rank Adam/RNG恢复、共享目录检查，
以及默认3×2命令、节点缺失/错误拒绝、混合节点身份、显式单机兼容。
这只是在一个CPU主机运行的六进程测试，**不是三台GPU节点的NCCL测试**。
三个shell的bash -n通过；dry-run未创建tmux或GPU进程。不虚构GPU验收通过。
