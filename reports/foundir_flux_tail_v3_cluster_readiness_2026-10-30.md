# V3 三节点两卡工程验收准备

用户明确授权改为三节点、每节点两卡，并允许 H100/H200 混用。本节覆盖旧的
单节点/同型号限制。代码与 CPU 检查已完成，未在用户 GPU 节点启动任务。
当前只交付 8 步工程资格，不自动启动 200 步或正式训练。

## 已知节点与尚未验证的部分

| NODE_RANK | MASTER/节点地址 | 用户贴出的 GPU |
| --- | --- | --- |
| 0 | 10.247.94.142 | 2×H200，143771 MiB/卡 |
| 1 | 10.246.88.49 | 2×H200，143771 MiB/卡 |
| 2 | 10.244.58.183 | 2×H200，143771 MiB/卡 |

三台显示同一 GPFS 项目挂载，项目 Python 均可执行。这不是已经测通 NCCL：
可达端口、跨节点 collectives、CUDA 显存/速度和逐像素 FLUX 基线复现仍待实测。
代码允许 H100/H200 在节点之间或节点内混用，要求每卡至少 75GiB、SM90，且
所有 rank 的 Python/torch/CUDA/NCCL/所需依赖版本一致。当前三台实际都是 H200，
因此即使本次跑通，也不能声称 H100/H200 混用已真实测过。

## 改了什么

- 保留 1×4、1×8，新增并默认 3×2=6。全局 rank 与 LOCAL_RANK 分开处理。
  源码、run ID、Python、运行环境、实际 host/UUID 和共享存储都在全体 rank 核对。
- 一个 torchrun/process group 贯穿 cache→mine→train→refresh→probe→verify，
  避免各节点反复重建 rendezvous 的竞态。全局 rank0 单独持有共享缓存锁并写准备
  元数据；每节点保留自己的进程锁和日志。rank0 准备失败会广播错误。
- 每步恰好 32 个 pair，六 rank 分到 6/6/5/5/5/5；每个 pair loss 乘 world/32，
  再由 DDP 平均。统计量同样按 pair 加权；没有补样、丢样或改 batch/lr。
  每步只有最后一个本地 micro backward 同步，不同 rank 本地累积次数可以不同。
- 每步模型、Adam、六份 RNG 和采样位置继续原子落盘。新增 execution 身份绑定，
  拒绝换 rank/节点/GPU UUID/型号或环境后冒充原样恢复。混卡 CUDA 的 bitwise
  恢复尚未实测，不以 CPU 测试替代该证据。
- 所有 24 份训练缓存和全部 8 张 training_fit_only probe 均保留，六 rank 不丢样。
  不读旧八张检查图、校准、最终1859或任何保护像素。
- 只改变五个已有 V3 绑定文件，并增加两个执行 helper；其余 59 个绑定文件
  （含 V1/V2 和 energy 架构/损失）未变。保持 512、冻结 Noise step400/50步、
  alpha=.30、双 uint8、原种子、8工程步/step4刷新及一次尾段建议。
- 新输出根 `checkpoints/foundir_flux_tail_relative_energy_v3_cluster`，新的
  qualification_3x2_v1 / qualification_1x4_v1 / qualification_1x8_v1 预检已生成。
  旧 V3 准备记录留在原根，不篡改旧 receipt，也不把新目录当作旧训练续跑。

## 验证结果

- 75/75 CPU 回归，无失败、错误或跳过；包括 4/6/8 分片/加权、混卡身份接受、
  不合法设备/拓扑拒绝、跨 cwd 启动和 tmux 显式参数传递。
- 三个 torchrun agent、每个两个 CPU worker 的真实 Gloo 测试通过：六 rank
  不等长累积后的 loss 和二阶参数梯度与单进程 32-pair 参考一致；下一步在
  模型/Adam/每rank Python与torch RNG 持久保存恢复后逐元素相同；六份参数一致；
  改 world 或设备映射身份会拒绝恢复。
- 该测试在一台物理 CPU 机器上运行，验证的是三 agent rendezvous 与六进程
  collective/保存逻辑，**不是三台 GPU 的网络、混卡或 NCCL 通过**。
  输入为16×16合成数据，energy使用缩小宽度；不代替512图的CUDA显存验收。
- 66 个绑定文件的 Python/JSON/shell 静态检查、原 24 来源的 4/6/8 元数据和
  实际父缓存 SHA 预检通过。没有 GPU 模型训练，没有图像解码或新质量结果。
- 本轮原文件、源码前后 SHA 和 CPU 日志留在
  `checkpoints/foundir_flux_tail_v3_cluster_preparation/20261030T171430Z/`，不上传资产。

## 三台各执行的命令

先在**三台终端各执行**共同设置：

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
export FLUX_TAIL_V3_PYTHON="$PWD/.conda/gpu-train-stable/bin/python"
export NNODES=3 MASTER_ADDR=10.247.94.142 MASTER_PORT=29683 CUDA_VISIBLE_DEVICES=0,1
```

然后按 IP 在三台分别启动，尽量在三分钟内全部启动：

```bash
# 10.247.94.142
NODE_RANK=0 bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh qualification qualification_3x2_v1 6
```

```bash
# 10.246.88.49
NODE_RANK=1 bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh qualification qualification_3x2_v1 6
```

```bash
# 10.244.58.183
NODE_RANK=2 bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh qualification qualification_3x2_v1 6
```

MASTER_PORT 必须在节点0空闲并可跨节点访问。没有自动 SSH 或远程进程控制。
无需安装 `ip` 命令。NCCL 默认使用 sockets（NCCL_IB_DISABLE=1），不硬编码网卡名；
如平台指定网卡，可显式设置 NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME，tmux 会
携带这些值。RDMA 切换须按实际网络验证，不能仅凭 GPU 型号推断可用。

入口先进行真实 host/GPU/版本/共享存储和标量、4MiB all-reduce 检查，通过后才
处理训练缓存。缺失节点、设备不空闲或网络出错会退出，不自动重试或杀别的任务。
若只想先查网络，可在三台把最后命令中的
`qualification qualification_3x2_v1` 改成 `connectivity connectivity_3x2_v1`；
它只运行通信/共享存储检查，不加载模型权重或数据，5分钟 cap，独立成本最多
0.5 GPU-hours，必须另计。默认 qualification 已含通信检查，无需另跑这一任务。

日志是三个独立文件：
`logs/foundir_flux_tail_relative_energy_v3_qualification_3x2_v1_node{0,1,2}.log`。
看到 tmux 创建不等于训练开始。验收须节点0有 `TAIL_RELATIVE_V3_RESULT_VERIFIED`，
且三份日志各有 `TAIL_RELATIVE_V3_END status=0`。此前的
`TAIL_RELATIVE_V3_CONNECTIVITY_PASS` 只代表通信检查通过。

重启使用相同三个命令、相同 run ID/拓扑/设备和环境；读取最后完整 step，未完成
一步可能重算，孤立文件保留。不同 world_size 或 rank 到设备映射不支持严格恢复。

## 预算与边界

六卡资格硬 cap 7200 秒（两普通小时）/12 使用 GPU-hours，每节点最多另有30秒
清理宽限。保留原 GPU-hours 上限；初始化和 rendezvous 等待也算本节点 wall cap。
三份节点日志各记 `local_gpu_seconds=2×elapsed`，成本取三者之和，不能每份日志
都乘全局六卡再相加。所有尝试累计保留；此前实验成本不清零。若实际独占超过
两卡的整节点，额外报告整机占用成本，不能按可见两卡假定没有额外占用。
没有该拓扑的实测 ETA，不套用旧八卡估时或承诺线性加速。

四卡/八卡仍分别为10800/5400秒资格 cap；可设 NNODES=1、NODE_RANK=0，
给独立 qualification_1x4_v1 / qualification_1x8_v1 和卡数4/8。
未来六卡200步上限48000秒（13小时20分）/80 GPU-hours，但当前不启动。
必须先有源码绑定匹配的资格结果，手动决定、fresh初始化，不用资格权重。

工程通过不代表方向已学会或泛化成功，8步无改善也不终判整条路线。
最终目标仍为 energy 独立 PSNR +.30dB、LPIPS 相对下降9.928107050377874%。
