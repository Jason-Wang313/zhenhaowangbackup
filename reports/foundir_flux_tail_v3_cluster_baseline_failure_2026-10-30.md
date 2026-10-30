# V3 三节点首轮资格：基线复现检查失败，尚未训练

在记录结果前，已 fetch 并完整阅读远端 `285e73a` 的
[设计演变与失败排查记录](foundir_energy_design_history_and_failure_playbook_2026-10-30.md)，
以及最新 AGENTS/current handoff。按其第7节先定位无引导基线，不能将本轮解释为 energy 学习失败。

`qualification_3x2_v1` 三节点均已以 status=1 结束；本报告是一次尝试的终态，
不是 V3 路线终判。观察来自共享日志和落盘记录，观察节点没有远程GPU进程控制通道。

| 项目 | 核验结果 |
| --- | --- |
| 实际硬件 | 3节点×2 H200，六rank；尚无本轮H100/H200混用实测 |
| 启动连通 | 共享存储检查、六rank身份、标量及4MiB NCCL allreduce通过 |
| 基础缓存 | 24/24记录，1706候选帧，complete.json存在 |
| 初始反例 | 0记录；六rank均在首次历史基线比较失败 |
| 训练与probe | 0/8 optimizer steps、0 probe、无训练checkpoint或result.json |
| 失败 | `RuntimeError: cached historical baseline changed` |

开始于2026-10-30 17:56:16 UTC，最后节点于18:00:15 UTC结束，
首启动至末退出239秒（3分59秒，.0663889普通小时）。各节点计费如下；
按本任务实际使用两卡计算，不将六卡乘到每个节点日志上。

| node rank | 开始UTC | 结束UTC | elapsed秒 | local GPU秒 | status |
| --- | --- | --- | --- | --- | --- |
| 0 | 17:56:16 | 18:00:14 | 238 | 476 | 1 |
| 1 | 17:56:18 | 18:00:15 | 237 | 474 | 1 |
| 2 | 17:56:19 | 18:00:14 | 235 | 470 | 1 |

合计1420 GPU秒，即 **.3944444使用GPU-hours**，包括启动/等待，不能在下轮清零。
历史其他任务成本另计；没有吞吐或后续训练ETA实测。

## 失败位置和解释边界

`mine(..., generation=0)` 从已绑定的原训练缓存读取 observation/proposal/seed，
调用 `FrozenFlow.session(..., expected_proposal=p)`。该函数先用冻结模型重新生成
50步FLUX结果，执行既定双uint8混合，再逐元素比较历史proposal；在
`smoke/flux_tail_energy_runtime_v1.py:53` 触发异常。此时尚未进行energy引导更新，
也未到达后面的tail与历史入口一致性检查。六rank都出现同一异常。

网络启动和基础缓存阶段已通过，但不代表全部CUDA训练链路通过。
本次重新核对66个绑定源码/配置哈希，全部一致；24个基础记录metadata SHA、
来源身份和源码绑定一致，对应tensor文件均存在。此次状态审计没有重新读取全部
基础tensor内容计算SHA，也没有重新解码原图或访问受保护像素。

当前运行时为Python3.12.13、torch2.7.0+cu128、CUDA12.8、NCCL2.26.2。
标量/4MiB allreduce的最大rank耗时约.10151/.11242秒，只是冷启动正确性检查，
不是稳定网络带宽测试。execution与证据文件SHA见同名JSON。

异常没有保存不同像素数、最大误差或连续输出差，因此尚不能区分量化边界小差异和
实质输出偏移。环境/内核/精度/硬件差异或推理路径问题仍未隔离；不能直接归因H200，
不能据此判断energy方向、选择或泛化失败，也不能认为源码SHA一致就已排除接线问题。

## 下一步可证伪检查（仅建议，尚未实施或启动）

先在一个原24训练来源上对齐权重、prompt、seed、crop、reference time、schedule、
alpha和双uint8输入，比较原缓存与新运行输出，记录不同像素比例、最大/平均绝对误差。
保留运行时/后端指纹，并按编码、初始噪声、采样、解码、两次量化定位首个分歧。
若需要GPU诊断，先冻结单来源范围、执行命令与预算；当前未授权额外诊断运行。
停止条件是取得足以定位首次分歧的证据或触及明确预算，不接续energy训练。

不关闭相等检查、不替换为较弱基线、不删或盲目重建缓存、不自动重试完整资格。
所有现有缓存、日志、权重和旧输出保留；本次仅更新文档。
200步trainfit仍须通过匹配资格后独立手动启动，不能由失败尝试自动接续。

证据：`logs/foundir_flux_tail_relative_energy_v3_qualification_3x2_v1_node{0,1,2}.log`；
`checkpoints/foundir_flux_tail_relative_energy_v3_cluster/qualification_3x2_v1/` 的
preflight、execution和attempt connectivity；同根cache_v1/base/complete.json。
原始运行资产不上传GitHub；报告JSON保存其SHA以供复核。
