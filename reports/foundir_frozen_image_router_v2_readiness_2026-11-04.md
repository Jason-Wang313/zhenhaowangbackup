# 一轮有边界的路由修复：冻结图像路由＋扩大训练覆盖

## 结论与授权

用户在“不是保证最后一次；做一轮修复和一次独立来源验证”的说明后授权execute。
本轮执行expanded_frozen_image_v2，不恢复energy/CNN/Restormer、旧400+8质量训练或修改五个FLUX专家。
仍是单退化预测top-1进入原内部专家bank，不把本轮称作double/triple混合退化已实现。
目前代码/测试与真实清单预检已完成；主代理没有目的地挂载/GPU，真实缓存和训练尚未启动。

7fe87e5的CPU核对已结束：40秒/0GPU-hours，无缓存错接、裁图错位或漏曝光。
Blur错例有相似训练样本，仍被VLM高置信覆盖；Low-light两种特征都接近Noise，
与训练Low-light距离偏大。训练集已640/640，但开发仍错两张，不能只延长旧数据训练。
这支持尝试更稳的修正学习与更丰富来源，不证明它们必能消除错误。

## 只做这一套固定修复

- 直接加载fresh_ce400_matched_v1的image_only最终step400图像MLP，全部冻结。
  保留的是这个实际33/40对照，不是联合VLM模型内20/40的图像分支。
- VLM MLP用fresh初始化，最后一层全零；初始合成分数严格等于冻结image-only。
  原结构仍是image logits + VLM delta，不额外加CNN、RGB refiner、LoRA merge或备用专家。
- 只训练VLM MLP，1600步/global32，每个训练样本20次曝光。
  AdamW lr.001、weight decay .01、label smoothing .05、梯度范数裁剪1。
  不改学习率扫描、不加噪声特征增强、不扫VLM系数/阈值、不针对开发错例硬编码回退。
- 在400/800/1200/1600四个预定点评估校准集，按最低CE选择、平局取最早。
  用CE而非只看正确数，避免完全忽视高置信错误的代价；仍不是实际恢复损失。
- 选择结果先不可变落盘，再加载独立holdout特征，一次评价冻结image-only和修正版。
  不挑holdout上的checkpoint、步数或系数，不把错误回灌本轮训练。
- 新旧训练数据量不同：冻结image-only只见旧640，新修正见2560。
  本轮是修复验证，不能宣称这是数据量匹配的VLM因果消融。旧匹配结果仍单独保留。

## 数据规模与真实清单核对

| 用途 | 每类记录/来源 | 总记录 | 跨类别合计不同来源 |
| --- | ---: | ---: | ---: |
| fit | 512 | 2560 | 2335 |
| calibration | 80 | 400 | 400 |
| holdout | 80 | 400 | 400 |

最初考虑每类512+128+128，但真实合格Low-light仅723来源，少于768，不能复制图片凑数。
因此在读新像素/训练前固定为512+80+80；其它四类同样数量。没有根据模型得分挑数量。
原640训练记录原seed/crop保留，并复用原producer特征；额外1920训练+400校准+400holdout，
共2720份新特征。新receipt明确记录producer/consumer，不重命名旧cache冒充新生成。

只从既有manifest的train单退化池选择，按512裁图可用性、来源与哈希过滤。
先固定稀缺类别来源，保留全部原训练身份；任何source/GT SHA/LQ SHA不得跨角色。
排除manifest非train来源/GT/LQ身份、旧40开发图及其来源；不会读取官方/封存测试像素。
同一来源可在训练组出现不同退化类型；不是宣称2560张全来自2560个不同clean来源。
新holdout与本轮路由训练及原640都来源分离；FLUX/Qwen预训练可能见过，
不能包装成整个系统完全未见过的新测试集。标签来自元数据，未做视觉标签审核。

已在本地只读真实manifest完成选择核对，没有解码像素：
items SHA256为333117f97705c42acefedbbee698098935da143c21a1bb6f176cf409f3fc4ceb。
目的地prepare必须得到同一SHA，否则停止。小报告foundir_frozen_image_router_v2_selection_2026-11-04.json。

## 成功标准与停止边界

holdout分类筛选须同时满足：修正版总体正确数严格高于冻结image-only、每类正确数不下降、
总体CE不高于对照。记录完整逐图及每类，不通过也正常结束并如实报告，不能自动扩预算/换配方。
这是下一步画质验证的筛选条件，不是论文目标或PSNR/LPIPS门槛。
本轮不生成恢复图；即使分类通过，也不能宣称原SoW、Rain/Blur或energy双目标通过。
后续需要在同一冻结协议下验证实际恢复质量，另外安排，不由启动器自动执行。

## 两阶段运行，不让GPU等待CPU训练

唯一执行根：/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。
环境：该根下.conda/gpu-train-stable/bin/python；不在旧HDD3执行，不用系统Python代替。

1. CPU代理同步最新GitHub并保留无关修改/大资产，先运行：
   `.conda/gpu-train-stable/bin/python -S smoke/repair_foundir_frozen_image_router_v2.py prepare`
   此步核验来源/旧checkpoint/源码身份和LQ文件存在，不加载模型或读像素。
2. 预检通过后CPU代理给用户四节点各两卡的特征命令：
   `bash smoke/start_foundir_frozen_image_router_v2_tmux.sh features <node_rank_0到3>`。
   节点编号各用一次，四节点共享同一根。各节点torchrun仅本地两worker，
   全局分片rank=2*node+local，不训练、不做NCCL、不需要旧node0地址/端口。
   八个worker各负责340个新记录；Qwen视觉特征与FLUX图像特征沿用原已验证提取代码。
   只读LQ，不读GT像素；不加载五专家生成bank、无需反传或恢复图。
3. 全部8个worker marker与四节点status0齐全后，释放GPU，CPU代理手动启动：
   `bash smoke/start_foundir_frozen_image_router_v2_tmux.sh train`。
   这是同一已授权修复的CPU训练阶段，不需要用户再提供GPU。
   入口不会在缓存结束时自动串起训练，也不会在训练结束后自动运行恢复评价。

特征入口支持每节点两张同型号H100/H200，节点间可混型号/驱动，继承软件和数值策略。
H100特征提取显存尚未实测，不能拿旧H200结果宣称已通过；本轮不是138GiB五专家推理bank。
若H100 OOM，保留已完成缓存并回传，不悄悄改精度/crop或额外启用新offload策略。
每node记录实际型号、驱动和worker峰值显存；四节点数值环境须匹配原签名。

特征每节点累计硬cap1小时，四节点两卡最多8使用GPU-hours，另各30秒清理。
CPU训练累计硬cap1小时/0GPU-hours，另30秒清理。都不是实测ETA，不承诺精确耗时。
失败时间累计；硬断缺结束账时先核实，不清零预算。无自动重试/扩大预算。
每份特征独立保存与SHA校验；同入口跳过已验证记录，不覆盖旧缓存/孤立文件。
CPU每完整step保存VLM/Adam/RNG/采样位置，保留0..1600共1601份，预计约18–22GiB，
首启须30GiB空闲；图像头冻结并绑定原checkpoint，检查点无需每步重复保存它。
损坏/步骤缺口失败即停，不静默回退；校准结果选择与holdout读取顺序锁定。
相同节点编号恢复时运行栈/型号/驱动须匹配已记录runtime；换环境需明确迁移，不能直接覆盖旧记录。

最终需要FROZEN_IMAGE_V2_RESULT_VERIFIED和训练入口FROZEN_IMAGE_V2_END status=0。
GPU特征完成只表示缓存完成，不是训练完成。结果文件：
reports/foundir_frozen_image_router_v2_terminal_2026-11-04.json。
只回传小报告及四节点/CPU耗时、退出码、峰值和失败累计成本；不上传特征、权重、图片或live logs。

## 测试边界

65项CPU测试通过（13项本轮新测试＋52项既有回归），Python编译、shell语法和diff检查通过。
本地CPU测试覆盖来源/图片哈希隔离、旧训练复用、固定20次曝光、八分片、图像头冻结、
VLM更新、完整1600步小维度合成流程、先选模型后读holdout、结果独立汇总核验、
重复完成不重训，以及下一步Adam/RNG精确恢复。合成结果不是新科学指标。
目的地真实模型/数据文件核验和8卡H100/H200缓存吞吐/显存仍待执行。
