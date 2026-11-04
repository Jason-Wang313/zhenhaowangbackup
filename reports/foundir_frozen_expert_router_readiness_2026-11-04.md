# 五个已有 FLUX 专家冻结后的路由训练：首轮工程验收

## 结论与下一步

用户已授权继续冻结专家训练 router。五库接入已完成，不再重跑：116/116 个单选输出
与同环境原入口逐像素相同；Rain 原100开发图维持34.30118094 dB / .06186744。
固定90%主专家混合会损害Rain和Haze等任务，因此本轮学习比例，不固定给其它专家最低份额。
代码已实现；本地没有目的地GPU挂载，尚未启动本轮，也没有本轮GPU速度、显存或质量结果。

这一轮只回答：能否正确提取特征、学会基本选专家、通过完整生成路径更新路由、保存及恢复？
不是正式训练，不宣称8步足以判断方法成败，不自动扩大预算。训练范围640条，开发检查40条。
所有历史模型、缓存、结果及已绑定源码不改动。

## 实际做什么

1. **保留五个已训练专家。**共享FLUX底座、原生VAE、四类BF16 LoRA及Rain step800 FP32 LoRA全部冻结。
   不是vanilla替代，不更新共享权重/专家，不添加CNN、Restormer、energy或RGB修复网络。
2. **图像路由＋VLM分数修正。**冻结FLUX的原生参考图token经img_in，按token统计均值和标准差，得到6144维图像特征。
   已适配的Qwen2.5-VL-7B visual merger token按token取均值，得到3584维VLM特征。
   两个LayerNorm→Linear(256)→SiLU→Linear(5)小MLP分别输出图像分数和VLM修正。
   VLM最后一层从零开始；两组分数相加再softmax。不用类别标签替代VLM，也不重训Qwen。
   这是原有Qwen视觉特征，不是FLUX文本编码器Qwen3，也不是Qwen生成的文字标签。
3. **先400步轻量类别初始化。**只训练图像分支，训练类别是监督目标而非模型输入；VLM修正乘0且保持零。
   global batch32，AdamW lr=.001、weight_decay0。400步是小MLP训练，不是400步FLUX反传。
4. **再8步真实修图损失训练。**共同训练图像和VLM路由，lr=.0001，保留同一个Adam状态。
   global32=8rank×4次累积；每张图真的跑所选26/50次生成并反传完整路径，不截断前缀。
   损失为10×RGB MSE（[-1,1]）＋LPIPS（Blur系数.2，其它.05）＋.1×类别CE。
   这组损失权重是首轮工程配方，不声称已最优。记录图像损失对比例的梯度，避免只看到CE在学习。
5. **输出协议保持原样。**最大分数专家决定原prompt/alpha/量化及生成终点；Rain沿原50步时间表第26次端点输出，
   其它四类50次。前26次允许五类混合，后24次Rain权重归零、其它四类归一化。
   混合发生在内部LoRA增量，不是平均五张成图，也不平均A/B因子。

离散最大值选择不反传；类别CE教会主专家选择，图像loss优化其固定主分支内的比例。
量化前向严格保留原单/双uint8流程，反向使用直通近似（STE），不是声称取整有真实恒等导数。
这种非连续协议、类别CE与图像质量目标之间的折中仍有风险；无单图无损保证，不承诺超过原专家。
也不能只凭开启VLM后分数变化就证明VLM贡献，后续匹配消融仍需要。

## 数据边界与检查口径

- 固定manifest SHA `e8c018879029c0bcc2da8d074cc4999541158ca97d07f48aadbcdddcdccc3ca7`，seed2026110401。
- 每类128个训练来源各一条，共640条；跨类合并实为634个不同来源，不冒称640个独立来源。
- 每类8个validation来源，共40条、跨类37个不同来源；与训练按source_group/GT哈希分离。
  选图仅元数据，固定512裁图/seed，不按结果选图；不做随机新增增强，不读取internal_test像素。
- Qwen和FLUX特征仅用该张LQ的同一个裁图，GT不进入特征、路由、prompt或终点选择。
- 训练前后（全局400/408）评价同40张：真实router对比已知类别单选对照。
  已知类别单选只用于描述原专家能力，**不是可部署路由**。
- 新40图和原Rain100图不是同一集合，不能拿新Rain8图直接宣称原100图34.2/.08门槛通过。
  报每类均值、选对数量、专家比例、退步数；开发不是官方/独立盲测，FLUX/Qwen历史是否见过不另作承诺。
- 本轮不读官方像素。历史授权允许官方SoW600反复评价与调参，不必等整个MoE冻结；但官方图/GT不进训练、
  loss/梯度、逐图GT选路；其他封存集仍关闭。官方原分辨率/pyiqa协议需独立入口，不能拿512开发结果冒充。
  依据：`reports/advisor_governance/foundir_official_sow600_moe_clarification_2026-11-03.json`。

## 反传与显存的重要实现

原推理类InferenceTrajectory带可变调用计数，不能用于checkpoint反传。本入口显式传调用index、时间、比例。
使用两层重算：完整FLUX调用，以及原生double/single block；原生VAE也重算。
每个block重算显式重新绑定本次专家比例，防止functional_call退出后读到恢复的NaN占位buffer。
Hopper原生VAE query chunk256（仍完整keys/values）、激活CPU转存、cuDNN workspace256MiB沿用。
不是新增图像卷积模型。完整多步反传比上一轮纯推理贵得多，不能按上一轮6分38秒估算。

在质量训练前，8rank各检查1张训练图、覆盖全部5类：新可反传路径与已验收推理入口
decoded最大差≤1e-6、uint8一致，且真实图像反传梯度有限；失败就保存失败并停止。
冻结参数全程requires_grad=False，不在优化器里，逐步检查无梯度/版本修改。

## 保存、资源与启动分工

- 4节点×每节点2张H100/H200，总8卡；单节点同型号、每卡≥75GiB；跨节点软件/指标版本一致。
  上轮接入有8H100实测，本轮完整反传尚未CUDA验收，不能据CPU通过声称H100/H200已实测。
- 每完成一步原子保存router、Adam、8rank RNG、next_global_slot、history及哈希；含step0，共409份。
  约12GiB状态文件，首启至少20GiB空闲；孤立未提交tensor保留，不删旧数据。
- 同run-ID、新的共同launch-ID恢复；只支持相同8rank拓扑。未提交步可能重算，特征/已完成评价逐图复用。
  CPU恢复测试通过不等于CUDA故障恢复已实测。不把资格权重用于formal，不自动加步。
- 全程暂无法给可信实测ETA；按特征缓存、首张反传和首个完整质量step更新。
  预算硬cap每节点3小时，即最多24使用GPU-hours，加最多30秒进程清理；不是保证3小时内一定结束。
  400步写盘也计入总wall time，不用短暂MLP计算速度估算全程。
- 输出：`checkpoints/foundir_frozen_expert_router_v1/qualification_4x2_v1`。
- 日志：`logs/foundir_frozen_expert_router_v1_qualification_4x2_v1_node<rank>.log`。
- tmux入口：`smoke/start_foundir_frozen_expert_router_v1_tmux.sh`，四参数依次为node_rank、当前node0地址、run-ID、新共同launch-ID。
  rendezvous默认29703；node0..3必须使用同地址/port/run/launch。脚本显式CUDA_VISIBLE_DEVICES=0,1。
- 主代理在隔离代码工作树实现/测试/push。目的地CPU代理安全同步、核验运行资产/解释器，再给用户当前四节点命令。
  不把主代理临时worktree或旧HDD3目录作为GPU运行目录，不清空脏文件、不上传模型/缓存/日志。
- 目的地预检：在新根用`.conda/gpu-train-stable/bin/python -u smoke/train_foundir_frozen_expert_router_v1.py preflight`。
  首次检查当前共享底座/VAE/Qwen身份，读数据清单但不读图；不是重扫所有历史checkpoint。
- 终态要求`FROZEN_ROUTER_RESULT_VERIFIED`和四节点`FROZEN_ROUTER_NODE_END status=0`。
  成功表示工程完整，不表示科学门槛通过。终态仅核对最终状态、8条路径验收、40图输出及汇总；不全扫历史409个tensor。

## 本地验证记录

单元测试覆盖零初始化/类别输入边界、原量化、全50/26轨迹梯度、显式gate嵌套重算、来源分离、
每步Adam/RNG恢复及损坏拒绝。另用真正上游FLUX double/single block的缩小模型检查原生采样一致性和反传，
不是仅用伪造线性网络代替所有测试。真实4B/CUDA精度、显存和吞吐仍由上述GPU验收回答。
共41项单元/原集成回归通过（其中15项为新增）。两进程CPU/Gloo测试实际global32累积、与串行结果近似一致，
重建模型/Adam/DDP reducer后下一步参数及Adam逐元素相同；这不是CUDA重启实测。
该CPU测试用重复的两rank RNG构造八项存档夹具，不冒称执行了8进程CUDA恢复。
Shell语法/Python编译、原五库回归及元数据真实选择同时核对；最终通过项数见同名JSON。

## 若失败，下一步怎么判断

- 前向parity失败：先修时间/CFG/精度/量化，不训练掩盖接入问题。
- 显存或速度不够：只调整重算/转存实现，保持生成、数据、loss、global32和预算；超过预算不自动扩大。
- CE能降但修图梯度缺失/不有限：修反传，不宣称已优化图像质量。
- 路由选错：看图像分类/新数据域；不要用开发GT硬选主专家美化结果。
- 选对仍因soft变差：和已知类别one-hot比较，查比例/兼容规则，不能归罪于专家权重被重训（本轮冻结）。
- 8质量步变化小：只是工程小试，结合吞吐和完整记录设计下一次单独授权的训练规模，不包装成路线终判。

下一步不是恢复Rain蒸馏、旧FFN-MoE或energy。当前目标是可靠使用已训练五专家，然后才谈混合退化和VLM消融。
