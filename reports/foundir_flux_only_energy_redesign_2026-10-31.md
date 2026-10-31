# FLUX-only energy：重做方案与当前交接

**最新资源规则：所有非轻量GPU工作默认3节点×每节点2张H100/H200，总6卡。
当前实施见 `foundir_flux_native_energy_3x2_readiness_2026-10-31.md`；覆盖本文单机4/8卡
默认。全局batch8及科学方案不变；真实三节点CUDA尚未验收，不直接恢复旧拓扑断点。**

**实施更新（同日）：** 用户已要求启动。新25步训练器/候选缓存/4或8卡tmux入口
已实现，9项CPU测试通过；真实4B GPU尚未启动/验收。见
`foundir_flux_native_energy_v1_readiness_2026-10-31.md`，下文“尚未实现”是方案冻结
时的历史状态。新global batch为8，三步和512图不变；无自动200步。

日期：2026-10-31。状态：用户要求重做；本文件冻结架构边界并提出具体实验方案。
**不是已实现/已通过的训练代码，不是 GPU 启动命令。**

## 1. 用户不可被旧文件覆盖的决定

- 恢复和 energy 的可训练图像主干都必须使用预训练 FLUX.2 Klein base 4B。
- 禁止外挂 CNN energy，包括 SpatialConditionalEnergy、TailQualityEnergy；禁止
  Restormer、residual Restormer、像素 refiner、CNN scorer 或用它们蒸馏本次模型。
- 不允许把旧 CNN 包装成“FLUX energy”，也不能只把类名改为 Transformer。
- 禁令适用于后续所有新恢复/energy实验，除非用户明确撤销；历史存档不是重启许可。
- 允许 FLUX 自身的 LoRA、线性读出头、VLM 条件线性投影；不增加独立恢复主干。
- FLUX 原生冻结 VAE 是编解码器；冻结 LPIPS 是用户要求的损失/指标。二者内部有
  卷积，但不是额外可训练 energy/restorer。因此准确名称是“FLUX-only 可训练主干”，
  不是“整个计算图零卷积”。不得隐瞒这个边界，也不得擅改原生 VAE 或 LPIPS 定义。
- 旧代码/结果保留为历史，禁止继续训练、部署或把旧 CNN checkpoint 作为初始化。
  停用旧路线不等于删除成果；无需重新运行历史基线差异诊断或 GT 可达性搜索。

最新用户消息优先于任何历史 ACTIVE/CURRENT。旧 CNN 的 200 步已经结束，不能再
按旧 AGENTS.md 的“下一步200步”启动一次。MoE/Rain/EBT/Restormer 仍非当前任务。

## 2. 第一原则：已有证据支持什么，不支持什么

任务不是让模型“觉得图片好看”，而是从输入图像可观察的信息中，判断一个修改是否
同时减少恢复误差、改善感知质量，并让判断的梯度实际给出有用的修正。

旧 CNN 在 24 个 Noise 来源训练、其中 8 个来源检查，bounded 单步 PSNR 增加
1.060337543 dB，LPIPS 相对下降 2.485951%，3/8 图 LPIPS 受损；目标是 +.30 dB
和 LPIPS 相对下降 9.928107050377874%。PSNR 已达标，LPIPS 不是“只差一点”。
这说明实际 FLUX 尾段具有可利用的修正空间，不证明当前能泛化，也不证明“CNN
几乎成功，所以更大的 FLUX 必然成功”。GT 多轮曾成功，但不是同预算单步证据。

采用 FLUX 的理由：复用已学到的图像/生成表示，避免再从零训练小 CNN 学质量。
同时训练实际修后图像，避免“评分拟合好但图修坏”。这两个改变都有问题依据，
仍是假设；FLUX 的 flow 表示不是天然校准的质量 energy，容量大也不会增加 GT
才知道的隐藏信息。最大风险仍是 LPIPS 的有效方向、源外泛化和二阶反传成本。

## 3. 确切架构：同一 FLUX 底座，两种明确的前向模式

### 3.1 固定生成模式 G

保留已验证 Noise FLUX：预训练 4B 底座 + 原 Noise LoRA，全部冻结。
保持当前部署定义：同一权重/种子/512 裁图/50 步/alpha=.30/双 uint8。
即 raw FLUX 先取整，再与输入按 .30/.70 混合，再取整。这不是 Restormer。
生成模式关闭所有新增 energy adapter/条件 token/读出，不改变原 FLUX 输出。
所有改进均相对同环境的这个固定基线；另保留历史基线对照。

### 3.2 FLUX energy 模式 E_theta

复用同一预训练 FLUX 和冻结 Noise LoRA；增加独立 energy LoRA（初始建议 rank16，
目标为现有50组 attention/linear 注入点）、VLM条件线性投影、线性标量读出。
训练更新这三类参数，不从零训练4B，不更新原Noise LoRA，也不只训练一个裸评分头。
LoRA残差零初始化以保留预训练起点；energy线性读出用小非零初始化，避免整个
energy恒为零而没有latent修正方向。必须验明LoRA和输入两侧确实有有限非零梯度。

输入为当前候选 latent z、实际 timestep t、degraded image reference、固定 proposal
reference，以及已有逐图 VLM 五个退化等级。后两张 reference 用原生冻结 VAE 编码；
只在 energy 模式增加 proposal reference，原生成模式不变。五等级经线性投影成为
条件 token；这是“VLM结构化诊断的投影”，不是谎称已经接入原始VLM hidden tokens。
不使用训练类别/GT退化标签替代 VLM 判断；当前小集合等级缺乏变化须如实报告。

FLUX Transformer 处理这些输入，在 final_layer 之前读取候选图像 tokens，排除
文字和 reference tokens。共享线性头逐 token 输出局部分数，最后平均成一个
标量 E_theta(z,t|x,p,c)。不接 Conv，不另加图像质量网络，不用 flow velocity
的模长冒充 energy。训练/推理使用同一 scalar energy。

实施依据：本地 third_party/flux2/src/flux2/model.py 的 Klein4BParams 为 hidden
3072、5个double blocks、20个single blocks；现有 forward 最后在 final_layer 前
得到图像 tokens。现有 LoRALinear 只有一套 adapter，**双模式尚未实现**，必须新增
显式、无原地参数切换的模式接口。禁止在活跃 autograd graph 中换权重/合并LoRA。
不承诺零改动地“插个head就好”，也不新增第二份独立可训练图像主干。

## 4. 实际怎么修图、怎么训练

### 4.1 保留已验证的修正位置，先做 1/2/3 次真正修正

从同一 FLUX 第48步保存状态 z^(0) 开始，固定剩余两步生成映射 G_tail：

    e_k = E_theta(z^(k), t48 | x, p, c)
    z^(k+1) = z^(k) - eta_k * normalized_gradient_z(e_k)
    y^(k+1) = deployed_output(G_tail(z^(k+1)))

每次均重新计算 energy；三个 k 是对同一尾段状态的优化迭代，不是把 diffusion
timestep 假称为三次修正。报告0/1/2/3次真实图像指标、耗时和FLUX调用/反传数。
不再用5个方向乘5个幅度的25候选搜索掩盖成本。eta/归一化/半径必须在训练侧冻结，
不在检查图上逐图看GT选幅度。它们不是保证不伤图的证明。

### 4.2 直接监督修后图像，而不是只拟合分数/导数缓存

核心训练目标（具体系数由训练侧初始尺度确定后冻结）：

    L = L_action_rank + sum_(k=1..3) [a*MSE(y_k,GT) + b*LPIPS(y_k,GT)
                                     + d*soft_harm_penalty(y_k, y_0)]

MSE 对齐用户 PSNR 目标；这相对原PPT的L1项是明确记录的改动，不再暗换。
LPIPS保留原指标网络/口径。soft_harm_penalty 分别惩罚相对基线的MSE和LPIPS变差；
只是训练损失，不能把软惩罚称为推理阶段的绝对no-harm保证。
三步完整展开，梯度通过 energy 的 latent 导数、冻结生成尾段及冻结 VAE 回到
energy LoRA/条件投影/读出；不能 detach 修正方向后宣称端到端学习了修图。
冻结权重不等于切断输入梯度。
现有FLUX使用scaled_dot_product_attention；Flash/fused内核不能未经测试就用于
二阶反传。先用支持高阶导数的实现做定向grad-grad核对，并检查checkpoint重计算
兼容性；再测正式512/三步图。不能用toy或一阶反传成功代替正式图验收。

取整不可微：训练采用原部署混合对应的连续图像损失，实际检查严格执行双uint8；
同时报告两者差异，不把round当可微或悄悄更改评测。现有混合/量化口径不重写。

### 4.3 完整动作对比，明确终态与中间状态的区别

同输入训练候选包括：原FLUX输出、真实错误FLUX专家输出、未恢复输入、欠恢复、
过恢复/重复执行FLUX、细节损坏反例，以及训练GT监督的较优FLUX候选。
全都不能来自Restormer/CNN。GT图/GT引导只作训练监督，不能成为部署可选输出。
欠/过恢复操作记录真实构造方式，不能把沿一个残差的七个幅度称为完整动作集合。

错误专家等最终RGB候选：用冻结VAE编码后，以 t=0 输入energy作终态排序。
实际尾段状态候选：使用真实 t48 和经固定G_tail生成的质量标签作排序/修图训练。
**不得把最终RGB的VAE latent冒充 t48 中间状态**；两种样本的时间/表示必须显式标注。
排序优先使用两个指标都明确更好的成对样本；指标互相冲突或近似平局的候选不强塞
错误的“绝对好坏”标签。动作类别不是GT质量排序的替代。

旧CNN权重、CNN梯度/反例刷新轨迹均不作新模型输入或初始化。只复用已核验且
确实来自冻结FLUX/训练GT的原始状态和候选；逐项记录身份。无需为换模型重跑全部
GT可达性实验，但新架构的候选构造和状态对应必须真实。

### 4.4 推理和自验证不读取GT

最多产生三次修正；只用同一energy、固定规则在原状态与修正状态中返回结果，
无可信下降时返回原图。最终报告固定3步终点与energy选择结果两组，分离修正能力
和选择能力。由这个实验检验self-verification是否可靠，而不是先宣称可靠。
基线score、每步score、真指标用于离线分析；真指标不得进入推理选择函数。

## 5. 重做的次序与规模

1. **实现并做定向工程检查。** 实际模块/优化器白名单必须显示：只有FLUX energy
   LoRA、线性投影和线性头训练；没有SpatialConditionalEnergy/TailQualityEnergy/
   Restormer/CNN refiner实例或checkpoint。核对energy-off的原输出、全三步梯度、
   参数更新、峰值显存、每步耗时、checkpoint/resume。检查保护的是新架构，不重开
   历史环境差1灰度级诊断。不自动执行任何GPU命令。
2. GPU工程资格建议25步，必须走正式三步图和512裁图，不能用低分辨率/单步冒充。
   支持单机4/8张H100或H200；实际节点和卡数按用户当前资源冻结。现有3x2已结束的
   拓扑不能默认变为本轮要求。完整三步二阶图是否装得下80GB尚未知；若不行，报告
   具体开销/错误，不能悄悄改回CNN、冻结全部LoRA、截断反传或降低评测协议。
3. 独立fresh的200步训练拟合检查：先用原24个训练来源保持可比性，固定原8图为
   training-fit probe，按25/50/100/200评估0/1/2/3步。资格权重不初始化此轮。
   这是能否学会的检查，不是最终数据规模，不因24来源拟合成功宣称泛化。
4. 若训练拟合有稳定双指标收益，再冻结更大规模合规Noise训练池和来源隔离开发集，
   训练并评估；规模按实际manifest身份/来源数报告，不把曝光次数说成新图片。
   正式数据不局限24张；但现在不虚构可用来源数、不直接承诺10k/100k步。
5. 未见来源通过后再扩Haze及其余类型/组合。当前不并行恢复MoE、Rain或其他旧分支。

训练拟合最好超过最终门槛并且逐图稳定，才有合理理由期待泛化；没有通用倍数保证。
不能凭平均PSNR过关就跨过LPIPS关卡。当前尚无新FLUX energy时长/显存测量，
不能沿用CNN的速度估算；资格测出后报告普通小时及实际/分配GPU-hours再决定长训。
每个完整step原子保存latest（模型/Adam/RNG/采样/拓扑），保留声明的里程碑；
恢复必须真正续训练状态，不能只载LoRA把重启说成resume。

## 6. 对照与成功标准

- 匹配原FLUX固定基线（energy off），不使用residual Restormer抬高起点。
- FLUX energy固定1/2/3步及energy自选结果：均无GT推理。
- 同预算额外FLUX采样、但无energy引导：选择规则先冻结，不允许GT择优。
  按实测时间/计算匹配，不能只匹配“步数”而忽略energy前向和二阶/输入反传。
- 去VLM条件、去动作排序、去实际修后损失的匹配消融，待主方法有效后执行。
  初始24来源条件变化很小，不足以建立VLM贡献；不得直接宣称创新已被证明。

最终联合门槛不变：相对同环境同协议原FLUX，平均PSNR至少+.30dB，平均LPIPS
相对下降至少9.928107050377874%。分清均值和单图通过率，记录受损样本和来源
不确定性；成功必须落在未参与训练/选参的来源，禁止official/sealed/internal/
parent-test像素训练、调参、选checkpoint或早停。训练fit不当作论文结果。

## 7. 本次交接状态

本次只更新PPT、规则和设计交接；没有实现FLUX energy训练器、没有新GPU运行，
没有把旧CNN checkpoint更名。下一步是实现第3/4节并通过第5节定向工程检查，
不是继续旧run_foundir_flux_tail_relative_energy_v3.py。

执行根仍为 `/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`；
远端执行Python为该根 `.conda/gpu-train-stable/bin/python`。本会话旧根只读审计，
文档在隔离worktree提交；主代理不能看到新节点实时GPU。接收代理从GitHub同步
明确文件、回报结果，不自行设计/启动训练。保留所有无关脏文件和历史产物。

相关证据：
- `reports/foundir_flux_tail_v3_direct_trainfit_terminal_2026-10-31.{md,json}`。
- `lowlight_restoration/flux_tail_learned_energy.py`（已弃用的CNN，不是新方案）。
- `lowlight_restoration/flux_tail_relative_energy_v3.py`（旧代理目标/单步路径）。
- 原PPT历史版本：Git `610d23ac9df682376743fe96b44fd753992a42c4` 下同名文件。
- 新PPT是本次用户要求的修订草案，不冒称原导师已确认这一具体架构。
