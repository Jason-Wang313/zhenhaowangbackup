# PPT energy → FLUX 尾段引导：设计演变、证据与排错决策记录

## 当前：右图token residual MoE四节点25步资格结束 — 2026-10-31

qualification_4x2_v1已于四节点8×H200正常结束，25/25步、四节点launcher status0、
step0/25各Rain100与Blur preview100、五份恢复收据齐全。工程通过，正式训练未启动。
Rain step25相对step0 PSNR -0.055246dB、LPIPS +3.773%；Blur preview PSNR +0.007440dB、
LPIPS +0.509%。预览指标无联合改善，不可宣称VLM贡献或完整Blur门槛通过。
完整日志与限制见reports/foundir_token_residual_moe_v1_qualification_4x2_terminal_2026-10-31.{md,json}。
资格权重不能seed正式训练；未来正式规模需另定，不自动长训。energy显存线保持原状。

## 当前：v6快照采集成功；计算仍OOM，未开始训练 — 2026-10-31

memory_alloc_trace_v6单H200运行54秒后status1，首个高阶更新OOM，峰值137.6805GiB。
一阶分数/梯度差均0；训练/optimizer更新0。快照8,521,833字节及26项源码SHA已核验。
异常后存活133.9051GiB，其中125.3557GiB分配栈含局部VJP与外层梯度checkpoint重计算；
这是实际分配证据，具体保留根因尚未隔离。历史30000事件已截断，捕获量不能当峰值。
完整日志/result/CPU摘要：reports/foundir_flux_native_energy_memory_v6_terminal_2026-10-31.{md,json}。
原pickle只留共享目录；所有旧资产保留。本轮无计算改动、无新GPU命令或自动重试。
下一工程排查应在CPU隔离嵌套checkpoint/VJP保留关系，不把采集成功当资格通过。

## 当前：v5仍OOM；下一步仅采集真实分配栈 — 2026-10-31

memory_isolated_vjp_v5的一阶数值差均0，首个高阶更新仍OOM（137.68GiB），
42秒status1；完整记录memory_v5_terminal报告。不能称OOM已解决或开始训练。
当前运算模块保持8e78eb9不变，只给单卡诊断加可选allocator历史/快照采集。
新ID=`memory_alloc_trace_v6`，FLUX_NATIVE_MEMORY_TRACE=1，CUDA_VISIBLE_DEVICES=0。
预计复现同一失败，目的是保存实际占用栈，不是新的候选修复/训练；没有optimizer更新。
cap仍1200秒加30秒清理，不自动执行。快照仅留共享目录，不上传GitHub。
采集成功看MEMORY_SNAPSHOT_SAVED及文件SHA；status1仍是原计算失败，不能包装通过。
读reports/foundir_flux_native_energy_allocation_trace_readiness_2026-10-31.md。

## 当前：一阶数值已匹配，隔离局部VJP输入边界待GPU — 2026-10-31

memory_native_vjp_v4真实4B分数与一阶梯度差均为0，但首个高阶更新仍OOM，
峰值136.81GiB，50秒后status1。完整记录memory_v4_terminal报告；无训练步。
CPU已复现内层求导沿共享Q/K/V祖先额外回溯。现为局部VJP各输入加可微clone，
不detach活跃梯度，不改原attention运算/配方；外层高阶梯度连接保留，26项CPU测试通过。
新短诊断：`bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_isolated_vjp_v5`，
单个空闲H200，CUDA_VISIBLE_DEVICES=0，原1%数值门槛、完整三步与1200秒cap不变。
无optimizer更新、无自动资格/长训，旧缓存/失败/step0全保留。真实CUDA容量仍待测。
详见reports/foundir_flux_native_energy_isolated_vjp_readiness_2026-10-31.md。

## 当前：保留原attention与autograd，仅逐attention重计算导数 — 2026-10-31

memory_heads_v3的真实4B梯度差8.1279%，仍未过1%，39秒后停止；完整报告315a09e。
下一候选撤去query/head分块与手写VJP，原native attention callable、拼接/矩阵形状、
math SDPA和原生autograd保留，只用自定义autograd边界与checkpoint重算各attention的VJP。
不更改TF32/matmul设置，不放宽数值门槛，不截断三步，不改配方。
25项CPU回归通过；真实4B/BF16数值与最终反传显存仍待GPU。
唯一下一短任务：一个空闲H200、CUDA_VISIBLE_DEVICES=0，
`bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_native_vjp_v4`。
仍先同状态1%梯度对照，再完整三步/真实尾段/全部loss反传，无Adam更新。
cap1200秒加30秒清理，无自动25步/长训。旧失败记录与资产全保留，不resume旧源码。
读reports/foundir_flux_native_energy_native_vjp_readiness_2026-10-31.md。

## 当前：query分块数值不合格；head分块待同门槛单卡检查 — 2026-10-31

memory_tiled_v2单H200运行38秒后status1，实际4B一阶latent梯度relative-L2差11.4771%，
超过1%门槛，故未做三步和完整反传。完整结果aa02a0c；不得放宽阈值或称显存已修复。
现改为每次一个独立attention head，保留每head完整Q/K矩阵和归约范围，
不用query分块的跨块K/V梯度累加。仍是完整softmax，保留全部三步和原配方。
25项CPU检查通过，含24heads/128head_dim的BF16检查；不能代替真实4B CUDA对照。
新单卡短诊断ID=`memory_heads_v3`，命令为
`bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_heads_v3`。
一个空闲H200、CUDA_VISIBLE_DEVICES=0，原1%梯度门槛和1200秒cap不变，无Adam更新。
旧数据/缓存/失败记录保留；不自动重启GPU或25步，不resume旧源码step0。
详见reports/foundir_flux_native_energy_head_attention_readiness_2026-10-31.md。

## 当前：最终反传仍OOM；energy attention分块版本待单卡验证 — 2026-10-31

memory_recompute_v1于18:14:37 UTC以status1退出，110秒/0.030556 GPU-hours；
三步更新/解码/loss已完成，但最终backward仍升至135.30GiB并OOM。记录ddaa5f0。
继续用户已授权的内存修复：energy full-softmax attention按128个query分块，
每块仍关注所有keys；导数逐块重算，生成路径继续原实现。没有稀疏化/截断三步。
24项CPU测试通过，包括完整三步旧新梯度/Adam、gradcheck/gradgradcheck及BF16检查。
数学等价不代表BF16逐位一致；实际4B输入梯度对照与CUDA容量仍待实测。
下一命令：`bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_tiled_v2`，
一个空闲H200、CUDA_VISIBLE_DEVICES=0，独立新ID；cap1200秒加30秒清理。
先同状态对照旧/新attention分数和一阶梯度，再完整三步/全loss反传；无optimizer更新。
不自动重启25步、不resume旧源码checkpoint。全部旧资产保留，非轻量训练默认仍3×2。
读reports/foundir_flux_native_energy_tiled_attention_readiness_2026-10-31.md。

## 当前：FLUX-native首步OOM；导数图重计算待GPU短诊断 — 2026-10-31

用户明确授权执行内存优化。旧3×2资格24候选缓存完成，首步OOM，三节点status1；
只有step0，0/25训练步，不能按下方“等待启动”重跑。完整日志已回传d039d3f。
新实现对energy及其create_graph输入梯度整体checkpoint，完整三步及配方不变。
19项CPU回归/数值对比通过；真实4B/BF16显存尚未实测，不能宣称OOM已解决。
当前只准备一个GPU的单样本全反传显存诊断，明确属于短容量测量例外：
`bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_recompute_v1`。
CUDA_VISIBLE_DEVICES=0，优先空闲H200，cap1200秒加30秒清理。无optimizer更新、
无训练步、无自动25/200步；原global batch8没有改为1。旧缓存/权重/日志保留。
旧qualification源码身份已变，不能直接resume旧step0。非轻量训练仍默认3×2。
详见reports/foundir_flux_native_energy_memory_readiness_2026-10-31.md。
目的地代理负责实现/CPU核验/同步本次获授权优化；用户执行GPU诊断。

## 2026-10-31 覆盖：停止CNN路线，FLUX-only重做

读 `foundir_flux_only_energy_redesign_2026-10-31.md`。用户明确禁用外挂CNN energy、
Restormer/refiner/其蒸馏；不是只改命名。旧CNN 200步已完成：8个训练来源
PSNR+1.06034dB，LPIPS降2.48595%（目标9.92811%），3/8图LPIPS受损。
这支持继续检验FLUX尾段修正空间，不证明FLUX energy必然容易成功。
旧CNN SpatialConditionalEnergy于10月15日引入，10月28日被FLUX-tail复用，
10月29日V3沿用。原PPT允许冻结proposal但未指定CNN；此前架构/目标变化未讲清。
新方案必须用FLUX内部tokens和energy LoRA，并直接监督真实修正后的图像；
不恢复旧CNN或用旧权重初始化。原生冻结VAE与固定LPIPS不属于新增恢复器。
本次仅设计/PPT/交接，尚无新训练器或GPU结果。下方旧下一步/状态都是历史。

更新：2026-10-30。面向接手代理，不依赖旧聊天窗口。
这份文件保存可复核的事实、设计理由、失败经验与未解问题，不把假设补写成定论。

### 最新证据：V3八步工程通过，用户要求继续200步

Git db03d94终态报告已回传，主代理从8条逐图值重算均值并核对59项Git内源码身份。
本轮问题是工程链路能否跑通；三节点6H200正常结束，8步及8个训练probe全部完成。
bounded相对当前原FLUX：PSNR +.608392dB，LPIPS降2.596406%，未过联合目标。
逐图各有3/8张PSNR/LPIPS退步；LPIPS总改善依赖index0，去掉该图后另外7图平均
LPIPS反而增加.00049921。不要把平均PSNR或全部接受候选误写成可靠自验证。
step8训练日志LPIPS方向cosine约.000433、MSE约.01976；这是已抽取训练对的
图像梯度诊断，不是新来源指标，也不能代替实际尾段probe；8步不足以终判可学习性。

下一可证伪问题：保持同一配方、fresh训练200步后，训练侧质量判断/导数及真实修正
是否同时改善，尤其LPIPS和逐图稳定性。不改loss，不重复已完成诊断，不加额外分支。
新任务及预算见 `foundir_flux_tail_v3_direct_trainfit_readiness_2026-10-30.md`。
当前尚未启动200步，主代理只能准备/Git交付，新节点CPU代理仍只同步和回报。

纠正旧“三关都未过”：GT多轮budget方案已在额外24来源平均达双目标（+.986179dB、
LPIPS降31.562025%，22/24单图通过）。它证明已测Noise来源存在改善空间，
不是当前单步/相同计算预算能达到同样收益，也不是无需GT的泛化。
在该24来源确认中，GT第一轮LPIPS仅降4.284375%，第四轮14.520211%，
故LPIPS不足不能直接归为没有空间或加训练必然解决；先完成已授权单步学习检查，
未来是否需要多步必须由学习后证据决定，不能本轮暗中改probe_updates。
通往首次未见来源结果最顺利为200步拟合+冻结模型评估两轮；若需要扩大训练/独立
校准则规划3–4轮。失败时轮数无法保证；Noise成功也不等于五类/混合退化或论文完成。

### 最新用户决定：不继续追历史bitwise，直接推进；本次只交付同步

用户明确取消额外基线迁移诊断前置要求。当前V3 wrapper不再将历史proposal送入
旧runtime的严格expected_proposal检查，而记录差异；旧runtime及量化函数本身不改。
新run的原始FLUX作为实际proposal，按run保存；训练条件与成对proposal监督一致。
理由：`quality_targets(candidate, GT)`的分数/导数不依赖proposal，所以旧候选仍可
作为训练样本；但作为参考的proposal条件与proposal候选必须成套换成当前run输出。
因此只替换这一参考及其标签，不重跑历史GT/FLUX候选生成，也不换损失/模型。
当前原始入口与tail零引导检查保留；probe同时对历史/current/FP32控制报告指标。
这允许开始验证学习，不等于证明迁移对画质无影响，更不是energy已经有效。
用户随后要求本轮只push并交付CPU代理同步提示词：不得自动GPU启动、不得提供
GPU运行命令。详见 `foundir_flux_tail_v3_direct_sync_2026-10-30.md`。

### 最新工程进展：训练前的历史基线不一致

V3三节点首次资格在initial mining的历史baseline检查退出，0训练步。
随后单来源双H200各两次的160阶段trace完全一致；相对历史图3630通道元素差1级。
这是训练前复现问题，不是V3 energy学习失败。原缓存和所有失败产物保留。
主代理CPU穷举65536种灰度混合发现1422种FP32/精确算术差1级，均在半整数边界；
这是量化敏感性证据，不是已锁定实际差异根因，也不授权放宽baseline检查。
新张量未挂载到主代理：交付固定只读CPU重放，由新根代理读取现成张量并回报，
不再重复GPU生成。主代理决策／远端代理只同步和回报的边界、命令和证据见
`foundir_flux_tail_cpu_replay_and_observer_handoff_2026-10-30.md`。
下一可证伪问题：现成raw与第二次取整能否解释实际旧新差异？若不能，保留上游
生成差异这一可能性；若能，也要区分算术匹配与历史因果证明，不偷换弱baseline。

后续真实CPU报告0b93b95已回答所检查的量化问题：两次取整及完整CPU重放与新GPU
完全一致，历史差异仍3630元素；FP64/精确算术扩大历史差异，不能用来修复。
输入归一化的约1.19e-7差异是否影响上游生成仍未知，没有锁定硬件或具体内核。
不重复相同诊断。下一建议是明确的新环境匹配对照与缓存迁移，而不是取消所有
检查或换弱基线；尚未批准/实现，没有新GPU启动。energy仍0训练步。

**先读本文件理解研究，再按最新 AGENTS.md / current handoff 执行。**
本文件不是新实验授权，不要求中断正在运行的任务，也不覆盖新的硬件/拓扑决定。
写作时已读取远端 `7532687`：新代理已完成迁移 CPU 验收，并按用户新决定准备
3 节点×2 GPU 的 V3。不要拿旧单节点八卡交接覆盖这些更新。
最新实际运行状态仍需查当前日志；此处不声称新 GPU 任务已经开始或完成。

## 1. 三分钟读懂我们究竟在解决什么

初始 PPT 的问题不是“再加一个网络把指标抬高”，而是：

> 已有恢复模型产出图片后，能否学会判断这次恢复是否正确，并用这个判断给出
> 有用的下一次修正？同一判断能否拒绝有害修改，而不是越修越坏？

最初设计是：原图 + VLM 退化描述 + 当前候选 → 条件 energy；
训练它比较同一输入下的正确恢复、残余退化、错误专家、过度恢复、细节破坏；
同时让 energy 对候选的梯度能修图。测试时没有 GT。

当前仍保留这个问题，但**修正位置变了**：

```text
原来的恢复任务：冻结 Noise FLUX 得到匹配的原始输出
                         ↓
从其 index48 latent 经实际剩余两步生成候选，energy 评价该候选
                         ↓
把 energy 的图像梯度经过 VAE / 冻结 FLUX 尾段传回 latent
                         ↓
修改 latent，重新运行尾段，再由预测分数决定是否接受
                         ↓
选择结束后，GT 仅用于报告实际 PSNR / LPIPS，不参与部署选择
```

FLUX 的权重不更新；训练的是约 101 万参数的 quality energy，不是从零训练大 EBT，
不是另一个直接预测 RGB 的 Restormer。VLM 当前是**每图五个退化等级**，不是在线
原始 token，也不是 MoE router。时间条件使用真实 shifted timestep，不用 step/50 假代。

我们已经证明：**用 GT 知道正确方向时，冻结 FLUX 的末段有足够共同改善空间。**
我们尚未证明：**不看 GT 的 energy 能学会这些方向、可靠选择、泛化，并以合理成本获益。**

因此当前最关键问题是“学到可用的质量梯度”，不是再证明 FLUX 完全没有空间，
也不是把自验证、首创或最终双指标通过写进结论。

## 2. 不同名字容易混淆，先分清

| 名字 | 实际是什么 | 不能混淆成什么 |
| --- | --- | --- |
| 旧 EBT / Compositional EBT / Generalist | 曾尝试从头训练恢复模型的历史分支 | 不是当前被冻结 FLUX 支撑的小 energy |
| Residual basis / Fidelity / Perceptual | 在已有输出上直接预测 RGB 修正的 Restormer 类辅助恢复网络 | 它的增益不是 energy 的增益 |
| 七个 alpha / complete-candidate scorer / pairwise gate | 比较已有候选、决定是否替换 | 不能自动代表“同一 energy 梯度实际修图”的完整 PPT |
| `flux_tail_budget_v3` | 使用 GT 的方向/步幅搜索诊断 | 不是当前可训练的 V3 |
| `flux_tail_energy_resumable_v2` | 旧 learned energy 的 25 步八卡资格，已完成且严重退步 | 不是需要恢复的训练 |
| `flux_tail_relative_energy_v3` | 当前相对质量 + 导数学习修复 | 不等于 GT oracle，也不是已证明有效的模型 |
| `..._v3_cluster` | 当前 V3 新拓扑输出根 | 工程迁移没有自动改变科学方法或证明效果 |

历史 EBT/Generalist 和 MoE 可以作为背景/对照经验，但用户已切换当前目标。
除非新决定明确要求，否则不恢复它们；不要因为名字含 EBT 就从头训练大模型。

## 3. 从原设计到现在：每次为什么改

各行样本集可能不同，只能看各自匹配基线的变化，不能把它们拼成同一进步曲线。
方括号证据编号对应文末文件；旧报告的“下一步”是当时决定，不是当前执行指令。

| 阶段 | 实际观察 | 当时改变什么、留下什么经验 |
| --- | --- | --- |
| 原 PPT 的候选排序 + 图像梯度修图 | 旧完整轨迹 V2 在 256 开发图 step5118：proposal 43.123698/.05036423，三步约43.124477/.05040073；几乎没修动且 LPIPS 略差 [A] | 不是“PPT 从未实现”。需要分别检查方向、幅度、排序，不能只看 loss 或工程完成 |
| 加强直接 residual basis | 另一匹配协议下 Fidelity 从43.124160/.05036207到43.397154/.02895504，即+.272993dB [B] | 证明辅助恢复网络有用；距旧绝对目标只差.027007dB，但这不是 energy 只需再涨.027007的永久目标 |
| 在强 basis 上预测局部幅度/候选 | 七 alpha 两轮0%替换，交叉熵1.95–1.98接近 ln7；完整候选 scorer 首轮过度替换，降低 Fidelity；独立校准无有效接受阈值 [C,D] | 把连续问题改成分类不自动增加信息。拒绝一切只是原样返回，不能算恢复收益 |
| 更直接的 pairwise gate | 完整终局82图相对同集 Fidelity：PSNR -.000792dB、LPIPS -.0001106 [A] | 已试过，不要再说“还没直接训练过候选优劣”。不能把子集绝对阈值通过当作相对增益 |
| 重试原完整 PPT 能量梯度 | 查出面积平均造成步幅缩放、局部排序标签未 detach 等数值隐患；修复后的 fixed32/300步训练拟合相对 Fidelity +.537637dB，LPIPS仅-.000519；选择比固定三步几乎无额外收益 [A,E] | 小集合能拟合，不证明独立来源改善，也不证明自验证学会。不能把“方向永远不可能”作为结论 |
| 用户明确采用 FLUX 基线、energy 独立增益 | 原图像输出 energy 25步：32训练图实际 uint8 指标完全无收益；连续修正很小 [F] | 微小更新可能被量化吞掉，但没有证明量化是唯一原因。之后用户决定直接转向 FLUX 生成末段 |
| 文献与可微尾段审查 | 区分预计成品梯度、近似梯度、真实剩余 rollout 梯度；接线、时间符号、量化和数值检查 [G,H] | 不能因为“diffusion 类似 EBM”就以为输出评分自动继承生成能力；必须真正穿过生成路径 |
| GT 诊断 V1 / V2 | V1约+.8dB但LPIPS不够；V2同轨迹+.869817dB、LPIPS降5.2709%，未达双目标 [H,I] | 两目标梯度强度和方向不同；已停住的轨迹增加最大步数不会自动继续改善 |
| GT 诊断 budget V3 + 另24来源确认 | 固定多方向+PSNR余量方案：旧八图+.700931dB/LPIPS降27.2731%；另24来源+.986179dB/降31.5620%，22/24单图通过 [J,K] | 有可达空间，但方向/选择/停止全部用GT；不是可部署、学习或泛化成功，且搜索成本不小 |
| learned-tail V2 25步 | 八检查图：原38.610993/.07752365，一次36.759309/.07801964，16次19.787051/.11268279；所有更新被接受且用最大幅度 [L] | 学到的评分与真实质量脱节。同一错误模型既给方向又批准自己，错误会被反复放大 |
| 当前 relative-energy V3 | fresh energy、相对质量差、梯度幅度/方向监督、自身反例刷新、一次实际修正及限幅诊断 [M] | 先回答训练侧能否学会；8步是工程资格，200步是后续独立拟合，不是自动上正式训练 |

**绝不能丢掉的归因纠正：**旧 residual 得到的漂亮 LPIPS 和 +.273dB 是真实项目成果，
但不是 PPT energy 已经成功。这不是贬低旧成果，而是避免把辅助模块的效果记错账。
同样，“完整 PPT 没试过”和“所有 energy 都彻底失败了”都不符合上述历史。

## 4. 三个不能混为一谈的问题

### A. 有没有更好的解？

GT 搜索在一些训练来源上已经给出“有”的证据。这个结论只限已测试来源、窗口和预算。
没在搜索里找到更好的解，也不能证明数学上不存在。

### B. 模型能否在不知道答案时找到它？

Oracle 知道真实干净图；energy 只看退化图、候选和 VLM 条件。
前者使用的信息更多。去噪本来就有不确定性，GT 的每次最佳决定未必都能从输入推断。
除此之外，还有训练分布覆盖、网络表示能力、优化目标、梯度学习、泛化等问题。
所以“GT 已经能过 → 学习只差一个 gate → 肯定能过”这条推理不成立。

当前 V3 甚至尚未完成训练侧可学习性检查，不能提前把失败归结为不可约信息缺失，
也不能简单说“数据少，直接100k步”。先定位它到底有没有学会已经给出的监督。

### C. 模型能否知道自己是否做对？

分数能排序，不代表局部导数正确；导数偶尔正确，不代表候选选择可靠；
对自己的分数做下降，更不等于真实 PSNR/LPIPS 改善。
目前独立校准/泛化没有完成。限制修改幅度是防失控措施，不是自验证成立的证据。

### D. 第一性原理：实际有用的不是名称，而是下降方向

暂时忽略取整，对真实质量代价 L、latent z 和小修改 Δz，局部变化约为
`ΔL ≈ ∇z L · Δz`。我们使用的修改来自预测energy，必须让这个内积在真实目标上
有利，而不只是让预测energy自己的内积有利。MSE和LPIPS是两个不同的真实目标。

即使候选RGB上的导数余弦变好，经过冻结生成器的Jacobian后，latent上的方向质量
也不一定按相同比例改善；有限步幅、混合精度和真实量化又会进一步改变实际结果。
这就是为什么同时需要“分数/图像导数诊断”和“真实FLUX尾段输出的成对指标”，
不能用任何一个训练loss或cosine代替实际probe。没有证据时先核对这条链路，
不要把“EBM”“TTS”“继承生成能力”当作会自动奏效的解释。

## 5. 当前 V3 每项设计对应哪个问题

科学配方见 [M]；实际执行文件以最新 origin/main 为准。

| V3 设计 | 要处理的问题 | 不能由此保证的事 |
| --- | --- | --- |
| 同图两个候选的相对 logMSE / logLPIPS 差分监督 | 图像之间的总体质量差可能掩盖“这次改动到底有没有帮助” | 降低回归 loss 不等于方向/选择已可靠 |
| 保留低权重绝对分数 | 当前接受策略仍需要质量量纲；只比较相对值不够 | 绝对分数不是外部质量认证 |
| 梯度幅度拟合 + 显式方向监督 | V2 导数弱且与真实方向不对齐；仅看排名不够 | 均值 cosine 增长不保证每张图收益 |
| 候选标签及真实 RGB 导数缓存 FP32 | 固定候选没必要每步重复昂贵目标反传；便于核验标签身份 | FP32/缓存不是新增图像信息，也不必然整体提速 |
| 自身生成的反例、训练中刷新 | 模型沿自己的错误方向可能走到普通训练候选没覆盖的位置 | 这是有依据的分布覆盖假设，不是已证明的唯一根因 |
| 旧V2权重只当初始反例生成器 | 已有坏轨迹可提供训练材料，避免把失败权重当良好初始化 | 不先假定每个自生成候选都是坏的；GT只在训练标注真实好坏 |
| 每个训练chunk检查各loss真实参数梯度范数 | loss数值小/权重大不等于对参数影响小/大 | 不根据一次范数就自动改权重 |
| 只做一次实际修改、同时报raw与bounded | V2反复自批最大步幅造成大范围压暗；先拆开单步方向和累积问题 | 不能用“被限幅拦住了”声称模型有效或多步TTS成立 |
| RGB RMS及通道均值偏移约束 | 针对已观察的全局颜色偏移，限制明显失控 | 阈值只由训练轨迹导出，未独立校准，不是实际no-harm保证 |

当前固定损失：
`relative 1 + absolute .1 + normalized-gradient 1 + direction 1`，
相对 SmoothL1 beta=.05；一半 pair 包含自身生成候选；遇零真实梯度不强行定义方向。
这些是待验证配方，不是经验上已经最优；不能在途中随意调配、还声称同一实验。

原始候选和新增反例只用原24训练来源。8张 probe 也来自这些训练来源，
旧8张检查图并没有转为训练。V3 不读独立校准池和最终1859条开发集。
200步足以回答什么，要看实际学习曲线；它不是证明总体方法可行或不可行的魔法数字。

## 6. “继承 FLUX 能力”在代码中具体是什么意思

当前 small energy 输出两种候选质量代价（logMSE、logLPIPS），不是 RGB。
输入包括 observation、proposal、candidate、两个图像差分、五级 VLM 条件与实际时间。
空间网络在聚合前保留多尺度信息，最后得到可比较的代价。

训练时：

- GT 只用于合法训练候选的质量标签和导数。
- 同候选的标签被冻结；不让模型通过改变“答案标签”来降低排序损失。
- 学 energy 的图像导数需要二阶参数反传，不能随手加 no_grad / detach 让其悄悄失效。
- 缓存训练不等于每个 optimizer step 都对50步 FLUX反传。

推理/训练侧 probe 时：

- 先走已冻结 FLUX 的前缀，取 index48。
- 真实走完剩余两步及 VAE 得到候选；energy 梯度通过这条路径传回 latent。
- 用候选方向/幅度搜索修改 latent，重新走尾段；FLUX 参数本身不训练。
- 连续可微代理用于求方向，真正量化图片用于最终报告；两者不能混报。
- 所谓预测 PSNR/LPIPS 是 energy 输出换算值，不是真实测量值。
- 选择及拒绝不读取 GT；决定输出后才计算真实指标。

主要代码入口（迁移版可能增加分布式包装，核心科学语义不变）：

- [energy与真实质量导数](../lowlight_restoration/flux_tail_learned_energy.py)：
  `TailQualityEnergy.score/forward`、`quality_targets`。
- [V3相对损失与一次受限修正](../lowlight_restoration/flux_tail_relative_energy_v3.py)：
  `relative_loss`、`parameter_gradient_audit`、`bounded_one_step`。
- [V3阶段调度](../smoke/run_foundir_flux_tail_relative_energy_v3.py)：
  `prepare`、`cache_base`、`mine`、`train`、`probe`、`verify`。
- [实际尾段运行](../smoke/flux_tail_energy_runtime_v1.py)及
  [可微尾段基础](../smoke/diagnose_foundir_flux_tail_oracle_v1.py)。
- [当前分布式约定](../smoke/flux_tail_distributed_v3.py)及
  [保存恢复](../smoke/flux_tail_relative_checkpoint_v3.py)。
- [科学配置](../configs/foundir_flux_tail_relative_energy_v3.json)。

## 7. 下一步结果不好时，按哪条路判断

**以下是定位顺序，不是授权自动跑这些额外实验。**
先读已有日志/JSON/源码，证据不足才提出最小的训练来源诊断及预算。
新增训练/阈值/数据/网络改动先冻结并获得对应决定；不悄悄读取开发集来试配方。

| 看到什么 | 首先核对什么 | 不要直接做什么 |
| --- | --- | --- |
| 训练还没开始、NCCL/路径/SHA报错 | 进程、环境、rank、共享文件、绑定身份、迁移记录 | 把基础设施故障解释为模型不行；删cache或关闭哈希检查 |
| 零引导结果对不上原FLUX | checkpoint、prompt、seed、crop、reference、50步schedule、alpha=.30、双uint8、精度对照 | 用较弱的新基线算增益，或先靠训energy补偿错误基线 |
| loss下降，但方向cosine近零且梯度幅度极弱 | 标签是否正确/固定、同图配对、梯度连通、各loss参数梯度、训练曝光 | 只看总loss说模型已学会；单纯放大步长 |
| 排序变好但真实一次修正仍变差 | 局部导数与真实目标方向、代理→实际rollout差异、步幅/量化、逐图trace | 认为排序好就等于可引导，或直接扩到16步 |
| raw继续压暗/选最大幅度，bounded全拒绝 | 生成候选实际质量与预测变化、全局RGB偏移、真实导数学习、自身反例覆盖 | 把“没再伤害”说成恢复成功，或放宽限制追求非零接受 |
| raw真实有益，但bounded挡住它 | 先核对逐图证据及训练导出界限是否过紧；后续需独立冻结的校准方案 | 在最终开发集上挑一个放宽阈值 |
| 有正确方向，但选择器接受坏候选 | 同图预测差分和实际差分、baseline评分、误接受/误拒绝分布 | 用GT筛回好图，再把结果称推理性能 |
| 训练probe有效，独立来源尚未知 | 标为training_fit_only，规划来源分离的下一阶段 | 声称泛化、SOTA或“PPT已成文” |
| 单步有效但多步恶化（未来阶段） | 逐步分布漂移、误差累积、候选排序/接受校准 | 认为同一energy会自动自纠错，或把轮数增加当免费收益 |
| 两个真实指标一好一坏 | 保留成对结果；查目标梯度/候选互补，不拼接不同方法最佳值 | 改门槛、只报有利指标，或把PSNR dB算百分比 |

8步资格只检查工程链路、保存、有限梯度、反例刷新和一次真实 probe：
没有明显收益不等于终判整条路线失败；有一次正收益也不等于科学成功。
若 CUDA/标签/梯度本身有错误，先修工程，不浪费200步。

后续200步若获准，fresh初始化并按预定刷新点执行。要看训练侧的方向、幅度、
相对质量和实际修图是否一致改善。若仍不能拟合，不优先归咎于“独立测试更难”，
也不跳到100k；先针对未学会的环节提出一个能区分原因的最小检查。
若学会训练来源而新来源失败，再讨论来源多样性、数据规模和泛化，而不是重复调步长。

没有预定、经验证的“cosine达到X就必过+.30”的公式；不要临时编一个通过线。
当前日志中的 `rank_correct/rank_valid` 是同图有效差分计数，须按有效总数解释，
不是样本成功率；cosine也不是正确像素百分比。
`parameter_gradient_audit` 是局部抽样诊断，不是整个训练集所有项的完整统计。
新增诊断如尚未实现，要明确说未实现，不假装现成结果存在。

## 8. 最重要的技术经验：不要重新踩这些坑

1. **先确定谁贡献了收益。** 固定基线、固定样本、固定crop/量化；辅助refiner、精度切换、
   增加计算、GT搜索与energy增益分别记账。旧43.397/43.424不是当前FLUX的绝对门槛。
2. **量纲和导数比loss表面系数重要。** GT V2中MSE/LPIPS梯度强度比中位约6.2，
   部分图方向相冲；把两个loss数值归一到1也没自动平衡。
3. **限制可能把可行路径截断。** GT实验发现，固定方向/每步双指标都不许退步会阻碍
   整体目标；多方向、使用超过PSNR目标的余量有帮助。但这是GT证据，
   不授权未校准energy在部署时任意牺牲PSNR。
4. **工程通过和科学通过是两件事。** status0、DDP一致、显存足够只证明可以运行。
   反过来，24来源/25步的负结果也不能作为所有EBM不可行的证明。
5. **不要把中间态适配说成全新或已经自动解决。** 文献已有预计成品引导、
   阶段条件和导数监督；当前实际FLUX分布、时间方向与量化仍需自己的验证。
6. **读过论文/官方代码不等于复现。** [G]保留12篇原文与代码审查范围及缺口。
   不再从摘要重新生成一个相同idea；但该审查也没有保证当前训练成功。
7. **更多步不是自动TTS收益。** GT确认24图用了8,350候选和18,704次FLUX forward计数
   （含checkpoint重算），不是16次前向。所有被拒候选、重试和基础生成成本都要记。
8. **规模既不能恐惧，也不能滥用。** 8步资格、200步拟合、覆盖更多来源的正式学习
   解决不同问题。不要把工程小样本当科学终判，也不要跳过可学习性就堆100k。
9. **保存完整状态。** 模型/Adam/RNG/采样进度/源码/拓扑都属于恢复身份。
   换卡数、节点或代码，不自动等于严格续训；保留旧结果和消耗，不能从新ID清零。
10. **不能证明首创就不写首创。** VLM+EBM、可微引导、能量自评分本身不自动新颖。
    论文仍需可部署改善、匹配计算对照、无VLM/普通质量评分/仅排序等必要对照，
    以及更广退化证据。它们是后续证据需求，不是本轮新增实验授权。

## 9. 不变量、已知未知和交接维护

不变量：

- Energy 独立增益：匹配 FLUX 的平均 PSNR +.30dB，
  LPIPS相对降幅 `1 - mean(new)/mean(baseline) >= .09928107050377874`。
- 同一方案同时满足两个目标；不按逐图百分比平均、不把dB作比例、不换弱基线。
- 当前 Noise-only、原24训练来源、512、Noise step400/50步/alpha=.30/双uint8。
- GT只用于合法训练和报告；不用于部署选择、接受、停止。没有Restormer补贴。
- 不读official/sealed/internal/parent-test和最终1859开发像素做训练、调参或选择。
- 200步不由资格权重初始化；不自动长训/扩预算；现有数据和所有记录保留。

目前仍未知：

- V3能否真正学会有用方向；改善来自哪个loss/反例组件。
- 24训练来源之外的泛化；VLM条件是否有独立贡献。
- 单步可学习性如何转成多步净收益和可靠拒绝；实际成本是否有竞争力。
- 独立校准阈值、正式训练规模、其他退化/复合退化的有效性。
- 新GPU拓扑上的真实吞吐和CUDA恢复性质，以最新资格为准，不拿CPU测试代替。

此文件写作时，远端运行计划是3×2卡资格，75项CPU回归及六rank CPU恢复测试已报告通过，
实际三节点CUDA/NCCL仍待验收 [N]。这些是新代理的记录，不是本文作者另跑一遍的结果。
后续运行状态看最新 handoff，不把这里的时间快照当永远有效的任务状态。

新代理每个重要结果后补一条决策记录，至少写：
**本轮问题 → 保持哪些条件不变 → 观察/证据文件 → 排除与未排除什么 →
下一步最小验证/停止条件 → 授权及预算 → Git提交。**
若改变本文结论，附新证据并保留原记录；不要靠聊天记忆或把失败记录删掉。

## 10. 证据索引与阅读顺序

快速接手：本文件 → 最新AGENTS/current handoff → [M,N] → 实际源码。
诊断时按下面索引深入，不要求每次重读全部历史、重做已完成实验。

- 原 PPT：[Novel PPT Approach](../presentations/novel_ppt_approach/Novel_PPT_Approach_VLM_Restoration_Energy.pptx)。
- [A] [PPT原文/实现对应、数值隐患、pairwise完整终局](foundir_vlm_and_novel_ppt_retry_audit_2026-10-27.md)。
- [B] [强residual basis最终结果](foundir_proposal_residual_basis_noise_v1_terminal_2026-10-17.md)。
- [C] [七alpha第二轮无增益](foundir_local_action_energy_noise_v1_pass2_2026-10-17.md)。
- [D] [完整候选scorer校准终判](foundir_complete_candidate_energy_risk_coverage_v1_terminal_2026-10-17.md)。
- [E] [fixed32完整energy训练拟合成功的边界](foundir_novel_ppt_trainfit_v1_terminal_2026-10-27.md)。
- [F] [FLUX输出像素energy资格：量化后零增益](foundir_flux_energy_qualification_v1_terminal_2026-10-27.md)。
- [G] [12篇关键文献/公式/代码深入审查与未复现范围](foundir_flux_tail_energy_guidance_deep_audit_2026-10-28.md)。
- [H] [尾段GT V1：接线通过，联合目标失败](foundir_flux_tail_oracle_v1_terminal_2026-10-28.md)。
- [I] [GT V2：方向尺度/量化/提前停止诊断](foundir_flux_tail_tradeoff_v2_terminal_2026-10-28.md)。
- [J] [GT budget V3：多方向与PSNR余量的消融](foundir_flux_tail_budget_v3_terminal_2026-10-28.md)。
- [K] [固定方案在另24训练来源上的确认](foundir_flux_tail_confirmation_v1_terminal_2026-10-28.md)。
- [L] [learned-tail V2：压暗、自批最大步幅、导数学习失败](foundir_flux_tail_energy_resumable_v2_terminal_2026-10-29.md)。
- [M] [relative-energy V3的科学设计与准备](foundir_flux_tail_relative_energy_v3_readiness_2026-10-29.md)。
- [N] [迁移后的最新三节点执行准备](foundir_flux_tail_v3_cluster_readiness_2026-10-30.md)。

## 11. 决策记录：三节点首轮停在历史基线复现 — 2026-10-30

- **本轮问题：** 迁移后3×2 H200能否完成V3八步工程资格；先满足原FLUX基线一致。
- **不变量：** 原24训练来源、冻结Noise step400/50步/512/alpha=.30/双uint8，
  energy架构/损失/global batch32和8步计划不变。无新增保护像素访问。
- **观察/证据：** [终态报告](foundir_flux_tail_v3_cluster_baseline_failure_2026-10-30.md)
  及同名JSON。六rank共享存储/NCCL启动通过，24基础记录完成；六rank在初始挖掘
  的历史proposal逐像素比较失败。0训练步、0probe，三节点status1，.3944444 GPU-hours。
- **排除与未排除：** 已经过网络启动阶段，66绑定源码SHA仍匹配；尚无energy学习结果。
  未保存输出差异幅度，不能归因H200、量化或运行时变化，亦未排除推理接线错误。
  基础metadata身份/SHA已复核；本次状态审计未重算全部tensor内容SHA。
- **最小下一验证/停止条件：** 建议单个合法训练来源比较历史/新基线及分阶段输出、
  像素误差和运行时指纹，定位首个分歧；取得区分原因的证据或触及预定预算即停。
  不靠新弱基线、关闭相等检查或训练energy绕过问题。此诊断尚未实现/启动。
- **授权及预算：** 本次为读远端设计记录、核验进度及记录终态；不自动新开GPU任务。
  原资格cap12 GPU-hours保留历史消耗，新诊断须单独冻结范围/预算，未获本轮状态请求授权。
- **Git提交：** 本条与终态报告同一focused docs提交，以285e73a为父版本；
  可用 `git log -- reports/foundir_flux_tail_v3_cluster_baseline_failure_2026-10-30.md` 定位。

## 12. 决策记录：短诊断改用单节点两卡 — 2026-10-30

- **问题与授权：** 用户要求短任务只用单节点两卡，长任务才跨节点，并明确继续下一步。
  这授权当前基线定位诊断，覆盖上一记录中“诊断尚未授权”的时间快照。
- **保持条件：** 原合法训练index0、冻结Noise step400/50步/512/原seed/alpha=.30/双uint8；
  复用原observation和prompt，不加载GT tensor，不训练energy，不改任何V3绑定文件。
- **证据设计：** 同一来源在两卡各生成两次，记录历史uint8差异、同卡重复/跨卡阶段差异，
  保留编码、50次模型输入/输出、解码和量化张量及运行时/driver/后端指纹。
  入口和检查见[两卡诊断准备](foundir_flux_tail_baseline_parity_readiness_2026-10-30.md)。
- **解释边界：** 同卡差异可暴露当前不稳定性；跨卡差异可暴露当前设备相关差异。
  若四次相同但不同于旧缓存，不能据此断言具体旧环境/硬件根因。旧中间态未保存，
  本次不能直接定位“历史→现在”的首个中间分歧；不冒称已恢复历史基线。
- **停止/预算：** 四次50步无引导生成完成，或任意错误/900秒cap即停，不启动后续训练。
  粗估3–8分钟尚未实测；两卡cap.5 GPU-hours，另30秒清理计实际成本，旧消耗保留。
- **当前状态：** 独立诊断已实现，GPU尚未启动。当前代理为CPU节点，交付单节点命令。
  结果出现后再更新判断，不根据预期填写指标。
- **Git提交：** 本条与准备报告/入口/CPU测试同一focused提交；
  用 `git log -- smoke/diagnose_flux_tail_baseline_parity_v1.py` 定位。

## 13. 决策记录：当前重复一致，历史基线仍有小幅确定差异 — 2026-10-30

- **问题/不变量：** 原训练index0、两卡各两次、原冻结模型/输入/seed/50步/双uint8，
  查当前重复性和相对历史差异；没有energy学习或GT图像解码。
- **证据：** [两卡终态](foundir_flux_tail_baseline_parity_terminal_2026-10-30.md)及同名JSON。
  四份160阶段trace文件逐字节相同；各次均相对历史有3630通道元素/3591像素变化，
  最大1灰度级。14文件SHA、68源码SHA、父缓存SHA及uint8计数独立复核通过。
- **支持/未排除：** 本来源当前同卡/卡间没有检测到不稳定性；历史偏差稳定且小。
  旧中间态不存在，仍无法区分生成与舍入分歧，未锁定环境/内核/硬件/路径根因。
  不将小误差改写成历史复现通过，也不扩大为全部来源/混卡资格通过。
- **下一最小验证/停止：** 先用已有raw、uint8和observation只读重算量化，查历史环境
  指纹；若不能区分上游与舍入，记录证据缺口再设计短对照，避免盲目完整重跑。
- **授权/预算：** 用户要求监测直到完成。已于18:40:24 UTC结束，92秒/.0511111
  GPU-hours；加前次失败=.4455556 GPU-hours。status2按预定规则保留结果。
  本监测未新开GPU任务、不接续资格/训练、不放宽parity。
- **Git提交：** 与本终态报告同一focused docs提交；
  `git log -- reports/foundir_flux_tail_baseline_parity_terminal_2026-10-30.md` 可定位。

## 14. 决策记录：停止外挂CNN，实施FLUX原生energy — 2026-10-31

- **用户决定：** 只用预训练FLUX作为可训练图像主干；禁止外挂CNN energy、Restormer、
  residual refiner及其蒸馏。用户已要求execute/start。不重跑历史灰度差异或GT搜索。
  本节覆盖上文所有继续V3/CNN及历史基线诊断的下一步；保留旧证据，不删旧成果。
- **已有证据：** 旧CNN的200步已结束。8个训练来源bounded PSNR增加1.060337543dB，
  LPIPS只降2.485951%，3/8图LPIPS退步，联合失败；不是源外泛化。
  不能把它说成“两个指标都差一点”，不能推断“改FLUX必然成功”。
- **问题到改动：** 复用FLUX图像表示，新增内部energy LoRA和线性读出；不从零训练
  独立图像评分网络。完整展开三次energy梯度更新，穿过冻结FLUX尾两步/VAE，用实际
  修后图像的MSE/LPIPS及同图动作排序训练；不只拟合缓存分数或导数。
- **不变量/显式变化：** Noise训练来源身份、512、原LoRA、50步、alpha=.30、双uint8
  与最终+.30dB/LPIPS相对降9.92810705%不变。新候选只由FLUX/固定图像处理构造；
  不继承旧CNN轨迹/权重。新工程global batch8、单机4/8卡，不冒充旧batch32协议。
- **已测/未测：** 小尺寸真实FLUX块的9项CPU测试通过，含三步能量更新经冻结生成尾段
  反传、有限差分、原生成不变、Adam/RNG续训。真实4B+VAE+LPIPS CUDA未运行；
  新GPU节点在主会话不可见。不能把PPT/源码就绪说成已启动或工程资格通过。
- **下一最小验证：** 新25步工程资格，8个训练来源probe报告0/1/2/3步和energy自选。
  检查真实梯度、显存、保存恢复及实际图像；8图不是泛化。GPU阶段上限2小时含缓存，
  8卡至多16、4卡至多8 GPU-hours，CPU核对和退出清理另计实际分配时间；不是耗时预测。
  不自动200步或正式训练，不因OOM暗换CNN/裁图/截断轨迹。
- **失败后如何判断：** 工程错误先定位实际栈；评分能排序但修图差则检查输入梯度和
  rollout一致性；PSNR改善而LPIPS停滞则是感知修正的有效方向仍未学会，不能直接
  推断容量不足或继续加步数。只有拟合证明有效后，才另冻独立来源检验与规模计划。
- **实施与Git：** 读 `foundir_flux_native_energy_v1_readiness_2026-10-31.md`。
  本条与native-energy代码/测试/交接同一focused提交；
  `git log -- smoke/train_foundir_flux_native_energy_v1.py` 可定位。

## 15. 决策记录：所有非轻量GPU实验默认改为3×2 — 2026-10-31

- 用户要求当前及后续非轻量GPU任务使用3节点，每节点2张H100/H200；允许节点间混用。
  轻量任务可少卡。此资源规则覆盖上文单机4/8卡和早期多节点默认，不改变FLUX-only边界。
- 当前资格仍25步/512/完整三次更新/global batch8。同一步两rank各2样本、其余各1样本，
  跨步轮换，按样本加权all-reduce；不默改batch=6/12。额外GPU不等于更低单卡显存。
- 所有节点须共享代码/资产/输出，身份和存储核对在加载4B前完成；日志锁按node rank
  区分，global rank0写主checkpoint，各rank RNG全部保存。旧checkpoint保留，不以
  改拓扑重启冒充精确续训。新fresh ID为qualification_3x2_v1。
- 本轮16项CPU测试通过，六进程Gloo对串行batch8一致、Adam/RNG恢复一致。
  这不是三台GPU主机；CUDA/NCCL/显存/真实吞吐仍需实际资格。没有启动GPU任务。
- GPU阶段每节点上限2小时，六卡上限12 GPU-hours，CPU核对和退出清理另计实际成本；
  不预测网络会加速、不自动长训。读 `foundir_flux_native_energy_3x2_readiness_2026-10-31.md`。
- 本条与3×2启动器/配置/测试/交接同一focused提交，用该readiness的git历史定位。
