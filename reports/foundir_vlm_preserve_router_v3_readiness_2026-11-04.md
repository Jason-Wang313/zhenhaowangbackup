# VLM路由修复V3：减少错误覆盖，同时保留纠错

## 当前要做什么

用户明确坚持PNG右图：**图像路由分数＋VLM token经过小MLP输出的修正分数**。
本轮只训练VLM修正头，不把image-only作为最终方案，不要求先训练新的image-only才能继续。
原image-only只作为冻结图像分支和诊断对照。五个已经训练好的FLUX专家、Qwen、VAE都不动。
不加入CNN、Restormer、RGB refiner、energy、额外门控网络或标签驱动的部署回退。
沿用真实LQ crop的Qwen视觉merger tokens均值，不把它说成VLM输出的退化文字或等级标签。

上一轮expanded_frozen_image_v2已经结束，GitHub 3d6d9fd；不要重启旧缓存/训练。
其独立400图：image-only350正确，VLM374正确；纠正36、改错12。
Blur77→76、Noise61→76、Haze76→69、Lowlight76→78、Rain60→75。
总体改善而每类不降门槛失败；未产生新PSNR/LPIPS。不能用这400图继续选新配方和检查点，
也不能把此次修复的校准结果称为新的独立泛化成功。

## 为什么改损失，而不是重训修图专家

冻结图像路由的参数，只能保证图像分支自身不被改坏；VLM加上的分数仍能把正确选项覆盖。
这轮问题是选错专家，没有证据说五个冻结修图专家的权重被改坏。
分类CE原本也惩罚错选，新约束不是提供新信息，而是更明确地提高错误覆盖的训练代价。
它可能改善取舍，也可能减少有用纠错或毫无改善；不是“保证no-harm”的方法或已证实的新贡献。

## 一套固定配方，不大范围扫参

复用V2的2560张训练/400张校准特征、原seed/crop/采样顺序与冻结图像权重。
加载fresh_ce400_matched_v1真正image_only step400图像MLP，冻结；VLM从与V2相同seed的
fresh零输出初始化开始，fresh AdamW。不从V2选中的step800继续，不继承旧Adam。

1600步/global32，每张训练记录仍20次曝光；lr=.001、weight_decay=.01、label_smoothing=.05、
clip_grad_norm=1，CPU8线程。只改下面这个损失项，系数固定1，不训练多个系数版本：

令训练真类别为y，原图像分数为b，最终分数z=b+VLM(v)。
margin(b,y)=b_y−max(b_错误类别)。新增惩罚为：

`mean_over_batch[ 1(argmax(b)=y) × relu(min(max(margin(b,y),0),0.2) − margin(z,y)) ]`

最终loss=原分类CE＋新增惩罚。

- 图像分支原本正确：不要求复制其完整分数，只要求保留至多0.2的正确排序余量。
  不强行继承极高置信度，也不让原本很小的正确余量突然承担0.2的额外要求。
- 图像分支原本错误：不保护错误答案，仍用CE让VLM纠正。
- mask与参考分数detach；只在训练用类别判断对错，推理不需要标签/GT，也不加新阈值。
- 惩罚按完整batch平均，不因某批正确样本少就把单样本权重放大。
- 每步记录CE、保护惩罚、受保护训练样本数、误改数及裁剪前梯度范数。

本轮不扫0.2或系数1；它们是预先固定的尝试，不是通过旧holdout挑到的最佳值。

## 只用校准集选一次，不把“完全不改”当成功

400/800/1200/1600评价同一400张校准集，与冻结image-only及V2选中step800的校准预测比较。
原始图像分数/样本/裁图必须完全一致；不把联合模型内的图像分支当独立对照。
候选必须同时满足：

1. 总正确数高于冻结image-only，每类正确数不少于它，CE不高于它。
2. 相比V2在**同一校准集**的误改数量严格减少，纠正数量不少于V2。

符合者中选择最低校准CE，平局最早。若没有符合者，保留最低CE检查点作失败诊断，
明确`calibration_screen_pass=false`，不能拿诊断选择冒充通过或部署。
旧400图的36/12只说明历史问题，不当本轮校准数量阈值，因为两组不是同一批图片。
已按V2逐图JSON重算：校准组为纠正28、误改12，其中Haze误改11；它与holdout的纠正36并非同一数字。
完整逐图/每类/每检查点计数保留；选择结果不可变落盘。
这是更严格的修复筛选，不意味着满足它是唯一可研究的取舍；未满足时先报告，不自动放宽规则。

## 新独立来源：先预留，当前不启动新的GPU

本地主代理已只读真实manifest，复现原680与V2全部3360条选择哈希后，
排除全部旧fit/calibration/holdout、旧40开发图，以及manifest所有nontrain来源/GT/LQ身份。
剩余Lowlight只有37个可用来源，不足再作每类80或400张的新验证。
因此固定预留**每类20、总100个独立来源**；按固定哈希选择，不看预测、图像或质量。
跨类别也不复用来源/GT/LQ哈希，预留SHA为：
`57596103ccf992e6f47dbcdb0bce0c1c0d3a8b7ee02e78837fe43b929b6043a7`。
目的地必须复现相同清单；不够/不一致就报告，不能换回旧400图。

当前只保存预留元数据，不加载它们的像素/特征，不提取新GPU缓存，不做恢复评价。
校准结果值得继续后，再单独安排该100图确认；每类20的统计余量有限，不能称全面论文验证。
来源分离仅针对路由训练与已有路由检查，不保证FLUX/Qwen历史预训练从未见过。
原2560对640数据量差异仍存在，不宣称数据量匹配的VLM因果消融。
当前仍是五类singleton top1，不是已经验证double/triple，也不是PNG全部能力或官方SoW通过。

## 目的地执行与保存

唯一执行根为`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`，
环境为该根`.conda/gpu-train-stable/bin/python`。主代理没有这个挂载，不能声称已启动现场训练。
旧HDD3冻结；本地只在隔离worktree做代码与合成CPU测试，不把合成结果当真实实验。

CPU代理安全同步origin/main，保留无关修改/权重/数据/缓存，阅读最新AGENTS顶部后执行：

`bash smoke/start_foundir_vlm_preserve_router_v3_tmux.sh`

它启动prepare→train→verify，无GPU命令。prepare只核验旧元数据/报告和预留新清单，
train只加载fit/calibration特征；旧holdout的JSON报告可用于父身份核验，但禁止加载其tensor。
CPU限制8线程、累计硬cap3600秒、0使用GPU-hours，失败耗时保留；缺结束账先调查。
旧相近训练+verify为350秒，本轮估计约6–15分钟，包含清单/缓存/状态I/O；不是实测保证。
1601份状态保存约18–22GiB，首启要求30GiB可用。每完整optimizer step保存VLM/Adam/RNG/slot。
同命令恢复同runtime最后完整步；损坏/缺步不回退，孤立未发布文件保留，不重命名旧状态。
校准选择只允许本轮固定四个检查点，重启补齐中断处的校准结果。
完整结果存在时只核验，不重复训练/评价。verify核查全部步marker齐全及step0/所选/最终文件SHA，
不是声称逐份重读1601个tensor。所有历史源码、已绑定缓存和结果保持不动。

必须看到`VLM_PRESERVE_V3_RESULT_VERIFIED`及`VLM_PRESERVE_V3_END status=0`。
结果`reports/foundir_vlm_preserve_router_v3_terminal_2026-11-04.json`；
日志`logs/foundir_vlm_preserve_router_v3_cpu_train.log`；状态根
`checkpoints/foundir_vlm_preserve_router_v3/preserve_correct_v3`。
CPU代理只push结果JSON及耗时/退出简报，勿push权重/特征/图片/live logs。
失败不自行改变配方、扩预算或启动GPU。下一阶段由主代理结合结果决定。

## 硬件纠正

V2特征提取实际每卡峰值137.473185GiB；之前“代码支持H100”不是H10080GB显存通过。
当前V3是CPU任务，不受这个显存问题影响。以后100图提取必须先解决硬件资格，
不能给80GB H100直接发旧提取入口，也不能悄悄改精度/裁图来声称兼容。

## 本地测试完成，现场训练仍待执行

78项CPU测试通过（13项本轮新测试＋65项既有回归），Python编译、stdlib-only入口和shell语法通过。
新测试覆盖错误覆盖惩罚的梯度方向、原图像错误时不受保护、正确但不同分数不被强制复制、
低margin/平局/空保护项有限、图像权重冻结、VLM更新、Adam/RNG下一步精确恢复、
禁止读取旧holdout和新预留特征、来源/GT/LQ排除、数据不足不补旧图、不能靠关VLM通过、
无合格检查点时只作失败诊断、完整1600步小维度合成流程与1601份状态、结果重算及重复完成不重训。
这些合成测试输出不是新分类/恢复科学结果。目的地真实preflight/训练尚未启动，不能宣称通过。
