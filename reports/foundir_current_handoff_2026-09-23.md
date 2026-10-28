# FoundIR current handoff — 2026-09-23

## MANDATORY: all future experiments support single-node 4/8 H100 or H200 — 2026-10-28

用户明确要求：从本次开始，所有后续实验、训练、评价、诊断和启动脚本都必须支持
单节点4×H100、4×H200、8×H100、8×H200。此规则覆盖历史“必须8卡”的操作限制，
但不是授权恢复旧实验或更改当前研究方向；历史脚本不因此自动视为已兼容。
每次只用同型号GPU，不混H100/H200，不跨节点。不得根据节点名推断型号。
当前实现以每卡至少75GiB可见显存为验收条件；其他显存规格须单独验证，不冒称已支持。

新agent必须遵守：
- 启动器显式接受4/8卡，检查实际CUDA可见数量、型号一致性和空闲状态；
  不把world_size、gather列表、样本分片、日志或GPU-hours硬编码成8。
  tmux必须显式传入卡数和CUDA_VISIBLE_DEVICES，不能依赖旧server环境。
- 训练用梯度累积保持原global batch、loss归一化、数据曝光量和optimizer-step口径；
  卡数减少不自动改学习率、裁图、loss、验证集、科学门槛或训练规模。
  若不能保持等价，必须先解释差异并取得用户确认，不能静默降低实验标准。
- 评价/诊断保持同一完整记录集合、crop、seed、指标、量化与权重；4卡顺序处理
  更多记录，不能丢掉一半数据。汇总防漏样本和重复计数。
- 换卡数续训必须核验optimizer/scheduler、RNG及sampler进度的可恢复性；
  不把重新初始化或数据顺序变化冒充严格续训，不覆盖已有run/artifacts。
- 每个新入口都测试4/8两种分片/汇总及CPU静态路径；分别记录实际GPU资格状态。
  “代码支持”不等于“4卡和8卡都已跑通”；无GPU时必须明确标待GPU验收。
- 分别报普通wall hours及实际使用卡数×耗时的GPU-hours。4卡可能更慢，
  不承诺精确2倍耗时；若实际占用整台8卡节点而只用4卡，还须另报8卡资源占用成本。
- 未来沿用通用命名/参数入口；历史_8gpu入口只能明确作兼容wrapper，
  不要让fresh agent误以为还必须8卡。保留历史结果，修改后focused commit/push。

历史Noise FLUX V1诊断已实现4/8支持，固定8训练来源、同46个arms、无新训练：
4卡每rank处理2张（先后独立加载、释放模型，不增加单卡batch），8卡每rank1张。
通用入口smoke/run_foundir_flux_tail_oracle_v1_gpu.sh <fresh_run_id> <4|8>；
tmux入口smoke/start_foundir_flux_tail_oracle_v1_tmux.sh <fresh_run_id> <4|8>。
V1历史预算cap：4卡3普通小时/12 GPU-hours；8卡1.5普通小时/12 GPU-hours，
另最多30秒清理。这是预算，不是实测ETA。4×H100已实测完成；8卡/H200尚未验收。
详细变更见reports/foundir_flux_tail_oracle_4_8gpu_compatibility_2026-10-28.md。

## ACTIVE: GT-only PSNR-budget / direction diagnosis V3 — 2026-10-28

用户明确授权：先解决 GT 诊断仍未通过双目标的问题，再检查更多来源；
不直接启动正式 energy 训练。此节覆盖下面终态/历史计划，不恢复其它分支。
V3 保持同八个 training sources、512裁图/seed、Noise step400、50步、
alpha=.30、双uint8量化；仅 index48 latent 真实两步rollout。
不只挑困难三图，不访问任何测试/封存像素及最终1859条开发集。
V1/V2绑定源码与结果不修改，新输出根 foundir_flux_tail_budget_v3。

四组匹配对照：balanced/mixed 方向 × monotone/psnr_budget 接受规则。
balanced 为 MSE/LPIPS 梯度各自RMS归一化后等权混合；mixed 同时试
LPIPS权重0/.25/.5/.75/1。共同步幅.0005/.001/.002/.005/.01。
monotone 不允许PSNR低于当前值；budget 达到原基线+.30dB后才允许
用掉超额PSNR，不能跌破目标；尚未达到时保持PSNR进度。
两组均不接受LPIPS恶化，并要求不损害原基线；先缩小PSNR目标缺口，
可行候选中优先最低量化LPIPS，再较高PSNR/较小步幅，固定顺序破平局。
最多16次尝试更新，1/2/4/8/16报告；达到目标不提前停，
无合法候选才停。共20轨迹报告项+2对照，全部尝试与拒绝保留。

四组共享排序及迭代上限；mixed每次25候选而balanced为5，
计算量不同，必须记录所有前向、重算、反向、解码和拒绝试步成本，
不能宣称等算力部署TTS优势。未接受更新时保留FP32 latent对照，
可能与历史BF16有极小差异，不用GT事后换回较好图来美化均值。
接受/停止直接用GT，仍不是可部署方法、energy学习或泛化证据。
原目标+.30dB与LPIPS相对下降9.928107050377874%不变，不承诺一定通过。

单节点同型号4/8 H100/H200；4卡顺序每rank两张，8卡每rank一张。
上限4卡1小时、8卡.5小时，均4 GPU-hours，另最多30秒清理；
这是cap不是ETA。独占八卡但用四卡须另报整机成本。
入口 smoke/start_foundir_flux_tail_budget_v3_tmux.sh <fresh_id> <4|8>；
建议四卡ID budget_4gpu_v1。Codex无GPU，不得把代码/CPU/preflight通过
说成GPU任务已经启动。无自动重训/数据扩展；不覆盖已有run。

更多来源计划：先检查八图双指标余量；通过后冻结单个方案，再检查
现有缓存中另外24个训练来源（索引8–31）。它们不是最终泛化测试，
本次V3不读取其像素，不自动扩展，也不逐图挑最佳方案冒充统一方法。
八图通过不能保证正式训练成功；学习和泛化仍须后续独立证据。
详见 reports/foundir_flux_tail_budget_v3_readiness_2026-10-28.md。

V3最终28项CPU测试、Python/shell静态检查和4/8卡权重/缓存元数据预检通过，
未解码原图。配置SHA256：
58ebc99b3bc92b4db4eaf80b3489bd3047bd0f6014e8ca26dbcdabe25a54ef9d。
GPU运行仍未启动，等待用户在GPU节点执行上述tmux入口。

## TERMINAL: V2 tail diagnostic complete; joint target still short — 2026-10-28

tradeoff_4gpu_v1 已正常结束，当前没有待续的 V2 任务，不自动启动 energy 训练。
4×H100 80GB，2026-10-28T20:44:07Z–20:47:38Z，launcher status=0。
总 211 秒 / .0586111111 普通小时 / .2344444444 使用 GPU-hours；
若独占整台八卡节点，对应 .4688888889 整机 GPU-hours（是否独占未确认）。
八训练来源×50报告项完整；基线逐像素复现、STE 前向一致、所有绑定源码/
配置哈希、sample/rank 完整性及独立均值重算均通过。每卡峰值 15.309712 GiB。
仅四卡 H100 实测通过；不冒称八卡/H200已实际验收。

原部署基线 39.64431142807007/.07384782587178051。
连续代理的 GT 安全轨迹最终为 40.51412868499756/.06995537877082825：
PSNR +.8698172569274902 dB，LPIPS 降 5.270902772020114%。
双量化 STE 轨迹为 40.512683391571045/.07000666065141559，
PSNR +.8683719635009766 dB，LPIPS 降 5.201460131045987%。
总体目标仍是 +.30 dB / LPIPS 降 9.928107050377874%，对应
39.944311428070066/.0665161346648535；全部48个引导报告项无一同时通过。
不得把这些 GT 结果当作训练出的 energy、VLM贡献、部署或泛化成绩。

两种代理均五张达到单图目标，三张（索引0/2/7）在固定方向/幅度下无安全
候选而停住；它们占原LPIPS总和约75.28%。所有轨迹至多接受四次更新，
step4=step8，不是继续更新没有涨。连续代理初始 MSE/LPIPS 梯度强度比
中位数6.197、范围1.472–42.466；平衡确实改善LPIPS，但还不够。
换STE没有解决缺口，不能因此说量化无影响。样本2没有接受修正，
保留FP32 latent对照，相对历史BF16仍有-.000534dB/+.0000516 LPIPS小差异；
不得声称每张图相对历史基线绝对无损。

本次证明生成中有双指标改善空间，尚未证明完整目标可行，更未证明energy可学习。
旧ACTIVE/readiness段落仅为本次历史计划。后续有限诊断或训练需新的用户授权，
不扩大预算/数据、不更改绑定源码或放宽门槛、不恢复Restormer/MoE/EBT。
报告及全50项均值：reports/foundir_flux_tail_tradeoff_v2_terminal_2026-10-28.{md,json}。
原始result SHA256：78e442d5f3a4cda82c15b2903b02d66a2ab790ac167df8bc12e9a8bcf1858625。

## HISTORICAL PLAN: GT-only LPIPS/PSNR trade-off diagnostic V2 — 2026-10-28

用户在看完V1完成结果后授权下一步。只做training-source-only的有限诊断，
不启动energy训练、不变更已有FLUX权重、不恢复Restormer/MoE/EBT。
V1结果及绑定源码保持不动；新入口/输出根为foundir_flux_tail_tradeoff_v2。
入口：smoke/start_foundir_flux_tail_tradeoff_v2_tmux.sh <fresh_id> <4|8>。

使用完全相同8训练来源/512裁图/seed/Noise step400/50步/alpha=.30/双uint8量化。
仅index48的真实剩余两步rollout，不再同时扫多个窗口，避免把预览误差混入。
先分开测normalized MSE与LPIPS的梯度RMS、比值和夹角，再比较：
MSE-only、LPIPS-only、原联合loss梯度、两种梯度各自归一化后的平衡方向。
原联合loss通过单独的联合backward获得，不用两个BF16 backward相加冒充原实现。
幅度冻结.0005/.001/.002/.005/.01；所有40个single-update arms保留。

两种目标：原continuous blend；保留真实双round前向但使用straight-through近似梯度。
STE不是取整的真实导数，不能宣称其通过有限差分。每个实际输出仍按真实量化评分，
并核验STE前向逐像素等于deployed_proposal，记录连续指标与量化指标差异。
每种目标额外一个最多8次的balanced GT-only轨迹：每次用同5步长试完，
只接受量化PSNR/LPIPS都不损害当前结果且至少一项改善的候选；
无安全候选或达到原双目标则停止，保留1/2/4/8步报告及全部接受/拒绝记录。
这依赖GT，不是可部署TTS/自验证，也不证明energy可学习。严格no-harm有限搜索失败
不能证明不存在需暂时权衡指标的其他路径，8训练图也不证明泛化。

共50报告arms；原BF16及同精度无引导control仍保留。
相对匹配FLUX的+.30dB/LPIPS下降9.928107050377874%目标不变，不以旧Restormer抵扣。
单机4/8同型号H100/H200；4卡每rank顺序2张，8卡每rank1张。
预算cap：4卡1小时，8卡.5小时，均4 GPU-hours，另最多30秒清理；
这是上限，不是已实测ETA。Codex无GPU，本轮只完成实现/CPU及元数据预检，
不声称新GPU任务已启动。完成前不得自动扩大数据或启动energy长训。
报告：reports/foundir_flux_tail_tradeoff_v2_readiness_2026-10-28.md。

## MILESTONE: four-H100 tail diagnostic complete; joint target NOT met — 2026-10-28

mechanism_4gpu_v1已正常结束，4×NVIDIA H100 80GB，2026-10-28T20:13:53Z–
20:22:22Z，launcher status=0，TAIL_ORACLE_RESULT_VERIFIED。总509秒/
.1413888889普通小时/.5655555556使用GPU-hours；纯worker阶段.0938851541小时。
8训练来源×46个arms完整；8/8基线逐像素复现，16/16 FP32预览/两步rollout
有限差分通过，最大相对误差.0097435826。单卡峰值22.5287GiB；源码/配置身份一致。
这实际验证了4×H100路径，不能扩称8卡或H200已完成GPU验收。

原基线39.64431142807007/.07384782587178051。
最佳PSNR固定arm preview_skip_i46_w0_rms0.005：
40.46877956390381/.07429557666182518，PSNR+.8244681358dB，但LPIPS恶化.606315%。
真实两步rollout+联合目标、小幅度.005：
40.45664739608765/.0739425360225141，PSNR+.8123359680dB，LPIPS仍恶化.128250%。
最佳LPIPS固定arm preview_skip_i48_w1_rms0.02：
39.27623128890991/.0733890135306865，LPIPS只降.621294%，PSNR反降.3680801392dB。
42个引导arms无一同时达到+.30dB和LPIPS下降9.928107%；本8图对应LPIPS目标.0665161347。

结论：GT引导在有限窗口有明确PSNR修正空间，不是简单“FLUX末段改不动”；
本次单步/权重/幅度配置尚未证明双指标可行，尤其LPIPS不足。
不是energy学会、不是VLM贡献、不是泛化或论文成功；没有训练energy。
下一步需先针对LPIPS目标/步幅及实际量化终图对齐提出有边界诊断，
不能直接据此自动启动energy长训，也不能由8图有限搜索否定全部路线。
报告：reports/foundir_flux_tail_oracle_v1_terminal_2026-10-28.md。
result SHA256：4985f367cca8bc1fb730dee173d111eb4929fc6ebda4b9d0db6da223d523f7b1。

## ACTIVE: Noise FLUX training-only tail mechanism diagnostic — 2026-10-28

用户明确授权执行最小机制诊断；此节覆盖下面所有旧Rain/Generalist/MoE/residual优先级。
唯一当前工作是冻结Noise FLUX生成过程的GT-only机制诊断，不训练energy或修改LoRA，
不恢复最终pixel descent、Restormer residual、MoE、EBT Generalist，不自动长训。
报告：reports/foundir_flux_tail_oracle_v1_readiness_2026-10-28.md。
入口：smoke/start_foundir_flux_tail_oracle_v1_tmux.sh mechanism_v1。

冻结复用已绑定qualification_v1缓存的前8条（8个training sources），512裁图；
不是最终1859条开发集，也不是泛化证据。GT只用于非部署诊断，全部测试像素仍禁止。
保持Noise step400、50步真实shifted schedule、reference time10、CFG1、alpha=.30、
两次uint8量化。零引导必须逐像素复现历史入口和缓存，否则立即失败。
仅在梯度数值检查阶段把模型转换FP32并关闭TF32，随后恢复原参数/buffer精度及TF32。
引导使用FP32 latent累积/BF16 FLUX；每个窗口都有同精度无引导control。

比较零基索引40/46/48处的一次预览梯度修正（完整及skip-J近似），
48处另做真实两步rollout梯度；不是默认最后几步最佳。
预先固定latent RMS幅度.005/.02/.08，目标为归一化MSE或MSE+LPIPS。
实际修改后重新计算FLUX velocity并完成剩余轨迹，按真实量化终图评分。
FP32完整预览及两步rollout均须通过有限差分，不能把skip-J叫精确梯度。
逐arm报告失败和收益，不用GT选优隐藏失败，不把有限搜索叫全局上界。
GT诊断成功不保证energy可学习；8来源失败也不证伪所有窗口/训练方案。
VLM等级仅记录来源，本诊断不用VLM引导，不能算VLM必要性证据。

单节点同型号4或8张H100/H200，一次一个任务；硬件空闲检查、节点锁、fresh ID、
输出不覆盖、源码/数据哈希绑定和timeout均启用。4卡上限3普通小时，
8卡上限1.5普通小时，均12 GPU-hours，另最多30秒清理；不是实测ETA。
记录实际forward/block recompute/backward/解码计数、显存和运行时间。
Codex当前无GPU设备；CPU/static及预检不等于实际GPU验收，不能说任务已启动。
正式energy仍须新决策/冻结配方，独立目标+.30dB及LPIPS降低9.928107050377874%不变。
继续简明中文汇报；每个实质节点focused commit/push并核验远端。

## REVIEW: FLUX tail-guidance deep paper/code audit — 2026-10-28

用户本轮要求先细读关键论文和代码，再复述计划；不是授权启动新GPU实验。
完成报告：reports/foundir_flux_tail_energy_guidance_deep_audit_2026-10-28.md。
12篇指定版本的正文/公式/实质附录已分段细读；9个官方仓库相关调用链做了静态核查。
不能说已复现全部论文、检查了所有图像或排除了所有bug；ResFlow-Tuner未找到方法代码，
D-Flow/KDS本轮只核原文/算法，明确保留实现核验缺口。

继续以冻结Noise FLUX生成过程中的条件energy引导为唯一当前方向，
不恢复最终pixel descent、Restormer residual、MoE或从零EBT。
零引导需复现50步/alpha=.30/两次uint8量化；保护全部封存/测试像素。
当前入口inference_mode不能直接启用输入梯度；需独立可微入口。
真实shifted t下预览为z_t-t*v，导数为I-t*J_v；完整预览梯度不等于真实终点梯度。
FlowChef跳过Jacobian是近似，并在公开路径复用旧velocity；不要宣称等价。
KDS附录关闭t/T<.3引导，FreeDoM使用中段窗口，故不能默认最后2–4步最佳。
IRED仍有二阶导；更低energy不保证正确；CEP有限候选目标不保证exact。

建议下次实施顺序（本轮未实施）：零引导一致性/梯度检查 →
training-source-only GT目标的真实tail可改进空间诊断 →
训练真实FLUX预览分布上的energy（排序和方向）→ 实际量化终图配对评估。
GT oracle只诊断，不得进入部署，oracle可行不证明可学习；小范围失败也不证伪全部方向。
保留energy独立+.30dB/LPIPS相对降低9.928107050377874%的原目标。
不默认复用25步energy，也不把32张工程资格当科学规模；先实测再冻结正式规模。
固定预算引导先于未校准的自适应停止；同一energy自验证仍须证据。
比较原基线及相同总计算/精度的无energy控制，计入全部反向、VAE和rollout。
本轮仅文档，无新训练代码/GPU任务，GPU-hours=0；窗口、预算、正式配方尚未冻结。

## ACTIVE: investigate FLUX tail guidance; abandon output-pixel descent — 2026-10-27

用户最新明确决定：不再继续当前FLUX最终输出上的直接pixel energy descent，
改为探索引导FLUX最后少量采样步骤。此决定覆盖此前“必须先证明pixel energy有效/
纹理受损才允许转采样引导”的前置条件；但用户本轮要求先广泛查论文和既有方法，
不是授权立即新训或把现有25步energy直接接入FLUX。
旧qualification已完成，无需杀进程；不得自动formal或恢复旧PPT/Restormer/MoE。
旧25步结果保留为工程通过/量化收益0，不宣称科学上已证伪整条energy路线。

调研依据：reports/foundir_flux_tail_energy_guidance_literature_2026-10-27.md。
已核查IRED、Reduce Reuse Recycle、CEP、DPS、Universal Guidance、FreeDoM、
On the Guidance of Flow Matching、D-Flow、FlowChef、Trust Sampling、
KDS和ResFlow-Tuner等原文方法，并只读检查IRED/FlowChef/flow_guidance公开代码。
中间态适配有已有解法：t/noise-conditioned energy；predicted-clean preview评分；
短段真实rollout后评分。不要再把它说成未解决、必须从零发明的障碍。
但解决输入分布错位不等于已有energy已经可靠或能通过PSNR/LPIPS门槛。

建议下一设计以FLUX实际末段的predicted-clean preview + 条件quality energy为起点，
保留冻结FLUX及其生成步骤；energy训练数据必须覆盖实际预览/候选分布。
不是直接复用25步energy，不是把IRED完整conditional score盲目加到已有conditional
FLUX向量场，不是把DDPM公式或DDPM噪声时标照抄到rectified flow。
本仓库t=1为噪声、t=0为clean；z0_hat=z_t-t*v(z_t,t)是单步endpoint估计，
不是精确最终结果；真实shifted schedule而非step_index/50决定t。
完整输入梯度是准确对照；FlowChef-style跳过FLUX Jacobian是可测试加速近似，
不是“完全不需要梯度”。可微preview仍需VAE/energy反传。

保留Noise step400/50步/alpha=.30部署基线、最终uint8量化、匹配开发协议和
energy独立+.30dB / LPIPS相对降低9.928107050377874%的目标。
不允许换成较弱raw FLUX，也不把GT、未知真实退化算子或perfect noise level当推理条件。
后续需零引导一致性、原50步基线、匹配总计算预算的无energy对照；
预算包括前向/反向、解码、评分/搜索，不能仅比步数。末段具体窗口和引导强度尚未冻结。
文献不证明只引导最后几步一定足够，也不保证同升两个指标。

Novelty边界：FlowChef已有FLUX引导；ResFlow-Tuner已有VLM语义条件/FLUX恢复/
中间轨迹搜索；普通energy guidance、TTS或其简单组合不能称首次。
本轮仅文献、代码只读核对及决策文档；无新GPU任务、无实验脚本改动。
单节点8xH100/H200约束保持；未来实验普通小时/GPU-hours待具体设计资格实测。

## MILESTONE: FLUX-only energy qualification completed; zero measured gain — 2026-10-27

qualification_v1于22:42:21Z正常结束，25/25步，launcher status=0且
FLUX_ENERGY_RESULT_VERIFIED。完整耗时992秒（16分32秒，.275555556普通小时，
2.204444444 allocated H100 GPU-hours）；不是先前保守预估的.5–1.5小时。
FLUX候选阶段106.3025267579942秒；energy平均20.694357499321924秒/步，
每rank峰值14.600883483886719 GiB。八rank全部参数hash一致，checkpoint哈希已复核，
运行绑定源码与当前文件一致。没有受保护像素访问，没有额外Restormer。

仅32条训练来源上的量化评估：baseline、energy 1/2/3和selected逐条记录指标完全相同，
均值40.41285765171051/.05783069503377192，即PSNR增益0、LPIPS改善0。
初始选择全为起点；训练后32张全选step3，但不产生量化指标收益。
这证明工程可运行，不证明energy正确评价画质、自验证或科学成功。
末步训练有非零修正（rank0平均.00011517549137352034，[-1,1]空间）；
幅度远小于一个8bit灰度间隔，符合量化掩盖微小修正的解释，但未存逐像素
最终float差值，不能据此断言所有像素都未变化或量化是唯一瓶颈。
25步仍在100步warmup内，不能仅凭本次验收终判方法失败，也不能假定扩训必然成功。

不要自动formal、扩退化、末段FLUX引导或恢复旧Restormer/MoE。
qualification权重不得seed正式训练。后续需另行冻结足够训练预算和独立开发验证，
目标仍是energy相对FLUX独立+.30dB / LPIPS降低9.93%，未改变。
报告：reports/foundir_flux_energy_qualification_v1_terminal_2026-10-27.md。
结果SHA256：1b906a58c423c3fcda6264143c86008b857be6c9727ab5b3ed904ab4e428df75。
checkpoint SHA256：3640540281ea20a9dd50788cd356366f0a4398b7613c73977cccdffc1766caf2。

## ACTIVE: FLUX-only energy qualification ready; sampler guidance conditional — 2026-10-27

用户授权先执行冻结FLUX输出上的energy修图接入与25步工程验收。
只有后续独立开发验证证明energy判断可靠、且像素修正导致纹理损伤限制收益，
才考虑引导FLUX最后少量采样步骤；当前不实现、不启动该备选。
未来必须匹配总计算预算的无energy FLUX对照，计入energy前向/反向与VAE开销，
不能只匹配采样步数。MoE继续暂缓，一次一个GPU任务，不恢复旧Restormer-energy。

新入口：smoke/start_foundir_flux_energy_qualification_v1_tmux.sh。
配置：configs/foundir_flux_energy_qualification_v1.json。
先单GPU离线BF16解量化Qwen生成Noise/Haze文本缓存，退出后八卡生成真实FLUX候选；
再退出全部FLUX/AE进程，仅八卡训练energy。不得使用Rain提示词缓存。
32条来自32个training source的固定512裁图；训练256同步裁图，验收512。
Noise step400、50步、先量化raw再固定alpha=.30再量化，保留原部署基线。
真实wrong-Haze expert、人工不足/过度恢复和低通损坏、训练GT、完整1/2/3步轨迹
均有明确身份。GT只用于训练/评分，不进入trajectory或推理选择。
不加载额外residual Restormer，不训练直接refiner。

只做fresh 25步工程验收；32张训练图上的指标不是泛化结果，不自动formal、
不允许用qualification权重seed正式训练。内部NAF beta/gamma初始化为.1，
输出head仍为0，初始energy轨迹必须严格等于FLUX基线；这是解除初始化退化，
不是已证明的画质提升。后续科学目标仍是energy独立增加>=.30dB并降低LPIPS
>=9.928107050377874%，正式匹配协议仍1859条/34来源/512，不拿旧43.x指标抵扣。
源/配置、权重、提示词、数据、裁图与候选缓存都有身份检查；训练/heldout来源和GT哈希隔离。

CPU单测、双rank Gloo二阶反传和真实权重/元数据预检已通过；GPU验收尚未执行。
预检确认54585训练条目/1755来源全部满足512尺寸，未读取图片。
单机同型号8xH100/H200，预计.5–1.5普通小时/4–12 allocated GPU-hours，
包含串行文本编码、候选生成和energy验收；以实测替换估计，再决定全量缓存预算。
当前Codex无GPU执行通道，需用户在GPU节点运行入口，不得声称已经启动。
详见reports/foundir_flux_energy_qualification_v1_readiness_2026-10-27.md。

旧stage1_retry1最新日志仅到580/1706，随后21:55:21Z出现launcher END status=0，
elapsed_seconds=13182（3.661667小时，29.293333 allocated GPU-hours）。
没有训练完成标志或最终科学验证，不能把退出码0当作实验成功。
未核验远端PID、未替用户杀进程；新入口用同节点锁、精确旧进程检查和空闲显存检查防争卡。
旧未完成nonzero_init Restormer草稿继续禁止启动，所有历史文件保留。

## ACTIVE: FLUX proposal + energy; measure energy-only relative gains — 2026-10-27

用户最新决定覆盖下面所有Restormer/residual-basis PPT active/live目标。
主实验先做Noise：冻结当前可用FLUX输出，VLM条件energy通过同一个energy的梯度修图。
不加额外residual Restormer或可训练直接refiner。MoE仍暂缓。
用户并非否认Restormer可作为其他实验基线；关键是energy收益必须独立计量。

门槛必须相对匹配FLUX基线重新计算：PSNR增加>=0.30 dB；
LPIPS降低比例>=0.005/0.050362067760033824=0.09928107050377874（约9.93%）。
不得沿用43.424160178005695/.045362067760033824，也不得说energy只差.027 dB
或LPIPS已经完成。基线已有提升不能抵扣energy目标。

保留已验证Noise FLUX step400 + 固定alpha=.30/50步的部署处理。
alpha只是算术混合，不是residual Restormer；不得换成较弱的raw FLUX制造增益。
最终比较使用同一1859条/34来源/512 crop/量化/指标/种子协议；旧256条43.x不适用。
仅当匹配重现旧FLUX基线时，换算目标约40.45055349220034/.062041844133373776。
这些只是目标，不是新结果。自选优于固定三步、VLM必要性和跨退化证据另行验证。

当前完成的是协议与配对增益检查，不是GPU launch readiness：
configs/foundir_flux_proposal_energy_noise_v1_plan.json；
reports/foundir_flux_proposal_energy_noise_v1_decision_2026-10-27.md。
先核对Noise prompt、现有FLUX输出身份和训练来源缓存吞吐，再接完整PPT候选/三步轨迹，
冻结预算和资格后才能启动；不直接外推旧Restormer训练速度，不自动全量缓存或长训。
单节点同型号8xH100/H200，保持全部受保护像素禁令。

旧stage1_retry1最后只读见step565，仍是Restormer-basis energy，不能当FLUX结果。
该旧任务不再是必须完成的研究前置条件；不自动恢复、延长或切改其绑定代码。
停止状态需实际GPU节点确认，不在此声称已停止，也不擅自杀进程或与它争卡。
未测试的nonzero_init_qualification Restormer脚本是已被新方向替代的草稿，禁止启动。
保留旧权重/日志/草稿，不销毁历史。所有汇报用简明中文，说明数据和增益归属。

## MILESTONE: PPT step400 checkpoint verified; training near step420 — 2026-10-27

截至20:56Z，stage1_retry1日志到420/1706（24.6189917936694%），持续更新，
无OOM/NaN/NCCL/runtime错误。100/200/300/400步均产生完整pt与JSON receipt。
step400 SHA82a36b802920972ec234bd217a8a4663f55366d64521620de3fe37692c1604a6
与receipt一致，23项源码绑定未变，next_sample_step=401。
这证明实际GPU运行能保存checkpoint；尚未在GPU上演练中断恢复。

320–420步的21个日志采样平均22.325224876484192秒/步，峰值18.720479488372803 GiB。
从420到1706纯训练外推7.9750664419885195小时；含最终验证暂估8–8.5小时/
64–68 allocated GPU-hours，依实际速度更新。不改计划或自动扩训。

科学信号仍弱：同一日志窗口平均gradient_cosine=-.0003576330413222552，
平均update_abs_mean=.00004885955797546904（[-1,1]图像空间）。
这是训练批次诊断，不是开发集PSNR/LPIPS，不把工程健康说成画质已改善。
尚无训练后开发验证，不能宣布PPT成功或失败；继续固定1706步验证。
MoE仍暂缓。本次仅查日志/JSON/哈希，未读取图片或重新运行训练/评价。

## LIVE: PPT stage1_retry1 is training on 8 H100 — 2026-10-27

stage1_retry1于18:15:39Z启动，八卡硬件/空闲显存、真实数据与源码绑定、
NCCL通信检查均通过。节点hostname为eight-h200-copy-copy--b212aa683678-4227pzjtb4，
实际硬件为8×H100 80GB；不按hostname中的h200判断型号。
初始256条开发验证已完成，记录无重复，Fidelity和1/2/3步及自选输出逐图完全一致：
43.39715355634689/.028955043449059303。它是零初始化基线，不是训练后的增益。

已确认真实训练第1/5/10步连续推进，step10于约18:23:36Z写入日志，
损失/参数梯度有限，未见OOM/NaN/NCCL错误。
step10约22.233646298001986秒，峰值18.720479488372803 GiB；
能量修图更新非零，但不代表已达到画质或自验证目标。
仅凭启动和几步训练，不能声称checkpoint保存/恢复已在本轮GPU验证。
第一个完整断点计划step100，第一个训练后开发验证仍为step1706。
暂保留10.5–13普通小时/84–104 allocated GPU-hours规划，后续以实测修正。

日志：logs/foundir_novel_ppt_full_data_stage1_v1_stage1_retry1.log。
输出：checkpoints/foundir_novel_ppt_full_data_stage1_v1/stage1_retry1/。
不要重复启动、改变绑定代码/配方或自动扩训；只监测固定计划。
只做PPT、MoE暂缓的用户决定继续有效。

## ACTIVE: PPT only; MoE deferred until PPT outcome — 2026-10-27

用户报告节点2被不明人员停止，并明确改为一次只推进一个实验。
唯一active research job现在是Novel PPT Approach全数据stage1。
这覆盖下面“两节点并行”、MoE改进优先级及其他旧active表述。
MoE的已完成11000步结果完整保留；不重启旧MoE，不启动新的梯度诊断、
LPIPS .20/.40对照或VLM训练。上一轮提出的MoE两组各2000步仅是建议，未执行。
只有PPT在来源隔离开发集验证有效，或用户明确决定放弃PPT后，才重新讨论MoE；
不得将tmux启动、工程验收或32张训练拟合成功当作PPT已work而自动启动MoE。

PPT stage1_v1日志最后停在加载冻结模型，只有launch.json，没有训练步记录、
validation文件或完整checkpoint/receipt，故没有可恢复的优化状态。
没有异常栈/END标记；停止由用户报告，具体原因和操作者未知，不能归因代码或外部管理员。
保留旧日志与目录不动，不宣称已远程核实旧节点PID。
在用户确认旧任务停止的节点2，用新ID重新从零启动相同冻结配方：
bash smoke/start_foundir_novel_ppt_full_data_stage1_v1_tmux.sh stage1_retry1

单机同型号8×H100/H200，完整54585条/1755来源训练池、固定256开发条目。
仍只到1706步，不加载trainfit/qualification权重，不自动扩到8530。
预计10.5–13普通小时/84–104 allocated GPU-hours，每100步保存完整恢复状态。
这是重新启动命令，不是假称从某训练步续训；Codex没有替用户启动GPU。
源码/config不改，节点锁及空闲显存检查继续拒绝同节点重复占卡。

## TERMINAL: migrated MoE finished; Blur LPIPS still short — 2026-10-27

节点1resume_v1已于17:28:57Z正常结束，status=0，10500→11000共500步。
单机8×H100，launcher1845秒/.5125普通小时/4.1 GPU-hours；
trainer.45572002159471997小时/3.6457601727577598 GPU-hours。
不要重启、延长或把完成的MoE当成仍在跑。节点2PPT全数据阶段授权与配方不变。

固定11000步：Rain100条34.41416323292532/.06439921100391074，仍通过34.2/.08。
Blur2595条26.61952123148184/.18833398326176268，PSNR过25.9，LPIPS未过.18，
还差约.008334。相对原配方26.74785967610281/.2319519135801782，Blur PSNR
下降.1283384446209673、LPIPS改善.04361793031841552。
预先规定的修复信号通过，但联合绝对门槛失败；不是VLM优势或novelty证明。
旧H200→新H100的迁移和中断恢复必须披露，不称严格同节点消融。

41项源码、终局权重/Adam哈希、逐图汇总和完整记录数已核验，未重跑或读取图片。
报告：reports/foundir_moe_blur_lpips_resume_v1_terminal_2026-10-27.md。
result SHA：d11d47c16403e1e70389e33a91ee73de8b3d6d70708660de92c775e986bcdd39。

## ACTIVE: node2 PPT full-data stage1 authorized — 2026-10-27

用户已在节点1启动MoE迁移，并要求节点2的PPT命令。这覆盖下面“第二节点待授权”。
两实验分开节点，各独占单机同型号8×H100/H200；不跨节点DDP，不停止或改写MoE。
本次节点1共享日志已通过通信并完成恢复起点Rain验证，尚不据此宣称续训已完成。

PPT fixed32/300步已结束，不重跑，不加载其权重。新energy从零训练，
使用完整54585条/1755来源Noise池，固定256条source-disjoint开发验证。
模型/loss/三步轨迹/候选/global32/100步warmup/8530步LR跨度全不变。
只运行至原定首个检查点1706步并停止，不自动继续8530；不增加额外方法。
入口：bash smoke/start_foundir_novel_ppt_full_data_stage1_v1_tmux.sh stage1_v1
预算10.5–13普通小时/84–104 allocated GPU-hours，按实际全数据速度更新。

新增独立入口，不改旧绑定源码。每100步保存权重、Adam、8 rank RNG、采样位置、
config/source哈希及原子receipt。续训须确认旧进程已停，在新ID从明确完整receipt恢复，
不得拿日志步数当断点，不重置optimizer/warmup，不从trainfit/qualification恢复。
1706 checkpoint后验证中断也可从其receipt重做固定验证；H100/H200迁移不承诺bit一致。
绝对PSNR/LPIPS门槛和相对Fidelity贡献分开；修图收益与自选收益分开。
不读取official/sealed/internal/parent-test像素。保留历史失败和旧数据协议。

配置SHA：dc7a41787e75731f2034644d58c5c9fdef2744a34f7ccf1e0c1fba723a2fab2b。
19项CPU测试、两个shell语法和真实权重/数据元信息/23项来源绑定预检查通过，零图片读取。
新增恢复功能尚未在GPU实测；八卡模型工程资格沿用原成功配方，不把CPU通过当GPU通过。
报告：reports/foundir_novel_ppt_full_data_stage1_v1_readiness_2026-10-27.md。
本节只授权、记录启动准备，不声称GPU已启动；看共享日志确认。

## ACTIVE: migrate unfinished MoE; PPT trainfit complete — 2026-10-27

用户要求换新节点，并明确单节点8×H100/8×H200都要兼容；不跨节点DDP。
这覆盖此前“整组必须同一实体节点”的限制，但不能抹去迁移对可比性的影响。
只恢复未完成blur_lpips4x：从保存的10500到11000，500步；日志10750不是checkpoint。
恢复权重、Adam、每rank RNG及336000采样位置，loss/LR/data不变。
新增独立resume_v1入口，不改旧绑定代码或旧输出；每100步保存一次以防再中断。
control已经完成，不重跑；新节点记录型号/UUID，最终比较明确跨节点及可能H200→H100，
不得称严格同节点消融或逐bit一致。旧严格比较器保持不变。

确认旧任务停用后，在一台新空闲八卡节点：
OLD_MOE_STOPPED=1 bash smoke/start_foundir_moe_blur_lpips_resume_v1_tmux.sh resume_v1
预计.5–.9小时/4–7.2 allocated GPU-hours。用户认为旧任务已停，但没有远程PID证明；
日志停更不是证明，flag代表用户确认，不能声称Codex杀掉或核实了旧进程。
迁移配置SHA：39afd8616ada4b8a22cac7e41d7453e999ce81f3e5cab161693cf26e8f462f4d。
13项CPU测试、真实权重/Adam/绑定预检查及语法通过；GPU尚未启动。
报告：reports/foundir_moe_blur_lpips_resume_v1_readiness_2026-10-27.md。

PPT fixed32/300步于13:15:42Z正常结束，8×H100，总1.7594444444444444小时/
14.075555555555555 GPU-hours。训练裁块上Fidelity43.393339693546295/.03337287811154965，
energy自选43.93097627162933/.032854294855496846，即+.5376365780830383 dB/
-.000518583256052807 LPIPS。只证明32个反复训练裁块可拟合，不证明泛化。
近质量排序正确率从.5976470588235294到.6023529411764705，未证明自验证有收益。
不要重跑/迁移已完成trainfit，不能拿其权重seed正式，不能自动开1706/8530步。
第二新节点留待PPT全数据阶段获授权；现有PPT入口本已兼容H100/H200。
终局：reports/foundir_novel_ppt_trainfit_v1_terminal_2026-10-27.md。

## MILESTONE: MoE control complete; treatment log stale — 2026-10-27

截至2026-10-27T14:46Z，共享日志证明两组在同一台8×H200上顺序启动。
control已完成11000步、engineering_pass=true；trainer耗.6895869781746943小时/
5.516695825397554 GPU-hours。最终Rain为34.496691444868425/.06531600838992745，
Blur为26.74785967610281/.2319519135801782，Blur LPIPS仍未过.18。

匹配10500步：control Blur26.673934584810702/.22695887966412578，
blur_lpips4x Blur26.58181729682454/.186522385071396。
修改版LPIPS改善.04043649459272977、PSNR下降.09211728798616292，
但PSNR仍过25.9；修改版Rain34.609037982905335/.06519713898887858也通过。
Blur LPIPS距离.18还差.00652238507139602；这是中期收益，不是最终达标。

第二组最后日志10750步、mtime约13:14Z，检查时已超过90分钟未更新。
没有异常栈、OOM/NaN、结束标记、11000检查点或comparison.json。
最后完整可恢复状态为10500。不能说正常运行，也不能断定外部停机或代码故障；
需要原训练节点进程/GPU状态。不得自动杀进程、重启、续训或延长。
本次仅核对日志/JSON/来源哈希，不读取新图片，不改变训练代码。
详见reports/foundir_moe_blur_lpips_pair_v1_status_2026-10-27.md。

## ACTIVE: same-node matched MoE Blur-LPIPS continuation — 2026-10-27

用户批准MoE修复对照，并明确两组必须在同一节点。不要按原配方重跑10k，
不要拆成两个4卡任务或放到两台节点。使用同一台同型号8×H100/H200，
control完成后顺序运行blur_lpips4x，每组独占8卡。
两组均从原image-only formal_v1的step10000恢复权重、Adam、每rank RNG和
样本位置320000，新增1000步到11000；第二组不能继承第一组结果。
唯一科学改动：Blur LPIPS系数.05→.20；Rain仍.05；其他损失、学习率、
数据、采样、架构和image-only router不变。不重置优化器或warmup。

入口：bash smoke/start_foundir_moe_blur_lpips_pair_v1_tmux.sh pair_v1
配置：configs/foundir_moe_blur_lpips_continuation_v1.json
配置SHA：f9840280eabc089567168ab55d7ce8f6431772f6fcb94ddbd208a3ff16449d20。
同节点顺序总预算2–3普通小时/16–24 allocated GPU-hours，依实际速度更新。
固定10500/11000完整评估Rain100和Blur2595，保持原裁块/指标。
只用11000判断，不挑中期最好，不按开发集搜索新权重，不自动扩训。
修改版Blur LPIPS相对control的来源bootstrap上界<0、Blur PSNR>=25.9、
Rain通过34.2/.08才是继续研究信号；Blur LPIPS<=.18仍是独立绝对目标。
不要把改善、工程通过或GPU利用率说成完整达标/VLM贡献/novelty。

启动器全程持有PPT/MoE和Rain/Blur节点锁，核对hostname及实体GPU UUID，
禁止与已运行PPT争卡；不得自动停止、恢复或修改PPT任务。
PPT fixed32/300-step检查的授权与历史不变，旧绑定源文件保持不动。
本轮7项CPU测试及语法通过，真实父权重/Adam/来源/数据元信息预检查通过，
预检查零图片读取；GPU尚未启动，CUDA工程行为仍须实际运行确认。
Codex无GPU执行通道，需要用户在空闲节点输入上述一个命令。
报告：reports/foundir_moe_blur_lpips_continuation_v1_readiness_2026-10-27.md。
继续严格保护数据、保留脏目录、简明中文汇报并focused commit/push。

## ACTIVE: PPT train-only fixed32/300-step learnability check — 2026-10-27

用户明确授权执行“能否学会修图”检查。当前PPT下一步不是8530步正式，
也不是继续25步资格权重：从零energy初始化，固定32训练来源的32个256裁块，
每步各曝光一次，训练300步，第0/100/200/300步检查相同训练裁块。
保留原energy/完整三步/loss/AdamW/100步warmup/8530步LR跨度，
只固定训练数据并缓存冻结输出。不得读取开发或受保护像素；原split元信息排重继续。
此实验只诊断训练拟合能力，不作泛化/论文成绩，权重永不初始化正式。
不自动开启1706或8530步。若学不动先分析目标/优化/容量；若能学再决定全数据阶段。

入口：bash smoke/start_foundir_novel_ppt_trainfit_v1_tmux.sh trainfit_v1
单节点同型号8×H100/H200，预算2.0–2.5普通小时/16–20 allocated GPU-hours，
缓存可能减少耗时，须看实测。旧速度下step100训练部分约36.4分钟，另加启动/检查。
逐项报告Fidelity、1/2/3步、自主选择的MSE/PSNR/LPIPS、梯度方向与改变量、
固定相近候选的真实质量排序准确率（同时报参与对数和step0基线）。
能量下降或都选最后一步不构成画质/自验证证据。10%近质量分组仅用于诊断读数，
不作用于训练/推理，不进行阈值调优；零有效对时结果为null。
5项CPU回归测试、语法与真实来源/权重预检查通过；未执行新GPU任务。
Codex无GPU/远端执行通道，需用户在空闲节点输入命令，不能声称已启动。
配置：configs/foundir_novel_ppt_trainfit_v1.json
SHA-256：dd3a7748f6c821bc870e685c1cd8613ffc113877b64f6a8773f5279a29326cb5。
详情：reports/foundir_novel_ppt_trainfit_v1_readiness_2026-10-27.md。

并行MoE diagnostic_v2已于11:14:22Z正常完成，status=0、64条完整记录，
8×H200耗331秒/.09194444444444444小时/.7355555555555555 GPU-hours。
全部51项来源绑定核验一致；零优化步。不要重跑它或把诊断成功说成MoE性能改善。
解释梯度结果属于下一步分析，尚未确定Blur唯一瓶颈。
终局：reports/foundir_moe_gradient_diagnostic_v2_terminal_2026-10-27.md。
PPT25步资格亦已结束，历史指标/成本不变。新文件隔离新增，旧绑定代码保持不动。
简明中文、保护数据、唯一HDD3运行根和专用Python、focused GitHub tracking规则继续适用。

## ACTIVE: isolated MoE diagnostic V2 retry; PPT qualification complete — 2026-10-27

用户要求监测两任务至终局，并在MoE错误后明确要求修复执行、重新给GPU命令。
MoE diagnostic_v1已于10:54:47Z status=1退出，耗234秒/.065小时/.52 H200 GPU-hours。
首个梯度抽样越界，无完整样本/result.json；零优化步，不能解释为MoE科学失败。
CPU已用真实3072×9216矩阵尺寸复现：FP32 linspace把合法末端28311551
舍入为越界28311552。改为int64整数索引是工程修复，不改变训练方案。

不要重启V1。新增隔离V2诊断和专用启动器，4项回归测试与真实权重预检查通过。
为保护同时运行的PPT源文件绑定，旧V1、共用runtime/config/launchers都保持不变。
在空闲MoE节点运行：
bash smoke/start_foundir_moe_gradient_diagnostic_v2_tmux.sh diagnostic_v2
每任务仍单节点8×H100/H200，预算.25–.75小时/2–6 GPU-hours。
新ID日志与输出不覆盖V1。Codex无GPU执行通道；V2尚未被共享日志确认启动。
详见reports/foundir_moe_gradient_diagnostic_v2_repair_2026-10-27.md。

PPT qualification_v1在8×H100完成25/25步，11:03:13Z status=0，
705秒/.19583333333333333小时/1.5666666666666667 allocated H100 GPU-hours。
32张Fidelity为42.97121322154999/.03313419208279811；最终energy自主选择为
42.97113627195358/.03313359947060235。PSNR略降、LPIPS极微下降，没有双指标收益。
工程通过不等于科学通过；25步/32张不能判定正式8530步/256张方向成败。
配置/19项来源/checkpoint哈希和逐图重汇总均独立核验通过，旧绑定代码未改变。
中位21.83572708698921秒/步，8530步单训练外推51.738542236671655小时/
413.90833789337324 GPU-hours，另需验证等开销。不得自动正式或从资格权重续训。
先明确实测成本；如优化执行效率则重做匹配资格，不能改配方后沿用旧receipt。
终局报告：reports/foundir_novel_ppt_energy_v1_qualification_terminal_2026-10-27.md。
所有历史与保护数据约束继续适用。

## ACTIVE: two retry GPU entry points ready, not launched — 2026-10-27

用户的“execute both”现已落实为两个独立入口。本节覆盖下面“没有新启动器”
以及旧 Rain-only、Generalist-first、energy 永久冻结等过时优先级；保留全部历史失败。
不能把执行准备完成说成训练已启动。当前 Codex 检查为 0 个 CUDA 设备，无远端执行通道。

A. MoE/VLM：运行不更新权重的训练来源诊断，而非再盲训 router。
冻结两个 step10000 模型，各检查16个 Rain 和16个 Blur训练来源。
比较7项损失在代表性共享LoRA/专家输出矩阵上的梯度强弱和方向。
L2范数为所选张量的精确值；余弦为固定坐标抽样，不是完整网络的精确梯度。
零 optimizer steps，不读开发/受保护像素，不据此直接宣布泛化或调好损失。
入口：bash smoke/start_foundir_dual_retry_v1_tmux.sh moe diagnostic_v1
预算 .25–.75 普通小时、2–6 allocated GPU-hours。

B. Novel PPT Approach：冻结强 Noise proposal 和 step8530 residual basis，
训练634032参数的候选相关能量模型；同一能量既排序又通过图像梯度修图。
使用规范化残差坐标、空间能量求和求梯度、detach后的质量标签、完整三步训练，
以及真实错误专家/重复恢复/不足或过度恢复候选，不是七档alpha选择器。
先25步GPU工程验收：32张开发子集，仅查数值、显存、checkpoint和完整推理；
不能用这25步判定方向成败。训练池54585条/1755来源，正式预设8530步、
每1706步验证固定256张。正式预算须由验收速度推算，不能自动启动，
不能从资格权重续训；正式入口要求匹配成功资格receipt，从零energy初始化。
入口：bash smoke/start_foundir_dual_retry_v1_tmux.sh ppt qualification_v1
资格预算 .25–1.0 普通小时、2–8 allocated GPU-hours。

两任务各占一台同型号8×H100/H200；两台空闲节点可并行，一台则顺序。
启动器有节点锁、空闲显存检查、180秒有界NCCL检查和新ID防覆盖。
两入口均已通过真实权重/数据元信息预检查（零图片读取），7项CPU测试及
完整三步能量二阶反传的双CPU-rank同步测试通过。CUDA显存/二阶反传仍待实测。
本轮实际GPU消耗0小时/0 GPU-hours，未产生新的科学结果。

不能把原Fidelity收益算作energy收益。分别报告原proposal、Fidelity、
1/2/3步修图和能量自主选择结果。绝对目标仍为43.424160178005695 PSNR、
LPIPS<=.045362067760033824；还须单独证明相对匹配Fidelity的PSNR提高且LPIPS不差。
最低能量和soft no-harm只是训练/选择机制，不是画质安全保证或novelty证明。
详见 reports/foundir_dual_retry_v1_readiness_2026-10-27.md。
冻结配置 configs/foundir_dual_retry_execution_v1.json：
69db2e2f8a0e2dd3675bb7999968d4109ac4abb708151f909045dfb8578426ef。
不得修改正在运行的绑定代码/配置。保留脏工作树、保护像素禁令、简明中文汇报，
并在启动准备、关键验证和终局结果时focused commit/push。

## ACTIVE: VLM/MoE diagnosis AND Novel PPT Approach retry — 2026-10-27

用户已明确授权同时推进两条路线，先分析实际瓶颈、阅读原 PPT 和旧试验。
本节覆盖下面所有禁止重试、Generalist-first、energy 永久冻结及终局后等待决定的旧优先级。
历史失败和终局数字仍有效；不恢复 EBT Generalist，不覆盖旧实验，不复跑未改的失败配方。

A. 保留导师批准的 VLM→router 主线。两个 MoE formal_v1 已正常结束：
Rain 达标，Blur LPIPS 为普通 .23083898932184047、VLM .2282890860716721，均未到 .18。
固定 Blur 专家也不能修复差距；当前主要证据指向恢复能力/训练目标，而非单纯选错专家。
VLM 是每图五个强度等级，不是 raw tokens；只加在全局 router logits。
下一步先在训练来源上检查 Blur 拟合及各损失的梯度作用，保持 Rain 监测。
不把数据集类别当 VLM 严重度真值，不按开发集选新权重/阈值，不直接合并不兼容旧 Blur LoRA。

B. 重新尝试原 PPT 的完整 candidate-dependent energy：同一 energy 既排序又通过
候选图梯度修图。旧 V2 已实现真实候选和 1/2/3 步梯度训练，效果失败；不能说从未试过。
强 residual basis 的 +.27299337834119797 dB 是 refiner 收益，不是 energy 收益。
最后 pairwise gate 在独立82张上相对 Fidelity 为 -.0007919683689010526 dB；
其 joint_gate_pass=true 只是绝对阈值，不代表选择器贡献。中期小幅正值不可后验选用。

本轮已读完整8页PPT和4张方法图、重汇总既有指标、复核旧 step5118 能量数值。
确认 mean-energy 更新有显式1/(H*W)缩放，且旧局部排序质量标签参与反传。
新隔离组件 restoration_energy_numerics.py 修正这两个数值约定，7项CPU测试通过。
这不证明画质提升或失败唯一原因；旧日志最后20个256-crop记录方向余弦约.1409，
不能仅放大步长。下一次完整 energy 重试须对比强 Fidelity 本身，检查方向/幅度，
真实好坏候选和1/2/3步效果，不再把七选一或普通refiner冒充完整PPT方法。

PPT 已按用户要求命名为 Novel PPT Approach，GitHub 路径：
presentations/novel_ppt_approach/Novel_PPT_Approach_VLM_Restoration_Energy.pptx
从 original_attempt_retry 改名，文件字节与 SHA-256 不变：
cdb8d5e0fe682d6cf30f0ba217d64c426e682fbbc7c70bd51e950ed38edacefb
名称不是经过验证的 novelty 声明；README 保留“曾试过失败、现在重试”的历史。

审计报告：reports/foundir_vlm_and_novel_ppt_retry_audit_2026-10-27.md
审计JSON：reports/foundir_vlm_and_novel_ppt_retry_audit_2026-10-27.json
数值复核：reports/novel_ppt_energy_numerics_audit_2026-10-27.json
本轮新增GPU用量0小时/0 GPU-hours。没有新GPU训练、没有已验收的正式重训命令。
用户授权两条后续实施；新GPU资格/正式训练前仍须冻结设计、来源隔离、预算和验收条件，
使用新ID，不把25步工程测试当科学成败，不承诺必过指标。正式规模依可学习性与实测成本决定。
每任务只用单节点8×H100/H200；若有第二空闲节点可独立并行，不用跨节点DDP。
当前Codex无GPU或远端执行通道；不能把准备完成说成已启动。
所有受保护像素禁令、唯一HDD3工作根/专用Python、简明中文汇报及focused push规则继续适用。


## TERMINAL: BOTH joint MoE formal arms complete; joint quality FAIL — 2026-10-27

This supersedes every earlier running/monitoring/pending statement below.
Both formal_v1 arms finished10000/10000 normally with launcher status0,
engineering_pass=true and no logged nonfinite loss/gradient, OOM or fatal
runtime/NCCL failure. image-only ended03:14:25Z on8xH100; VLM03:41:07Z on8xH200.
The monitoring request is fulfilled. Do not resume, extend, relaunch, retune,
or start another experiment without a new user decision.

Frozen final Rain100: image34.73973430724847/.06654019097797573;
VLM34.742482612955634/.06634078806499019. BOTH pass Rain34.2/.08.
Frozen final FULL Blur2595: image26.655435556096634/.23083898932184047;
VLM26.61442463761589/.2282890860716721. BOTH pass Blur PSNR25.9, but
BOTH FAIL LPIPS<=.18. Do not describe engineering completion as joint success.

The existing final comparator passed matched config/all29 sources/data/input
metrics and recomputed paired rows. Rain VLM benefit is inconclusive (both
source-bootstrap intervals include zero). Blur VLM-minus-image is
-.04101091848074167 dB / -.0025499032501683986 LPIPS: perceptual/fidelity
tradeoff, NOT dominance. This is one seed on different H100/H200 hardware,
not training-seed uncertainty or a hardware-matched causal proof.
At identical VLM weights, switching VLM off changes Rain almost negligibly;
on matched Blur100, VLM on lowers LPIPS but also lowers PSNR. No novelty/SOTA
or VLM-necessity claim is established. This remains Rain/Blur only, not five
types, coupled restoration or raw-token conditioning.

Both final weight/Adam file SHA-256 values independently match their
step10000 resume receipts; next_sample_position320000. Result/metric hashes
and current bound sources/config were checked. No protected pixels or extra
GPU evaluation were accessed in this terminal audit.
Launcher costs: image26910s=7.475h/59.8 H100 GPU-hours;
VLM27969s=7.769166666666667h/62.153333333333336 H200 GPU-hours.
Total121.95333333333333 allocated GPU-hours, excluding prior qualifications.

Report: reports/foundir_joint_moe_v2_terminal_2026-10-27.md
Machine-readable: reports/foundir_joint_moe_v2_final_comparison_2026-10-27.json

## MILESTONE: image-only step6000 — Rain PASS, Blur still short — 2026-10-26

image-only scheduled step6000: Rain10034.3965064204137/.07232339138630778
passes both Rain goals; Blur100 preview25.542105757879234/.29018930681515487
still fails both Blur goals. Both metrics improved since3000, but Blur remains
about.358 dB short and.11019 LPIPS above its ceiling. This is not joint success
or fullBlur2595. VLM step6000 is pending. Full step6000 resume receipt saved,
next_sample_position192000. Continue BOTH frozen10000-step runs to terminal
completion as requested; no retuning, checkpoint selection or extra GPU work.
Report: reports/foundir_joint_moe_v2_step6000_2026-10-26.md.

## MILESTONE: BOTH step3000 Rain PASS, Blur still FAIL — 2026-10-26

Both formal arms completed the scheduled step3000 evaluation.
Rain100 image-only34.32609266908536/.07442104332614691;
VLM34.32999647989338/.0712783659575507: both pass Rain AT THIS CHECKPOINT.
Blur100 preview image-only25.399459516822663/.31455456842668356;
VLM25.38343235445578/.30998902085004376: both still fail Blur.
VLM has lower LPIPS on both tasks, near-equal Rain PSNR and slightly lower
Blur PSNR. This is an encouraging perceptual trend, NOT consistent two-metric
VLM superiority, a final joint PASS, significance, novelty or fullBlur2595.
Both saved full step3000 receipts (next_sample_position96000) and continue.
Preserve fixed10000-step final selection; do not select step3000 or retune.
User requested monitoring BOTH jobs to terminal completion. No new GPU job,
protected-data access or source/config change.
Report: reports/foundir_joint_moe_v2_step3000_2026-10-26.md.

## MILESTONE: both joint MoE formal arms at step1000 — 2026-10-26

The user requested monitoring BOTH formal_v1 jobs until terminal completion.
Continue read-only monitoring; do not stop, restart, retune, change bound code,
auto-launch another experiment or interpret successful execution as quality PASS.

Both first scheduled post-training evaluations are available.
Rain100 image-only: 33.9101104544897/.07389099790481851;
VLM: 33.92674730344711/.07357312987558544.
Blur100 PREVIEW image-only: 24.841265019393163/.35753815472591666;
VLM: 24.776332465206465/.35622713952558116.
VLM is slightly better on Rain, but has lower Blur PSNR and better Blur LPIPS.
No consistent VLM advantage or absolute quality PASS is established.
Both improve Blur versus initialization but worsen Rain. Keep the frozen
10000-step schedule; next quality milestone is3000, then6000 and10000.
Full Blur2595 and final constant-expert/VLM-off diagnostics remain pending.

Both step1000 complete resume receipts are present, next_sample_position32000.
Each binds the same config/source identities. Checkpoint hashes in the report
are trainer-recorded; this milestone audit hashes metric/receipt files only.
Hardware remains8xH100 versus8xH200, with the documented step0 numerical caveat.
Report: reports/foundir_joint_moe_v2_step1000_2026-10-26.md.

## ACTIVE: BOTH formal MoE arms confirmed training — 2026-10-26

Read-only shared-log check at 2026-10-26T20:05:52Z confirms BOTH formal_v1
arms are training, superseding earlier "VLM not started" statements below.
image_only: 480/10000 on 8xH100; vlm_assisted: 60/10000 on 8xH200.
Both advance with finite logged losses/gradients, peak memory about 42.57 GiB,
and no observed traceback/OOM/NCCL fatal or launcher END. These are log-based
observations, not direct remote GPU telemetry. Do not restart or retune.
The VLM launch was at 19:54:58Z; image-only at 19:45:55Z.

Config, all 29 source bindings and data identities match. Inputs/record IDs
and input metrics match exactly. Formal step0 OUTPUTS differ slightly across
H100/H200: VLM-minus-image Rain +.0003037634628881847 dB /
-.000008241008035841935 LPIPS; Blur100 preview +.0009170851552084969 dB /
-.000034258328378222735 LPIPS. Do not claim bitwise-identical formal outputs,
attribute these tiny differences to VLM efficacy, or compare runtime as
hardware-matched. Earlier same-H100 qualification equality remains valid.

image-only step250 weights/Adam/resume receipt are saved; independent file
hashes match the receipt, next_sample_position=8000. No post-training
validation exists yet: first at step1000; full Blur2595 only at step10000.
Keep fixed final selection, frozen recipe and separate single-node jobs.
No protected pixels, new GPU launch or running-code/config edit in this check.
Detailed status/hashes: reports/foundir_joint_moe_v2_parallel_formal_start_2026-10-26.md.


## ACTIVE: independent parallel formal arms on TWO nodes — 2026-10-26

The user reported another free GPU node and requested checking the current
run before starting the VLM contrast there. The image-only job is healthy:
image_only/formal_v1 started 2026-10-26T19:45:55Z on8xH100 at
eight-h200-copy-copy-copy--daf90a2bdaa1-k3xjvzod7q. A read-only audit observed
advancing steps50->70, finite logged losses/gradients, peak42.56962299346924
GiB and no terminal marker. All29 bound source hashes and data identities
match qualification; step0 Rain100/Blur100 metrics exactly match its starting
point. No post-training quality milestone exists yet (first at1000).
Do not stop, restart, retune or edit hash-bound training code/config.

The earlier sequential-only plan was a ONE-NODE resource constraint.
It is now superseded: VLM/formal_v1 may run concurrently on the user's
OTHER free homogeneous8xH100/H200 node. Prefer8xH100 to match the active
hardware; if using8xH200, record the difference and do not call runtime a
hardware-matched comparison. This is TWO independent single-node jobs,
not a16-GPU cross-node DDP job. No master IP, node ranks, nnodes change,
global-batch change or cross-node communication setup is needed.

Keep the same frozen code/config, seed, source/crop order, training data,
losses and10,000-step schedule. Each formal arm independently starts from
original Rain step4000, NEVER from image-only progress or qualification.
Each uses all8 GPUs on its own node. Existing /tmp flock/tmux sockets are
node-local; output/log paths include the arm and therefore do not collide
in the shared workspace. Do not start VLM on the busy image-only node.
No launcher changes are needed or authorized merely for parallel scheduling.

At this check VLM/formal_v1 has no log/output yet. Codex has no GPU/node
execution channel; provide the existing command for the OTHER GPU terminal:
bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh formal vlm_assisted formal_v1
Do not claim VLM started until its own shared log confirms it. No GPU job
was started or current job interrupted by this scheduling/documentation turn.

VLM budget stays5.45-9.06 wall hours /43.63-72.45 allocated GPU-hours.
Parallel completion can be roughly one-arm5.5-9.1-hour scale from the second
launch if throughput is comparable, not the old11-18-hour serial wait.
Total compute for both arms remains87.26-144.89 GPU-hours; parallelism does
not halve compute. Shared-storage contention or hardware differences may
extend wall time. Scientific acceptance/quality claims remain unchanged.

## ACTIVE: paired formal joint MoE phase authorized — 2026-10-26

The user explicitly requested "start now" for the formal phase after BOTH
25-step engineering qualifications passed and the paired budget was reported.
This supersedes all older "formal unauthorized" statements below. The
immediate launch is image_only/formal_v1 for10,000 steps on one homogeneous
8xH100/H200 node; vlm_assisted/formal_v1 is the planned matching second arm,
run sequentially on the same node after the first finishes. Do not start both
tmux jobs together or auto-start a second job from a monitoring turn.

Use the existing frozen runner/starter/config unchanged. Both formal arms
independently initialize from original Rain step4000 with fresh Adam and the
same seed/data/crop order/losses. Do not pass a resume receipt and do not
initialize from either qualification. Formal validates BOTH qualification
receipts and checkpoint hashes before training. The two tasks remain Rain
and Blur, not five-type or coupled restoration. No protected pixels.

Read-only formal launch check completed: the trainer's actual
require_qualifications validator passed both receipts and all four weight/
optimizer file hashes against current config/source/data identities.
Both formal_v1 output/log paths are unused; shell syntax checks passed.
No training/evaluation or GPU launch was performed by this readiness check.

Per arm10,000 steps,320,000 presentations (160,000 per task), with frozen
evaluation at0/1000/3000/6000/10000 and fullBlur2595 only at10000. Keep the
predeclared fixed final checkpoint, no interim selection or retuning.
Checkpoint at250, every1000 and milestones includes weights/Adam/eight-rank
RNG for later same-arm exact resume into a fresh ID if explicitly needed.

Budget from matched8xH100 qualifications: per arm5.45-9.06 wall hours /
43.63-72.45 allocated GPU-hours; both sequentially10.91-18.11 wall hours /
87.26-144.89 GPU-hours. These are estimates, not guaranteed efficacy or speed.
The first post-training milestone at1000 is provisionally35-50 minutes from
launch including startup/evaluation. At this handoff neither formal ID has a
log/output; the Codex shell has no GPU, so user GPU-node execution is required.
Never claim it is running until the new shared launch log confirms it.

First command: bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh formal image_only formal_v1
After the first arm completes, the second command is:
bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh formal vlm_assisted formal_v1
Do not join detached tmux starters with && to imply sequencing; starter exit
only means session creation, not completed training. No new GPU job was
launched by this CPU agent. Preserve all successful qualification receipts.

## TERMINAL: BOTH joint MoE V2 qualifications PASS — 2026-10-26

vlm_assisted/qualification_nvls_v2 completed 25/25 steps on eight H100 80GB
GPUs at 2026-10-26T19:34:33Z, launcher status 0, engineering_pass=true.
Total launcher 278 seconds = .07722222222222222 wall hours /
.6177777777777778 allocated H100 GPU-hours. Peak42.583311557769775 GiB;
median training step1.6632833430194296 seconds. On every rank, both FFN
experts, shared LoRA A/B, image router AND VLM projector updated.
Maximum VLM logit adjustment .05847013369202614; insertion error0.
This proves the VLM path executes/learns, NOT a restoration-quality benefit.

Both arms have identical frozen config, all29 source hashes, data identity,
and byte-identical step0 metric files on Rain100/Blur100. At step25:
Rain VLM33.86719081854514/.07618030223064125 versus image-only
33.86460822453734/.07575711041688919, delta+.0025825940077979226 dB /
+.00042319181375205517 LPIPS. Blur100 PREVIEW VLM
24.269234594175536/.4070244254171848 versus image-only
24.26987270296453/.4072400273568928, delta-.0006381087889941739 dB /
-.0002156019397079878 LPIPS. Tiny mixed differences show no clear VLM
advantage at25 steps. Both lose Rain quality versus initialization; both
improve Blur versus initialization but Blur LPIPS remains worse than input.
No fullBlur2595, absolute quality success, significance or novelty claim.

The user's monitoring request is fulfilled. No job was auto-launched and
no formal training is authorized. Both engineering gates now pass, so the
next USER decision can be paired10,000-step formal arms on the same single
eight-GPU node. Fresh independent initialization from original Rain step4000,
never from either qualification; keep data/seed/recipe/evaluation unchanged.
Using the slower measured1.6934155019698665 s/step and the existing launcher
overhead formula, provisional per-arm wall5.453931949916296-9.055897924874444
hours /43.631455599330366-72.44718339899555 allocated GPU-hours.
Both arms sequentially:10.907863899832591-18.111795849748887 wall hours /
87.26291119866073-144.8943667979911 GPU-hours. This includes estimated
evaluation/save/I-O overhead, is not measured full-run speed or a guarantee.
Both successful qualifications cost1.5911111111111111 allocated H100 GPU-hours.

Result SHA-256:
2f589ddf3fc6a2e666f74107fb85062c802149d57a5d9dc4572a91c6566c9506.
Report: reports/foundir_joint_moe_v2_vlm_qualification_terminal_2026-10-26.md.

## ACTIVE: VLM-assisted joint MoE qualification authorized — 2026-10-26

The user explicitly said execute for the separate VLM-assisted 25-step
engineering qualification. No further approval is required for this exact
qualification; formal training remains unauthorized. Run
vlm_assisted/qualification_nvls_v2 on one homogeneous 8xH100/H200 node
through the existing tmux starter, with its bounded NCCL probe first.
Initialize independently from original Rain step4000 with fresh optimizer;
NEVER initialize from the image-only qualification or any failed MoE.
The same frozen config, all 29 bound source files, data, seed, losses and
evaluation protocol match the completed image-only qualification. No code
or scientific protocol changes were made for this authorization.

At this handoff, no VLM launch log/output exists and the Codex shell has
no CUDA device node. Do not claim it started: the user must execute the GPU
command below, then inspect the shared log. Do not launch on the CPU shell.
Expected wall time is provisionally 10-20 minutes / 1.3333-2.6667 allocated
GPU-hours, based on the matched image-only run's 7.3 minutes; not a measured
VLM duration. Retain the earlier conservative .81-hour / 6.47-GPU-hour
envelope for slow startup/shared I/O. The launcher still prints its original
conservative estimate; do not mutate hash-bound launch code to change it.
Once complete, verify both engineering receipts and compare matched
initial/final metrics; 25-step quality changes do not establish novelty.
Then report BOTH-arm measured formal budget and wait for formal permission.

Command: bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh qualification vlm_assisted qualification_nvls_v2

## TERMINAL: joint MoE V2 image-only qualification PASS — 2026-10-26

image_only/qualification_nvls_v2 finished all 25 steps normally on eight
NVIDIA H100 80GB GPUs at 2026-10-26T19:20:40Z, launcher status 0.
The no-update NCCL/DDP probe passed, followed by engineering_pass=true,
zero insertion error, finite training/checkpoint validation, and observed
updates to both FFN experts, image router and shared LoRA A/B on all eight
ranks. VLM was inactive as intended. Peak memory 42.57999897003174 GiB;
median training step 1.6934155019698665 seconds. Total launcher cost:
438 seconds = .12166666666666667 wall hours / .9733333333333334 allocated
H100 GPU-hours. This validates the repaired path on this H100 node, NOT
the original H200 node or a uniquely proven cause of the earlier CUDA401.

Matched Rain100: step0 34.17715539550872/.07015681236982346 ->
step25 33.86460822453734/.07575711041688919. Rain regresses by
.31254717097137785 dB and .00560029804706573 LPIPS; PSNR fails >=34.2.
Matched Blur100 PREVIEW: 23.8596342447602/.4188850187882781 ->
24.26987270296453/.4072400273568928 (+.41023845820432925 dB,
-.01164499143138531 LPIPS). Blur output LPIPS still exceeds input
.37266220796853305. No full Blur2595 evaluation, no quality gate PASS,
no demonstrated VLM benefit and no novelty conclusion. Do not extend or
retune from this 25-step engineering result.

All 29 bound source hashes, config/receipt/history/metric identities and
weight/optimizer SHA-256 were independently checked. No protected pixels.
Result SHA-256:
249071478a7f68d3d3bc8d812e9736497fee3d19a2ca5d3f60147c0c6ab5d278.
Report: reports/foundir_joint_moe_v2_image_only_qualification_terminal_2026-10-26.md.

The user's monitoring request is fulfilled; this launcher is finished.
Next planned task is a separate fresh vlm_assisted/qualification_nvls_v2
25-step qualification, only on user launch instruction. It does not require
image-only quality to pass first. No VLM/formal job was auto-launched.
Both matching engineering receipts are required before any formal launch
decision and measured budget. Never initialize formal from qualification.
Preserve both the failed qualification_v1 and successful qualification_nvls_v2.

## ACTIVE: authorized joint MoE V2 transport repair / retry — 2026-10-26

The user explicitly approved repairing the failed image-only qualification.
This supersedes the previous wait-for-repair-approval instruction, not the
scientific contract. The actual runner now enforces NCCL_NVLS_ENABLE=0 and
NCCL_P2P_DISABLE=0 inside the tmux job, independent of inherited server values.
Before loading restoration models/data, it runs the existing eight-rank
cure_nccl_rehearsal_v1.py (scalar all-reduce + tiny DDP backward, no optimizer
update or image access) under a 180-second timeout plus 30-second kill grace.
Failure/timeout exits without training; success continues ONLY the requested
25-step qualification. Detailed NCCL logging is scoped to the probe process.
No model, config, dataset, loss, training step count or protected-data scope
changed. Formal and the other arm do not auto-start.

Use a fresh image_only / qualification_nvls_v2 run ID. Never overwrite the
failed qualification_v1 log. Default receipt paths now use qualification_nvls_v2
for both arms; explicit paths remain supported. The VLM arm still requires
a separate launch decision after checking this retry's engineering result.
CPU launcher tests cover forced env values, probe-before-training order,
failure/timeout blocking, training error propagation and existing-log safety.
All five launcher tests plus all 18 original joint-MoE/data CPU tests passed.
Actual H200 transport recovery, memory safety and training remain unverified
until the user's GPU-node retry; the Codex shell has no GPU access.
Budget per qualification remains provisional .20-.75 wall hours / 1.6-6.0
allocated GPU-hours; the new probe adds at most 210 seconds (.058333 hours /
.466667 allocated GPU-hours) if its timeout/kill grace is exhausted.

Command: bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh qualification image_only qualification_nvls_v2
Repair report: reports/foundir_joint_moe_v2_nccl_repair_2026-10-26.md.

## TERMINAL: joint MoE V2 first launch failed at NCCL initialization — 2026-10-26

The image_only / qualification_v1 launcher ran on eight NVIDIA H200 GPUs
from 2026-10-26T18:57:06Z to 18:58:49Z, exit1, 103 seconds
(.02861111111111111 wall hours / .2288888888888889 allocated GPU-hours).
All ranks failed in dist.init_process_group('nccl', device_id=device),
CUDA401 "the operation cannot be performed in the present state".
No optimizer step, model loading, output directory, checkpoint or evaluation
was reached. This is initialization failure, NOT scientific quality failure.
The user requested monitoring through completion; that monitoring is fulfilled.

The current launcher lacks NCCL_NVLS_ENABLE=0 / NCCL_P2P_DISABLE=0,
which the existing 2026-09-26 H200 recovery explicitly needed and verified.
The new failure matches that earlier pattern, but this run has no detailed
NCCL NVLS trace, so do not present the specific transport cause as proven.
Do not infer that the hardware is faulty or recommend changing the model.
The proposed next step is the existing H200 transport compatibility setting,
a no-update communication check, then a fresh-ID retry of the SAME25-step
qualification. No launcher/code change or GPU retry was made during this
monitor-only turn. Await user instruction for the repair/retry; preserve the
failed log and never reuse qualification_v1. Neither formal nor VLM arm starts
automatically. All preceding CPU readiness remains CPU-only.

Report: reports/foundir_joint_moe_v2_image_only_launch_failure_2026-10-26.md.
Failed log SHA-256:
bbee8607da8ed02d3b6befff33c01e0339f772da31b9f6ca703d33521493cfb8.

## ACTIVE: joint image-only versus VLM-guided MoE — 2026-10-26

The user explicitly cancelled the serial prerequisite "new one-step Blur
expert must pass first, then image-only MoE, then VLM". This supersedes ALL
older launch holds/order statements below for this new V2 experiment only.
Do not resume failed Blur-only, frozen-Rain FFN-only MoE, EBT or energy runs.
Original successful Rain/old Blur checkpoints remain untouched.

The task is a matched TWO-TASK Rain/Blur study, not yet five-type all-in-one
or coupled restoration. Both arms jointly train the same full FFN experts,
image routers and ONE shared rank32 LoRA at all 50 original attention/stream
targets. The old restriction "freeze Rain/shared attention, train FFNs only"
is removed. Shared LoRA is not a LoRA expert bank. Both arms initialize from
original Rain step4000, use identical empty text conditioning and one-step
endpoint inference. Do not claim one-step Blur feasibility is already proven.

Only the VLM-assisted arm adds per-image frozen Qwen predicted five-slot
strengths through a 5->256->2 MLP to each image router's logits. This is the
advisor-allowed LABEL/STRENGTH interface, NOT raw VLM tokens. Full prediction
cache is joined strictly by ID/source/LQ+GT hashes; never use
type_oracle_levels_json or GT for inference. Existing VLM prediction
center448 crops differ from random512 restoration crops; disclose this
image-level-condition limitation. Qwen preprocessing cost is additional,
not zero merely because predictions are cached.

Both arms use task-neutral decoded RGB/latent/LPIPS/high-pass supervision.
Remove Rain-specific change-mask weighting, unchanged-region identity, and
Restormer teacher; retain remaining coefficients. Train-only task CE and
250/500 hard-route warmup are identical in both arms, absent at inference.
This bundled repair is not a single-factor causal ablation of old failures.
VLM superiority must be assessed ONLY against its matching repaired baseline.

Contract: configs/foundir_rain_blur_joint_vlm_moe_v2.json
SHA-256: 3bdd59b3fb3c010cc7a93ab299f3f67fe85cc5eb5a6f380f610d4338398baebe.
Data: unchanged 38,334 Rain + 105,057 Blur training records, source-balanced
16+16 per batch32, random512/D4; unchanged Rain100/Blur2595 development
crops with Blur100 previews. No protected/official/sealed/internal pixels.

First run separate fresh25-step engineering qualifications for BOTH arms on
one homogeneous 8xH100/H200 node. Each estimate .20-.75 wall hours /
1.6-6.0 allocated GPU-hours, provisional. No GPU is available in the Codex
shell; no new GPU job has been launched by the agent. No automatic formal
or other-arm launch. Qualification checkpoints must NEVER seed formal.
Image-only scientific-quality failure is no longer a veto on the matched
VLM arm; engineering failure still blocks an unsafe formal launch.

Prepared formal recipe: 10,000 steps per arm, 320,000 presentations total /
160,000 per task (not unique records, not guaranteed complete traversal).
Same seed/crops/order/optimizer/loss/inference across arms. This is half the
old Blur expert's 320,000 presentations per task, not a matched old-model
training-budget claim. Measure BOTH qualifications and report wall hours /
GPU-hours before a user formal launch decision. Require both valid
engineering receipts; never require image-only to pass the SOW first.
Save complete weights/Adam/eight rank RNGs at250, every1000 and milestones;
resume only same-arm incomplete formal into a fresh output ID.
Training-state storage is about117 GB per arm before metrics/images.

Final fixed-step comparison reports per-task PSNR/LPIPS, paired source-level
bootstrap intervals, win fractions, expert use and constant-expert/VLM-off
diagnostics. Preserve Rain >=34.2/<=.08 and Blur >=25.9/<=.18 as separate
absolute goals. A tiny gain over a bad baseline is not paper success.
No broad first-VLM-MoE, novelty-established or SOTA claim is authorized.

CPU readiness: 18 tests passed (including two-rank DDP, full-model meta
parameter audit, checkpointed gradients and exact resume). Real-asset
preflight passed; all146,086 allowed records have hash-matched predictions.
No GPU qualification or new scientific result exists yet.
Readiness/commands:
reports/foundir_rain_blur_joint_vlm_moe_v2_readiness_2026-10-26.md.
Launcher:
bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh qualification image_only qualification_nvls_v2
Run the VLM qualification separately after checking the first engineering
result; no standalone Blur quality gate in between.

## TERMINAL: Blur-only formal complete — quality gate FAIL — 2026-10-26

This supersedes every older LIVE/ACTIVE/unlaunched Blur-only status below.
formal_v1 completed all 3,000 steps and full Blur2595 evaluation at
2026-10-26T16:49:10Z, launcher exit 0. Engineering PASS, joint quality
gate FAIL. PSNR 26.096176888792407 passes >=25.9, but LPIPS
0.3260975599297016 fails <=.18 by 0.14609755992970158.
Old matched MoE is 26.10596660902048/0.3271988157297777:
single-task changes only -0.009789720228074117 dB /
-0.0011012558000761397 LPIPS. Removing routing/mixed training did
not materially solve this Blur branch. Do not claim a working MoE or novelty.

Input is 25.849544629199475/0.29110622298011024; LPIPS is worse on
2280/2595 records. Final matched Blur100 preview is
25.067034539869926/0.37915658176876604; never compare the
full2595 final mean directly against earlier100-record preview means.
This control does not prove one-step FLUX is impossible or isolate a specific
loss/prompt/update-scope cause. Original Rain/old Blur remain untouched.

Launcher 3371 seconds = 0.9363888888888889 wall hours /
7.4911111111111115 allocated H100 GPU-hours. Median step
0.8779867839766666 seconds; rank0 peak 29.185157299041748 GiB.
48,000 Blur presentations, 12 complete checkpoints. Source/config/metric
identities and final artifact hashes verified. All 424,673,280 parameters and
10 Adam states independently verified finite FP32, Adam step 3,000, next
sample position 48,000, eight rank RNG states. No protected pixels or runtime error.

The user's monitoring request is fulfilled. No job is active from this
launcher. Do not auto-extend, restart, tune, add VLM, or launch another
experiment. Preserve all results and wait for the user's next decision.
Report: `reports/foundir_blur_single_endpoint_formal_terminal_2026-10-26.md`.
Result SHA-256:
`6224638f9b2a65f15f00839f338eb7cb0782b5c4ba028de16af41bad7bb2d1b9`.

## MILESTONE: Blur-only formal step 2,000 — 2026-10-26

formal_v1 remains LIVE beyond 2,000. Matched Blur100:
24.91918972080556/0.38884398228721695. Relative to step1,000:
+0.06952350409890329 dB / -0.0028100751596502604 LPIPS.
Old MoE at step2,000 is 24.823461629491018/0.38983108070679007;
single-task advantage is only +0.09572809131454108 dB /
-0.000987098419573118 LPIPS. Output LPIPS remains worse than input
0.37266220796853305. No full Blur gate or solved-branch claim.
Continue frozen final 1,000 steps and full2595 evaluation, no extension/retune.

Source/config/record/metric identities and weight/Adam hashes verified.
Checkpoint saved 2026-10-26T16:28:57.101968+00:00; next sample position32,000.
Cost to checkpoint: 0.5994727688888889 wall hours /
4.795782151111111 allocated GPU-hours, not terminal cost.
Report: `reports/foundir_blur_single_endpoint_formal_progress_2026-10-26.md`.

## MILESTONE: Blur-only formal step 1,000 — 2026-10-26

formal_v1 remains LIVE beyond 1,000; continue unchanged to 3,000 and full
Blur2595 evaluation. On the matched Blur100 preview, single expert is
24.849666216706655/0.3916540574468672, versus old MoE at the same step
24.8058798027948/0.3938900137133896.
Only +0.04378641391185667 dB / -0.0022359562665223986 LPIPS.
Output LPIPS remains worse than input 0.37266220796853305.
Do not call this a solved Blur branch, a working MoE, or a full-development
gate pass. No restart, extra tuning, VLM, or new experiment is authorized.

Checkpoint/Adam hashes, source/config identities, record identities and
metric means verified; checkpoint step1,000, next Blur position16,000.
Checkpoint saved at 2026-10-26T16:11:34.685216+00:00;
0.30991256 wall hours / 2.47930048
allocated GPU-hours since launch, not terminal cost.
Report: `reports/foundir_blur_single_endpoint_formal_progress_2026-10-26.md`.

## LIVE: Blur-only formal training started — 2026-10-26

The user launched `formal_v1` at 2026-10-26T15:52:59Z on one 8xH100 node
and asked to monitor through successful completion. Initial Rain100/Blur100
parity passed with exact original metrics; finite optimization is observed
beyond step 30. This supersedes the prior authorized-but-unlaunched status.
Do not duplicate, restart, retune or stop the frozen run without user direction.
Continue through 3,000 steps, the full 2,595-record Blur evaluation and launcher
END status. Normal execution is not proof of the PSNR/LPIPS quality gate.

Session: `blur_single_formal_v1`.
Log: `logs/foundir_blur_single_endpoint_v1_formal_v1.log`.
Output: `checkpoints/foundir_blur_single_endpoint_v1/formal_v1/`.
Same original Rain initialization, no qualification continuation, no router/VLM.
Save every 250 steps; evaluate at 1,000/2,000/3,000 as frozen. Preserve all
artifacts. Codex sees shared files from its CPU-only node, not GPU telemetry.
Estimate remains 1.11-2.14 wall hours / 8.88-17.12 allocated H100 GPU-hours,
to be updated from measured progress. No extension or next experiment is authorized.

## AUTHORIZED: fresh Blur-only formal launch — 2026-10-26

After qualification passed, the user requested the official formal launch
command. The frozen 3,000-step Blur-only formal is now authorized under run ID
`formal_v1`. Start from the original successful Rain initialization, NOT the
25-step qualification weights. No router/VLM, loss change or schedule change.
Qualification/source/config identities match; formal output/log were absent
at prelaunch check. The agent has NOT launched a GPU job; wait for the user's
node launch and verify actual logs before marking it live.
Command: `bash smoke/start_foundir_blur_single_endpoint_v1_tmux.sh formal formal_v1`.
Expected 1.11-2.14 wall hours / 8.88-17.12 allocated GPU-hours on 8xH100;
H200 is supported but its speed is not measured by this qualification.

## MILESTONE: Blur-only qualification PASS; formal not launched — 2026-10-26

qualification_v1 completed 25/25 steps on one 8xH100 at 2026-10-26T15:44:10Z,
launcher exit 0. This supersedes the older GPU-qualification-pending status.
Initial Rain100 and Blur100 per-record PSNR/LPIPS parity errors are exactly 0.
Five FFN blocks updated, finite training, no NaN/OOM/NCCL/runtime failure.
Final Blur100 preview: 24.17730141751229/0.39682197285816073,
versus initialization 23.788320945242468/0.42077609404921534;
deltas +0.3889804722698216 dB / -0.02395412119105461 LPIPS.
PSNR improves on 50/100 and LPIPS on 93/100. Engineering PASS only:
no full Blur2595 evaluation and no formal scientific success yet.

Launcher: 331 seconds = 0.09194444444444444 wall hours /
0.7355555555555555 allocated H100 GPU-hours.
Median step 0.9119640614371747 seconds; rank0 peak
29.184199810028076 GiB. Formal estimate now 1.11-2.14 wall hours /
8.88-17.12 GPU-hours, including provisional full-eval/checkpoint IO.
Source/config/metric identities and saved artifact hashes verified.
All 424,673,280 parameters and 10 Adam states independently verified finite
FP32; Adam step 25, next sample position 400, eight saved rank RNG states.
The user's monitoring request is fulfilled. No automatic formal or VLM.
Formal must start fresh from original successful Rain, never qualification.
Keep the frozen 3,000-step/48,000-Blur exposure protocol and save every 250.
Report: `reports/foundir_blur_single_endpoint_qualification_2026-10-26.md`.
Result SHA-256:
`d1d5e1ca2ae616b3c9ea9473381b125ca79cb37b23b9f95f44e40bbe9f4110f0`.

## ACTIVE: Blur-only one-step diagnostic, no router — 2026-10-26

This supersedes older active/hold instructions for the immediate next task.
After the failed joint Rain+Blur gate and weak fixed-expert Blur result, the
user authorized a standalone Blur branch diagnostic. Implement/qualify ONLY
this control; do not resume the failed MoE, EBT/energy, add VLM or alter the
successful original Rain/old Blur experts.

Use `configs/foundir_blur_single_endpoint_v1.json`, SHA-256
`a273b55ccfd9ca128b976ea5a6fa314ce862d77273a4127f0e5836e58957eefb`.
Same original Rain step4,000 initialization, five full image FFNs with ONE
expert per block and no router (424,673,280 trainable FP32 parameters).
Shared model/Rain LoRA stay frozen. Keep the same fixed Rain prompt, one-step
t=1/zero-noise FP32 endpoint, losses and LR1e-5; teacher weight remains zero.
Train Blur only. Batch16 (8 ranks x 2 accumulation), 3,000 steps =48,000 Blur
presentations. Exact Blur record/crop/augmentation positions match the old
MoE's 16-Blur half of each batch. Scale mean Blur loss by .5 to preserve its
original mixed-objective contribution; do not silently renormalize it.

This is a no-routing/single-task diagnostic, NOT a parameter-matched or
pure-router causal ablation. A failure would not prove one-step FLUX is
impossible; frozen Rain conditioning, update scope and objective remain
possible limitations. No new scientific improvement is available yet.

First run fresh25-step `qualification_v1` on one homogeneous 8xH100/H200.
Require initial per-record Rain100 AND Blur100 parity, finite gradients,
actual FFN updates, FP32 weight/Adam saves and valid matched evaluation.
No automatic formal. Formal must restart original Rain, never qualification.
Formal evaluates at0/1000/2000/3000; fullBlur2595 only at final. Target
>=25.9 PSNR / <=.18 LPIPS, with per-record regressions reported separately.
Save complete state every250 steps. Resume requires a finished hash-bound
`.resume.json`, formal mode, unchanged code/config, Adam and absolute
sampling state, and a fresh output ID. Never overwrite a prior run.

Qualification estimate .15-.50 wall hours /1.2-4.0 allocated GPU-hours.
Formal provisional1.0-2.5 hours /8-20 GPU-hours, to be re-estimated from
qualification throughput and checkpoint/evaluation IO. No protected data.
Agent runtime is CPU-only; GPU qualification remains pending user launch.
Readiness checks PASSED: 28 CPU tests including two-rank DDP, exact Blur
exposure/crop matching and uninterrupted-versus-resumed state equality;
Python/shell/JSON syntax and full real-asset read-only preflight passed.
These are not GPU memory qualification or scientific quality results.
Readiness: `reports/foundir_blur_single_endpoint_readiness_2026-10-26.md`.
Launch: `bash smoke/start_foundir_blur_single_endpoint_v1_tmux.sh qualification qualification_v1`.
Continue Mandarin-only, plain-language reporting and focused GitHub tracking.

## TERMINAL: image-only Rain/Blur MoE formal — joint gate FAIL — 2026-10-26

This supersedes every older LIVE/ACTIVE/unlaunched MoE status below.
formal_restart_v1 completed 3,000/3,000 steps plus full evaluation and
fixed-expert diagnostics at 2026-10-26T13:30:15Z, launcher exit 0.
Engineering pass, but JOINT SCIENTIFIC GATE FAIL. No new run is active
from this launcher; do not automatically extend, restart, retune, run dense,
or add VLM. User requested monitoring through completion, which is fulfilled.

Final Rain100: 34.36398316275458/.07267702806275338, both >=34.2/<=.08 pass.
Final full Blur2595: 26.10596660902048/.3271988157297777, PSNR>=25.9
passes but LPIPS<=.18 FAILS by .14719881572977772. Do not call the two-task
baseline successful, novelty established or SOTA. Blur input itself is
25.849544629199475/.29110622298011024: output improves PSNR only
+.2564219798210061 dB and WORSENS LPIPS by .036092592749667474.
LPIPS worsens vs input on 2,324/2,595 Blur records.

Same final Blur100 preview reaggregated from full output:
24.966814240094454/.3831212885398418. Fixed expert1 on those same records
is 25.041084236119197/.38290703640086576, only slightly better and still
poor. Do not blame the main gap solely on the router. Explicit VLM remains
deferred until a genuinely working image-only baseline. Next proposed work,
if requested, is an evidence-based Blur quality audit, not an automatic
architecture/training change. Preserve original successful Rain/old Blur.

Launcher cost: 6,847 seconds = 1.9019444444444444 wall hours /
15.215555555555556 allocated H100 GPU-hours on 8xH100, excluding prior
interrupted run/qualification. Trainer-only 1.8740228895263862 hours /
14.99218311621109 GPU-hours. Median step 2.001327611040324 seconds;
rank-0 peak allocated memory 37.40372896194458 GiB.
96,000 presentations; unique Rain27,228 / Blur30,808. No protected pixels.

All source/config/checkpoint hashes and metric aggregates verified.
Final 50 Adam states are finite FP32, step=3,000, next_sample_position=96,000.
Initial Rain per-record equality remained exact. No NaN/OOM/NCCL/runtime
error. Do not confuse successful completion with scientific gate success.
Report: `reports/foundir_rain_blur_image_moe_formal_terminal_2026-10-26.md`.
Machine-readable: `reports/foundir_rain_blur_image_moe_formal_terminal_2026-10-26.json`.
Result SHA-256: `e13f70ca2169f0f1d0bb584764ab3176ecad16700c96a40d3eac5c1b08e42254`.

## MILESTONE: image-only MoE formal restart step 2,000 — 2026-10-26

formal_restart_v1 remains LIVE beyond 2,000. Rain100:
34.350766710211964/.07022410760167987, both gates pass, improves on 1,000.
Blur100 preview: 24.823461629491018/.38983108070679007, only
+.017581826696218883 dB / -.004058933006599541 LPIPS versus 1,000.
Blur progress has slowed markedly; no evidence warrants promising a joint
pass after the final 1,000 steps. Preview is still NOT full Blur evaluation.
Weights/Adam hashes, FP32 weight header, record identities and metric means
verified. Save time 12:48:41 UTC, cumulative 1.209202438888889 wall hours /
9.673619511111111 allocated H100 GPU-hours since restart (not terminal cost).
Continue unchanged through 3,000 and full Blur evaluation. No extension,
retuning, dense arm or VLM authorized.
Report: `reports/foundir_rain_blur_image_moe_formal_progress_2026-10-26.md`.

## MILESTONE: image-only MoE formal restart step 1,000 — 2026-10-26

formal_restart_v1 remains LIVE and has passed its first saved milestone.
Rain100: 34.238711513652106/.07147581973578782, both >=34.2/<=.08 gates pass.
Relative to original Rain initialization: +.011278978157413633 dB,
-.0022033531684428342 LPIPS. Blur100 preview:
24.8058798027948/.3938900137133896, improved but far from target, especially
LPIPS; this is NOT the full 2,595-record Blur gate. No two-task success claim.
Task-average router weights now clearly distinguish Rain/expert0 and
Blur/expert1; do not claim causal routing superiority without controls.
Weights/Adam saved and SHA-verified, FP32 weight header and metric identities/
aggregates verified. Save time 12:14:14 UTC, about .635 wall hours /
5.08 allocated H100 GPU-hours since restart launch (not terminal cost).
Continue unchanged to 2,000 and 3,000; no automatic other arm/VLM.
Report: `reports/foundir_rain_blur_image_moe_formal_progress_2026-10-26.md`.

## ACTIVE: Rain-initialized image-only Rain/Blur MoE — 2026-10-26

The user explicitly approved executing the plain image-only MoE plan after
the Rain gate passed. This supersedes the prior Rain-before-MoE hold and the
failed raw-FLUX one-MLP pilot. The only new task is Rain+Blur ordinary DiT-MoE;
no VLM branch, LoRA expert bank, EBT/energy, pixel refiner, extra task or mixed
degradation experiment. Preserve the successful Rain and existing Blur experts.

Contract: `configs/foundir_rain_blur_image_moe_v1.json`, SHA-256
`8988e2374ad736f3e406945f1d9db83ae14587d0aba28a04dbf9cdc900ab5f12`.
Start from the hash-bound successful Rain step-4,000 LoRA on FLUX.2 Klein 4B.
Keep that single Rain LoRA and shared backbone frozen. Replace image MLPs in
all five double-stream blocks with two ordinary full FFN clones per layer;
learn only those FFNs and image routers (853,313,290 FP32 parameters).
There is no second external MoE and no bank of LoRA experts. The dense
control trains one FFN per same block (424,673,280 parameters); disclose the
capacity/compute difference, never call it a parameter-matched causal test.

Both tasks use the SAME fixed cached Rain text context and SAME one-step,
t=1, zero-noise FP32 endpoint. The fixed prompt preserves Rain initialization;
it does not expose each image's task label. No runtime Qwen/VLM.
Do not load the old 50-step Blur LoRA into this one-step path or retrain that
old expert. The NEW MoE must learn Blur from allowed paired GT. Keep the
existing eight Rain loss terms/coefficients, with the auxiliary Restormer
teacher enabled only on Rain; its weight is zero on Blur.

Use 38,334 Rain / 105,057 Blur training records, balanced 16+16 per global
batch32, source-balanced within each task, aligned random 512 crops and D4.
Cross-task development source groups and GT hashes are excluded; the audited
pools have zero such overlaps. Keep the exact 100 Rain development records
and all 2,595 original Blur development crops. A deterministic 100-source
Blur preview is for intermediate diagnostics only, never the full Blur gate.
No official/sealed/internal/parent-test pixels.

Run ONLY a fresh 25-step `moe/qualification_v1` first on one homogeneous
8xH100/H200 node. Before updates, replay all 100 Rain records and require
per-record agreement with the successful checkpoint (PSNR error <=1e-4,
LPIPS error <=1e-5) AND preserve its 34.2/.08 gate. Then exercise hard training
task routes, transition, and image-only routing (qualification hard6/end12).
Task labels never enter validation/inference; fixed-expert diagnostics do
not use labels either. Require finite gradients, actual expert/router
updates, memory safety, F32 model/Adam files and matched evaluation artifacts.

Formal is configured at 3,000 fresh optimizer steps, 96,000 presentations
(48,000 per task, NOT full traversal), with hard warmup250/transition-end500.
Validate at 0/1000/2000/3000, using full Blur only at formal final. Formal
requires a passing SAME-ARM qualification and never seeds from it. The dense
arm needs its own qualification. No automatic formal/other-arm/VLM launch.
Do not claim a working MoE merely from numeric gates: inspect expert usage
and constant-expert ablations, and compare against the matched-data dense arm.
Rain gate is >=34.2/<=.08; full Blur average gate >=25.9/<=.18, with tail
regressions reported separately as in the historical Blur result.

24 CPU tests, a separate two-rank CPU DDP test and real-file preflight passed.
The user then launched `moe/qualification_v1` on 8xH100. It TERMINALLY
COMPLETED at 2026-10-26T10:41:54Z, launcher exit 0, 25/25 steps, engineering
pass. Initial Rain per-record PSNR/LPIPS errors are both exactly zero.
Both expert branches and routers updated; finite training, FP32 checkpoint
and optimizer artifacts, no OOM/runtime failure. Recorded rank-0 peak memory
37.39800262451172 GiB; median step 1.7897918989765458 seconds.
Launcher: 484 seconds, .13444444444444445 wall hours,
1.0755555555555556 allocated H100 GPU-hours. Trainer-only timer:
.10672626883945971 hours / .8538101507156777 GPU-hours.

Do NOT describe this engineering pass as a working two-task MoE or gate pass.
Rain fell from 34.22743253549469/.07367917290423065 to
33.77674230661362/.07843095208052546 (PSNR gate FAIL, LPIPS pass).
Blur preview improved from 23.788320945242468/.42077609404921534 to
24.169359410868655/.4072999455779791; this is only 100 records, NOT the full
Blur gate. Fixed-expert diagnostics do not establish successful task
specialization, and task-average router weights remain close to half/half.
Preserve the successful original Rain checkpoint and the qualification.
Never resume qualification as formal or repeat its used run ID.

The user explicitly AUTHORIZED the fresh 3,000-step MoE formal on 2026-10-26
after reviewing qualification. Use run ID `formal_v1`, from the ORIGINAL
successful Rain, with unchanged 250/500 warmup and frozen protocol. Never
initialize from qualification weights. Qualification receipt/source/config
identities match and formal_v1 output/log were absent at launch handoff.
The user launched formal_v1 at 2026-10-26T10:59:57Z on one 8xH100 node.
GPU/data preflight and initial 100-record Rain parity PASSED (both maximum
per-record errors 0). Formal step 1 completed with finite loss/gradient.
INTERRUPTION UPDATE: the last formal_v1 log is step 810, last modified
2026-10-26T11:30:30Z, with no launcher END or traceback. The user reports
possibly stopping the node/job accidentally and asks to continue. This does
NOT prove an internal training error or that all old GPU processes are dead.
No step checkpoint or Adam file exists: first save was at 1,000. Only receipt,
initial parity and step-0 metrics/images exist. Therefore there is NO saved
state from which to resume step 810, and no trained development result.

Preserve formal_v1. Once the user confirms the old job/node is stopped, use
fresh run ID `formal_restart_v1` with the SAME frozen 3,000-step recipe and
original successful Rain initialization:
`bash smoke/start_foundir_rain_blur_image_moe_v1_tmux.sh formal moe formal_restart_v1`.
This is a FROM-ZERO FORMAL RESTART, not optimizer/sample-position resume.
Do not reuse the qualification weights or overwrite the old run. The launcher
lock prevents same-node overlap only; never infer cross-node exclusion.
The user launched formal_restart_v1 at 2026-10-26T11:36:08Z on 8xH100.
GPU/data preflight and exact initial Rain parity passed; step 1 completed
with finite loss/gradient. This RESTART is now LIVE. User requested monitoring
until terminal completion. Do not duplicate/restart/retune/interrupt it.
New session: `rain_blur_moe_formal_restart_v1`.
New log: `logs/foundir_rain_blur_image_moe_v1_moe_formal_restart_v1.log`.
New output: `checkpoints/foundir_rain_blur_image_moe_v1/moe/formal_restart_v1/`.
The old formal_v1 paths below describe preserved interrupted history only.
Preserve the unchanged 1,000/2,000/3,000 evaluations; normal execution does not
establish a scientific gate pass. Codex observes shared artifacts from its
CPU-only node, not direct GPU process telemetry.
Session: `rain_blur_moe_formal_v1`; log:
`logs/foundir_rain_blur_image_moe_v1_moe_formal_v1.log`.
Output: `checkpoints/foundir_rain_blur_image_moe_v1/moe/formal_v1/`.
No dense/VLM run is authorized by this launch. Qualification's 6/12 warmup
is only an engineering stress test, not a reason to retune formal.
Measured provisional formal estimate: 1.94–3.24 wall hours / 15.53–25.90
allocated H100 GPU-hours; full Blur evaluation and filesystem IO add uncertainty.
Result SHA-256: `5c2d2b8cdcfc8177c58d938bb87e0e1c76eb4cc81a7ec4c6cae66a927215c071`.
Terminal report: `reports/foundir_rain_blur_image_moe_qualification_2026-10-26.md`.
Readiness: `reports/foundir_rain_blur_image_moe_readiness_2026-10-26.md`.

## TERMINAL PASS: Rain FLUX continuation completed — 2026-10-25

This supersedes every earlier active/live/failed/unstarted Rain status below.
`continue_v1` completed the authorized 3,000→4,000 continuation on 8xH100:
2026-10-25T23:20:16Z to 23:51:52Z, launcher exit 0, no runtime/OOM/nonfinite/
NCCL failure. Final fixed 100-record development PSNR/LPIPS is
34.22743253549469/.07367917290423065: BOTH >=34.2 / <=.08 gates PASS.
PSNR margin is .027432535494689603 dB, LPIPS margin .006320827095769352.
Compared with step 3,000, PSNR gains .21904474864805223 dB; LPIPS worsens
slightly by .0002468797005713008. Do not call this improvement in both means.
The train probe is 34.171933008547796/.05955105775501579 and is NOT the gate.
This passes the frozen development protocol, not an independent official test,
a novelty claim or a SOTA claim. Near-clean over-restoration remains: 14/24
records with input PSNR >=34 lose PSNR versus the input.

The successful mergeable Rain expert is:
`checkpoints/foundir_rain_flux2_precision_resume_v1/continue_v1/step_004000.safetensors`
SHA-256 `020113ce209728452917d49859da7be220f638f0d27f6f87f182ffdab0a8fc2e`.
Inference remains FLUX LoRA only; Restormer was a frozen training-only teacher.
Keep the expert and all historical checkpoints; do not continue training or
change the successful recipe automatically. The Rain prerequisite for the next
image-only DiT-MoE stage is now cleared. Only after a working image-only MoE,
add VLM token routing; no MoE job has been launched by this completion.

Final LoRA/Adam tensors, Adam step=4000, next_sample_position=128000,
configuration/source identities, metric means and matched record/input
identities were independently verified. No protected pixels. Additional
exposure is 32,000 presentations / 17,181 unique records / 5,780 sources;
cumulative presentations 128,000. Total launcher cost is 31m36s,
.5266666666666666 wall hours / 4.213333333333333 allocated H100 GPU-hours.
Trainer-only clock is .5022755164518539 hours / 4.018204131614831 GPU-hours.
Terminal evidence: `reports/foundir_rain_precision_resume_progress_2026-10-25.md`
and `reports/foundir_rain_precision_resume_terminal_2026-10-25.json`.
No Rain training remains active; do not mistake the older LIVE sections for
current instructions.

## MILESTONE: Rain continuation step 3,500 — 2026-10-25

Development is 34.0641582441431/.07644298265688121 on the identical 100
records. Relative to step 3,000: PSNR +.05577045729646102 dB, LPIPS worsens
by .0030106894532218598. PSNR still misses 34.2 by .1358417558569016 dB;
LPIPS remains below .08. This is a modest PSNR improvement, NOT a joint gate
pass or evidence of uniformly improving perception. Per-image wins are
57/100 PSNR, 35/100 LPIPS, 28/100 both. Train probe is
33.95975334657288/.06225986051140353 (not generalization evidence).
Saved LoRA is 100 F32 tensors; all 100 Adam states remain F32/finite with
step=3500 and next_sample_position=112000. Source hashes remain unchanged,
and aggregate means/record identities match the frozen evaluation files.
Training resumed normally beyond 3,500. Continue only to the authorized
4,000-step terminal milestone without retuning or early termination.

## LIVE: Rain step-3,000 continuation running — 2026-10-25

The user launched `continue_v1` on 8xH100 at 2026-10-25T23:20:16Z and
requested monitoring through terminal completion. GPU preflight passed.
The log explicitly confirms LoRA step=3000, Adam step=3000, exact optimizer
state restoration, first sample position=96000 and LR=1e-5. Step 3,001 and
subsequent steps have finite loss/gradients and real A/B updates. Do not
restart, duplicate, interrupt, retune or advance to MoE. Continue unchanged
through the frozen 3,500/4,000 validations; scientific success still requires
development PSNR >=34.2 and LPIPS <=.08, not merely normal process exit.
Early steady steps are ~1.5 s with ~24.86 GiB peak allocated memory per rank.
The earlier "not launched" wording below is now historical.
Progress: `reports/foundir_rain_precision_resume_progress_2026-10-25.md`.

## ACTIVE: true Rain continuation 3,000 → 4,000 — 2026-10-25

The user explicitly authorized exactly 1,000 more optimizer steps from the
completed full-data `formal_v1` step 3,000. This supersedes the older
"no continuation authorized" status, but not the frozen Rain-before-MoE gate.
Use `configs/foundir_rain_flux2_precision_resume_v1.json` and the new
`smoke/train_foundir_rain_flux2_precision_resume_v1_ddp.py`; do not edit or
restart the completed parent trainer/config, and never start with fresh Adam.

Restore the hash-bound step-3,000 FP32 LoRA AND all 100 Adam moment states /
step counters. Preserve LR 1e-5, the eight loss terms/coefficients, batch32,
512 crops, source-balanced schedule, augmentation, teacher and evaluation.
Resume absolute sample positions 96,000 through 127,999: exactly 32,000 new
presentations, cumulative 128,000. Validate/save only at global steps 3,500
and 4,000 on the same 100 development records and 32 training-probe records.
No protected data, new architecture, pixel refiner, EBT or energy experiment.
Do not automatically extend beyond 4,000 or advance to MoE.

The resume code has passed real LoRA/Adam/source/data preflight, CPU exact
state restoration and interrupted-versus-uninterrupted update tests. Old
checkpoints do not contain global RNG states: data sampling/crop/D4 are
stateless by absolute position, AE encoding uses its mean, anchor noise and
dropout are zero. No claim of bitwise uninterrupted GPU equivalence.
The original files/artifacts remain unchanged. New output namespace:
`checkpoints/foundir_rain_flux2_precision_resume_v1/continue_v1/`.
Log: `logs/foundir_rain_flux2_precision_resume_v1_continue_v1.log`.

Launch on ONE homogeneous 8xH100/H200 node:
`bash smoke/start_foundir_rain_flux2_precision_resume_v1_tmux.sh continue_v1`.
Codex has no GPU; no GPU continuation has been launched here. Session creation
alone is not success: verify `RAIN_RESUME_STATE_RESTORED` and step 3,001.
Estimate 40–50 normal wall minutes / 5.3–6.7 allocated GPU-hours on 8xH100;
replace with observed throughput after launch. Current baseline remains
34.00838778684664/.07343229320365935, PSNR gap .19161221315336263 dB.
Do not promise passing 34.2/.08. Readiness:
`reports/foundir_rain_precision_resume_readiness_2026-10-25.md`.

## TERMINAL: full-data Rain formal completed — 2026-10-25

This supersedes every earlier LIVE/unstarted status for this run.
`formal_v1` completed 3,000/3,000 steps on 8xH100 at
2026-10-25T22:55:59Z, launcher status 0. Final/best development is
34.00838778684664/.07343229320365935; training probe is
33.984372473834554/.060009161999914795. LPIPS passes .08; PSNR remains
.19161221315336263 dB below 34.2. Thus engineering completion succeeds but
the joint Rain gate is NOT passed. Do not advance image-only MoE/VLM routing.
Relative to matched step0, gains are +1.9236860265013505 dB and
-.015389716671779755 LPIPS. The final 1,000 steps add .384159465303874 dB;
do not promise an unrun extension will pass. No automatic continuation.
Actual exposure: 96,000 presentations / 36,156 unique records / 5,780 sources.
Total launcher cost: 1h36m33s, 1.6091666666666666 wall hours and
12.873333333333333 allocated H100 GPU-hours. All final artifacts, F32 tensor
headers, source/config identities and aggregate metric means were verified.
Final LoRA: `checkpoints/foundir_rain_flux2_precision_full_v1/formal_v1/step_003000.safetensors`,
SHA-256 `336d412299cd7cd9ff919d8983b2b4c435105161496510c0cec8748785c15904`.
No next run is active. Any extension/recipe change requires a fresh user
decision and frozen protocol; preserve all prior artifacts.
Complete report: `reports/foundir_rain_precision_full_formal_progress_2026-10-25.md`.

## LIVE: full-data Rain formal started — 2026-10-25

`formal_v1` started at 2026-10-25T21:19:26Z on 8xH100 and logged finite
training steps after matching the qualification contract. User requests
monitoring until terminal completion. Do not restart, duplicate, retune,
interrupt, or advance to MoE. Frozen schedule: 3,000 steps, evaluations at
0/250/500/1000/2000/3000; keep engineering success separate from 34.2/.08.
Log: `logs/foundir_rain_flux2_precision_full_v1_formal_v1.log`.
Output: `checkpoints/foundir_rain_flux2_precision_full_v1/formal_v1/`.
Launch/config source snapshot is backed up at commit 761a1f5 and the
qualification evidence at 2ad559c. The previous unstarted status is historical.

## Latest: full-data Rain qualification PASSED — 2026-10-25

The 25-step full-data qualification completed on eight H100s at
2026-10-25T21:16:59Z, launcher status 0. Actual exposure: 800 records from
800 unique training sources. Development PSNR/LPIPS improved from
32.08470176034529/.0888220098754391 to
32.64423398615503/.08657967696897685; the 34.2/.08 gate remains false.
Train-probe result: 32.66665987243047/.0708855870179832. This is engineering
qualification, not evidence that the final scientific target will pass.
Finite gradients, actual A/B updates, saved F32 LoRA/optimizer, matched
metrics and unchanged source/config hashes were verified. No protected data.
Total launcher elapsed 173 seconds (.04805555555555555 wall hours,
.3844444444444444 allocated H100 GPU-hours). Median step 1.5106424172408879
seconds, reported peak allocated memory 24.854560375213623 GiB.
Update the formal estimate to 1.5-2.0 wall hours / 12-16 allocated GPU-hours,
subject to sustained full-data IO. Formal has NOT been launched.
The already-authorized independent 3,000-step formal can now be started on
the user's GPU node with
`bash smoke/start_foundir_rain_flux2_precision_full_v1_tmux.sh formal formal_v1`.
Initialize from original step-2,000 LoRA/fresh Adam, never qualification.
Do not change source/config after qualification or auto-advance to MoE.
Full evidence: `reports/foundir_rain_precision_full_qualification_2026-10-25.md`.

## ACTIVE: full-data FP32 Rain FLUX — 2026-10-25

The user explicitly authorized full-data matched-resolution training after
the 300-step fitting diagnostic. This supersedes the earlier instruction to
wait for a new full-data decision. Do not resume the 32-image fit run.
Keep Rain development gate >=34.2 PSNR / <=.08 LPIPS before image-only MoE
and then VLM-token routing. No pixel refiner, EBT or action-energy branch.

Use `configs/foundir_rain_flux2_precision_full_v1.json` and its new trainer.
Independent initialization remains the original hash-bound direct-endpoint
step-2,000 LoRA with fresh Adam; neither diagnostic nor qualification weights
may seed formal. Preserve FP32 LoRA/Adam, BF16 compute, shared FP32 endpoint,
teacher training-only, rank32 and all eight original loss terms/coefficients.
The pool is all 38,334 Rain records / 5,780 sources. Cycle uniformly through
source groups then each group's records, random aligned 512 crops and D4
augmentation. Reflect-pad only the 35 training records with short side <512;
never resize or change development images. No decoded training-image cache.
Two spawned workers per rank prefetch two batches each; verify image SHA-256
before first decode and again if file stat changes. Metadata-path checks run
once on rank0, not redundantly on all ranks.

First run a fresh 25-step full-data engineering qualification on one homogeneous
8xH100/H200 node. Then independently train 3,000 steps, batch32, evaluating
at 0/250/500/1000/2000/3000. Formal requires a passing matching qualification
receipt and cannot auto-start. Keep the same 100-record development gate and
32 training probe; probe scores are not SOW results. No protected data.
Qualification estimate .10-.50 wall hours / .8-4 allocated GPU-hours.
Provisional formal estimate 1.5-3.0 wall hours / 12-24 allocated GPU-hours;
replace with full-data qualification throughput (cached 32-image speed is not
a guarantee). Report actual unique records/sources and sample presentations.

Eight CPU tests and full artifact/data preflight passed. No GPU full-data
qualification has been launched from this GPU-less Codex environment.
Start on the user's node:
`bash smoke/start_foundir_rain_flux2_precision_full_v1_tmux.sh`.
Readiness: `reports/foundir_rain_precision_full_readiness_2026-10-25.md`.

## Latest terminal update: Rain precision fit complete — 2026-10-25

The independent `fit_v1` completed 300/300 steps on 8xH100 at
2026-10-25T20:59:43Z, launcher status 0. Training-fit reaches
35.09303906168583/.04928288975497708 (numeric target passes); development
reaches 32.44553214852097/.08547242213506251 (Rain SOW gate fails).
Step-100 development was 32.54685884616055/.08776275758864358.
Do not describe training-fit success as Rain gate success or evidence that
FP32 alone caused the gain. This proves fitting ability on these 32 fixed
training inputs, not unseen-source generalization. Of 24 development inputs
with input PSNR >=34, 20 still lose PSNR after restoration at step 300.
The run is complete: do not resume/extend it on the same 32 images or
auto-launch full-data training. Next proposed work should address
generalization with a frozen full-data, matched-resolution protocol and
measured budget, requiring a new user decision before GPU launch.
Preserve the Rain gate -> image-only MoE -> VLM router order. No pixel refiner.
Total launcher time: 546 seconds (9m06s), .15166666666666667 wall hours,
1.2133333333333334 allocated H100 GPU-hours. This is faster than the prior
provisional estimate. Full result and limits:
`reports/foundir_rain_precision_fit_terminal_2026-10-25.md`.

## Latest update: precision qualification complete; fit authorized

25-step qualification ended normally at 2026-10-25T20:39:00Z on eight H100s.
Development: 32.61941190970921/.08720199926523492, improving both metrics
but still below the 34.2/.08 joint gate. Training-fit:
33.364734543215874/.06714100111275911. Parameter updates and finite gradients
were verified. No protected data used. Full report:
`reports/foundir_rain_precision_qualification_2026-10-25.md`.
User authorized independent 300-step fitting diagnosis from original
step-2,000 LoRA and fresh optimizer, not qualification weights or full-data
training. On the user's GPU node run:
`bash smoke/start_foundir_rain_flux2_precision_fit_v1_tmux.sh fit fit_v1`.
Expected 15-20 minutes / 2.0-2.67 allocated GPU-hours. Fit launch is not yet
confirmed; no log existed at this update, and Codex has no local GPU.

## 2026-10-25 current: Rain numerical repair, no pixel refiner

The latest user instruction is Rain DiT SOW pass first, ordinary image-only
MoE second, VLM-token router third. Blur already passes. The user rejected
the small pixel refiner proposal. Its code is untrained historical work;
do not launch it. The former direct-endpoint formal already completed and
failed: best step 2,000 was 32.085315339143946/.08882607079111039.

The authorized next run fixes BF16 trainable/Adam state and unequal latent
precision in training versus inference. Both now use one shared endpoint
function with FP32 observation-minus-prediction; model compute stays BF16.
Use the new `foundir_rain_flux2_precision_fit_v1` config/trainer/launchers.
The exact best step-2,000 LoRA initializes each independent run with a fresh
optimizer. First 25 engineering steps, then an independent 300-step fit
diagnostic on 32 deterministic training sources at 512 resolution. The same
100-source development set is evaluated separately at the unchanged gate.
The original loss coefficients are retained. No extra refiner or MoE is
trained, and no long full-data run is auto-launched. Four CPU tests and the
32+100 data/artifact preflight pass; no GPU result exists yet.

Launch on the user's single 8xH100/H200 node:
`bash smoke/start_foundir_rain_flux2_precision_fit_v1_tmux.sh`

Log: `logs/foundir_rain_flux2_precision_fit_v1_qualification_v1.log`.
Readiness, limits, budgets and later fit command:
`reports/foundir_rain_flux2_precision_fit_v1_readiness_2026-10-25.md`.

## 2026-10-17 active: explicit pairwise safe gate CPU-ready

Following the terminal confidence-ranking failure, the active replacement is
a fresh `1,199,043`-parameter pairwise safe gate. It directly predicts gain,
perceptual harm and relative log-MSE for each deployable alternative versus
fidelity. It does not reuse the failed scorer or train on obvious external
negatives. Structured VLM severity and candidate identity remain explicit
inputs.

Probability thresholds are selected only on source-disjoint calibration
groups and frozen before evaluation. No valid calibration policy means exact
fidelity fallback. Compilation, configuration/Bash checks, checkpoint hashes
and `7/7` focused tests pass. The next action is a fresh 25-step qualification
on 8xH100/H200, estimated at `.08–.25` wall-clock hours and `.64–2.0`
GPU-hours. Full contract:
`reports/foundir_pairwise_safe_candidate_gate_noise_v1_readiness_2026-10-17.md`.

## 2026-10-17 risk/coverage audit rejects current scorer

The source-disjoint no-training audit completed in `203` seconds (`.0564`
wall-clock hours, `.4511` H100 GPU-hours). On 174 calibration records from 17
source groups, every nonzero acceptance threshold reduces PSNR below the
fidelity-only `43.341621968938014`; even `.4849%` acceptance reaches only
`43.339416964300746`. Calibration therefore selects zero coverage.

On the disjoint 82-record/17-group evaluation partition, the output equals
fidelity exactly at `43.514986642977085/.03269933527846616`.
`scorer_has_deployable_signal=false`. This terminally rejects threshold repair
and the present heteroscedastic scorer. The next route is explicit pairwise
safe-versus-candidate advantage/perceptual-harm learning with source-disjoint
confidence calibration. Full record:
`reports/foundir_complete_candidate_energy_risk_coverage_v1_terminal_2026-10-17.md`.

## 2026-10-17 formal1 stopped; risk/coverage audit CPU-ready

The user stopped complete-candidate energy `formal1` after pass 1 failed.
Shared telemetry stopped after step `2,285` at UTC
`2026-10-17T20:23:20Z`; direct tmux termination produced no normal END marker.
Never resume it or use its post-pass-1 partial state.

The next experiment is a short, no-training source-disjoint risk/coverage
audit of the immutable step-1,706 checkpoint. The fixed 256 development rows
are split by source group into calibration and evaluation halves. Acceptance
threshold selection uses calibration only; evaluation is consumed once. This
determines whether the scorer contains any genuine high-confidence ordering
signal. If it fails, reject the current scorer and move to explicit pairwise
safe-versus-candidate learning rather than adjusting another threshold.

Static checks and `6/6` focused tests pass. Expected cost is `.08–.20`
wall-clock hours and `.64–1.60` allocated GPU-hours on one 8xH100/H200 node.
Full contract:
`reports/foundir_complete_candidate_energy_risk_coverage_v1_readiness_2026-10-17.md`.

## 2026-10-17 complete candidate energy pass 1 fails

`formal1` reached step `1,706` with numerically healthy training and no
protected-data access, but inference-only selection is substantially worse
than fidelity. Selected is `43.18257042020559/.03665485502278898`; fidelity is
`43.39715355634689/.028955043449059303`. This is
`-.21458313614130042 dB/+.007699811573729676 LPIPS` relative damage.

The core failure is over-acceptance: `96.69189453125%` of regions abandon the
safe fidelity candidate, and unchanged proposal alone is selected on
`43.511962890625%`. The intended uncertainty gate is therefore not calibrated
at pass 1. The joint gate is false. The frozen run continued beyond step
1,900 because no pass-1 stop rule was declared; do not restart, retune, or
interrupt it without a new user decision. Full record:
`reports/foundir_complete_candidate_energy_noise_v1_pass1_2026-10-17.md`.

## 2026-10-17 active: complete candidate-conditioned energy CPU-ready

The user explicitly paused the EBT Generalist gate and authorized the complete
candidate-conditioned Noise energy after the 32x32/64x64 GT oracle passed.
The previous seven-alpha and continuous-alpha selectors remain terminal and
must not resume.

The new implementation freezes the existing proposal and hash-bound
step-8,530 residual basis. A `1,193,907`-parameter shared scorer sees the
observation, proposal, actual candidate, two candidate-relative residuals,
and structured VLM severity. It predicts separate relative local MSE and
spatial-LPIPS risks plus uncertainty on 32x32 regions. Wrong-expert,
under-restored, detail-damaged and repeated-proposal outputs are training
negatives only and are never deployable actions. Fidelity is the fail-closed
baseline.

Static/CPU qualification is complete: compilation, JSON/Bash checks, all six
focused tests, spatial-LPIPS forward/backward, and every manifest/checkpoint
hash check passed. No GPU job has been launched from the Codex container,
which has no visible GPUs. The exact next action is one fresh 25-step
engineering qualification on a homogeneous 8xH100/H200 node. Its checkpoint
must never seed formal training. Estimated cost is `.2–.6` wall-clock hours
and `1.6–4.8` allocated GPU-hours; replace this estimate with measured data.
Full contract:
`reports/foundir_complete_candidate_energy_noise_v1_readiness_2026-10-17.md`.

## 2026-10-17 complete-candidate oracle passes both protocols

The no-training audit completed successfully in `235` seconds (`.0653`
wall-clock hours and `.5222` allocated H100 GPU-hours). Both 32x32 and 64x64
protocols pass all predeclared feasibility conditions.

- Fidelity: `43.39715355634689/.028955043449059303`.
- 32x32 oracle: `43.52247426658869/.03426853639393812`, or
  `+.1253207102417946 dB` beyond fidelity; five candidates exceed 1% wins and
  flip-aligned winner stability is `.77020263671875`.
- 64x64 oracle: `43.512206099927425/.03413928690861212`, or
  `+.11505254358053207 dB` beyond fidelity; four candidates exceed 1% wins and
  stability is `.818603515625`.

Both oracle LPIPS values remain safely below `.045362067760033824`. Candidate
wins are meaningfully distributed among over-fidelity, fidelity,
under-fidelity and unchanged proposal, while wrong-expert/detail-damaged
negatives are much worse than fidelity. This resolves the feasibility gate in
favor of implementing the complete candidate-conditioned energy. These are
GT-selected upper bounds, not deployable performance. Result SHA-256:
`7f7bbfa757580efeaeeb457ed32cb764e948c369393c907f6361cff7a8d0111d`.
Full record:
`reports/foundir_complete_energy_candidate_oracle_noise_v1_terminal_2026-10-17.md`.

## 2026-10-17 active: complete-candidate oracle after seven-alpha stop

The user stopped local-action `formal1`. Its final log record is step `3,910`;
the last complete checkpoint and validation are step `3,412`. Direct tmux
termination produced no normal END marker. Never resume this run.

The seven-alpha selector is now recognized as a narrow diagnostic, not the
complete advisor-PPT method. The next authorized action is no-training only:
audit the strong step-8,530 residual basis with genuinely different deployable
candidates (proposal, fidelity, perceptual, under/over fidelity, and
fidelity/perceptual blends), while treating wrong-expert and detail-damaged
outputs only as contrastive negatives. On the fixed 256-record development
set, measure 32/64-patch GT oracle metrics, candidate win shares, and aligned
horizontal-flip winner stability. Do not start energy training until this
candidate pool shows at least `+.08 dB` oracle headroom beyond fidelity,
LPIPS remains within the frozen gate, and winner labels are not random.

## 2026-10-17 local candidate-action formal pass 2: still no action

At step `3,412`, local action remains exactly equal to frozen fidelity:
`43.39715355634689/.028955043449059303`. The selector again accepts `0%` of
alternative regions and chooses alpha `1` everywhere. Relative to proposal it
is unchanged at `+.27299337834119797 dB/-.02140702431097452 LPIPS`, leaving
`.02700662165880202 dB` to the PSNR gate.

Two full passes now show no deployable change. Recent ranking cross-entropy is
roughly `1.95–1.98`, essentially the seven-class uniform reference `1.94591`.
This is a strong failure signal for the current formulation, not a promising
trend. The healthy run has already entered pass 3 and continues unchanged
unless the user explicitly stops it. Full record:
`reports/foundir_local_action_energy_noise_v1_pass2_2026-10-17.md`.

## 2026-10-17 local candidate-action formal pass 1: safe no-change

At step `1,706`, the local-action output is
`43.39715355634689/.028955043449059303`, exactly equal to frozen fidelity.
The fail-closed selector accepted `0%` of alternative regions and selected
alpha `1` everywhere. Relative to proposal this remains
`+.27299337834119797 dB/-.02140702431097452 LPIPS`; PSNR is still
`.02700662165880202 dB` short of the joint gate, while LPIPS passes widely.

This confirms no-harm behavior but is not evidence that the candidate-energy
mechanism works. Recent candidate-ranking cross-entropy remains about `1.98`,
which is not better than the uniform seven-class reference `ln(7)=1.94591`.
Continue the unchanged frozen schedule so later passes can determine whether
ranking begins to generalize; do not alter the live decision margin or loss.
Full record:
`reports/foundir_local_action_energy_noise_v1_pass1_2026-10-17.md`.

## 2026-10-17 local candidate-action energy qualification passed

Fresh run `qualification_v1` completed on 8xH100 from `15:08:02Z` to
`15:11:53Z` with launcher status `0`, finite training, checkpoint and matched
256-record development validation. Cost was `.0642` wall-clock hours and
`.5133` allocated GPU-hours. All protected-data access flags are false.

At step 25, the fail-closed selector accepts zero alternative regions and
chooses alpha `1` everywhere. Therefore local action output exactly reproduces
fidelity: `43.39715355634689/.028955043449059303`, equal to
`+.27299337834119797 dB/-.02140702431097452 LPIPS` versus proposal. This is
the intended safe engineering baseline, not evidence that action selection
improves quality. The qualification checkpoint must not seed formal training.
Next action is a fresh-ID, from-zero 8,530-step formal run.

Checkpoint SHA-256:
`ec98308fefd05fffa92a2dd6d05e76dbc2c38363b7ad251ce220d23c2e8be5f4`.
Validation JSON SHA-256:
`36512bcf48f63aea88e161d611f3bf7888056aba6f737a99a0a37db69794d48f`.

## 2026-10-17 active decision: local candidate-action energy

The user stopped `foundir_spatial_residual_amplitude_noise_v1/formal1` and
rejected further continuous-alpha regression. The launcher ended with status
`120` at `2026-10-17T14:53:05Z`; the last logged step was `5,525`, while the
last complete checkpoint and development validation remain step `5,118`.
Never resume or seed from this stopped run.

The active replacement freezes the Noise proposal and step-8,530 residual
basis, then learns to compare seven local restoration actions:
`alpha={0,.5,.75,1,1.25,1.5,2}`. The model sees only inference-available
observation/proposal/fidelity/residual tensors and predicts one relative cost
per candidate per 8x8 region. Training GT labels which action is locally
better. Validation and inference never use GT for action selection. Alpha
`1` is the safe baseline; initialization and uncertain decisions fall back
exactly to the frozen fidelity output.

This addresses the measured failure directly: continuous regression learned
an average alpha and recovered only `+.0031436309 dB` beyond fidelity, while
the GT-local oracle proves that the frozen direction contains about
`+.2626 dB` additional PSNR headroom beyond fidelity. The new test asks whether
explicit action comparison can recover the missing `.0270066217 dB` needed
for the original `+.30 dB` proposal-relative gate. It is not yet evidence of
an image-quality gain.

## 2026-10-17 spatial-amplitude formal pass 3: improvement flattening

At step `5,118`, spatial is `43.400297187268734/.0317960577121994`.
Compared with frozen fidelity, PSNR is `+.0031436309218444425 dB` better and
LPIPS is `.0028410142631400966` worse. Relative to proposal, it remains
`+.27613700926303864 dB/-.018566010047834425 LPIPS`, so LPIPS passes but PSNR
is `.023862990736961353 dB` short. Pass-2 to pass-3 PSNR gain grew by only
`.0007849782705344696 dB`; the deployable transfer is flattening far below the
GT oracle. Continue the last two frozen passes unchanged. Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass3_2026-10-17.md`.

## 2026-10-17 spatial-amplitude formal pass 2: tiny PSNR gain

At step `3,412`, spatial output is `43.3995122089982/.033109975053776`.
Compared with frozen fidelity, PSNR is now `+.00235865265130997 dB` better,
while LPIPS is `.004154931604716697` worse. Compared with proposal, the result
is `+.27535203099250793 dB/-.017252092706257827 LPIPS`: LPIPS still clears the
target comfortably, while PSNR remains `.02464796900749206 dB` short. The
predictor has begun to recover a small deployable PSNR benefit but remains far
below the GT-local oracle. Continue the frozen five-pass run unchanged. Full
record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass2_2026-10-17.md`.

## 2026-10-17 spatial-amplitude formal pass 1: slight regression

`formal1` is live and reached step `1,706`. Spatial output is
`43.395900413393974/.029359601011492487`, versus frozen fidelity
`43.39715355634689/.028955043449059303`: PSNR is lower by
`.00125314295291574 dB` and LPIPS is worse by `.000404557562433184`. The map
is learning nontrivial variation (`mean 1.0358318537`, standard deviation
`.1107148176`) but has not yet generalized to a metric improvement. Continue
the frozen run unchanged; pass 1 alone is not a justified stop or retune.
Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass1_2026-10-17.md`.

## 2026-10-17 spatial-amplitude GPU qualification passed

`qualification_v1` completed successfully on `8 x H100 80GB` from
`13:19:04Z` to `13:20:14Z`, with status `0`, finite training and matched
checkpoint/validation output. At step 25 the amplitude map remains near its
identity initialization (`mean 1.0000218451`, standard deviation
`.0004146491`); spatial output is therefore effectively identical to frozen
fidelity (`43.39715509861708/.028954966111086833` versus
`43.39715355634689/.028955043449059303`). This proves engineering safety but
is not enough optimization for a scientific conclusion.

The next action is a fresh, from-zero `formal1` run for 8,530 steps. Never seed
it from qualification. Estimated H100 cost is `2.5–3.5` wall-clock hours and
`20–28` allocated GPU-hours; update the estimate at step 1,706. Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_qualification_2026-10-17.md`.

## 2026-10-17 deployable spatial-amplitude predictor CPU-ready

The user authorized the narrow follow-up supported by the oracle. Proposal and
the step-8,530 residual basis remain frozen; only a `223,393`-parameter CNN is
trained to predict one `[0,2]` amplitude per 8x8 patch from observation,
proposal, fidelity and their residual. Its zero-initialized output starts at
exact alpha `1`, reproducing fidelity. Training GT creates analytic local
amplitude targets, but inference and development validation use no GT-time
selection.

JSON/Python/Bash checks, two focused unit tests, strict frozen-checkpoint
loading and diff checks pass. The next action is a fresh 25-step eight-GPU
qualification; its checkpoint must not seed the formal run. Readiness:
`reports/foundir_spatial_residual_amplitude_noise_v1_readiness_2026-10-17.md`.

## 2026-10-17 final fidelity-residual oracle: local headroom confirmed

The GT-only audit completed successfully in `72` seconds (`.020` wall-clock
hours, `.160` allocated GPU-hours), exactly reproducing the checkpoint's
proposal and fidelity metrics. No global amplitude improves enough to cross
the joint gate; maximum global PSNR remains at alpha `1.000`. However, the
GT-derived local 8x8 hard-amplitude upper bound is
`43.65973751991987/.03243445468615391`, or
`+.5355773419141769 dB/-.01792761307387991 LPIPS`, and all 8/16/32/64 local
patch variants pass.

This proves the fidelity direction contains enough local information, while a
single image-wide scale cannot recover it. These GT-dependent patch results
are not deployable model metrics. The next scientific problem is narrowly
defined: learn a spatial amplitude map from inference-available inputs without
GT, rather than simply extending training or changing a global residual
scale. Result SHA-256:
`ce6ab400c5003a630f3e05fed60a5e1fe9e6d6514af86cddc1fed6b9e65e2f66`.
Full report:
`reports/foundir_proposal_residual_basis_noise_v1_oracle_terminal_2026-10-17.md`.

## 2026-10-17 final fidelity-residual oracle launch-ready

The user authorized the no-training GT-only feasibility audit of the frozen
step-8,530 fidelity residual. The implementation evaluates global amplitudes
`0–2` in `.025` increments and analytic local 8/16/32/64-patch amplitudes on
the same fixed 256-record source-disjoint development subset. It verifies both
proposal and alpha-1 fidelity protocol reproduction before writing output and
reports fixed-candidate plus per-image Pareto upper bounds. This cannot be
reported as deployable inference quality.

Python compilation, Bash syntax, focused unit tests, and diff checks pass. The
Codex container exposes no GPU, so launch the prepared script on one
homogeneous eight-GPU H100/H200 node. Estimated cost is `.10–.17` wall-clock
hours and `.8–1.36` allocated GPU-hours. Do not begin continuation, energy, or
gate training before reviewing this result. Readiness details:
`reports/foundir_proposal_residual_basis_noise_v1_oracle_readiness_2026-10-17.md`.

## 2026-10-17 residual-basis formal terminal: near miss

`formal1` completed normally at step `8,530` with status `0`. Final fidelity
is `43.39715355634689/.028955043449059303`, or
`+.27299337834119797 dB/-.02140702431097452 LPIPS` versus proposal. It passes
the LPIPS target widely but misses the `+.30 dB` PSNR requirement by only
`.02700662165880202 dB`; the joint gate is therefore not passed. The final
pass added `+.03130790591239929 dB`, so optimization was not completely
stationary. The remaining PSNR gap corresponds to only about `.620%` further
MSE reduction relative to the final fidelity output.

The successful run used `2.5869` wall-clock hours and `20.6956` allocated
H100 GPU-hours. Do not resume or extend it. Next, run only a GT-only
amplitude/local feasibility audit of the frozen final fidelity residual to
determine whether its direction already contains enough headroom for the
gate. Do not start an energy/gate or continuation before reviewing that audit.
Full record:
`reports/foundir_proposal_residual_basis_noise_v1_terminal_2026-10-17.md`.

## 2026-10-17 residual-basis formal pass 4: PSNR plateau

At step `6,824`, fidelity is `43.365845650434494/.02899720575715037`, equal
to `+.24168547242879868 dB` PSNR and `-.021364862002883456` LPIPS versus the
proposal. It still improves both metrics strongly, but PSNR gained only
`+.011031992733478546 dB` over pass 3 and remains `.05831452757120131 dB`
short of the joint gate. This is a clear plateau signal, so final gate success
is now unlikely rather than assured. Finish the unchanged fifth pass to obtain
the frozen endpoint; do not retune the active job. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass4_2026-10-17.md`.

## 2026-10-17 residual-basis formal pass 3: near joint gate

At step `5,118`, fidelity reached `43.354813657701015/.030396450317311974`,
or `+.23065347969532013 dB` PSNR and `-.01996561744272185` LPIPS versus the
proposal. It now needs only `.06934652030467986 dB` more PSNR to clear the
joint `+.30 dB/-.005 LPIPS` gate. The PSNR gain increased by
`+.1251758486032486 dB` from pass 2 rather than saturating. This makes gate
success likely, but it is still interim development evidence. Continue passes
4 and 5 unchanged. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass3_2026-10-17.md`.

## 2026-10-17 residual-basis formal pass 2: hopeful trend confirmed

At step `3,412`, fidelity reached `43.22963780909777/.03339703518213355`,
improving over proposal by `+.10547763109207153 dB` and
`-.016965032577900274` LPIPS. PSNR gain therefore increased from pass 1's
`.04423 dB` into the predeclared `.09–.12 dB` hopeful pass-2 range, while the
LPIPS improvement also grew. The joint direction is real, but the PSNR gain
still needs another `.194522 dB` to clear the `+.30 dB` gate. Continue the
unchanged run; do not infer a linear endpoint or retune from pass 2. Full
record:
`reports/foundir_proposal_residual_basis_noise_v1_pass2_2026-10-17.md`.

## 2026-10-17 residual-basis formal pass 1: joint improvement

`formal1` is live on `8 x H100 80GB` and reached step `1,706` normally. The
fidelity head is `43.16839203238487/.0400125195938017`, versus frozen proposal
`43.124160178005695/.050362067760033824`: `+.044231854379177094 dB` PSNR and
`-.010349548166232125` LPIPS. It therefore improves both metrics and already
beats the old direct residual's GT-only oracle (`+.042452 dB/-.003946 LPIPS`)
in both dimensions. The new restoration-native basis is producing useful
correction information rather than merely learning a better gate.

The perceptual head trades more PSNR for LPIPS
(`43.0077431127429/.03262401805659465`), showing that the two heads have now
separated as intended. The fidelity head already clears the LPIPS component
of the `+.30 dB/-.005` gate by more than twofold, but its `.0442 dB` PSNR gain
is only about 15% of the PSNR requirement. This is promising pass-1 evidence,
not final success. Continue the unchanged five-pass run; do not retune or
restart it. Pass 1 took about `.458` wall-clock hours and `3.66` allocated
GPU-hours. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass1_2026-10-17.md`.

## 2026-10-17 residual-basis GPU qualification passed; formal run next

The fresh `qualification_v1` run completed successfully on `8 x H100 80GB`
from `2026-10-17T09:57:50Z` to `09:59:08Z`, with launcher status `0`, finite
training, and a saved step-25 checkpoint plus matched 256-record development
validation. The run took about `.0217` wall-clock hours and `.173` allocated
GPU-hours. All protected-data access flags are false.

Proposal is `43.124160178005695/.050362067760033824` PSNR/LPIPS. At step 25,
the fidelity head is `43.13571598380804/.05056087289904099` (`+.011555805802345276`
dB and LPIPS worse by `.00019880513900716323`), while the perceptual head is
`43.13552286475897/.050557308552015456` (`+.01136268675327301` dB and LPIPS
worse by `.00019524079198163236`). The near-identical heads and small LPIPS
regression are not a scientific failure after only 25 warm-up steps; this run
only establishes that the engineering path works.

The next action is a fresh, from-zero `formal1` run for the frozen five-pass,
`8,530`-step schedule. Never resume or seed it from `qualification_v1`. A
conservative runtime estimate is `3.5–6.0` wall-clock hours on one eight-GPU
node (`28–48` allocated GPU-hours), with pass 1 used to update the estimate.
No energy/gate training is authorized before the residual basis demonstrates
adequate headroom. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_qualification_2026-10-17.md`.

## 2026-10-17 residual-basis causal test authorized and CPU-ready

The user explicitly authorized fixing the oracle-proven lack of information in
the direct residual. The active experiment freezes the existing proposal and
uses a `26,116,852`-parameter restoration-native Restormer to learn two
independent proposal corrections from degraded image, proposal, and their
difference. The fidelity head directly optimizes MSE; the perceptual head uses
a stronger LPIPS term while both retain MSE and 8x8 local no-harm supervision.
Energy and gate training remain off until this basis proves sufficient.

JSON, Python, Bash, a focused unit test, and real step-250 checkpoint
initialization pass. The next action is a fresh 25-step engineering
qualification on one homogeneous 8xH100 or 8xH200 node. It must not seed the
formal run. Details:
`reports/foundir_proposal_residual_basis_noise_v1_readiness_2026-10-17.md`.

## 2026-10-17 direct-residual oracle terminal: joint gate fails

The step-`5,118` GT-only feasibility audit completed normally on all 256 fixed
source-disjoint development records. The best maximum-PSNR oracle output uses
GT-fitted 8x8 patch amplitudes on 255/256 images and keeps the proposal on one:
`43.16661264002323/.04641580081442953`, versus proposal
`43.124160178005695/.050362067760033824`. This simultaneously improves both
metrics, but by only `+.04245246201753616 dB` and `-.003946266945604293`
LPIPS, far below the `+.30/-.005` gate. Among target-selecting Pareto points
that do clear the LPIPS reduction, the best PSNR gain is only
`+.03746754676103592 dB`. No Pareto point gains `+.30 dB` PSNR, so the frozen
direct residual does not contain a combination capable of passing the joint
gate. A learned gate over the same candidates cannot fix that ceiling.

The successful rerun took `207` seconds (`.0575` wall hours) on 8 GPUs, about
`.46` allocated GPU-hours. Including the earlier false-failure run, the audit
used about `.94` allocated GPU-hours. Result SHA-256:
`6400d78624dd2a9fe2cc2f67e622dc9371fd44243346df4f5977c6c17fc8a317`.
Full report:
`reports/foundir_proposal_energy_v2_direct_residual_oracle_terminal_2026-10-17.md`.

## 2026-10-17 oracle first launch: protocol-tolerance false failure

The first direct-residual oracle launch completed all 256 records but exited
before writing its result because the protocol-reproduction assertion was too
strict. The rerun proposal was `43.124160178005695/.050362067760033824`
versus checkpoint validation `43.123698368668556/.05036422960256459`: only
`.00046180933713935146 dB` and `.000002161842530766122` LPIPS apart. This is
far below the scientific `+.30/-.005` gate and is consistent with a tiny
GPU-kernel evaluation difference, not a changed dataset or model. The
checkpoint/config hashes and 256-record protocol all matched. The assertion
now permits at most `.001 dB/.00001 LPIPS` and records the exact reproduction
delta in the output JSON. Re-run the same launcher; no training or recipe
change occurred.

## 2026-10-16 V2 stopped; direct-residual oracle is next

The user stopped `noise_energy_v2_formal1` externally after the last logged
step `6,655`. Do not resume it. Since no later milestone checkpoint was
written, the diagnostic input remains the hash-bound step-`5,118` checkpoint.
The user authorized a GT-only feasibility audit on the same fixed 256-record
source-disjoint development subset. It tests global residual amplitudes,
GT-fitted local patch amplitudes, and a per-image Pareto selection over those
candidates. It performs no training and accesses no protected set. The result
can establish whether the learned direct residual contains enough information
to clear `+.30 dB/-.005 LPIPS`; it cannot be reported as deployable quality.
Implementation and single-node launcher:
`smoke/evaluate_foundir_proposal_energy_v2_oracle_feasibility_ddp.py` and
`smoke/run_foundir_proposal_energy_v2_oracle_feasibility_v1_8gpu.sh`.

## 2026-10-16 V2 pass-3 interim

`noise_energy_v2_formal1` reached step `5,118` (three complete effective
passes) and continued to step `6,635/8,530` as of `19:09 UTC` on `8 x H200`.
On the fixed 256-record source-disjoint development set, frozen proposal is
`43.123698368668556/.05036422960256459` PSNR/LPIPS; the adaptive energy arm
is `43.12445606291294/.05040028668190644`, and the direct refiner is
`42.9707455933094/.04185937133115658`. No arm simultaneously meets the
predeclared `+.30 dB` and `-.005 LPIPS` improvement gate. This is interim
evidence, not a reason to alter or stop the frozen five-pass run. Full metrics,
artifact hashes, and protected-data flags:
`reports/foundir_proposal_relative_energy_noise_v2_pass3_2026-10-16.md`.

## 2026-10-16 V2 pass-2 interim

`noise_energy_v2_formal1` resumed from step `1,706` and passed step `3,412`
on `8 x H200`; it was still advancing beyond step `4,100` at the last check.
The fixed development validation shows adaptive energy
`43.12455575913191/.05040373227166128` versus its frozen proposal
`43.123698368668556/.05036422960256459`. The change is only
`+.0008573904633522034 dB` and `+.000039502669096691534 LPIPS` (worse LPIPS), so
the method has not met the gate. The direct refiner improves LPIPS but lowers
PSNR. Continue the frozen five-pass run without retuning. Details and artifact
hashes: `reports/foundir_proposal_relative_energy_noise_v2_pass2_2026-10-16.md`.

## STOP: 2026-10-15 MAX-COMPLETE PPT ENERGY V2 ONLY

The user explicitly chose one final, complete implementation of the PPT
proposal-relative energy idea before any DiT-MoE fallback. V1 must finish its
already-running 8,530-step endpoint and then stop. Do not extend V1.

V2 is restricted to exactly six corrections supported by the V1 failure:
LPIPS-aware training; real frozen-model wrong/under/over/detail-damaged
candidates; trained 1/2/3-step unrolling; per-step PSNR/LPIPS no-harm plus the
option to retain the proposal; per-image frozen-Qwen severity; and spatial
local energy supervision. Do not add a new backbone, dataset, degradation,
specialist, MoE, router, or compositional mechanism. Noise data, frozen
Restormer proposal, source-disjoint validation, protected-data boundaries, and
the `+.30 dB/-.005 LPIPS` gate remain unchanged. This section supersedes the
older instruction to switch directly from V1 to DiT-MoE.

### 2026-10-15 V1 terminal and V2 launch-readiness update

V1 has now completed its frozen five-pass endpoint at `8,530` steps and is no
longer running. Its final best energy arm was
`43.227015025913715/.05713279500741919` versus the unchanged proposal
`43.124160178005695/.050362067760033824`: PSNR improved only
`+.10285484790802033 dB` while LPIPS worsened `+.006770727247385367`.
Therefore V1 failed the `+.30 dB/-.005 LPIPS` gate and must not be resumed.
It consumed `3.1357351445996513` wall-clock hours and
`25.08588115926418` allocated H100 GPU-hours. Full record:
`reports/foundir_proposal_relative_energy_noise_v1_terminal_2026-10-15.md`.

The exact-six-change V2 implementation now exists and passes compilation,
JSON/Bash/diff checks and five direct CPU tests. A PyTorch naming collision
(`self.half` versus `nn.Module.half()`) was corrected before GPU launch.
No V2 GPU job has been launched yet. The next action is exactly one fresh
25-step, single-node eight-GPU engineering qualification; it is not scientific
evidence and must not be reused as the formal five-pass run. Readiness record:
`reports/foundir_proposal_relative_energy_noise_v2_readiness_2026-10-15.md`.

The 25-step qualification subsequently completed successfully as
`noise_energy_v2_qualification1` on `8 x H100 80GB`: launcher status `0`, all
`488` model tensors finite, all `488` trainable entries represented in the
optimizer state, checkpoint and 256-record validation written, and every
protected-data flag false. Step-25 metrics are intentionally non-scientific
and essentially unchanged from the proposal. The qualification checkpoint
must not seed the formal run. The next action is a new-ID, from-zero `8,530`-
step V2 run. Full record:
`reports/foundir_proposal_relative_energy_noise_v2_qualification_2026-10-15.md`.

## MANDATORY: keep coherent progress pushed to GitHub

Do not let model code, experiment contracts, fixes, research decisions, or
terminal results accumulate only in this dirty workspace. After every coherent
material milestone, make a focused commit and push it to `origin/main`.

Required push points are: launch-ready code/config; a substantive corrected
failure; an advisor/user priority change; each meaningful qualification or
pass-level validation; checkpoint selection; and every terminal report. Do not
commit per-step logs while a job is running, but ensure the exact code/config
that launched it is remotely recorded, then push the validated result later.

Because this workspace contains more than one thousand historical modified or
untracked paths, never use `git add -A`, `git add .`, `git clean`, or
`git reset --hard`. Fetch first, stage only an explicit file allowlist, exclude
all data/checkpoints/weights/images/caches/telemetry/live logs/large manifests,
scan staged content for secrets, run syntax and diff checks, then push. When
the dirty workspace conflicts with remote history, construct the commit in an
isolated temporary worktree based on `origin/main`; leave the live experiment
workspace untouched. A push is complete only after verifying the remote commit
hash and reporting it to the user.

The Git remote is `origin` (`Jason-Wang313/zhenhaowangbackup`) and the intended
tracking branch is `main`. The canonical remote base recovered on 2026-10-15
is `5003e1e51bf9373fbdb352cc0420d712d396f114`. Credentials must remain outside
the repository; never record the previously exposed token or any new token.

### 2026-10-15 historical workspace backfill

The accumulated reproducible text backlog was audited against current
`origin/main` and prepared as two explicit Git snapshots: executable
source/config/tests first, then reports/context. This is archival preservation,
not a reactivation of older research branches. The topmost STOP decision in
this handoff and `AGENTS.md` remains authoritative.

Large generated manifests, checkpoints, weights, datasets, images, caches,
telemetry, and live logs are intentionally excluded and remain untouched
locally. The stale lowercase `agents.md` and old top-level session handoffs are
also excluded so they cannot compete with the authoritative context files.
Full scope and validation record:
`reports/github_historical_workspace_snapshot_2026-10-15.md`.

## STOP: 2026-10-15 ADVISOR-APPROVED NOVELTY SEQUENCE

This section is the current terminal decision and supersedes every older active
research priority below. Existing experiment records remain historical
evidence. Do not resume the from-scratch EBT Generalist or start specialist
work.

### What was decided

The advisor reviewed the ideas shown in the PPT and said they are reasonable,
but correctly asked whether any of them already work. They do not yet have
metric evidence. The approved sequence is therefore:

1. **First test the PPT energy idea.** Keep the existing FLUX.2/Restormer
   restoration models frozen. Learn a small restoration-energy model that sees
   the degraded input, the current restored candidate, and the VLM's explicit
   degradation description. Train it to recognize whether that candidate is
   correctly restored, still under-restored, over-restored, produced by the
   wrong expert, or has lost details. Use its gradient for only a few bounded
   correction steps.
2. **Require a real metric improvement at an adequate scale.** Begin with the
   frozen Noise development protocol already specified in `AGENTS.md`. Compare
   with the unchanged FLUX/Restormer output and a matched feed-forward refiner.
   A lower learned energy is not success by itself. The first 500 steps are
   only a code, memory, throughput, and numerical-stability check; missing the
   final metric target there does not reject the method. Once qualification
   passes, use the full 54,585-record Noise training split and run at least one
   complete effective data pass. Continue for predeclared additional passes if
   the source-disjoint validation curve is still improving. Decide failure only
   after adequate exposure shows saturation/no gain, not after an intentionally
   tiny pilot.
3. **If the energy idea does not work, switch to the advisor's DiT-MoE
   fallback.** Let a VLM first state the degradation type, severity, and
   composition, then use this diagnosis to control experts inside a FLUX/DiT
   restoration network. This fallback is sequential, not a parallel active
   experiment.

In one sentence: **try the project's own restoration-energy idea first; if it
fails, test explicit VLM-guided expert selection inside DiT.**

### Frozen experiment scale and available hardware — 2026-10-15

The experiment may run on one homogeneous node containing either
`8 x NVIDIA H200` or `8 x NVIDIA H100 80GB`. There is no cross-node execution,
and a mixed H100/H200 node is forbidden. Both profiles preserve global batch
`32`: H200 uses micro-batch `4/GPU` with accumulation `1`; H100 uses
micro-batch `1/GPU` with accumulation `4`. The H100 adjustment only reduces
per-forward memory and does not change steps, data order, or data passes.

The first energy experiment uses the complete `54,585`-record Noise training
split. The planned global batch is `32`, so one complete effective data pass is
`ceil(54,585 / 32) = 1,706` optimizer steps. The scale is:

| Stage | Total optimizer steps | Effective passes | Meaning |
|---|---:|---:|---|
| Systems qualification | `500` | `0.293` | Code, finite gradients, memory, throughput, validation, checkpoint/resume only |
| First scientific checkpoint | `1,706` | `1` | First result allowed to inform the learning curve |
| Primary formal run | `8,530` | `5` | Minimum adequately scaled main judgment |
| Conditional continuation cap | `17,060` | `10` | Continue only if pass 4→5 validation still improves without overfitting |

Validation and checkpointing occur at every completed pass. The matched table
must include the unchanged frozen FLUX/Restormer proposal, a parameter-matched
feed-forward refiner, and energy refinement with 1/2/3 candidate-gradient
steps. The official/sealed/internal/parent-test and frozen 660 pixels remain
forbidden for training, tuning, checkpoint selection, or deciding the step
count.

`100,000` steps are not currently planned because they equal approximately
`58.6` passes over this Noise split. Increasing beyond the 10-pass cap requires
a later explicit decision supported by the pass-by-pass development curve.

The exact duration is pending a real single-node throughput measurement on the
GPU family actually used. After the 500-step qualification, calculate normal
wall-clock hours from that family's measured seconds/step and calculate
allocated cost as `wall hours x 8 GPUs`. Never transfer an H100 timing estimate
to H200 or vice versa, and do not use a historical cross-node measurement.

The current single-node implementation is:

- `configs/foundir_proposal_relative_energy_noise_v1.json`
- `lowlight_restoration/proposal_relative_energy.py`
- `smoke/train_foundir_proposal_relative_energy_noise_v1_ddp.py`
- `smoke/run_foundir_proposal_relative_energy_noise_v1_8gpu.sh`

The launcher auto-detects one homogeneous 8-GPU H100/H200 node. Start with the
500-step systems qualification:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && \
PROPOSAL_ENERGY_RUN_ID=noise_energy_v1 \
PROPOSAL_ENERGY_TOTAL_STEPS=500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash smoke/run_foundir_proposal_relative_energy_noise_v1_8gpu.sh
```

Do not launch the 8,530-step continuation until qualification has produced
finite objectives, a checkpoint, development validation, exact-resume
evidence, and a measured hardware-specific time projection.

#### 2026-10-15 first-launch import failure and fix

Run ID `noise_energy_v1` terminated before model/data/optimizer access with
status 1. All eight workers failed while importing a reused legacy evaluator:
that evaluator imported OpenCV, whose native module required unavailable
`libGL.so.1`. This was a code dependency error, not an H200/H100, CUDA, NCCL,
memory, data, or model failure. Completed optimizer steps: `0`; allocated GPU
training work was effectively zero apart from process startup.

The new training script now implements its required manifest validation,
source-group-disjoint split checks, deterministic sampling, pair loading,
alignment, augmentation, and validation crops locally using PIL/NumPy. It no
longer imports the OpenCV-based evaluator or the legacy Restormer training
script. Python compilation, Bash syntax, JSON parsing, and diff checks pass.
Preserve the failed log and retry with a new run ID:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && \
PROPOSAL_ENERGY_RUN_ID=noise_energy_v1_retry1 \
PROPOSAL_ENERGY_TOTAL_STEPS=500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash smoke/run_foundir_proposal_relative_energy_noise_v1_8gpu.sh
```

#### 2026-10-15 retry1 qualification terminal result

`noise_energy_v1_retry1` completed all 500 optimizer steps on one node with
`8 x H100 80GB`, global batch 32 (`1/GPU x accumulation 4`). Wall time was
`.1976426635413534 hours`; allocated cost was `1.5811413112655281 GPU-hours`.
No OOM, NCCL error, non-finite objective, or protected-data access occurred.

On the fixed 256-record source-disjoint validation set, the frozen proposal was
`43.124160178005695/.050362067760033824`. Energy refinement produced:

- 1 step: `43.124949902296066/.05044266952609178`;
- 2 steps: `43.125487580895424/.050498137210524874`;
- 3 steps: `43.126018427312374/.05055382074897352`.

Thus the short run changes PSNR by only `+.00079` to `+.00186 dB` and worsens
LPIPS by `.000081` to `.000192`. It is not a method pass. The matched direct
refiner reached `43.231114752590656/.05958490435659769`: `+.106955 dB` but a
much worse `+.009223` LPIPS. Per the frozen scale policy, 500 steps are a
systems qualification rather than a scientific rejection.

A post-run schedule audit found that v1 tied cosine-LR horizon to the requested
stop step, so this 500-step run reached minimum LR at step 500. Resuming the
checkpoint to 1,706 would make LR jump upward and would not be one continuous
scientific trajectory. The fix freezes LR horizon at the 17,060-step cap for
all milestones. Preserve retry1 as systems/throughput evidence, but start a
fresh corrected run to the first scientific checkpoint at 1,706 total steps.
Do not resume `step_000500.pt` and do not jump directly to 8,530.

Full report:
`reports/foundir_proposal_relative_energy_noise_v1_qualification_2026-10-15.md`.
Checkpoint SHA-256:
`dffad058a9b9697fb76c580144b482e22dddbb5ae2bfb618e0b1134e1f107cdf`.

#### 2026-10-15 corrected pass-1 scientific result

Fresh run `noise_energy_v1_pass1` completed `1,706` optimizer steps on
`8 x H100 80GB` in `.6484166688554817` wall hours, or
`5.187333353174229` allocated GPU-hours. It completed normally with no OOM,
NCCL, non-finite, or protected-data-access error.

On the same frozen 256-record source-disjoint development validation, the
proposal remains `43.124160178005695/.050362067760033824`. Energy refinement
at 1/2/3 steps reaches respectively
`43.20002077519894/.05506254433294089`,
`43.21786245703697/.05692732973329839`, and
`43.22869956493378/.058651921212913294`. Thus more steps add only
`+.07586` to `+.10454 dB` while worsening LPIPS by `.00470` to `.00829`.
The direct refiner reaches `43.24551533162594/.06125720706040738`, likewise a
small PSNR gain with materially worse LPIPS. No arm passes the predeclared
`+.30 dB/-.005 LPIPS` technical gate.

This is a weak one-pass result, not a code failure and not yet the frozen
five-pass method decision. Resume the exact same run ID and checkpoint to
`8,530` total steps. Do not restart, change hyperparameters, or access the
official/sealed/internal/frozen-660 endpoints. Estimated remaining H100 time is
about `2.59 wall hours` and `20.75 allocated GPU-hours`, plus small validation
overhead. Full report:
`reports/foundir_proposal_relative_energy_noise_v1_pass1_2026-10-15.md`.

Pass-2 and pass-3 validations were subsequently produced at steps `3,412` and
`5,118` while the same run continued normally. At pass 3, the best perceptual
energy arm is Energy-1 at `43.19604617357254/.05484414789316361`, versus the
unchanged proposal `43.124160178005695/.050362067760033824`. This is only about
`+.07189 dB`, while LPIPS is still worse by about `.00448`. No energy arm has
passed the frozen `+.30 dB/-.005 LPIPS` gate. Continue unchanged through the
already-running five-pass endpoint; do not tune from these intermediate
records. Interim report:
`reports/foundir_proposal_relative_energy_noise_v1_pass2_pass3_interim_2026-10-15.md`.

### Why the MoE fallback is not a clean first claim

- FoundIR-v2 already uses MoE, but uses an SDXL backbone rather than DiT.
- MiM-DiT already combines MoE and DiT for All-in-One image restoration.
- MiM-DiT directly looks at degraded image features to choose experts. It does
  not first use a VLM to explicitly report degradation type, severity, and
  composition.
- Therefore the broad claim "MoE + DiT" is already occupied. The only current
  differentiation hypothesis is whether an explicit VLM diagnosis makes
  expert selection and composition better than ordinary image-feature routing.
- Merely adding VLM input to a router is not enough. A matched comparison must
  show better mixed/OOD restoration or expert composition using the same DiT
  backbone and training protocol.

### Evidence status

- Advisor approval: confirmed.
- PPT/action-contrastive energy mechanism: implementable hypothesis, not yet a
  working image-quality result. It must receive an adequately scaled test;
  500-step qualification alone cannot reject it.
- VLM-guided DiT-MoE: fallback hypothesis, not yet implemented or validated.
- No GPU experiment was launched by this documentation update: `0` wall-clock
  GPU hours and `0` allocated GPU-hours.

### Required communication style for future handoffs

Lead with the answer in simple Mandarin, then give technical detail. Explain
every necessary English term in ordinary words, explicitly separate facts from
hypotheses, and always identify both methods and the evaluation protocol behind
a comparison. Never turn advisor approval or code execution into a performance
claim. Report duration in normal wall-clock hours plus allocated GPU-hours.

## STOP: 2026-10-12 FORMAL REAL-DATA GENERALIST ACTIVE

The Generalist-only real-data continuation is active on the frozen
`654,881`-pair train split. It implements hierarchical composition/category
balancing, EMA, raw/EMA development validation, exact checkpoint provenance,
and resumable 8/24/32/40-GPU execution. CPU preflight, epoch-18 hash-bound GPU
qualification, every selected node's I/O gate, and the final multi-node
benchmark all passed before launch. The full readiness record is
`reports/foundir_vlm_energy_real_generalist_v1_readiness_2026-10-11.md`.

### 2026-10-12 formal topology selection

The user selected four nodes / 32 GPUs for the three-pass real-data Generalist
continuation. The frozen profile uses batch 2 per GPU, global batch 64, 10,233
optimizer steps per pass, and 30,699 total steps. Initial estimate: 25–29 wall
hours and 800–928 allocated GPU-hours. The selected initialization is the
epoch-18 joint checkpoint, SHA-256
`666fb152b50024357b6c44ea96469012e5405acb67a4ed08ad5618da1aca182e`.
The epoch-18/config-hash qualification and all four node-local I/O checks were
mandatory launch gates and passed before the active run started.

Those gates have now passed. Epoch-18 qualification completed on 8 x H200 with
status `foundir_vlm_energy_real_generalist_qualification_pass`, config SHA-256
`3c6c8f50092f35331e39c498332f538ba0d5e9ef942f7bbacd028a0c6e79e73e`,
and initialization SHA-256
`666fb152b50024357b6c44ea96469012e5405acb67a4ed08ad5618da1aca182e`.
The four selected nodes each passed the 4,096-pair, 32-worker I/O gate at more
than 120 pairs/s. Hardware identity proved that the allocation is
heterogeneous: node 0 has 8 x H200 and nodes 1--3 each have 8 x H100 80GB.
All 32 devices are Hopper compute capability 9.0. The corrected mixed-Hopper
benchmark passed all ten optimizer steps at `3.171203` mean measured schedule
seconds/step. Formal run `real_generalist_4node_mixed_hopper_v1` is active with
batch/GPU 2 and global batch 64. Do not restart or duplicate it.

## 2026-10-12 pass-1 real-data diagnostic — WIP, not the 660 gate

Pass 1 completed all 10,233 steps and wrote a resumable checkpoint, SHA-256
`7a19231ba772f7dbfc5927ca80b325338574573749bdc2c871104fcc02fe4d88`.
The 16,461-record 256x256-crop validation produced:

| Weights | Overall PSNR | Overall LPIPS |
|---|---:|---:|
| Raw | `26.706398` | `.199866` |
| EMA | `26.293138` | `.202940` |

Raw tier metrics were:

| Tier | Records | PSNR | LPIPS |
|---|---:|---:|---:|
| Singleton | 8,269 | `28.829671` | `.172490` |
| Double | 7,613 | `23.916114` | `.242836` |
| Triple | 579 | `33.070977` | `.025842` |

These crop metrics are not protocol-compatible with the frozen 660
full-resolution reference and must not be presented as a pass. They only show
that the model has not collapsed and isolate Double LPIPS as the current risk.

Triple contains only `blur+low_light+noise`, while Double contains five
heterogeneous categories:

| Double category | Records | Raw PSNR | Raw LPIPS |
|---|---:|---:|---:|
| `blur+noise` | 579 | `42.135151` | `.017810` |
| `haze+low_light` | 1,580 | `22.383274` | `.196602` |
| `low_light+noise` | 1,743 | `24.340798` | `.242655` |
| `haze+rain` | 1,580 | `20.962397` | `.277549` |
| `blur+low_light` | 2,131 | `21.945075` | `.312665` |

The sampler assigns one third of presentations to each composition tier.
Because Triple has one category and Double has five, the Triple category gets
approximately five times the exposure of each Double category. LPIPS also has
weight `.05` and enters the objective only every four optimizer steps, so PSNR
can improve before hard Double perceptual artifacts are resolved. Do not alter
the active frozen run; inspect pass 2/3, freeze one checkpoint, then perform one
matched 660 evaluation.

## Frozen explanation: what the old FLUX/Restormer system lacked vs FoundIR

This is the exact pre-real-EBT conclusion, not a new experiment proposal.

1. **Data scale/domain.** Most FLUX work used synthetic/task-specific data from
   roughly 1,600 clean sources. FoundIR metadata covers 1,010,064 pairs,
   including 378,440 Low-light-containing records; the current safe real split
   provides 654,881 training pairs. The referenced full archive is about 4.85
   TB.
2. **Training exposure.** FLUX experts were selected after hundreds to roughly
   10,000 task-specific steps. Released FoundIR Generalist is checkpointed at
   step 1,300,000 and its Low-light specialist at iteration 60,000 / epoch 413.
   Steps are not batch-normalized, but the exposure gap remains substantial.
3. **Joint composition learning.** The project system was VLM routing plus
   heterogeneous FLUX.2/Restormer experts, not one jointly trained Generalist
   over all eleven admitted singleton/mixed categories.
4. **Sequential interface failure.** On 53 correctly routed Triple records,
   Noise added only `+.069859 dB`; Low-light added `+6.807510 dB` but overshot
   GT luma by `+29.503`; Blur worsened 49/53 and lost `.272308 dB`.
5. **Simulator/domain mismatch.** Synthetic strength and operator order do not
   fully match FoundIR; the Noise/JPEG audit found a codec-order and
   noise-visibility mismatch.
6. **EMA maturity.** Short FLUX/Restormer selections lacked a mature EMA path.
   Real EBT adds EMA `.9999` and matched raw/EMA validation after every pass.
7. **Task alignment, not parameter count.** FLUX.2 is a much larger pretrained
   generative DiT, while FoundIR's approximately 36.3M-parameter UNetRes is
   restoration-native. Generative capacity did not substitute for paired
   pixel-restoration alignment.
8. **Full-system specialist refinement.** Full FoundIR can apply Low-light or
   Weather specialists after its Generalist. This is separate from the current
   Generalist-only gate; specialist research remains paused until every
   Generalist tier passes.

## STOP: 2026-10-11 GENERALIST-FIRST DECISION — SPECIALIST WORK PAUSED

The user has frozen the research priority: focus exclusively on improving the
EBT Generalist. Do not design, implement, train, tune, benchmark, or launch a
Low-light, Weather, trajectory-energy, residual-energy, or other specialist.
The 2026-10-11 specialist prior-art audit remains useful deferred research, but
specialist work may resume only after the EBT Generalist has exceeded the
released FoundIR Generalist under the matched gate below.

### Decisive Generalist gate

Use the existing 660 full-resolution whole-image composition set and the exact
frozen metric protocol: `full_rgb_uint8_image` PSNR and
`area_weighted_nonoverlap_512_rgb_uint8` LPIPS. Compare Generalist against
Generalist; neither side may run a specialist. The immutable released baseline
is in
`checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json`
(SHA-256
`ab4a1b0330c732432072917dca4ba74012bf7efe166e347e43aee4461a1e09e6`).

An unambiguous pass requires the EBT Generalist to beat the released FoundIR
Generalist in both metrics in every tier:

| Tier | Records | FoundIR Generalist PSNR | FoundIR Generalist LPIPS |
|---|---:|---:|---:|
| Singleton | 300 | `27.2323939564508` | `.1744119563086327` |
| Double | 300 | `22.245453630403492` | `.2066620594635684` |
| Triple | 60 | `24.9455711880497` | `.03775338436922618` |
| Overall, reporting only | 660 | `24.757709920211013` | `.17664758756638468` |

The overall row is descriptive and cannot hide a failed Singleton, Double, or
Triple tier. Full FoundIR Generalist-to-specialist values are not the target of
this gate and must not be substituted for the released Generalist values.

### Current run and exact next sequence

The 40,197,726-parameter synthetic `formal_3node_v1` run completed and its
epoch-18 checkpoint is the hash-bound initialization of the active real-data
run `real_generalist_4node_mixed_hopper_v1`. Preserve both runs. Do not restart,
duplicate, reconfigure, or interrupt the active four-node run.

After the real-data run reaches a terminal state:

1. audit all expected checkpoints, validation files, hashes, logs, and terminal
   markers;
2. select a checkpoint using development/validation evidence only;
3. stop or extend only from the frozen development trend, never from official,
   sealed, parent-test, or internal-test feedback;
4. freeze one Generalist checkpoint and run one matched 660 whole-image
   Generalist-versus-Generalist evaluation;
5. reopen specialist research only if every tier's PSNR and LPIPS gate passes.

The passed mixed-Hopper benchmark projected approximately `27.04` pure-training
wall hours and `865.36` allocated GPU-hours on 32 GPUs; validation and
checkpoint overhead make the operational estimate approximately 28--32 wall
hours / 896--1,024 GPU-hours.

## STOP: 2026-10-08 tensor-shard I/O fix READY — GPU QUALIFICATION NOT RUNNING

The one-shot tensor bank is complete: 60,000 train records x eight deterministic
views = 480,000 paired 128x128 samples in 469 contiguous shards (about 44 GiB).
All shard hashes pass. CPU build took `0.096557 hours`, with zero GPU use and no
triple/official/sealed access. Manifest SHA-256:
`561037670c88ead7a115687ef19da51eed061c329fd53b8e9b383b598fb5f3ff`.

The v3 51,200-sample schedule is duplicate-free and exactly balanced at 25,600
singleton / 25,600 double. Shard-local sequential CPU throughput is about
`6,002 img/s`, approximately 37x cached-PNG v2 and 2.34x above measured GPU
feed demand. Terminal bank report:
`reports/foundir_compositional_ebt_tensor_shards_v1_terminal_2026-10-08.md`.
GPU readiness report:
`reports/foundir_compositional_ebt_tensor_shard_smoke_v3_readiness_2026-10-08.md`.

Exact command:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_TENSOR_SHARD_RUN_ID=shard400_v3 CUDA_VISIBLE_DEVICES=0,1 bash smoke/run_foundir_compositional_ebt_tensor_shard_smoke_v3_2gpu.sh
```

Stop after v3 qualification; do not automatically launch 100k.

## STOP: 2026-10-08 target cache READY; cached v2 rerun READY — NOT RUNNING

The immutable 66,000-image clean+nuisance target cache completed in
`1183.632873 seconds` (`0.328787 hours`) using CPU only. All 60,000 train and
6,000 validation PNGs passed a second full SHA-256 pass; triple count is zero.
Cache size is approximately 24 GB. Target manifest SHA-256:
`64bcea7cdb95ae1f399d469dfbb76b4078dbc98e501081084e54a9d086b65178`.
Terminal report:
`reports/foundir_compositional_ebt_scaled_targets_v1_terminal_2026-10-08.md`.

Cached learning v2 is CPU-preflighted. It is bit-identical to v1 for the same
record schedule, crop/augmentation seeds, all three nuisance families, and
fixed validation tensors. The only intervention is cached-target read versus
on-the-fly regeneration. Readiness report:
`reports/foundir_compositional_ebt_scaled_learning_cached_v2_readiness_2026-10-08.md`.

The first `cached400_v2` launch failed at import before data/model/optimizer
access because the direct script path did not yet expose the workspace root to
the sibling `smoke` import. Both arms exited status 1 and no result exists. The
import ordering is fixed; direct-script import and tensor-identity regression
pass. Preserve the failed run and use `cached400_v2_retry1`.

Exact launch command:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_SCALED_CACHED_RUN_ID=cached400_v2_retry1 CUDA_VISIBLE_DEVICES=0,1 bash smoke/run_foundir_compositional_ebt_scaled_learning_cached_v2_2gpu.sh
```

Estimated wall time `0.03–0.25 hours`, allocated cost `0.06–0.50 GPU-hours`.
Stop after v2 and compare against real400_v1; do not auto-launch 100k.

## STOP: 2026-10-08 scaled EBT real-data learning smoke READY — NOT RUNNING

The next bounded diagnostic is CPU-preflighted: matched Plain and
Compositional 26,043,648-parameter EBT arms, one-step descent, batch 128, 400
updates, one homogeneous H100/H200 GPU per arm. Each arm sees the identical
51,200-record unique deterministic train schedule and the same 512-record
source-disjoint lower-order validation subset before and after training.

Target reconstruction for `none/jpeg/resample` nuisance families passes, and
the OpenCV BGR-to-RGB target conversion has an exact source-crop identity
test. No triple, sealed, or official data is accessible. No checkpoint is
saved. Readiness report:
`reports/foundir_compositional_ebt_scaled_learning_smoke_v1_readiness_2026-10-08.md`.

Exact launch command:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_SCALED_LEARNING_SMOKE_RUN_ID=real400_v1 CUDA_VISIBLE_DEVICES=0,1 bash smoke/run_foundir_compositional_ebt_scaled_learning_smoke_v1_2gpu.sh
```

Estimated wall time is `0.10–0.75 hours`, allocated cost `0.20–1.50
GPU-hours`. Stop after the terminal result; do not automatically launch 100k.

## STOP: 2026-10-08 scaled EBT H100/H200 kernel benchmark READY — NOT RUNNING

The first launch at `09:50:21Z` correctly stopped in hardware preflight: the
node exposed four H100 80GB GPUs but the initial gate required H200. No worker,
optimizer update, data access, checkpoint, or terminal result occurred. Preserve
run ID `h200_benchmark_v1` and its log as failure evidence; do not reuse it.
The corrected gate accepts four homogeneous H100 or H200 GPUs.

A bounded synthetic benchmark is CPU-preflighted for two credible larger EBT
sizes: 11,266,048 parameters (384x6) and 26,043,648 parameters (512x8). Each
is tested with official-style one-step and current-project three-step energy
descent. Four independent profiles run concurrently on four homogeneous
H100/H200 GPUs and scan local batches 8/16/32/64/128. Additional GPUs are not
wasted on duplicate profiles.

Config, Python, shell, exact parameter counts, a miniature higher-order full
update, and no-CUDA fail-closed behavior pass. The current Codex shell exposes
no CUDA device, so the benchmark has not started and has consumed zero
GPU-hours. Readiness report:
`reports/foundir_compositional_ebt_scaled_kernel_benchmark_v1_readiness_2026-10-08.md`.

Exact launch command:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_SCALED_BENCHMARK_RUN_ID=h100_benchmark_v1 CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/run_foundir_compositional_ebt_scaled_kernel_benchmark_v1_4gpu.sh
```

Estimated wall time is `0.08–0.75 hours`; allocated cost is `0.32–3.00
GPU-hours`. This benchmark uses synthetic tensors, reads no train/triple/
sealed/official data, and saves no checkpoint. At completion, inspect
`checkpoints/foundir_compositional_ebt_scaled_kernel_benchmark_v1/h100_benchmark_v1/result.json`.
Do not automatically launch full training from this benchmark.

## STOP: 2026-10-08 lower-order EBT scaled-data v1 READY — NO GPU TRAINING

The source-disjoint scaled lower-order data contract is frozen and verified.
It contains 60,000 train records (30,000 singleton + 30,000 double) and 6,000
validation records (3,000 + 3,000), drawn from 1,600 train and 200 validation
source IDs with zero source overlap. All five singleton degradations and all
ten unordered pairs are balanced. Triple records and parent test records are
both exactly zero.

Every selected degraded PNG and every unique source passed SHA-256
verification. A separate `--check` pass and independent manifest aggregation
also passed. No held-out triple, FoundIR 60-image triple endpoint, sealed
internal, or official FoundIR test was accessed. No GPU job was launched and
the cost is `0.0 GPU-hours`.

Frozen output manifest:
`manifests/foundir_compositional_ebt_scaled_data_v1.jsonl`, SHA-256
`d7add2be0ec4edff95034caee1287b4746acffcec76d43ab3f288503bbf3b449`.
Machine-readable metadata:
`manifests/foundir_compositional_ebt_scaled_data_v1.json`, SHA-256
`44f4546a941fa7e150f51eb6e148b7bfb64d4346cbb8a342d73a82c970e51c7a`.
Readiness report:
`reports/foundir_compositional_ebt_scaled_data_v1_readiness_2026-10-08.md`.

The exact next task is a bounded throughput/memory benchmark of a credible
larger EBT configuration on the available 8xH200 node, followed by a frozen
wall-time and GPU-hour estimate. Do not automatically launch 100k training
with the 146,432-parameter toy model. Do not tune on or inspect triple data.
Full scaled training requires a recorded model/data-scale governance decision.

## STOP: 2026-10-07 Compositional EBT 10k steps-only v2 TERMINAL PASS

The matched 10k run completed normally on two H100 80GB GPUs from `20:34:16Z`
to `20:41:24Z` (`7m08s`, approximately `.238 allocated GPU-hours`). Each
independent arm completed 10,000 optimizer steps. Both checkpoints, arm
results, terminal result, and advisor report exist; no process remains.

Both arms pass lower-order learnability on the 30-parent source-disjoint
singleton/double development set:

- Plain: `13.881658 -> 17.041216`, gain `+3.159558 dB`; Smooth L1
  `.020455223 -> .009882082`.
- Compositional: `13.881658 -> 16.099879`, gain `+2.218221 dB`; Smooth L1
  `.020455223 -> .012273887`.
- Plain minus Compositional: `+.941337 dB` and `-.002391806` Smooth L1.

Plain also leads on singleton and double subsets; its double gain is
`+3.575398 dB` versus Compositional `+2.231884 dB`. Train loss materially
decreases in both arms (last-25/first-25 ratios `.474358` and `.535821`). The
correct conclusion is that EBT is viable and 300 steps were insufficient, but
the current lower-order evidence does not support a Compositional advantage.

Result:
`checkpoints/foundir_compositional_ebt_lower_order_scale_v2/scale10k_v2/result.json`,
SHA-256
`9a4e73d261d9bc03b7996133588f90f80e1c130a4c1e4f5f5ecf83a2e61dd3c0`.
Terminal report:
`reports/foundir_compositional_ebt_lower_order_scale_v2_terminal_result_2026-10-07.md`.

No held-out triple, FoundIR 60-image triple endpoint, sealed internal, or
official FoundIR test was accessed. Stop before 100k or triple. The next step
requires an explicit choice: one-time frozen held-out-triple diagnostic,
official-recipe/model/data scale redesign, or 100k steps-only overfit study.
Do not infer that more steps alone will close the `0.941337 dB` lower-order gap.

## 2026-10-07 Compositional EBT 10k steps-only v2 READY — NOT YET RUNNING

The advisor/user directed `maybe after 10k then we consider scale to 100k or
not, execute`. Governance hold
`foundir_compositional_ebt_lower_order_scale_v2_design-3cae9b0bf329cfa0`
is resolved and bound to that exact instruction.

v2 restarts matched Plain and Compositional arms from the identical seed and
changes exactly one scientific variable from v1: optimizer steps increase
from 300 to 10,000 per arm. Model (146,432 parameters), data, crop/presentation
schedule, batch 4, AdamW, LR, weight decay, fixed candidate step size 10,
three-step differentiable descent, final-step checkpoint rule, and access
boundaries are unchanged. The two independent arms run concurrently on two
homogeneous H200 GPUs without DDP, preserving per-arm batch semantics.

CPU preflight passed with identical initial loss and finite higher-order
gradients. Python compilation, JSON parsing, shell syntax, governance, hashes,
and whitespace checks pass. Config SHA-256:
`7d307614e00fdaa9ec6cb0f26bcc3ea2124543708ed03327eb2f409a201c44bd`.
Trainer SHA-256:
`70fc49a45bda1e908b5a1e8a561952494067e494363a6def46cf66041182a3e7`.

The Codex shell has no visible GPU, so v2 has not started and has consumed zero
GPU-hours. Exact external 8×H200-node command (uses GPUs 0 and 1 only):

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_LOWER_ORDER_SCALE_RUN_ID=scale10k_v2 CUDA_VISIBLE_DEVICES=0,1 bash smoke/launch_foundir_compositional_ebt_lower_order_scale_v2_2gpu.sh
```

Estimated total wall time is 8–20 minutes and total cost `.27–.67 GPU-hours`.
Monitor the main log plus the Plain/Compositional arm logs named by the
launcher. The workflow freezes both final checkpoints before one lower-order
development evaluation, then stops. It does not access held-out triples and
does not automatically enter 100k steps.

## 2026-10-07 interpretation lock: EBT remains a viable path

The advisor/user explicitly rejects interpreting the 300-step toy pilot as
evidence that EBT itself is non-viable. Preserve this interpretation in future
work. The official ICLR EBT image recipe trains for 100,000 steps and publishes
successful ID/OOD denoising results; the local v1 pilot used only 300 steps per
arm and a 146,432-parameter restoration adaptation. Its small positive
development gains establish that the implementation can learn a correction,
while its frozen convergence-gate failure establishes only that this bounded
pilot was insufficient.

Brutal qualification: the local toy recipe is not otherwise identical to the
official image recipe. Official S1 uses a large 24-block, width-1536 model,
batch 128, one candidate-gradient step, learnable alpha initialized to 3000,
alpha LR multiplier 9000, peak LR `1e-4`, weight decay `.01`, 10,000-step
warmup, and 100,000 optimizer steps. Local v1 uses depth 2/width 64, batch 4,
three candidate-gradient steps, fixed alpha 10, LR `2e-4`, weight decay
`.0001`, no warmup, 60 train parents, and 300 steps. Therefore a steps-only
scale-up is the correct next controlled test, but not an official-recipe
reproduction and not sufficient by itself to claim paper-level EBT training.

Recommended next bounded decision: restart matched Plain/Compositional arms
from the identical seed for 10,000 fixed steps each while changing no training
or model hyperparameter. This is a 33.3x step-only intervention. Use two H200s
to run the two independent arms concurrently without changing per-arm batch or
optimizer semantics; using all eight GPUs for this 146k-parameter, batch-4 toy
would waste compute and likely increase overhead. Only after this clean test
should official stability/scale ingredients be introduced in a separate,
explicitly labeled experiment.

## STOP: 2026-10-07 matched Compositional EBT lower-order pilot TERMINAL FAIL

The authorized matched lower-order pilot ran to terminal completion on one
NVIDIA H100 80GB HBM3 from `18:40:56Z` to `18:41:41Z`. The measured training
and evaluation body took `24.270043` seconds; 600 optimizer steps cost
approximately `.006742 GPU-hours`. Both final checkpoints and the result are
complete. No process remains.

The frozen scientific status is `lower_order_learnability_fail`. Both arms did
improve the 30-parent source-disjoint development set over degraded input:
Plain `13.881658 -> 14.141973` (`+.260315 dB`) and Compositional
`13.881658 -> 14.175251` (`+.293593 dB`); Smooth L1 also improved by
`.001190057` and `.001337115`, respectively. Compositional minus Plain is only
`+.033279 dB` on this lower-order diagnostic.

The formal failure comes from the frozen convergence gate. Last-25/first-25
train-loss ratios are `1.361700` Plain and `1.377185` Compositional, above the
required `.9`. A read-only complete-epoch audit gives last-five/first-five
ratios `.952996` and `.966471`: modest improvement, still below the required
10% reduction. Changing crop difficulty makes the raw trend noisy, but the
frozen gate must not be rewritten after seeing the result.

Result:
`checkpoints/foundir_compositional_ebt_lower_order_pilot_v1/pilot_v1/result.json`,
SHA-256
`0bfcc30805387e22b8ede15e91e1aa13b6c3d8335d924695d0d99d746ce4b78f`.
Terminal report:
`reports/foundir_compositional_ebt_lower_order_pilot_v1_terminal_result_2026-10-07.md`.

Held-out R0 triple images, FoundIR 60-image triple endpoint, sealed internal,
and official FoundIR test access are all false. The fail created unresolved
hold `foundir_compositional_ebt_lower_order_pilot_v1-c31644dbf2370957`.
Stop: do not rerun, scale, change the gate, or access triple data without a new
advisor/user decision.

User hardware note for future experiments: one homogeneous `8×H200` node is
available. The current 146k-parameter pilot used one H100 because eight-way DDP
would add overhead rather than reduce its 24-second runtime. Design later
substantial experiments for the available 8×H200 node when data/model scale
can benefit from it.

## STOP: 2026-10-07 matched Compositional EBT lower-order pilot CPU-ready and authorized

The advisor/user explicitly authorized the next EBT step with `sure, execute
new step now`. The new governance hold
`foundir_compositional_ebt_lower_order_pilot_v1_design-5d467aa6f7d278cd`
is bound to that exact bounded scope and resolved. The older Phase-0 decision
was preserved under `reports/advisor_governance/decisions/`.

The frozen fail-fast pilot compares Plain dense descriptor mixing with
atomic/pairwise diagonal Compositional mixing. Both arms contain 146,432
parameters and use identical initialization, 60 singleton/double train
parents, 30 source-disjoint singleton/double development parents, data order,
crops, three-step differentiable energy descent, optimizer, and 300 optimizer
steps. Final step is fixed; no metric checkpoint selection occurs.

CPU preflight passed with identical initial Plain/Compositional loss
`0.0009887457126751542`, finite outputs, and finite higher-order gradients.
Three focused EBT unit tests, Python compilation, shell syntax, governance,
and whitespace checks pass. Config SHA-256 is
`a69e48b35fd1e58b03a864e8d1da15bfc5fdd1e3a8f5401c43b5ae67f6844d87`.

The current Codex shell has no `nvidia-smi` or `/dev/nvidia0`, so training has
not started and GPU-hours remain zero. Exact one-GPU H100/H200 command:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_LOWER_ORDER_RUN_ID=pilot_v1 CUDA_VISIBLE_DEVICES=0 bash smoke/launch_foundir_compositional_ebt_lower_order_pilot_v1_gpu.sh
```

Estimated wall time is 10–30 minutes; cost is `.17–.50 GPU-hours`. The run must
stop after the lower-order gate. If either arm fails to improve source-disjoint
development PSNR and Smooth L1 over degraded input, do not access triples. If
both pass, both checkpoints remain frozen and a separate governance decision
is required before any one-time 40-image held-out-triple evaluation. FoundIR
60-image triple, sealed internal, and official test access remain prohibited.

Readiness report:
`reports/foundir_compositional_ebt_lower_order_pilot_v1_readiness_2026-10-07.md`.

## STOP: 2026-10-07 Compositional EBT Phase-0 CUDA PASS

The one-GPU zero-update smoke completed normally on an NVIDIA H100 80GB HBM3
from `18:14:47Z` to `18:15:05Z` (18 seconds, approximately `.005 GPU-hours`).
It ended with `FOUNDIR_COMPOSITIONAL_EBT_PHASE0_END status=0` and no residual
experiment process.

Both exactly matched 143,360-parameter Plain and Compositional EBT arms had
finite nonzero candidate gradients, finite second-order parameter gradients,
and decreasing energy after one synthetic candidate update. Optimizer steps
were zero. Triple endpoint, sealed internal, and official FoundIR test access
were all false.

Result:
`checkpoints/foundir_compositional_ebt_phase0_v1/gpu_smoke_v1/result.json`,
SHA-256
`74d19448da5a498e0e5cc6e18fa2fcf78ddb19f87a88c3fb211c265b01ac22ff`.
Terminal report:
`reports/foundir_compositional_ebt_phase0_terminal_result_2026-10-07.md`.

This proves only CUDA/mechanical readiness, not restoration performance or
novelty. Stop before training. Exact next action is to design and freeze a
matched singleton/double-only Plain-versus-Compositional EBT pilot with no
triple checkpoint-selection leakage and a predeclared one-time triple endpoint.

## Historical Phase-0 CPU readiness — superseded by CUDA PASS above

The advisor/user explicitly chose the lean EBT first step and said `just
execute`. A new hash-bound governance decision authorizes only the pinned
official EBT paper/source audit, minimal matched Plain/Compositional energy
implementation, CPU tests, and a zero-optimizer-step CUDA smoke. It does not
authorize training, checkpoint selection, triple endpoint access, sealed
internal access, or official FoundIR test access.

The official EBT repository is pinned at
`third_party/ebt_official` revision
`19420cbeae655bbf11930219a675ade6897019e8` under Apache-2.0. No released
image checkpoint exists in that revision. The published image comparison is
COCO 2014 `128x128` denoising (`sigma=.1` train, `sigma=.2` OOD test), not
coupled FoundIR restoration. Its 99% statement is specifically 300 DiT versus
3 EBT forward passes.

Phase-0 implements exactly parameter-matched Plain and atomic/pairwise
Compositional EBT arms. Python compilation, three focused unit tests, finite
nonzero candidate gradients, finite second-order parameter gradients, and
one-step energy descent all pass on CPU. Both tiny arms contain 143,360
trainable parameters; optimizer steps are zero. CPU result:
`checkpoints/foundir_compositional_ebt_phase0_v1/cpu/result.json`, SHA-256
`fe4484e21c15307e578f3559c4aac8e51c20a946d992c56af8a31985e0fdcc74`.

At this historical checkpoint the coding shell exposed no CUDA device and no
executable system tmux, so the GPU smoke had not yet run. The workspace-local
tmux was subsequently repaired and the CUDA smoke completed as recorded in the
terminal section above.

Audit report:
`reports/foundir_compositional_ebt_phase0_audit_2026-10-07.md`.

## STOP: 2026-09-27 stage-trajectory first launch failed before step 1; retry ready

The first external 8×H200 launch started at `09:18:29Z` and failed during
`dist.init_process_group()` around `09:19:55Z`. All ranks reported NCCL/CUDA
401 (`operation cannot be performed in the present state`). It never reached
cache construction, model forward/backward, optimizer step 1, checkpoint
creation, double validation, or triple endpoint access. No process remains and
no terminal marker exists. This is a runtime initialization failure, not a
scientific result.

The failure matches the previously verified H200 NVLS issue. The launcher now
exports `NCCL_NVLS_ENABLE=0`, keeps P2P enabled, and first runs an eight-rank
scalar all-reduce plus full adapter DDP forward/backward rehearsal with zero
optimizer steps. Only a passing immutable rehearsal permits formal training.
The correction changes no scientific protocol and passes Python compilation,
shell syntax, unit tests, CPU preflight, governance, and `git diff --check`.

Exact retry:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/run_foundir_stage_trajectory_compatibility_v1_8gpu.sh
```

Do not launch a duplicate. The current Codex shell has no `nvidia-smi`, so it
cannot perform the retry itself.

## Historical: 2026-09-27 stage-trajectory compatibility v1 initial CPU readiness

The advisor explicitly selected the correctly-routed triple restoration
failure instead of fail-closed routing as the paper direction and then said
`execute`. A bounded `StageTrajectoryCompatibilityAdapter` tiny gate is now
fully implemented and CPU-ready. The current shell has no `nvidia-smi`, so no
GPU process has started.

The 257,235-parameter adapter is exact identity at initialization and fuses
the frozen LQ/Noise/Low-light/Blur stage trajectory with a bounded residual.
Router and expert weights remain frozen. Training and selection contain only
source-disjoint `15Lowlight_Blur` and `16Lowlight_Noise` records; no triple is
used. All 221 source groups from the 660-image endpoint are excluded.

- train: 49 Low-light+Blur + 40 Low-light+Noise, 51 groups total;
- validation: 13 Low-light+Blur + 6 Low-light+Noise, 13 groups total;
- optimizer steps: 1,000 on exactly 8 homogeneous H100/H200 GPUs;
- promotion: both double validation categories must improve PSNR;
- only after promotion: one frozen 60-image full-resolution triple endpoint;
- official FoundIR sealed test access: false.

CPU preflight, exact-identity check, finite-gradient rehearsal, two focused
tests, Python compilation, shell syntax, hashes, source isolation, governance,
and `git diff --check` pass. Governance hold
`foundir_stage_trajectory_compatibility_v1_design-e435439b0673fefc` is bound to
the advisor's explicit scope and resolved.

Exact external launch:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/run_foundir_stage_trajectory_compatibility_v1_8gpu.sh
```

Estimated total wall time: H200 `0.20--0.40 hours`; H100 `0.25--0.50 hours`.
Do not launch a duplicate. Readiness report:
`reports/foundir_stage_trajectory_compatibility_v1_readiness_2026-09-27.md`.

## STOP: 2026-09-27 triple sequential-interaction diagnosis COMPLETE

The user rejected risk-controlled fail-closed routing as the main paper
novelty and explicitly selected the correctly-routed triple restoration
failure for closest-method reproduction and targeted improvement. Do not
resume the aborted CURE two-track training and do not reopen routing or
official singleton evaluation.

A CPU-only audit measured all existing full-resolution stage images for the
53 correctly routed `18Lowlight_Blur_Noise` records. It ran no inference or
training and accessed no official FoundIR test image. Frozen sequence:
`Noise -> Low-light -> Blur`.

- degraded input: `10.110227 dB`;
- after Noise: `10.180086 dB` (`+0.069859`, nearly inert);
- after Low-light: `16.987595 dB` (`+6.807510`) but mean luma overshoots GT
  by `+29.503`;
- after Blur/final: `16.715288/.181321`; Blur worsens 49/53 records and loses
  `0.272308 dB` while luma, gradient, and Laplacian errors all increase;
- Released FoundIR Generalist: `24.986867/.038079`;
- Full FoundIR Generalist -> Low-light specialist: `33.238410/.032058`.

The same-transition double-composition control strengthens the diagnosis:

- `Noise -> Low-light` is `+9.899631 dB` on 43/43 correct
  Low-light+Noise records, but only `+6.807510 dB` on triple: a `-3.092122 dB`
  transfer deterioration. Its post-stage luma bias changes from `+0.057` on
  double to `+29.503` on triple.
- `Low-light -> Blur` is `+0.125191 dB` on Low-light+Blur (32/52 improve),
  but `-0.272308 dB` on triple (only 4/53 improve).

Thus routing error is ruled out on this subset and ordering is secondary:
skipping Blur leaves only `16.987595 dB`, still `7.999272 dB` below the
Generalist. The evidence supports a composition-domain/inter-expert
compatibility failure. The next bounded hypothesis is whether compatibility
learned only from singleton/double expert interfaces can transfer to an
entirely held-out triple. A direct triple-trained specialist may improve
metrics but is not, by itself, credible novelty.

A GT-derived scalar-exposure oracle further separates the failure. On the 53
correctly routed records, per-image least-squares scaling raises post-Low-light
from `16.987595` to `20.631794 dB` (`+3.644199`) and post-Blur from
`16.715288` to `20.477888 dB` (`+3.762601`). This is a non-deployable upper
bound, not a method result. Exposure overshoot is therefore substantial, but
scalar correction alone still leaves roughly 4.36--4.51 dB to the Released
Generalist, proving that color/structure/detail interaction also remains.

New evidence:

- `smoke/diagnose_foundir_triple_sequential_stages.py`;
- `reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.md`;
- `reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.json`;
- `reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.csv`;
- `smoke/diagnose_foundir_expert_interaction_transfer.py`;
- `reports/foundir_expert_interaction_transfer_diagnosis_2026-09-27.md`;
- `reports/foundir_expert_interaction_transfer_diagnosis_2026-09-27.json`;
- `smoke/diagnose_foundir_triple_exposure_oracle.py`;
- `reports/foundir_triple_exposure_oracle_diagnosis_2026-09-27.md`;
- `reports/foundir_triple_exposure_oracle_diagnosis_2026-09-27.json`.

## STOP: 2026-09-27 routing paper ablation COMPLETE; CURE tracks remain stopped

The user stopped both CURE two-track training jobs externally before baseline
training. They are user-aborted, have no terminal scientific result, and must
not be resumed. The older `LIVE` section below is preserved as historical
launch-time evidence and is superseded by this status.

The requested CPU-only paper reaggregation is complete. It reused the frozen
16,461-record development predictions and existing restoration metric
sidecars; it performed no model update, GPU work, new inference, or
official/sealed-image access. At exactly 13,603 accepted records (82.6377%
coverage), the ordinary eleven-class head ranked only by confidence has 92
wrong accepted routes, while the final multi-head router has 17: 75 fewer
errors, or an 81.52% reduction at matched coverage. The final accepted-error
rate is .1250%, versus .6763% for the ordinary head.

The initial categorical consensus was not automatically beneficial: at its
87.6435% coverage it had 201 wrong routes, versus 153 for the pure eleven-head
confidence ranking at the same accepted count. The defensible contribution is
therefore the final compositional/ambiguity/physical fail-closed design, not a
generic claim that more heads always help.

The 60-image Blur+Noise+Low-light subset independently confirms the remaining
failure is restoration composition: 53 correctly routed records still reach
only `16.715288/.181321`, while the seven Low-light fallbacks reach
`18.651010/.147128`. Do not reopen routing or singleton metric optimization.

Canonical new artifacts:

- `reports/foundir_routing_paper_ablation_2026-09-27.md`;
- `reports/foundir_routing_paper_ablation_2026-09-27.json`;
- `reports/foundir_routing_paper_ablation_2026-09-27.csv`;
- `reports/foundir_routing_risk_coverage_curve_2026-09-27.csv`;
- `smoke/reaggregate_foundir_routing_paper_ablation.py`.

Exact next paper task: use this frozen evidence to write the method/related-work
claim around risk-controlled fail-closed restoration routing. Any new model
experiment should target the separately identified triple expert-interaction
failure and requires a new, specific hypothesis; do not repeat the routing
ablation or restart CURE.

## 2026-09-27 CURE two-track formal training LIVE on two nodes

The user launched the two approved tracks on separate homogeneous 8×H200
nodes. `published_style` started at 01:21:23 UTC and `strict_holdout` at
01:24:23 UTC, with the frozen config SHA-256
`01a428b7eced044cc26e5a6a939010d23eade49627d6e776d19938acef823119`.
Both acquired distinct track-local launch locks, passed governance, full-data
preflight, H200/NCCL/DDP, and the no-update full-loss rehearsal, then entered
formal embedder training. At 01:34:24 UTC, published-style was at epoch 15/200
and strict-holdout had completed epoch 14/200. Both produced their immutable
epoch-10 embedder checkpoint (144,974,255 bytes), validation was finite, and
there was no traceback, OOM, NCCL error, or terminal marker. Latest observed
validation included published epoch 14 `loss=0.2894 accuracy=0.909` and strict
epoch 13 `loss=0.1876 accuracy=0.943`. Remote PIDs are not visible in this
shell; advancing shared logs and checkpoints are authoritative liveness
evidence. Do not relaunch either track.

This is only stage 1 of 3. Each track must finish embedder 200 epochs, baseline
300 epochs, and CURE 300 epochs before any frozen CCDD-11 triple evaluation.
Continue monitoring both existing logs. Do not access test metrics early, and
do not infer the long CURE-stage wall time from embedder throughput.

## 2026-09-27 CURE GPU rehearsal TERMINAL SUCCESS — data preflight PASS

The corrected `v1_retry2` no-update rehearsal on homogeneous 8×H200 ran
01:17:32–01:18:01 UTC and ended with both
`no_update_cuda_rehearsal_pass` and `CURE_GPU_REHEARSAL_END status=0` in
`logs/cure_two_track_gpu_rehearsal_v1_retry2.log`. The eight-rank NCCL/DDP
gate passed. The original CURE batch-three full-loss forward/backward had
finite loss `0.2047935277`, finite trainable gradients, and peak GPU memory
`37,857,186,304` bytes (~35.25 GiB); measured first-use forward/backward
was 1.4942 seconds. No optimizer update occurred. This single-batch timing
is **not** a defensible end-to-end training rate, because it excludes
DataLoader/HDF5, optimizer, DDP all-reduce, checkpointing, and the other two
stages. Preserve the earlier failed `v1` and `v1_retry1` logs as provenance.

Both full-data gates also passed: `smoke/build_cure_h5_v1.py --check` verified
three HDF5 databases with 25,704 rows each, and
`smoke/preflight_cure_two_track_v1.py` verified the two archive SHA-256
digests, source-disjoint views, zero strict-track triple exposure, class order,
and matched optimizer-update budgets (10,200 embedder + 60,300 baseline +
321,300 CURE = 391,800 updates per track). No CCDD-11 test evaluation or
FoundIR official sealed test has run. The one-line frozen-unused-parameter
runtime patch in the pinned CURE working tree is documented immediately below;
retain and disclose it in all reproduction reporting.

The approved two-track training is now technically gate-ready, **but not
started**. A genuine full-training time/cost estimate must be measured from
real stage throughput, not inferred from the 1.4942-second rehearsal sample.
Each track needs a homogeneous eight-H100/H200 node; independent nodes can run
the tracks in parallel, while one node requires sequential execution. The
Codex shell does not see the GPU node and cannot start that training itself.

## 2026-09-27 CURE GPU rehearsal retry1 failure — legacy unused DDP parameter

`logs/cure_two_track_gpu_rehearsal_v1_retry1.log` ended at 01:11:24 UTC with
`status=1`. It passed H200 profile, eight-rank NCCL/DDP, released checkpoint
loading, and the full-loss forward/backward; the extra gradient gate then
reported `missing or nonfinite restorer gradients`. Source audit identified
the exact static unused parameter `encoder.conv_esstesss.weight`, explicitly
retained in the released OneRestore architecture for checkpoint compatibility
but never used in `forward`. The pinned official CURE working tree now has a
**one-line local runtime patch** in `cure/models/onerestore.py` to set only
that legacy layer's `requires_grad_(False)`. The weight, state-dict key and
forward numerics remain unchanged. This patch is required for the original
default DDP wrapper to avoid an unused-parameter reduction failure on later
iterations; the clone's HEAD remains `a294116...` but its working tree is
intentionally dirty and must **not** be described as an unmodified checkout.
File SHA-256 after patch:
`63e78fb5c73fbe71aba69b5107235649c66766109b9874f30af478d981182920`.
Single-process CPU forward/backward confirmed every remaining trainable
parameter has a finite gradient. No optimizer update or CCDD-11 test
evaluation has run.

Next GPU-node command, preserving both failed logs:

```bash
CURE_REHEARSAL_RUN_ID=v1_retry2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang/smoke/run_cure_gpu_rehearsal_v1_8gpu.sh
```

Do not start training until this attempt ends with both
`no_update_cuda_rehearsal_pass` and `CURE_GPU_REHEARSAL_END status=0`, and
the full source/archive preflight passes. This is a runtime compatibility
patch, not a new scientific method or a changed loss/data protocol.

## 2026-09-27 CURE GPU rehearsal retry needed — checkpoint reader fixed

The first GPU rehearsal ran on homogeneous 8×H200 from 00:59:14 to 01:00:18
UTC. Profile and eight-rank NCCL/DDP checks passed. The single-GPU full-loss
rehearsal then failed **before forward/backward or any optimizer update** with
`KeyError: 'state_dict'`: its ad-hoc checkpoint reader expected a wrapped
training checkpoint, whereas the SHA-verified released OneRestore restorer is
a plain `OrderedDict`. The failure is preserved at
`logs/cure_two_track_gpu_rehearsal_v1.log` (terminal `status=1`).

`smoke/cure_gpu_rehearsal_v1.py` now uses the pinned CURE repository's
`cure.checkpoint.load_model`, which supports both formats. CPU loading of the
actual released restorer passed with all 189 state entries; Python compile
and launcher `bash -n` passed. The launcher now accepts a safe unique
`CURE_REHEARSAL_RUN_ID` to retain the failed log and refuse duplicate retries.
The exact next GPU-node command is:

```bash
CURE_REHEARSAL_RUN_ID=v1_retry1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang/smoke/run_cure_gpu_rehearsal_v1_8gpu.sh
```

Monitor `logs/cure_two_track_gpu_rehearsal_v1_retry1.log` for
`no_update_cuda_rehearsal_pass` and `CURE_GPU_REHEARSAL_END status=0`.
The three full HDF5 files are present, but do not launch training until their
`--check`/preflight and the corrected GPU rehearsal pass. No CCDD-11 test
metrics or FoundIR official sealed test were accessed.

HDF5 `--check` subsequently passed all three databases: each has exactly
25,704 aligned patch rows and the frozen class order; the strict baseline has
no triple classes. Full source/archive preflight is running separately. GPU
retry had not started at the latest log check.

## 2026-09-26 CURE two-track execution checkpoint — HDF5 build active

The user confirmed that the CURE two-track authorization represents the
advisor's explicit decision. The exact hash-bound governance decision remains
resolved (`smoke/advisor_governance.py check` passes). The three full HDF5
databases are being built serially by the pinned original CURE builder via
`smoke/build_cure_h5_v1.py`; the live process writes
`logs/cure_two_track_h5_build_v1_retry1.log`. At the latest observed check,
`published_baseline.partial.h5` had grown to about 60 GB and the first source
pass was about 23% complete. Do not start another builder or delete/overwrite
the partial file. No CURE training or test evaluation has run.

The independent no-update GPU rehearsal launcher
`smoke/run_cure_gpu_rehearsal_v1_8gpu.sh` passed `bash -n`; the three focused
CURE test modules passed four tests, and the rehearsal Python modules passed
bytecode compilation. This Codex shell has no visible GPU or `tmux`, so the
GPU rehearsal has **not** run. It first checks a homogeneous eight-H100/H200
profile, then eight-rank NCCL/DDP, then a batch-three original CURE full-loss
forward/backward without optimizer update on one GPU. Its log is fail-closed
against duplicate launch. Do not infer full-training wall time from the R0
pilot or call a CPU-only check a CUDA pass. Run the rehearsal on a real GPU
node and use its measured time/memory to estimate the 391,800 optimizer
updates per track before launching training.

Next: monitor the existing HDF5 builder; when it ends, run its `--check` and
`smoke/preflight_cure_two_track_v1.py`, then complete the GPU rehearsal and
only afterward launch the two approved original-code training tracks. FoundIR
official sealed test and the 60-image FoundIR triple endpoint remain excluded.

## 2026-09-26 CURE two-track reproduction: authorized preparation, no GPU result yet

The user explicitly approved the CURE original-code two-track reproduction and
confirmed they are the advisor or are relaying the advisor's explicit approval.
The new diagnostic-only governance hold
`cure_two_track_reproduction_v1_design-e847b2e4404be017` is resolved by an
exact hash-bound decision. The prior R0 decision was preserved verbatim at
`reports/advisor_governance/decisions/foundir_triple_onerestore_r0_v1_design_frozen-4409a69aba5bf9fe.json`.

The official CURE repository is pinned under `third_party/cure_official` at
`a294116ad3d7dce58723a21bdd64f74b085e7b13`. CCDD-11 revision
`4784b88382f8616008ac9ae1e9be36c75f871f4e` is being downloaded into
`data/third_party/ccdd11` (74.06 GB train and 16.39 GB test archives). The
released CURE/OneRestore checkpoints were independently SHA-verified against
their Hugging Face LFS OIDs. A CPU inference on the official low+haze example
completed with non-collapsed output; this is only a runtime preflight, not a
two-track result.

The approved protocol is frozen in `configs/cure_two_track_reproduction_v1.json`.
`smoke/prepare_cure_two_track_v1.py` creates source-disjoint symlink views and
keeps strict-track triple image exposure at zero while matching embedder
presentations per epoch. Its synthetic source-disjointness/exposure unit test
passes. `smoke/estimate_cure_h5_v1.py` will estimate HDF5 storage before any
large database build. No CURE training or CCDD-11 test evaluation has run. The
Codex shell has no visible GPU. Do not claim completion or issue a GPU launcher
until archive integrity, extracted source layout, HDF5 budget, and actual-image
output-collapse preflight pass. FoundIR sealed test and the 60-image FoundIR
triple endpoint remain excluded.

Original-code causal qualification: `EmbeddingClassifier` freezes its text
branch by default; `PromptEncoder` subsequently loads only that text branch,
not the trained visual classifier. Both tracks retain the same 12-row GloVe
prompt vocabulary. Thus seeing triple images in published-style embedder
training need not affect downstream restoration; the baseline's triple-image
exposure is the likely consequential difference. Confirm this by comparing
text-tensor hashes after both embedder runs, not by assumption.

Data acquisition update: both CCDD-11 ZIPs have now passed the Hugging Face
LFS SHA-256 checks (`half_train` `fd957ac7...`, `half_test` `2ed9fd82...`).
`half_test` was safely extracted (12,400 members); the frozen evaluation
manifest has 200 sources per triple class, 400 full-restoration records, and
2,400 selective records with same-source `1deg`/`2deg` targets. Manifest SHA:
`80b6233a87ef80dc650351f96881fd5b3488816231082c9821d1c11e6bdc7106`.
`half_train` safe extraction is in progress. The parallel downloader had one
short HTTP range; it resumed exactly from the recorded offset and the complete
train archive subsequently passed the published whole-file hash. The official
code repository and dataset card do not declare a license in the inspected
metadata; retain code/data for internal reproduction and do not redistribute
them without separate license review.

## 2026-09-26 closest-work claim gate: CURE pretraining exposure qualified

The latest literature/code audit is
`reports/foundir_triple_novelty_claim_gate_2026-09-26.md`. CURE already
covers selective subset supervision and order consistency. Its *CURE
fine-tuning* excludes triples, but the released protocol initializes from a
OneRestore baseline and prompt embedder trained with triple classes. This is
an end-to-end exposure distinction, not a claim of test-image leakage.
DAME-Net already evaluates unseen combinations up to four factors, and a 2023
paper already models degradation operation histories. No new method novelty
has been established.

Next research action: reproduce CURE published-style versus strict
end-to-end triple-free training under matched source split and update budget.
Only if a real gap appears should an order-conditioned intervention method
be designed and checked against degradation-history prior art. The terminal
R0 v1 result remains frozen; no new GPU job was launched by this audit.

## STOP: 2026-09-26 CSR novelty claim withdrawn after CURE full-text audit

The earlier provisional claim that matched counterfactual subset supervision
plus selective removal and held-out triple generalization might be a novel
combination is superseded. CURE (arXiv:2607.03044, July 2026) already trains
with matched selective-restoration pairs and per-factor intermediate losses,
adds order consistency, excludes triples from its fine-tuning stage, and
tests on triple composites. The earlier baseline and embedder stages do see
triple classes. Full evidence and citations are in
`reports/foundir_triple_novelty_cure_prior_art_correction_2026-09-26.md`.

R0 v1 remains terminal and scientifically failed due output collapse. A
read-only checkpoint probe confirms strongly negative pre-clamp residuals and
nearly all-black clamped outputs; it did not prove why residuals drifted.
No new GPU job has been launched. The next paper step is a bounded,
CURE-aware novelty formulation and closest-work comparison. The old CSR
mechanism cannot be presented as a new contribution.

## STOP: 2026-09-26 OneRestore-style R0 TERMINAL — RUNTIME SUCCESS, SCIENTIFIC FAILURE

The user-authorized R0 ran on one homogeneous 8×H200 node from 12:30:12 to
12:35:51 UTC and exited successfully (`status=0`). All three 1,200-step final
checkpoints, 210 full-image outputs, provenance and metric sidecars, terminal
`result.json`, report, and end marker are present and hash-verified. No
official FoundIR test or 60-image FoundIR triple endpoint was accessed.

The frozen primary gate **failed**. Development PSNR/LPIPS: A
`6.810193/0.967408`, B `7.007366/0.984490`, C `6.140267/0.950947`.
Held-out triple PSNR/LPIPS: A `7.010534/0.987038`, B
`7.182890/1.004814`, C `6.636498/0.966714`. B's LPIPS is worse than A by
0.017082; C's PSNR is worse than B by 0.867099 dB. Outputs have collapsed
near black: B mean pixel value is ~0.000005, with 99.9953% endpoint-saturated
pixels. All three arms are worse than the degraded input on every development
image. This is a valid completed runtime but not usable restoration evidence.

Canonical R0 terminal artifacts:

- `checkpoints/foundir_triple_onerestore_r0_v1/result.json`
- `reports/foundir_triple_onerestore_r0_v1_terminal_result_2026-09-26.md`
- `reports/foundir_triple_onerestore_r0_v1_terminal_audit_2026-09-26.md`
- `logs/foundir_triple_onerestore_r0_v1.log`

The R0 stop condition is met. Do not rerun the frozen R0 or automatically
start R1, R2, continuous severity, spatial factorization, or CSR. A future
design must address output collapse and preserve the held-out triple boundary.

## 2026-09-26 OneRestore-style R0 GPU PACKAGE READY — USER AUTHORIZED, NOT YET RUNNING

The user explicitly authorized this bounded diagnostic with the exact statement
`execute`. Do not ask for another approval to fix an in-scope implementation or
runtime defect. Authorization is limited to the frozen three-arm R0:
`A_unconditioned`, `B_fixed_categorical`, and
`C_fixed_categorical_cdrl`. Stop after the R0 terminal result. Do not
automatically enter continuous severity, spatial factorization, R1, R2, or CSR.

The complete implementation, evaluator, and launch path now exist and pass CPU
preflight. A protocol defect found during review was fixed before GPU use: the
trainer now freezes all three final checkpoints before the first held-out-triple
export/access. The official FoundIR test and the 60-image FoundIR triple endpoint
remain prohibited and have not been accessed.

Core launch files:

- `configs/foundir_triple_onerestore_r0_v1.json` — frozen SHA-256
  `415ffaa126393e0125ca070830596c413645919836e290b7a16970b2f27c2c64`
- `lowlight_restoration/onerestore_r0.py`
- `smoke/train_foundir_triple_onerestore_r0_ddp.py`
- `smoke/evaluate_foundir_triple_onerestore_r0_pyiqa_ddp.py`
- `smoke/run_foundir_triple_onerestore_r0_v1_8gpu.sh`
- `smoke/launch_foundir_triple_onerestore_r0_v1_8gpu.sh`

Verification completed on 2026-09-26 UTC: governance resolved; trainer preflight
including real frozen-VGG16 CDRL forward/backward passed; offline pyiqa 0.1.16
LPIPS-Alex preflight passed; 12/12 focused tests passed; Python compilation,
shell syntax, and whitespace checks passed. The rendered cache remains complete
at 620 PNGs. The Codex shell has no `nvidia-smi`, so it could not start the GPU
job. No R0 process, checkpoint, result, or GPU-hours exist yet.

Exact next command on one homogeneous eight-H100 or eight-H200 node:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash smoke/launch_foundir_triple_onerestore_r0_v1_8gpu.sh
```

Estimated total normal wall time, including three sequential arms, all exports,
LPIPS/PSNR, aggregation, sidecars, and terminal report: H200 0.75–2.0 hours
(center 1.25), H100 1.0–2.5 hours (center 1.5). Aggregate cost is H200 6–16
GPU-hours (center 10) or H100 8–20 GPU-hours (center 12), using exactly one
eight-GPU node. The dominant uncertainty is the absence of an external-node
benchmark, especially frozen VGG16 CDRL backward and DDP/shared-storage
throughput.

Expected terminal artifacts are
`checkpoints/foundir_triple_onerestore_r0_v1/result.json` and
`reports/foundir_triple_onerestore_r0_v1_terminal_result_2026-09-26.md`. At
terminal completion, report the R0 result and stop.

## 2026-09-26 triple counterfactual R0 CPU READINESS COMPLETE

The CPU-only reproducibility package prescribed by the literature memo is
complete. It freezes 130 source-disjoint AgenticIR synthetic parents: train
20 singleton + 40 double; development 10 singleton + 20 double; held-out 40
triple. The held-out set covers all ten triple combinations. The builder
expands 620 ordered-subset nodes and 1,710 counterfactual pairs.

Full build and independent `--check` pass. Six focused tests pass. Every source
hash is verified; triple gradient leakage and triple checkpoint-selection
leakage are both zero. Neither official FoundIR test nor the 60-image FoundIR
triple endpoint was accessed. No model was implemented or trained, no GPU ran,
and GPU-hours are zero.

Frozen identities:

- config SHA: `f099113d41867bf42f4143c70ebc7cbbc31d393368d81c7cc0fea3f9ca1b0cd8`;
- manifest SHA: `29fafafec8929d57136068433f081c8998053df0ec638b8f654a5f69e0c4ac85`;
- metadata SHA: `82302da80acde9de29a8964f4125c69b8f893686db81bc31bd5d140afa97f948`;
- readiness report:
  `reports/foundir_triple_counterfactual_reproduction_r0_cpu_readiness_2026-09-26.md`.

Two implementation defects were caught and fixed without changing protocol:
nested lattice operator objects are now deep-copied so mutation cannot evade
validation, and combination names now use the project canonical order instead
of alphabetical sorting.

This is not GPU-ready. A OneRestore-style diagnostic restorer/conditioning path
is a model-family/architecture experiment and requires a governance design hold
plus explicit user/advisor decision before neural implementation or launch.
There is no GPU command or defensible end-to-end runtime estimate yet. Exact
next task after authorization: implement only the R0 fixed categorical
descriptor + CDRL arm, add no-update/forward-backward/DDP checks, benchmark it,
publish normal wall-time and GPU-hour estimates, run the smallest R0 gate, and
stop before R1/R2.

## 2026-09-26 triple-composition literature/reproduction memo COMPLETE

The user directed a return to the literature-driven paper-novelty task after
freezing the successful double-composition aggregate. A read-only literature,
public-code, and local-evidence audit is complete. No GPU training or new
evaluation was launched; GPU-hours are zero.

The prior working hypothesis, factorized continuous type/severity conditioning,
cannot by itself be claimed as novel. CDG-IR (TMM 2026,
DOI `10.1109/TMM.2026.3702504`) already predicts multi-dimensional continuous
composite degradation intensity over five dimensions. F2D-Net (Knowledge-Based
Systems 2026, DOI `10.1016/j.knosys.2026.116411`) already factorizes a shared
transport backbone into degradation-specific increments with pixel-wise spatial
gating. DPPD, CDIR, UniCoRN, and MDIR further occupy dynamic prompting,
local/context-aware processing, controllable multi-head diffusion/MoE, and
continuous compact conditioning.

The revised, still-provisional research question is held-out composition
generalization through counterfactual subset supervision: train only on
clean/single/double synthetic compositions, explicitly supervise selective
removal against rendered subset targets, and evaluate on triples excluded from
training and checkpoint selection. Continuous severity and factorized control
are baselines, not the claimed contribution.

Local data truth supports this protocol: AgenticIR manifests preserve
per-component levels, operator parameters, application order, source hashes,
and source-disjoint splits. The 60 FoundIR Low-light+Blur+Noise validation
records have composition labels but no authoritative per-component severity;
such labels must not be invented. Existing Qwen evidence also shows
composition-complexity degradation: triple type-set exact `.75` and combined
type+severity exact `.4667`.

Full memo:
`reports/foundir_triple_composition_literature_reproduction_memo_2026-09-26.md`.

Exact next task: prepare a CPU-only R0 reproducibility package: hash-pin the
public OneRestore revision, build a source-disjoint matched synthetic
clean/single/double mini-manifest with held-out triples, validate
counterfactual-lattice parameter/order consistency and triple exclusion, add
unit tests, and emit a governance-bound design/readiness report. Stop before
any GPU launch for user/advisor review.

## Operating instruction — implementation fixes do not require repeated approval

The user explicitly directed that once an exact implementation fix is known,
apply it directly without asking for approval again. This applies to scoped
code/runtime compatibility fixes that do not change the scientific protocol,
data, model, optimizer, checkpoint-selection rule, or evaluation target. Record
the fix and verification evidence in this handoff. New scientific experiments
or protocol changes still require their normal authorization.

## 2026-09-26 route-aware 660-image integrated Restormer v2 TERMINAL COMPLETE

The user authorized the formal 660-image paper-table integration after the
Low-light+Haze Restormer experiment. A new route-aware, hash-bound artifact
replay completed normally from `2026-09-26T10:40:24Z` to `10:41:02Z` and
emitted `FOUNDIR_COMPOSITION_INTEGRATED_RESTORMER_V2_END status=0`.

The frozen router has 53 accepted `haze+low_light` routes, five
`low_light_fallback` routes, and two abstentions among the 60 Low-light+Haze
records. Only the 53 accepted routes use the new direct Restormer; fallback and
abstention behavior remains fail-closed. Consequently the deployed category
result is `21.168830/.199704`, not the capacity-only all-60 result
`22.215445/.182860`.

Terminal 660-image metrics:

- route-aware project: `25.647200 PSNR / .168062 LPIPS`;
- Released FoundIR Generalist: `24.757710/.176648`;
- Full FoundIR Generalist -> Specialist: `27.169232/.167938`;
- project minus Generalist: `+.889490 dB / -.008586 LPIPS`;
- project minus Full FoundIR: `-1.522032 dB / +.000124 LPIPS`.

The 300-image double aggregate is now `24.321537/.192313`, beating Full
FoundIR `24.184117/.194418` by `+.137420 dB / -.002105 LPIPS`. Freeze the
double aggregate. Triple Low-light+Blur+Noise remains unchanged at
`16.941122/.177332` versus Full FoundIR `33.677913/.031967` and is the next
paper-novelty diagnostic target.

This run performed no fresh neural inference for the 607 unchanged records.
It generated 660 new per-record audit sidecars by verifying and binding every
source sidecar and project output SHA-256, then replaying their immutable
metrics; this provenance must be disclosed in the paper. It avoids about 10.75
hours of redundant 8-GPU inference without changing the deterministic result.
All 660 sidecars are complete, there are zero temporary files, and official
FoundIR test access is false. Wall time was 38 seconds; GPU-hours were zero.

- config SHA-256:
  `c30581a701f6b08b6c66503b7b681bcc1f386ca43799a74d387367ab280ad6ad`;
- result SHA-256:
  `511940015fb3e3898f7da4ce7f31958dfb7f0bf9175062002de801cd2fcf7c1f`;
- terminal report:
  `reports/foundir_composition_integrated_restormer_v2_terminal_result_2026-09-26.md`.

Exact next task: return to the literature-driven paper novelty work. Analyze
the triple failure and reproduce the most relevant compositional-restoration
mechanism before proposing or launching a new scientific method. Do not
automatically add another category-specific specialist.

## 2026-09-26 Low-light+Haze direct Restormer v1 TERMINAL COMPLETE

The frozen `09Lowlight_Haze -> clean` Restormer experiment completed normally
after exact-resume from the immutable step-3000 checkpoint. The resumed run
started at `2026-09-26T07:08:48Z`, all eight ranks reported
`resume_step=3000`, and the runner emitted
`FOUNDIR_LOWLIGHT_HAZE_RESTORMER_V1_END status=0` at
`2026-09-26T08:37:16Z`. There was no new traceback, OOM, CUDA failure, or NCCL
warning after the resumed-run start marker.

Frozen development selection:

- step 6000: `21.732965 PSNR / .200751 LPIPS` on 128 source-disjoint,
  full-resolution development records;
- step 3000: `20.767505/.221818`, so step 6000 improves by
  `+.965460 dB / -.021067 LPIPS`;
- the frozen rule
  `maximize_worst_normalized_development_gate_margin_then_earlier_step`
  correctly selected step 6000;
- selected checkpoint:
  `checkpoints/foundir_lowlight_haze_restormer_v1/step_006000.pt`, SHA-256
  `a6f9523a6b9b44ab95d693c29038bb12378ba050b575104e62b6d9b5d1ef0cc3`;
- development result SHA-256:
  `7f6289c661c9cd9ecfc8ec4cdb6ed3a6ef329d915dde9bbe25691562c00d83f3`.

Frozen final 60-image comparison, which was not used for checkpoint selection:

| System | PSNR | LPIPS |
|---|---:|---:|
| Project Restormer step 6000 | 22.215445 | .182860 |
| Full FoundIR | 20.939309 | .159126 |
| Project minus Full FoundIR | +1.276136 dB | +.023734 |

The experiment substantially improves the previous project result
`13.112670/.251747` by `+9.102775 dB / -.068887 LPIPS`. It beats Full FoundIR
on PSNR but not LPIPS, so `beats_full_foundir_both_metrics=false`; the double
composition objective is not yet achieved on both metrics. This is a mixed
development/comparison experiment, not the official sealed singleton test, and
`official_foundir_test_accessed=false`.

Terminal artifacts and audit:

- final result:
  `checkpoints/foundir_lowlight_haze_restormer_v1/final_comparison/result.json`,
  SHA-256 `26266d6ff414919fab77b6c01126afb56c8dba6f43e154483fd75e796d9d78ae`;
- exactly 60 final PNGs and 60 per-record JSON sidecars, with no temporary
  files; record IDs match one-to-one;
- all 60 sidecars have `status=complete`; recomputed means match the terminal
  result, and all 60 recorded output SHA-256 values match their PNGs;
- resumed terminal wall time: 1 hour 28 minutes 28 seconds on 8xH200,
  approximately 11.80 GPU-hours;
- the earlier corrected run used 2 hours 27 minutes 49 seconds before external
  node/session termination; both full-run segments total 3 hours 56 minutes
  17 seconds of active 8-GPU time, approximately 31.50 GPU-hours, excluding
  short diagnostics and downtime;
- do not rerun this frozen v1 automatically. The exact next scientific task is
  to analyze and improve the remaining `.023734` LPIPS gap without using the
  final 60-image comparison for iterative checkpoint selection; any new
  experiment/protocol requires its normal authorization.

## 2026-09-26 Low-light+Haze NCCL recovery correction

The initial run and the first `NCCL_P2P_DISABLE=1` diagnostic both failed before
optimizer step 1. Full NCCL evidence identifies `transport/nvls.cc` with CUDA
error 401 while NVLS multicast remained enabled. P2P disabling does not disable
NVLS. The exact code-only correction is therefore `NCCL_NVLS_ENABLE=0` with P2P
retained, explicit CUDA `device_id` process-group binding, current-device CUDA
seeding, and a two-stage diagnostic (scalar all-reduce followed by Restormer DDP
forward/backward with no optimizer step). If that bounded diagnostic passes, the
same frozen run resumes automatically. No scientific protocol changes.

Implementation and CPU-side verification completed at 2026-09-26 UTC:

- `NCCL_NVLS_ENABLE=0`, `NCCL_P2P_DISABLE=0` are enforced by recovery;
- trainer, evaluator, and diagnostic explicitly bind `device_id=device`;
- trainer seeds the CPU default generator and current CUDA device only;
- diagnostic performs scalar all-reduce before constructing Restormer DDP;
- the tmux starter now enters recovery rather than bypassing it;
- verbose `NCCL_DEBUG=INFO,COLL` is scoped to the bounded diagnostic and is not
  intentionally inherited by subsequent full training;
- Python compilation, shell syntax, governance binding, frozen preflight, and
  launcher `--check` all pass;
- the current Codex shell has no `nvidia-smi`, so GPU execution cannot be
  initiated from this shell. On the existing 8xH200 shell, run exactly:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/recover_foundir_lowlight_haze_restormer_v1_8gpu.sh`.

Runtime update at 2026-09-26T03:38:07Z: the corrected diagnostic passed all
stages on 8xH200 and wrote
`checkpoints/foundir_lowlight_haze_restormer_v1/nccl_recovery/result.json`.
The unchanged frozen run then started automatically at 03:32:27Z and reached
step 150/6000 with finite loss, no new traceback, and 8-GPU utilization. The
old CUDA 401 lines earlier in the append-only log belong to the 03:00 failed
attempt. Observed 128-crop throughput is about 1.55 seconds/step. Because crops
increase to 192 at step 1500 and 256 at step 4000, the original 1.7--2.7 hour
estimate is not yet supported by runtime evidence; a cautious first projection
is roughly 6.5 hours for training plus milestone/final evaluations, to be
refined after the first 192-crop interval. The first recovered run inherited
diagnostic NCCL verbosity; the recovery script has been corrected so future
full runs will not inherit it. The current run remains scientifically valid.

Runtime update at 2026-09-26T07:05:20Z: that GPU node/session stopped abruptly
at 06:00:16Z after reporting step 5450/6000. The GPU telemetry stopped at the
same time, the runner lock is free, and there is no new traceback, OOM, NCCL
warning, or normal END marker; this is consistent with external node/session
termination rather than a model/code failure. The latest immutable resumable
milestone is step 3000 (`2d8baa0a...`); step 5450 was not checkpointed. Step
3000 development is `20.767505 PSNR / .221818 LPIPS`, improved from step 1000
`18.356467/.301158` but still short of the frozen `20.939309/.159126` gate.
Launcher/preflight checks pass and exact-resume is automatic. On a fresh
8xH200 shell run:
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/launch_foundir_lowlight_haze_restormer_v1_8gpu.sh`.
It resumes at step 3000; expected remaining wall time is about 1.4--1.8 hours
plus startup variance, approximately 11.2--14.4 GPU-hours.

## Historical readiness record — superseded by Restormer v1 terminal completion

The user explicitly authorized `yes, restormer, execute`. A bounded direct
`09Lowlight_Haze -> clean` Restormer workflow is frozen and CPU-ready. It uses
the public task-native Restormer deraining checkpoint as initialization and no
released FoundIR weights. Every source group represented in the frozen
660-image comparison is excluded from training and development selection.

- training: 75,370 records / 7,514 source groups, 6,000 steps, global batch 32;
- development selection: 128 source-disjoint full-resolution records at steps
  1,000, 3,000, and 6,000;
- final gate: one frozen 60-image `09Lowlight_Haze` comparison against Full
  FoundIR `20.939309/.159126`, not used for checkpoint selection;
- config SHA-256:
  `d60d67ba2a285fa0cb0691ac04f9630e2dc3129f1bf95d7563934ce777235cd3`;
- manifest SHA-256:
  `afa0e22b27c9c9773be67454a36c80bfcc7fd3e26f97b91be7577d17bd8b8a0e`;
- readiness:
  `reports/foundir_lowlight_haze_restormer_v1_readiness_2026-09-26.md`;
- governance hold `lowlight_haze_direct_restormer_v1_design-e3c715f7298fa7c2`
  is resolved by the user's scope-limited decision;
- CPU preflight, Python compilation, shell syntax, source isolation, immutable
  hashes, H100/H200 guards, and launcher `--check` pass.

No GPU is visible in the current Codex shell, so the job has not launched and
no training result exists. Exact next action is one external launch on a
homogeneous 8xH100 or 8xH200 node. Total estimate: H200 100--160 minutes
(center 125; 13.3--21.3 GPU-hours), or H100 130--200 minutes (center 160;
17.3--26.7 GPU-hours). The job includes training, all milestone evaluations,
selection, and final comparison. Do not launch a duplicate or automatically
start Low-light+Blur afterward.

## 2026-09-25 official SOW600 Released FoundIR comparison TERMINAL COMPLETE

The user-authorized direct singleton comparison completed normally on the same
600 official SOW pairs. The frozen terminal project v8 beats Full FoundIR on
both PSNR and LPIPS in every category:

| Category | Project PSNR / LPIPS | Full FoundIR PSNR / LPIPS | Project margin |
|---|---:|---:|---:|
| Blur | 28.092635 / .144089 | 25.577556 / .180644 | +2.515080 dB / -.036555 |
| Noise | 40.048323 / .056601 | 39.807950 / .064671 | +.240373 dB / -.008070 |
| Haze | 22.677420 / .110437 | 19.084373 / .224938 | +3.593046 dB / -.114500 |
| Rain | 37.809175 / .052901 | 34.293876 / .079429 | +3.515298 dB / -.026529 |
| Low-light | 19.322459 / .148799 | 18.240826 / .238644 | +1.081634 dB / -.089846 |

The 600-image weighted means are Project `29.168753/.103485`, Released
Generalist `25.821272/.169251`, and Full FoundIR `26.626220/.164044`.
Project minus Full FoundIR is `+2.542533 dB/-.060559`.

- result:
  `checkpoints/foundir_official_sow600_released_full_pipeline_v1/result.json`,
  SHA-256 `afdef408521989bc449497c637403642612a9687e5004878f4687a765532b9d6`;
- terminal report:
  `reports/foundir_official_sow600_released_full_pipeline_terminal_result_2026-09-25.md`;
- artifacts: 600 Generalist PNGs/metadata, 250 Specialist PNGs/metadata, 600
  metric sidecars;
- independent audit: pass;
- runtime: `2026-09-25T01:19:59Z` to `01:38:23Z`, 18m24s on 8xH200,
  approximately 2.45 GPU-hours, status 0;
- training/model updates: none.

The singleton objective is already achieved; do not reopen singleton training
merely to beat FoundIR. Per the user's ordered plan, the next research stage is
double compositions, followed by triple composition only after the double stage
is frozen. Keep official sealed singleton data metric-only and outside all
training and iterative checkpoint selection.

## Historical readiness record — superseded by terminal completion above

The user explicitly authorized a direct singleton comparison on the same 600
official SOW pairs: terminal project v8 versus Released FoundIR Generalist and
the documented Generalist -> Specialist pipeline. This is evaluation only;
official pixels/GT remain forbidden for training or gradient updates.

The implementation and full CPU preflight are complete. Preflight verified all
600 manifest rows, 1,200 LQ/GT hashes and dimensions, released checkpoint
hashes, canonical commit `5003e1e` project metrics, `pyiqa` 0.1.16, LPIPS
weights, governance, and syntax. No GPU is visible in the current Codex shell,
so the run has not launched and the output directory is empty.

- config:
  `configs/foundir_official_sow600_released_full_pipeline_v1.json`, SHA-256
  `3dff5bb4341d3f97c15470fe2d8bfaab7c9b22af051c13171d38fa986224802d`;
- readiness:
  `reports/foundir_official_sow600_released_full_pipeline_readiness_2026-09-25.md`;
- expected scope: 600 Generalist outputs, 200 Weather Specialist outputs, 50
  Low-light Specialist outputs, and 600 metric sidecars;
- estimate: 24--38 minutes on 8xH200 (center 30), or 30--48 minutes on 8xH100
  (center 38).

Exact next action is one launch on a homogeneous 8-GPU node, then monitor to
terminal completion, independently audit artifacts, write the terminal report,
and freeze the per-category singleton gaps. Per user direction, subsequent
quality work must address singleton categories first, double compositions only
after singleton success, and triple composition last.

This is the current FoundIR operational handoff. It supersedes
`reports/foundir_current_handoff_2026-09-21.md` for project status and the
next action. Historical results remain immutable evidence and must not be
rewritten as the current state.

## 2026-09-24 full FoundIR composition comparison TERMINAL COMPLETE

The follow-up 660-image full-resolution development comparison is now complete
for the complete released FoundIR inference pipeline. The earlier 2026-09-23
result ran the released Generalist only and must be labeled accordingly. The
new run reused those immutable Generalist PNGs, then applied the documented
Weather specialist to 120 records and Low-light specialist to 240 records;
the other 300 records retained Generalist output.

Overall 660-image means are:

- project: `24.914822 PSNR / .172793 LPIPS`;
- Released FoundIR Generalist: `24.757710 / .176648`;
- Full FoundIR Generalist -> Specialist: `27.169232 / .167938`;
- project minus Full FoundIR: `-2.254410 dB / +.004855 LPIPS`.

Thus the project slightly beats the released Generalist-only aggregate but
does not beat the complete FoundIR pipeline overall. Triple
Low-light+Blur+Noise is the dominant gap: project `16.941122/.177332` versus
Full FoundIR `33.677913/.031967`.

Terminal identities:

- result:
  `checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json`;
- result SHA-256:
  `ab4a1b0330c732432072917dca4ba74012bf7efe166e347e43aee4461a1e09e6`;
- config SHA-256:
  `c1dc876671622fd4d0541b52ca28a6cf95c036f181249d275ba370c98d835760`;
- report:
  `reports/foundir_composition_full_pipeline_terminal_result_2026-09-24.md`;
- artifacts: 360 specialist PNGs, 360 specialist metadata sidecars, 660 metric
  sidecars;
- runtime: `2026-09-24T23:11:09Z` to `23:16:22Z`, 5m13s on 8xH200,
  approximately .70 GPU-hours;
- model updates: none; official FoundIR test accessed: false;
- independent terminal audit: pass.

This remains `DIAGNOSTIC ONLY / 非主线`. It does not alter the already-terminal
official SOW600 v8 pass. Exact next action is paper/architecture analysis of
the observed mixed-degradation gap; do not restart this comparison or reopen
official singleton training.

## Current status in one paragraph

The five project-trained single-type systems already pass their complete
source-disjoint development mean SOW gates. In particular, Blur is not an
unfinished development task: the frozen Blur FLUX.2 step-10000 scalar-residual
system reaches `26.559141 PSNR / .151993 LPIPS`, and the fail-closed final
eleven-class router integrated over all 2,595 Blur development records reaches
`26.614726 / .156974`; both are above the Blur SOW thresholds `25.9 / .18`.
The user confirms that the final official sealed evaluation is also complete
and all five SOW types are well above threshold. The final all-five official
result artifact, exact metrics, and SHA-256 are not discoverable in the current
workspace snapshot as of `2026-09-23T05:28Z`; the latest conventional
`official_result.json` files visible here are historical v1--v6 intermediate
rounds and must not be presented as the current final outcome. Bind the final
official artifact here as soon as its path is supplied or synchronized.

## Scope and system identity

- Workstream: VLM Guided All-in-One restoration only; the separate bright/color
  model workstream remains out of scope.
- Categories: Blur, Noise, Haze, Rain, and Low-light plus the six admitted
  mixed categories; JPEG remains excluded.
- Deployed structure: Qwen degradation inference, fail-closed routing/fallback,
  and frozen project-trained FLUX.2 or Restormer experts. It is a VLM-routed
  multi-expert system, not one monolithic restoration checkpoint.
- The official 600-pair scope was authorized on 2026-09-21. Authorization:
  `reports/advisor_governance/foundir_official_sow600_qwen_authorization_2026-09-21.json`,
  SHA-256 `001c70b975e6c7b3cc1683e1b9ac82e0a62d73f06020290c415ad308132fefc0`.
- The authorization permits documented retraining, tuning, and retesting, but
  official images/GT remain forbidden as training examples or gradient targets.

## Frozen source-disjoint development evidence

Registry:
`reports/foundir_five_type_source_disjoint_development_registry_v1.json`.

| Type | Frozen project system | Records / groups | PSNR | LPIPS | Mean SOW |
|---|---|---:|---:|---:|---:|
| Blur | FLUX.2 step-10000, `LQ + .75*(FLUX-LQ)` | 2,595 / 165 | 26.559141 | .151993 | pass |
| Noise | FLUX.2 step-400, `LQ + .30*(FLUX-LQ)` | 1,859 / 34 | 40.150553 | .068880 | pass |
| Haze | FLUX.2 step-3000 | 1,580 / 158 | 23.311888 | .101155 | pass |
| Rain | project-trained Restormer step-5000 | 835 / 117 | 42.284347 | .047452 | pass |
| Low-light | FLUX.2 step-400, `LQ + .925*(FLUX-LQ)` | 1,400 / 21 | 19.779135 | .154058 | pass |

Blur's final-router record-weighted development evidence is
`checkpoints/foundir_blur_record_weighted_final_eleven_router_v2/result.json`:

- 2,223 correct routes, 372 safe abstentions, zero wrong routes;
- `26.614726 PSNR / .156974 LPIPS`;
- PSNR margin `+.714726 dB / +2.7596%`;
- LPIPS headroom `.023026 / 12.7920%`.

Do not call Blur unresolved based on the older 128-record source-group-balanced
rehearsal. That rehearsal intentionally overweighted rare hard groups and is a
tail/distribution diagnostic, not the complete record-weighted Blur result.

## Work completed after the stale 2026-09-21 handoff snapshot

### Official authorization and intermediate rounds

The 2026-09-21 handoff stopped while the first Noise Restormer experiment was
still described as pending. The following work subsequently completed:

1. Official SOW600 access was explicitly authorized for the Qwen-routed five
   SOW categories, 600 pairs total.
2. Historical official v1 established the initial full-resolution baseline.
3. Historical v2 froze improved Rain and Haze paths; both official categories
   passed.
4. The first Noise Restormer attempt (historical v3) failed official Noise.
5. A project-trained Noise MSE Restormer reached `42.719396/.066883` on all
   1,859 development records. A 250-step MSE+LPIPS continuation reached
   `42.629757/.056313` on the same development set.
6. Historical official v4 passed Noise PSNR but narrowly failed LPIPS:
   `39.511948/.105248`.
7. Historical official v5 passed Noise at `39.467855/.094125`. At that point
   Noise, Haze, Rain, and Low-light all passed that intermediate official round.
8. Historical official v6 was a Blur-motion audit. It returned
   `21.831774/.389771` for that candidate and failed. It is preserved as a
   rejected intermediate candidate, not the current final sealed result.

Historical official result SHAs:

| Round | Result SHA-256 |
|---|---|
| v1 | `9a17808e6982f363e92e0423681715b5d879847f0d1b1e941666a918bdb2fdb0` |
| v2 | `b238430b24b77cc6cae2ff3eab3a26fd07c5e61c39085a960285a70485f58ebe` |
| v3 | `cea5206525414eab43c71f503960eb8ad6a6b5643fdf480656a0ceec1940eddd` |
| v4 | `8a0221d6a8257b9e6f00a3699432c3c97304ec649d61fa73bc514c9a5f4d8fb6` |
| v5 | `61948d5df57e85cd029a638779323a72e9ce1e45e5548a0cac22a0e4ee14ca72` |
| v6 | `22256877a8fa0e0fd4b8582c6a4c3447d66cb8e432cfef80f7dc46689f15cd61` |

### Blur diagnostics completed after v6

- The training-only severity audit covered 71,393 Blur records represented by
  2,782 source groups. Representative mean input PSNR was `22.373173`, and
  2,021 groups fell inside the disclosed official input-PSNR range. Evidence:
  `reports/foundir_blur_training_severity_audit_v1.json`, SHA-256
  `4478f834bcadde8afcf23babf6b82f05246c9487f90e81d3a6319442e3956c61`.
- The 2,000-step heavy motion MSE+LPIPS Restormer gate failed at
  `22.200263/.229873` on 44 source-disjoint development records.
- The external FFTFormer diagnostic failed qualification at
  `19.949758/.188968` on the 150 official Blur records and is non-deployable.
- These rejected diagnostics do not invalidate the already-passing complete
  source-disjoint Blur development system or the user-confirmed final sealed
  five-type pass.

## Live 660-image whole-image composition comparison

On 2026-09-22/23 a separate read-only development comparison was prepared and
launched. This is a composition/whole-image characterization run, not a missing
single-type SOW gate and not another official sealed evaluation.

Frozen identities:

- config: `configs/foundir_composition_whole_image_v1.json`, SHA-256
  `b9b04fc4242524cfa44a2f2ac038bd73266d4bf14809dad20ce2b5929a6de0cb`;
- manifest: `manifests/foundir_composition_whole_image_v1.csv`, SHA-256
  `a94128195ffd219f7e94b44c1ec98291c75e727f17e2c3af88cf1c2f9b6282a9`;
- 660 validation images, 60 per each of eleven categories;
- compare the frozen routed project system against released FoundIR on whole
  images with tiled inference;
- no model update and no official-test access.

Verified live snapshot at `2026-09-23T05:28Z`:

- one homogeneous 8xH200 instance;
- all eight GPUs at 100% utilization, about 19,046 MiB per GPU;
- complete routed stage counts: Rain 111, Blur+Noise 53, Noise 137;
- Haze in progress with 105 immutable stage PNGs written;
- no traceback, OOM, NCCL error, terminal result, or end marker;
- final `checkpoints/foundir_composition_whole_image_v1/result.json` does not
  exist yet.

Monitor only:

```bash
tail -f logs/foundir_composition_whole_image_v1.log
```

Do not restart or duplicate the job merely because its remote PID is invisible
from the Codex shell. Advancing telemetry, log records, and immutable PNGs are
authoritative liveness evidence.

## Final official sealed result binding still required

User-authoritative current status: **all five official sealed SOW types pass
well above threshold**. Do not regress the handoff to v6 or describe Blur as
unfinished.

The following provenance fields still need to be copied from the final result
artifact once it is visible in this workspace:

- exact result path and SHA-256;
- exact five PSNR/LPIPS pairs and percentage margins;
- frozen router/config/checkpoint identities;
- run start/end, hardware, and total wall time;
- output/sidecar counts and terminal marker;
- whether this was one run or a later version after historical v6.

Until these are bound, cite the user's confirmation for the terminal sealed
status and cite the verified development artifacts for exact Blur numbers.
Never substitute the failed historical v6 JSON as the final result.

## Exact next action

1. Continue monitoring the already-running whole-image composition comparison;
   do not launch another GPU job.
2. At terminal completion, verify all 660 project outputs, all 660 released
   FoundIR outputs, all metric sidecars, `result.json`, hashes, and end marker.
3. Write its terminal report and update this handoff with actual timing and
   metrics. Treat it as composition/robustness evidence, not as reopening the
   already-passed five-type sealed result.
4. Locate or synchronize the final all-five official sealed result, then bind
   its exact metrics/path/SHA in this handoff.
5. After both records are complete, freeze the deliverable package and prepare
   the reproducibility/technical report. Do not start another Blur/Noise model
   search merely to enlarge already-passing margins.

## Compute-sizing rule retained

Choose GPU count from measured workload and end-to-end wall time, not available
inventory. Prefer one tested 8-GPU node for jobs under roughly one to two hours;
use embarrassingly parallel multi-node inference only when it materially saves
wall time. Preserve global batch, optimizer steps, LR semantics, data ordering,
and checkpoint comparability. Every future GPU command requires a total
end-to-end wall-time estimate before launch.

## Potential SOTA follow-up recorded on 2026-10-12

This is deferred planning, not a new active launch. It does not supersede the
Generalist-first gate: finish `real_generalist_4node_mixed_hopper_v1`, select
raw versus EMA only from frozen validation evidence, and run exactly one
matched 660-image Generalist-versus-Generalist evaluation. Do not tune from the
660 result. The project is not yet demonstrably SOTA because pass-1 metrics are
crop diagnostics and Double LPIPS remains the main unresolved gate risk.

Only after EBT beats released FoundIR Generalist in both PSNR and LPIPS for
Singleton, Double, and Triple should this public-benchmark plan be activated:

| Experiment block | Purpose | New training? |
|---|---|---:|
| Frozen 660 whole-image gate | Establish matched FoundIR Generalist result | No |
| CDD-11 matched train/eval | Public composite Singleton/Double/Triple evidence | Yes |
| Standard 5D matched train/eval | SOTS/Rain100L/BSD68/GoPro/LOL comparability | Yes |
| No-VLM ablation | Isolate VLM conditioning | Yes |
| No-energy/direct-restoration ablation | Isolate learned iterative energy | Yes |
| Fixed steps versus adaptive TTS | Quality/compute curve | No |
| Self-verification on/off | Isolate verification benefit and cost | No |
| Efficiency evaluation | Params/FLOPs/memory/latency/descent steps | No |

The minimum package is eight blocks, four new training runs, and roughly six
to eight evaluation jobs. A broader AiOIR SOTA claim additionally requires a
separately trained standard 3D setting, real-world benchmarks, and
robustness/OOD evidence, for roughly 12--15 blocks and six to eight training
runs.

Main CDD-11 targets (PSNR/SSIM) are AllRestorer `33.72/.9436`, DAR-Net
`29.73/.888`, ClearAIR `29.34/.886`, MoCE-IR-S `29.05/.881`, and OneRestore
`28.47/.878`. AllRestorer is the strongest published claim even though a
reproducible public implementation was not verified; unrestricted SOTA wording
cannot omit it. Main standard-5D average targets include D3Net `32.36/.901`,
DAR-Net `31.02/.921`, DPC-Net `31.00/.923`, DFPIR `30.64/.913`, MoCE-IR
`30.58/.919`, and CyclicPrompt `30.52/.935`. FoundIR-v2, VLU-Net, InstructIR,
Perceive-IR, ClearAIR, DFPIR, and EBT are also mandatory related-work
comparisons for the claimed VLM-conditioned learned-energy/adaptive-TTS/self-
verification intersection. Do not compare unmatched metrics or imply that one
method leads every task.

Measured planning baseline: the current 30,699-step schedule is approximately
8.97 wall hours per pass, hence about 27 wall hours and 864 allocated
heterogeneous GPU-hours for three passes on 32 GPUs. Four mechanical repeats
would be an unapproved ceiling of about 108 serial wall hours and 3,456
GPU-hours. Before any CDD-11, 5D, or ablation run, perform a 500-step protocol-
specific throughput qualification, targeted at no more than 0.5 wall hour and
4 allocated GPU-hours on eight GPUs, and derive the real budget from matched
epochs and unique-sample exposure.

The intended defensible claim is narrowly scoped to composite/mixed Generalist
SOTA or best matched average. Winning the project-specific 660 comparison is a
necessary first gate but is not, by itself, universal All-in-One Image
Restoration SOTA. Specialist work remains paused.
