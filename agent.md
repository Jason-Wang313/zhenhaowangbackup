## Blur真实参数梯度诊断已准备：两检查点×固定8训练图 — 2026-11-06

当前官方21.819776/.241321未过25.9/.18，只推进Blur。64训练图已有20张错误方向，先核对损失的实际参数作用，不把RGB梯度比例当LoRA更新证据。
固定当前前8训练来源/裁图，原初始化和当前step200，真实10*MSE/.1*LPIPS分别求全部LoRA参数梯度；每图及NCCL汇总global8分别报告norm/cosine/内积。原初始化复现LQ，step200复现已有64图对应PNG；每检查点求导前后参数SHA不变，无optimizer/参数更新，GT仅现有训练8图。
4项CPU测试、静态检查和prepare通过，2.037856秒/GPU0；待发布入队round11_blur_loss_gradients_v1，node3双H100。入口smoke/diagnose_foundir_blur_loss_gradients_v1.py，输出foundir_blur_loss_gradients_v1。无专用持久观察器，接手看真实ended，再finish/推报告，勿重复提交。
原始混合SGD方向的一阶loss内积只是诊断，不等于Adam实际行为或有限步/官方收益。仅官方完整150为质量门槛，无新测试集gate。不改旧绑定配方，不盲追加训练，依据真实结果冻结下一方案；完整自主研究授权继续有效。详见reports/foundir_blur_loss_gradients_v1_readiness_2026-11-06.md。

## Blur64训练图诊断：训练侧也存在错误方向；先核对loss参数梯度 — 2026-11-06

固定前64当前训练图/最终step200只读诊断完成，两rank/launcher退出0，参数SHA前后不变，PNG/记录核验通过。48.074520秒/.026708067 GPU-hours，CPU1.422280秒；两个delta试验及对应诊断共2.562817654 GPU-hours，早期历史另计。
training_fit_only：LQ23.384541/.347390→23.672547/.232197，PSNR仅+.288006dB、LPIPS降33.16%；20/64方向非正相关，28/64 PSNR差于LQ。不能只归因官方分布，不能包装为修复成功。官方仍21.819776/.241321失败；只用官方150的25.9/.18质量门槛，Blur未过不推进其它类别。
已有旧资格输出空间证据：加权LPIPS RGB梯度范数为加权MSE的5.079–12.415倍；这不是参数梯度主导的证明。下一固定小诊断：当前前8训练图，原初始化和最终200，分别测10*MSE及.1*LPIPS的真实LoRA参数梯度范数/夹角，无optimizer/参数更新、不新增测试gate。不先改变已绑定配方，不用官方GT调系数。此参数梯度诊断尚未实现/提交；下一代理依此推进，完整自主研究授权持续，不需重复许可。
四台worker当前空闲，round10_blur_coverage_fit_v2已结束，无待续训练；全部资产保留。报告reports/foundir_blur_coverage_fit_v2_result_2026-11-06.json及terminal.{json,md}。不要重复64图任务。

## Blur下一项已准备：固定64训练图只读拟合诊断 — 2026-11-06

官方1024来源delta200仍21.819776/.241321失败；解释报告dfaca80已推送。先检查训练侧是否学会正确修正，不盲加步、不推进其它类别。
固定当前训练来源前64张/原512裁图、同一step200权重，PNG固定后才读训练GT统计LQ/候选指标和修正方向；不更新参数、不跑旧50步、不设新测试集gate。单节点node3两H100，每rank32张；权重SHA前后不变，全部64图及PNG核验。只标training_fit_only，不是泛化证据。
smoke/diagnose_foundir_blur_coverage_fit_v2.py及run_foundir_fullstack_blur_coverage_fit_v2.sh已实现，prepare/语法检查通过；当前待发布提交，以共享queue round10_blur_coverage_fit_v2实际状态为准。输出foundir_blur_coverage_fit_v2；短任务无独立持久监视器，接手需检查实际ended，再执行CPU finish并推报告，勿重复提交。旧prelaunch receipt已备份，旧全部运行资产保持。
完整自主研究授权继续；后续依据结果冻结方案。唯一质量验收仍完整官方Blur150的25.9/.18，未过前不推进其它类别。详见reports/foundir_blur_coverage_fit_v2_readiness_2026-11-06.md。

## Blur1024来源对照失败：有单项变化，无联合收益；先诊断训练侧 — 2026-11-06

完整官方150/2250块、两rank及launcher退出0、CPU核验通过，终态fa375b5已推送。新21.819776/.241321：比128来源delta200的PSNR+.666617dB，但LPIPS变差6.9881%；比旧50步FLUX的PSNR+.273014dB，LPIPS变差19.0991%。未过25.9/.18，不采用、不推进其它类别。100/150 PSNR差于LQ；仅30/150比前轮两项都好。
来源对照保持初始LoRA hash、200步/1600曝光、loss/lr/推理/官方150及指标不变；改1024个1080p来源，分辨率/多样性/重复次数共同变化，不能拆分因果。官方LQ评分完全一致。
只读CPU输出误差统计：修正能量下降48.44%，平均cosine .019366→.125740，但仍90/150方向非正相关（与前轮重合70），旧FLUX24/150。不是单纯“更弱”或已证实数据根因；大部分正确修正仍未学会/迁移。无GT拟合系数、候选/逐图选择。
本轮1.240189363 GPU-hours，两个delta试验加路径诊断共2.536109587，旧更早历史另计。训练1023.809839秒、官方964.034458秒、缓存244.496556秒；所有旧资产保留。四台worker空闲，当前监视器/上传器正常结束，无待续训练。
下一受控步骤：固定当前1024训练来源前64张、最终step200，只读训练拟合/修正方向诊断；不更新参数、不增加测试集门槛、不重新跑旧50步。尚未实现/提交。先区分训练侧学不好与迁移失败，再依据证据固定下一方案；完整自主研究授权继续有效，不因本轮失败要求重复许可，不盲加旧步或改官方阈值。仅官方FoundIR作质量验收，Blur过25.9/.18前不动其它类别。
详见reports/foundir_blur_coverage_v2_interpretation_2026-11-06.{json,md}、foundir_blur_coverage_control_v2_comparison_2026-11-06.json及foundir_blur_coverage_direction_v2_CPU_2026-11-06.json。

## Blur1024训练来源对照终态 — 2026-11-06

完整官方150生成和CPU完整性核验结束；按同一最终200输出判断双门槛。
完整官方150指标：{'lpips': 0.24132125635941823, 'psnr': 21.81977596282959, 'ssim': 0.7275497882643466}；25.9/.18联合通过=False。
仅官方FoundIR作质量门槛，Blur未过前不推进其它类别。不自动追加步数/切换任务。查看reports/foundir_blur_coverage_v2_terminal_2026-11-06.{json,md}。保留全部资产和旧失败。

## Blur下一轮：1024个1080p训练来源，固定200步→官方150 — 2026-11-06

用户授权自主完整修复，仍仅Blur、仅官方FoundIR质量门槛；25.9/.18前不推进其它类别。上一轮官方21.153159/.225559退步；8训练图四路径一致，只是诊断进展，无修复收益。
本轮控制训练曝光1600张/global8/200步及全部原数值实现不变，只把128混合尺度固定裁图换为1024不同1080p训练来源、每来源固定512裁图。原Blur step10000/fresh Adam，不继承失败step200、不加其它网络。来源分辨率/多样性共同变化，不能拆分因果；更少重复可能欠拟合，未证明有效。
目的地CPU预检35.119302秒/退出0：2048图SHA及裁图、模型/提示词/指标身份通过，5项针对性测试通过。相同数学CUDA资格复用，新数据初始化/每步工程检查仍执行。源码发布后由node3双H100执行新缓存→fresh200→完整官方150→CPU收尾；其它节点不启动，无开发集gate/逐图选择/官方GT拟合，失败停链。不能把准备/入队称实际GPU训练开始。
持久监视器smoke/watch_foundir_blur_coverage_v2_cpu.py，monitor和upload分离；首次/50/100/200/终态推小报告。队列round10_blur_coverage_{cache,train,official}_v2；输出foundir_blur_delta_coverage_v2和foundir_blur_coverage_official_v2。保留旧全部资产/失败和费用，无自动加步/重试。详见reports/foundir_blur_coverage_v2_readiness_2026-11-06.md。按共享心跳/实际step/终态核验，勿重复提交。

## Blur路径诊断终态：四路径一致，官方失败未改变 — 2026-11-06

固定前8张训练图、同一step200权重，无更新参数。两rank和node3 launcher均退出0；44.047787秒、.024470993 GPU-hours，CPU finish .425097秒。缓存四张量与重算完全一致，四路径float/uint8完全相同，GPU参数SHA前后不变。只排除所测路径差异，不宣称排除所有bug。
训练图LQ23.885945/.346004、旧50步22.408155/.186820、新24.164456/.235645；新模型相对LQ仅+.278510dB，LPIPS仍差于旧FLUX。官方全150保持21.153159/.225559失败，不能归为单纯加步或已证实纯泛化原因。
仅官方FoundIR作质量门槛，Blur25.9/.18前不推进其它类别。训练图仅实现诊断；所有失败/费用保留。本轮含诊断累计1.295920224 GPU-hours，历史另计。全部worker空闲，以共享心跳为准，无自动追加旧128训练。
报告reports/foundir_blur_delta_path_v1_result_2026-11-06.json及terminal.{json,md}。下一轮须固定可证伪的数据/学习假设，不重跑旧失败路线，不用官方GT拟合或逐图选输出。

## Blur200终态失败；先核验训练/官方推理路径一致性 — 2026-11-06

只以官方FoundIR作各类验收门槛，仍Blur优先。最终200步官方150/150、2250/2250块、两rank及launcher退出0，CPU完整性核验通过，67128e9已推送。结果旧FLUX21.546762/.202622，新21.153159/.225559：PSNR下降.393603dB、LPIPS变差11.319935%，未过25.9/.18；111/150比LQ的PSNR差，只有38/150比旧FLUX两项都好。不采用、不宣布修复、不转其它类别。
两步资格、缓存、200步训练和官方生成共1.271449232任务GPU-hours。训练1122.147524秒，官方960.149235秒，官方CPU finish35.423898秒；全部资产保留，旧开发评价未提交。四台worker空闲，无待续训练。
额外只读CPU误差恒等式/DCT诊断135.560801秒、0GPU-hours：新修正与GT−LQ方向平均cosine .019366，90/150非正相关；旧为.236089、24/150。新改动幅度更小但像素误差反而更大，不是仅“大步幅过冲”；按图平均全图常量色偏只占1.1686%误差。只做统计，无GT拟合系数、候选生成或GT回退；复算全150的PSNR匹配pyiqa。
32,440,320参数中96.4773%在BF16下仍改变；训练online loss下降，不能代替固定权重训练图评价或官方收益。新旧路径差异尚需排除，暂不盲加训练步数或把失败直接归因数据。
下一项已授权实现诊断：固定前8张原训练图，step200不更新参数，比较缓存参考/重新计算参考/官方requires_grad=false推理/训练checkpoint模式四路径，核对tensor、float/uint8和实际加载权重SHA。只用node3双卡，不另设测试集或质量放行门槛，不读其它类别。入口smoke/diagnose_foundir_blur_delta_path_v1.py；当前prepare后待提交，以共享队列实际状态为准。
报告reports/foundir_blur_delta_official_v1_result_2026-11-06.json、reports/foundir_blur_delta_official_error_identity_v1_CPU_2026-11-06.json。训练来源56/128为1920×1080，其余多种分辨率；这只是分布描述，不是已证实根因。先核验路径，再确定下一轮受控修复。

## 永久验收口径：只使用官方FoundIR测试集作为各类别门槛 — 2026-11-06

用户再次明确：always use official FoundIR test，不使用其它测试集作为gates。后续Blur/Noise/Haze/Rain/Lowlight及统一600验收均执行本规则；训练侧拟合/诊断可用于定位问题，但任何其它开发/测试集成绩不能替代或阻挡官方验收。官方GT仍不得用于参数拟合、阈值调参或逐图挑输出；反复官方反馈需披露为非独立盲测。当前继续已启动的Blur-only修复与官方150评价，官方25.9/.18联合通过前不推进其它类别；其它四类最终门槛及完整600统一复现目标不变。

## 最新用户指令：只用官方Blur150作为验收，不再以其它验证集放行 — 2026-11-06

用户明确要求直接用官方测试集，不再使用其它测试集作为前置门槛。仍严格Blur优先，完整官方150张PSNR≥25.9且LPIPS≤.18后才推进其它类别。本节覆盖下方固定32开发放行规则；当前已启动的128来源200步训练保持不变，未启动的128训练/32开发GPU评价不再派发，改为最终step200的官方150张完整原尺寸恢复评价。不得重跑已完成资格、缓存或训练。
官方GT只在输出固定后计算指标，不用于参数训练、逐图选择、阈值拟合或oracle回退。使用官方结果反复改进属于开发集式使用，必须披露已有曝光，不能再称独立盲测；保留全部失败及费用，不改变25.9/.18门槛。
旧CPU调度器blur_delta_v2_monitor已在旧evaluate job尚未提交时停止，GPU训练round9_blur_delta_train_v1继续。独立上传器仍保留；新官方150入口/目的地预检和接续监视器准备中，不启动其它类别。代码须先发布并核验，再向共享worker派发。旧GPU绑定源码、receipt、权重、缓存不改。

## Blur原生预测差：CUDA资格通过，固定200步运行；上传与调度分离 — 2026-11-06

当前仍只推进Blur，完整官方150张25.9/.18双目标通过前不动其它类别。原生LoRA预测差两步资格已真实通过：8图初始化浮点及uint8均完全等于原LQ、MSE/LPIPS梯度非零、两rank权重一致、step1真实恢复后step2完成；退出0，138.20684秒、.07678158 GPU-hours，峰值16.96299GiB/卡。资格报告58411a7已推送。
参考缓存160/160完成，退出0，68.10502秒、.03783612 GPU-hours。node3两H100上fresh200步pilot正在执行，首次训练报告2a5a0d6已推送。训练/开发结果均不得当官方通过；旧资产、energy及其它类别均保留。
GitHub连接曾超时，现采用独立CPU调度和上传器，避免上传阻塞已发布的实验；GPU绑定源码4182f8c未改。v1观察器已停止，当前smoke/watch_foundir_blur_delta_network_v2_cpu.py：monitor会话blur_delta_v2_monitor，upload会话blur_delta_v2_upload，日志logs/foundir_blur_delta_velocity_v2_{monitor,upload}.log。持久请求publication_requests及已发布published在checkpoints/foundir_blur_delta_velocity_v1；接手先查这些，不重复提交/启动。网络失败只延迟小报告，GPU失败仍停链。
固定顺序为200步→全128训练/32开发评价→CPU finish→小报告；无自动其它类别或官方评价。step1/50/100/200报告先原子落盘，再独立上传。所有GPU任务仍使用同一已发布commit及固定配方；轻任务只用node3双卡，不占其它三台。

## 当前Blur-only执行：原生预测差小试验已CPU预检 — 2026-11-06

继续只攻Blur；官方150张PSNR≥25.9且LPIPS≤.18之前不推进其它类别，无累计GPU-hours上限。当前新方案为LQ + 冻结VAE解码差，latent修正来自训练LoRA与冻结原Blur LoRA在同一LQ/t1状态的velocity差；两次模型前向，不是旧50步。仅原生LoRA训练，不加RGB网络，不恢复energy或其它类别。
固定128训练/32开发来源、512裁图；先node3双H100两步工程资格，CPU核验通过后缓存160图，再独立fresh Adam/原LoRA初始化200步、global8、lr1e-5、loss=10*MSE_RGB01+.1*LPIPS。最终全160图评价，不选中途检查点，不读官方像素。开发放行25.9/.18且两项优于旧50步、最大PSNR退步≤1dB；开发通过仍非官方成功。
7项CPU测试及真实两进程global8梯度SUM/Adam对照通过，目的地预检77.589859秒、退出0、0GPU-hours；权重/来源/320图哈希及裁图、旧160缓存、指标身份核验通过。此时尚未CUDA验收，不声称已修复。
入口smoke/run_foundir_fullstack_blur_delta_v1.py；持久观察器smoke/watch_foundir_blur_delta_v1_cpu.py按qualify→CPU核验→cache→train→evaluate→finish执行，失败停链不自动重试。每步原子保存Adam/RNG/采样位置及SHA，step1/50/100/200和终态推小报告。不修改旧源码/资产，轻任务只用node3两卡，node0–2保留空闲。
详见reports/foundir_blur_delta_velocity_v1_readiness_2026-11-06.md及preflight JSON；真实CUDA速度、显存和结果待日志确认，不引用旧ETA。

## Blur当前：频段候选失败；局部几何诊断完成；继续Blur原生保真修复 — 2026-11-06

仍严格只推进Blur，官方150图25.9/.18双目标通过前不动其它类别，无累计GPU-hours上限。round8缓存160/160、评分160/160、所需节点全部status0；CPU核验通过并推送ce1793b9。缓存+评分实耗1.053419215GPU-hours；全部旧资产保留。
固定六频段候选开发PSNR23.891740→24.660468，但LPIPS.217380→.330545、最大单图PSNR退步5.067924dB，不采用、不跑官方。训练侧也LPIPS退步。不能把MSE下降当双指标成功。
CPU频率解释80.689261秒：128训练/32开发，残差与目标修正方向相关性低；逐图用GT拟合六系数的不可部署诊断开发PSNR25.773539。该值不是clip/量化后的严格理论上界；不继续无依据扫幅度。
新增固定前8训练图局部对齐诊断21.495747秒，0GPU-hours：原22.408154/.186847；LQ参考DIS23.134434/.192608；GT参考诊断25.334012/.160353。仅LQ方法尚未联合改善，不采用；GT结果不可部署、不是泛化成功，只说明可修正空间错位贡献了部分误差。未读官方或开发像素，3项采样坐标/身份/输入检查通过。
下一方向准备：原生FLUX条件残差映射，从LQ保留位置，以同一LQ状态下训练LoRA与冻结旧Blur LoRA的预测差作为latent修正；冻结VAE解码差加回原LQ，初始应还原LQ，必须GPU实测核验。只训练FLUX原生LoRA，不增加CNN/RGB输出网络、不恢复旧多任务或energy。现仅方案，尚无训练入口/CPU或CUDA预检通过，不宣称已修复。
准备阶段必须核对历史：已有degraded_anchored_flow_batch、多任务anchored训练、Blur16/512来源末段监督和LQ-start推理失败；不得把旧方案改名当新发现。拟复用当前128训练/32开发来源和旧缓存，先固定小试验再按训练/开发证据决定后续，不直接官方扫描或长训。
报告reports/foundir_blur_frequency_evidence_v1_CPU_2026-11-06.json、reports/foundir_blur_local_alignment_v1_CPU_2026-11-06.json及reports/foundir_blur_first_mechanism_update_2026-11-06.md。

## ACTIVE PRIORITY RESET：只攻Blur，官方通过才推进其它类别 — 2026-11-06

用户最新明确重设执行顺序：当前唯一推进类别为Blur；官方150张完整评价须PSNR≥25.9且LPIPS≤.18后才允许推进Rain/Noise/Haze/Lowlight。此规则覆盖下方旧Rain优先、Noise先行及“所有类别准备后才单类官方”的安排。系统活动goal文字无法中途编辑，实际执行必须服从本节最新用户指令，不得把旧目标伪标成功。
始终第一性原理：Blur150/150路由正确，当前21.546762/.202622，91/150 PSNR低于LQ；不能通过改路由解决。先验证频段残差是否造成无效/有害细节，再依据训练/开发证据决定后续修复；六系数只是待验证候选，不保证填平4.353238dB缺口。
新独立Blur-only执行链复用已绑定128训练/32开发来源和冻结50调用专家：node0/1/2六卡生成160图缓存→CPU训练侧解析拟合六系数→node0两卡完整评分→CPU核验/推小报告。node3保留空闲；Rain对照尚未提交，Noise旧队列不启动。
本链CPU绑定预检通过，7项队列检查通过；实际GPU是否开始必须查新job/日志/输出，不以监视器tmux创建当成功。任务round8_blur_frequency_cache_v1与round8_blur_frequency_score_v1；监视器smoke/watch_foundir_fullstack_blur_first_v1_cpu.py。
开发平均PSNR改善、LPIPS不恶化且最大单图PSNR退步≤1dB，才将冻结候选送官方150图；开发失败则留在Blur诊断/修复，不转其它类别、不盲测官方。GT仅训练拟合或输出冻结后的评分，不参与官方逐图选择/阈值拟合。官方已反复使用，披露历史曝光，不能称独立盲测。
无累计GPU-hours上限；每轮固定范围、故障停止条件和实际费用仍记录。保留全部历史资产/energy及未提交修改，不reset/clean、不修改运行绑定代码。完整全栈执行和小报告推送授权继续有效，无需再次索取GPU终端命令。
详见reports/foundir_fullstack_blur_first_v1_readiness_2026-11-06.md。

## ACTIVE GOAL：五类统一端到端达标；始终第一性原理；无累计算力上限 — 2026-11-06

用户已明确授权代理创建目标并自行执行完整闭环，取消100 GPU-hours累计上限。本节覆盖历史“仅准备/待手动启动/无后续授权”及旧累计预算限制；不等于无意义重复或隐瞒成本。
最终唯一完成条件：冻结同一无GT路由/输出选择的统一入口，从官方600张原始LQ重新生成全部结果，Blur≥25.9/≤.18、Noise≥38.9/≤.10、Haze≥22.1/≤.19、Rain≥34.2/≤.08、Lowlight≥18.9/≤.28（PSNR/LPIPS），保存全部逐图、配置/权重SHA、退出码和可重复命令。单类缓存、训练探针或CPU通过不算目标完成。
始终从第一性原理分析误差来源、目标与优化机制；明确可证伪假设、固定对照和开发判断后再花GPU，不盲加步、不改名重跑失败路线。每轮写明范围/预期费用/成功与故障停止条件，记录所有失败和累计实际GPU-hours；没有全局费用上限，故障任务仍应诊断处理。
代理负责安全同步、实现、预检、共享worker提交、真实运行监控、故障处理、CPU收尾和GitHub小报告推送；不要每轮只给命令或重复要许可。用户询问/质疑不是取消。全部原始数据/缓存/权重/energy和旧记录保留，不reset/clean，不改运行绑定文件。
当前4台8H100：node0/1/2沿用v2，node3仅nodes=[3]独立双卡。轻任务单节点；四节点联合入口未适配不得盲用。下一步Rain匹配原loss/log-MSE两组各50步对照，然后推进Noise/Blur；Haze/Lowlight候选保留，全部候选准备后统一600复现。
训练/开发用于制定方案，官方GT不得拟合参数、调阈值或逐图挑输出；披露旧来源曝光和官方反复使用，不能称独立盲测。工程、训练、开发、官方成功明确区分；不保证研究必然达标、不伪造成功。

## 当前终态：四台worker已核验；Rain新损失仅CPU候选 — 2026-11-06

第四台已由用户启动并实际核验共享registration/新鲜heartbeat，node0–3均idle、无未完成队列；8个GPU UUID互异。无需重复启动worker。node3 IP10.244.90.116，只接nodes=[3]独立双卡任务，四节点联合job仍不支持；轻任务优先单节点。
Rain四组诊断已结束，新逐图log-MSE候选6项CPU测试通过；32固定训练探针×800/900/1000共96输出核对完成，5.908933秒、退出0、0GPU-hours。没有新GPU训练或待结束GPU任务。
参考误差仅取原800训练探针中位数.001339064795，eps=(2/255)^2；保留LPIPS权重.05，log-MSE在参考误差处匹配原MSE权重10的局部梯度。模型/推理未改，未读开发或官方像素。
全32探针的原量化MSE和原目标在1000均比800差，所以8图目标权重冲突不是完整根因证明；新损失不是已证实修复，不保证官方通过。LPIPS复用已核验记录，未重新计算；uint8诊断不等同连续RGB训练loss。
GPU试验配方/步数/开发验证/预算/入口尚未冻结；不追加原1000训练、不据官方改选900、不宣称只剩GPU运行。下一设计需匹配原loss与新loss对照，不能凭CPU通过启动长训。
详见reports/foundir_rain_logmse_objective_v1_CPU_2026-11-06.{md,json}及reports/foundir_fullstack_four_workers_verified_2026-11-06.json。保留energy、全部旧失败记录和大资产。

## Rain交叉诊断终态；第四节点独立入口已准备 — 2026-11-06

Rain固定8训练图×四组完成，node0两H100，72.19355秒/.04010753GPU-hours，GPU与CPU退出0；16个原版本输出均逐像素/指标复现。报告推送9e5a4a1。
800/800为33.92244867/.07303950；800前缀+1000末次为33.60182729/.07200584；1000前缀+800末次为33.93427530/.07287066；1000/1000为33.57745226/.07193444。
固定800前缀后PSNR仍下降.32062138dB，改前缀的附加影响仅-.02437503dB；本8图不支持前缀漂移是主要问题，不能据此直接冻结前缀训练。
更具体线索：固定前缀换末次，算术平均量化MSE改善约4.33%，平均逐图PSNR反降；12806原45.4293降40.4876dB，MSE增3.12倍。说明当前平均MSE目标与平均log-MSE/PSNR权重存在实际冲突；尚非官方整体根因终判。
下一方向先检查逐图误差归一化/log-MSE和原模型相对保真约束，仅训练侧制定，不盲调官方或追加相同训练。本轮已结束，无自动下一GPU任务。
第四台用户实测node3 IP10.244.90.116，notebook e087f2fc-829f-4a0d-8aa5-5df18b6cd56a，两卡H100/torch2.7.0+cu128。新独立worker入口start_foundir_fullstack_worker_node3_v1_tmux.sh已准备，尚待该节点一次性启动并实测共享心跳/UUID。
不修改正在使用的三节点v2。新node3只支持nodes=[3]独立任务，旧worker跳过；不能派四节点联合job给旧v2。四节点联合训练须另行适配，不能宣称已通过NCCL。
详见reports/foundir_fullstack_rain_prefix_cross_v1_interpretation_2026-11-06.json及reports/foundir_fullstack_node3_extension_v1_readiness_2026-11-06.md。

## 已授权执行：Rain固定8训练图前缀/末次四组对照 — 2026-11-06

用户execute，仅本轮诊断。索引12800–12807固定，不按分数挑；800/1000前25调用×800/1000第26调用，四组32输出、432模型调用，GT仅输出固定后训练侧评分。
node0两卡，硬cap15分钟/.5GPU-hours，不训练、不访问开发/官方像素，不自动启动Noise/Blur或后续训练。CPU测试5项及真实权重/缓存预检通过；必须原800/800与1000/1000探针逐像素复现才解释结果。
状态漂移只是待验证假设，不保证冻结前缀能修好。按全部8图固定均值/逐图条件差报告，不逐图选输出，不称泛化或官方成功。
任务round7_rain_prefix_cross_v1，独立输出foundir_fullstack_rain_prefix_cross_v1；源码发布核验后由现有worker执行，持久CPU观察器负责核验与小报告推送。保留全部旧资产。
新增第四节点等待用户实测IP/型号/UUID及共享目录后登记，不沿用当前三节点v2编号；本轻量任务不因新增节点扩大资源。
详见reports/foundir_fullstack_rain_prefix_cross_v1_readiness_2026-11-06.md。

## Rain终态失败；下一步先隔离前缀与末次更新 — 2026-11-06

Rain100/4680块完整结束，续跑三节点status0及CPU finish通过，终态推送106de594。step1000为31.89185747/.07479696，PSNR比800下降.14307249dB，仍差2.30814253dB；不称修复成功，不采用1000为统一默认。
32训练探针PSNR也下降.36603074dB，不能仅归因官方分布。100张只对应10个GT场景，3/10场景PSNR改善。旧step800只是较好参考，并未通过官方PSNR。
代码发现同一LoRA同时改变前25次和第26次，但梯度仅穿最后调用；前缀漂移是待隔离假设，不是已确认根因。下一步建议合法训练图prefix800/1000×readout800/1000的固定小型对照，单节点、不训练、不重跑官方100；该诊断入口尚未实现/启动。
用户上一条是在询问统一复现，不是要求停止。代理误取消已经记录，用户明确要求完成后恢复；不要再把询问/质疑自动当作取消命令。旧cancel/-15日志和全部费用保留。评价两attempt共14.04867389GPU-hours，训练另计。
旧统一入口仍绑定800，新Rain专用入口绑定1000；最终统一部署必须显式锁定已接受的五专家/路由/推理流程，不承诺旧命令自动升级或单类通过等于统一复现。
Noise/Blur尚未启动；原9bd8fbe编排等待旧Rain终态，需要先适配已完成的resume终态再使用，不能删除失败标记伪造成功。
详见reports/foundir_fullstack_rain_failed_update_v1_2026-11-06.{md,json}；不自动追加当前Rain训练或官方扫描。

## 就绪但未启动：Noise纹理路由及Noise/Blur六频段修复 — 2026-11-06

用户要求推进到只剩GPU运行。本轮代码、真实CPU预检、单入口编排和小报告发布已准备；未提交round6队列，未启动新GPU。Rain评价保持运行。
Noise705参数独立CPU分类器固定400步失败：calibration Lowlight75/77、confirmation16/17，原为77/77、17/17；12.833秒，0GPU-hours。不部署、不做官方阈值扫描。
下一候选：42,153参数LQ纹理分类器，1514训练/157校准/37确认来源，固定400步/global32，单节点两卡，逐步保存；仅修改原Noise/Lowlight二分类，不输出RGB。GPU效果待验证。
Noise/Blur专家保持冻结；各128训练/32开发固定来源和512裁图，50步原生输出。各6个共享DCT频段系数只以训练GT闭式拟合，开发/官方不拟合；不同于旧统一倍率，但不保证解决局部几何。
21项CPU测试通过；Noise/Blur各320文件SHA/裁图、12项模型提示词SHA及pyiqa完整state身份通过；路由1708来源LQ六视图已准备。CUDA、NCCL、速度和指标尚待实测。
入口start_foundir_fullstack_noise_blur_v1_tmux.sh需本轮完整提交SHA；显式启动后先等Rain终态报告，再Noise路由→Noise缓存/CPU拟合/单节点评分→Blur同流程→CPU核验推送→停止。当前未启动该监视器。
缓存可三节点六卡；轻量路由/评分只node0两卡。固定本轮硬cap共11GPU-hours，旧Rain及历史成本另计，失败不自动重试/扩预算，不自动官方测试/部署或统一600复跑。
保留Haze/Lowlight、energy和所有旧资产。下一步只需实际执行该有限GPU链；最终是否修好仍需真实结果，不能保证通过或提前声称官方成功。
必读reports/foundir_fullstack_noise_blur_repair_v1_readiness_2026-11-06.md；无SSH执行方法仍见reports/foundir_shared_gpu_executor_runbook.md。

## CPU后续终态：Noise小试验未修复官方错误，Blur平移不采用 — 2026-11-06

Noise21参数物理统计头固定400步已完成：校准修正1条、确认37/37保持，但官方150条路由全部未变；10张Noise仍误送Lowlight。训练129.533秒、官方诊断32.380秒、GPU0，不部署、不调官方阈值或追加步。
原冻结路由平均概率替代平均logit同样不改变150图；以上失败和官方来源重叠均保留，不包装为Noise成功。正确专家参考自身37.693592/.110755也未过原双目标。
Blur150图整数平移CPU诊断完成54.672秒/GPU0。按LQ对齐仅3张非零、共同内部PSNR平均-.013414dB，不追加该方案GPU实验。GT诊断有效98/150，均值不能当全150增益；不排除局部/亚像素问题。
历史Blur512来源已经做真实50步46+4梯度训练而泛化失败；LQ锚定flow代码/旧实验也已存在，不能把这些改名当新方案。
Noise/Blur来源计划继续保留，尚无新专家训练或GPU预检；下一步围绕真实去噪与跨场景判别、局部保真设计有限对照，不重跑已失败方案。
Rain800→1000已完成、CPU核验与终态推送04535de；三节点现执行round5_rain_step1000_3x2_v1，图块真实推进，仍独占当前6卡。其余500输出包含已接受Haze，未覆盖。
Rain评价实测早期约10.5秒/块/卡，完整4680块约2小时多；此前粗估偏乐观，后续按实际完成率更新。没有新增Noise/Blur GPU队列。
详见reports/foundir_fullstack_noise_blur_CPU_followups_terminal_2026-11-06.{md,json}，全部旧资产及绑定源码保留。

## 正在推进：Noise/Lowlight固定CPU路由小试验；Blur来源准备 — 2026-11-06

用户明确要求继续Noise、Blur下一步。六卡仍由Rain固定1000及Rain100评价顺序占用；本节只授权已限定CPU试验，不提交额外GPU或自动部署。
Noise/Lowlight冻结路由4901缓存+官方150重放完成；平均概率替换平均logit完全未改变150图，不能当修复。误判来自多块共同分类，需检查输入特征的区分能力。
来源审计新增事实：旧路由fit池159裁图/30跨类别来源组与官方GT文件精确重叠；路由训练用LQ特征和类别，不是GT像素训练。官方不称独立盲测；新过滤不能消除旧权重曝光。
新CPU试验：原路由已为Noise/Lowlight时，20个LQ亮度/对比度/高频统计输入21参数线性margin修正；其它三类路由完全保持。固定400步、batch64、lr.001，无超参/阈值/选步扫描，不改FLUX专家。
新fit来源Noise1001/Lowlight513；calibration80/77、confirmation20/17，已排除165个匹配裁图及其来源组；只fit更新。四项CPU测试及prepare通过，待实际训练/评价。
训练保存后CPU固定官方150无标签推理，先落盘决策再复用已有GPU输出评分；缺输出如实报告，不能伪造或按GT选。CPU阶段cap1800秒，不自动部署或扩大预算。
Blur同步完成来源候选计划：与Noise各预留128训练/32开发来源，只做元数据，不是GPU训练已获具体规模授权。所有旧资产和运行绑定源码不变。
详见reports/foundir_fullstack_noise_physical_router_v1_readiness_2026-11-06.md；科学结果与失败均需小报告推送。

## 并行CPU准备：先Noise再Blur，六卡继续留给Rain — 2026-11-06

用户要求先查当前进度，再做Noise、Blur；当前仅3节点6卡。保留Rain固定1000及已安排Rain100评价，未新增Noise/Blur GPU任务。
已只读重算Noise官方100/开发1859、Blur官方150/开发2595的已有记录，退出0，分别3.042/4.598秒，0GPU-hours，无像素解码、拟合或新恢复。
Noise10张错路由全部为同一GT来源，均进Lowlight；正确专家参考37.693592/.110755仍未过38.9/.10。不能用官方ID/GT标签硬编码修路由。
旧Noise开发输入40.474110本已过PSNR，输出40.150553；原LoRA只16训练来源/400步。绝对门槛通过不能替代真实去噪增益证据。
Noise16训练来源与官方600无精确文件哈希重叠，但不证明语义分离/全历史无曝光。全图常量色偏占正确专家误差约24.70%，须训练侧核实，不在官方拟合偏置。
Blur150路由全对，91张PSNR差于输入；全图常量色偏只占误差约.178%。不复制Haze修复，不重跑失败的残差扫幅、LQ起点或旧tiny-set方案。
开发/官方checkpoint SHA一致，但来源、裁图/拼接、seed、指标器不同，不能将跨集差异归因于单一原因；当前没有新官方通过结果。
详见reports/foundir_fullstack_noise_preparation_v1_2026-11-06.{md,json}及blur对应报告；GPU顺序仍Rain收尾/评价→Noise→Blur，全部类别通过后才统一600端到端复跑。

## 所有接手代理必读：无SSH的共享目录GPU执行通道 — 2026-11-06

本项目已具备GPU常驻执行器，代理职责包含已授权任务的提交、实际执行核验、监控、CPU收尾和结果报告推送，不限于给用户命令。
不得仅因当前CPU环境无CUDA/SSH就声称无法执行；先检查现有共享worker心跳、队列及CPU持久监视器，避免重复启动或提交。
原理：用户在每个GPU节点一次性启动worker → 代理向共享GPFS写job → worker在GPU节点本地执行 → 心跳/日志/退出码/结果写回共享目录。GitHub不是实时队列。
当前入口smoke/foundir_fullstack_executor_3x2_v2.py，队列checkpoints/foundir_fullstack_executor_3x2_h100_v2；现有三台H100已启动worker。新节点仍须一次性启动及重新核验登记。
方法可供其它代理/节点使用，但现有v2绑定当前3台IP、H100、每台2卡及项目路径；其它节点/型号/项目须单独适配，不可盲用旧编号或改运行中绑定代码。
轻量GPU任务只用一节点；重任务才按收益跨节点。源码先发布并验证SHA，再submit；完成须实际rank/step/输出和退出状态核验，tmux创建不等于GPU成功。
接手前检查已运行监视器和自动后续任务，同一轮明确一位提交负责人；失败不自动重试、保留所有资产，不扩大授权预算。当前Rain训练和后续Rain100已有持久监视器，不要重复派发。
完整命令、节点映射、状态检查、提交参数、取消/恢复边界及历史来源：reports/foundir_shared_gpu_executor_runbook.md（必读）。
历史：6e0f985引入此通道，e7d0a2b适配当前三台；用户已在三节点启动一次性worker。代理回合结束后持续运行的是worker/监视器，不是语言模型自动在线。
本规则只明确执行能力和交接职责，不改变下面当前实验顺序、训练终点或科学口径。

## 当前后续：固定Rain100评价，训练结束后自动衔接 — 2026-11-06

Rain step800→1000正在三节点六卡训练，step805独立核验和首次五步报告已推送fa5f461；监视器已恢复，GPU无中断。
按用户execute及全流程授权，新增CPU监视器等待三节点训练退出0、CPU终态核验和报告发布，再执行最终1000权重prepare并提交Rain100评价；无需用户重复GPU命令。
只重生成Rain100/4680块，原26调用endpoint/同seed/512+128拼接/pyiqa不变。其余500图采用已接受清单，必须保留Haze协调成果。
当前仅元数据计划与8项CPU测试通过；最终权重preflight和GPU评价尚待训练结束，不能称Rain已修复。所有训练绑定源码未改。
任何失败保留并报告、不自动重试、不追加训练、不用900替换1000。评价结束推小报告/退出码/费用，后续按真实结果决策。
所有报告继续披露旧权重3共享GT来源/30官方Rain的历史曝光；官方是自适应基准，不是独立盲测。全100主指标加30/70描述性分组，不删样本美化门槛。
Haze单独端到端复跑暂停；Rain→Noise→Blur修好后才统一600从原始LQ重生成。当前新自动链仅含本轮训练收尾和Rain评价，无其它实验。
详见reports/foundir_fullstack_rain_step1000_v1_readiness_2026-11-06.md。

## Rain续训已实际运行，里程碑报告持续监控 — 2026-11-06

三节点六张H100的NCCL和step800模型/Adam恢复均通过；已保存step815，训练继续至固定1000，不自动扩步。
独立CPU已核验step805完整状态、SHA、六rank与next_slot25760，退出0；最近计算约26.5秒/步，另计保存和探针。
CPU报告监视器首次因Markdown末尾空行未通过git diff检查而退出，GPU训练未受影响；已修复发布格式并保留原失败日志。
监视器负责首次五步、900及终态小报告推送与最终CPU核验，不自动重试GPU。源码smoke/watch_foundir_fullstack_rain_continue_v1_cpu.py。
训练探针不是官方成绩；旧权重有已确认3来源/30张官方Rain的历史GT来源曝光，新数据过滤不能抹去此事实。
固定1000的Rain100评价另行绑定最终权重；保留Haze/Lowlight已通过候选，全部类别修好后才统一600端到端复跑。

## 当前执行：Rain固定step800→1000续训，三节点六卡 — 2026-11-06

用户明确execute。保持原26调用endpoint、MSE×10/LPIPS×.05、lr1e-5、global32和512，只新增200步/6400呈现；不恢复旧任务或加新RGB网络。
完整step800模型/Adam已验证并将复制，新world6每rank6/6/5/5/5/5样本，loss/32后梯度SUM；六进程CPU与global32对照通过。
8→6 rank RNG重新设定，不是bitwise恢复；使用当前部署VAE runtime，先重算当前环境step800的32训练探针，900/1000同样检查，全部training_fit_only。
重要：预检发现并确认旧on-policy六次呈现涉及3个官方GT来源，对应30张官方Rain记录；旧权重有历史曝光，不能称干净held-out泛化。
新训练排除这3个来源组及所有关联裁图，25600起首6400合规缓存到32001；逻辑slot与实际cache_index显式映射。不删除旧数据，不隐瞒旧曝光。
新目录checkpoints/foundir_fullstack_rain_continue_3x2_v1，job round4_rain_continue_3x2_v1，master10.244.211.17:29841。逐完整步保存800..1000。
训练粗估100–140分钟/六卡，实际按首步更新；训练阶段每次cap3小时，不自动扩步/重试。六卡CUDA/NCCL状态须看真实日志。
固定1000后再准备本轮Rain100官方评价；Haze/Lowlight结果保留。五类都过才统一600端到端复跑，不因训练探针达标就声称Rain修好。
首次801..805、900和终态须推小报告，包含实际费用及已知数据重叠限制。旧资产/旧绑定源码全保留。
详见reports/foundir_fullstack_rain_continue_v1_readiness_2026-11-06.md及CPU预检/数据审计JSON。

## Rain诊断完成：细节误差为主，尚未修复 — 2026-11-06

遵守最新顺序：Haze单独端到端复跑暂停且未入队；先Rain、Noise、Blur，全部候选通过后再统一600端到端GPU验收。
Rain100图/4680块原拼接逐像素复现，重叠RMS3.124灰度级（明显低于Haze21.054），常量色偏解释19.62%的重叠差异。
32像素块均值误差平均占19.74%，GT理想消除该块均值的诊断增益仅+.986367dB，低于当前2.165070dB缺口；非可部署输出或一般低频上界。
8.54%的LQ颜色分量低于GT，最大差56，不能假定所有雨都是非负亮分量叠加，因此没有启动单侧裁剪或直接复制Haze修正。
两项CPU诊断均退出0，GPU-hours0，无训练/新恢复/GPU队列。当前Rain仍32.034930/.075014未过PSNR，十项总计仍5过。
下一步检查合法训练来源上的细节拟合与训练/官方口径差异，再冻结有限FLUX更新；不盲扫后处理、不把官方GT用于训练或逐图选择。
详见reports/foundir_fullstack_rain_diagnosis_v1_2026-11-06.{md,json}及overlap/additivity JSON。

## 用户调整执行顺序：先修其他类别，最后统一端到端复跑 — 2026-11-06

用户明确暂停刚授权的Haze200端到端复跑，要求先处理其它类别，等剩余类型修好后再统一从输入重跑GPU验收。
Haze复跑尚未提交队列、未生成新图块、未消耗GPU；不用取消/杀进程。三节点执行器均空闲。
当前顺序：Rain → Noise → 尚未通过的Blur；保留Haze固定重叠协调及Lowlight已通过结果。
五类现有候选达到原十项门槛后，再冻结统一流程、从原始LQ重生成官方600图并统一评分；不能将缓存对照通过冒称端到端已复现。
当前先只读检查Rain原100图/4680块及历史26调用endpoint协议，判断误差来源；不能无证据直接复制Haze修正。
轻量CPU/评分只用CPU或单节点；需要重生成/训练时按明确收益选择资源。所有旧资产、失败和费用保留。
本节覆盖上一条Haze先端到端复跑的执行顺序；不恢复旧训练或旧四节点队列。

## 最新终态：Haze官方200张平均双指标通过 — 2026-11-06

round3_haze_consensus_single_node_v1已完整结束，node0两张H100退出0、2worker齐全，600记录/200候选核验通过。
Haze由21.963813/.123679改善为22.857610/.120848，超过22.1/.19双门槛；PSNR+.893797dB，LPIPS下降2.2888%。
188/200张PSNR改善、174/200张LPIPS改善；最差PSNR退步1.309206dB，不能说逐图无损。无GT按图回退。
只有固定重叠块色偏协调：原模型/路线/图块不变，无训练，无新FLUX生成；GT仅计分，不参与偏置求解。
GPU评分104.201秒/.057889664 GPU-hours；CPU审计364.894秒、拼接666.357秒、核验4.965秒另计。node1/2保持待命。
固定Haze策略及全部200输出保留为后续统一方案的已通过候选：checkpoints/foundir_fullstack_haze_consensus_v1/accepted_followup_manifest.json。
不改旧producer或旧绑定源码，不重跑本轮，不逐图拼最优。当前全部十项5项通过：Haze、Lowlight双项过，Rain仅LPIPS过，Blur/Noise仍未过。
下一优先级Rain，再Noise，Blur暂缓；不能无诊断直接把Haze修正套到其他类型。官方是自适应验收基准，非独立盲测。
详见reports/foundir_fullstack_haze_consensus_v1_terminal_2026-11-06.{md,json}及result JSON。

## 当前执行：Haze重叠块色偏协调单次对照 — 2026-11-06

用户已授权execute，先查现有Haze重叠图块，再做证据支持的最小干预。原模型、路由、所有旧输出保留。
当前新方案是确定性拼接修正，不训练FLUX或RGB网络：相邻图块重叠区RGB均值差构成图，按重叠面积加权最小二乘求每块常量偏置。
约束加权拼接后的修正场整图RGB均值为零；只消除相对块间偏差，不拟合整图GT色彩。裁剪和uint8量化后均值可能轻微变化，必须记录。
系数固定1，不扫强度、不按GT选图。仅对原conditional_v4预测Haze的200图应用；另400图沿用原输出和指标，完整600图汇总。
CPU只读核验旧tiles后生成唯一候选，GT像素仅用于后续评分。采用单节点node0两卡评分，node1/2待命；不重跑FLUX、训练或路由。
新目录checkpoints/foundir_fullstack_haze_consensus_v1，job round3_haze_consensus_single_node_v1。
合成检查只证明数学接线，不能证明官方改善；需同时通过Haze22.1/.19，记录所有退步及十项总状态。
详见reports/foundir_fullstack_haze_consensus_v1_readiness_2026-11-06.md。

## 当前优先级：先处理Haze，再推进Rain/Noise — 2026-11-06

用户明确要求先处理Blur之外较容易的三类，优先Haze。Lowlight已有双项通过，保留；Blur暂停新GPU尝试，历史资产全部保留。
Haze官方200图当前21.963813/.123679，PSNR距22.1差.136187dB，LPIPS已过.19；这是最小当前缺口，不保证最容易或必过。
已复用现有200条误差记录完成CPU审计：200/200相对输入PSNR改善；RGB逐图均值偏差平均占平方误差17.299%，方向混合。
此分解仅定位问题，不是可部署增益；禁止用官方GT拟合偏置、逐图选择输出，禁止直接套官方平均偏置。
先核对当前step3000专家合法训练侧残差与训练/部署口径，再冻结有依据的最小干预。历史step400固定残差与全局RGB affine失败不重跑。
轻量诊断默认CPU/单节点，不因三台可用而全用；原纯FLUX路线、官方十项门槛、完整报告和全流程执行职责继续有效。
详见reports/foundir_fullstack_haze_priority_v1_2026-11-06.{md,json}。

## 最新终态：固定LQ起点官方对照失败，旧输出保留 — 2026-11-06

round2_blur_lq_start_v1已完整结束：3节点退出0、6worker、2250块、150张新Blur+450张原图结果，CPU核验通过。
Blur：原FLUX21.546762/.202622；固定LQ起点21.820219/.356149；未处理输入21.863903/.353603。
PSNR相对原模型小幅改善，LPIPS明显恶化，两项均未过25.9/.18，甚至略差于未处理输入均值。
不部署本轮，不自动扫更多起点；此单点失败不等于所有训练匹配的anchored flow不可行。
本轮没有训练/重路由/读其它测试集/GT挑输出。node2首输出延迟期间不曾盲目终止或重启，之后正常完成。
轻量任务单节点规则与纯FLUX开发/官方门槛区分继续有效。下一轮须针对真实恢复保真与状态分布匹配，不能将保留输入当恢复成功。
详见reports/foundir_fullstack_lq_start_v1_terminal_2026-11-06.{md,json}及result JSON。

## 当前执行：固定LQ latent起点官方对照 — 2026-11-06

用户已授权execute。新run round2_blur_lq_start_v1：只对原conditional_v4预测为Blur的150张官方图重生成；
另外450图沿用未变的原专家输出/指标，最终仍完整600图验收。不是重训练、重路由或Restormer恢复。
唯一固定干预：原50步shifted时间表选择第一个t<=.5，即512图块1024 token时index43、t=.4716968238353729，
以(1-t)*encoded_LQ+t*同seed噪声为初态，执行原suffix7步。保留原模型/LoRA/提示、reference_time10、CFG1、
原alpha=.75双uint8、512/128拼接及pyiqa口径。官方GT只在输出落盘后评分，不选起点/阈值/候选。
这是推理反事实；初态条件分布不同于训练clean+noise，不能预先宣称必改善或失败即路线不可能。
2250块/15750次native CFG模型调用，属于较重完整生成，使用现有3节点×2H100，每rank375块；
轻量任务默认单节点规则不变。本轮预计20–40分钟仅粗估，须按实际日志更新。常驻执行器领取，不要求用户重新开终端。
4项核心CPU接线测试、600图六分片完整性、源/权重/提示哈希预检通过；GPU执行状态须看新队列日志。
旧绑定源码与资产不改；新输出checkpoints/foundir_fullstack_lq_start_v1逐块/逐图保存，失败不自动重试。
6worker完成且3节点退出0后CPU finish核验并推送小报告、费用与所有失败结果。
详见reports/foundir_fullstack_lq_start_v1_readiness_2026-11-06.md。

## 当前问题澄清：纯FLUX专家的开发通过，不等于官方通过 — 2026-11-06

用户所指是替换Restormer后的纯FLUX专家；不能拿更早异构v8系统身份差异代替回答。
Blur step10000/alpha.75当时在2595开发512裁图上均值26.559141/.151993过双指标，但另有尾部安全门槛失败；
当前官方150图为21.546762/.202622。Rain step800当时开发100图34.301181/.061867通过，
原报告明确官方/封存未访问；当前官方100图为32.034930/.075014，PSNR未过。
五专家接入PASS仅证明同环境原入口与bank单选输出一致，以及Rain开发100图复现，不是官方质量通过。
当时未做官方同口径验证，推迟发现差距；不测试本身不是数值退化原因。来源分布、512裁图与完整图拼接、
seed、torchmetrics与pyiqa等因素尚未逐项隔离，不能直接断言全部是过拟合、GPU差异或某一评分器导致。
后续报告分开写工程/接入通过、开发集指标通过、官方指标通过；只有原官方600口径十项都过才记官方目标完成。
对当前纯FLUX路线继续分析/改进，不擅自恢复旧Restormer或把旧v8缺失配置当作当前工作的前置阻塞。
详见reports/foundir_pure_flux_development_vs_official_2026-11-06.md。

## 最新结果：单节点Blur幅度对照已完成，四组均未达标 — 2026-11-06

round1_blur_strength_single_node_v1只用新node0两张H100，509.335秒/.282963854 GPU-hours，退出码0；
600条完整结果与600张候选PNG核验通过，CPU finish 25.609秒/退出码0。node1/2未参与。
残差0.5/0.75/1.25/1.5倍的Blur PSNR/LPIPS分别为22.129605/.275399、21.939185/.230787、
20.988326/.188090、20.336427/.184869；原部署21.546762/.202622。四组均未过25.9/.18双门槛。
不部署这些倍数，不拼接最好指标，不继续无边界扫系数；这个有限对照不构成所有FLUX路线不可行的证明。
下一轮重点是内容空间对应及训练/部署一致性，不能盲目重启历史小样本末段监督或空间锚定失败配方。
官方GT仅评分/分析，训练限合法训练数据；轻量任务默认单节点的规则继续有效。
详见reports/foundir_fullstack_blur_strength_v1_terminal_2026-11-06.{md,json}及result JSON。

## 资源选择规则：轻量任务默认单节点 — 2026-11-06

用户明确要求：容易且单节点足够的任务只用一个节点，不因有3台可用就默认分布式。
CPU汇总、分析和文档工作在CPU执行；短GPU评分/小诊断默认新node0单节点，按需要用1或2卡。
仅长训练、完整昂贵生成或单节点资源不足等有明确收益时使用多节点，并记录理由与实际GPU-hours。
节点数只改变执行分片/吞吐，不静默减少样本、改指标、global batch或科学门槛。
常驻执行器保持3台可用；每项任务通过nodes列表选择参与节点，其余节点保持待命，不要求用户重新启动。
此前600图输入基线已全部完成并核验，报告提交678e2e3；不要重跑，该三节点队列现无未完成任务。
全流程改进职责继续有效，保存历史资产、源码身份与小报告；先冻结下一轮方案再执行。

## 当前硬件与队列：三台新实例、六张H100 — 2026-11-06

用户确认当前仅3节点×每节点2张 NVIDIA H100 80GB HBM3（81559MiB）；旧四节点映射不再适用。
按用户浏览器顺序：新node0 notebook=37497631-24c8-4a19-907e-494c701599ab，IP=10.244.211.17；
新node1 notebook=2551e049-18b3-4c01-8490-735ced1e58cd，IP=10.244.47.55；
新node2 notebook=55ef95b9-862f-4ffa-a2ed-6eaa7156ccad，IP=10.245.39.185。
用户终端实测GPU空闲；代理仍需以新worker心跳、设备登记和任务日志确认真实执行。

旧四节点队列round1_official_input_baseline_v1已取消，未启动、无GPU结果；源码、receipt、队列与取消记录保留。
新常驻入口：bash smoke/start_foundir_fullstack_worker_3x2_v2_tmux.sh <0|1|2>。
新共享队列checkpoints/foundir_fullstack_executor_3x2_h100_v2，与旧队列隔离；启动核对IP、H100型号、显存和GPU UUID。
当前唯一排队工作是官方600张未处理输入的基线评分：新run round1_official_input_baseline_3x2_v2，6worker各100张。
不重跑已完成训练、路由或FLUX恢复。评分口径、全部600张与旧父资产不变；父身份读取保留原4节点绑定，执行分片为6。
新结果根checkpoints/foundir_fullstack_official_input_baseline_3x2_v2；GPU结束后CPU运行
`.conda/gpu-train-stable/bin/python -S smoke/audit_foundir_fullstack_official_baseline_3x2_v2.py finish`并推小JSON及耗时/退出码。
全流程研究和执行职责继续有效；后续新任务按当前3×2资源实现，不把旧八卡参数或旧节点编号静默复用。
详见reports/foundir_fullstack_3x2_h100_readiness_2026-11-06.md。

## 当前最高优先级：本工作区全流程负责，持续改进直到官方FoundIR达标 — 2026-11-06

用户明确授权：另一执行端目前不可用，后续工作统一在这里推进；分析本轮问题、改进实现/训练、
在官方FoundIR测试集评价，未通过则继续下一轮，直到原官方验收条件通过。
本代理职责是完整研究与工程执行：根因分析、方案与代码、训练/推理/环境修复、CPU/GPU任务协调、
监控恢复、完整性核验、指标解释、实验账本和GitHub同步；不是只检查日志、给GPU命令或等待另一代理决策。
本节覆盖历史“CPU代理仅同步/报告”“本轮不允许改配方/新训练”“没有后续授权”等任务阶段限制。
新一轮改动可自主实施，但必须明确依据、冻结配置并保留历史版本/资产，不能改写正在运行的绑定源码。

当前验收范围延续官方五类600张原图：Blur150/Noise100/Haze200/Rain100/Lowlight50。
统一部署配置下五类PSNR/LPIPS十项全部通过才完成：Blur≥25.9/≤.18，Noise≥38.9/≤.10，
Haze≥22.1/≤.19，Rain≥34.2/≤.08，Lowlight≥18.9/≤.28。不能拼接不同轮次最好指标冒充统一通过。
只用官方测试集做效果验收；训练及监督只用合法训练数据。官方GT不进loss/梯度、逐图选路或选输出。
用户已授权根据官方结果迭代，因此这是反复使用的自适应基准，不能称独立盲测或未见过泛化证据。
保留所有失败结果和真实成本，工程退出0不等于指标通过，不通过时基于证据修正，不原样反复耗算力。

已完成V4官方轮（24d1434）：600/600，四节点与finish均status0；V4总体25.421806/.141534，
相对image-only +.749665dB/LPIPS下降4.739055%，仅Lowlight双项通过。已知类别参考也未全部达标，
下一轮不能只加路由训练就声称能解决专家恢复差距。官方累计106.544444GPU-hours，历史费用另列。
默认复用四节点×两卡H100/H200及现有共享环境，实际可用资源以现场确认和runtime为准。
有GPU执行通道就直接执行授权任务；若环境没有控制通道，先建立可审计的执行入口并给一次性启动步骤，
明确未启动事实，不能把缺远程通道推诿成只负责给命令，也不能假报GPU进度。
完整官方评价无需每轮重复索取已给出的执行授权；不自行扩大到其他官方类别或外部付费资源。
每轮及时推小进度和终态报告，保留数据/权重/缓存/图像/原日志及本地无关修改，不上传大资产。

## 官方V4评分身份修复：复用原路由与不限时授权 — 2026-11-05

四节点首次render在评分身份比较处全部退出，0恢复块；失败费用0.918888889 GPU-hours保留。
用户已授权修复。只新增metricfix执行层，不修改已绑定的原producer/contract/unlimited执行层或receipt/plan。
评分网络先按原默认设置在CPU构造并核验完整state哈希，再将整个Module迁移到GPU并复核全部state。
保留pyiqa空dummy_param的注册身份，不删除真实权重检查、不改变指标默认设置/精度/评分计算。
本轮仍是固定官方600图、三组、step4800、四节点×两卡H100/H200，原21,828块计划与不限时授权不变。
新CPU命令：`bash smoke/run_foundir_official_vlm_v4_metricfix.sh prepare`。
通过后四GPU终端分别运行：`bash smoke/start_foundir_official_vlm_v4_metricfix_tmux.sh <0|1|2|3>`。
新日志`logs/foundir_official_vlm_router_v4_4x2_render_node<N>_metricfix.log`；原失败日志/attempt不覆盖。
监测必须同时检查METRIC_DEVICE_VERIFIED、实际TILE落盘及退出码；CPU通过不当作CUDA已通过。
不重跑训练/route/plan，不重建缓存，不新增图像测试集或独立测速，不自动重启循环。
8worker完成、4节点status0后CPU运行`bash smoke/run_foundir_official_vlm_v4_metricfix.sh finish`，推小报告和累计耗时。
详见reports/foundir_official_vlm_v4_metricfix_readiness_2026-11-05.md。

## 当前授权：立即恢复官方600评价，不设最长时限 — 2026-11-05

用户原话：“no need for max hour limit, just execute and start the exp”。
这已批准本轮四节点×两卡的完整render及finish；不再索要具体GPU-hours批准，
不设置wall-hour/GPU-hour上限，也不伪造一个很大的有限cap。本节覆盖下方预算等待规则。
仅取消本轮恢复/评分的时间上限；模型/step4800/官方600/三组对照/种子/精度/拼接/指标不变。
不重训、不重新route、不使用其他测试集、不新增测速或质量gate、不自动反复重启错误任务。

route/plan已正常完成：600图、8worker、四节点status0；VLM590/600，图像路由546/600；
21,828块恢复计划SHA256为5f3823ed90320639bf804e7e29d0a891a5e0edb92e8b2761b51ac55fd90ea53d。
保留全部已完成缓存和历史费用；旧123.24 GPU-hours/15.4小时只是外推，不是已测速度或门槛。
不修改已绑定的原producer源码/receipt/plan，以免使现有缓存失效。
新增unlimited执行覆盖层记录用户授权；render_budget中上限为null，shell不调用timeout。
保留逐块/逐图原子保存、校验、节点锁、错误退出、失败尝试和实际GPU-hours记录。
相同命令可复用已核验进度；中断账目缺失需核对，不静默删除或从零重跑。

CPU代理同步GitHub后直接执行：
bash smoke/run_foundir_official_vlm_v4_unlimited.sh authorize
此命令只是落盘用户已给出的授权，不是再请求批准，不重新生成特征/决策。
然后在原四节点分别启动：
bash smoke/start_foundir_official_vlm_v4_unlimited_tmux.sh <0|1|2|3>
若CPU代理没有GPU进程控制权限，立即给用户四条对应命令，不谎称已经启动。
8个WORKER_COMPLETE且四节点END status0后，CPU运行：
bash smoke/run_foundir_official_vlm_v4_unlimited.sh finish
最终要求OFFICIAL_VLM_V4_RESULT_VERIFIED和finish END status0；不因退出0宣称指标达标。
只push小进度/终态报告，不上传权重、缓存、PNG或live logs。继续简单中文、事实与估计分开。
主代理当前无新工作区/GPU控制通道；已实现启动覆盖层不等于现场启动。
详细说明：reports/foundir_official_vlm_v4_unlimited_readiness_2026-11-05.md。

## 最新执行：官方V4支持四节点×两卡 — 2026-11-05

用户授权修复缺失pyiqa并适配四节点。本节覆盖本轮官方V4的旧两节点默认值；官方唯一测试集规则继续有效。
默认`FOUNDIR_OFFICIAL_NNODES=4`，4节点各2张H100/H200>=75GiB，8个独立worker；节点编号0..3。
显式设为2保留旧两节点兼容。各阶段都必须使用相同节点数，不支持同run静默换拓扑。
四节点用独立run `official_vlm_v4_frozen4800_v1_4x2`及带`_4x2`报告，旧两节点失败记录保留。
同节点两卡同型号，跨节点允许H100/H200，软件/精度策略签名一致；本轮CUDA仍待现场验收。
官方600对、冻结step4800、专家/原图拼块/评分/seed不变，不重训、不运行其他测试集。
route总cap仍12 GPU-hours：四节点每节点累计5400秒（两节点10800秒）。失败成本不得删除；旧失败prepare43秒另列历史账。
四节点8个ROUTE_COMPLETE、4节点END status0完整后CPU代理运行plan，推小preflight/route_plan及耗时简报。
完整恢复仍须用户看到精确计划后批准具体GPU-hours；批准值按实际world分配，不自行填写批准。
用户已明确授权补齐指定环境的pyiqa==0.1.16及官方评分权重；不得自动升级现有torch/CUDA/NumPy或改变指标协议。
依赖修复、prepare结果及具体命令见本轮四节点状态简报；CPU通过不等于官方CUDA或指标通过。

## 最高优先级：以后只用FoundIR官方测试集做测试与效果验收 — 2026-11-05

用户明确要求：今后不得再使用其他测试集做任何测试、对照、调参反馈或效果验收；
只能使用FoundIR官方测试集。不得再次新建/复用自建100图、开发集、内部封存集、
训练裁图probe等来代替官方结果或作为进入官方评价的前置效果门槛。本节覆盖下方
所有冲突的旧计划。历史结果保留并标明历史，不删除、不伪装成官方结果。
训练仍只能使用训练数据；训练loss和合成张量的CPU工程单测不是图像效果验收。
不得把“只用官方测试”解释为可以用官方GT做训练loss/梯度、逐图挑路由或输出。
官方集此前已多轮使用，必须保留每轮配置与结果，不能宣称从未使用的盲测。

当前授权：冻结V4已选step4800和五个既有FLUX专家，评价官方五类全部600对：
Blur150、Noise100、Haze200、Rain100、Lowlight50。不是完整20类1500对，
不自动扩到混合退化/其它类别。评价原图完整尺寸，使用官方pyiqa口径；
禁止以512裁图torchmetrics分数替代。对照为原image-only、本轮VLM和已知类别专家参考，
最后一组只是参考，类别标签不得进入前两组部署路由。无新训练或检查点选择。
V4历史100图结果已完成，99/100选对；不再重跑该集合或根据其绝对分数继续改专家。
旧官方runner含Restormer和旧Qwen路由，不能直接启动；必须接入当前纯FLUX冻结模型。
现场执行根仍为/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang；
主代理没有该目录挂载，未启动官方GPU评价。

官方入口已实现，尚未现场启动：smoke/run_foundir_official_vlm_router_v4.sh。
CPU代理先prepare，通过后提供两个节点route 0/1命令；四worker完成后运行plan。
官方600原图需至少19460块，不是600个512裁图；按上轮速度外推，恢复最少约110
GPU-hours、四卡约27.5普通小时（估算非承诺，三组不同路由还会增加）。
route阶段累计上限12使用GPU-hours；完整render须用户看到精确计划后明确批准预算，
不得自动沿用或扩大旧18GPU-hours，不自行填写authorize_render批准记录。
不增加质量门槛、不重训、不改配方；预算确认只是防止擅自启动长时间GPU任务。
每块/整图保存可恢复，CPU129项测试通过但官方CUDA/指标未实测；只同步小报告。
详见reports/foundir_official_vlm_router_v4_readiness_2026-11-05.md。

## 用户最新资源要求：V4改为两节点×两张H100 — 2026-11-05

本节覆盖下方V4四节点/H200限制。用户仅能提供两节点共4张H100，已要求实现兼容。
仍执行同一轮V4：13917训练裁图/400校准/100确认、6400步/global32、原损失/精度/seed/专家/恢复协议不变。
改用四个独立worker，不需要MASTER_ADDR或跨节点NCCL；节点编号只能0/1，各两张同型号H100/H200>=75GiB。
features/render必须四worker和两节点status0完整后再继续下一阶段，不遗漏或重复样本。
原137–138GiB峰值的具体分配来源尚未隔离，不能说已经证明是cuDNN workspace。
新显存路径仅在VAE encode/decode范围禁用cuDNN，使用原FP32的PyTorch原生卷积；保留query chunk256。
没有量化、降精度、缩图、换专家或自动offload。此为显式数值执行策略调整，不保证与旧H200缓存逐像素相等。
旧2960特征保留并复用，其producer不变；新缓存receipt绑定新的卷积策略和源码，报告须说明差异。
H100仅代码/CPU检查完成，真实CUDA显存/吞吐/指标仍待运行；不能把改硬件检查当GPU验收。
若OOM/非有限/超时，保留缓存/状态并回报，不自动重试、改策略或扩大预算。
特征每节点累计cap10800秒(3h)，两节点×两卡合计12GPU-hours；恢复每节点cap5400秒(1.5h)，合计6GPU-hours。
CPU训练仍1小时，prepare/finish各10分钟；总GPU预算仍18GPU-hours，不将减卡包装为速度保证。
入口不变：prepare；用户在两个节点分别运行features 0/1；CPU代理train；用户render 0/1；CPU代理finish。
本轮仍无旧分类门槛放行要求；工程身份、有限和预算检查保留。不得启动旧八分片入口或旧实验。
CPU代理可修改并发布上述资源适配，但不自行更改科学配方。GPU尚未自行启动。
详细适配记录：reports/foundir_expanded_vlm_router_v4_2node_h100_2026-11-05.md。

## 当前授权：一轮冻结图像路由修复＋独立来源验证 — 2026-11-04

本节覆盖下方“仅CPU诊断、不启动新训练”。用户明确授权一轮有边界修复与一次独立来源验证。
7fe87e5审计已结束：680条身份/特征复制/来源分离/20次曝光通过；不要重复该诊断。
Blur更像路由泛化问题，Low-light特征混淆/覆盖偏弱，未证明唯一根因，不能承诺这是最后一次。

当前expanded_frozen_image_v2：保留fresh_ce400_matched_v1真正image_only step400图像MLP并冻结，
fresh VLM MLP零输出起点，只训练VLM分数修正；不是继续联合更新image分支。
五个已训练FLUX专家、VAE、Qwen都冻结，不加CNN/Restormer/refiner、不恢复energy或旧400+8训练。
训练2560（每类512），calibration400、holdout400（每类各80），严格来源/GT/LQ哈希隔离。
原计划各128校准/验证不成立：合格Low-light仅723来源，少于512+128+128；未读新像素前改为80。
旧640训练特征复用，2720份新特征；旧40开发图及其来源排除，不针对两错例调参或补数据。
本地真实manifest预检已通过：fit2335不同来源、cal400/holdout400来源，三组无交叉。
新holdout只保证路由训练来源分离，不保证FLUX/Qwen预训练从未见过。

固定1600步/global32/每图20曝光，AdamW lr.001/wd.01/label_smoothing.05/grad_clip1；
400/800/1200/1600只用calibration最低CE选一次，选择先落盘，再读取holdout评分。
不扫系数/阈值、不挑holdout检查点。holdout筛选：总体正确数提高、每类不降、CE不升。
分类筛选不是PSNR/LPIPS/SoW成功；此轮不生成恢复图，也不自动下一阶段。
修正头见2560而冻结image-only只见旧640，本轮不是数据量匹配的VLM消融。

目的地CPU代理安全同步GitHub，保留无关修改/大资产，先用已验收环境执行：
.conda/gpu-train-stable/bin/python -S smoke/repair_foundir_frozen_image_router_v2.py prepare
通过后给用户四节点各两卡特征命令，节点编号0..3各一次：
bash smoke/start_foundir_frozen_image_router_v2_tmux.sh features <node_rank>
只本地torchrun两worker/全局八分片，不需要master IP/NCCL；H100/H200代码支持，CUDA显存待实测。
新任务只加载冻结编码器，不把旧138GiB五专家bank推理说成H100已合格。
八个worker完整、四节点status0后释放GPU，CPU代理手动执行同一授权阶段：
bash smoke/start_foundir_frozen_image_router_v2_tmux.sh train
CPU8线程，最终须FROZEN_IMAGE_V2_RESULT_VERIFIED和FROZEN_IMAGE_V2_END status=0。
每node缓存累计cap1小时（总最多8GPU-hours）；CPU累计cap1小时；不自动重试/扩预算。
逐图缓存及逐step VLM/Adam/RNG/slot保存；0..1600快照约18–22GiB，首启需30GiB空闲。
只有代码/CPU测试/真实清单预检已完成，主代理未启动真实GPU缓存或训练。

唯一执行根仍为/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang；
Python为该根下.conda/gpu-train-stable/bin/python。禁止在冻结旧HDD3执行。
详见reports/foundir_frozen_image_router_v2_readiness_2026-11-04.md。
CPU代理只推小结果/耗时/退出/峰值报告，主代理分析；不推tensor、权重、数据或live logs，不自行改配方。

## 已结束：两张严重错选的CPU原因核对；不启动新训练 — 2026-11-04

匹配画质补测已正常结束（dd06f76），2H200/76秒/status0；不要再跑四图。
VLM38/40对image-only33/40：平均+1.280642dB/LPIPS下降.004140，但Blur两项退步；
相对旧soft，PSNR改善但LPIPS仍差.002063，不能宣称全面达标或官方测试成功。

用户授权分析错选与训练覆盖。主代理已重算全部40图分支分数：
VLM版内部图像分支20/40，VLM修正帮19张、害1张，最终38/40。
内部图像分支不是独立image-only的33/40对照；联合训练只监督相加分数。
0007534 Blur原内部选择正确，被VLM强改Haze；0479048 Low-light两个完整模型都错选Noise。
VLM训练640/640、CE .001782，开发CE .451962，不能把追加同数据步数当必然修复。
详细证据/后续条件分支见reports/foundir_router_generalization_audit_2026-11-04.md，
以及foundir_router_score_audit_2026-11-04.json。不以关VLM就能解决或“唯一缺数据”下结论。

当前唯一新入口为CPU只读诊断：全部680条缓存身份、标签映射、裁图、来源分离和曝光，
及全部40图相对训练图像/VLM特征的固定近邻覆盖。不是训练、路由阈值搜索或像素评价。
禁止对开发图扫系数/阈值、挑checkpoint、硬编码回退、改标签或回灌训练。
不读GT/LQ像素、不加载大模型/reference latent、不恢复CNN/Restormer/energy或旧MoE训练。

目的地CPU代理安全同步origin/main，保留本地无关修改、大资产及绑定源码后，直接运行：
bash smoke/run_foundir_router_generalization_audit_v1_cpu.sh
根仍为/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang，
使用.conda/gpu-train-stable/bin/python，CPU8线程/0GPU-hours/单次cap600秒+30秒清理。
无自动重试/扩预算；超时回报日志。无需GPU，也不用给用户新的GPU启动命令。
本机未挂载目的地，缓存覆盖尚未实测；不能把静态阅读/合成单测当现场核对通过。
成功须ROUTER_FEATURE_COVERAGE_PASS和ROUTER_GENERALIZATION_AUDIT_END status=0。
仅推reports/foundir_router_feature_coverage_2026-11-04.json及耗时/退出简报；
不推模型、缓存、图片或live logs。主代理据结果决定下一步，不让CPU代理自改训练。

## 已结束：只补4个输出，完成匹配的路由画质对照 — 2026-11-04

本节覆盖下方CPU训练待执行等旧状态。fresh_ce400_matched_v1已正常结束（23ac684）：
CPU8线程/212秒/0GPU-hours，两组各400步，image_only开发33/40正确，image_plus_vlm为38/40。
这是分类结果，不是新PSNR/LPIPS。不要重复CPU训练，不恢复旧400+8或energy任务。

用户已授权当前matched_top1_four_missing_v1。两组固定最终400步预测，均用预测top-1，
送入同一冻结五专家FLUX内部bank，保持原prompt/alpha/量化/crop/seed/读出。
只补4个此前不存在的input+predicted-expert输出；其余76/80行在核验原JSON及输出SHA后复用。
共47个不同预测路由，其中43个已有；原40个正确类别参考+旧4个错误路由缓存为44项，
不把缓存目录项数误算成预测数量。不用GT改选专家、不按质量挑缓存、不重新生成40图。

缺失：image_only的0009805/0391222/0377282→Haze；image_plus_vlm的0007534→Haze。
完整key前缀validation_foundir_train_。四个新输出均原Haze50步/alpha1。
GT仅生成后计分；正确类别单选只是参考，不是实际预测。报告两组各40图/五类/逐图及配对差值。
相对旧step408 soft同时改变训练和部署策略，不能把整段提升都算作VLM贡献；
单seed小开发集不代表官方SoW/Rain100达标、混合退化或泛化成功。

本轮轻量例外：单节点2×H200，每卡至少139GiB；不占四节点八卡。
旧真实纯推理峰值137.933GiB/卡，H10080GB不符合当前内存要求，入口明确拒绝。
不为兼容H100擅改精度/offload/科学口径。保留父运行软件/指标/内存策略，允许记录驱动差异，
不宣称跨驱动逐像素相等。累计cap900秒/15分钟/.5使用GPU-hour，另最多30秒清理；
此前四图77秒只作参考，不保证本轮ETA。若整机独占，另记整机占用成本。
每图原子保存，可恢复已核验完成图；失败时间累计，缺失结束账需核实，不自动重启/扩预算。

主代理已完成33项CPU测试（11新+22回归）、shell/Python静态检查；测试夹具不是新实验指标。
尚未在目的地/GPU运行。唯一执行根为/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。
CPU代理安全同步origin/main，保留无关脏文件、大资产和历史绑定文件，然后用已验收环境运行：
.conda/gpu-train-stable/bin/python -S smoke/evaluate_foundir_supervised_router_restoration_v1.py prepare
此预检不读像素/加载模型；通过后给用户单节点两张H200的tmux入口：
bash smoke/start_foundir_supervised_router_restoration_v1_tmux.sh
不要发旧四节点或CPU训练命令。成功须MATCHED_RESTORATION_RESULT_VERIFIED和MATCHED_RESTORATION_END status=0。
只推reports/foundir_supervised_router_restoration_terminal_2026-11-04.json和简短耗时/退出报告；
不推tensor/图片/权重/缓存/live logs。不自动训练或启动下一阶段。
详细范围、恢复规则见reports/foundir_supervised_router_restoration_readiness_2026-11-04.md。

## 历史授权：CPU完整监督训练VLM路由；单退化预测top-1 — 2026-11-04

用户在四图补测后授权execute修复。四图补测已终态：a68d01a，2H200/77秒/.042778使用GPU-hours；
不再跑旧四图或400+8质量资格。四错例主要损失来自选错专家及其原协议，直接把旧路由全部改hard不能解决；
正确36图又有混合损害。五个专家一直冻结，不能说专家权重被训坏。

明确代码缺口：旧400分类步VLM关闭，VLM仅训练8质量步/256条样本曝光；
argmax所选prompt/读出协议无离散导数，虽然soft权重仍有真实修图梯度，不能视为VLM已充分学会五类。
这不证明唯一根因，也不保证新分类训练一定能泛化。本轮不盲目延长昂贵完整FLUX反传。

当前唯一新任务fresh_ce400_matched_v1：CPU、原640训练特征/40开发特征，五类各128/8；
两组均fresh模型/Adam、相同图像MLP初始化、同样本顺序、400步/global32/lr.001/纯分类CE。
image_only不启用VLM，image_plus_vlm从第1步启用VLM修正；两组各曝光训练记录20遍。
不读取旧qualification权重，不改专家、底座、VAE或Qwen，不加CNN/Restormer/refiner。
只读已有image/vlm缓存，不取reference latent，不解码GT/LQ像素，不重生成缓存。
训练类别只用于CE；推理只接图像/VLM特征，不传类别。两组都完成后才读40开发特征，
固定最终400步，无开发集挑步/早停。报告分类混淆、原4错例转变及新增错误，不虚构新PSNR/LPIPS。

当前singleton部署候选用模型预测top-1/one-hot进入原内部FLUX bank；
只要选对即可沿用已验证单专家能力，不默认混入其它专家。选错仍会失败；
这不是“hard本身修好错误”，也不是double/triple混合退化方案，更不是新论文贡献已成立。
生成质量验收留待CPU结果回传后明确安排，不自动GPU/正式训练。旧纯推理峰值137.933GiB，
H10080GB的完整路径依然未验收；CPU阶段不受GPU型号影响，不等于已修好显存。

默认CPU8线程，显式CUDA_VISIBLE_DEVICES为空；累计硬cap30分钟（非ETA）、0GPU-hours。
每完整步保存模型/Adam/CPU+Python RNG/slot及SHA，两组含step0共802份，约24GiB，
首启需30GiB空闲。全部旧文件不动，同入口恢复相同CPU运行栈；损坏不回退，失败时间不清零。
不要把所有checkpoint/缓存推GitHub。主代理已通过11项新CPU测试（含合成双组400步与精确恢复）；
本机无目的地缓存，实际训练未执行，不能把合成测试当实验结果。

目的地CPU代理同步最新origin/main，保留无关脏文件/大资产，直接运行：
bash smoke/start_foundir_router_supervised_repair_v1_tmux.sh
这是CPU任务，无需用户启动GPU；prepare→train→verify均已在入口内。
纯元数据预检可用.conda/gpu-train-stable/bin/python -S smoke/train_foundir_router_supervised_repair_v1.py prepare。
唯一执行根仍为/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang；
不在旧HDD3执行，不用系统Python替代已验收环境。GPU入口本轮没有新增。
成功须SUPERVISED_ROUTER_FINAL_VERIFIED及SUPERVISED_ROUTER_CPU_END status=0；
回传reports/foundir_router_supervised_repair_terminal_2026-11-04.json及简短耗时/退出记录。
只推小报告，不上传模型/Adam/特征/live logs。主代理负责分析与下一步，不让CPU代理自改配方。
详见reports/foundir_router_supervised_repair_readiness_2026-11-04.md。

## 当前授权：只补4张模型自选主专家的单选输出 — 2026-11-04

用户已授权execute。本节覆盖下方“只做CPU分析/尚未授权补测”。CPU报告f66143f已回传并独立核对：
36张主专家正确，soft相对正确one-hot平均PSNR低.057323dB、LPIPS高.004949；
4张错误平均低11.168748dB、高.116694。错误图仍混合选择/推理协议和混合比例的影响，
不能把全部损失叫分类损失。五个专家始终冻结；不恢复任何旧训练，不自动formal。

当前唯一新任务：predicted_hard_four_v1，四张固定错误图各生成一次模型step408已选主专家的one-hot输出。
不重新训练/运行router，不用GT改选专家，不改原权重、裁图、种子、prompt、alpha、量化或50步流程。
只读这4张原开发图，GT只在完整生成后评分。禁止加图、遍历五专家、逐图选最优或重跑40图基线。
保留历史C=正确类别单选、S=原soft；新H=模型argmax单选，拆分S-C=(H-C)+(S-H)。
符号采用两项均正数表示损害；H-C包含主专家附带的prompt/alpha等协议，不是纯分类的因果估计。

本次属于轻量推理例外：单节点2张同型号H100/H200，各2图，不占4节点8卡，不做反传/NCCL训练。
建议沿用已跑通的H200；H100仅代码路径支持、待实测，不能据此声称138GiB的完整训练已适配H100。
沿用旧数值软件、LPIPS与内存策略，记录GPU/driver差异，不宣称跨硬件逐像素相等。
全任务累计cap30分钟/1使用GPU-hour，另最多30秒清理；不是实测ETA，不自动扩预算。
每图独立原子保存uint8与小JSON；相同入口跳过已核验完成图；孤立tensor保留。
失败尝试成本累积；若节点硬断导致attempt结束账缺失，先根据日志补记，不默默清零预算。

主代理已实现入口，22项CPU回归通过；本机没有目的地挂载/GPU，尚未运行真实补测。
目的地CPU代理安全同步GitHub后运行：
.conda/gpu-train-stable/bin/python -S smoke/diagnose_foundir_router_hard_counterfactual_v1.py prepare
只读旧元数据并写新receipt，不解码图片、不占GPU。通过后给用户单节点tmux入口：
bash smoke/start_foundir_router_hard_counterfactual_v1_tmux.sh
只在唯一新工作根执行；不提供旧四节点训练命令。日志logs/foundir_router_hard_counterfactual_v1.log。
终态需要ROUTER_HARD_RESULT_VERIFIED和ROUTER_HARD_END status=0。
CPU代理只推小报告reports/foundir_router_hard_counterfactual_terminal_2026-11-04.json，
另简短说明exit/耗时/两卡GPU-hours及累计失败成本；不要推图片/tensor/权重/live logs。
详见reports/foundir_router_hard_counterfactual_readiness_2026-11-04.md。

## 当前状态：路由资格已结束；只做CPU逐图误差拆分 — 2026-11-04

用户授权执行终态后的误差分析。本节覆盖下方“资格待启动/运行中”的历史状态。
qualification_mixed_driver_4x2_v1已完成400分类＋8质量步、40图前后评价，四节点status0；
8H200实测，峰值138.089GiB/卡，不代表H10080GB训练通过。不要重新启动旧资格或自动formal。
五个专家仍冻结；当前不是共享LoRA/专家被训练改坏。主专家最终36/40正确，错4张在Noise/Low-light。
Haze8/8选对仍比同图单选低.958344dB且LPIPS高.009953，证明选对也可能混坏，但不是每图都退步。
这不是VLM因果贡献消融，不把8质量步当路线终判；资格权重不seed正式训练。

当前新入口只有CPU脚本smoke/analyze_foundir_frozen_router_errors_v1.py：
在唯一新工作根用.conda/gpu-train-stable/bin/python -S运行。只读receipt/result、
400/408 checkpoint小JSON和120个评价JSON，不读图片、大tensor、不重新算指标、不占GPU。
核对身份/均值后导出40图完整记录、正确/错误组差距贡献、主专家转变及缺失对照清单，
输出reports/foundir_frozen_router_error_breakdown_2026-11-04.json。
主代理本地已完成汇总分析/11项CPU单测，但看不到新挂载；真实逐图拆分须由目的地CPU运行并推小报告，
不能把测试夹具当真实实验结果。成功标志FROZEN_ROUTER_ERROR_BREAKDOWN_PASS。

正确36张已有soft与同主专家one-hot对照；错误4张的差距混合了选择与比例影响，不能全算分类损失。
若后续需要精确因果拆分，最多补这4张“模型argmax主专家one-hot”的输出，不能用GT选主专家或逐图择优。
当前仅生成缺失清单，未启动补测/新训练，不自动扩预算。先回传CPU报告，再决定配方。
跨节点不同驱动/型号授权、官方SoW600重复评测授权保持；均不等于H100显存已合格。
保持历史绑定源码/权重/缓存不动，保留无关脏文件；只focused push小报告。
详见reports/foundir_frozen_router_error_analysis_readiness_2026-11-04.md。

## 用户明确要求：跨节点允许不同 NVIDIA 驱动版本 — 2026-11-04

用户授权当前及后续跨节点入口支持 H100/H200 混用，以及节点间 NVIDIA 驱动版本不同。
当前实际节点为 H100/H200、570.124.06/595.58.03（含 proprietary/open kernel module）。
不得仅因跨节点驱动 release 不同拒绝运行；保留原始逐节点驱动身份，不能改写成同一版本。
仍要求每节点内部两卡同型号且至少75GiB，跨节点 PyTorch/CUDA/cuDNN/transformers/评分版本及显存策略一致。
各节点在同一run内的runtime身份仍严格不变；换驱动续训不冒称严格恢复。
真实CUDA初始化、NCCL通信、前向parity和有限梯度检查必须通过；代码支持不等于已通过GPU资格。
不得热改正在运行任务绑定源码或收据、自动杀任务或扩预算。历史驱动必须相同条款被本节覆盖，旧脚本不自动视为已适配。
当前使用独立入口 smoke/start_foundir_frozen_expert_router_mixed_driver_v1_tmux.sh，
run=qualification_mixed_driver_4x2_v1，launch=launch_frozen_router_mixed_driver_v1，port29705。
训练科学配方/数据/步数/global32和每节点3小时cap不变；旧尝试成本仍须累计，重试不代表授权额外预算。
旧资格当前未保存任何训练step，新ID记录的是修改运行兼容策略后的独立尝试，不冒充checkpoint续训。
另有node0:29703连接超时记录；驱动兼容修改不代表网络已连通，启动前应核实当前地址与端口。

完整项目规则见 AGENTS.md；本文件记录用户指定的混合驱动要求。
