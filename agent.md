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
