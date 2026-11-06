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

## 当前授权：V4扩大训练＋五类恢复完整实验 — 2026-11-05

用户明确要求推进更大实验，不再被一串小资格试验卡住。本节覆盖下面“仅审计/暂不训练”。
V3及fit审计已结束（bfcaef5）：训练97.15→98.55%，校准91.5→90.5%，不继续旧2560固定裁图加步数。
当前一次完整实验expanded_conditional_v4，prepare→特征→CPU6400步→100图五类恢复→汇总。
没有额外25步训练或“每类分类零退步”放行门槛；有限/身份/预算/完整性工程检查不能跳过。

方向仍是PNG右图：原冻结图像路由＋VLM token分数修正。image-only只作原分支和对照，不替代VLM。
新VLM修正头同时读取原图像五分数（中心化/5），VLM视觉特征3584维LayerNorm后拼接，MLP256输出5。
原图像分支、已训练五个FLUX专家、Qwen、VAE全冻结；不加CNN/Restormer/refiner/energy。
fresh零输出VLM/fresh Adam，6400步/global32/lr.001/wd.01/smoothing.05/clip1；
保持保护hinge权重1/余量.2，不扫系数。只用LQ图像/视觉token，不冒称生成的退化文字等级。

真实清单已核对：fit13917裁图、4639个来源×类型、跨类型4192不同来源；不是13917独立来源。
Blur/Noise/Haze/Rain各1024来源×3裁图，Lowlight543×3；校准400，独立确认100每类20。
保留原2560训练+400校准缓存2960份，新提取11457份。原400holdout与nontrain全部来源/GT/LQ排除。
Lowlight543包含合法的其他类别旧fit来源，不把旧验证移入训练；类内样本不够不凑数。
展开SHA 568ffd35a6cad695a95f1d86f90a2826f606e596a10f4b5b11f58351e64d6432。
1600/3200/4800/6400按校准最低CE选一次、平局最早，selection落盘后才读新100确认特征。
分类退步仍继续预定恢复评价，不据确认集重选模型/步数。预提取确认特征不等于允许训练读取。

在新100图比较原image-only、旧V3 step800、新V4及已知类别专家参考。前三组无GT选专家；
第四组明确是类别参考，不是部署/GT挑最优。固定top1进入原FLUX bank，同输入同路由只生成一次。
保留四类50步、Rain第26次调用endpoint、原alpha和量化；不改专家、不拼最好指标、不按GT回退。
报告同图PSNR/LPIPS总体与每类，保留原五类数值目标。100裁图结果不能宣称完整官方SoW通过；
仍是singleton top1，不宣称混合退化、逐token MoE或论文消融全完成。大数据+新结构不是单因素消融。

执行根/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang；
Python为其.conda/gpu-train-stable/bin/python。主代理无目的地挂载，代码完成不代表现场已启动。
GPU使用四节点×2H200，每卡≥139GiB，独立八分片，无跨节点NCCL。旧峰值137–138GiB，
80GB H100本入口明确未合格并拒绝，不静默改精度/offload/裁图冒称兼容。
CPU代理安全同步保留修改/大资产，按同一授权实验协调：
1. bash smoke/run_foundir_expanded_vlm_router_v4.sh prepare
2. 给用户四节点各自的 bash smoke/start_foundir_expanded_vlm_router_v4_tmux.sh features <0..3>
3. 全8worker/四节点status0后CPU代理运行 bash smoke/run_foundir_expanded_vlm_router_v4.sh train
4. TRAIN_VERIFIED/status0后，无需再等分类门槛，给用户同四节点的tmux入口 render <0..3>
5. 全部渲染成功后CPU代理运行 bash smoke/run_foundir_expanded_vlm_router_v4.sh finish
阶段顺序是资源交接，不是额外小实验/科学gate；CPU代理不自行改配方/扩大预算或提前读保护像素。

cap：特征每节点90分共12使用GPU-hours；恢复每节点45分共6使用GPU-hours；CPU训练1小时；
prepare/finish各10分。失败成本累计，超时保留进度不自动加预算；GPU总cap18小时单位是GPU-hours。
每图/每完整optimizer步保存，0..6400状态约80–100GiB，prepare需150GiB空闲；旧产物不动。
只push训练/最终报告和耗时退出简报，不push权重/特征/图片/live logs。普通小时与GPU-hours分开。
最终须EXPANDED_V4_RESULT_VERIFIED和finish的EXPANDED_V4_END status=0。
详见reports/foundir_expanded_vlm_router_v4_readiness_2026-11-05.md；继续简单中文、结论先行。

## 历史执行：V3训练侧核对；不盲目追加步数 — 2026-11-04

用户授权先核对已保存的800/1600步模型，再决定是否给VLM修正头补充图像路由五个分数。
最终仍坚持右图：图像路由＋VLM token修正；image-only仅冻结分支/对照，不是替代方案。
本节覆盖下面V3“待训练”状态；不要再次运行V3训练或旧GPU缓存。

V3已结束，73bc85e：1600步/345秒/CPU8线程/status0；校准366→366→363→362正确，
四个检查点均不满足每类不降。800步仅失败诊断：同校准组误改12→11、纠错28→28。
Haze仍73→66，Lowlight77→76；不把校准366和旧holdout374混成同一验证集。
没有新100图确认或PSNR/LPIPS。原专家、缓存、checkpoint与历史源码保持不动。

本轮fit800_1600_v1：仅加载2560训练特征，前向评价固定两个保存模型；
校准预测直接复用原JSON。记录总体/每类纠错与误改率、保护损失及历史last100训练统计。
按原图像路由正确/错误数作分母，不能直接比较2560与400图的错误张数。
不训练、不改结构、不读calibration/holdout/预留100图特征或新GT/LQ像素，不自动GPU。
若训练已学会而校准仍差，支持泛化差距；若训练也差，先查目标/优化。
“让VLM修正头同时看图像路由五个分数”仍是待结果支持的候选，不是已证实根因或已实现方案。
后续由主代理读回实际结果决定；CPU代理不自行加步数、换结构或调参。

唯一执行根/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang，
使用其.conda/gpu-train-stable/bin/python；主代理无目的地挂载，不能声称已现场运行。
CPU代理安全同步GitHub保留本地修改/大资产后直接运行：
bash smoke/run_foundir_vlm_preserve_fit_audit_v1_cpu.sh
CPU8线程/0GPU-hours/累计cap600秒+30秒清理；与旧CPU训练共用锁，不杀无关任务。
成功须VLM_FIT_AUDIT_RESULT_VERIFIED和VLM_FIT_AUDIT_END status=0。
仅push reports/foundir_vlm_preserve_fit_audit_2026-11-04.json与耗时/退出简报；
不push训练特征、权重、逐图缓存、图片、live logs。保持简单中文、事实与假设分开。
详见reports/foundir_vlm_preserve_fit_audit_readiness_2026-11-04.md。

## 历史授权：只修复右图VLM分数修正，CPU训练V3 — 2026-11-04

用户明确坚持PNG右图“图像路由＋VLM token修正分数”。image-only只是冻结分支/诊断对照，
不是最终替代方案，也不应成为必须先另训一个image-only才能继续的前置条件。
本节覆盖下方V2待运行及旧energy/EBT/MoE优先级。用户已授权execute本轮V3修复。

V2 expanded_frozen_image_v2已结束，3d6d9fd：独立400图350→374正确，纠正36/误改12，
Haze76→69、Blur77→76，其余三类改善；每类不降筛选失败，无新PSNR/LPIPS。
不重跑旧GPU缓存或V2训练。V2真实特征峰值137.473185GiB/卡，不能声称80GB H100已通过。

新preserve_correct_v3仍是原图像MLP冻结＋fresh零输出VLM MLP；五个训练好FLUX专家、
Qwen、VAE全冻结。只用已有LQ图像6144维/Qwen视觉merger均值3584维特征。
不加CNN/Restormer/RGB refiner/energy/备用门控，不把视觉tokens说成生成的退化文字标签。
保留V2的2560训练/400校准、1600步/global32/相同seed顺序/lr.001/wd.01/smoothing.05/clip1。
新增一个训练惩罚：仅对图像路由原本正确的训练样本，惩罚最终正确类别被错误类别反超；
具体hinge余量=min(原正确margin,.2)，系数固定1，完整batch平均。图像路由原本错时不保护。
这是分类CE之外的代价调整，不增加新信息、不保证新图no-harm，也不保证一定改善。

400/800/1200/1600只评校准集。候选须总体正确数高于图像对照、每类不降、CE不升，
并比V2同校准预测误改更少且纠错不少；符合者中选最低CE，平局最早。
无符合者仍保留最低CE供失败诊断，screen=false，不能把“VLM什么都不改”当成功。
旧400 holdout不加载特征、不作选型；不能把校准成绩称独立泛化/画质/官方SoW或混合退化成功。
没有自动后续GPU或扩大预算。训练2560对冻结基线640仍非数据量匹配VLM消融。

真实manifest元数据已核对：排除旧所有fit/calibration/holdout/旧40与nontrain身份后，
Lowlight仅剩37来源；固定预留每类20，共100不同来源，source/GT/LQ全隔离。
预留SHA 57596103ccf992e6f47dbcdb0bce0c1c0d3a8b7ee02e78837fe43b929b6043a7。
当前只保存元数据，不读其像素/特征、不启动提取。未来确认需单独安排，100图不是论文充分验证，
也不保证FLUX/Qwen从未见过。不得返回旧400图冒充全新盲测。

唯一执行根/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang，
环境为其.conda/gpu-train-stable/bin/python。主代理此处无目的地挂载，不能声称已启动训练。
CPU代理安全同步GitHub保留无关修改/大资产后，执行：
bash smoke/start_foundir_vlm_preserve_router_v3_tmux.sh
该入口CPU-only prepare→train→verify，8线程、累计cap1小时、0GPU-hours，无需GPU命令。
首启需30GiB，0..1600每步保存VLM/Adam/RNG/采样slot，同runtime恢复；旧产物不动。
完成须VLM_PRESERVE_V3_RESULT_VERIFIED与VLM_PRESERVE_V3_END status=0。
只回传reports/foundir_vlm_preserve_router_v3_terminal_2026-11-04.json和耗时/退出简报；
不push权重/特征/图像/live logs，不自行改配方/预算。主代理分析，CPU代理执行与回报。
继续用简单中文、先结论，区分事实和假设；focused commit/push保证新代理看到同一决定。
细节及成功/停止边界：reports/foundir_vlm_preserve_router_v3_readiness_2026-11-04.md。

## 历史授权：一轮冻结图像路由修复＋独立来源验证 — 2026-11-04

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

# Legacy instructions — current authority moved to AGENTS.md

## 当前授权：冻结已有五专家，训练图像路由＋VLM分数修正 — 2026-11-04

本节覆盖下方旧“先五库接入/未训练”状态。用户已授权下一阶段。五库接入已结束：
116/116单选与同环境原入口逐像素相同，Rain原100开发图34.30118094/.06186744；
固定90%混合损害Rain/Haze等，不直接部署，也不是可学习router已失败。

当前只运行独立路由工程资格：五份已有FLUX LoRA、底座、原生VAE全部冻结；
四类BF16和Rain step800 FP32保留，无CNN、Restormer、RGB refiner、vanilla替代或energy。
图像分数来自冻结FLUX参考token的mean/std6144维；VLM修正来自已适配Qwen2.5-VL
visual merger token均值3584维。各过小MLP，VLM末层零初始化，分数相加再softmax。
先400步图像分类初始化（VLM关、lr.001），再8步完整26/50次生成反传（VLM开、lr.0001）；
global32、同Adam，loss=10 RGB MSE＋LPIPS(Blur .2/其它 .05)＋.1 CE。
类别仅训练监督，不作为实际router输入或GT选路；不强制给其它专家保留最低比例。

沿用已接入兼容规则：主专家argmax决定prompt/alpha/量化/读出，Rain原50步时间表第26次端点，
其它50次；第26次之后Rain份额归零、其它归一化。内部混LoRA增量，不是平均五张成图。
完整生成路径显式call-index/gate重算，禁止复用推理计数器；block重算必须重新绑定gate。
离散argmax不反传；uint8前向严格原口径、反向STE近似，不宣称是真实取整导数。
在质量训练前8张训练图覆盖五类，检查新grad路径与原推理像素一致、反传有限；失败停止。

固定每类128训练来源各1条，共640条/634唯一来源；每类8开发来源，共40条/37唯一来源，
训练与开发按来源/GT分离。Qwen/图像特征只读同一LQ裁图。400/408评价固定40图，
报告已知类别单选对照与实际router的每类指标、选对情况和退步。单选对照不是部署方法；
新Rain8图不冒充原Rain100门槛，8步不宣称科学成功或VLM因果贡献。
本轮无官方像素；官方SoW600重复评测/调参授权仍有效，不另加“整套冻结才准测”的限制。
官方GT不得入训练/loss/梯度/逐图选路；其它封存集不开放，512开发不能冒充官方原分辨率pyiqa。

4节点×每节点2H100/H200=8GPU，单节点同型号>=75GiB；完整反传CUDA显存/速度待验收。
每完整步存router/Adam/8rank RNG/slot/hash，0..408共409份，约12GiB，首启需20GiB空闲。
同run+新共同launch恢复，不支持换world_size；不得用资格权重seed formal，不自动加预算/正式训练。
每节点cap3h、总24使用GPU-hours加清理；不是实测ETA，不能沿用上轮纯推理6分38秒估算。
当前无本轮GPU结果；主代理实现测试push，目的地CPU代理同步预检后给当前IP四节点命令。
唯一GPU根仍为/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang；
旧HDD3只读资产来源，不能在那里执行GPU。保持所有历史结果及无关脏文件。

新入口smoke/start_foundir_frozen_expert_router_v1_tmux.sh：
node_rank/current_node0_address/run_id/fresh_shared_launch_id四参数，
run=qualification_4x2_v1，默认port29703。先目的地CPU preflight，不在旧目录试启动。
终态FROZEN_ROUTER_RESULT_VERIFIED和四节点status0。轻量核对，不全扫409份历史tensor。
详见reports/foundir_frozen_expert_router_readiness_2026-11-04.{md,json}。

## 当前授权：五类已有 FLUX 专家接入；官方SoW重复评测权限澄清 — 2026-11-03

Rain step800已经正常完成：原100开发图34.30118094/.06186744，同一checkpoint过双门槛；
不是待续训任务。用户现授权接入新Rain、保留四类原专家、冻结专家训练router的分阶段序列。
当前先交付五库接入资格；覆盖下方旧“不得转MoE”的阶段限制，不启动其它energy任务。

已有四类原BF16 LoRA＋新Rain FP32 step800，全部冻结，共享原FLUX/VAE；
不再使用vanilla分支替代已训练权重，无CNN/Restormer/新RGB网络。
主专家argmax决定原prompt/alpha/量化及读出：Rain按原50步时间表第26次端点输出；
其它四类保留完整50次。内部混合在0..25次允许五类，26..49次Rain权重置0、其余归一化。
这是待验证兼容规则，不宣称五库soft已有效；不是平均五张输出，也不统一改26或50步。
推理计数器禁止用于autograd训练；未来router必须显式传call-index/route。

本轮116输入：四类各2张训练＋Rain8张训练＋原Rain100开发；原入口/one-hot全量，
仅16训练图加.9主专家/其它各.025手工soft，共248输出。开发只复现Rain历史成绩，不用GT选路。
单选decoded≤1e-6且uint8一致；报告历史差异，Rain100还需复现历史像素及34.2/.08。
soft只描述执行/指标/逐图退步，不当已学会router或VLM贡献。未训练、未构建优化器。
各图原子落盘可恢复；旧绑定源码、所有模型/缓存/结果保留，无自动正式训练或扩预算。

4节点×每节点2张H100/H200，总8卡，单节点同型号>=75GiB；共享已验证运行环境。
新五库采用Rain的Hopper显存策略；旧四库137.73GiB峰值不冒称H100已过，新五库CUDA待测。
粗估15–35分钟/2–4.67使用GPU-hours，硬cap1小时/节点、总8GPU-hours加清理。
入口smoke/start_foundir_five_expert_integration_v1_tmux.sh，rank/当前IP/run/fresh launch四参数，
run=rain800_bank_4x2_v1，port29701。主代理实现测试push，目的地预检后给四节点命令；
主代理看不到新GPU挂载，不冒称已启动。终态FIVE_EXPERT_RESULT_VERIFIED和四节点status0。

已找到2026-09-21明确授权：五类官方SoW600允许反复测、重新训练后再测与调参。
用户本轮重新确认，当前MoE亦按此约定；不要继续说“必须等整套MoE冻结后才准测官方”。
仅该600范围；官方图/GT不入训练、loss/梯度或GT逐图路由，其他封存集不开放。
保留每轮结果和修改，反复用过的官方benchmark不叫独立盲测。
当前接入检查仍不读官方像素；之后官方评测不是必须先五类开发过线才可做。
官方须原分辨率/匹配pyiqa协议，不能拿本轮512/torchmetrics成绩冒充官方。
详见reports/foundir_five_expert_integration_readiness_2026-11-03.md及
reports/advisor_governance/foundir_official_sow600_moe_clarification_2026-11-03.json。

## 当前授权：Rain同配方step600→800，新增200步 — 2026-11-03

父on-policy任务已正常结束：step600开发33.99953558/.06457076，PSNR还差.20046442dB，
LPIPS过线；四节点status0，1小时12分32秒/9.64777778GPU-hours。它不再是待运行任务。
用户已授权保持配方真续训200步；本节覆盖下方旧step400起点/等待GPU的状态。

恢复on-policy父step600完整LoRA/Adam/8rank RNG/600条历史/采样位置19200，
不是旧native50的step600或2000，不fresh Adam。新增601..800，共6400次呈现，
槽位19200..25599，复用已有缓存，不重新生成teacher。不改变lr1e-5/global32、
MSE×10/LPIPS×.05、当前FLUX前25次无梯度＋第26次成图反传、512/CFG1/ref10/alpha1/
单uint8；仅FLUX和原生VAE恢复，无CNN/Restormer、新机制或其它路线。

700/800各评价固定32训练图和原100开发图，分开均值；训练探针仍12800..12831，
开发64032..64131，不进新训练。700重算同环境冻结600对照，800复用；保留历史起点。
同一checkpoint PSNR>=34.2且LPIPS<=.08才过，不拼接不同步；最终800和700各自结果
都保留，不自动加步或转MoE。官方/封存/最终测试禁止，开发不是新盲测。

仍4节点×每节点2H100/H200=8卡，>=75GiB；单节点同型号，软件/指标环境一致。
训练核心已8H100实测，新的续训入口/真实CUDA恢复未验收；混卡/H200不冒称已验证。
预计75–90分钟/10–12GPU-hours，首700约35–45分钟，按本轮实测更新；cap每节点3小时，
24GPU-hours加清理，不自动扩预算。首次100GiB空闲，600..800每完整步保存，共201份。
旧文件不改；同run+fresh共同launch恢复，700评价中断先补完，不支持换world_size严格恢复。
终态轻量查关键状态/导出/均值/退出，不全扫历史tensor，不重算LPIPS。

新入口smoke/start_foundir_rain_onpolicy_continue_v1_tmux.sh，
run_id=continue600_to800_4x2_v1，首次launch_id=launch_continue600_to800_v1，端口29698。
主代理实现/CPU测试/push，目的地预检后给当前IP四节点命令；主代理无GPU挂载，未启动。
目的地首次601..605、700、800及时推小报告，不补造历史发布时间，不上传实时日志/资产。
终态RAIN_ONPOLICY_CONTINUE_RESULT_VERIFIED＋四节点status0；结果写result.json。
详见reports/foundir_rain_onpolicy_continue_readiness_2026-11-03.md。

## 当前授权：Rain实际生成状态训练，step400起新增200步 — 2026-11-03

用户已授权执行新方案；本节覆盖下方“固定中途评价待执行/不训练”的旧状态。
100开发图固定prefix26已正常完成：33.129105/.07536413，相比同权重full50提高
1.448504dB；LPIPS过线，PSNR仍差1.070895dB。不是科学门槛已通过。

当前任务独立恢复step400 LoRA、Adam、8rank RNG、采样位置12800，新增200步到
全局600。每张图用当前FLUX跑原50步时间表前25次，无梯度；第26次预测x_t−t*v
与原生VAE解码参与反传。当前模型前缀每次重生成，不复用旧状态；截断前缀导数，
不宣称全26次端到端训练。只用GT RGB MSE×10和LPIPS×.05；移除旧flow/latent/
teacher项，lr1e-5/global32不变。是恢复旧状态后改变目标，不冒充原配方连续训练。
不加CNN/Restormer/energy，不新生成教师，不动其它专家或另一代理任务，不自动MoE。

复用旧槽位12800..19199的6400份缓存；仅LQ/GT进入新流程，不重新扫全池生成缓存。
新0/50/100/200步检查本轮实际前32训练呈现，明确training_fit_only；训练前32图
新路径与既有prefix26须逐像素相同。仅最终200步评价原100开发图，和step400
固定prefix26匹配，报告均值及逐图退步；不用GT选图，不扫读出位置。
同一输出PSNR>=34.2且LPIPS<=.08才过；开发已使用，不是盲测。
官方/封存/最终测试像素禁止。新检查点目录独立，历史绑定文件不修改。

仍4节点×每节点2H100/H200=8卡，单节点同型号、>=75GiB，跨节点混用仅代码支持。
完整CFG双分支末次反传的CUDA显存/速度待测，不把CPU测试当GPU通过。
原Hopper激活CPU转存/VAE分块/原生checkpoint保留。粗估45–120分钟/6–16GPU-hours，
每节点cap3小时，合计24GPU-hours加清理；按首5步（含保存开销）更新，不自动扩预算。
每步保存完整状态，新400..600共201份，首次需100GiB空闲，不删旧资产。
同run+fresh共同launch恢复；换world_size不支持严格恢复。终态只轻量检查关键状态、
已提交指标和退出码，不全扫历史tensor，不重跑LPIPS。

新入口smoke/start_foundir_rain_onpolicy_midpoint_v1_tmux.sh；
run_id=step400_onpolicy200_4x2_v1，首次launch_id=launch_onpolicy_midpoint_v1，端口29696。
主代理实现/CPU测试/push；目的地安全同步、preflight后给当前IP四节点启动命令。
主代理看不到新GPU挂载，不声称已启动。首5步/50/100/终态推小报告，不推权重/实时日志。
终态需RAIN_ONPOLICY_RESULT_VERIFIED及四节点status0；result.json与verified引用一致。
详见reports/foundir_rain_onpolicy_midpoint_readiness_2026-11-03.md。

## 当前授权：固定中途终点预测，原100开发图确认 — 2026-11-03

用户授权执行：固定step400、call index25（第26次调用），用50步原时间表的前缀，
输出x_t−t*v经原生VAE解码的清晰终点。真正26次停止，不是重新安排26步时间表。
只比较这个固定候选与同权重完整50步及已缓存一步教师，不再扫位置、权重或阈值。
原32训练来源诊断已正常完成，64/64逐像素复现；本轮允许原100开发图，仅用于评分。
不读官方/封存/最终测试，不训练、不改旧绑定文件，不自动MoE/energy或继续长训。
100图是已使用过的开发集，step400此前按开发表现选择；不是新盲测，不能包装为独立测试。
固定候选同一均值PSNR>=34.2且LPIPS<=.08；报告逐图退步，不拼接最好值，不用GT选输出。
额外检查真prefix26与full50同位置读出一致、重跑full50与历史图一致；不一致不冒称匹配过门槛。
仍4节点×2H100/H200=8卡，原runtime/裁图/seed/CFG1/ref10/alpha1/单uint8保持。
入口smoke/start_foundir_rain_midpoint_development_v1_tmux.sh；
run_id=step400_call25_dev100_v1，首次launch_id=launch_midpoint_development_v1，端口29694。
每图保存，同run/fresh共享launch恢复；锁与空闲检查保留，不杀旧进程。粗估6–15分钟，
cap30分钟/4使用GPU-hours加清理，>=10GiB空闲。执行成功不等于质量通过。
主代理实现/CPU测试/push；目的地preflight并给用户四节点当前IP命令，轻量终态简报推GitHub。
详见reports/foundir_rain_midpoint_development_readiness_2026-11-03.md。

## 当前授权：Rain训练式预测与真实50步诊断 — 2026-11-03

用户授权诊断，不是继续扩大训练。原200→2000训练按原授权收尾，不停进程、不修改绑定代码；
完成且四节点卡空闲后，手动运行独立诊断。固定当前Hopper运行的step400与1500，
原32训练诊断图，比较GT构造路径一次预测、真实生成状态一次预测及最终50步。
GT路径含答案，不是部署的一步模型；这些训练来源裁图也不能直接叫过拟合训练图。
报告实际source/record/crop曝光和GT-VAE误差、CFG/精度控制，不凭单项差异断言根因。
不读100开发图/官方/封存像素，不训练、不换模型、不自动MoE/energy或进一步长训。
原始32图指标父报告已确认H100训练到1500；下方“所有H100尚未实测”是旧状态，
但本新诊断自身仍待CUDA执行。4节点×2H100/H200总8卡，保留软件/评分一致约束。
入口smoke/start_foundir_rain_prediction_gap_v1_tmux.sh；run_id=trainprobe32_steps400_1500_v1，
首次launch_id=launch_prediction_gap_v1，当前node0 IP由目的地确认，端口29692。
每图保存；重启同run/fresh共同launch；与训练共用节点锁，不杀旧任务。粗估10–30分钟，
硬cap1小时/8使用GPU-hours加清理，>=10GiB空闲；不自动扩预算。
主代理只实现/CPU测试/push，目的地preflight后给用户四节点命令，结束轻量报告推GitHub。
静态发现旧续训末尾写light_result.json但verified查result.json；若仅此收尾错误，
分别报告已完成步骤/评价与退出失败，不重训、不补造status0、不热改绑定文件。
详见reports/foundir_rain_prediction_gap_readiness_2026-11-03.md。

## H100/H200显存适配与跨节点混用 — 2026-11-03

用户要求当前Rain续训支持H100。当前续训入口已加入H100 80GB/H200代码路径：
4节点×每节点2卡；每节点两卡同型号、每卡至少75GiB，节点之间可混H100与H200。
本节覆盖当前续训入口原H200-only/139GiB限制，不扩称所有历史脚本支持H100。

受限显存路径统一使用VAE全key/value的query256分块、保存的非叶中间激活CPU转存、
cuDNN卷积工作区256MiB及plan cache限制。保持512/global32/micro1/累积4、loss、lr和50步不变。
工作区设置不是总显存上限；H100显存、CUDA反传、混卡NCCL及速度均尚未GPU验收，不能声称已经跑通。
CPU转存增加主机内存/PCIe成本，不沿用3–6小时作为新实测保证，不自动扩大8小时预算。

允许从已绑定父H200 step200迁移模型/Adam/RNG/采样进度到H100/H200，但不承诺跨GPU逐比特等价。
同新run恢复仍要求逐节点原始runtime完全一致；真实驱动release和全部软件/指标版本必须一致。
VAE分块可能带来浮点舍入差异；原父缓存保持只读并记录producer，新增缓存用当前runtime生成。
旧receipt/源码绑定不覆盖；兼容版首次使用新run_id=continue2000_hopper_4x2_v1、
launch_id=launch_continue2000_hopper_v1，从父step200开始，不复用可能存在的旧续训目录。
如果旧续训已推进超过200，不自动丢弃它的进度或停止进程；先报告，再明确迁移方案。
始终中文回复；里程碑/终态推简报，只做轻量检查，不追加全量CPU审计，不自动启动GPU。
详见reports/foundir_rain_h100_h200_compatibility_2026-11-03.md。


## 当前授权：原生50步Rain从step200真续训，最多到2000 — 2026-11-03

用户已明确将上限从1000改为2000并要求continue；本节覆盖下方旧200步“待启动”状态。
不重新做GPU资格、不重训step0、不动另一代理energy或原四个专家，不自动启动MoE。
父native50_stage1_4x2_v1四节点status0、200步/201快照/528评价和CPU核验已完成；
开发31.084100/.083605，尚未通过34.2/.08。本路线累计11.972222222 GPU-hours保留。

新任务是恢复该step200的LoRA、Adam完整状态、8rank RNG、历史和next_global_slot=6400。
旧文件不改；新独立目录从200号快照开始，用新receipt明确记录父身份，不冒称fresh训练。
父step200状态SHA 8fdd6a0d5c6682ab616ba65580477329f3155aecab9da94ec6f1f943720cedcb；
父receipt SHA ac284c0ed2ffe1d7291c7c4462f31632187036f1885d29d221f41c9333be61c2。
新的第一步为201，采样位置6400..6431；不是将原6400曝光重复九遍。
总上限2000，即最多新增1800步/57600曝光、累计64000；随机来源采样，不称完整1.67轮。

数据池38334记录/5780来源、source-uniform采样、512/D4、global32、lr1e-5、五项loss、
BF16 FLUX＋FP32 LoRA/Adam、50步/CFG1/ref10/alpha1/单uint8和逐图seed全保持不变。
仍仅FLUX和原生VAE恢复，冻结一步Rain仅弱监督；不加CNN/Restormer或新机制。
400/700/1000/1500/2000分别评价同32训练诊断＋100开发图，两组不混均值。
旧132图缓存复用，step200指标沿已完成父报告，不再重新跑step0/200或旧GPU资格。
训练图/开发图严格隔离，official/sealed等保护像素不读，开发不是盲测。
同checkpoint必须PSNR>=34.2且LPIPS<=.08；不能拿不同步最优值拼接。
连续两次里程碑PSNR不升且LPIPS不降则保存并暂停人工审查；单次波动不停止。
非有限值/运行错误保留最后完整步并失败退出，不自动改配方/增加步数/恢复质量暂停。

按200→400→700→1000→1500→2000分段，只先生成当前段的新训练缓存；
原一步FLUX生成后释放，再载入学生完整保存状态，避免两个恢复模型同时占显存。
分段准备缓存对随机数的影响在恢复rank RNG时消除；CPU精确恢复测试不等于CUDA逐比特保证。
每完整step存LoRA/Adam/8rank RNG/采样位置，最多200..2000共1801份新快照，旧快照保留。
全部新增缓存逐张落盘可恢复；不重命名旧缓存充新。中断后先补完当前评价再前进。

仍4节点×每节点2H200、总8卡，每卡>=139GiB；不是单节点8卡，H10080GB未资格。
入口smoke/start_foundir_rain_native50_continue_v1_tmux.sh，四参数仍node_rank、当前node0地址、
run_id=continue2000_4x2_v1、首次共用launch_id=launch_continue2000_v1，端口29690。
同run_id＋fresh launch可恢复工程中断；不能用换ID丢进度，也不能绕过质量暂停。
本轮全程粗估3–6普通小时/24–48GPU-hours，未实测；每节点cap8小时，
合计64GPU-hours加清理，不自动扩预算。第一个400步检查粗估30–70分钟，须按实测修正。
新缓存＋逐步快照估计1.3–1.6TiB，首次要求1800GiB可用；恢复按已提交文件估算余量，
只是保守工程估算，不保证其它任务写盘时仍充足。空间不足先报告，不删除历史资产。

结束默认只核对退出状态、已提交指标、最新完整恢复状态/最终导出等关键文件；
不扫描全部历史tensor，不独立重跑LPIPS，不再追加全量CPU核验。
RAIN_NATIVE50_CONTINUE_LIGHT_CHECK_PASS和四节点status0不是质量达标；
light_result.json可能是paused_for_quality_review或完成2000，必须据实际last_step报告。
目的地代理在各里程碑及终态回传简报到GitHub，保留耗时/真实GPU-hours/核验范围；
不上传权重、数据、tensor、实时日志，不自动续训或扩大任务。

已通过11项CPU回归及两进程Gloo小模型step200迁移、201/202精确恢复；不是CUDA验收。
主代理负责实现/CPU测试/push，目的地安全同步并执行preflight --require-parent，
核对所需父step200状态和资产后给用户当前IP的四节点命令。不重做GPU资格。
本地主代理看不到新GPU挂载，不能声称新GPU任务已启动；详见
reports/foundir_rain_native50_continue_readiness_2026-11-03.md。

## 用户沟通与轻量结果检查要求 — 2026-11-03

用户明确要求：始终用中文回复。今后的GPU结束检查默认跳过耗时的全量CPU核验，
只读取退出状态、已提交指标摘要及关键保存文件，注明未做全量核验，不冒称完整验收通过。
本要求覆盖历史默认执行全量CPU核验的流程；除非用户另行要求，不自动追加该核验。
GPU结束后的简短真实报告仍须推送GitHub；不上传权重、tensor或原始日志，不自动续训。


## 当前授权：原生50步Rain起点，匹配训练首200步 — 2026-11-02

用户明确要求先把已有原生50步Rain改好，再考虑内部MoE。本节覆盖下方旧endpoint转flow
stage1的待执行状态；不恢复其训练，也不动另一代理energy、四个原专家或旧结果。
旧stage1已200步/6400曝光/528评价完成，但最终核验超时，没有RESULT_VERIFIED。
开发28.946285/.099400，低于同图一步教师34.226794/.073640；累计8.722777778
GPU-hours（含旧资格）保留，不能把8GPU-hours说成8个普通小时。

新起点固定旧native source_diverse_pilot step2000，SHA
2c4537258783f51ad15163b1a3377ce811d1418d51b4da68b5d75ef502bd45af。
它原100图30.859952/.083477，未达标；不是那个达标的一步Rain。
旧100图与当前100图的record/source/文件哈希/crop一致，旧1580训练来源与它们分离。
但seed与FP32 LoRA计算路径变了，旧指标只作历史参考，必须重算当前step0。

当前任务复制原生50步BF16 LoRA并无损提升到FP32，fresh Adam；不是原optimizer轨迹续训，
也不继承一步模型、旧资格或新200步转换模型的权重。训练目标仍GT为主＋冻结一步FLUX
弱RGB示范，恢复模型只有FLUX/原生VAE；LPIPS仅损失/评分，不加CNN/Restormer恢复器。
沿用上一轮完整38334条/5780来源池、lr1e-5、global32、512随机裁图/D4、全部loss系数、
采样顺序和noise/t种子。首200步6400曝光，5012条/3867来源；不是全池已经遍历。

只改变初始化；step0/50/100/200各评价32训练诊断＋100开发图，分别汇总。
目标仍同一checkpoint开发PSNR>=34.2且LPIPS<=.08；开发只评价，官方/封存像素禁止。
不承诺200步过线；结束后人工决定是否续训，不自动加步、改loss或启动MoE。

明确复用父formal_stage1_4x2_v1的6532份缓存，不复制/重命名冒充新缓存。
新receipt绑定父receipt SHA 8b2209f880307af41eec03c571a8c98cf0661f7714ac8d4f63d63cd96dfb75ae；
逐张核对行身份、裁图、教师、tensor SHA、有限值/形状，评价GT latent仍必须为零。
主代理看不到目的地缓存，CPU源资产通过不等于目的地缓存可用；目的地必须--require-cache。
若缺失或不匹配就回报，不能自动重生成整池/换缓存/扩大预算。

仍4节点×每节点2H200=8卡，>=139GiB/卡，不是4卡/单节点，也未资格H10080GB。
新入口smoke/start_foundir_rain_native50_warmstart_v1_tmux.sh，
参数node_rank、当前node0地址、run_id、fresh共享launch_id；端口29688。
run_id=native50_stage1_4x2_v1，首次launch_id=launch_native50_stage1_v1。
每步保存LoRA/Adam/各rank RNG/采样进度，201份；同ID＋fresh launch恢复本段。
新磁盘约75–85GiB，至少100GiB空闲，父缓存必须保留；不删旧资产。

GPU阶段粗估25–55分钟/3.33–7.33使用GPU-hours，未实测；仍cap60分钟/8GPU-hours加清理。
GPU输出GPU_COMPLETE且四节点status0只表示训练/评价完成，CPU_VERIFICATION_PENDING。
不要让八卡等待CPU全量核验。目的地CPU代理随后运行
smoke/verify_foundir_rain_native50_warmstart_v1_cpu.sh <run_id> <launch_id>，
CUDA_VISIBLE_DEVICES空，cap90分钟，无新采样/训练；CPU耗时另报，不假定平台自动停止计费。
必须最终有RAIN_NATIVE50_WARMSTART_RESULT_VERIFIED及CPU_VERIFY_END status0才称完整核验通过；
质量门槛仍单独判断。无论成功/失败/超时都回传小报告到GitHub，不上传tensor/实时日志。
主代理实现/CPU测试/push；目的地安全同步、检查资产并给当前IP的四节点命令。
已通过19项CPU回归、两进程DDP下一步精确恢复、真实旧LoRA100张量FP32加载一致性，
及60份源码绑定的源资产预检；目的地缓存/CUDA仍待验收。
本地没有GPU和新根挂载，不能声称已启动。详见
reports/foundir_rain_native50_warmstart_readiness_2026-11-02.md。

## 当前授权：Rain正式数据训练前200步，嵌入早期评价 — 2026-11-02

用户明确取消独立32图过拟合阶段，要求进入下一阶段训练、在50/100/200步检查真实输出。
本节覆盖下面旧25步待重试、单独trainfit计划及其它历史ACTIVE优先级。未发布的trainfit草稿
已移除，没有启动/产物；不恢复energy/旧MoE/router，不把正式训练重命名成32图拟合。

父qualification_4x2_driverfix_v1已四节点status0，25步、800曝光、26快照齐全；
同8训练图：原一步Rain33.696391/.034100，新多步step0为9.972354/.816801，
step25为16.417211/.394619，8/8双指标弱于教师。工程通过不是质量通过，旧失败
加资格累计.721666667 GPU-hours不清零。原模型与全部旧结果保持不动。

当前新入口smoke/start_foundir_rain_flow_formal_stage1_v1_tmux.sh，四参数node_rank、
当前node0地址、run_id、fresh共享launch_id；端口29686。
资源仍4节点×2H200=8卡，每卡>=139GiB，唯一GPU根为迁移后的multimodal-discrete-diffusion。
run_id=formal_stage1_4x2_v1，首次launch_id=launch_formal_stage1_v1；不复用旧IP或旧run。
主代理无法启动目的地GPU；负责实现/CPU测试/push，目的地安全同步/资产预检/给用户命令。

这是正式训练的首200 optimizer steps，不是200步完整训练已足够的承诺；到200停止，
不自动扩大步数。从原成功Rain step4000 FP32 LoRA复制初始化、fresh Adam，
不复用旧资格step25权重；新的正式checkpoint/Adam/RNG可在后续明确授权后继续，
不必再做一次资格或重新初始化。当前同ID/fresh launch恢复本段中断，源代码/world/runtime变更拒绝。

使用完整38334条、5780来源的原Rain训练池，均匀选来源再选记录；
512随机裁图与D4增强，小图reflect padding（池中35条）；GT仍主要监督，
冻结原一步FLUX仅弱RGB示范，不加载Restormer。模型/loss/lr/global batch32不变。
本段固定6400次曝光，5012不同记录、3867来源，不声称已遍历全池。
按绝对position固定采样/增强/noise/t，缓存6532份（6400训练+32训练诊断+100开发评价）。
缓存逐张保存、逐张核验；不把整套缓存复制进每个rank内存。

step0/50/100/200评估同32训练图和原冻结100来源Rain开发图，分别汇总；
开发裁图沿manifest原crop_x/y，单uint8及PSNR/LPIPS算术与原Rain口径一致；
开发图绝不进入反传，GT latent仅训练记录编码，评价GT只评分。开发是可调参开发集，
不是官方/封存测试，也不代表FLUX底座预训练没见过。其他保护像素仍禁止。
目标仍开发PSNR>=34.2且LPIPS<=.08；比较同图原一步教师，不能拿训练均值替代开发成绩。
早期开发两指标都未改善则milestone标review_required并回传检查，不自动改loss或扩大预算；
首200步不要求50步就过门槛。目的地代理在50/100步核验已提交摘要并回传小报告，
终态无论成功/失败/超时都必须回传GitHub，不上传实时日志/tensor/权重。

每完整步原子保存LoRA/Adam/8rank RNG/采样进度，共201份；评价保持训练RNG，
中断后先补完当前milestone再训练，已保存评价跳过。新缓存+快照预计约160GiB量级，
启动前至少220GiB可用空间，保留全部旧资产。额外缓存与开发评价吞吐未实测；
粗估25–50分钟/3.33–6.67使用GPU-hours，硬cap每节点60分钟、合计8GPU-hours加清理，
超时保留进度，不自动续跑/扩预算。前25步训练本体实测均值1.509秒不含保存/评价/新缓存，
不能据此保证新全程只需5分钟。CPU通过不等于新CUDA/NCCL运行已通过。
详见reports/foundir_rain_flow_formal_stage1_readiness_2026-11-02.md。

## Rain资格驱动检查修复 — 2026-11-02

以AGENTS.md顶部新修复为准：旧qualification_4x2_v1训练前失败，0步，无质量结果。
只修正跨节点驱动描述比较，保留实际驱动版本/全部软件评分环境检查和严格断点恢复。
同步后重新CPU预检，新run_id=qualification_4x2_driverfix_v1、launch_id=launch_driverfix_v1，
由目的地代理使用当前node0地址给四节点×两张H200的原25步命令，不覆盖旧收据/运行目录。
本地主代理未启动GPU，不自动formal或其它路线。详见AGENTS.md与
reports/foundir_rain_flow_runtime_guard_fix_2026-11-02.md。

## 必须回传GPU终态简报 — 2026-11-02

用户明确要求：每次GPU任务结束后，目的地代理必须核验日志、结果和退出状态，
将简短的真实报告提交并推送至GitHub main，核验远端提交后告知用户；口头汇报不能替代推送。
成功、失败、超时或中断都应记录，不隐瞒退步；报告包含run/launch ID、完成量、指标、
耗时和GPU-hours、核验边界、日志路径及SHA。不得上传权重、图片、tensor或原始/live日志。
保留本地修改和全部运行资产，不因报告而自动启动、重试或接续训练。


## 当前授权：独立Rain多步适配25步资格，四节点H200 — 2026-11-02

用户在protocol_routed工程通过后明确execute。本节覆盖下方“Rain训练器尚未交付”；
当前交付独立新Rain资格，不训练router、不重跑旧五类或共同协议，不改另一代理energy。
旧protocol_routed_4x2_v1四节点status0，8/8单选像素完全相同、soft可执行；
固定.9混合使六图至少一指标变差，只有Low-light两图均改善，不能当普遍无损或VLM贡献。

新Rain从成功precision_resume continue_v1 step4000的FP32 LoRA复制初始化，fresh Adam；
不是随机重训、不是原优化轨迹续训，也不是把一步残差重复50次。原Rain只读保留，
作为冻结FLUX教师在固定训练裁图上生成一次RGB示范，缓存后释放；不加载Restormer。
四个原50步专家和共享base/VAE全部冻结，不加载为可训练参数，不改变其任何权重文件。

新分支学习native reference noise-to-clean：
z_t=(1-t)*z_GT+t*noise，目标v=noise-z_GT，预测clean=z_t-t*v。
GT主要监督：flow MSE1、latent L1 .25、RGB MSE10、LPIPS .05；
原Rain RGB示范只加L1 .1辅助，不当所有t的速度真值。预测clean减法、LoRA、Adam为FP32，
FLUX计算BF16。真实forward_kv_extract按块重算，VAE decode也重算，无新增修图网络。
LPIPS AlexNet只作为损失/评分；不加CNN energy、Restormer恢复器或额外教师。

只从原38334训练记录中按固定哈希取32个不同来源，512中心裁图、不增强；
完整主manifest非训练source/GT哈希隔离复查。每步32图各一次，micro1×累积4×8rank，
25步共800曝光，只是工程资格。前8张训练图在step0/25跑真实50步，
并同时报告同图冻结一步Rain；是training_fit_only，不是开发、泛化或正式34.2/.08达标。
保留Rain原prompt、CFG1/ref10、最终一次uint8/alpha1，不搜索输出系数。
只看随机t loss下降不能判成功；必须报告真实50步两指标，差就如实报，25步不作路线终判。

4节点×2张H200、共8卡，每卡≥139GiB。仍用唯一迁移后的multimodal-discrete-diffusion根。
入口smoke/start_foundir_rain_flow_adaptation_v1_tmux.sh，四参数node_rank、当前node0地址、
run_id、fresh共享launch_id，默认端口29684。预计10–30分钟/1.33–4 GPU-hours未实测；
每节点cap60分钟、总8 GPU-hours加清理，不自动扩预算。至少20GiB新空间，不删旧资产。
每完整step原子发布LoRA/Adam/各rank RNG/采样位置及SHA；同run_id换fresh launch_id恢复，
未落盘一步可能重算，不支持换world/runtime严格恢复。资格权重不得seed正式。
缓存逐图、probe逐图保存；四节点失败联动。原step4000始终保留。

主代理实现/测试/推送；目的地代理安全同步、CPU资产预检后给四节点命令，不能自行改配方。
36项CPU检查、两进程DDP下一步精确恢复、真实源资产预检及44份源码绑定复核通过。
GPU尚未启动，显存/速度/四节点NCCL训练仍待实测；通过后不自动正式长训、MoE或energy。
详见reports/foundir_rain_flow_adaptation_v1_readiness_2026-11-02.md。

## 当前执行：保留主专家原协议的内部混合验收 — 2026-11-02

用户在共同prompt／alpha=1损害原能力之后授权execute。本节覆盖下方“下一步待制定”；
不重跑共同协议对照，不续训旧五类模型，不直接启动Rain或正式MoE，也不动另一代理energy。

新接口接收显式专家比例：全部比例用于FLUX原50处LoRA增量的内部混合；
比例最大的主专家决定它原来的prompt、alpha和uint8顺序。平局按固定专家顺序。
不平均alpha、不强制共同prompt、不生成多张图再融合；一条50步FLUX轨迹只生成一张图。
Blur/Noise/Low-light保留.75/.30/.925和双uint8，Haze为1与单uint8。
目标是保留单选时原能力，完整GPU仍待验收；主专家切换不连续，soft质量未知，不承诺no-harm。

本轮仅8张父任务training图、每类前两张，固定选取不看指标，每rank一张。
两组为one_hot和主专家.9／另三份各.1/3；全是手工工程比例，不是学会的image/VLM router。
单选必须decoded最大差≤1e-6且uint8完全一致；soft只验有限输出、原协议绑定与可运行性，
照实报告原/单选/soft三组PSNR和LPIPS。各类只有2张，不能当泛化、SoW或VLM贡献证据。
GT只在该图两组输出保存后评分；开发/official/sealed/internal/parent-test等保护像素不读。
恢复模型仅FLUX＋原生VAE；AlexNet只评分，不加CNN/Restormer恢复器或教师。

资源固定4节点×2张H200、总8卡，每卡≥139GiB；继承上轮对应节点软件/硬件环境。
入口smoke/start_foundir_protocol_routed_flux_v1_tmux.sh，参数为node_rank、当前node0地址、
run_id、fresh共用launch_id，默认端口29683。新输出根foundir_protocol_routed_flux_v1；
逐arm保存并校验恢复，四节点联动，结果汇总重新读tensor验证单选，不只信pass字段。
复用父四条prompt缓存，不重新运行Qwen。全部原权重、源码、缓存和结果不修改。
预计5–15分钟/.67–2 GPU-hours，尚未实测；cap每节点30分钟、合计4 GPU-hours加清理。
本地61项CPU回归、源资产预检及47份源码绑定复核通过；GPU未启动。
实际显存与完整soft采样仍待目的地验收，本地主代理看不到目的地父缓存。

主代理focused push；目的地代理安全同步、CPU --require-parent预检后给四节点命令，
保留脏文件和运行资产，不复用旧IP、不自动启动。通过后再依据soft结果准备Rain25步资格，
Rain适配训练器尚未交付，工程通过不自动授权长训。
详见reports/foundir_protocol_routed_flux_v1_readiness_2026-11-02.md。

## 当前终态：共同prompt／alpha=1未保住四类能力 — 2026-11-02

common_protocol_4x2_v1四节点H200 status0，128条×4组评分与最终核验齐全。
只换共同prompt影响较小但仍无四类平均双指标全不退步；统一alpha=1损害明显。
仅换输出Noise/Low-light/Blur PSNR为-4.983307/-.909835/-.825348dB；
两者都换为-4.795476/-1.001667/-.894802dB。不能用部分LPIPS收益遮住PSNR损失。
Haze逐图输出处理控制通过。协议未选定，不在当前共同方案上直接启动Rain训练。
node0墙钟386秒，四节点.844444 GPU-hours；峰值137.738088GiB，均H200。
256份原/新tensor SHA及汇总/受损数独立复算通过，未独立重跑LPIPS。
只读同128条训练GT评分，training_fit_only，不是泛化/SoW或router贡献。
下一步先制定输出协议兼容方案，保留原分母；不自动搜索、训练或修改energy。
小报告：reports/foundir_existing_flux_common_protocol_v1_terminal_2026-11-02.{md,json}。

## 当前执行：共同提示词／输出协议对照，四节点H200 — 2026-11-02

用户在128/128静态继承通过后明确授权下一步，并确认四节点全部是H200。
当前只执行方案B：固定共同提示词与输出处理的训练侧2×2对照；不再为H100兼容阻塞，
本入口明确要求4节点×每节点2张H200（每卡≥139GiB），总8卡。不动已完成重放及另一代理energy。
旧“没有下一任务授权”在本节被覆盖；这不是自动正式MoE或Rain训练授权。

复用 `replay_4x2_v1` 原128张已保存输出；相同权重/50步/512/crop/seed，只新增共同提示词采样。
四组：原提示词＋原输出；共同提示词＋原输出；原提示词＋统一输出；共同提示词＋统一输出。
统一输出候选仅alpha=1、一次uint8，不再混LQ；这是待检验的最简单候选，不是已经决定抛弃
原任务alpha，也不预设所有类型不退步。不做alpha搜索、逐图GT选方案、soft混合或router训练。

为比较质量，本轮新增读取这128条相同training记录的配对GT，仅算PSNR/LPIPS；元数据逐项核对，
同crop、GT哈希和路径严格绑定，所有开发/official/sealed/internal/parent-test及最终1859像素仍不读。
这些图已被原FLUX微调见过，结果只描述训练侧接口影响，不是泛化/SoW/论文成功。
恢复模型仍只有FLUX＋原生VAE；LPIPS AlexNet只评分，不是新增CNN恢复器/教师/energy。

入口 `smoke/start_foundir_existing_flux_common_protocol_v1_tmux.sh`，四必填参数仍为
node_rank、当前node0地址、run_id、fresh共用launch_id。默认端口29682；新输出根完全隔离。
原输出复用要求各node保持父任务对应runtime；新节点/环境不匹配则拒绝，不偷偷换弱基线。
逐图保存、失败联动、四节点done后汇总；结果完成不自动选共同协议，不自动启动Rain或长训。
预计15–30分钟／2–4GPU-hours（新任务未实测）；每节点硬cap1小时，总8GPU-hours加清理。
主代理推送代码；目的地代理安全同步、检查共享父缓存和LPIPS依赖后给四节点命令，不冒称本地已启动。
详见 `reports/foundir_existing_flux_common_protocol_v1_readiness_2026-11-02.md`。

## 当前终态：四节点H200静态单选重放128/128通过 — 2026-11-02

replay_4x2_v1 / launch_3d22527_v1四节点status0；四份done及verified齐全。
四类各32张原入口与bank完整50步对照，decoded最大差0、uint8全部一致。
目的地CPU独立重读256tensor及SHA并重算128对，全部通过；32源码/26energy哈希未变。
各节点592/588/584/584秒，node0墙钟9分52秒，总1.304444 allocated GPU-hours。
实测均为H200，峰值allocated137.734817GiB；H10080GB未通过，不能从旧≥75GiB门槛扩称兼容。
只证明静态单选继承；无GT/新质量指标、无soft/router或Rain训练验收，不自动下一任务。
下一步共同prompt/输出处理及Rain多步适配仍待交付；全部资产保留。
小结果/身份/成本：reports/foundir_existing_flux_replay_4node_terminal_2026-11-02.{md,json}。

## 最新资源纠正：四节点×每节点两卡，总八卡 — 2026-11-02

用户明确纠正：当前已有FLUX专家继承/MoE路线必须使用 **4 nodes × 2 GPUs = 8 GPUs**，
不是单节点八卡，也不是四节点各八卡。此节覆盖下方单节点4/8卡文字及旧迁移默认；
`dbd8b29` 的单节点交付是配置错误，没有启动GPU，不应继续使用它的命令。
另一代理energy工作不改，不借此重启旧任务。

已改为四节点torchrun，每节点只见CUDA 0,1；node_rank=0/1/2/3，global_rank=node_rank*2+local_rank。
每节点两卡同型号H100或H200、每卡≥75GiB；节点之间可以不同系列，但同图原入口/bank
必须在同一rank/GPU系列和绑定环境比较。四节点必须共享新根和输出目录，使用当前node0地址，
绝不复用历史IP。旧资产审计config中的single_node字段是历史策略，不改其哈希冒充新审计；
当前执行拓扑以 `configs/foundir_existing_flux_replay_v1.json` 为准。

本轮仍是128训练LQ、四类各32、零训练静态单选重放；科学口径、权重与像素权限不变。
每节点保存独立runtime/日志/成本，node0仅在四节点完成后汇总全部128图；没有四节点完成标记
不能报告成功。每次启动四节点共用同一fresh launch ID，恢复保留run ID但换launch ID，
核验逐图存档后跳过完成项；拒绝旧单节点receipt。预算每节点2h，总计最多16 GPU-hours加清理；
成本按四份每节点2×耗时相加，不把每个节点误算八卡。CUDA/实际跨节点网络尚待验收。
主代理推送后只给同步提示；目的地代理保留脏文件安全同步、CPU检查，取得当前node0地址后
给四条分节点命令。详见 `reports/foundir_existing_flux_replay_4node_readiness_2026-11-02.md`。

## 2026-11-02最新：零训练完整50步专家单选重放

目的地资产已通过；当前入口 smoke/start_foundir_existing_flux_replay_v1_tmux.sh，
不是旧retention/Rain/energy。四类各32训练LQ，原单LoRA对四分支bank固定单选，
不读GT或最终开发像素，不训练，逐图保存并独立复算。默认单节点8卡，兼容4卡H100/H200。
尚无GPU结果；目的地同步/CPU预检后再给用户命令。完整说明见
reports/foundir_existing_flux_replay_v1_readiness_2026-11-02.md及AGENTS.md顶部。
Rain适配和动态VLM路由未交付，不从下方旧历史指令启动其他任务。

## 2026-11-02：已有LoRA复用是当前MoE方向

用户已批准保留Blur/Noise/Haze/Low-light原多步FLUX，单独适配Rain，再内部VLM残差路由。
旧五类retention正式1000步不启动；另一代理energy不动。右图未规定FFN或采样步数。
已交付CPU资产审计/LoRA混合算子/静态单选安装器；不是完整GPU重放或Rain训练器。
不要恢复下方历史优先级。原成功一步Rain只读保留，不把其直接当flow速度专家。
最新方案/边界见AGENTS.md顶部和reports/foundir_existing_flux_experts_plan_2026-11-02.md。
唯一GPU根为新的multimodal-discrete-diffusion目录；旧根仅只读。默认单节点8卡、
兼容4卡同型号H100/H200，不沿用旧跨节点命令。下一步目的地安全同步/CPU审计回传，
无新GPU命令、不自动formal、不读取保护或最终1859 Noise像素。

## 当前：retention资格通过，旧任务保持；正式入口待交付 — 2026-11-01

四节点qualification_4x2_v1均status0，25步batch40、每类200次、五收据齐全。
Rain/Blur/Noise PSNR分别+.081055/+.020909/+1.306323dB，LPIPS均改善；
Low-light+.049010dB且LPIPS改善，Haze-.010419dB但LPIPS改善，新增任务适应仍有限。
未复现旧配方数dB遗忘；不等于完整五类门槛或VLM贡献证明。
总墙钟863秒，四节点1.912222 GPU-hours；最终模型/Adam SHA已核验。
下一计划独立正式1000步batch40，从原成功step2000＋fresh Adam开始，资格不得seed正式。
目前正式入口仍禁用，需确认预算并交付新正式配置/launcher，不手改资格步数。
未启动新GPU、不自动mixed；全部资产和energy保留。
完整逐图/逐步/原始日志：reports/foundir_token_moe_retention_v1_terminal_2026-11-01.{md,json}。

## 当前执行：五类MoE保护训练修订，先25步验收 — 2026-11-01

本节覆盖下方“只读诊断／尚未改配方／旧五类25步入口”的MoE执行指令。
用户已授权从成功VLM step2000修订训练；旧诊断完成，不重复，不续训退步step2025。
当前只有新25步资格入口，GPU未启动；主代理实现/测试/push，目的地CPU代理同步、
预检后给用户四节点命令，不能自动formal/mixed或触碰另一代理energy工作。

保持五块、每块两个FLUX FFN专家＋一套共享LoRA＋VLM加性router，不加CNN/Restormer。
加载step2000全部权重，包括既有VLM；fresh Adam，新阶段0→25，不冒称原优化轨迹续训。
每步五类各8张，batch40=8rank×5累积；LoRA/专家LR降到1e-6，router/VLM仍1e-4。
旧模型只对相同训练裁图生成六项误差标量。当前损失除以旧加权总误差，并分别惩罚
MSE/LPIPS超过旧误差；不是旧router蒸馏，不增加第二个常驻模型，不把GT输入router。
这是假设驱动的修复，尚未证明改善，不能承诺开发集绝不退步或宣称VLM独立贡献。

25步每类200次曝光；阶段0/5/25沿用各100张预览。每5步保存新batch40完整
权重/Adam/8rank Python、NumPy、PyTorch、CUDA随机状态及采样位置；旧batch32收据拒绝。
资格不seed正式。下一计划正式1000步、40k曝光、每类8k，须看资格后确认预算和入口，
不把25步当最终能力判决，也不照失败配方自动长训。

训练与开发按来源及相同GT哈希隔离；路径在MillionIRData/Train不代表开发图参与训练。
参照只读训练GT；开发GT只评分。official/sealed/internal/parent-test像素不进入本次工作。
官方测试留到方案/超参/checkpoint冻结后，开发达标不等于官方达标或SOTA。

资源4节点×2张H100/H200，共8卡，节点内同型号；GPU共享根仍
/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。
入口 smoke/run_foundir_token_moe_retention_v1_4x2.sh（不要用旧five_single入口），
配置 configs/foundir_token_moe_retention_qualification_v1.json。
资格暂估.3–.8小时/2.4–6.4 GPU-hours，单次1.5小时cap、80GiB空闲；未实测不保证。
完整配方、理由、数据边界、CPU验收及同步/恢复说明：
reports/foundir_token_moe_retention_v1_readiness_2026-11-01.md。

## 当前：五类干扰诊断完成，必须处理body变化 — 2026-11-01

四节点diagnosis_4x2_v1均status0，零训练更新，1000次自重放最大差0。
固定旧比例后Rain/Blur仍下降7.173306/3.583147dB，单改router不足。
body仍包含共享LoRA与专家，未隔离唯一根因或训练梯度冲突；不能直接归因VLM。
Noise交互明显，不按简单百分比分摊。node0墙钟239秒，四节点.525556 GPU-hours。
下一步应从成功step2000修订旧任务保持和body更新约束，先冻结新配方/规模/预算/入口。
不续训退步2025，不用旧两类formal或25步入口替代五类正式训练；本次未改配方/启动GPU。
完整四组逐图及日志：reports/foundir_token_moe_interference_v1_terminal_2026-11-01.{md,json}。

## 当前授权：定位五类退步，只做零训练交叉检查 — 2026-11-01

用户授权检查上一轮退步原因，不是照原配方正式长训。五类25步已结束，不重跑、
不续训2025、不启动mixed；成功step2000及全部历史资产保留。
本地回传CPU审计已完成：Rain/Blur/Noise分别99%/96%/74%的预览图PSNR退步；
Rain后四层仍几乎全选原专家0，不能仅归为“选错专家”。Low-light起点平均像素MSE
约为Rain的109.52倍，这是开发误差尺度，不是训练梯度主导的证明。唯一根因尚未确定。

下一步是同一500张开发预览的四组只读干预：旧/新修图权重 × 旧/新每图每块专家比例。
先从两个原模型记录实际soft比例，再固定比例重放，避免仅换router参数但上游特征仍变。
两模型各500张自重放必须浮点最大差≤1e-5且uint8一致。GT只用于评分；
跨组合是诊断，不是可部署提升，不能用其最佳指标宣称科学成功或VLM贡献。
不创建optimizer、不反传、不写权重，不加CNN/Restormer/专家，不做image-only消融。

新入口 smoke/run_foundir_token_moe_five_single_interference_v1_4x2.sh；
配置 configs/foundir_token_moe_five_single_interference_v1.json。
生产MoE/资格训练器和另一代理energy源码保持不变。GPU诊断尚未启动；
主代理实现/测试/push，目的地代理安全同步、CPU资产预检后给用户GPU命令。
资源仍4节点×每节点2张H100/H200；迁移后multimodal-discrete-diffusion共享根不变。
预算.2–.5小时/1.6–4.0 GPU-hours，每次1小时cap，至少5GiB新空间，不删旧文件。
完成后依据四组结果决定优先处理路由漂移还是共享权重干扰，不自动更改训练配方。
详见 reports/foundir_token_moe_interference_v1_readiness_2026-11-01.md；
已执行CPU证据在 reports/foundir_token_moe_interference_cpu_audit_2026-11-01.json。

## 当前：五类single 25步工程验收结束；旧任务明显退步 — 2026-11-01

qualification_4x2_v1四节点8×H200，全局2000→2025、新增25步，四节点status0。
每类160次曝光、五份完整收据、两次各500张固定预览；工程PASS。
Rain PSNR -7.253410dB / LPIPS +.025153；Blur预览 -3.654619 / +.051980；
Noise预览 -1.540234 / +.022325。Haze预览 +2.702537 / -.102344，
Low-light预览 +8.642499 / -.258797。旧任务明显退步，不以新任务改善掩盖。
这不是完整五类门槛通过或VLM独立贡献；25步不能作路线终判。
node0墙钟747秒，四节点合计1.652222 GPU-hours；最终模型/Adam SHA核验通过。
终态不自动formal/mixed/加步，不改配方；资格权重不得seed formal，保留成功父step2000。
完整日志与逐图核验：reports/foundir_token_moe_five_single_v1_terminal_2026-11-01.{md,json}。

## 当前授权：扩展五类single，再接入现有混合退化 — 2026-11-01

本节覆盖下方“仅限Rain/Blur”“下一步续训到2000”及更早Rain/EBT优先级。
用户已批准扩展右图③：五类接入验收 → 五类适应训练 → 五类single＋现有五种double、
一种triple联合训练。暂不做image-only消融，不另训五个独立专家，不加CNN/Restormer/energy。
无需五类全部先过SOW才允许混合阶段；先确认新增任务能学习且旧任务没有严重退化。
另一代理的FLUX-native energy显存线独立，不修改其代码或停止其任务。

本次交付只到五类25步工程验收，CPU已通过，GPU尚未启动；正式规模在实测后冻结，
不自动formal或mixed。从已成功的VLM step2000恢复权重、Adam和8rank RNG到2025，
不是重置VLM/fresh Adam，不从旧step11000重新训。五类新采样＋取消原两类router CE
是明确的阶段变化，不能冒称原两类配方原样续训。仍每块两个FLUX FFN专家，
不是五类各一个专家；路由由恢复损失学习，不输入GT类别。
Rain/Blur LPIPS权重.05/.20不变，新增三类.05；其余loss/LR、一步端点和batch32不变。

可用五类训练池311,921条；本次每类160次曝光、各100张固定开发预览。
Noise/Low-light预览仅34/21个来源，不是各100独立来源。预览不能替代完整五类门槛。
每5步保存完整状态；本次资格权重不能seed正式训练，formal仍从成功step2000出发。
step2000历史完整结果Rain34.610111/.064487、Blur26.755594/.176108保留；
均过最低门槛，但未证明VLM独立贡献，Blur LPIPS仍不如原独立50步专家.151993。

资源仍4节点×每节点2张H100/H200，共8卡；节点内同型号，跨节点可混用。
GPU共享运行根仍为 /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。
不要复用旧IP、不要在旧agent-driven-world-model目录启动。
主代理在隔离worktree实现/测试/push；目的地CPU代理安全同步、资产预检后给用户GPU命令。
保留本地dirty、全部旧资产和energy工作；GitHub是代码/决策交付源，不含大权重/数据。
预算.3–.8小时/2.4–6.4 GPU-hours，每次启动1小时硬cap，至少80GiB空闲，不自动删文件。
入口 smoke/run_foundir_token_moe_five_single_v1_4x2.sh；
配置 configs/foundir_token_moe_five_single_qualification_v1.json。
精确同步、恢复、数据与测试说明：
reports/foundir_token_moe_five_single_v1_readiness_2026-11-01.md。

## 当前：VLM-token MoE续训2000步完成；完整开发最低双任务门槛通过 — 2026-11-01

continue_4x2_v1真续训从父step1000恢复，四节点8×H200，1000个新步到全局2000，
四节点status0；1000/1500/2000完整Rain100＋Blur2595及10份收据齐全。
终态Rain34.610111/.064487、Blur26.755594/.176108，均过预设绝对门槛。
相对同运行step1000，Rain PSNR+.336425dB、LPIPS变差.002326；
Blur PSNR+.050452dB、LPIPS改善.007321。原step0基线另行保留并核验。
最低联合门槛通过不等于VLM独立贡献或独立来源泛化；无匹配消融，
历史独立50步Blur LPIPS=.151993仍优于本次.176108。
node0墙钟10597秒，合计23.541111使用GPU-hours。终态已停，不自动加步。
完整日志与核验：reports/foundir_token_residual_moe_continue_v1_terminal_2026-11-01.{md,json}。

## 当前授权：优先改善VLM-token性能，真续训1000→2000 — 2026-11-01

用户选择暂不做image-only匹配对照，继续当前右图③。不得擅自启动对照、改架构、
加CNN/Restormer或触碰另一代理的energy显存工作。本次主代理准备代码/测试/push，
目的地CPU代理安全同步并预检后给用户GPU命令；这里尚未启动GPU。

下一步固定从formal_4x2_v1 step1000恢复权重、Adam、八rank RNG和采样进度，
再训练1000步到总step2000；不是fresh Adam、不是从原step11000或25步资格重新训练。
原数据、损失、学习率、batch32和VLM加性router不变。资源仍4节点×2张H100/H200；
节点内同型号、节点间可混用。共享根仍为multimodal-discrete-diffusion的新目录。
续训起点1000以及1500/2000均评估完整Rain100＋Blur2595，每100步完整保存。
Rain≥34.2/≤.08，Blur≥25.9/≤.18仍是最低质量门槛；续训不能保证通过。
终态同时报告相对原step0和本次step1000的变化，不以换分母掩盖退步。

必须分清三者：原独立50步Blur FLUX＋.75固定输出混合=26.559141/.151993，
已经通过SOW平均指标（另有最差单图安全检查失败）；接VLM前image-only MoE
=26.619521/.188334；当前VLM一步MoE=26.705142/.183430。
当前只比MoE起点改善，LPIPS仍不如原独立Blur；不是“原Blur一直没过SOW”。
统一一步MoE未加载原50步Blur LoRA，不能把其旧能力视为已被完整继承。
暂无匹配消融，不能把所有收益归给VLM；原独立专家保留为历史性能参照。

入口/配置：`smoke/run_foundir_token_residual_moe_continue_v1_4x2.sh` /
`configs/foundir_token_residual_moe_continue_v1.json`。
预算2.5–3.5小时、20–28 GPU-hours；每次启动4小时硬cap，至少160GiB新空间，
不删旧checkpoint/cache。首次自动加载唯一hash-bound step1000；中断后新run ID
和第四参数指定本次continuation完整收据。到2000停止，不自动延长。
详见`reports/foundir_token_residual_moe_continue_v1_readiness_2026-11-01.md`。

## 当前：右图③正式1000步结束；联合画质门槛未过 — 2026-11-01

formal_4x2_v1四节点8×H200正常结束，1000/1000步、11份恢复收据、四节点status0。
同运行step0→1000：Rain100 PSNR -0.143334dB、LPIPS下降3.453%；
Blur2595 PSNR +0.085780dB、LPIPS下降2.602%。Rain绝对门槛通过；
完整Blur LPIPS=.1834295仍高于.18，联合门槛失败。没有匹配消融，VLM贡献未证明。
墙钟node0 9431秒，四节点合计20.951667使用GPU-hours。
完整日志/结果与核验见reports/foundir_token_residual_moe_formal_v1_terminal_2026-11-01.{md,json}。
正式训练已终止，不自动延长或启动其他分支；energy源码保持原状。

## 当前执行：已批准右图③ MoE 正式训练1000步 — 2026-10-31

本节覆盖下面“未来正式规模需另定／未获正式批准”的旧说明。用户已授权正式训练；
25步资格已结束，不重跑，不从资格权重继续。**本次只实现、核验、推送正式入口，
本会话尚未启动GPU。**目的地代理同步后做CPU preflight，再给用户四节点启动命令。

正式配置：`configs/foundir_token_residual_moe_formal_v1.json`；
入口：`smoke/run_foundir_token_residual_moe_formal_v1_4x2.sh`。
只做右图③，4节点×每节点2张H100/H200，共8卡；节点内同型号，跨节点允许混用。
共享运行根仍是
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
从原image-only step11000权重初始化，VLM末层归零、fresh Adam，
保留原数据、batch32、损失和学习率；不加CNN/Restormer，不扩展其他对照。
step0与1000评估完整Rain100＋Blur2595，250/500评估Rain100＋固定Blur preview100。
每100步及250步保存权重、Adam、八rank RNG；中断用新run ID和同模式完整收据恢复，
保留原step0基线，最终步支持只补评估，不自动延长训练。

估计2–3小时/16–24 GPU-hours（不是保证；资格纯训练中位5.946秒/步）。
正式需新提取34,695份特征；每次启动4小时硬上限，至少160GiB新空间，禁止删旧资产腾空间。
资格只证明工程可运行，未证明指标改善；正式即使过绝对门槛也不等于已证明VLM贡献。
详细核验、同步和恢复说明：
`reports/foundir_token_residual_moe_formal_v1_readiness_2026-10-31.md`。
另一位代理的FLUX energy显存工作保持独立，不修改/停止其代码或任务。

## 当前：右图token residual MoE四节点25步资格结束 — 2026-10-31

qualification_4x2_v1已于四节点8×H200正常结束，25/25步、四节点launcher status0、
step0/25各Rain100与Blur preview100、五份恢复收据齐全。工程通过，正式训练未启动。
Rain step25相对step0 PSNR -0.055246dB、LPIPS +3.773%；Blur preview PSNR +0.007440dB、
LPIPS +0.509%。预览指标无联合改善，不可宣称VLM贡献或完整Blur门槛通过。
完整日志与限制见reports/foundir_token_residual_moe_v1_qualification_4x2_terminal_2026-10-31.{md,json}。
资格权重不能seed正式训练；未来正式规模需另定，不自动长训。energy显存线保持原状。

## 最新用户授权：只做右图 VLM residual-router；资源最终为4×2 — 2026-10-31

用户在3×2与4×2之间更正后，最后确认使用8卡：**4个节点，每节点2张H100或H200**。
这是跨节点八卡，不是单节点八卡；覆盖下方旧3×2默认。后续非轻量GPU工作默认4×2，
轻量CPU/单卡诊断例外不变。节点间可分别为H100/H200，每节点内两卡同型号。
不要复用历史node0 IP；启动需当前地址与NODE_RANK=0/1/2/3，global batch仍32。

用户已授权现在只实现/验证右图③：保留image router，冻结VLM的视觉token均值
经过小MLP后加到专家logits，再softmax混合同一套FLUX FFN experts。
不同时训练image-only/concatenation两组，不添加energy、CNN、Restormer或新专家库。
从image-only MoE step11000权重初始化，VLM末层零初始化、fresh Adam；
这不是旧八卡任务的精确续训。保留Rain LPIPS .05 / Blur .20及原端点监督。
先做25步工程资格，工程通过不等于科学达标，资格权重不得seed正式训练。
范围仅Rain/Blur；不能声称已覆盖五类或混合退化，不能将已有image-only收益归给VLM。

当前代码与同步清单见
`reports/foundir_token_residual_moe_v1_readiness_2026-10-31.md`。
GPU未启动，本会话看不到迁移后的GPU目录。先安全同步、目的地CPU资产核验、
获取四节点地址后再给用户命令；不擅自启动正式长训或补两组消融。
另一位代理仍负责FLUX-native energy显存排查，不改动/覆盖他的代码，不擅自停止其任务；
其下方状态记录属于另一条工作线，不是阻止本次MoE实现的优先级门槛。
现有energy三节点启动器不会因资源文字更正自动变成四节点；未来重训练须单独适配验收。

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

**最新永久资源规则：所有当前/后续非轻量GPU工作默认3节点×每节点2张H100/H200，
总6卡，允许节点间混用。轻量任务才可少卡；不能按旧段落退回单机8卡。**
读AGENTS.md顶部和 `reports/foundir_flux_native_energy_3x2_readiness_2026-10-31.md`。
当前fresh run为qualification_3x2_v1，全局batch8不变；原4/8卡ID不可直接resume。
下方4/8卡启动设定均为历史，GPU尚未启动。

最新实施状态：FLUX-only 25步代码已实现、9项CPU测试通过，尚未GPU启动/验收。
读 `reports/foundir_flux_native_energy_v1_readiness_2026-10-31.md`，不得运行旧CNN/V3。
先同步，再在GPU节点运行新native-energy入口；不自动200步。

## 2026-10-31 最新覆盖：FLUX-only，禁止外挂 CNN / Restormer

只按 AGENTS.md 顶部和 `reports/foundir_flux_only_energy_redesign_2026-10-31.md`
执行。旧CNN的200步已完成；不得续训/重启SpatialConditionalEnergy或TailQualityEnergy，
不得使用Restormer/refiner/其蒸馏。当前为FLUX自身Transformer表示上的energy重做，
训练器已实现、CPU通过、未启动GPU。下方旧“下一步200步/继续V3”全部失效。
原生冻结FLUX VAE及固定LPIPS保留；它们不是新增可训练CNN。历史代码/结果不删除。

## READ FIRST: workspace handoff, 2026-10-30

The sole active root is
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`.
Fetch `origin/main` and read its `AGENTS.md` and
`reports/foundir_workspace_migration_handoff_2026-10-30.md` before acting.
The old copied local checkout may be stale; do not blindly pull/reset over it.
All operational priorities, old workspace paths and mandatory old-handoff
reading orders BELOW are historical wherever superseded by those documents.
Do not restart an old Rain/MoE/Restormer/EBT/25-step task from this legacy file.
Next: read the latest AGENTS.md completed-qualification/200-step section and
reports/foundir_flux_tail_v3_direct_trainfit_readiness_2026-10-30.md.
The eight-step 3x2 qualification completed; the user approved an independent
fresh 200-step trainfit, not a repeat qualification or restored qualification weights.
Historical bitwise parity is recorded, not a launch gate. Do not repeat diagnostics.
The source-session agent leads decisions/code; the destination agent only
synchronizes the specified files to the shared root and verifies hashes.
The destination CPU observer does not launch GPU jobs; the user launches the
approved trainfit on the GPU nodes. No source/config/recipe changes are needed.
No new-root GPU verification has been done by the source-workspace agent.
Preserve safety/protected-data restrictions and unrelated files; speak Mandarin.

---

No interference with any other folders at all.

Before every work session, explicitly tell the user: "I am working in zhenhaowang, which is the correct workspace."

USER COMMUNICATION LANGUAGE

Communicate with the user in Mandarin Chinese in commentary and final messages.
Keep necessary technical terminology in English when that is clearer or more
precise. Repository code and existing technical documents may continue using
their established language and style.

FRESH-SESSION CONTINUATION — 2026-09-12 (LATEST OPERATIONAL AUTHORITY)

- Read this entire `agents.md` first, then read
  `SESSION_HANDOFF_2026-09-12.md` completely before inspecting, editing,
  launching, stopping, or restarting project work.
- `SESSION_HANDOFF_2026-09-12.md` is the latest authority for live runtime
  state and the exact next task. It supersedes the stale operational status
  and next-task statements in the 2026-09-11 sections below and all older
  handoffs. It does not override the mainline goal lock, advisor governance,
  workspace boundary, destructive-action rule, H100/H200 portability rule, or
  protected-test restrictions in this file.
- The current authorized work is the already-running FLUX.2 isolated-Blur
  full development evaluation requested by the advisor. Monitor that same job
  to completion. Do not restart it merely because its remote PID is invisible
  from the Codex shell; use its advancing telemetry, log, and immutable
  per-record sidecars as authoritative liveness evidence.
- At each completed checkpoint, and at any terminal success or failure, update
  `SESSION_HANDOFF_2026-09-12.md` with the verified timestamp, counts, metrics,
  and exact next action before ending the work session.

GOAL LOCK AND MANDATORY ADVISOR REPORT GATES — 2026-09-11

- The active main goal is the advisor-approved VLM-guided FLUX.2 All-in-One
  restoration system for Blur, Noise, Haze, Rain, and Low-light. Released
  FoundIR generalists, specialists, Restormer, routing experiments, and other
  reference systems are diagnostic/comparison evidence only. They must never
  replace the FLUX.2 mainline or be reported as its result.
- Before changing the model family, restoration backbone, architecture,
  evaluation target, data split, checkpoint-selection rule, or acceptance
  protocol, stop and obtain an explicit advisor/user decision. A promising
  diagnostic result is not authorization to change the goal.
- Generate an advisor-ready report artifact immediately after each of these
  gates: Stage-0/overfit; full-training completion; development checkpoint
  selection; sealed evaluation; any runtime or scientific failure; and every
  proposed model-family or architecture switch. Do not rely on chat memory.
- A terminal scientific failure or runtime failure is a hard stop. Only
  read-only audit and advisor-report preparation are allowed until the advisor
  response is recorded. Do not launch another training, evaluation,
  specialization, routing, or diagnostic branch during this hold.
- Every advisor report must state the exact model, config path and SHA-256,
  checkpoint path and SHA-256, evaluation split, metrics versus SOW with
  pass/fail, sealed/official-test access state, known root cause or uncertainty,
  and the exact decision requested from the advisor.
- Every diagnostic/reference report must display `DIAGNOSTIC ONLY / 非主线` and
  explicitly state that it is not a FLUX.2 mainline or SOW final result.
- Machine enforcement is authoritative: `smoke/advisor_governance.py` writes
  immutable gate reports plus `reports/advisor_governance/advisor_hold.json`.
  Every FoundIR experimental launcher must run its `check` command and fail
  closed while the hold is unresolved. Work may resume only when
  `reports/advisor_governance/advisor_decision.json` records a matching hold
  ID/hash, advisor authority, exact decision, statement, and timestamp. Never
  fabricate this file or infer approval from silence.
- A proposed model-family/architecture change must be registered with
  `advisor_governance.py emit --require-decision` before any implementation or
  GPU launch. This creates the hold even though no experiment has run yet.
- The sealed internal split consumed by FLUX.2 v3 remains consumed. Do not run
  it again, and never access the official FoundIR test, unless the advisor
  explicitly changes the frozen protocol and the user records that decision.

CURRENT AUTHORITATIVE FLUX.2 STATUS AND HOLD — 2026-09-11

- FLUX.2 v3 full training completed `56,250/56,250` steps on 2026-09-08.
  Development selected
  `checkpoints/foundir_five_type_flux2_dit_v3/full_curriculum/step_042500.safetensors`
  with SHA-256
  `8e7b30710fa78d297c27a78c96063052561d4200d31806daac4b0c8a3a867766`.
- Its one-time sealed internal evaluation ended in
  `internal_protocol_scientific_failure`: only Low-light LPIPS passed, so the
  frozen SOW gate passed `1/10` metric checks. The official FoundIR test was
  not accessed and is not authorized.
- The released FoundIR + Qwen routing work performed afterward is diagnostic
  evidence, not FLUX.2. It was a goal drift and must not be continued or used
  as the mainline result.
- The active report/hold is
  `reports/advisor_governance/advisor_hold.json`. Send its bound Markdown
  report to the advisor now. Until the matching advisor response is recorded,
  do not run another FoundIR experiment or propose a self-selected recovery
  route. This status supersedes stale next-task instructions in earlier
  session handoffs.

ADVISOR-AUTHORIZED BLUR-ONLY SANITY SCOPE — 2026-09-11

- The advisor response recorded in
  `reports/advisor_governance/advisor_decision.json` authorizes exactly one
  bounded diagnostic: train FLUX.2 on isolated Blur only to determine whether
  the current training/inference implementation can learn restoration.
- Disable Qwen degradation predictions and all mixed degradations. Retaining
  FLUX.2's frozen native text encoder with one fixed deblurring prompt is not
  Qwen degradation conditioning.
- First audit the code and run a 16--32-pair tiny overfit. If tiny overfit
  fails, stop at code diagnosis. Do not start Blur-only full training.
- Only if tiny overfit passes may Blur-only full training start. Freeze the
  source-group-disjoint validation examples before training. At every
  candidate checkpoint, infer the same fixed validation examples and save
  `Input | Output | Ground Truth` triptychs. Reject visually broken
  checkpoints before metric comparison; never select by PSNR/LPIPS alone.
- Whether FLUX.2 Blur-only succeeds or fails, stop and prepare the advisor
  report. Do not proceed to Z-Image, reintroduce Qwen predictions, add mixed
  degradations, reuse sealed internal, or access official FoundIR test without
  a new explicit advisor decision.

H100/H200 PORTABILITY REQUIREMENT — 2026-09-11

- All subsequent substantial GPU training and evaluation launchers in this
  workspace must support both homogeneous eight-GPU `NVIDIA H100 80GB HBM3`
  and homogeneous eight-GPU `NVIDIA H200` instances unless a required hardware
  feature is proven unavailable and the user/advisor explicitly approves the
  narrower hardware scope. Never infer GPU type from an instance hostname;
  detect the actual CUDA device properties at runtime.
- Hardware-specific microbatch and gradient-accumulation profiles are allowed
  for memory safety and speed only when they preserve the frozen global batch,
  optimizer-step count, sample/presentation order, LR semantics, loss,
  checkpoint schedule, seeds, and evaluation protocol. Record the selected
  hardware, microbatch, accumulation, and any cross-hardware resume in the
  training artifact.
- CPU preflight and focused tests must validate both profiles. A launcher must
  reject mixed H100/H200 ranks, unsupported memory sizes, and profile arithmetic
  that changes the global batch. Do not repeat the 2026-09-11 mistake where an
  H200-only guard blocked an otherwise valid H100 run before optimizer step 1.

FRESH-SESSION CONTINUATION — 2026-08-21

- Read this entire `agents.md` file first. Then read
  `SESSION_HANDOFF_2026-08-21.md` completely before inspecting, editing, or
  launching project work.
- `SESSION_HANDOFF_2026-08-21.md` supersedes
  `SESSION_HANDOFF_2026-08-18.md` for the current operational status and exact
  next task. Older handoffs remain preserved historical evidence.
- The advisor-authoritative architecture, workspace boundary, safety rules,
  gate discipline, and protected-FoundIR policy in this file remain higher
  authority than any handoff. If a handoff conflicts with those rules, obey
  this file and report the conflict.

DELETION AND DESTRUCTIVE-CHANGE CONFIRMATION RULE

Ordinary code, configuration, documentation, and test edits do not require a
confirmation round. When the user's requested scope is clear, carry them out
directly and preserve unrelated work.

Ask for one necessary confirmation only before a genuinely destructive action
that could cause difficult-to-recover loss, such as deleting important data,
overwriting an irreplaceable artifact, moving a critical file without a safe
copy, running `git reset`, or rewriting repository history. Before that action:

- identify the exact destructive targets and what will be preserved;
- use the narrowest possible target and verify it immediately before acting;
- act only after the user confirms that exact scope;
- report exactly what changed afterward.

Never infer permission to delete related, neighboring, generated, backup, or
environment files from a request to delete one item.

You are the execution agent for the image-restoration synthetic-data quality gate.

Read and obey the existing AGENTS.md first.

WORKSPACE BOUNDARY
Only work inside /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang unless the user explicitly asks otherwise. Do not inspect, modify, or run commands in other users' folders.


Do not inspect, modify, delete, or execute commands in any other user's directory. Do not reuse previous project paths, datasets, checkpoints, environments, manifests, or results unless directly verified inside this workspace.

Assume this is a fresh start. Treat every dataset, checkpoint, download, environment, manifest, and result as UNKNOWN until checked directly.

READ FIRST — AUTHORITATIVE ADVISOR-PROVIDED MAIN PROJECT ROUTE — 2026-08-18

This section is the first project-route authority for every fresh agent. If a
later historical plan, experiment handoff, or result conflicts with this
section, follow this section. Preserve the contract text and experimental
history below as evidence; do not mistake them for the current implementation
route.

Advisor architecture image:

- canonical filename: `advisor_vlm_guided_all_in_one_main_route.jpg`
- verified SHA-256 of the advisor upload:
  `02e32805142e64a7c191469fa4fbe4d4169914b25e3fa9df75853d4cde266194`
- this image is the advisor-provided main-route authority; preserve it, keep it
  tracked with this document, and do not substitute or regenerate it

The authoritative route has two stages:

1. Train a degradation-aware VLM with supervised cross-entropy. A degraded
   image and a degradation-assessment question are inputs; degradation type
   and independent severity labels are targets. Following the advisor diagram,
   train the vision encoder and multimodal VLM path while keeping the text
   tokenizer and LLM decoder frozen. This is real degradation-aware VLM
   training, not a frozen-feature linear probe.
2. Validate the trained VLM on the downstream image-enhancement domain it must
   analyze. Synthetic validation alone is insufficient. Keep the official
   FoundIR test protected from tuning; use it only for a frozen, predeclared
   final evaluation, and use a separate labeled real-domain validation set for
   any iterative VLM development.
3. Freeze the validated degradation-aware VLM. For restoration, feed the
   degraded image to both the frozen VLM and the trainable diffusion model;
   use the VLM's predicted degradation types and severities to guide the
   diffusion restoration model. Then evaluate the complete frozen system on
   the official FoundIR task protocol.

CURRENT ROUTE DECISIONS AND DEPRECATED PROTOTYPES

- AgenticIR is the approved synthetic-degradation basis. GenDeg is abandoned,
  removed from the active workspace, and must not be downloaded, restored, or
  restarted. Historical Git commits, reports, manifests, and logs may mention
  it only as superseded evidence.
- The frozen `Qwen2.5-VL-7B-Instruct` mean/std visual-feature extractor plus
  7,169-parameter scalar ridge head is a preliminary linear-probe baseline,
  not the final VLM. It achieved only `68%` severity accuracy on its independent
  synthetic validation and did not perform the advisor-directed cross-entropy
  VLM fine-tuning.
- The one-learned-context-token Qwen-conditioned FLUX.2 run is a preserved
  negative/near-null ablation, not the final conditioning architecture. On the
  internal synthetic L2-L5 test it changed PSNR by only `+0.03683 dB`, SSIM by
  `+0.000163`, and LPIPS by `-0.000357`, with no convincing visible benefit.
- Those internal synthetic results are not FoundIR results and do not establish
  an SOW Stage 1 pass. Do not continue optimizing that frozen-ridge/one-token
  route unless the user explicitly reauthorizes it as a diagnostic.
- The verified Lowlight data, manifests, Qwen/FLUX environments, FLUX
  restoration baseline, evaluator, official FoundIR acquisition, and negative
  experimental evidence remain reusable foundations for the authoritative
  route.

ADVISOR CONFIRMATION — FOUNDIR SCOPE AND JPEG EXCLUSION (2026-09-06)

The advisor confirmed that the first FoundIR work covers only blur, noise,
rain, haze, and low-light, including single and mixed combinations. JPEG and
all other degradation categories are excluded from the first version and from
its main training, validation, and acceptance gates.

The first FLUX route is: freeze the VAE and FLUX native Qwen3 text encoder;
use DiT LoRA rank 32/alpha 32 on the agreed image-attention and text-image
interaction layers; start with the native FLUX.2 reference-image token path;
add a restoration adapter only if a small overfit demonstrates that the native
path is insufficient. The frozen Qwen's actual predictions are the primary
restoration condition. Oracle/generic/shuffled conditions are optional
ablations, not deployment substitutes.

User execution decision (2026-09-06): do not run the optional Qwen-condition
ablations in the current FLUX mainline. Training, checkpoint selection, and
development/internal evaluation use the frozen physical-v8 Qwen predictions.
Do not run oracle, generic, or shuffled/wrong condition comparisons unless the
user explicitly re-authorizes them later. Released FoundIR model comparison is
a restoration benchmark comparison, not a Qwen-condition ablation, and remains
allowed. Five-percent clean GT-to-GT identity pairs remain allowed as training
regularization; generic-condition dropout on degraded pairs is zero.

Operational prohibition: never create a JPEG gate, mix JPEG examples into the
five-type main corpus, or treat a post-JPEG failure as failure of the
advisor-approved five-type task. A JPEG experiment requires explicit advisor
re-authorization and must be labeled auxiliary/diagnostic. The prior
post-target-JPEG v9 run was agent-initiated, not advisor-approved; preserve its
results only as historical evidence.



1.	合作需求/任务
1.1 研究领域
随着人工智能、计算机视觉与多模态大模型技术的发展，图像增强正由面向单一退化的任务模型，逐步转向通用化、语义感知化和大模型化。传统方法通常分别针对低照度、去噪、去模糊、去雨、去雾、超分辨率、色彩校正等问题建模，但真实场景中的图像退化往往复杂耦合，单任务模型在跨场景泛化、统一任务建模和语义一致性保持方面仍存在不足。同时，图像拍摄中也面临着色偏、饱和度衰减、对比度不足、画面灰雾过重、局部细节与色彩层次丢失等退化问题。因此，构建具备多退化适应能力和统一增强能力的 All-in-One 图像增强模型，以及能够实现高保真色彩映射与影调还原的亮色大模型，已成为图像处理与视觉智能领域的重要方向。

近年来，视觉语言模型 (VLM) 凭借跨模态表征、语义理解和开放词汇泛化能力，为图像增强提供了新的技术路径。本项目围绕“亮色大模型”和“VLM Guided All-in-One 图像增强模型”展开研究，以多模态大模型、扩散模型和退化建模为基础，引入语言指令、语义先验和视觉内容理解能力，探索面向复杂场景的通用图像增强方法，实现更加自然、稳定且符合人类视觉偏好的图像质量提升。

1.2	研究目标
本项目面向手机终端图像在拍摄、显示、传输及放大处理过程中出现的复杂质量退化问题，围绕图像色彩与影调恢复以及多类型退化统一建模等关键技术开展研究。项目以扩散模型、图像编辑大模型、视觉语言模型（VLM）和图像退化建模技术为基础，构建由“亮色大模型”和“VLM Guided All-in-One 图像增强模型”组成的通用图像增强技术体系，提升模型对真实场景复杂退化的感知、分析、恢复和泛化能力。

对于亮色大模型，本项目重点解决手机图像经过拍摄、传输或放大处理后出现的色偏、饱和度衰减、对比度不足、画面发灰发蒙、局部雾感增强、明暗层次压缩、纹理细节丢失以及色彩层次不丰富等问题。项目将分析上述退化现象的形成机理及其表现特征，构建包含多种色彩退化、光亮退化和细节退化类型的退化模型。通过对退化类型、退化强度及多种退化组合方式进行建模，生成与真实终端图像退化特征相接近的训练样本，构建高质量、规模化的成对训练数据集，为亮色增强模型训练提供数据基础。在模型设计方面，亮色大模型将以具有较强图像编辑能力的生成式大模型为基础，通过数据微调、条件控制等方式，引导模型学习退化图像与高质量目标图像之间的色彩光亮映射关系。同时，通过引入结构、主色调和语义上的一致性约束，避免处理后存在的结构变化、内容幻觉和颜色漂移，最终实现高保真色彩映射、自然光影重建和稳定的细节恢复。

对于VLM Guided All-in-One 图像增强模型，本项目重点解决真实场景中退化类型多样、不同退化相互耦合、退化强度难以预先确定等问题。项目将研究利用视觉语言模型对图像内容、图像质量和退化属性进行联合分析的方法，使模型不仅能够判断图像是否存在质量问题，还能够进一步识别图像中噪声、模糊、雨纹、雾霾、低照度等退化的种类、程度，为后续图像增强模型提供准确、可解释的任务引导信息。项目将构建面向退化图像理解的多模态感知数据集，为不同退化图像配置退化类型、退化强度、图像内容等多维标注信息，并通过视觉问答、退化描述、任务指令等形式组织训练数据。在此基础上，对视觉语言模型开展退化感知微调和指令微调，使其能够将输入图像转换为结构化的退化分析结果或增强提示信息，例如判断主要退化类型、估计退化程度，并生成相应的图像增强指令。在统一增强模型构建方面，将以视觉语言模型作为高层语义理解模块，以生成式图像恢复模型作为底层图像增强模块。视觉语言模型输出的退化描述将用于引导图像增强模型进行恢复。通过统一的模型框架，实现去噪、去模糊、去雨、去雾和低光照增强等多类任务的联合建模，减少传统方案中针对不同任务分别训练和部署多个独立模型所带来的模型冗余与维护成本。针对真实图像中可能同时存在多种退化的情况，项目将进一步研究复合退化数据构建和多任务联合训练方法，使模型具备对未知退化组合和不同退化强度的适应能力。

最终，本项目将形成完整的亮色大模型和VLM Guided All-in-One图像增强模型技术方案。其中，亮色大模型实现面向手机图像的高保真色彩恢复、全局影调优化和局部细节增强；VLM Guided All-in-One图像增强模型实现对多种退化类型的智能识别、统一建模和自适应恢复。

1.3 交付成果
序号	交付
1	亮色大模型技术报告与完整算法包一份
2	VLM Guided All-in-One图像增强模型技术报告与完整算法包一份
2.	项目阶段
阶段	起止时间	任务描述（含目标）	交付件	验收标准	验收方法
1阶段	T—T+6 	1．调研当前主流图像增强模型及 VLM 技术路线，分析其在色彩恢复、亮度增强和多退化图像增强中的能力与不足，制定亮色大模型与 VLM Guided All-in-One 图像增强模型的整体方案。
2．围绕手机图像放大后发灰、发蒙、色彩不准等问题，设计色彩退化类型，构建亮色增强训练数据；同时评估现有图像增强模型效果。
3．选取代表性生成式图像增强模型和视觉语言模型作为基础模型，开展数据微调、指令微调和初步验证，提升模型在亮色增强和多退化统一增强场景下的恢复能力。	1．亮色大模型技术报告一份。
2．亮色大模型算法包一份，包含亮色数据模拟退化代码，中期模型的训练代码。
3．VLM-Guided All-in-One图像增强技术报告一份。
4．All-in-One图像增强算法包一份，包含数据模拟退化代码，VLM和图像增强中期模型的训练代码。	亮色大模型
1.技术报告需包含相关研究方案和结果；
2.客观指标：在相关benchmark上测试，MUSIQ大于等于55、BRISQUE小于等于30；
All-in-One图像增强
1.技术报告需包含相关研究方案和结果；
2.客观指标：All-in-One图像增强模型在FoundIR-Blur去模糊测试集上，PSNR大于等于20.7dB，LPIPS小于等于0.21。在FoundIR-Noise去噪测试集上，PSNR大于等于31.1dB，LPIPS小于等于0.12。在FoundIR-Lowlight低光照增强测试集上，PSNR大于等于15.1dB，LPIPS小于等于0.34。在FoundIR-Haze去雾测试集上，PSNR大于等于17.7dB，LPIPS小于等于0.24。在FoundIR-Rain去雨测试集上，PSNR大于等于27.4dB，LPIPS小于等于0.1。	客观指标按任务验收标准验收，主观效果由华为方提供测试数据，组织评审效果；技术文档由验收小组组织评审；源代码由华为方提供人员复现。
2阶段	T+6——T+12 	1．完善亮色大模型训练流程，扩大色彩退化数据规模，提升模型对手机放大图像中发灰、起雾、色偏、饱和度衰减等问题的修复能力；
2．优化VLM guided All-in-One 图像增强模型，引入视觉语言理解能力，实现对低光照增强、去噪、去模糊、去雨、去雾等多类退化任务的统一建模与增强；
3．完成模型性能评估与系统验证，从色彩准确性、视觉自然性、结构一致性和用户主观观感等方面验证模型效果，形成亮色大模型和All-in-One图像增强模型两个成果。	1. 完整亮色大模型算法包一份，包括亮色退化数据合成代码、模型训练代码、模型测评代码。
2.完整All-in-One图像增强模型算法包，包括数据合成代码、VLM训练代码、图像增强模型训练代码、模型测评代码。	亮色大模型
1.客观指标：模型在相关benchmark上测试，MUSIQ大于等于65、BRISQUE小于等于25；
2.模型处理后和手机实拍效果的压制比（优+平）/（劣+平）大于1；
3.技术文档通过验收小组集体评审；
4.源代码验证训练、测试可复现。
All-in-One图像增强
1.客观指标：All-in-One图像增强模型在FoundIR-Blur去模糊测试集上，PSNR大于等于25.9dB，LPIPS小于等于0.18。在FoundIR-Noise去噪测试集上，PSNR大于等于38.9dB，LPIPS小于等于0.1。在FoundIR-Lowlight低光照增强测试集上，PSNR大于等于18.9dB，LPIPS小于等于0.28。在FoundIR-Haze去雾测试集上，PSNR大于等于22.1dB，LPIPS小于等于0.19。在FoundIR-Rain去雨测试集上，PSNR大于等于34.2dB，LPIPS小于等于0.08；
2.模型处理后和手机实拍效果的压制比大于1；
3.技术文档通过验收小组集体评审；
4.源代码验证训练、测试可复现。	客观指标按任务验收标准验收，主观效果由华为方提供测试数据，组织评审效果；技术文档由验收小组组织评审；源代码由华为方提供人员复现。
T：合同生效日期
 T+N：合同生效日期后N个月

CONFIRMED RESPONSIBILITY SCOPE AND LOWLIGHT-FIRST SOW GATE — 2026-08-15

The project user is responsible only for the `VLM Guided All-in-One` image
enhancement workstream. The separate `亮色大模型` workstream is owned and
handled by another project member; it is out of scope for this workspace and
for this agent's execution goals. Keep the original SOW text above as contract
context, but do not plan, implement, train, evaluate, or claim completion of
the `亮色大模型` deliverables unless the user explicitly changes this scope.

For the user's All-in-One workstream, FoundIR-Lowlight may be evaluated before
Blur, Noise, Haze, and Rain. The current independent SOW Stage 1 Lowlight gate
is:

- dataset: the official FoundIR-Lowlight evaluation split;
- metrics: restored-image PSNR and LPIPS;
- pass thresholds: PSNR `>=15.1 dB` and LPIPS `<=0.34`;
- systems: first the FLUX.2 restoration baseline, then the same restoration
  backbone with Qwen-VL degradation type/severity conditioning;
- evidence: exact test manifest, checkpoint, code revision, configuration,
  command, output images, and metric JSON/CSV must be retained.

Passing this gate may be reported as `FoundIR-Lowlight SOW Stage 1 sub-target
passed`. It must not be reported as complete SOW Stage 1 acceptance because
the original All-in-One acceptance table independently lists Blur, Noise,
  Lowlight, Haze, and Rain. The other four tasks are not prerequisites for
  running or passing the Lowlight gate; they remain later work for the full
  All-in-One SOW Stage 1 deliverable.

3.	项目监控和管理 请按项目实际情况进行拟定，此部分将写入合同中，交流过程、记录需留存。
本项目采用如下项目管理机制（供参考）：
1.	项目双月月报。
2.	每隔4-6个月的面谈交流项目进展。
3.	根据实际需要安排的其它交流，如电话、邮件、电话会议等。

CURRENT FROZEN AGENTICIR LOW-LIGHT PILOT RECIPE

The frozen low-light recipe is the official AgenticIR synthetic `dark`
degradation followed by project-level luminance calibration.

Implementation and source:

- AgenticIR revision: `9640a291480dee3ba8f2974125d4ee9e3440f3d6`
- official function: `third_party/AgenticIR/installation/tune_depictqa/add_single_degradation.py::darken`
- source images: `data/clean/*.png`
- source image is preserved byte-for-byte as L1
- official darkening method: `darken_type="linear mapping"`
- reproducibility: fix NumPy seed per source; current source `001` seed is
  `2026080600`, source `002` seed is `2026080610`, and source index `i` uses
  `2026080600 + i*10`

Official AgenticIR linear mapping:

- convert BGR to HSV;
- use the V channel;
- sample `dst_max` around `[230, 190, 150, 120]` for levels 1–4 with the
  official normal variation and clipping rules;
- map V by `(V-vmin)/(vmax-vmin)*dst_max`;
- convert back to the original color representation.

Final project labels and luminance intervals:

- L1: clean source, ratio `1.00`
- L2: AgenticIR level 1, target ratio `0.50`
- L3: AgenticIR level 2, target ratio `0.38`
- L4: AgenticIR level 3, target ratio `0.30`
- L5: AgenticIR level 4, target ratio `0.14`

Here `ratio` means final mean luma divided by clean mean luma, with luma
computed as `0.2126R + 0.7152G + 0.0722B` and averaged over all pixels. The
acceptance intervals are:

- L1: `0.95–1.00`
- L2: `0.48–0.52`
- L3: `0.36–0.40`
- L4: `0.28–0.32`
- L5: `0.12–0.16`

The calibration multiplies each RGB pixel in the official AgenticIR output by
`target_luma/raw_luma`, clips to `[0,255]`, and saves a PNG. It does not alter
the spatial structure or add another learned degradation.

The approved tiger reference is in
`smoke/agenticir_low_light_linear_mapping_calibrated_l5_14/`; the independent
validation image is in `smoke/agenticir_low_light_selected_002/`.

CURRENT AGENTICIR SEVERITY POLICY

The current 400-image AgenticIR noise/blur/haze/rain pilot uses one fixed
severity per degradation type and must not be described as a five-level
hierarchy. Low-light is the only current AgenticIR degradation with five
explicit ordered levels. For a later severity-labeled all-in-one dataset,
noise, blur, haze, and rain should also be regenerated with four explicit
severity levels plus clean, using the same non-overlapping interval policy.
Do not expand those four types before the 900-image pilot has passed visual and
training validation.

CURRENT FROZEN AGENTICIR NOISE PILOT RECIPE

The frozen five-level Gaussian-noise recipe is clean L1 plus four AgenticIR
noise severities. It is an isolated noise hierarchy; no darkness calibration
is applied.

- official operator: `basicsr.data.degradations.random_add_gaussian_noise_pt`
- noise is color Gaussian noise with zero mean and standard deviation `sigma`
  in the 0–255 pixel-intensity scale;
- source L1 is copied unchanged;
- L2 sigma range: `[16.848, 49.9824]`
- L3 sigma range: `[37.575, 75.15]`
- L4 sigma range: `[75, 105]`
- L5 sigma range: `[159.6, 199.5]`
- current tiger smoke output: `smoke/agenticir_noise_levels_001_custom_l2_down20_again/`
- independent validation output: `smoke/agenticir_noise_selected_002/`
- validation source seed: `2026080710`

For each level, one sigma is sampled uniformly inside the assigned range,
Gaussian noise is added per RGB channel, and values are clipped to `[0,255]`.
These sigma ranges were visually tuned from AgenticIR's original ranges and
must not be confused with low-light luminance ratios.

CURRENT FROZEN AGENTICIR RAIN PILOT RECIPE

The frozen five-level rain inspection recipe uses butterfly source
`data/clean/002.png`. L1 is a literal clean copy. The current selected output
keeps L2-L3 at the official AgenticIR strengths and strengthens L4-L5 to make
the visual severity gaps clear.

- official basis: `third_party/AgenticIR/installation/tune_depictqa/add_single_degradation.py::add_rain`
- project runner: `smoke/run_agenticir_selected_rain.py`
- seed: `2026080723` for NumPy and Python `random`
- current output: `smoke/agenticir_rain_selected_002_l4_150_l5_340/`
- L1: clean source copy
- L2: `Normal(mean=5, std=5)`
- L3: `Normal(mean=60, std=5)`
- L4: `Normal(mean=150, std=10)`
- L5: `Normal(mean=340, std=20)`

The operator creates a random noise map, keeps sparse bright pixels using
`value*0.01`, blurs it, forms diagonal streaks with width `3`, random length
`20–39`, and random angle `-30–29` degrees, then adds the streaks to the
source and clips to `[0,255]`. The previous rain folders are retained for
comparison; only the folder above is the current selected recipe.

LOW-LIGHT-FIRST TRAINING AND EVALUATION PLAN

The first experiment is intentionally low-light-only. Do not scale to 100,000
images or train on the other four degradation types until this gate passes.

- The first credible pilot uses `2,000` unique clean source images and five
  low-light levels per source, giving `10,000` paired samples.
- Split by source image, never by generated variant: 80% train, 10% validation,
  and 10% internal test.
- Keep source images disjoint from FoundIR evaluation images; record source IDs
  and generation seeds in a manifest.
- Train and compare two systems: a restoration baseline without VLM
  conditioning, and the same restoration model with VLM-predicted degradation
  type/severity conditioning.
- The VLM predicts structured low-light information; it does not directly
  produce PSNR. FoundIR Lowlight PSNR is measured on the restoration model.
- The SOW first-stage FoundIR Lowlight target is PSNR `>=15.1 dB`.
- Record PSNR, LPIPS, checkpoint, config, code revision, environment, seed,
  source split, and exact degradation parameters for reproducibility.
- If the 10,000-pair pilot fails, diagnose recipe/domain mismatch and baseline
  quality before increasing data. If it passes, scale to `5,000–10,000` clean
  sources (`25,000–50,000` pairs) before considering a 100,000-image corpus.

REQUIRED MANIFEST FIELDS FOR THE LOW-LIGHT PILOT

Each generated sample should record: clean source path and source ID, degraded
path, degradation type, project level, official AgenticIR level, darkening
method, seed, raw luminance, final luminance, target ratio, calibration factor,
AgenticIR revision, generator script revision, and train/validation/test split.

CONFIRMED MULTI-DEGRADATION DATA COMPOSITION

The dataset will contain samples with one, two, or three simultaneous
degradation types. This interpretation was confirmed by the project advisor.

- Single degradation: one clean crop receives one degradation, for example
  `low_light L3`, `noise L2`, `blur L4`, `haze L5`, or `rain L3`.
- Two degradations: one clean crop receives two degradations, for example
  `low_light L3 + noise L2`, `blur L4 + haze L3`, or `rain L2 + noise L4`.
- Three degradations: one clean crop receives three degradations, for example
  `low_light L3 + noise L2 + rain L4` or `blur L4 + haze L3 + rain L2`.

Severity is independently assigned to each active degradation in a
combination. Active degradations use project levels L2-L5 only. Project L1 is
reserved for a clean-only crop and is never attached to an active degradation
type. The canonical structured label is:

```json
{
  "degradations": ["low_light", "noise"],
  "levels": {"low_light": 3, "noise": 2}
}
```

The generator and manifest must preserve the degradation list, each component
level, application order, component parameters, source ID, crop coordinates,
and random seed.

The degraded-record composition is frozen at `1:1:1` for single, double, and
triple degradations. This ratio is directly supported by the official
AgenticIR DepictQA synthesis implementation at
`third_party/AgenticIR/installation/tune_depictqa/synthesize_lq.py`: its loop
uses `n_degra=1,2,3` and generates exactly `5,000` images for each category.
That evidence supports only the degraded-record `1:1:1` ratio. AgenticIR uses
eight degradation names in that script, whereas this project selects five:
Lowlight, Noise, Blur, Haze, and Rain. AgenticIR internal levels 1-4 are four
actual degraded strengths; they map to project L2-L5 and must not be confused
with project L1.

CURRENT FROZEN AGENTICIR BLUR PILOT RECIPE

The frozen five-level motion-blur inspection recipe uses butterfly source
`data/clean/002.png`. L1 is a literal clean copy. L2-L5 use the AgenticIR
motion-blur implementation, with the project-selected doubled settings and
fixed random seed.

- official implementation basis: `third_party/AgenticIR/installation/tune_depictqa/add_single_degradation.py::add_motion_blur`
- project runner: `smoke/run_agenticir_selected_blur.py`
- seed: `2026080720` for both Python `random` and NumPy
- output: PNG at the source resolution; original blur outputs are preserved
- L1: clean source copy
- L2: radius `10`, sigma `12`
- L3: radius `20`, sigma `18`
- L4: radius `30`, sigma `30`
- L5: radius `30`, sigma `48`

The runner samples a motion angle uniformly from `[-90, 90]` degrees. It
constructs a normalized one-dimensional Gaussian kernel of width
`2*radius+1`, with weights proportional to
`exp(-i^2/(2*sigma^2))`, distributes those weights along the sampled motion
direction, shifts the image with replicated borders, and clips the result to
the uint8 range `[0,255]`. Radius controls the motion-path length; sigma
controls the Gaussian spread and softness.

CURRENT FROZEN AGENTICIR HAZE PILOT RECIPE

The frozen five-level haze inspection recipe uses butterfly source
`data/clean/002.png` and depth map `data/agenticir/haze/002/predict_depth.mat`.
L1 is a literal clean copy. L2-L5 use the AgenticIR atmospheric-scattering
operator with the project-selected strengthened beta schedule.

- official operator: `third_party/AgenticIR/installation/tune_depictqa/add_single_degradation.py::add_haze`
- project runner: `smoke/run_agenticir_selected_haze.py`
- seed: `2026080722` for NumPy and Python `random`
- output: PNG at the source resolution
- L1: clean source copy
- airlight means A: `[0.60, 0.70, 0.80, 1.00]` for L2-L5
- A sampling: `Normal(A_mean, 0.015)`, with minimum `0.4`
- current beta means: `[2.4, 3.6, 7.2, 10.8]` for L2-L5
- beta sampling: `Normal(beta_mean, 0.30)`

For each pixel, normalized depth `d` gives transmission `t=exp(-beta*d)`,
then the result is `image*t + A*255*(1-t)`, clipped to `[0,255]`. The
current beta means are 6 times AgenticIR's original `[0.4, 0.6, 1.2, 1.8]`.

CURRENT SOW STAGE 1 PLAN — DATA AND DEGRADATION UNDERSTANDING

The immediate SOW Stage 1 work is to construct and evaluate Qwen-VL as a
structured degradation-understanding model. Qwen-VL will receive a degraded
image and predict the degradation types and independent severity levels.

Confirmed data composition:

- Single samples contain one degradation, such as `low_light L3` or `rain L4`.
- Double samples contain two simultaneous degradations, such as
  `low_light L3 + noise L2`.
- Triple samples contain three simultaneous degradations, such as
  `low_light L3 + noise L2 + rain L4`.
- Each active degradation in a combination has its own independent project
  L2-L5 level.
- Project L1 is always clean. No degradation operator is applied. Here a
  literal clean image means an unmodified clean crop after shared preprocessing,
  not necessarily a byte-identical copy of the original full-resolution source
  file.
- A clean-only record has the canonical label
  `{"degradations":[],"levels":{}}`. Labels such as `low_light L1` or
  `low_light L1 + noise L1` are forbidden.
- AgenticIR internal levels 1-4 map to project L2-L5; AgenticIR internal level
  1 is a real degraded strength, not project L1.
- The five-level convention is a project design choice for structured labels,
  not an exact SOW requirement.

CONFIRMED 100,000-RECORD QWEN-VL CORPUS POLICY — 2026-08-17

- Use approved clean sources from DIV2K, MiO100, and a selected LSDIR subset.
- Split by original source image before cropping. Crops from one source must
  stay in the same train, validation, or test split.
- Generate aligned fixed-size crops. For haze, estimate depth on the full
  image first and crop the RGB image and depth map identically.
- Produce `100,000` multimodal records in total: `80,000` train, `10,000`
  validation, and `10,000` fixed test records. Split original sources before
  cropping; the fixed test is never used for training, checkpoint selection, or
  tuning.
- The project-selected composition is `10,000` clean-only, `30,000` single,
  `30,000` double, and `30,000` triple records. The 10% clean quota is a
  project choice that gives Qwen-VL a negative class; it is not derived from
  AgenticIR.
- Preserve `1:1:1` across the `90,000` degraded records. Split each composition
  source-disjointly as follows: train `8,000/24,000/24,000/24,000`, validation
  `1,000/3,000/3,000/3,000`, and test `1,000/3,000/3,000/3,000` for
  clean/single/double/triple respectively.
- Within the degraded groups, balance the five single combinations at `6,000`
  each, the ten double combinations at `3,000` each, and the ten triple
  combinations at `3,000` each. Randomize source, crop, independent L2-L5
  levels, parameters, application order, and seed inside each quota, while
  recording every realized value in the manifest.
- Run a small smoke test covering clean-only records, all five degradation
  types, and all three degraded composition categories before producing the
  full dataset.

LOW-LIGHT 10,000-PAIR PILOT CLEAN-SOURCE RULE

The pilot uses exactly 2,000 verified, unique clean sources and produces five
aligned low-light records per source. The approved composition is:

- 100 existing MiO100/AgenticIR HQ images;
- all 800 images from the DIV2K training split only;
- 1,100 images selected from the LSDIR training split;
- total: 2,000 sources and 10,000 low-light pairs.

Stage the inputs separately under `data/clean_pool/mio100`,
`data/clean_pool/div2k`, and `data/clean_pool/lsdir`. Before source-level
train/validation/test splitting, the source-pool gate must verify readable RGB
content, minimum crop-compatible resolution, exact SHA-256 uniqueness,
near-duplicate screening, dataset-prefixed source IDs, exact per-dataset quotas,
and dataset provenance. Source/license notes should be recorded when available,
but missing manual license metadata does not block this pilot. FoundIR overlap
screening is an optional dataset-leakage check, unrelated to the FLUX.2 GPU smoke:
run it only if a FoundIR SHA-256 index is available, otherwise record
`not_run_no_index` and continue. It is good to have before final FoundIR metric
reporting, but it is not mandatory and must never block source acquisition,
the 10,000-pair pilot, restoration training, or smoke testing. Never use DIV2K validation/test images or
arbitrary web-scraped images. Do not generate the 100,000-record dataset at this
stage; finish and validate only this 10,000-pair pilot.

SOURCE ACQUISITION STATUS — 2026-08-13

- MiO100/AgenticIR: 100 images copied to `data/clean_pool/mio100`.
- DIV2K: the official `DIV2K_train_HR.zip` was downloaded, integrity-tested,
  and all 800 train-HR images were extracted to `data/clean_pool/div2k`.
- Partial source-pool verification passed for 900/900 sources with no rejected
  files: readable RGB, minimum resolution, exact SHA-256, and perceptual-hash
  checks all passed.
- LSDIR: the official Hugging Face repository is auto-gated. Its file listing
  is visible, but downloading `shard-00.tar.gz` requires the user's Hugging
  Face account to be authorized. After the user runs `hf auth login` in the
  network-enabled CPU terminal, download only the official training shard and
  run `smoke/source_pool/finalize_lsdir_sample.sh`. This is the only current
  blocker to reaching 2,000 sources; it is not a GPU or code blocker.

Canonical Qwen-VL target output:

```json
{
  "degradations": ["low_light", "noise"],
  "levels": {"low_light": 3, "noise": 2}
}
```

Every record must preserve clean/degraded paths, source ID, crop coordinates,
degradation list, independent levels, application order, exact parameters,
random seed, generator revision, and source-level split.

Qwen-VL evaluation:

- Keep one fixed source-disjoint test set that is never regenerated or used for
  training.
- Measure degradation-type accuracy and macro-F1, severity accuracy, combined
  label exact match, severity MAE, and valid structured-output rate.
- Qwen-VL accuracy is an internal degradation-understanding metric; it is not
  the FoundIR restoration metric.

CURRENT ADVISOR-ROUTE COMPLEMENTARY MODULE

The advisor route is now explicit: train and validate a degradation-aware VLM
first, then freeze it and use its full structured degradation description to
guide the trainable diffusion restoration model. The current Lowlight slice is
the first implementation target. FoundIR PSNR/LPIPS evaluate the complete
VLM-guided restoration pipeline, not standalone VLM predictions. The SOW
Stage 1 Lowlight sub-target is PSNR `>=15.1 dB` and LPIPS `<=0.34`.

HISTORICAL FRESH-THREAD ROUTE — SUPERSEDED

An earlier route compared a frozen Qwen visual-feature extractor/ridge head
and a one-token diffusion condition using synthetic data. That route was a
diagnostic negative/near-null ablation, not advisor-directed VLM fine-tuning.
It is retained only as historical evidence; do not resume it as the main
project route, and do not describe it as Qwen fine-tuning.

ADVISOR-DIRECTED BACKBONE POLICY

The advisor directed the project to prioritize Z-Image or FLUX-style DiT
diffusion models. LoRA/adapter training is permitted. The fallback order is
SDXL, then SD2.1, if the larger DiT models are too difficult to run or produce
poor restoration results. Do not use InstructIR, DFPIR, MoCE-IR, AdaIR, or the
other previously shortlisted conventional All-in-One models as the final
backbone; they are reference/comparison methods only.

The agent may choose the most suitable publicly available Z-Image or FLUX
checkpoint after checking local/offline availability, license, image-to-image
or restoration compatibility, GPU memory, code reproducibility, and LoRA or
adapter support. If the choice between two viable candidates is genuinely
ambiguous, run a small identical low-light comparison and record the result.
Do not ask the advisor about fallback order; it is already decided. Ask only
if a specific model choice cannot be made from evidence.

MODEL-CONCEPT DEFINITIONS

- DiT: a diffusion transformer. It iteratively denoises image latent tokens.
- VAE: the image/latent encoder-decoder. It converts RGB images to compressed
  latents and converts restored latents back to RGB. Freeze it initially.
- Text encoder: converts text into conditioning embeddings for the diffusion
  model. Freeze it initially unless the selected checkpoint requires another
  documented arrangement.
- Condition adapter: a small trainable projector/MLP or LoRA module that
  converts Qwen-VL's structured degradation result into embeddings accepted by
  the DiT. It is a connector, not a second restoration model.
- Restoration adapter: the trainable image-conditioning path that lets the
  pretrained generative DiT use the degraded image as an input and preserve
  its content while restoring it. It may be implemented with an image encoder,
  ControlNet-like branch, residual adapter, or another architecture supported
  by the selected checkpoint.
- Injection: feeding the condition embedding into suitable DiT blocks through
  cross-attention, AdaLN/FiLM, residual addition, or LoRA. Choose the least
  invasive method compatible with the actual DiT code; do not assume
  cross-attention exists before inspecting the architecture.

STAGE 0 — REPOSITORY AND MODEL AUDIT

Before editing code or downloading anything, inspect this workspace and record:

1. Current Python environments, GPU visibility, available disk, and network
   restrictions.
2. Candidate checkpoint names, sources, licenses, sizes, and checksums.
3. Whether each candidate supports image-to-image conditioning rather than
   text-to-image generation only.
4. VAE, text-encoder, tokenizer, sampler/flow, resolution, precision, and LoRA
   requirements.
5. The smallest reproducible inference command and a plan for deterministic
   output.

STAGE 0 AUDIT STATUS — 2026-08-13 (HISTORICAL SNAPSHOT)

The local audit is recorded in `reports/model_audit_stage0.md`. At that time
the Codex session had no visible GPU and no local Z-Image, FLUX, SDXL, or SD2.1
candidate weights. The later FLUX.2 Lowlight baseline is recorded in the
completed-result sections below; do not interpret this old snapshot as current
training status.
The official FLUX.2 source is now available in the shared workspace at
`third_party/flux2` revision `50fe5162777813d869182b139e83b10743caef15`; the GPU
environment must install it from that local path because it cannot reach GitHub.
The first GPU candidate is FLUX.2 [klein] 4B Base because the official repository
documents image editing, recommends Base for fine-tuning/LoRA, and lists the 4B
model under Apache-2.0. Its offline end-to-end GPU smoke test passed on an NVIDIA
H100 80 GB: all FLUX.2, autoencoder, and Qwen3-4B-FP8 files passed exact size,
SHA-256, and SafeTensors checks; a 512x512 two-step image was generated at
`output/sample_0.png`. This proves that the local stack loads and performs basic
generation. It did not prove restoration quality, faithful degraded-image
conditioning, scene-identity preservation, or FoundIR performance. FLUX.1-dev
was not the first choice because it is a gated 12B model under a non-commercial
license.

ENVIRONMENT AND GPU EXECUTION RULE

The coding-agent terminal may be CPU-only. The newly provisioned user GPU
environment is reported as 169 CPU cores, 1,800 GB system RAM, and 8 NVIDIA
H200 GPUs (expected 141 GB HBM per GPU); verify with `lscpu`, `free -h`, and
`nvidia-smi` before the first run. The user launches GPU work in a separate
GPU-enabled interface. Both interfaces mount the exact same workspace
filesystem, so repository files, model weights, logs, outputs, and
workspace-local environments are visible on the other side. GPU visibility,
running processes, shell variables, and packages installed outside the shared
workspace are not automatically shared.

The agent must therefore:

- prepare and verify code, configs, manifests, and commands in this workspace;
- provide one exact copy-paste **tmux** command for every GPU gate, creating or
  attaching to a named session before launching the job;
- include the matching `tmux attach -t <session>` or log-monitor command so a
  browser/VS Code tab can be closed without terminating the job;
- make the intended `CUDA_VISIBLE_DEVICES` set explicit and use a checked-in
  launcher plus an explicit shared-workspace interpreter;
- keep training/inference commands offline-compatible when required;
- never claim GPU training or evaluation completed unless the user supplies
  terminal output or the shared workspace contains verifiable outputs;
- after the user runs a command, inspect its logs/checkpoints/metrics here and
  give the next command only after the current gate is reported.

REPOSITORY AND GITHUB CHECKPOINT RULE

The top-level `zhenhaowang` repository is the project source of truth. Important
code, configuration, validation scripts, environment specifications, and local
patches to ignored or nested third-party repositories must be represented by
files tracked by the top-level Git repository. Large model weights, generated
datasets, private assets, virtual environments, caches, and runtime outputs must
remain excluded.

After a major implementation milestone or a newly passing end-to-end smoke test:

1. Run the relevant validation or smoke command and save its concise result.
2. Inspect `git status` and `git diff`; do not silently include unrelated user
   changes, credentials, model weights, datasets, or generated caches.
3. Commit the validated source/configuration changes with a descriptive message.
4. Push the current branch to its configured GitHub remote when connectivity and
   authorization are available. Never claim that a push succeeded without
   checking the command result and remote branch state.

The successful FLUX.2 smoke launcher, preflight, reproducible environment
specification, and the local FLUX.2 source patch must all be tracked by the
top-level repository. Editing only `third_party/flux2` is insufficient because
that directory is an ignored nested Git checkout.

Before any GPU command, the user should verify from the GPU interface:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

CPU-side generation, manifest validation, static code checks, and small smoke
tests may run here. GPU model loading, diffusion inference, LoRA/adapter
training, Qwen-VL training, and FoundIR evaluation must be executed in the
GPU-enabled interface by the user unless a GPU is explicitly exposed to this
session.

Do not start Qwen-VL training, full synthetic-data generation, or a large
checkpoint download until this audit identifies one viable backbone.

REQUIRED HANDOFF ARTIFACTS

The fresh agent must leave:

- model-audit report and candidate decision;
- reproducible baseline inference/training command;
- reproducible Qwen-conditioned command;
- exact data manifest and source-level split;
- baseline and Qwen checkpoints;
- metric JSON/CSV and comparison table;
- sample outputs and failure-case contact sheet;
- an update to this file describing what was actually run.

STOP CONDITIONS

Stop and report instead of improvising if: no candidate supports degraded-image
conditioning; a checkpoint is unavailable or its license is unsuitable; the
model hallucinates content in the baseline; the baseline cannot run on the
available GPU; Qwen output cannot be injected without changing the baseline
architecture; or the two experiments cannot be made computationally fair.

LOW-LIGHT 10,000-PAIR PILOT — COMPLETED 2026-08-14

The low-light-first pilot is complete. This milestone covers only the approved
10,000-pair pilot; it does not authorize or claim generation or training on a
100,000-image dataset.

Authoritative artifacts:

- verified source manifest: `manifests/low_light_source_pool_2000.json`
- paired-record manifest: `manifests/low_light_pilot_10000.csv`
- generation metadata: `manifests/low_light_pilot_10000.json`
- generated files: `data/low_light_pilot/` (runtime data; never commit to Git)

Final verified composition:

- 2,000 unique clean sources: MiO100 `100`, DIV2K train `800`, LSDIR train
  `1,100`
- 10,000 paired records and 10,000 physical output files
- L1, L2, L3, L4, and L5: exactly `2,000` records each
- source-disjoint split: train `1,600`, validation `200`, test `200`
- physical layout: `2,000` clean files and `8,000` degraded files
- no generated `.partial` or temporary PNG files remained at completion
- FoundIR overlap screening was not run because no FoundIR hash index was
  available; as frozen above, this is recommended but not a pilot blocker

The interrupted generation was resumed with
`generate_low_light_pilot.py:v3-resume`. Resume mode verifies existing clean
files by SHA-256 and existing degraded files by exact expected pixels before
reusing them. It refuses to overwrite any differing existing output. The final
run safely reused `5,296` files and wrote only the `4,704` missing files.

The independent validator was run over all 10,000 physical files with two
low-memory workers and completed with:

```text
LOW_LIGHT MANIFEST PASS: 10000 pairs, 2000 sources
levels {'L1': 2000, 'L2': 2000, 'L3': 2000, 'L4': 2000, 'L5': 2000}
source_splits {'train': 1600, 'validation': 200, 'test': 200}
source_datasets {'mio100': 100, 'div2k': 800, 'lsdir': 1100}
```

The CPU coding-agent container exposed many host CPUs but had a 4 GiB cgroup
memory limit. Sixteen workers were therefore rejected after an OOM termination;
the successful reproducible configuration used two workers with OpenCV/BLAS
thread counts limited to one per worker. The validator retains every original
fail-closed hash, image-readability, shape, luminance, level-set, split-leakage,
and dataset-composition check while adding bounded parallelism and progress
output.

The authoritative rerun/verification command is:

```bash
LOW_LIGHT_WORKERS=2 smoke/run_low_light_pilot_10000.sh
```

The launcher defaults to safe resume, the verified 2,000-source manifest, two
workers, and one numerical-library thread per worker. A rerun verifies and
reuses matching generated files and fails closed instead of overwriting any
differing output.

LOWLIGHT BASELINE CPU READINESS — VERIFIED 2026-08-15

This status is CPU-side readiness evidence only. No new GPU training,
restoration evaluation, visual identity decision, or FoundIR result is claimed.

- The Stage 1 architecture audit is recorded in
  `reports/lowlight_stage1_architecture_audit.md`. It explicitly identifies the
  rectified-flow training objective as project-derived because the local BFL
  FLUX.2 checkout is inference-only and contains no official training loop.
- The trainer now publishes checkpoint artifacts without overwrite, records
  and verifies adapter/optimizer SHA-256 values, reconstructs a missing JSONL
  metric after a completed-checkpoint interruption, rejects partial or
  differing artifacts, and records Git/FLUX revisions, exact config,
  command, environment, and output paths.
- The loss-decrease gate now compares an initial and final deterministic probe
  using fixed records, crop, noise seed, and timestep, rather than unrelated
  stochastic minibatches.
- Nine CPU unit tests passed. They cover deterministic aligned crops,
  rectified-flow/reference layout, zero-init LoRA behavior and gradient scope,
  LoRA checkpoint round trip, config rejection, deterministic index ordering,
  interrupted-summary recovery, non-finite metric rejection, and refusal to
  reuse a failed evaluation.
- Python compilation, launcher shell syntax, and `git diff --check` passed.
- CPU preflight passed both with `--skip-large-hashes` and with all model/text
  encoder SHA-256 checks enabled. It verified 16 records, 16 unique sources,
  all L1-L5 levels, 50 LoRA targets, 8,110,080 trainable parameters, and config
  SHA-256
  `62740e0e0f26c4001057adf9d955259b7c8d89553245a2c1565a22599e7d0804`.
- The CPU allocation exposed 128 logical Intel Xeon Gold 6530 CPUs, a 300 GiB
  cgroup memory limit, about 1.85 GiB cgroup use at inspection, and about
  964 TiB free on the shared filesystem. The validated interpreter reported
  Python 3.12.13 and PyTorch 2.7.0+cu128 with CUDA 12.8 support, but
  `torch.cuda.is_available()` was false in this coding-agent terminal.
- Full 8,000-pair training remains unauthorized until the offline H100
  16-pair automatic gate and human contact-sheet identity review both pass.

LOWLIGHT FLUX.2 OVERFIT16 GATE — PASSED 2026-08-15

The bounded 16-pair FLUX.2 Lowlight overfit gate passed on the H100 at
guidance 1.0. The original guidance 4.0 evaluation is retained as a failed
diagnostic: it over-amplified color/exposure and failed the L1 identity floor.
Training used only the conditional prompt path, so guidance 1.0 is the matched
conditional-only inference setting and is frozen for the next baseline stage.

- Exact result report: `reports/lowlight_overfit16_result_2026-08-15.md`
- Machine-readable gate: `configs/lowlight_overfit16_gate.json`
- checkpoint SHA-256:
  `94ecd1c32e7a137ed153b581ea0b12c879d7e5e3c0698375a3d2293ecb6481c7`
- passing metrics SHA-256:
  `9ffb568a500eb378a2ad81ccd4ff4f329bf47b3dcb589dbe5d12623aeba96edc`
- L2-L5 mean PSNR: `9.2784 -> 22.1979 dB`
- L2-L5 mean SSIM: `0.4287 -> 0.8537`
- restored L1 mean SSIM: `0.7731`
- automatic gate: pass
- visual identity gate: pass; all scenes, objects, and layouts are preserved,
  with no structural hallucination observed

The overfit result authorizes preparation and preflight of the full 8,000-pair
baseline. It does not itself authorize a FoundIR or SOW Stage 1 success claim.

LOWLIGHT FULL BASELINE PIPELINE — CPU-VERIFIED READINESS 2026-08-15

The fixed full baseline pipeline is implemented for the already approved
Lowlight-only scope. This is readiness evidence, not a GPU training result,
FoundIR result, Qwen-VL result, or SOW Stage 1 completion claim.

- Frozen config: `configs/lowlight_flux2_full.json`
- Config SHA-256:
  `c71235efe88e8f3bc98ca863350af372a78aa35f1a2cc065a1867c5bb6bfde18`
- Launcher: `smoke/run_lowlight_baseline_full.sh`
- Split evaluator: `evaluate_flux2_lowlight_split.py`
- Readiness report: `reports/lowlight_full_baseline_readiness_2026-08-15.md`
- Training: 8,000 records / 1,600 sources / 4,000 steps
- Evaluation: fixed 1,000-record validation then fixed 1,000-record internal test
- Inference: 50 steps, guidance 1.0
- Metrics: PSNR, SSIM, and offline AlexNet LPIPS, overall and per level
- Restart evidence: hashed per-image JSON sidecars, restored PNGs, aggregate
  metrics, and stratified/worst-case contact sheets

Authorization fails closed unless the exact full config and the passing
overfit gate match. Runtime authorization also re-hashes the overfit checkpoint
and its passing guidance-1 metrics. Fifteen CPU unit tests, Python compilation,
shell syntax, the 8,000-record preflight with large model hashes skipped, and
the independent preflight with all FLUX/VAE/text-encoder hashes passed before
the full H100 launcher was run.

LOWLIGHT FULL INTERNAL BASELINE — PASSED 2026-08-16

The fixed FLUX.2 full baseline completed 4,000 H100 training steps and complete
evaluation on the 1,000-record validation and 1,000-record internal-test
splits. Both automatic gates and the contact-sheet visual identity review
passed. This authorizes preparation of the Qwen-VL Lowlight conditioning
comparison only; it is not an official FoundIR result or an SOW Stage 1 completion
claim.

- Result report: `reports/lowlight_full_baseline_result_2026-08-16.md`
- Machine-readable gate: `configs/lowlight_full_baseline_gate.json`
- Checkpoint:
  `checkpoints/lowlight_baseline_full/step_004000.safetensors`
- Checkpoint SHA-256:
  `fa51deac872e3df57f147631361841039c386ef43f58a17a56fb68b9924e8f22`
- Training-summary SHA-256:
  `4581852a5ed35a1654f239040cd050d5ee40b462a22b57215c50f8afb3cd538b`
- Validation-metrics SHA-256:
  `b1b78a9afa5174786702ac17fc2db44524ba374a2264c42067ba68fa3ff08e51`
- Internal-test-metrics SHA-256:
  `6bac93cb0a45556787364c553242d15463b2ea0a5a18d5738054b2a87e1a1da4`
- Validation L2-L5: PSNR `24.9051`, SSIM `0.8913`, LPIPS `0.0411`
- Internal test L2-L5: PSNR `24.7723`, SSIM `0.8858`, LPIPS `0.0440`
- Validation L1 identity: PSNR `31.4672`, SSIM `0.9317`, LPIPS `0.0172`
- Internal-test L1 identity: PSNR `31.6674`, SSIM `0.9266`, LPIPS `0.0185`

The stratified and worst-case sheets preserve scene structure and identity
without structural hallucination. Genuine failure cases remain: several
extremely dark or nearly uniform crops are under-restored, including
`div2k_0452_L5` validation at `3.01 dB`, and some individual L1 crops show
tonal/texture changes despite the strong mean identity score. Preserve these
outliers in the later fair comparison.

GPU COMMAND RELIABILITY, SESSION RETENTION, AND ACCELERATOR UTILIZATION RULE — 2026-08-19

These rules apply to every current and fresh agent. They supplement the
existing GPU execution and repository rules above. Their purpose is to prevent
copy/paste launch failures, preserve terminal evidence after a job exits, and
minimize end-to-end experiment wall-clock time without silently changing the
scientific experiment.

COMMAND CONSTRUCTION AND PREFLIGHT

- Before giving the user any GPU command, verify the exact workspace, launcher,
  config, interpreter, and executable paths that the command will use. Run
  Python compilation, launcher shell syntax, config parsing, and the relevant
  CPU preflight first. Never assume that a utility on the coding-agent PATH is
  also on the GPU-shell PATH.
- Resolve `tmux` before launch, check that it is executable, and run
  `<path>/tmux -V`. As of 2026-09-07, this coding terminal has verified
  `/usr/bin/tmux` version 3.4. The path `.tools/tmux-env/bin/tmux` is currently
  a socket rather than an executable and must not be presented as a fallback.
  Do not give bare `tmux` when its presence in the user's GPU shell has not
  been demonstrated.
- Prefer a checked-in launcher over a long inline training command. Commands
  shown to the user must be short, literal, and safe to paste one line at a
  time. Use `cd` and short shell variables on separate lines when necessary.
  Do not allow a long absolute path or command to be visually wrapped into
  executable newlines, and do not rely on fragile backslash continuations.
- Give only one launch action at a time, followed by separate attach and log
  inspection commands. State exactly which lines should be pasted together and
  which should be run only after the launch succeeds.
- Before retrying a command after any shell error, inspect the shared log,
  output directory, checkpoint directory, and terminal markers to determine
  whether the job never started, is still running, completed, or failed. Never
  launch a duplicate process based only on a missing tmux session.

TMUX LIFETIME AND PERSISTENT EVIDENCE

- A tmux session must remain inspectable after its training command returns.
  Do not make the training command the session's only process unless tmux
  `remain-on-exit` is enabled. Prefer creating a named session with a persistent
  shell and then sending the checked launcher to that shell. This lets the user
  attach after success or failure and inspect the terminal history.
- Every GPU launcher must persist unbuffered stdout and stderr to a unique log
  under this workspace, while preserving the real pipeline exit status with
  `set -o pipefail`. The log must contain a UTC launch marker, exact workspace,
  config, Git revision, visible device set, environment check, terminal result
  marker, and UTC completion marker with the numeric exit status. A failure
  path must write its completion marker too.
- Always provide the exact workspace-local tmux attach command and the exact
  persistent-log command. Explain that a missing session does not imply lost
  logs and that a nonzero gate exit can be an intentional automatic failure
  rather than a crash.
- After the user launches a job, do not claim that it is running until a live
  process, advancing shared log, advancing step artifact, or checkpoint has
  been verified. After it exits, inspect the terminal marker, complete log,
  machine-readable summary, per-step artifact count, checkpoint hashes, and
  failure records before reporting or proposing another run.

GPU COUNT AND WALL-CLOCK MINIMIZATION

- The default objective is to minimize total end-to-end wall-clock time while
  preserving the predeclared samples, optimizer steps, global/effective batch,
  learning-rate semantics, seed policy, precision, checkpoint selection, and
  evaluation protocol. Do not default to one GPU merely because single-GPU
  code is easier.
- Choose the GPU count adaptively using prior operational evidence, the job's
  size, expected distributed overhead, implementation correctness, and sound
  engineering judgment. Use all eight H200s concurrently for substantial jobs
  when they are reasonably expected to shorten wall-clock time. Use fewer GPUs
  for genuinely small jobs when startup, preprocessing, or communication would
  probably dominate. Do not turn either choice into a rigid universal rule.
- Existing trustworthy operational evidence may justify the decision even when
  it did not come from a perfectly controlled scaling experiment. In this
  workspace, the passing eight-GPU Qwen DDP run achieved about `15.20`
  samples/second versus about `1.055` samples/second in the earlier single-GPU
  run, a greater-than-tenfold observed speedup. Although those schedules were
  not a formal apples-to-apples scaling study, this is sufficient evidence to
  prefer eight GPUs for the current substantial Qwen pilot.
- Do not require a dedicated 1/2/4/8 scaling gate before every substantial run.
  Ordinary run evidence is normally sufficient: record the selected GPU count,
  end-to-end or steady-state throughput, wall time, per-GPU utilization and
  memory, global/effective batch, gradient accumulation, precision, and samples
  or steps seen. A short initial observation of the real run may also confirm
  that the chosen devices are active and that performance is reasonable.
- Run a formal controlled multi-GPU timing study only when the fastest useful
  GPU count is genuinely uncertain, observed scaling is unexpectedly poor, a
  large amount of compute is at stake without relevant prior evidence, or
  distributed execution may change scientific semantics. Select the fastest
  scientifically equivalent configuration when such a study is actually
  needed; do not create a ceremonial benchmark gate merely to authorize an
  otherwise well-supported run.
- If a substantial launcher supports only one GPU while additional GPUs are
  likely to provide a meaningful speedup, first make a reasonable effort to use
  a correct distributed path. This requirement is practical rather than
  pedantic: it must not block useful work for an exhaustive benchmark when
  existing evidence already supports the multi-GPU choice.
- Monitor all selected GPUs during the actual run and persist utilization and
  memory evidence in the workspace. If utilization is low, investigate input
  loading, preprocessing, batch sizing, gradient accumulation, checkpoint
  frequency, attention kernels, and communication before accepting a long
  runtime. Never claim `full utilization`, `maximum efficiency`, or `minimal
  training time` without recorded measurements.
- Scaling must not silently change the experiment. If distributed execution
  changes effective batch size, optimizer-step meaning, learning rate,
  ordering, numerical determinism, or number of samples seen, either preserve
  equivalence or declare and justify the change before launch. Keep a fair
  comparison against the fixed single-GPU or prior baseline.

QWEN REAL LOWLIGHT ADAPTATION GATE — CPU-VERIFIED READINESS 2026-08-20

STATUS NOTE — 2026-08-21: This section is a preserved pre-run readiness
snapshot. The gate subsequently completed and failed its automatic scientific
gate. Do not launch it from this section. Read `SESSION_HANDOFF_2026-08-21.md`
for the verified result, consumed-holdout restrictions, and exact next task.

The frozen synthetic-pilot step-600 VLM failed both its original real-domain
free-generation gate and the post-failure constrained-scoring diagnostic. The
failures are legitimate status-2 results, not crashes, and remain immutable:

- original result SHA-256:
  `85542baf277c72a442e12d6da4f04e02051e45f7d586020472253437b6ebceda`;
- constrained result SHA-256:
  `8cb62790a423438d2d62e096d0985075a3c4c2e179e950d2beb5a0e61ab44552`;
- neither result may be relabeled as a pass.

The next authorized gate is a bounded source-disjoint real adaptation using
only the official LOL-v2 Real-captured Train split plus frozen synthetic
replay. It is CPU-ready but has not yet produced a GPU result.

- config: `configs/qwen_lowlight_real_adaptation_step600.json`
- config SHA-256:
  `0a817d70d88f172d6d16d37853ce9c3cd77308446b3eed3d5e5cdab0c3b3b541`
- real manifest SHA-256:
  `91f95644f852f628891513f116c0a3607b8de980bbe5252c360ff41d9bcc9ab7`
- real metadata SHA-256:
  `2b9f35ffdc2543a4a99bf95fbb4eb61938a022c54cba4569a14863c5b536f06f`
- proxy-label disclosure: `reference_based_proxy_not_human_ground_truth`
- source-disjoint splits: train 673 records / 190 scenes, validation 143 / 41,
  and untouched fresh test 145 / 41
- initialize all visual weights from frozen synthetic step 600, visual SHA-256
  `b0e65a8597077973a2a103dc3369e0cb1ea703de95e2e4841d359c3d34eff77f`
- keep the language decoder and LM head frozen; reinitialize the optimizer
- train 200 optimizer steps at global batch 40; every step contains 20
  L1–L5-balanced real presentations and 20 L1–L5-balanced synthetic-replay
  presentations
- schedule SHA-256:
  `8ae8e4e06a306c3828423374e39cecac4e3be2cf90b2b08564005b928a62c71f`
- checkpoint selection uses only real-validation macro-per-level
  teacher-forced loss; evaluate the frozen fresh real test exactly once after
  selection
- retain a 40-source / 200-record balanced synthetic validation replay with
  bundle SHA-256
  `1981939ac92de5412df990f39f9d7c2ee8776ec537e76dec059a81378407cfc3`
- FoundIR is not used.

The same exact global schedule supports either two or eight GPUs. Two GPUs use
microbatch 5 per rank and four gradient-accumulation steps; eight GPUs use
microbatch 5 per rank and one accumulation step. The checked launcher requires
the explicit literal device set `0,1` or `0,1,2,3,4,5,6,7`, accepts H100 or
H200 devices with at least 75 GiB each, monitors every selected GPU, locks
against duplicates, writes persistent logs and terminal markers, and retains
the tmux pane after exit. Use the available two H100s promptly for this small
gate when waiting for eight H200s would increase end-to-end wall-clock time;
use eight H200s if they are available before launch. This is an adaptive
engineering choice, not a mandatory scaling benchmark.

Readiness evidence is recorded in
`reports/qwen_lowlight_real_adaptation_readiness_2026-08-20.md`. Thirty focused
tests, Python compilation, shell syntax, contract validation, the CPU preflight
with large base-model hashes skipped, and the CPU preflight with all hashes
enabled passed. Both CPU preflights deliberately used `--no-write-artifact`;
the GPU launcher must create the immutable full-preflight artifact with CUDA
visible before training.

Do not start diffusion or FoundIR from readiness alone. If the adaptation gate
passes, stop and obtain user review before beginning Lowlight diffusion
conditioning. If it exits status 2, preserve and diagnose the failure; do not
lower thresholds, tune on the fresh real test, or relabel proxy metrics.

REAL-SEVERITY AND FOUNDIR OBJECTIVE CORRECTION — 2026-08-21

This section is a later authoritative correction to the real-domain validation
procedure. It does not change the controlled synthetic L1--L5 schema or the
advisor's VLM-guided architecture.

- FoundIR-Lowlight has paired LQ/GT restoration targets and is evaluated by
  restored-image PSNR and LPIPS. It has no official project L1--L5 labels.
- Do not ask humans to assign precise absolute L1--L5 levels to unrelated real
  photographs. Without same-scene exposure anchors, those levels are not
  objectively identifiable and cannot support a credible quantitative gate.
- The screened Retinex-Net singleton images remain an unlabeled qualitative
  robustness pool. The proposed 190-component absolute L1--L5 annotation is
  stopped and must not be revived as a prerequisite.
- Objective real severity development uses same-scene exposure ordering:
  darker exposure implies non-decreasing predicted low-light severity. Use
  soft canonical-target scores, pairwise concordance, rank correlation,
  adjacent inversions, EV consistency where available, and prediction span so
  a constant output cannot pass.
- Relative exposure ordering establishes only ordinal sensitivity. The
  decisive VLM evidence is causal improvement of paired real restoration under
  baseline, oracle, predicted, and shuffled conditions. If oracle conditioning
  cannot beat baseline, redesign the restoration-conditioning interface before
  further Qwen adaptation.
- Reusable real paired development suites and leave-one-dataset-out validation
  replace the repeated search for newly named "fresh tests." Official FoundIR
  remains protected until the complete system and inference protocol are
  frozen.
- The current first-principles SOTA strategy is recorded in
  `reports/foundir_lowlight_sota_strategy_2026-08-21.md`. It prioritizes the
  official FoundIR training distribution and released teachers, realistic
  camera formation, a fidelity-first degraded-to-clean FLUX/Z-Image bridge,
  VAE/high-frequency ceiling tests, causal condition ablations, and native
  full-resolution deployment.
- Official FoundIR metadata contains 39,962 isolated-Lowlight training pairs
  and 378,440 pairs in categories containing Lowlight. Training-set access and
  teacher distillation are major unexploited levers. Never use FoundIR test
  pairs for training, calibration, or iterative selection.

The frozen same-scene development manifest is
`manifests/qwen_lowlight_real_exposure_ordering.csv`, with its contract in
`manifests/qwen_lowlight_real_exposure_ordering.json`. It contains 15 MEF
exposure stacks, 67 exact-unique images, and 151 comparisons separated by at
least one EV. It was selected and hashed before model inference, contains no
FoundIR image, and is explicitly a reusable development suite rather than an
absolute-label holdout.

PAIRED-REAL GATE-A EXECUTION UPDATE — 2026-08-21

- The reusable non-FoundIR paired-real suite is frozen at
  `manifests/lowlight_real_paired_development_v1.csv`: 174 records, 82 scene
  components, 102 LOL-v2 real-adaptation validation records, and 72 SICE
  reusable validation records. It excludes L1, consumed-test records, and
  exact FoundIR overlap. Project levels are proxy metadata only.
- The target-fitted global photometric oracle completed without model
  inference. Scene-balanced PSNR is `8.4356 -> 24.1209 dB` and SSIM is
  `0.2475 -> 0.6571`; the PSNR gain is `+15.6853 dB` with 95% scene-bootstrap
  CI `+14.7717` to `+16.6036`. The oracle accesses each development target and
  is never an inference method. It makes the bounded learned photometric head
  the first model component to implement.
- The released FoundIR generalist checkpoint is present at step 1,300,000 with
  SHA-256
  `2603a0be47d0844d4c9cea542ecca039e0e4aaa39d45d41064bfbdc13c0830c9`.
  The Lowlight specialist SHA-256 is
  `f834248e145d020d5a02f257b7ceb383b579add2522938b3e1e0bd460abe6`.
  No teacher inference has run.
- The next GPU gate is the two-H100 FLUX.2 VAE ceiling at
  `configs/lowlight_vae_ceiling_v1.json`. It uses literal
  `CUDA_VISIBLE_DEVICES=0,1`, 130 exact-unique paired-development targets,
  direct 512-pixel crops, full-resolution 1024-pixel overlapping tiles, and
  PSNR/SSIM/LPIPS/high-frequency/seam diagnostics. It does not use FoundIR.
- The VAE result must decide latent-only versus pixel-frequency bypass before
  the restoration bridge is implemented. Do not start official FoundIR test
  inference from VAE readiness or from the photometric oracle.

FOUNDIR FIVE-TYPE FLUX.2 VAE DECISION AND V3 ROUTE — 2026-09-07

- Current scope is the advisor-approved five-type FoundIR training route:
  Blur, Noise, Rain, Haze, and Low-light, including admitted mixtures. JPEG is
  prohibited. Qwen stays frozen. The current mainline contains no
  oracle/generic/shuffled-wrong Qwen-condition ablations. The official FoundIR
  test remains protected and must not be accessed without explicit user
  approval after the internal protocol is frozen.
- The corrected eight-GPU v2 runtime preflight passed. Its 44-record
  FoundIR-training-only VAE ceiling finished at `2026-09-07T00:50:35Z` and
  returned `pixel_refinement_required`, not a crash. Exact result:
  `checkpoints/foundir_five_type_flux2_dit_v2/vae_ceiling_result.json`, SHA-256
  `1702e40a8938fcbf659f0442a6bce51e2d87d69aa261e6c335da3dffa866eba9`.
  Failures were Noise HF correlation `0.89056 < 0.90000`, Low-light HF
  correlation `0.86160 < 0.90000`, and Low-light+Haze PSNR
  `34.72370 < 35.00000` plus HF correlation `0.82381 < 0.90000`. No FLUX
  optimizer step started in v2.
- V3 keeps FLUX.2 Klein 4B as the restoration backbone, with frozen VAE,
  frozen native Qwen3 text encoder, native reference-image tokens, and DiT
  LoRA rank/alpha 32/32. It adds only a zero-initialized, bounded, 33,251-
  parameter local pixel refiner after VAE decode. The refiner receives decoded
  RGB, aligned LQ RGB, and their difference; its per-channel correction is
  bounded to `0.25` in normalized `[-1,1]` space. It is part of the single
  deployed restoration checkpoint, not an independent generator.
- V3 reuses the frozen v2 VAE measurement rather than wasting eight GPUs on
  the identical calculation. Reuse is allowed only when exact source-config,
  result, manifest, VAE-weight, selection-seed, thresholds, category coverage,
  failure status, and protected-test identities pass.
- The v3 overfit trains FLUX LoRA and the pixel refiner jointly. Its 32-pair
  evaluation is balanced across all 11 admitted categories and all composition
  sizes. In addition to overall loss and restoration gain, the refiner itself
  must improve PSNR and must improve high-frequency correlation on Noise,
  Low-light, and Low-light+Haze before full training can start.
- The v3 launch route is
  `smoke/launch_foundir_five_type_flux2_main_v3_8gpu.sh`. It requires exactly
  eight homogeneous H100 or H200 GPUs and literal devices
  `0,1,2,3,4,5,6,7`. Inspect the v3 stage-0 result before treating the route as
  scientifically viable. Never call CPU readiness or launch success a training
  pass.

FOUNDIR SOW METRIC CORRECTION — 2026-09-07 (SUPERSEDES THE V3 HF GATE)

- The FoundIR restoration SOW names PSNR and LPIPS as its objective metrics.
  High-frequency correlation is not an SOW metric and is removed completely
  from the active five-type FoundIR evaluation and gate logic. It must never
  block overfit admission, full training, checkpoint selection, or final SOW
  reporting.
- The first v3 200-step overfit and its result are immutable historical
  evidence. That run measured PSNR but omitted LPIPS and then stopped on the
  superseded high-frequency-correlation gate. Preserve its checkpoint, result,
  and logs exactly; do not relabel or overwrite that historical result.
- The active SOW-aligned continuation reuses the exact historical v3 step-200
  checkpoint only to evaluate the same 32 FoundIR-training-only overfit records
  with both PSNR and offline AlexNet LPIPS. Its new result is isolated under
  `native_reference_overfit_sow/`. If the historical checkpoint is absent, the
  corrected overfit trainer may reproduce the 200-step gate from scratch.
- The corrected overfit gate requires training-loss reduction, final-output
  PSNR improvement over LQ, and final-output LPIPS improvement over LQ. It does
  not use the absolute SOW thresholds because a 32-record overfit subset is a
  learnability check, not the official FoundIR evaluation.
- Development metrics use the source-group-disjoint FoundIR training
  validation split. Checkpoint selection uses PSNR and LPIPS jointly across all
  five tasks. A separate sealed source-group-disjoint internal-test split is
  consumed once after selection. These training-archive splits estimate
  generalization but do not replace or claim results on the official FoundIR
  test.
- The official FoundIR test remains protected. Only after the architecture,
  checkpoint, and inference protocol are frozen may the user explicitly
  authorize one final official-test evaluation. Only that result can establish
  the contractual FoundIR PSNR/LPIPS outcome.

FOUNDIR QWEN CONDITION-EXPORT OPTIMIZATION — 2026-09-07

- The frozen physical-v8 Qwen condition semantics do not change: every LQ
  center crop is still evaluated through five independent dialogues for Blur,
  Noise, Rain, Haze, and Low-light, using the same model, adapter, prompts,
  greedy decoding, and canonical condition conversion.
- The optimized exporter shards complete image records across eight ranks,
  opens each local image once rather than once per dialogue, and evaluates up
  to 64 same-type dialogues per GPU call. It does not combine the five
  questions or replace Qwen decoder decisions with a classifier.
- Existing 512-record condition chunks are immutable and resume-compatible.
  Before the optimized engine may write any new chunk, it must regenerate one
  complete existing chunk and obtain exact row-for-row equality, including raw
  Qwen output and canonical restoration text. If exact equality fails, the
  exporter automatically uses the legacy engine.
- The exporter optimization is not yet GPU-validated merely because CPU tests
  pass. Treat the first `FOUNDIR_PHYSICAL_V8_OPTIMIZATION_VALIDATION` marker as
  the runtime gate, and report its exact equivalence and measured throughput.
  The `--benchmark-only` mode performs this gate without writing a new chunk,
  so it may run on a separate eight-GPU instance while the legacy exporter
  continues; only its explicit benchmark-pass marker authorizes a switchover.
- The active legacy process that started before this source change is
  unaffected and may continue producing valid chunks. Never discard or
  overwrite those chunks. The official FoundIR test remains inaccessible.

RUNTIME RESULT AND APPROVED PARALLEL EXPORT ROUTE — 2026-09-07

- The record-sharded large-batch candidate was tested on one complete legacy
  512-record chunk. It took `53.85048` seconds (`9.50781` images/second) and
  failed exact condition equivalence beginning at record
  `foundir_train_0000017`. The warmed legacy exporter was faster at roughly
  42--44 seconds per 512-record chunk. Never activate the rejected optimized
  engine or treat its intentional benchmark exit as a training crash.
- Further acceleration must preserve the legacy `generate_outputs` call
  exactly. The approved route is chunk-level multi-instance partitioning:
  eight mutually exclusive partitions, with every partition using its own
  eight-rank legacy Qwen process. No image/question batching, prompt, decoder,
  crop, weight, or condition conversion changes.
- On each of two homogeneous eight-H200 instances, four partition workers may
  run concurrently because the measured legacy footprint is about 19.8 GiB
  per GPU and four copies remain well within 143.8 GiB. The H200 group runner
  rejects other hardware or insufficient memory. H100 remains supported by
  the original one-worker main route; do not use the four-worker H200 launcher
  on H100.
- Partition workers never publish the final combined CSV and never enter FLUX
  training. After all eight partition-pass markers exist, restart exactly one
  normal v3 main pipeline: it verifies all immutable chunks, publishes the
  combined conditions, builds the prompt cache, and continues FLUX training.
  The old unpartitioned exporter must be stopped before partition workers
  start so it cannot race or redundantly recompute their chunks.
