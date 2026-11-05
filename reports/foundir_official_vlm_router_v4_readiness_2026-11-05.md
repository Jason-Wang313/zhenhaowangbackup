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

四节点当前入口（在四台终端分别执行route 0/1/2/3）：
`FOUNDIR_OFFICIAL_NNODES=4 bash smoke/start_foundir_official_vlm_router_v4_tmux.sh route <节点编号>`。
CPU prepare/plan也显式传`FOUNDIR_OFFICIAL_NNODES=4`；四节点日志为`logs/foundir_official_vlm_router_v4_4x2_<阶段>_node<编号>.log`（CPU阶段无node）；attempt与缓存独立绑定四节点run。
四节点普通时间外推按实际GPU-hours/8，不能保证精确两倍加速；route阶段不是恢复评价。

# 冻结V4的官方600张评价入口

## 当前状态与长期规则

用户已授权官方评价，并明确要求以后测试和效果验收只用FoundIR官方测试集。
不得再用自建100图、开发集、训练裁图probe、内部封存集等做测试/对照/效果验收或
作为官方评价前的质量门槛。历史产物保留，不重跑、不改名成官方。
训练仍使用训练数据；CPU合成张量单测可用于检查程序，不能宣称恢复指标。
官方GT不得进入训练loss/梯度或部署选路，已使用多轮的官方集不称盲测。

这次不是新训练，也不选择新checkpoint。固定已选V4 step4800：
`383d40b63e31d79283ec5a33193413fee914a5082a8888c5c918737ccdd6d971`。
原image-only step400、Qwen、VAE和五个已有FLUX专家全部冻结。
旧官方入口包含Restormer及旧Qwen路由，不能直接启动或悄悄借用其恢复输出。
新入口仅复用当前V4的纯FLUX专家bank、真实Qwen视觉token和官方评分协议。

主代理没有新工作区挂载或GPU；本地未启动官方任务，没有新PSNR/LPIPS结果。
129项CPU合成/工程回归通过，其中13项新官方评价测试；Python编译、shell语法、
标准库CLI导入通过。没有以这些测试冒称GPU或原图指标验收。

## 精确范围和原图推理

官方五类600对：Blur150、Noise100、Haze200、Rain100、Lowlight50。
清单 `manifests/foundir_official_sow600_qwen_v1.csv`，SHA256：
`dfe5a05ba13f941ac57734e2a96eb21048ad463f4248f306d2fb2d9c6d48866e`。
只读旧迁移源的清单元数据确认了此哈希及尺寸；没有在旧工作区执行模型/任务或读测试像素。
不是全部20类1500对，不自动扩展混合退化。

同图三组：原image-only、本轮conditional V4、已知类别专家参考。
第三组只用于区分路由损失和正确类别专家的恢复水平，不是可部署结果、不是GT选最佳图。
所有部署路由仅输入LQ特征；不输入路径中的类别、标签、GT或质量分数。

原图不能硬裁到512后就当官方评价。预先固定如下整图推理，不看官方分数调规则：

- 512块、128重叠、末块贴边；小图按轴reflect到512，长度1的轴用edge。
- 每块按原V4计算图像分数和VLM修正分数；各自对全图所有块的分数取算术平均，
  再做一次全图top1选择。不增加网络、不逐块选不同专家、不加入soft mixture。
  这是由原512训练输入扩展到整图的明确规则，不声称与旧整图Qwen路由相同。
- 三组使用相同每块seed，seed仅由固定数值2026110502、record ID及x/y确定。
- 五专家仍保留原权重、prompt、alpha和量化；四类50调用，Rain50步时间表第26调用endpoint。
  VAE沿用已实测V4原生FP32卷积/attention query256，不额外量化/offload。
- 相同图片、相同专家的结果只生成一次。每块先保存原生uint8结果，再用sqrt-Hann
  overlap平均、权重floor1e-3，最终round uint8 PNG，裁掉padding后严格等于原图尺寸。
  这是继承当前专家量化的拼接策略，不宣称逐像素等于旧混合系统的浮点拼接。
- 全部三组输出保存后，才读取该图GT评分。完整原图RGB，外部不resize/crop/shave，
  `pyiqa==0.1.16`的默认PSNR/LPIPS/SSIM，与已读取的FoundIR `cal_metrics.py`调用一致。
  metric模型state哈希在CPU准备时冻结，GPU再次核验；版本/权重不匹配报错，不自动安装替换。
- 分类和恢复报告按每类及600图等权平均，不把五类宏平均混作600图均值。
  每类PSNR/LPIPS目标保持原SoW；不把本轮VLM增益混成旧energy双目标。

## 关键：真实工作量不能沿用小裁图预算

元数据确认最少19,460块；四worker各4915/4915/4815/4815块。
分类分布的块数：Blur2250、Noise4550、Haze6220、Lowlight1760、Rain4680。
最大原图3984×2652。三组去重后的实际恢复量介于19,460与58,380块。

上一轮117块恢复用了0.6605556实际GPU-hours，按该速度外推，最少块数约110
GPU-hours；四卡约27.5普通小时。三组路由不同时还会增加。加载开销、原图指标、
新硬件和数据I/O会影响速度，这不是承诺，也不是已消耗成本。
因此不能把上一轮2.47 GPU-hours或18 GPU-hours上限套成这轮完整评价的承诺。
历史V4成本2.473888889 GPU-hours继续保留，不被新run清零。

科学方案不新增小试验/质量gate，但工程分为：CPU prepare → 官方LQ特征/选路 →
固定完整render plan和精确去重数量 → **用户明确批准恢复预算** → 原图恢复/评分 → finish。
route不读取官方GT像素，也不运行恢复。分阶段是核对大额算力成本，不是看效果挑配方。
未批准恢复预算时，render入口明确拒绝；CPU代理不得凭本readiness自行填120或其它预算。

## 资源、保存和入口

唯一执行根：
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
Python：该目录下`.conda/gpu-train-stable/bin/python`，不能用旧HDD3执行目录或系统Python替代。
默认两节点×两张同节点同型号H100/H200>=75GiB，四独立worker、无跨节点NCCL。
节点型号可以不同，但精度/软件策略签名必须匹配；每个节点内同型号。
继承V4四H100特征和H200裁图恢复证据，但新官方整图metrics的CUDA显存仍待实测。
同run严格绑定节点runtime，换硬件/驱动需显式迁移，不静默改变或重用不同数值路径。

新目录：`checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1`。
每块特征、原生uint8、整图PNG、决策、逐图指标均独立原子落盘和校验；相同命令跳过已核验项。
半写临时文件不覆盖已完成产物；孤立attempt账目需要核对后才能恢复，失败耗时仍计预算。
prepare要求100GiB空闲。旧检查点/缓存/数据/输出全保留。

1. CPU代理安全同步后：
   `bash smoke/run_foundir_official_vlm_router_v4.sh prepare`
2. prepare通过，CPU代理给用户两个节点各自的命令：
   `bash smoke/start_foundir_official_vlm_router_v4_tmux.sh route 0` / `route 1`
3. 四个ROUTE_COMPLETE、两节点END status0后，CPU代理：
   `bash smoke/run_foundir_official_vlm_router_v4.sh plan`
   推送小preflight和route_plan报告，向用户说明精确工作量，不在此处看GT或挑方案。
4. 仅在用户明确批准具体恢复GPU-hours之后，CPU代理记录批准：
   `.conda/gpu-train-stable/bin/python -S smoke/evaluate_foundir_official_vlm_router_v4.py authorize_render --gpu-hours <批准数值> --approval-note '<用户批准的原意和时间>'`
   这条命令不代表本文件已授权任何数值。批准文件绑定当前render plan，不能覆盖修改。
5. CPU代理提供两个节点`...tmux.sh render 0` / `render 1`。
6. 四worker、两节点END status0后：
   `bash smoke/run_foundir_official_vlm_router_v4.sh finish`

route累计cap每节点10800秒，共12使用GPU-hours；不是预计一定耗满12小时。
render上限由批准数值换算到两节点，各节点墙钟秒数=批准GPU-hours×3600/4，
失败尝试累计，不自动扩大。CPU prepare/plan各1800秒，finish900秒；清理最多30秒。
日志在`logs/foundir_official_vlm_router_v4_<阶段>.log`追加；不推实时日志或大文件。
最终必须有`OFFICIAL_VLM_V4_RESULT_VERIFIED`和finish `OFFICIAL_VLM_V4_END status=0`。
科学失败也要完整报告；工程退出0不等于SoW全部通过。

## 测试证据与禁止误读

新13项测试覆盖：官方600身份/路径白名单、四分片完整性、边界覆盖、极小/奇数/大尺寸
identity拼接、缺块和dtype拒绝、三组同seed、平均分数与标签隔离、argmax篡改拒绝、
正确的600图加权汇总、三组去重/GT读取顺序/断点/PNG损坏拒绝、显式预算、失败成本、
全600条路由计划绑定及特征tensor篡改拒绝。
其余116项为既有合成CPU回归。全部不读取其它真实测试集，没有新的恢复效果结论。

新CPU代理只负责同步、现场资产预检、按上述授权协调阶段、推送小报告。
不能自动扩大scope、重训/换checkpoint/换专家、回到旧100图，不能为省时间改裁图或评分。
