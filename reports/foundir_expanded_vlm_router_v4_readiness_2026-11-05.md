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

# V4：一次有边界的扩大训练＋五类真实恢复评价

## 现在就推进完整实验，不再排一串小资格试验

用户已授权execute扩大实验。仍坚持PNG右图：图像路由分数＋VLM token辅助修正，
只训练VLM修正头；五个已经训练好的FLUX专家、Qwen、VAE和image-only图像头全部冻结。
image-only只是冻结分支/评价对照，不是取代VLM的最终方案，也不另训image-only挡住推进。
不加CNN、Restormer、RGB refiner、energy、外部质量模型、GT推理门控或额外备用专家。

旧V3与CPU审计均已结束，不再运行：bfcaef5审计58秒，训练正确率97.15→98.55%，
校准91.5→90.5%；训练误改率.566→.479%，校准3.152→4.011%。支持泛化差距，
不支持继续在原2560固定裁图上盲目加步数；不证明唯一根因已找到。

本轮完整链路已经实现：prepare→四节点特征→CPU6400步及固定选型→四节点恢复→CPU汇总。
不插入新的25步/小规模资格训练，不因为某个类别分类少对一张而停止预定画质评价。
工程错误/非有限/文件身份错误/超预算仍须停，不能跳过工程安全检查来追求“完成”。

## 固定模型与训练配方

图像分支加载fresh_ce400_matched_v1真正image_only step400，参数冻结；不使用联合模型的内部弱分支。
VLM从fresh零输出初始化，fresh Adam，不继续V3 step800或旧Adam。
新修正头输入：Qwen视觉merger均值特征3584维，经过LayerNorm；加上原图像路由五个分数，
分数减自身五维均值后固定除以5。拼接→Linear256→SiLU→Linear5，最后一层零初始化。
最终分数仍是原图像分数＋修正分数，图像分数给修正头的路径detach。
这只是让修正头看见“它在修正什么”；不是新增GT信息，也不保证泛化问题会消失。
不是VLM生成退化文字或等级标签，仍是原视觉tokens特征。

6400 optimizer steps，global batch32，seed2026110501，AdamW lr.001/wd.01，label smoothing .05，
grad clip1。沿用V3的正确路由保护hinge（余量上限.2，权重1），不再扫系数/换loss/加调度器。
类别均衡，各类40960次曝光；类内均匀轮流选来源，再轮流选择该来源的不同裁图。
数据/架构/seed同时改变，因此本轮是整体改进验证，不是单因素VLM因果消融。
不宣称CPU路由训练与全模型训练一样耗算力；GPU用于一次特征提取和真正恢复评价。

## 数据扩大多少：真实清单已核对，不把裁图当新来源

原fit2560裁图/跨类别2335不同来源；新fit13917裁图/跨类别4192不同来源：

| 类别 | 来源数（该类） | 512裁图 | 每裁图曝光 |
| --- | ---: | ---: | ---: |
| Blur | 1024 | 3072 | 13–14 |
| Noise | 1024 | 3072 | 13–14 |
| Haze | 1024 | 3072 | 13–14 |
| Low-light | 543 | 1629 | 25–26 |
| Rain | 1024 | 3072 | 13–14 |

总4639个“来源×退化类型”，不是4639个全不同clean来源；跨类别来源数实际4192。
每个选定记录最多3个不同位置的512裁图；512×512原图只能提供1个位置，不复制充数。
本次真实清单全部选定记录有3个有效位置，原裁图保留，新增两个位置按固定哈希确定。
保留旧2560训练与400校准缓存，合计复用2960；新提取11457份，含新预留100图。

Low-light不能凑成1024来源。旧“剩37”是排除所有旧角色（也含其他类fit来源）的数目；
本轮允许训练角色跨类别共享来源，仍严格禁止跨fit/calibration/confirmation，因此最终合格543，
不是把旧验证来源挪回训练，也不是违反先前预留。类别均衡导致Low-light每来源重复更多，明确报告。
这还不是百万级训练，增加的是合理可用的来源及裁图变化，不承诺规模必然解决问题。

三组source/GT SHA/LQ SHA全隔离；旧400 holdout和manifest全部nontrain身份排除。
旧40开发图亦由nontrain排除，不回灌。保留原400校准集，只用于预定选型。
确认集保持此前预留的每类20/共100不同来源，选择清单不变；新图特征可以预提取，
但CPU训练入口在写出不可变selection之前禁止加载confirmation特征。
只使用现有训练collection；官方/封存/internal/parent-test/final1859像素都不读。
来源分离针对本路由训练/已有路由检查，不保证FLUX/Qwen历史预训练没见过。

展开清单SHA：568ffd35a6cad695a95f1d86f90a2826f606e596a10f4b5b11f58351e64d6432。
本地已复现原680/V2清单与预留100清单、核对完整6400步采样；无图像解码。
目的地prepare必须复现此SHA，不能临时替换缺失图片。

## 选一次模型，完整评价，不再用分类门槛拦截

1600/3200/4800/6400评同一400校准集，按最低CE选一次，平局最早；不看确认结果选模型。
选择先落盘，再加载新100图特征，只计算一次三种模型预测：

1. 原冻结图像路由（不另训）。
2. 旧V3在校准选出的step800（失败诊断参考，不冒称已批准部署）。
3. 新V4条件VLM路由。

第四组为已知类别专家参考，不是可部署模型，也不是GT挑最佳图像。三个模型均只用LQ特征推理。
固定预测top1进入原五专家内部bank：同图同seed同crop同专家只生成一次，各组复用。
不按PSNR/LPIPS挑输出、不运行多个候选再用GT选最佳，不改变alpha/量化/专家权重。
保持Blur .75、Noise .30、Haze1、Lowlight .925、Rain1及各自原双uint8/单uint8口径。
四类原50步；Rain原50步schedule第26次调用后endpoint读出，不让Rain多跑或把其他类截断。
GT图像在输出生成后才加载计分；只有第四组参考在事先计划里使用类别标签，清楚标注。
固定100图×4组，去重后100–400个实际render，完整每图/每类PSNR与LPIPS及成本都报告。
本轮仍为五类singleton预测top1内部选择，不宣称soft mixture、逐token动态路由或double/triple完成。

## 怎么判断，不事后改标准

预先分别报告：

- 新V4相对image-only和旧V3的同图总体及各类PSNR/LPIPS差。
  “总体双指标改善”要求对两组都PSNR不降、LPIPS不升，并且对每组至少一项严格改善。
- 同时完整报告各类退步，不能以总体提升掩盖某类严重下降，也不把它自动包装为五类完成。
- 保留原数值目标：Blur≥25.9/≤.18，Noise≥38.9/≤.10，Haze≥22.1/≤.19，
  Lowlight≥18.9/≤.28，Rain≥34.2/≤.08。

这些数值在新100张512裁图上的状态只作本轮报告，不等于历史完整开发集/官方原图SoW通过。
旧官方门槛未改，但官方集合与其原图协议没有在本轮执行；不能拿100图替代完整官方结果。
分类只作解释，不再要求“每类正确数不能少一张”才获准做恢复评价。
100张/每类20不构成充分的论文泛化或独立VLM消融；本轮结束后再决定更大确认/消融，不自动追加。

## 执行与预算：一套实验，按依赖交接，不额外申请小资格任务

唯一执行根：/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。
Python：该根.conda/gpu-train-stable/bin/python。主代理无此挂载，代码推送不等于GPU已启动。
四节点每节点两张H200，global shard=2×node+local；只独立特征/推理分片，无跨节点NCCL/master IP。
当前旧提取与bank实测峰值约137–138GiB，GPU启动器要求每卡至少139GiB且H200。
80GB H100本入口明确拒绝；不悄改精度/offload/裁图冒称兼容，不要求用户盲目再试OOM。

CPU代理安全同步GitHub保留本地修改/大资产后，依次协调以下已授权阶段：

1. CPU预检：`bash smoke/run_foundir_expanded_vlm_router_v4.sh prepare`。
   不要绕过wrapper直接运行.py，否则阶段耗时账不完整。通过后给用户四个GPU节点对应命令。
2. 每个节点node=0/1/2/3各运行一次：
   `bash smoke/start_foundir_expanded_vlm_router_v4_tmux.sh features <node>`。
   全8 worker marker和四节点status0后，释放GPU。
3. CPU代理自行运行：`bash smoke/run_foundir_expanded_vlm_router_v4.sh train`。
   如需脱离终端可用`bash smoke/start_foundir_expanded_vlm_router_v4_tmux.sh train`。
   等6400步、selection、predictions/render_plan和TRAIN_VERIFIED/status0；不因分类退步要求额外用户批准。
4. CPU代理给用户同四节点的恢复命令：
   `bash smoke/start_foundir_expanded_vlm_router_v4_tmux.sh render <node>`。
   不同GPU节点的同一node编号运行栈分别锁定在本阶段receipt；阶段间可更换符合签名的H200节点。
5. 全8渲染worker与四节点status0后，CPU代理运行：
   `bash smoke/run_foundir_expanded_vlm_router_v4.sh finish`。
   收集所有成功/失败attempt成本、复核图像文件SHA与原生协议，推最终小报告。

CPU代理只负责安全同步、执行既定阶段、回传小报告和GPU命令；不自行改配方/选不同模型或扩大预算。
中间训练报告可先push，主代理查看不构成新的分类停机门槛；工程错误或非有限仍先回报处理。

| 阶段 | 资源 | 粗估耗时 | 累计硬cap |
| --- | --- | --- | --- |
| prepare | CPU8线程 | 数十秒至数分钟 | 10分钟 |
| 新11457缓存 | 四节点×2H200 | 30–60分钟 | 每节点90分钟，共12使用GPU-hours |
| 6400步路由训练 | CPU8线程 | 20–45分钟 | 1小时，0GPU-hours |
| 100图四组恢复 | 四节点×2H200 | 15–40分钟，依去重数 | 每节点45分钟，共6使用GPU-hours |
| finish | CPU | 数分钟 | 10分钟 |

这些是参考旧吞吐的估算，不是本轮实测。GPU总cap18使用GPU-hours，另超时最多30秒清理/attempt，
失败成本累计；缺结束账先调查，不能换ID清零或自动扩预算。若独占更大整机需另报整机成本。
不把四节点普通小时相加当用户等待时间，也不把GPU-hours说成普通小时。

每份特征/每张恢复图原子落盘，重启跳过SHA通过的已完成内容；孤立文件保留不覆盖。
每optimizer step保存VLM/Adam/RNG/采样位置，0..6400共6401份；约80–100GiB状态与少量新缓存，
prepare要求150GiB空闲、训练首启120GiB余量。所有旧状态保留；同命令精确恢复同CPU运行栈。
最终verify检查全部marker连续、step0/选中/最终checkpoint文件SHA、所有恢复文件SHA和协议，
不是声称重读6401份全量权重。新run/旧缓存producer身份分别绑定，不修改历史绑定代码。

训练报告：reports/foundir_expanded_vlm_router_v4_training_2026-11-05.json。
最终报告：reports/foundir_expanded_vlm_router_v4_terminal_2026-11-05.json。
状态根：checkpoints/foundir_expanded_vlm_router_v4/expanded_conditional_v4。
日志：logs/foundir_expanded_vlm_router_v4_<phase或phase_nodeN>.log。
完整结束须EXPANDED_V4_RESULT_VERIFIED及finish阶段EXPANDED_V4_END status=0。
只push训练/最终小JSON与耗时/退出简报；不push权重、特征、恢复图、逐图缓存或live logs。

## 验证边界

113项CPU回归通过（13项新V4测试及100项既有相关测试）；Python编译、stdlib-only入口和shell语法通过。
真实manifest元数据选择与6400步完整采样计数已在本地核对，未解码图像。
本地CPU测试覆盖新五分数条件输入、零初始化、冻结图像、训练/Adam/RNG精确恢复、
多来源/多裁图隔离、旧缓存继承、稀缺类/小图片不伪造数据、先选模型后读确认特征、
固定预测与四组去重、全部八分片、画质门槛与官方声明隔离、失败成本记账、完整小维度合成链路。
没有新的GPU或目的地真实训练结果，合成测试不当科学成绩。新增GPU入口仍需现场启动实测。
