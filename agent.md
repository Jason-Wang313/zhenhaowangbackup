## 当前授权：两张严重错选的CPU原因核对；不启动新训练 — 2026-11-04

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
