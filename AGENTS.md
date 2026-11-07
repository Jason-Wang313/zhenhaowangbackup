## Blur5632只读分解：索引6空间误差退步，训练继续 — 2026-11-07

同8训练整图3712→5632仅索引6双退步：PSNR-3.485338dB、LPIPS坏26.910582%。MSE增.003795153，整体RGB均值偏差分量反而减.000007311、剩余空间误差增.003802464；固定输出增量与GT残差cosine-.106263。不是整体色偏即可解释，也不能只说方向正确但幅度大；这是快照输出差，不是optimizer梯度。原因未隔离，不推断遗忘/过拟合/代码故障。
索引7的MSE改善约75.36%来自整体RGB均值偏差下降，不能把全部平均收益等同细节去模糊；索引3/5主要改善空间误差。已看全部8图480px宽联系表，骑车人物、树林和文字仍有模糊/拖影；不是全分辨率视觉验收。PNG仅留本地。报告foundir_blur_fullgrid_1x2_v1_snapshot5632_decomposition_2026-11-07.{json,md}。
成功只读分析6.860741秒，GPU0。初次FP64通道累计误差1.89777e-13触发1e-14守卫，改用uint8残差精确int64求和，保留原容差，最大误差2.78e-17；GPU输出/指标/源码不变，失败已记录。GT只解释固定训练输出，不调阈值/选图/部署。
当前node3双H100固定9472训练继续，monitor908410/upload908408；查实时steps/心跳，勿重启或改绑定bcc3b6d源码。5632训练均值23.239442/.175716仍非官方成功；官方仍21.948140/.202188未过25.9/.18。下一7552/9472快照→唯一最终官方150→核验/推报告；仅Blur，全部资产/energy/失败和成本保留，全栈自主授权继续。

## Blur5632训练整图均值继续改善，1张相邻快照双退步；官方仍未通过 — 2026-11-07

5632固定同8训练整图完成并独立核验128新PNG SHA、身份/模型/receipt绑定和均值；原始step5632报告7c8857f已由持久上传器推送。1792→3712→5632均值20.575352/.240456→22.192278/.190853→23.239442/.175716。相对1792，PSNR+2.664089dB、LPIPS降26.923744%，8/8图两项改善；相对3712，+1.047164dB/降7.930813%，7/8两项改善。
旧PSNR退步来源索引3/5/7均回升；新回退为索引6 foundir_train_0053870（第7张），3712→5632 PSNR25.110912→21.625574、LPIPS.113034→.143452。仍优于1792，但不能称每张稳定改善；原因未隔离，不推断遗忘/过拟合/代码故障。报告foundir_blur_fullgrid_1x2_v1_snapshot5632_2026-11-07.{json,md}完整保留8图，不选图/中途权重。
仅training_fit_only，.175716不是官方LPIPS过关。最新官方仍21.948140/.202188，未过25.9/.18；只攻Blur。固定9472终点不变，下次7552/9472训练快照，之后唯一最终模型完整官方150→核验/跨轮比较/报告。5632真实checkpoint/100Adam/2rank RNG/采样核验通过；运行source bcc3b6d不改。
当前仍node3(10.244.90.116)两H100真实训练，monitor908410/upload908408存活；接手查实时foundir_blur_fullgrid_1x2_v1及队列round20_blur_fullgrid_resume_1x2_train_v1_node3，不因本历史快照重启/重复提交。旧四卡失败、3688孤立文件、energy及全部资产保留。累计成本快照见foundir_blur_queue_cost_snapshot_2026-11-07，范围是当前队列明确Blur作业，非全部项目成本。全栈自主执行授权继续。

## Blur3712只读误差分解：3张PSNR退步，双卡训练继续 — 2026-11-07

同8训练整图8/8 LPIPS下降、5/8 PSNR提高。另3张PSNR下降.305/.456/.199dB；固定输出变化与GT残差cosine .002684/.070964/-.002913，方向对齐弱，不能称方向已学会。整体RGB均值偏移仅解释前两张MSE退步约6.6%/<.1%，第三张整体色偏分量改善但空间误差增加；不是统一亮暗偏移就能解释。训练GT仅用于固定输出描述，不用解析GT缩放选图或部署。已看全部8图缩略联系表，树林/汽车背景仍明显拖影。
7.013576秒只读CPU分析，GPU0，原PNG/GT SHA及MSE分解/PSNR原eps口径通过；首次分析漏eps导致检查失败已如实记录，原GPU指标不变。报告foundir_blur_fullgrid_1x2_v1_snapshot3712_decomposition_2026-11-07.{json,md}，图片仅留共享目录。
当前仍node3双H100固定3687→9472，monitor908410/upload908408，查新foundir_blur_fullgrid_1x2_v1和共享queue实际step，勿改绑定源码/重复提交/重启旧四卡。3712训练均值22.192278/.190853只是training_fit_only；官方仍21.948140/.202188未过25.9/.18。下一固定5632/7552/9472→唯一官方150→核验/报告/跨轮比较，所有资产/energy保留、仅Blur。全栈自主授权持续。

## Blur双H100已恢复；3712同8训练整图均值改善，官方仍未通过 — 2026-11-07

源码bcc3b6d已推送且10份GPU/控制源码SHA核对；node3(10.244.90.116)实际领取round20_blur_fullgrid_resume_1x2_train_v1_node3。两rank NCCL通过、完整3687模型/100Adam/历史/采样/RNG映射逐值核验通过（step3688报告5e9a7a5已推）；3689两卡真实CUDA重载通过。原15360缓存复用，global8/每步原8样本/原loss/lr/9472终点不变；world4→2不称bitwise，不恢复失联node1的旧任务。
3712第一遍完整网格快照已完成：同8训练整图20.575352/.240456→22.192278/.190853，PSNR+1.616925dB、LPIPS降20.628976%，5/8图两项都改善。独立核验128份PNG SHA及均值；仅training_fit_only、非官方/泛化通过，不改固定9472。最新官方仍21.948140/.202188未过25.9/.18。
2026-11-07T07:08:20.529213+00:00快照已保存3731，新两卡含保存约4.813秒/步，训练暂余7.68小时，剩余训练快照及官方150另计。接手查实时新目录checkpoints/foundir_blur_fullgrid_1x2_v1；monitor908410/upload908408，身份daemons.json，均实际存活，负责5632/7552/9472→唯一官方150→CPU核验/跨轮比较/小报告。新观察器已内置比较，不重启旧monitor1877938/comparison2052208。GPU正在运行，不改绑定源码或重复提交。
旧node1终态缺失，原四卡轮成本至少8.197129152GPU-hours，历史和全部资产/energy保留；当前恢复轮费用另计。只有node3健康，未扩大训练步数；全栈自主修复授权持续，仅Blur。详见reports/foundir_blur_fullgrid_1x2_v1_snapshot3712_2026-11-07.{json,md}及readiness。

## Blur两卡续训已实现并CPU预检，待发布后派发 — 2026-11-07

node1仍失联，仅node3(10.244.90.116)两H100心跳新鲜。新foundir_blur_fullgrid_1x2_v1从原完整3687模型/Adam/历史/采样位置恢复，原rank2/3 RNG映射到新0/1；world4→2不称bitwise。每卡4样本、global8/每步相同8样本/原loss/lr/1024来源全15视图四遍不变，仍固定9472，仅剩5785步。旧15360缓存直接复用，全部旧目录/energy/失败记录及3688孤立tensor保留。
7项CPU测试和语法通过，真实父状态/缓存清单预检5.742887秒；官方完整150元数据计划通过。新队列round20_blur_fullgrid_resume_1x2_{train,official}_v1_node3，源码发布后由watch_foundir_blur_fullgrid_1x2_v1_cpu.py monitor/upload接续；PID将写新目录daemons.json，接手查真实状态，不重复启动。3688 CPU核验真实迁移，3689 CUDA重载；3712/5632/7552/9472固定同8训练全图仅描述，最终9472完整官方150→核验/跨轮比较/小报告自动推送。
当前尚未派发GPU；粗估剩余8–11小时/16–22GPU-hours，须按实测更新，无累计cap但不增加9472终点。旧round20至少8.197129152GPU-hours且node1费用不完整，不能清零。仅Blur，官方仍21.948140/.202188未过25.9/.18。全栈自主修复授权持续。详见reports/foundir_blur_fullgrid_1x2_v1_readiness_2026-11-07.{json,md}。

## Blur round20主连接中断；完整3687已核验，待恢复运行 — 2026-11-07

node1最后心跳2026-11-07T06:43:57.221713+00:00，node3 TCPStore EOF/Broken pipe；实例/系统根因未确认，不能认定GPU显存或模型问题。精确取消round20训练对子job，node3 status=-15且worker已idle；node1缺退出receipt，不伪造。monitor1877938和comparison2052208已退出，upload已完成failed发布b2916885fa33b5244e58c3d54b286c7bf6d1dc8f；下方“训练仍正常”均为历史快照，接手必须查实时状态。
最后完整3687实际SHA/100Adam/四rank RNG/next_global_slot29496已CPU核验，保留新增1895步。3688孤立tensor保留但无receipt，不作为完成步。缓存15360完整保留，3712快照/新官方结果尚无，剩余目标仍9472（5785步）。恢复不从1792重训、不重缓存。原四rank须先恢复节点/处理新attempt运行身份；只剩node3时先实现验证world4→2的global8等价迁移，不能冒称bitwise。已询问node1平台状态，尚未提交恢复GPU。
本轮费用至少8.197129152GPU-hours，node1终态缺失，不能报完整成本。官方仍21.948140/.202188未过25.9/.18，仅Blur、全栈修复授权持续，保留全部旧资产/energy。见reports/foundir_blur_fullgrid_v1_master_loss_2026-11-07.{json,md}和step3687_state核验。

## Blur1024来源完整网格四遍终态 — 2026-11-07

paired GPU job failed: train；保留全部资产，无自动重试。
只用官方作质量门槛，Blur未过不推进其它类别。完整自主修复授权继续；本轮不自动追加旧步数。见reports/foundir_blur_fullgrid_v1_failed_2026-11-07.{json,md}。全部资产保留。

## Blur四H100训练已开始；官方仅小幅改善，尚未通过 — 2026-11-07

round20完整缓存15360/15360，旧4096只复用、新11264生成，两缓存job退出0、共1.451627582 GPU-hours。正式1792父模型/完整Adam/四rank RNG/历史/采样逐值恢复通过；初始120块和8训练全图PNG逐一复现，1794四rank真实CUDA重载通过。详见reports/foundir_blur_fullgrid_v1_parent_restore_2026-11-07.json。
2026-11-07T05:11:31.798831+00:00快照：node1/node3四H100真实训练step1911，新增119/7680，目标9472；最近约3.100秒/步，训练暂余6.51小时，官方另8–12分钟。两个worker心跳及monitor1877938/upload1877939/comparison2052208真实cmdline核验正常；步骤/ETA仅快照，接手查实时记录，不重启或改绑定源码。
完整官方仍1792的21.948140/.202188，未过25.9/.18；相对原FLUX仅+.401dB/LPIPS改善约.21%，不能说已修复。单训练图25.692→30.010只支持训练能力，不代表官方。当前固定1024来源全部15视图×4遍，正式父1792→9472，不用单图诊断权重，最终唯一官方150；3712/5632/7552/9472快照仅training_fit_only，终态自动核验/推报告/跨轮比较。只攻Blur，全栈自主授权持续，保留全部旧资产/energy。进展见reports/foundir_blur_fullgrid_v1_progress_assessment_2026-11-07.{json,md}。

## Blur终态跨轮比较观察器已运行；缓存链继续 — 2026-11-07

CPU比较源码3613da5已推送并核验，3项比较契约测试通过。独立postprocess PID2052208实测存活，身份在checkpoints/foundir_blur_fullgrid_v1/comparison_daemon.json，日志comparison.log；只等主链terminal发布、两官方job退出0后运行同150图固定200/1280/1792/9472跨轮比较并推报告，不派发GPU。终态后检查published/official_comparison.json，不要重复启动该观察器。
主monitor1877938/upload1877939持续控制round20 node1/node3四H100缓存→固定1792至9472→完整官方150→核验/小报告。当前快照缓存9697/15360，四rank输出持续增长、两worker心跳正常；接手查实时状态，不把本旧计数当最新进度，不重启或改绑定源码。
官方仍21.948140/.202188未过25.9/.18，仅Blur。全部旧资产/energy保留，训练和单图诊断不作为新质量gate，完整自主修复授权持续。详见reports/foundir_blur_fullgrid_v1_postprocess_live_2026-11-07.json。

## Blur完整网格终态将自动补充同官方150跨轮比较 — 2026-11-07

当前round20 node1/node3四H100缓存链正常继续，GPU绑定源码/配方不变；不重新提交任务。新增独立CPU入口compare_foundir_blur_fullgrid_official_v1_cpu.py，待固定9472完整官方150终态后，与原FLUX、200、1280、1792在同一150图/无GT路由/评分口径上比较，独立重算均值、逐图双好/双差和双门槛；禁止混拼最好指标或GT选输出。
3项标准库测试通过，覆盖多模型单项最好不能拼联合通过、缺失/重复/GT身份变更、均值与基线篡改。新postprocess_foundir_blur_fullgrid_v1_cpu.py仅等待主控制器发布terminal并确认两官方job退出0，再运行比较和推小报告；不生成图片、不训练。源码发布后将独立后台启动，身份写foundir_blur_fullgrid_v1/comparison_daemon.json。当前官方9472结果尚不存在，不能提前运行比较或声称改善。
主monitor1877938/upload1877939仍负责固定缓存→1792至9472→官方150→终态回传，身份见daemons.json。正式候选仍旧1792，官方21.948140/.202188未过25.9/.18；仅Blur继续，全部旧资产/energy保留。接手查实时queue/缓存/step，不改绑定源码或重复启动。

## Blur完整网格链已真实启动：四rank新缓存及输入核验通过 — 2026-11-07

源码da011aca已发布，全部10份新GPU/控制源码远端SHA逐一核验；queued报告ecfa36fa已推。物理node1/node3四H100实际领取round20_blur_fullgrid_cache_v1_node1/node3，四rank均已生成新缓存，未见早期异常。快照1794026615.1246603：4951/15360，其中旧4096复用、新855；每rank首份新tensor实际SHA/FP32有限/schema/LQ像素逐值一致核验通过（CPU1.705569秒）。当前尚未进入训练，接手查最新cache/step/queue，不重启或改绑定源码。
持久monitor PID1877938、upload PID1877939，身份在foundir_blur_fullgrid_v1/daemons.json；两worker心跳新鲜。固定缓存→CPU完整性→正式1792完整四rank状态恢复至9472→唯一最终官方150→核验/小报告。1793及每遍3712/5632/7552/9472回传；训练初始同8全图/120块必须复现，1794 CUDA重载。无需用户再输入GPU命令。
完整1024来源×15视图×4遍，global8/模型/loss/lr不变，诊断单图权重绝不使用。官方仍21.948140/.202188未过25.9/.18，仅Blur；快照拟合不是质量通过。所有旧资产/energy保留，异常停链不自动扩步/盲重试。源码/readiness与live_cache报告含固定范围和粗估7–9小时，按新轮实测更新；完整自主修复授权持续。

## Blur完整部署网格四遍候选已实现并CPU预检 — 2026-11-07

新round20_blur_fullgrid_{cache,train,official}_v1_node{1,3}：原1024训练来源、每来源全部15网格视图，共15360条；4遍无放回、每视图4次，global8。正式1792完整模型/Adam/四rank RNG→固定9472（新增7680），不使用单图诊断权重；原架构/loss/lr/CPU浮点归一化不变。覆盖和曝光共同变化，不声称单因素因果或保证通过。
15项CPU检查通过：真实四进程global8 Adam与同world精确恢复、采样完整性、源/坐标缓存复用、两独立worker信封/故障隔离、训练快照及完整官方150汇总。目的地预检112.179765秒：父状态/2048训练文件/4096旧缓存实际SHA、120旧块及8全图SHA通过；旧缓存只加不可变硬链接，无字节修改。指标CPU身份核验2.748769秒。还须新生成11264参考缓存，未重跑旧缓存。
训练前先要求同前8训练图120块及拼接逐像素复现1792；每遍固定8训练全图快照仅training_fit_only。每完整步保存全部Adam/RNG/哈希；1794真实CUDA重载。最终9472唯一官方150评价，门槛25.9/.18，不挑中途官方最好；目前官方仍21.948140/.202188失败，仅Blur继续。
持久控制器smoke/watch_foundir_blur_fullgrid_v1_cpu.py monitor/upload负责缓存→完整性→训练→最终官方→CPU核验/小报告，1793及每遍3712/5632/7552/9472回传。身份将写foundir_blur_fullgrid_v1/daemons.json；当前尚待源码安全发布后启动，接手查真实队列/daemon/心跳，不重复提交，不改绑定源码。仅健康node1/node3各两H100，node0/2心跳陈旧不使用。
粗估7–9小时/28–36使用GPU-hours，按实测更新；逐步state约新增2.8TiB、另有新缓存，预检可用空间大于4TiB；无累计GPU上限但本轮固定范围，异常停链不盲重试。保留全部旧资产、energy及失败费用；全栈自主修复授权持续。详见reports/foundir_blur_fullgrid_v1_readiness_2026-11-07.md。

## Blur单来源16上下文诊断结束：可同时拟合；下一固定完整网格四遍候选 — 2026-11-07

round19固定200步结束，node3双H1001009.929472秒/.561071929GPU-hours，两rank/launcher退出0；CPU收尾12.113812秒含启动、解释2.381399秒，原始终态518ceba已推并核验三份SHA。monitor1123736正常退出，无待续本轮GPU，勿重跑。
单训练全图25.692104/.124584→30.010347/.074466；16/16视图相对起点双项改善，38对重叠预测MSE下降57.169721%，完整图方向cosine .730217→.897127。13/16优于LQ，另3右侧背景块仍PSNR低于LQ且幅度过量，不能声称no-harm/普遍修复。只证明一来源多上下文能学，官方仍21.948140/.202188未过25.9/.18，仅Blur。
下一固定候选（尚未实现/预检/提交）：原1024来源全部15部署网格视图15360条，4遍无放回=7680新增步；从原正式1792完整模型/Adam/四rank RNG到9472，绝不使用单图诊断1992权重。原架构/loss/lr/global8不变；旧4096参考缓存核验后复用，其余新根缓存；健康node1/node3共四H100。每遍同前8训练全图描述性快照，非新质量gate；唯一最终9472完整官方150，不挑中途官方最佳。
覆盖和曝光量共同改变，不称单因素因果或保证通过。粗估合计7–9小时/28–36GPU-hours，按新轮实测更新，无累计GPU上限但本轮固定范围；逐步保存完整state、终态小报告推送，全部旧资产/energy保留。先实现采样/缓存复用/完整恢复及CPU预检，再安全发布/派发，不能调用旧四视图入口。完整自主授权持续，下一代理按此推进无需再次许可。
详见reports/foundir_blur_singleframe_fit_v1_interpretation_2026-11-07.{json,md}及terminal/result。

## Blur单来源16上下文固定200步诊断已结束 — 2026-11-07

round19_blur_singleframe_fit_v1，GPU退出码0，CPU收尾0。无待续本轮GPU，勿重复；详见reports/foundir_blur_singleframe_fit_v1_terminal_2026-11-07.{json,md}。仅training_fit_only，诊断权重不得部署或seed正式训练。官方仍21.948140/.202188未过25.9/.18，继续仅Blur。接手先读本轮证据再冻结下一项，不自动追加步数；全栈自主修复授权持续，所有资产/energy保留。

## Blur单来源16上下文：新增100步仍改善，固定200继续 — 2026-11-07

round19 node3双H100仍正常训练；100步固定全图29.402836/.082272（初始25.692104/.124584，50步28.858677/.088792）；16视图均值32.058145/.078101，16/16相对起点双项改善、13/16优于LQ。单训练图能力证据，不代表跨来源或官方；已看联系表，运动边缘改善但仍有局部重影/失真，不包装视觉全修复。继续固定1992，不挑中间权重。
原monitor PID1123736继续等待终态后CPU核验/推小报告；接手查job/心跳/step，不改绑定源码或重复启动。终态后另运行只读analyze_foundir_blur_singleframe_fit_v1_cpu.py，核验所有快照/重叠预测差异并推解释。官方仍21.948140/.202188未过25.9/.18，仅Blur；诊断权重禁止部署/seed正式训练，全部旧资产/energy保留。

## Blur单来源16上下文：新增50步快照已完成，固定200继续 — 2026-11-07

round19仍由node3双H100正常训练；50步固定快照全图25.692104/.124584→28.858677/.088792，16视图均值26.521845/.122707→31.276242/.084731，13/16两项优于LQ。仅一个训练来源上的能力证据，不能当官方或跨图修复；无提前停止/挑权重，继续到固定1992。官方仍21.948140/.202188未过25.9/.18，仅Blur。
运行绑定代码不变；monitor PID1123736仍负责结束后CPU收尾/报告推送，身份见foundir_blur_singleframe_fit_v1/monitor.json，接手查实时job/心跳/step不要重启。新增只读CPU解释脚本analyze_foundir_blur_singleframe_fit_v1_cpu.py须在完整终态后运行，比较全部固定快照/16逐视图/重叠预测差异，未产生新推理或GT拟合，之后推小报告。所有资产/energy保留，诊断权重不得部署/seed正式训练。

## Blur单来源16上下文诊断已真实训练，初始逐像素复现通过 — 2026-11-07

round19_blur_singleframe_fit_v1由node3双H100实际领取运行；16参考缓存完成，初始16PNG和全图SHA逐一等同冻结1792旧输出。CPU3.273295秒核对父1792和新1792模型/完整Adam/参数顺序/历史/采样逐值相同，明确四rank RNG转两rank重新seed；实际1793 state SHA/100 Adam/参数更新通过。截至1794024565.7874634保存至1802，即10/200新增步；接手查最新step/心跳，不重启。
持久monitor PID1123736，身份在checkpoints/foundir_blur_singleframe_fit_v1/monitor.json；脚本watch_foundir_blur_singleframe_fit_v1_cpu.py等待本job结束→CPU finish→小报告和费用自动推送，无后续GPU。新增0/50/100/200快照、每完整步保存，固定1992结束；禁止修改运行绑定源码。无自动续步、官方或其它类别。
本轮仅training_fit_only，诊断权重不得部署/seed正式训练；官方仍21.948140/.202188未过25.9/.18，只攻Blur。全部旧资产/energy保留。源码9bef437已推；详见reports/foundir_blur_singleframe_fit_v1_live_2026-11-07.json。全栈自主修复授权持续，终态依据实证再决定下一方案。

## Blur单来源16上下文拟合诊断已准备 — 2026-11-07

官方仍21.948140/.202188，相比原FLUX仅+.401377dB/LPIPS降.214263%，未过25.9/.18；72/150 PSNR差于LQ，不能说稳定修复。仅Blur继续，旧round17/18不重跑。
新round19_blur_singleframe_fit_v1：原1024第一来源0034314，15部署网格块+原[985,489]裁图共16；完整1792模型/Adam复制到独立目录，四rank转两rank显式重建RNG，不称逐位等价父训练。原模型/loss/lr/global8不变，固定200新增步到1992，每视图100次曝光。新增0/50/100/200保存16视图及拼接；初始必须逐像素复现1792的16输出，逐完整步保存Adam/RNG/哈希。仅training_fit_only，无新质量gate、无官方评价，诊断权重绝不部署/seed正式训练。
6项CPU测试通过，含真实两进程global8 Adam汇总及同world精确恢复；目的地预检12.472429秒/GPU0，父状态/16输出/源图/指标SHA通过。轻任务仅node3两H100，粗估18–25分钟，按实测更新。源码安全发布后提交共享worker；接手查实际job/心跳/step，勿重复。
持久CPU观察器smoke/watch_foundir_blur_singleframe_fit_v1_cpu.py仅等待本job终态→CPU finish→小报告/耗时/退出码推送，无后续GPU。daemon身份写foundir_blur_singleframe_fit_v1/monitor.json；尚未有该文件时不要称观察器已运行。所有旧资产、energy和失败费用保留；全栈自主修复授权持续。

## Blur1792同8全图对照完成：差距缩小主要来自原裁图退步 — 2026-11-07

round18_blur_multiview_context_v1结束，node3双H10090.243148秒/.050135082GPU-hours；8图/120块/8原裁图、两rank和launcher退出0，模型/参考SHA前后不变，CPU finish2.655758秒。复用240旧/新块PNG的CPU统计7.087600秒，无新预测。无待续本轮GPU，勿重跑。
同512：独立23.727759→22.954543，完整图同位置22.499734→22.503520；差距−1.228024→−.451023，主要由原裁图退步.773216贡献，不能当上下文修复。完整8图19.635696/.235742→20.575352/.240456，PSNR+.939656但LPIPS坏约2%。新训练过32块仍9/32 PSNR差于LQ；它们仅各一次新增曝光，不足以宣判模型不可学或纯官方分布问题。
下一固定训练诊断（尚未实现/提交）：原1024列表第一个来源，全部15部署网格块+原512裁图，共16视图；复制原1792完整模型/Adam，world4→2显式重建两rank RNG；原模型/loss/lr/global8保持，固定200新增步到1992，每视图100次曝光。新增0/50/100/200记录全部16视图和完整拼接，初始复现1792，逐步保存。仅node3两卡，18–25分钟粗估；training_fit_only，无新质量gate、官方或自动后续，诊断权重绝不部署/seed正式训练。
该实验验证能否同时拟合同一图的多个上下文，不盲扩1024来源长训。实现和CPU预检/发布后自主派发，完整授权持续。官方仍21.948140/.202188未过25.9/.18，只攻Blur。所有旧资产/energy/失败费用保留。详见reports/foundir_blur_multiview_context_v1_terminal_2026-11-07.{json,md}与multiview_tile_fit_v1_CPU报告。

## Blur1792同8训练全图上下文对照已准备 — 2026-11-07

官方1792仍21.948140/.202188，25.9/.18未过；上一轮完整终态/比较/方向分析已推5fecb91。当前只隔离新裁图训练是否缩小上下文差异，不盲加步，不转其它类别。
新入口smoke/diagnose_foundir_blur_multiview_context_v1.py，输出foundir_blur_multiview_context_v1；冻结唯一1792，固定原同8训练来源，8全图/120部署块+8原裁图，共256模型调用。旧1280完整图/独立裁图结果直接复用，比较同像素与中心384、完整图；目标评分沿用相同历史数值路径，输出固定后读训练GT。training_fit_only，无新测试gate/optimizer/GT选图/阈值。
3项CPU配对检查、语法通过；目的地CPU预检3.063151秒，父权重/源数据/旧结果SHA核验，官方像素未读。源码安全发布后仅node3双H100提交round18_blur_multiview_context_v1；预计2–4分钟/.067–.133GPU-hours，实际以队列/日志为准。当前无独立持久控制器，接手查job是否已提交/真实ended，勿重复启动。
两rank/launcher退出0后运行本新入口finish，核验权重不变/8图/120块/SHA/同LQ评分，推小报告和耗时。轻任务不扩大节点；旧round16/所有权重/缓存/energy/失败资产保留。完整自主修复授权持续。

## Blur四新裁图官方150结束：小幅平均收益，仍失败；下一项同8训练全图对照 — 2026-11-07

round17已结束，无待续训练/评价；512新增步→最终1792、官方150/2250块、6个子job全部退出0，CPU完整性通过；monitor/upload正常结束。原始终态8f87385已推且10份远端小报告SHA一致。新轮共3.018048999GPU-hours，全部历史资产/失败/费用保留。
官方1792为21.948140/.202188：比原FLUX+.401377dB、LPIPS改善仅.214263%；比1280 PSNR+1.011152dB但LPIPS坏2.886299%；25.9/.18均未过。72/150 PSNR仍差于LQ，平均仅比LQ+.084237dB。存在小幅官方联合平均收益，不得说完全零进展，也不得说修复成功或稳健通过。仅Blur继续。
CPU固定输出误差分解：方向cosine .110052→.190417，非正相关63→48、另幅度过量37→24，修正平方幅度下降61.31%。GT仅事后统计、无系数拟合/逐图选择；下一步不盲加训练。
下一固定短诊断：新1792权重、同原8训练来源的8全图/120块+8原裁图，单node3两H100，旧1280输出复用，比较全图/同像素/中心384的上下文差异是否缩小；无optimizer/新训练/新质量gate，training_fit_only。入口尚未实现/提交，勿重跑旧round16或改其绑定源码。完整自主执行授权继续，先实现CPU预检并发布再派发。
最终模型参数hash1776f5e755050728c106d0d13bead0b7bc4f0fe6e3d5cddac9a4525ffc99fe86；完整状态见foundir_blur_multiview_v1/steps/step_001792.json。详见reports/foundir_blur_multiview_v1_interpretation_2026-11-07.{json,md}、official_comparison及multiview_direction报告。

## Blur1024来源四新裁图终态 — 2026-11-07

完整官方150结束并CPU核验，按同一最终1792输出判断双门槛。
完整官方150：{'lpips': 0.2021881063779195, 'psnr': 21.948139508565266, 'ssim': 0.7361772159623922}；25.9/.18联合通过=False。
只用官方作质量门槛，Blur未过不推进其它类别。完整自主修复授权继续；本轮不自动追加旧步数。见reports/foundir_blur_multiview_v1_terminal_2026-11-07.{json,md}。全部资产保留。

## Blur本轮终态须补充同官方150跨轮对照 — 2026-11-07

当前round17四卡训练→最终1792官方150链继续，不修改运行绑定源码、不重启GPU。新增纯标准库CPU脚本smoke/compare_foundir_blur_multiview_official_v1_cpu.py，待本轮official finish完整退出0后运行。它读取已固定的200/1280/1792三份完整结果，核验同150输入/GT文件身份/无GT路由/基线/评分协议，独立重算均值并比较原FLUX、200、1280与1792；不混拼最好指标，不选权重或逐图输出。
3项CPU测试通过，覆盖不同模型单项最好不能拼联合通过、缺失/重复/GT身份变化拒绝、均值与基线篡改拒绝。当前1792结果尚未产生，不能提前运行或伪造结果。终态控制器先推原始完整结果；接手随后执行此比较脚本并将新comparison JSON/MD小报告推送。无需新增GPU或其它测试gate。
当前训练/监控以foundir_blur_multiview_v1实际step/daemons/worker heartbeat为准。Blur未过25.9/.18之前不转其它类别；全部旧资产保留。

## Blur四新裁图训练已启动：完整父状态与CUDA重载通过 — 2026-11-07

round17缓存4096/4096完成，两node退出0，约510秒/.563941765GPU-hours；原资产不动。node1/node3四H100训练实际运行，截至1794020457.6403143已保存step1307/1792。四rank NCCL及1282真实CUDA状态重载通过；CPU逐值核对父1280与新起始1280的模型、完整Adam、4rank RNG、参数顺序/历史/采样完全一致（receipt因新数据方案不同）。1281实际checkpoint SHA/100 Adam/4 RNG/next_global_slot核验并推送1e4a7e5。不是从头重训或只创建tmux。
monitor PID3518028、upload PID3518027仍存活，身份在foundir_blur_multiview_v1/daemons.json；健康node1/node3心跳新鲜。按固定1792→唯一最终官方150→CPU核验/小报告继续，1536/1792及终态自动回传；接手查实际step/心跳/报告，勿重复提交或修改绑定源码。当前每步含保存约3.18秒，训练暂余25.7分钟，官方另约8–12分钟，非保证。
官方仍20.936988/.196516未过25.9/.18，仅Blur继续。训练loss不可当官方改善，不恢复其它类别或盲加本轮步数。完整自主修复授权持续，所有失败/费用/energy/旧资产保留。详见reports/foundir_blur_multiview_v1_live_training_2026-11-07.json及parent_restore报告。

## Blur四新裁图链已真实启动：四rank缓存生成与首份校验通过 — 2026-11-07

源码51996fd已发布并逐文件SHA核验，queued报告32d7c17已推送。物理node1/node3四H100实际领取round17_blur_multiview_cache_v1_node1/node3，四rank均已产生新缓存；快照1794019850.4793422为446/4096。每rank首份tensor的SHA/有限FP32/schema及原LQ逐值一致CPU核验通过，不是仅tmux/入队。两个worker心跳正常，无早期异常；接手查最新缓存/step/queue，勿重复启动。
持久monitor PID3518028、upload PID3518027，身份在checkpoints/foundir_blur_multiview_v1/daemons.json。固定链缓存→CPU完整性→1280至1792（512新增步）→唯一最终官方150→CPU核验/小报告；1281/1536/1792及终态推送。当前快照尚未进入训练。下一步由控制器接续，无需用户GPU命令，不改已绑定源码；仅异常停链，保留全部资产。
官方仍20.936988/.196516（原FLUX21.546762/.202622），25.9/.18未过；训练/诊断进展不等于官方收益。仍只攻Blur。详见reports/foundir_blur_multiview_v1_live_cache_2026-11-07.json及readiness；无累计GPU上限不改变本轮固定范围。

## Blur四新裁图训练链已准备，官方仍未改善通过 — 2026-11-07

官方Blur150当前1280为20.936988/.196516，原FLUX21.546762/.202622；PSNR退步、LPIPS略好，25.9/.18均未过。训练拟合改善和上下文诊断不是官方修复；最新终态/误差分解已推ff4ccd5。仍只攻Blur。
下一轮round17_blur_multiview_{cache,train,official}_v1_node{1,3}：原1024来源，每来源固定seed/record_id/坐标SHA排序选4个不同部署网格512裁图，4096视图不是4096来源。输入浮点归一化对齐官方CPU路径（原GPU归一化存在微差）；原模型/loss/lr/global8不变。完整1280模型/Adam/四rank RNG和采样位置恢复，固定续到1792（512新增步），只评最终官方150；新数据有意变化，不称旧固定裁图逐位等价延续。无GT选裁图/输出/权重，不设其它测试gate。
10项CPU测试含四进程global8 Adam汇总/同world精确恢复、视图覆盖与缓存键隔离、队列入口检查通过。目的地预检21.459116秒，2048训练图文件SHA及父状态核验。新缓存foundir_blur_multiview_v1，旧缓存/权重/energy/所有失败保持；GPU尚待源码发布后派发，勿调用旧exposure入口。
持久控制器smoke/watch_foundir_blur_multiview_v1_cpu.py monitor/upload：健康node1(10.244.47.55)/node3(10.244.90.116)双H100，缓存→CPU完整性→四卡训练→最终官方150→CPU核验/小报告；1281/1536/1792及终态推送。daemon PID写新输出daemons.json，接手先查实际queue/心跳/缓存/step，勿重复提交。每GPU阶段异常停链并保留，无自动重试/扩步；完整自主修复授权持续。
暂估缓存5–12分钟、训练28–40分钟、官方8–12分钟，合计约45–65分钟/3–4.33使用GPU-hours；以新轮实测更新，不保证质量通过。详见reports/foundir_blur_multiview_v1_readiness_2026-11-07.md。

## Blur全图上下文差异已定位；下一轮改训练裁图覆盖 — 2026-11-07

round16_blur_full_context_v1已结束，8全图/120块/8无缓存裁图、两rank及launcher退出0，前后模型/参考SHA不变，CPU完整性核验通过。node3双H10088.128545秒/.048960303GPU-hours，CPU finish内部3.067172秒、误差分解10.418525秒。无待续本轮GPU任务，勿重跑；全部资产保留。
同一前8训练来源的512像素：独立无缓存23.727759/.188496→全图截取22.499734/.209507，PSNR−1.228024dB，中心384仍−1.076556；6/8下降。无缓存与旧缓存平均PSNR仅差−.003048，存在微差但远小于上下文差异。注意该8图不是此前64图均值26.806/.1440。
原生拼接全部逐像素复现；加权分块MSE .008484854→融合浮点.008092948→uint8 .008094351，融合降低误差，较大差异已在不同裁块预测里。完整8图仅PSNR+.307dB、4/8退步。联系表已查看，部分轮廓/纹理随裁块改变。8图不能证明官方全部失败只有这一原因，更不是修复通过。官方仍20.936988/.196516，门槛25.9/.18未过，只攻Blur。
下一轮拟在原1024来源上用四个不同部署网格裁图替代反复单个固定裁图，保持loss/lr/global8，从完整1280模型/Adam固定续到1792（新增512步）后只评最终官方150。新缓存键/采样覆盖/同四rank恢复实现和CPU预检尚未完成，尚未提交新GPU任务；勿调用旧固定裁图入口冒充新轮。不是用GT选择裁图/输出，不增其它测试gate。长任务优先健康node1/node3四卡，轻任务单node3两卡。
详见reports/foundir_blur_full_context_v1_terminal_2026-11-07.{md,json}及context_blending_v1_CPU报告。全栈自主修复授权继续，保护所有旧修改/大资产/energy。

## Blur同前8训练来源全图上下文诊断已准备 — 2026-11-07

同64固定训练图改善而官方失败，当前只隔离上下文/拼接与来源迁移，非新训练。冻结最终1280/参考LoRA，原1024来源前8张1920×1080，用官方同款512/128拼接120块；另8原裁图无缓存复现，总256模型调用。复用旧8裁图输出，比较全图、同512像素、固定中心384；输出先固定再用训练GT评分，training_fit_only，无新质量gate/GT选择。
3项CPU几何测试、语法、目的地预检5.244058秒通过；源码发布后派发round16_blur_full_context_v1，仅node3双H100。入口smoke/diagnose_foundir_blur_full_context_v1.py prepare|run|finish，新输出foundir_blur_full_context_v1。无独立持久控制器，接手查共享job/心跳，勿重复提交。预计2–4分钟，异常停不自动重试。
两rank及launcher退出0后CPU核验8图/120块/8对照和前后权重SHA，推小报告。官方仍20.936988/.196516，未过25.9/.18，只攻Blur；所有旧资产/energy保留，不盲加步。详见reports/foundir_blur_full_context_v1_readiness_2026-11-07.md。

## Blur同64训练图固定模型改善，官方仍失败；下一步隔离全图上下文 — 2026-11-07

round15_blur_exposure_fit_v1已完成，64/64、两rank参数SHA前后不变、launcher退出0、CPU核验通过。node3双H10050.060766秒/.027811537GPU-hours；CPU finish内部2.750213秒（不含导入/启动）。无待续本轮GPU任务，勿重跑。
同64训练图200步23.672547/.232197→1280步26.805872/.143962，58/64两项更好；仍8/64 PSNR差于LQ，4方向非正相关，mean cosine .666742。当前模型确实学到这些训练输入上的修正；该training_fit_only结果不是官方通过。官方150仍20.936988/.196516，未过25.9/.18，只有LPIPS相对原FLUX略好而PSNR更差。
下一有限诊断：固定前8当前训练来源和最终1280，跑与官方相同的全图512/128拼接，再比较完整图及原训练裁图位置，区分上下文/拼接和来源迁移。现只是计划，入口尚未实现/提交；不得盲加训练步数，不新增其它测试gate，不用GT拟合或逐图选输出。轻任务仍只用node3两卡。
完整自主执行授权继续，只攻Blur；保护所有旧资产/energy/失败及费用。结果见reports/foundir_blur_exposure_fit_v1_terminal_2026-11-07.{md,json}及comparison/result。前一1280官方终态f090a0fb，失败解释及CPU方向分析9ef17405已推送。

## Blur1280官方仍失败；下一项同64训练图只读诊断 — 2026-11-07

round14b两节点四卡训练1280/1280、官方150/150与2250块全部结束，四个子job退出0，CPU完整性核验通过，终态f090a0fb已推送且10份远端文件SHA核验一致。monitor/upload正常结束，无待续本轮GPU任务；禁止重跑或盲加该训练。新增训练+官方共4.449628435GPU-hours，旧资产/费用保留。
完整官方20.936988/.196516，相比原FLUX PSNR−.609774dB、LPIPS改善3.013581%；相比200步PSNR−.882788dB、LPIPS改善18.566616%，两门槛25.9/.18均未过。十轮在线训练MSE/LPIPS虽下降57.26%/45.00%，不能当官方修复。只读方向分析63/150非正相关、另37方向正但幅度过量；仍只攻Blur，不转其它类别。详见exposure_2x2_v2_interpretation报告。
下一项仅物理node3双H100，用最终1280权重重评原同64训练输入/裁图，与旧200逐图比较；没有optimizer/新训练/新质量门槛，不重跑官方。新入口smoke/diagnose_foundir_blur_exposure_fit_v1.py，新输出foundir_blur_exposure_fit_v1。CPU预检6.694160秒、语法通过；发布后提交round15_blur_exposure_fit_v1，实际GPU以共享job/心跳为准。
无独立持久调度器；接手先查本job是否已提交，勿重复。两rank/launcher退出0后CPU finish并推小报告和费用。完整自主执行授权继续；保留所有旧修改/大资产/energy。详见reports/foundir_blur_exposure_fit_v1_readiness_2026-11-07.md。

## Blur1024来源十遍曝光终态 — 2026-11-07

完整官方150结束并CPU核验，按同一最终1280输出判断双门槛。
完整官方150：{'lpips': 0.1965160647034645, 'psnr': 20.93698792775472, 'ssim': 0.726844864725331}；25.9/.18联合通过=False。
只用官方作质量门槛，Blur未过不推进其它类别。完整自主修复授权继续；本轮不自动追加旧步数。见reports/foundir_blur_exposure_2x2_v2_terminal_2026-11-07.{json,md}。全部资产保留。

## Blur已通过1024步保存核验，继续固定1280→官方150 — 2026-11-07

2026-11-07T01:42:56.371153+00:00记录：物理node1/node3共4H100的round14b_blur_exposure_train_v2仍在运行；step1024实际checkpoint SHA、100份Adam状态、四rank RNG、next_global_slot8192已CPU核验退出0，报告f2940a1已推送。monitor/upload持久进程存活，接手先查实时step/心跳，勿重复启动或改绑定源码。
同1024来源前八轮在线训练MSE .008256→.004290、LPIPS .262615→.159421，均逐轮下降；仅training_fit_only，非固定模型/官方结果。官方仍旧21.819776/.241321失败，25.9/.18之前只攻Blur。保持最终1280唯一权重→完整官方150→CPU核验/小报告推送，不挑中间官方最好checkpoint，不自动扩大本轮步数。
详见reports/foundir_blur_exposure_online_epoch8_2026-11-07.{md,json}及step1024_state报告。保留所有旧资产/失败/energy；官方GT不拟合、不逐图挑输出。

## Blur四卡已真实运行；官方尚无双指标改善 — 2026-11-07

此节更新下方四卡准备状态：物理node1/node3共4H100，四rank NCCL及step202实际CUDA模型/Adam/RNG重载全部通过；step201实际checkpoint SHA/Adam/四rank RNG/采样CPU核验通过。2026-11-07T01:02:27.375202+00:00快照已保存至298/1280，两worker心跳新鲜、monitor/upload均存活。步骤与ETA是该时间快照，接手必须查最新共享日志，勿重启或改绑定源码。
原1024来源/配方/global8、原step200模型及Adam→固定1280→唯一最终官方150保持。当前完整官方21.819776/.241321相对原21.546762/.202622，PSNR略好但LPIPS更差；尚无官方联合收益，未过25.9/.18。8训练图26.408422/.136296仅证明小样本拟合，不证明官方修复或步数是唯一根因。
持久CPU监控继续512/1024/1280和终态核验/推小报告；GPU无需用户重输命令。官方未过只攻Blur，不推进其它类别；保留旧六卡草案、energy及全部资产。详见reports/foundir_blur_exposure_2x2_v2_progress_assessment_2026-11-07.{md,json}。

## 当前Blur曝光试验改为健康node1/node3四卡；六卡准备保留未启动 — 2026-11-07

启动前node0/2心跳中断，未有明确退出；随后node0恢复，node2仍陈旧。当前采用持续新鲜的物理node1(10.244.47.55)/node3(10.244.90.116)，两节点各2H100，master:29879。原六卡job从未提交，旧3x2源码/预检/资产全部保留；不要启动round14三节点任务。
科学范围不变：原1024来源/配方/global8，原完整step200模型+Adam→固定1280，总每来源10遍→唯一最终权重官方150。不使用8图拟合权重。world2→4显式重建四rank RNG，非逐位等价父训练；同四卡保存恢复完整，202后真实GPU重载。每rank2真实样本、loss/8+SUM。8项CPU测试含真实4进程Adam/精确同world恢复通过；预检11.341932秒，官方150完整38/38/37/37分片元数据通过。
新控制器分别给既有node1 worker(nodes=[1])和node3 extension(nodes=[3])提交同协议的成对子job；不改旧worker、不发送其不支持的联合nodes列表。两任务通过新torchrun入口协作，故障协调取消本组peer。实际四rank NCCL待日志，不把CPU通过当GPU开始。
持久CPU脚本smoke/watch_foundir_blur_exposure_2x2_v2_cpu.py monitor/upload；输出foundir_blur_exposure_2x2_v2，PID记daemons.json，日志logs/foundir_blur_exposure_2x2_v2_{monitor,upload}.log。队列round14b_blur_exposure_{train,official}_v2_node{1,3}；先查真实step/心跳，勿重复提交。201/512/1024/1280和终态推小报告。
完整官方150的25.9/.18仍唯一质量门槛，旧21.819776/.241321失败；Blur未过不推进其它类别。全栈授权继续，保留所有旧资产/energy。详见reports/foundir_blur_exposure_2x2_v2_readiness_2026-11-07.md。

## Blur固定1024来源十遍曝光链已准备：三节点六卡 — 2026-11-07

固定8训练图100步已证明可拟合，非官方通过。当前保持原1024来源/配方/global8，从原完整step200模型与Adam迁移六卡到固定总1280（每来源10遍），再评价唯一最终1280的完整官方150；不使用8图拟合权重、不增加其它测试gate、不挑中途官方最好checkpoint。官方旧21.819776/.241321仍失败，25.9/.18之前只攻Blur。
两rank→六rank保留模型/Adam/step/采样位置，显式重建6rank RNG，非逐位等价父训练恢复；旧父checkpoint/RNG原封保留。每步2/2/1/1/1/1真实样本、loss/8、全局SUM，global8/lr不变。迁移后同六卡可恢复，202步做真实CUDA状态重载。7项CPU测试含六进程Gloo/Adam/精确同world恢复通过，目的地预检14.163213秒，官方150元数据375块/卡已核验；实际CUDA/NCCL待日志。
发布后启动smoke/watch_foundir_blur_exposure_v1_cpu.py的monitor/upload两个独立CPU进程，训练→CPU收尾→官方150→CPU核验→简报。201/512/1024/1280推小报告；网络故障不阻塞已发布GPU任务，工程故障仅取消本轮job，不重试/扩步。队列round14_blur_exposure_{train,official}_v1，node0/1/2双H100，node3空闲。
接手先查foundir_blur_exposure_v1/daemons.json、共享queue实际日志/step/心跳及publication_requests/published；不要重复启动或修改绑定源码。预计训练45–90分钟/官方6–12分钟，仅粗估。保留所有旧资产/energy/失败及费用。详见reports/foundir_blur_exposure_v1_readiness_2026-11-07.md。完整自主修复授权继续。

## Blur固定8图100步完成：映射可拟合；官方仍失败 — 2026-11-07

round13_blur_fixed8_fit_v1已结束，100/100、两rank及launcher退出0、CPU finish通过；node3双H100 519.128156秒/0.288404531GPU-hours，CPU3.111293秒。101步receipt连续，40图快照/SHA/均值核验，独立核验step1/50/100实际checkpoint；非全部大文件重哈希。无待续本轮训练，所有旧资产保留。
training_fit_only：同8图LQ21.114951/.423268→step25 22.784127/.321262→step50 24.831643/.190802→最终100 26.408422/.136296；8/8双项改善，PSNR+5.293470、LPIPS降67.7992%。原1024来源step200在同8图仅20.479493/.287003。说明当前原生预测差映射有小样本学习能力，但来源数/曝光/跨图组成共同改变，不能独证官方是步数不足；不从本8图模型启动部署/正式训练。
官方150保持21.819776/.241321未过25.9/.18，唯一质量门槛仍官方；Blur未过不推进其它类别。完整结果/40逐图/100步日志/费用见reports/foundir_blur_fixed8_fit_v1_result_2026-11-07.json及terminal.{json,md}，勿重跑round13。
下一方向保持原1024来源/配方，受控增加训练曝光→固定最终步官方150，不改loss、不从8图权重初始化、不挑官方最好中途checkpoint。先核对同world恢复或多节点global8的等价与RNG限制，再冻结步数/拓扑；当前尚未实现/提交下一训练。完整自主研究授权继续，保留旧失败/energy/全部资产，无累计GPU上限但逐轮固定范围和费用。

## Blur固定8图fresh100已真实运行：首步状态核验通过 — 2026-11-07

round13_blur_fixed8_fit_v1已由node3双H100真实领取并训练。第0步8/8浮点输出精确复现LQ；第1步实际checkpoint SHA、参数更新、100份Adam状态/两rank RNG/next_global_slot=8独立核验通过。训练图原21.114951/.423268，step1为21.241245/.424260，早期LPIPS变差如实记录，不称修复。
固定总100步/同8来源每步一次，0/1/25/50/100快照、每完整步保存；模型/配方未变，不自动官方或长训。唯一质量门槛仍完整官方150的25.9/.18，当前官方21.819776/.241321失败。其它类别不动，所有旧资产保持。
源码提交77300828cabf41d198c2b25a48bb4e38c629f376。无独立持久观察器；接手查共享round13队列/心跳、foundir_blur_fixed8_fit_v1的step及快照，不能重复提交。两rank/launcher退出0后执行smoke/train_foundir_blur_fixed8_fit_v1.py finish并推完整小报告和成本；工程/拟合不等于官方成功。

## Blur固定8训练图100步拟合诊断已准备 — 2026-11-07

继续只攻Blur，官方150的25.9/.18是唯一质量门槛；当前21.819776/.241321仍失败。历史16图50步flow/末段监督不是当前原生预测差映射，不重复旧任务。
当前1024训练来源前8条/固定512裁图/原缓存，fresh原step10000 LoRA与Adam，global8每图每步一次，共100步800曝光。模型/loss/lr/数值路径不变。0/1/25/50/100固定全8图快照与逐图指标、每步Adam/RNG/SHA；只标training_fit_only，无其它测试gate、不选最好checkpoint、无自动官方或长训。
11项CPU测试/语法通过，目的地预检12.978861秒/GPU0，16图SHA/裁图、8缓存及模型/指标核验通过。发布后提交round13_blur_fixed8_fit_v1，只用node3双H100；预计8–12分钟，非实测保证。入口smoke/train_foundir_blur_fixed8_fit_v1.py，输出foundir_blur_fixed8_fit_v1。
无独立持久观察器，接手查实际队列/step/五组快照，结束后CPU finish/推小报告，不重复提交。少量训练图学会不等于官方通过；若100步不行也不能宣判模型不可学。依据结果再制定下一方案，全栈授权继续。保留旧全部资产/energy/失败，勿改旧运行绑定源码。详见reports/foundir_blur_fixed8_fit_v1_readiness_2026-11-07.md。

## Blur两次实际更新核验完成：有学习，无官方联合收益 — 2026-11-07

只读重放step0→1及199→200、原global8和rank RNG，四checkpoint SHA/Adam/采样核验，原loss精确复现/gradient norm固定容差通过；16逐图、两rank/launcher退出0、CPU finish通过，无optimizer更新。node3双H100 60.297005秒/0.033498336GPU-hours；CPU1.779189秒。四worker现无待续训练，以心跳为准。
第1步MSE8/8改善、LPIPS8/8恶化，平均PSNR+.138572、LPIPS+.005266；第200步均值两项微弱改善，但3/8两项都差、只有2/8两项都好。原weighted loss两步均下降；不能说优化器完全没学习，也不能把此诊断当修复成功。实际delta方向已核对，不依据RGB比或原始SGD方向盲改LPIPS系数。
官方150仍21.819776/.241321，较旧FLUX PSNR+.273014但LPIPS差19.0991%，未过25.9/.18；仍只攻Blur，无其它测试gate。两批不能概括全部200步，未证明官方唯一根因。完整记录见reports/foundir_blur_actual_updates_v1_result_2026-11-07.json与terminal.{json,md}，不要重复round12。
下一项先核对历史是否已有等价小样本充分拟合实验，再冻结少量训练图原生delta学习能力对照，区分映射/优化学不会和覆盖/跨图干扰。尚未实现/提交新训练；training_fit_only，不是新质量gate。完整自主研究授权继续，保留全部资产和失败，不盲加旧200步、不转其它类别。

## Blur历史真实更新重放已准备：step0→1及199→200，无新训练 — 2026-11-07

官方Blur150仍21.819776/.241321，未过25.9/.18，只推进Blur；没有联合质量改善。继续全栈自主修复，先确认实际Adam有限步效果，不由RGB/原始SGD梯度比直接改loss。
四个原checkpoint的SHA/step/采样位置/FP32参数/Adam/RNG通过CPU预检，10.120323秒/GPU0；4项CPU测试及语法通过。恢复各步真实global8、原rank RNG及before/after参数，重算combined loss和gradient norm须匹配原日志，再报告分项梯度与真实delta内积及同批有限步前后指标。无optimizer更新、不重跑200步、不读官方像素；training_fit_only，不设新测试gate。
入口smoke/diagnose_foundir_blur_actual_updates_v1.py prepare|run|finish，输出checkpoints/foundir_blur_actual_updates_v1。源码发布后提交round12_blur_actual_updates_v1，只用node3双H100；无独立持久观察器，接手查真实queue ended及两rank，然后CPU finish/推小报告，勿重复提交。预估2–4分钟，不是实测承诺。重放超固定容差须如实失败，不盲放宽容差。
保留旧所有源码/权重/缓存/energy及失败费用。依据实际结果再制定下一训练方案，不把诊断完成称为修复成功。详见reports/foundir_blur_actual_updates_v1_readiness_2026-11-07.md。

## Blur参数梯度诊断完成：MSE范数更大；下一步核对真实已保存更新 — 2026-11-06

固定8训练图×初始化/step200，16逐图+2全批，两个rank/launcher退出0、CPU核验通过。初始化/step200路径复现，参数求导前后SHA不变。106.187254秒/.058992919 GPU-hours；全部旧资产保留。
完整global8加权LPIPS/MSE参数梯度norm比为.184308/.363637，cosine为−.290730/−.408207；16逐图MSE norm均更大。RGB LPIPS梯度大不等于参数LPIPS主导，不依据旧RGB比值削弱LPIPS。两全批原始混合SGD方向一阶降MSE/升LPIPS，但不能据此推断实际Adam更新或有限步效果。
下一有限诊断：只读原已保存step0→1和199→200，在各自原采样global8上重算分项梯度/真实参数delta内积/有限前后loss，区分实际更新与跨样本干扰；不执行optimizer更新或新训练、不重跑官方、无新测试集gate。入口尚未实现/提交，当前所有worker空闲，以心跳为准。完整自主研究授权继续。
官方21.819776/.241321仍失败；唯一质量验收仍完整官方Blur150的25.9/.18，未过不推进其它类别。报告reports/foundir_blur_loss_gradients_v1_result_2026-11-06.json及terminal.{json,md}。勿重复此梯度任务或盲调系数。

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

# Project agent instructions

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

## 当前修复：Rain资格驱动描述误拦截 — 2026-11-02

用户已授权修复。旧qualification_4x2_v1 / launch_ccecd3a_v1四节点在训练前失败，
0步、0缓存、无指标，总.187777778 GPU-hours；不能写成OOM或Rain学习失败。
四节点驱动实际同为570.124.06，仅GCC构建后缀22.04/22.04.2不同。
现在跨节点比较解析出的NVIDIA驱动版本，其余runtime/metric字段仍严格一致；
未知驱动格式拒绝启动，原始完整runtime收据保留，同节点恢复仍要求原始环境完全一致。
此补丁不改25步、32来源、loss、学习率、模型、数据或四节点×两张H200资源规格。

源码绑定已变化，禁止原地改旧receipt哈希或覆盖旧run。目的地代理安全同步最新main，
保留本地修改及所有资产，重新执行本轮CPU preflight并写新报告路径；
通过后提供同入口四节点命令，统一新run_id=qualification_4x2_driverfix_v1、
新launch_id=launch_driverfix_v1，使用当前node0地址，不沿用历史IP。
原qualification_4x2_v1全部保留；新ID不是续训，无旧训练步可继承。
本地主代理只修复/测试/推送，不能直接启动目的地GPU；CUDA/NCCL及训练质量仍待验收。
只重试同一25步资格，不自动formal、router/MoE训练或energy任务。
细节见reports/foundir_rain_flow_runtime_guard_fix_2026-11-02.md。

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

## 当前入口：四类已有LoRA完整50步单选重放 — 2026-11-02

用户已授权继续。目的地39e4575资产审计和12项CPU测试通过，retention日志SHA差异
已核实为CR换行归一化，原文件SHA一致，不再阻塞。当前交付零训练GPU重放，不启动Rain训练。
四类各32训练来源、128张512 LQ；同prompt/seed/crop/量化比较原单LoRA与四分支bank固定单选。
不读GT/开发/保护像素，不算新PSNR/LPIPS，不把静态单选通过当动态VLM路由或科学成功。
入口 smoke/start_foundir_existing_flux_replay_v1_tmux.sh；配置 configs/foundir_existing_flux_replay_v1.json。
主代理先推送，目的地安全同步/4与8卡CPU预检后给用户GPU命令，不假称已启动。
默认单节点8卡、保留4卡同型号H100/H200；绝不复用旧跨节点地址。每图原输出/bank输出
和SHA不可变保存，同ID同world/运行环境核验恢复，不能自动换卡数混合已有结果。
每图decoded最大差≤1e-6且uint8完全一致；128/128汇总从保存tensor独立复算。
离线Qwen短暂生成四个固定prompt后退出，FLUX阶段无Qwen/Restormer/CNN/LPIPS模型。
无自动后续训练。8卡未实测估计.5–1.5小时/4–12 GPU-hours，硬cap2小时/16GPU-hours；
4卡粗估1–3小时，同GPU-hours，硬cap4小时。每次attempt成本保存，旧失败成本不清零。
完整边界、恢复、CPU代理操作见 reports/foundir_existing_flux_replay_v1_readiness_2026-11-02.md。
下一步是本重放验收；Rain多步适配训练器及共同prompt/输出协议仍未交付。

## 当前用户授权：已有LoRA专家库＋Rain多步适配 — 2026-11-02

本节覆盖旧五类retention正式1000步计划及更旧“唯一energy/Rain/EBT”优先级。
用户批准保留四类成功50步FLUX专家，单独适配Rain，最后接右图内部VLM残差router。
不续训失败2025，不启动旧retention正式；另一代理FLUX-native energy工作不动。
右图没有规定FFN、禁止LoRA或50步。新expert bank明确不同于旧双FFN＋共享LoRA。
旧PPT handoff的no-LoRA约束在此被用户新决定覆盖，不宣称导师已额外批准所有实现细节。

第一阶段已交付CPU资产审计、冻结LoRA混合算子和静态单专家重放安装器；
还没有完整GPU重放/动态router/Rain训练器，不冒称已训练或新指标达标。
按原安装位置保存各LoRA，混合B_i(A_i(x))，不平均A/B因子；Rain endpoint暂时拒绝进入
flow bank。原一步Rain只读保留，作为未来训练示范，GT仍主要监督；不运行Restormer/CNN。
四类提示词与输出alpha不同，需先原协议单选等价、再确认共同条件，不能只凭都是50步
就混合。计划Rain从成功step4000的独立副本适配，fresh Adam，不是严格续训或随机底座。
详见 reports/foundir_existing_flux_experts_plan_2026-11-02.md。

唯一GPU根仍为 /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。
旧目录只读迁移来源；主代理在隔离worktree提交，目的地代理同步并核验新目录资产。
本轮遵守用户最新资源规则：单节点同型号8×H100/H200，保留4卡兼容，不沿用旧四节点IP。
当前没有GPU启动命令/耗时实测，不借用旧一步MoE吞吐估算50步。
official/sealed/internal/parent-test及最终1859 Noise像素不读；本轮仅核验manifest元数据。
历史训练池隔离检查不等于FLUX预训练数据审计；CPU算子等价不等于完整CUDA恢复等价。
下一步：目的地安全同步并执行只读CPU审计、回传小报告；再交付原协议GPU重放和Rain资格。
禁止按下方旧标题自动启动旧任务。按focused allowlist提交/push，不上传权重/数据/实时日志。

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

## 永久默认GPU资源：三节点，每节点两卡（3×2）— 2026-10-31

用户最新明确要求：**当前及以后所有非轻量GPU实验，默认使用3个节点，每节点
2张H100或H200，总计6卡。** 这不是只针对一次实验的偏好。此节覆盖下方所有
单机4/8卡默认、禁止跨节点及旧3×2/1×2优先级。不能让新代理按旧记录改回八卡单机。
允许不同节点分别是H100/H200；当前启动器要求每节点内部两卡同型号、每卡至少75GiB。
六卡是一个跨节点同步训练任务，不是三个独立任务；不把GPU显存加起来当单卡显存。

轻量CPU检查、小规模推理/连通性检查等确实不需要六卡时，可以用CPU或单节点1/2卡，
先说明任务与用卡数。**4B完整三步二阶训练即使只有25步也不默认算轻量。**
不自动从3×2降为单机，不因网络问题暗改训练规模；其他重任务拓扑须用户另行明确。
现有单机2/4/8卡入口仅保留兼容，不是新的默认或自动失败回退。

当前FLUX-only实验按 `reports/foundir_flux_native_energy_3x2_readiness_2026-10-31.md`
执行：fresh ID=`qualification_3x2_v1`；每节点CUDA_VISIBLE_DEVICES=0,1，
NODE_RANK分别0/1/2，共用当前node0的MASTER_ADDR/端口。不要复用聊天中的旧节点IP。
全局batch仍8：每步两rank算2样本、四rank算1样本，跨步轮换并按样本加权汇总；
不能悄悄改成6或12。512图、完整3次更新、25步/原损失/学习率/双目标均不变。
每step仍保存6个rank的RNG、Adam和全局采样位置；4/8卡或旧源码checkpoint保留，
不得直接当新六卡resume。新3×2断点仅允许同身份/拓扑/源码严格恢复。
启动前自动检查跨rank通信/共享目录/代码身份；主会话无GPU，不冒称已完成三节点CUDA验收。
本轮16项CPU测试通过，含六进程Gloo梯度对串行batch8、Adam/RNG续训；不代表NCCL通过。
GPU阶段每节点上限2小时，六卡合计上限12 GPU-hours，CPU核对与退出清理另记实际成本。
未实测吞吐，不承诺三节点更快；不自动200步/正式训练。

## 当前执行：FLUX-only 25步入口已实现；CPU通过，GPU未启动 — 2026-10-31

本节的单机4/8卡为初版历史快照；当前拓扑/启动ID以顶部3×2规则与新报告为准。

用户已要求execute/start。读 `reports/foundir_flux_native_energy_v1_readiness_2026-10-31.md`。
本节只更新实施状态，不取消下方FLUX-only禁令。新代码用FLUX内部rank16 LoRA和
线性energy读出，完整三次latent更新及实际MSE/LPIPS训练；没有外挂CNN/Restormer。
9项CPU小尺寸真实FLUX块/二阶梯度/Adam续训测试通过，未验证真实4B CUDA显存/速度。
主会话没有GPU且新根未挂载，不能声称已启动；先同步GitHub文件，再由GPU节点启动。

唯一新入口：`smoke/start_foundir_flux_native_energy_v1_tmux.sh qualification_v1 8 fresh`。
支持单机4/8张H100或H200；全局batch8，4卡累积2/8卡累积1，512图、完整3次更新。
先25步工程验收；不自动200步/正式训练。GPU阶段上限2小时（含缓存，不含CPU核对），
不是预计耗时。每完整step持久保存Adam/参数/RNG/采样，严格同身份resume。
新权重只含FLUX energy LoRA/条件投影/读出；不继承CNN权重或CNN反例。
来源沿用24个已审计训练来源；8个probe是训练来源，不能称泛化。旧GT/历史基线诊断
不重跑，旧CNN/V3启动命令仍禁用。真实GPU错误必须如实记录，不暗换骨干/截断反传。

源码/启动配置已准备不等于GPU资格通过；TMUX_CREATED不等于训练开始。
目的地CPU代理只同步/回报；用户有GPU访问时执行新入口。不要直接跑旧CNN命令。

## 唯一当前任务：FLUX-only energy 重做；禁止 CNN / Restormer — 2026-10-31

本节覆盖下方所有历史 CURRENT/ACTIVE/下一步200步。用户明确要求只用 FLUX，
放弃外挂 CNN energy 和 Restormer。当前旧 CNN 200步已结束，不得续训/重启。
必须先读 `reports/foundir_flux_only_energy_redesign_2026-10-31.md`。

硬边界：可训练图像主干只能是预训练 FLUX.2 Klein base 4B。
此禁令适用于当前及后续所有新恢复/energy实验，除非用户明确撤销；不是仅本轮偏好。
禁止 SpatialConditionalEnergy、TailQualityEnergy、任何外挂CNN energy/scorer、
Restormer/residual Restormer/pixel refiner，也禁止用它们蒸馏本次模型。
不能换名字保留CNN；不能只给旧CNN加FLUX输入就称为FLUX-only。
允许FLUX内部energy LoRA、线性读出和VLM条件投影。原生冻结VAE及固定LPIPS
损失/指标保留，内部卷积不是新增恢复器；不得声称全计算图零卷积。
原FLUX恢复LoRA保持冻结作为生成路径；新增energy LoRA在FLUX Transformer内
训练，energy必须读取FLUX内部图像tokens。共享底座、显式区分生成/energy模式，
不能在反传图存活时原地换权重。旧CNN权重/梯度/反例轨迹不初始化新实验。

训练要直接监督energy引导后的实际图像（MSE+LPIPS，完整1/2/3步展开），并含
同图不同恢复动作的质量排序；不再仅拟合缓存分数/导数冒充完整PPT。
最终图t=0和真实尾段t48状态分开建模，不能混同。GT仅在训练/离线评估使用。
保持原FLUX基线/双uint8/alpha=.30，目标仍是energy独立+.30dB与LPIPS相对
下降9.928107050377874%，必须最终在独立来源验证。禁止保护测试像素泄漏。

实施状态以顶部同日更新为准：FLUX-only训练器已实现、CPU检查通过，GPU尚未启动。
下一步是新25步入口的真实4B二阶梯度/显存/resume验收，不是运行旧V3命令。
先定向工程资格，再独立fresh拟合；通过才扩合规训练数据和独立来源评估。
更大FLUX不保证成功，也不能说旧CNN已接近联合过关（LPIPS仅降2.49%）。
不再重复历史基线差1灰度级诊断/GT搜索；保留旧证据，不删除历史代码和结果。
单机4/8张H100/H200兼容是新实现目标；按用户实际资源冻结拓扑，不自动跨节点。
时长和GPU-hours由新架构实测后报告，禁止继承CNN速度或隐瞒二阶反传成本。

执行根仍是 `/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
GitHub是交付源；目的地代理只同步指定文件/回报，不自动运行GPU。
旧workspace/旧ACTIVE/Generalist/Rain/MoE优先级均不能覆盖本节。
报告必须用通俗中文，明确“实际实现 / 待实现 / 已测结果 / 假设”，不得再省略主干身份。

## CURRENT:8步资格完成；用户授权推进独立200步拟合 — 2026-10-30

用户在检查终态后明确要求continue。本节覆盖下面本次只同步/尚无GPU结果的旧状态。
下一任务是 `trainfit_3x2_direct_v1`，mode=trainfit，沿用刚成功的3节点各2张H200，
引用 `qualification_3x2_direct_v1` 的工程结果；fresh模型/Adam，不能加载资格权重。
不修改已经验收的源码、配置、24来源、损失、学习率、global batch32和单步probe。
200步在50/100/150刷新反例，每完整step保存；真实图片指标仅在200步结束后计算。
不要把中途loss日志或反例刷新说成中途PSNR/LPIPS验证。

资格终态见 `reports/foundir_flux_tail_v3_direct_terminal_2026-10-30.{md,json}`：
三节点status0，8/8步与8/8训练probe；bounded相对当前FLUX +.608392dB，
LPIPS仅降2.596406%，联合目标未过；3/8图PSNR退步、3/8图LPIPS退步。
总38分58秒、3.894444使用GPU-hours。主代理重算均值并核对59项Git内源码SHA一致；
8项third_party源码及远端checkpoint本体不在本地主仓库，不能冒称本地全量重验。

读 `reports/foundir_flux_tail_v3_direct_trainfit_readiness_2026-10-30.md` 执行后续交接。
六H200预计11–13普通小时/66–78使用GPU-hours；硬上限13小时20分/80使用GPU-hours。
这是按187.332秒/step外推，不是已完成时长。硬上限或外部中断后保留全部进度，
恢复必须同拓扑/设备/运行环境；不自动重试或扩大总预算。
主代理不能访问新GPU节点；本次只准备并push交接，没有声称200步已启动。
目的地CPU代理继续只同步/回报，不自行启动GPU；用户在GPU节点启动已批准任务。
不重跑8步资格、GT搜索或历史基线诊断，也不恢复Rain/MoE/EBT/Restormer分支。

旧“三关都未通过”说法已过时：GT多轮搜索已在另24来源确认平均双目标，22/24单图通过；
它不是当前单步energy的同预算上界或泛化证明。下一步先检验学习，不再重复GT关卡。
200步后若可学习性成立，再单独冻结未参与拟合/选参的来源评估；不是本次自动任务。

## CURRENT AUTHORITY:直接推进 V3；本次交付只同步代码、不启动 — 2026-10-30

用户明确取消额外的历史基线诊断/迁移验收前置关卡，要求直接推进训练。
随后又明确本次交付范围：主代理只修改、测试、推送代码并给CPU代理同步提示词；
不向用户提供GPU启动命令，不让接收代理自动运行GPU。此节覆盖下方所有
“历史逐像素必须一致／先审批迁移契约／再重复诊断”的旧要求。

本版V3以当前run的原始FLUX输出为proposal。旧图差异只记录，不阻断；
同环境原始入口与BF16尾段一致性、当前run内参考图一致性、数据/SHA/finite/
checkpoint保护仍保留。旧标量量化、FrozenFlow及V1/V2绑定源码未改。
训练条件和成对比较中的proposal候选/标签/导数使用同一当前参考；其他历史
候选仍是合法训练样本，保留其真实质量监督，不把旧proposal条件混入新候选。
最终同时报告当前原FLUX、历史图、FP32零引导、raw/bounded energy结果，
分别给出增益，不能把环境/精度收益记到energy名下。不读取保护图像。

新输出根 `checkpoints/foundir_flux_tail_relative_energy_v3_direct`；旧输出/缓存
不覆盖。bootstrap按run保存，不跨硬件任务共用。仍fresh 8步、step4刷新；
200步仍为之后独立手动fresh任务，不自动接续；每完整optimizer step保存不变。
默认单节点2卡H100/H200，保留1x4、1x8、3x2兼容；global batch32不变，
两卡每rank16对，pair loss权重world/32。单节点忽略遗留跨节点rendezvous变量。
CPU代理只按 `reports/foundir_flux_tail_v3_direct_sync_2026-10-30.md` 和对应SHA清单
同步到新共享工作根、核对文件；不重跑基线诊断、不改配方、不启动训练。
当前没有新GPU结果，CPU通过不等于CUDA资格通过；收到用户后续启动请求再行动。

## CURRENT:主代理决策，目的地代理只同步和回报；只读 CPU 核对 — 2026-10-30

用户明确保留原会话代理为主代理：负责方案、代码、结果判断和 focused GitHub push。
新工作区代理负责安全同步主代理指定的文件、执行明确交付的检查、采集实际结果，
将小型 JSON/简短报告推送 GitHub；不得自行改方案、参数、基线或启动/重试 GPU。
直接的新用户指令优先于此分工。GitHub 是异步报告渠道，不是实时进程控制通道。
远端日志不可见时如实说明，不据此宣称任务已停止、正常或被监控。

本次用户要求 CPU 检查由主代理自行做。主代理已完成本地浮点量化穷举和旧缓存
SHA 检查，但新工作根及新 diagnostic tensors 在主代理挂载中不可见。
不要把“有 CPU”误写成“已经拿到新张量”。无需新增 GPU 实验。
新节点已执行 `smoke/replay_flux_tail_quantization_cpu_v1.py`，报告提交0b93b95。
主代理已收到真实重放JSON（SHA02b4990d0be484ce8dd72e7641153375ea5d5059208db26d7714f506539d40a8），
核对报告/源码身份与统计关系；原始远端tensor仍未在主代理重新读取。
两次CPU取整及完整重放与新GPU图差异均为0，仍与历史图相差3630通道元素。
改FP64/精确有理数反而扩大历史差异，不能用这些替代算法修复或替换基线。
CPU/GPU输入归一化存在约1.19e-7差异，但只替换第二次取整输入没有改变结果；
它是否通过上游生成放大尚未知，不能声称归一化/H200/内核已确定为根因。
此CPU检查已完成，不重复命令或双卡生成诊断；energy仍0训练步。
分析与下一步建议见 `reports/foundir_flux_tail_cpu_replay_and_observer_handoff_2026-10-30.md`。
建议下一步明确冻结新环境的匹配对照及受影响缓存迁移办法，保留旧基线对照；
这是待批准的工程迁移建议，不授权删相等断言、重建缓存或重启资格/训练。

## CURRENT RESULT: two-GPU baseline diagnostic finished — 2026-10-30

Read `reports/foundir_flux_tail_baseline_parity_terminal_2026-10-30.{md,json}`.
Single-node two-H200 baseline_parity_2gpu_v1 completed at18:40:24 UTC,
DIAGNOSTIC_COMPLETE and status2 (expected historical mismatch, not a crash).
One original training source, two repeats/GPU: all four full trace files are
byte-identical, all160 stages agree. Historical image differs at3591/262144
RGB pixels (1.369858%), maximum1 uint8 level;3630 channel elements differ.
14 artifact hashes,68 source hashes and parent cache hash independently checked;
uint8 counts independently recomputed without GT decoding. No energy training.
Cost92 wall seconds/184 GPU-seconds=.0511111 used GPU-hours; including prior
three-node failure=.4455556 GPU-hours. Peak allocated8.80852GiB per GPU.
This does not locate the historical root cause or qualify mixed GPUs/all24
sources. Preserve exact parity and the old baseline. Next minimal step is
read-only quantization replay from saved tensors and old runtime provenance;
no new GPU job was started by this monitoring turn. Short work remains1x2.

## CURRENT AUTHORITY: short work on one two-GPU node — 2026-10-30

User now requests single-node TWO-GPU execution for short diagnostics/jobs;
reserve cross-node execution for long-running work. This overrides older
4/8-only and blanket 3x2-default rules for short work. Mixed H100/H200 remains
allowed with actual identity checks. Do not reduce global batch or silently
reinterpret six-rank training checkpoints as two-rank resumes. Current V3
qualification/trainfit launchers remain 3x2/1x4/1x8; this update adds a separate
short diagnostic, not an unverified two-rank training mode.

User said to continue with the next diagnostic: one original training source,
two repeats on each of two local GPUs, original frozen baseline only. Read
`reports/foundir_flux_tail_baseline_parity_readiness_2026-10-30.md`.
Entry: `smoke/run_flux_tail_baseline_parity_2gpu_v1.sh` with explicit qualified
FLUX_TAIL_V3_PYTHON, CUDA_VISIBLE_DEVICES=0,1. No tmux, torchrun, MASTER_ADDR,
NODE_RANK or inter-node networking required. Estimated 3–8 minutes (unmeasured),
cap900 seconds/.5 used GPU-hours plus30s cleanup. Fresh diagnostic output root;
all old V3 bound code/cache/contracts remain unchanged. No learning, GT tensor
decode, parity relaxation, baseline substitution or automatic qualification.
CPU observer cannot launch on GPU nodes; GPU diagnostic remains pending until
its actual log/results appear. Status2 with DIAGNOSTIC_COMPLETE means measured
historical mismatch, not missing diagnostic evidence. Partial/error runs remain.

## CURRENT TERMINAL: V3 first 3x2 attempt stopped at baseline parity — 2026-10-30

Read `reports/foundir_flux_tail_v3_cluster_baseline_failure_2026-10-30.{md,json}`.
qualification_3x2_v1 ended on all nodes at 18:00:14–15 UTC, status=1.
Six H200 ranks passed startup shared-storage/NCCL checks and completed 24 base
records (1706 frames), then all failed `cached historical baseline changed`
in initial mining, before energy-guided updates. Zero optimizer steps/probes,
no training checkpoint/result. This is not evidence of energy learning failure.
66 source hashes and 24 base metadata hashes/identities rechecked successfully;
tensor contents were not rehashed in this terminal status audit. Difference
magnitude was not saved; runtime/kernel/hardware/path cause remains unisolated.
Total used cost: 1420 GPU-seconds = .3944444 GPU-hours; wall span 239 seconds.
Preserve everything. No auto-retry, relaxed parity, cache replacement or trainfit.
Next proposal is one allowed training-source baseline-localization diagnostic,
with separately scoped command/budget; none started by this status audit.
This terminal snapshot supersedes older GPU-pending statements, not 3x2 authority.

## REQUIRED CONTEXT: design history and failure reasoning — 2026-10-30

Before interpreting V3 quality results or changing its scientific recipe, read
`reports/foundir_energy_design_history_and_failure_playbook_2026-10-30.md`.
It connects the original PPT, residual/selector attempts, FLUX-tail GT evidence,
failed learned V2 and current V3; it records what each result does and does not
establish and how to localize the next failure. Do not restart old experiments
or count residual/GT gains as learned-energy gains. Do not interrupt a live job
merely to read it. This context requirement does NOT override the current
hardware/runtime authority immediately below or authorize additional trials.
At each material milestone, append evidence and the next falsifiable decision
to the reasoning record as well as the runtime handoff; push focused updates.

## CURRENT AUTHORITY: user-authorized 3x2 cross-node V3, mixed Hopper allowed — 2026-10-30

The user explicitly changed the available allocation to three nodes with two
GPUs each and authorized cross-node H100/H200 mixing. This supersedes ALL
older single-node / homogeneous-only / default-eight-GPU restrictions for V3.
Latest supplied node evidence: all three currently expose two H200s, >=75GiB,
with the same GPFS project mount and executable project-local Python.
Map node ranks 0/1/2 to 10.247.94.142 / 10.246.88.49 / 10.244.58.183.
These are user-provided identities, not an agent-run CUDA/NCCL qualification.

Read `reports/foundir_flux_tail_v3_cluster_readiness_2026-10-30.{md,json}`.
V3 now defaults to 3x2=6 ranks, retaining 1x4 and 1x8; mixed H100/H200 SM90
with >=75GiB each is allowed, including within a node. All nodes require the
same code/runtime and shared project storage. No architecture/loss/metric,
source, crop, seed, threshold or training-step changes are authorized here.
Global batch remains 32: rank counts 6/6/5/5/5/5, EACH pair weighted world/32
before DDP's rank average. Never drop/pad pairs or average rank means equally.

Default run is qualification_3x2_v1, eight steps with mining refresh at four.
New output root is checkpoints/foundir_flux_tail_relative_energy_v3_cluster;
all prior V3 artifacts remain frozen in their old root. Five V3 bound files
changed and two helpers were added; 59 prior bound files, including all V1/V2
producers and the energy architecture/loss, retain their prior hashes.
One torchrun/group spans all stages; only global rank zero locks/writes shared
preparation metadata. Each node has its own locks and appended log. Fresh
per-rank shared-storage tokens and actual NCCL reductions precede data access.
Resume binds global/local ranks, host/GPU UUID/model and runtime identities;
changing the mapping requires separate audit, not a claimed exact resume.
Same-topology CPU resume is exact; mixed-GPU CUDA bitwise equivalence is NOT
claimed. Keep every complete-step checkpoint and all six rank RNG states.

75 CPU regressions and a real six-rank Gloo test via three two-worker torchrun
agents passed: serial 32-pair mean/gradient equivalence, durable model/Adam/RNG
exact resume, rank agreement and rejection of incompatible resume. That test
ran on ONE physical CPU host. 4/6/8 train-only metadata/hash preflights passed.
Actual three-node CUDA/NCCL, memory, throughput and FLUX parity remain pending;
no GPU job was started by this preparation, no training-fit metrics exist.

Launch the same run on all three user GPU nodes with explicit NODE_RANK=0/1/2,
NNODES=3, MASTER_ADDR=10.247.94.142, MASTER_PORT=29683 and the qualified
FLUX_TAIL_V3_PYTHON. Commands are in the readiness report. No automatic SSH,
remote job kills, retries, trainfit or formal training. Six-card qualification
hard cap is 7200 seconds / 12 total used GPU-hours (+30s cleanup), preserving
the prior GPU budget. Count SUM(2*node_elapsed) across three node logs, not
six times each log. Startup/rendezvous waiting counts against each node cap.
Do not promise the old single-node ETA. Independent/manual fresh trainfit is
still gated by matching qualification; six-card cap 48000 seconds / 80 GPU-hours.

## CURRENT VERIFIED: destination CPU migration complete; GPU pending — 2026-10-30

This section updates the migration prerequisites below. Read
`reports/foundir_flux_tail_v3_workspace_migration_2026-10-30.{md,json}`.
The new workspace was actually checked: 147 SHA checks, 67 CPU tests,
two-rank second-order exact resume, and regenerated 4/8-GPU source/cache
preflights passed. V3 had only CPU metadata; old metadata was archived,
not rewritten. No V3 GPU run/result exists from this session; CUDA count=0.

The two V3 launchers now derive ROOT from their script location and require
an explicit `FLUX_TAIL_V3_PYTHON`; they no longer hardcode the old workspace.
The destination's `.conda/gpu-train-stable/bin/python` passed CPU runtime
checks (Python 3.12.13, torch 2.7.0+cu128). Only this copied interpreter's
execute permission was restored, 0644→0755; dependencies were not changed.
Its CUDA execution and baseline parity are still unqualified. Do not use the
stale `.venv` configurations or silently substitute system Python.

On the user's homogeneous eight-H100/H200 node, from the new root, the
prepared engineering command is:
`FLUX_TAIL_V3_PYTHON="$PWD/.conda/gpu-train-stable/bin/python" bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh qualification qualification_v1 8`.
This session prepared the command only; no GPU task or long training started.
Keep the 5400-second / 12-used-GPU-hour qualification cap and all train-only
restrictions. No automatic trainfit/formal. Four-GPU compatibility remains.
Unrelated local changes and all runtime assets remain; the main checkout was
not reset/pulled, and focused updates are committed from an isolated worktree.

## CURRENT AUTHORITY: new-workspace migration and V3 handoff — 2026-10-30

- The sole active project root is now
  `/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`.
  This explicitly supersedes all older HDD3 / agent-driven-world-model workspace
  rules below. The old workspace is a frozen migration source, not a fallback
  execution directory. Do not launch or resume jobs there.
- GitHub `Jason-Wang313/zhenhaowangbackup`, `origin/main`, is the authoritative
  source for code, decisions and handoff updates. The user already copied the
  files to the new root; subsequent edits in the old root do NOT synchronize.
  Preserve copied weights/data/caches and all unrelated dirty files. GitHub
  does not contain those large runtime assets and cannot replace their copy.
- Read `reports/foundir_workspace_migration_handoff_2026-10-30.md` FIRST, then
  the top of `reports/foundir_current_handoff_2026-09-23.md` and the V3 readiness
  report. Lowercase `agents.md` and all older ACTIVE/LIVE headings are historical
  wherever they conflict with this section. Never revive their old priorities.
- Immediate next action: safely synchronize the relevant GitHub files, verify
  the copied assets, fix and qualify relocation of paths/runtime/cache links,
  then run the V3 **8-step engineering qualification** on one 8xH100/H200 node.
  Do NOT run the current launcher verbatim: its ROOT/PYTHON still reference the
  old workspace, and changing bound files requires a recorded migration.
- V3 is the train-only relative FLUX-tail energy repair, NOT the failed V2
  25-step run, old GT-only oracle, Restormer residual, MoE, Rain or EBT Generalist.
  Latest verified evidence at this handoff: 67 CPU tests and CPU DDP exact
  resume passed before migration; no V3 GPU log/result was found in the source
  workspace. Inspect the destination before assuming no newer run exists.
- Only after a matching qualification passes may a separately/manual-launched,
  fresh 200-step train-fit run be considered. No qualification-weight reuse,
  no automatic formal training, no automatic budget increase. Final energy-only
  goals remain +0.30 dB PSNR and 9.928107050377874% relative LPIPS reduction.
- Default: all eight GPUs on one homogeneous H100/H200 node; retain four-GPU
  compatibility. No cross-node jobs. 15 CPU cores / 500GB RAM is acceptable to
  qualify, not a measured zero-slowdown guarantee. Preserve global batch 32.
- The last validated Python environment was
  `/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3`.
  Its availability on the new node is UNKNOWN, and copying the project folder
  does not copy this sibling environment. Verify the interpreter/dependencies;
  do not guess a new environment path or silently use system Python. A missing
  runtime blocks GPU launch until an equivalent environment is qualified.
- Never use official/sealed/internal/parent-test or final-1859 development
  pixels for training, tuning or selection. V3 currently uses only its original
  24 training sources; all probes are training fit, not generalization evidence.
- Report in simple Mandarin: outcome first, facts versus hypotheses separated.
  Commit/push focused milestones using explicit file lists and secret scanning;
  never upload checkpoints, datasets, caches, credentials or live logs.

## ACTIVE: train-only relative FLUX-tail energy repair V3 — 2026-10-29

用户在第一性原理分析后明确授权execute。本节覆盖旧V2终态的“没有新任务授权”；
不恢复旧25步任务、Restormer、MoE或EBT Generalist。当前执行修复的第一阶段：
先确认energy在训练来源上能学会一次正确修改；不是直接上正式泛化训练。

保持冻结Noise FLUX step400、50步、512图、alpha=.30和双uint8口径。
energy architecture保持原1,010,274参数，不加RGB输出网络。仅原24训练来源可读，
旧8张检查图、calibration池、最终1859开发集及所有保护像素均不读。
所有训练集probe明确标记training_fit_only，不得包装为来源分离/泛化指标。
用户最终+.30dB和LPIPS下降9.928107050377874%的双目标不变。

新训练从fresh模型/Adam开始；旧V2 step25只作为冻结反例生成器，绝不seed新训练。
初始反例在24训练来源上由旧energy自行走4步生成，再用训练GT标注好坏；
不先假定所有生成候选都是坏图。学习期间也刷新当前模型自身产生的4步压力轨迹，
这些无边界轨迹只是训练反例采集，不是部署时允许连续走4步。
现有候选的log MSE/log LPIPS与真实RGB导数只计算一次，FP32缓存。
一半训练pair含self-mined候选；加强同图数值差分监督、保留低权重绝对分数、
梯度幅值拟合，并增加有零梯度保护的方向监督。记录各loss的真实参数梯度范数。

qualification_v1：8工程步，step4刷新当前模型反例，最后做8训练图的一次真实FLUX尾段probe。
这8步只验收链路、显存、保存、反例刷新；未改善不能当整条路线失败。
trainfit_v1：独立fresh200步，50/100/150刷新反例，最终做相同训练图probe。
代码已具备此入口，但必须先有hash匹配的qualification工程通过结果，并手动启动；
不能自动接续，不能从qualification权重初始化，不因工程通过就声称科学可行。

同时报告原FLUX、FP32无引导、原始一次建议及受限一次建议；受限版本最多一次，
限制相对原proposal的RGB RMS和通道均值偏移，界限从训练GT轨迹首步的95分位×1.25
导出并保留1/255量化下限。这只是训练侧防失控约束，未经独立校准，不是no-harm保证。
无候选时原样返回原FLUX；全部拒绝=没有收益。独立校准、来源分离确认与多步扩展
留到训练侧学会之后，当前没有实现/启动它们，不得暗中用检查图选阈值。

默认用满单节点8×H100/H200，同时支持同型号4卡；global batch32对保持不变。
每完整optimizer step原子保存模型/Adam/每rank RNG/采样位置/哈希；相同命令恢复，
不同world_size严格恢复不支持。每张候选缓存/probe独立落盘，可跳过已核验记录；
异常中断留下的孤立tensor文件保留，重试用新唯一文件名，不覆盖。旧目录不动。
67项CPU回归和两进程真实二阶DDP精确恢复已通过；真实训练缓存预检通过。
Codex没有GPU，尚未启动本轮GPU；4/8和H100/H200的CUDA显存/速度仍待实测。

八卡资格粗估35–75分钟（4.67–10使用GPU-hours），硬cap90分钟/12使用GPU-hours。
四卡暂估70–150分钟（同GPU-hours），硬cap180分钟；不是实测倍速保证。
八卡200步训练拟合粗估7–10小时/56–80使用GPU-hours，硬cap10小时；
四卡硬cap20小时/80使用GPU-hours。须按新资格实测更新，不能自动扩预算。
FP32标签/导数/反例/逐步checkpoint预计占25–40GiB，不能上传这些大文件至GitHub。

默认八卡资格命令：
bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh
四卡须独立ID：
bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh qualification qualification_4gpu_v1 4
未来trainfit第四参数可明确指定通过的qualification ID，不把四卡绑定八卡目录。
终态须TAIL_RELATIVE_V3_RESULT_VERIFIED和TAIL_RELATIVE_V3_END status=0。
详见reports/foundir_flux_tail_relative_energy_v3_readiness_2026-10-29.{md,json}。

## TERMINAL RUN: FLUX-tail qualification complete; learned guidance harms quality — 2026-10-29

本节覆盖下方LIVE状态：qualification_8gpu_v2已于2026-10-29T00:50:19Z正常结束，
8×H100，25/25训练步、8/8工程检查图、RESULT_VERIFIED、launcher status0。
3425秒=57分05秒=.95139普通小时/7.61111使用GPU-hours；旧四卡成本另计，
两次合计11.36222使用GPU-hours。没有待续训任务，不自动formal或扩大预算。

独立核验了step0–25共26份检查点文件SHA、全部8图结果文件SHA、源码绑定及7个
报告项的均值。每个完成步均保存；最终checkpoint SHA256为
2eebdafb0b6a30829de48792b8eced1037cbfcfc3e0114ee986f1f8b70da4ed1。
训练均值124.46875秒/步，八卡参数一致，峰值allocated显存31.8921GiB/卡。

工程通过不等于恢复有效。八图原FLUX为38.6109929085/.07752364664；
energy一次为36.7593092918/.07801964320；预定16次终点为
19.7870512009/.11268278910，即PSNR下降18.82394dB、LPIPS变差45.3528%。
无引导FP32对照38.6112513542/.07753595291，排除精度路径本身造成这种量级崩坏。
8/8图第一次PSNR就下降，16次后8/8两项都恶化；128次更新全被接受且全选.01最大步幅。
预测评分却认为质量改善：当前energy判断与真实质量脱节，自检未能拦住有害更新。

训练末5步梯度匹配loss均值1.000584、MSE/LPIPS方向cosine约-.051905/-.001187。
这些是训练采样诊断，不是评价图梯度测量；显示25步后尚无学会正确修正方向的证据。
推理将非零梯度归一化，因此微弱但错误的方向仍能触发大步更新；同一不准energy
再给更新打分，产生反复错误接受。完整根因尚未隔离，不能说已排除所有实现问题。

此次仅24来源/25步工程资格，不能据此宣判所有learned-energy路线不可能，也不能
因工程通过直接正式长训。资格权重不可seed formal。所有历史结果、缓存、模型保留。
后续应先确认训练侧的梯度/排序能否真正学会，再决定训练规模和部署约束；本次检查
未授权新GPU任务、调整阈值或改写已绑定实验。
报告：reports/foundir_flux_tail_energy_resumable_v2_terminal_2026-10-29.{md,json}。

## LIVE: eight-H100 training and per-step save verified — 2026-10-28

qualification_8gpu_v2 已在用户节点真实启动，不再是“待GPU运行”。
2026-10-28T23:53:14Z启动；八张卡均实测为H100 80GB，rank0–7全部参与。
旧四卡任务已精确停止：最后日志step7，23:53:44Z以status1结束，属主动切换，
不是新八卡训练崩溃。32图缓存全部复用，旧日志及快照保留；旧step1–7从未
保存optimizer checkpoint，不能恢复，当前是同初始化重算而不是无损续训。

已独立读取并核验step0/1/2不可变文件的SHA256；step2模型参数有限且已更新，
Adam共236个参数状态均为step2，next_global_slot=64，8份CPU/CUDA RNG齐全，
八卡参数一致。每卡峰值allocated显存31.8915GiB。前两步分别125.5294/124.6914秒，
接近旧四卡246–250秒的两倍速度；global batch/科学方案不变。
核验时2/25步，剩余训练约48分钟（约6.4使用GPU-hours），评价时间另计。
旧四卡已消耗3376秒/.9378普通小时/3.7511使用GPU-hours；若计整台八卡占用，
为7.5022整机GPU-hours。不得从历史账中删除这笔成本。

这是八卡启动、有限训练和逐步落盘验收，不是25步完成或科学成功。
GPU中断后真正重启尚未实测；仅CPU二阶DDP的精确恢复测试通过。
不要为测试恢复而打断正常任务，不改正在运行的绑定源码/配置。
同run-ID启动命令可恢复最后完整步，未完成的一步可能重算；不自动formal/扩预算。
当前未有新PSNR/LPIPS结果，早期gradient cosine接近零，不宣称energy已学会。
详见reports/foundir_flux_tail_energy_resumable_v2_live_2026-10-28.{md,json}。

## ACTIVE: use all eight GPUs; durable per-step FLUX-tail energy qualification — 2026-10-28

用户明确要求切换到八卡并保存进度。本节覆盖下面四卡首轮资格的执行优先级；
科学方案仍是同一个25-step learned-energy qualification，不扩训练规模、不自动formal。
默认单节点8×H100/H200，显式CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7。
仍保留4卡兼容入口，但兼容4卡不等于默认只用4卡；整机八卡空闲时默认用满八卡。

旧qualification_4gpu_v1已完成全部32图缓存（24训练/8检查），四卡训练实测
每步约246–250秒；原30–60分钟全程估算与90分钟旧超时不足，不能继续照此估算。
旧程序只在step25保存权重，当前中间optimizer状态只在GPU进程内，无法热补保存。
不要谎称换八卡是旧中间状态的无损续训。全部旧缓存/日志保留；新入口复用缓存，
从同初始化重算旧未保存训练步。切换时记录旧最终日志步数及日志快照，不包装重算。
Codex无GPU/远程进程控制通道；截至准备完成，尚未停止旧GPU进程或启动新八卡任务。

新任务：checkpoints/foundir_flux_tail_energy_resumable_v2/qualification_8gpu_v2。
已完成元数据与全部32缓存SHA核验，只链接原缓存，不重新生成、不读保护像素。
旧绑定源码/配置/缓存均不修改。新元数据同时记录原producer和新consumer代码身份。
global batch仍32候选对；八卡每rank累积4对，四卡累积8对。损失、lr、512裁图、
25步、来源划分、无GT推理和+.30dB/LPIPS下降9.928107050377874%目标不变。

每个完整optimizer step（含step0）原子保存模型、Adam、各rank Python/torchCPU/CUDA
RNG、显式warmup状态、next_global_slot、累计训练日志及哈希。fsync后发布不可变
checkpoint/receipt，再更新latest指针；旧步全保留。读取最高完整receipt，不因latest
指针滞后丢进度；损坏校验失败时禁止悄悄回退。未完成的一步可能需要重算，不承诺
断电瞬间未落盘计算可恢复。默认相同run-ID重复命令恢复最后完整步，不能新建ID
冒充续训。当前已验证同world_size恢复；换world_size恢复明确拒绝，不能冒称bitwise等价。
每张评价图完成后原子保存结果，重启跳过校验通过的已完成图；GT仍只在选择结束后计分。

已通过52项CPU回归，以及两进程真实二阶DDP的Adam/RNG恢复：
恢复后下一步参数与不中断路径逐元素相同。八卡缓存导入真实预检通过。
这不是八卡CUDA资格通过；新八卡训练、显存、吞吐、指标均待用户节点运行。
不因25步指标无收益就宣称路线终判，也不因CPU通过就声称科学成功。

八卡重算25步+评价粗估55–75分钟，约7.3–10.0使用GPU-hours（未实测八卡加速）；
旧四卡已消耗成本另计。新每次尝试硬cap：八卡5400秒、四卡10800秒，均12使用GPU-hours，
另最多30秒清理；超时保留所有已提交步骤，不自动重启/扩预算。若独占八卡仅用四卡，
须另计整机资源成本。整个历史累计成本不得被新run-ID清零。

用户GPU节点一条命令：
bash smoke/start_foundir_flux_tail_energy_resumable_v2_tmux.sh
默认run-ID qualification_8gpu_v2、8卡；同命令将来用于断点恢复。
启动器先准备/核验缓存，再通过精确脚本名、train phase、run-ID、4卡参数和cwd识别
旧进程，核对GPU没有无关任务，只SIGTERM该旧torchrun并等待退出，不使用宽泛pkill。
旧内存权重无法保存的事实会明确打印并记录；如果旧step25 checkpoint意外已经存在，
拒绝丢弃/重算，必须先处理其迁移。旧日志快照保存到新run目录，旧目录不删除。
随后获取节点锁、检查全部八卡空闲，执行train→evaluate→verify。
日志logs/foundir_flux_tail_energy_resumable_v2_qualification_8gpu_v2.log追加保存各次尝试。
终态需要TAIL_ENERGY_V2_RESULT_VERIFIED和TAIL_ENERGY_V2_LAUNCH_END status=0。
报告reports/foundir_flux_tail_energy_resumable_v2_readiness_2026-10-28.{md,json}。

## ACTIVE: learned FLUX-tail quality energy — 25-step qualification — 2026-10-28

用户在24来源GT确认完成后明确授权“下一步，执行”。本节覆盖下面全部历史
ACTIVE/TERMINAL中的优先级：当前只执行新energy的25步工程资格，不重跑GT诊断，
不恢复Restormer residual、MoE或旧EBT Generalist，不自动正式训练。
上一阶段GT平均+.98618dB/LPIPS下降31.562%只是可行性，不是energy已学会。

冻结FLUX/LoRA、VAE和LPIPS。新1,010,274参数条件quality energy预测log MSE与
log LPIPS，输入LQ、原FLUX结果、候选、差分、真实每图Qwen退化等级及shifted t。
同时监督分数、候选图像梯度和同来源候选排序；二阶训练只穿过小energy。
它不输出RGB，不加Restormer；推理梯度经真实FLUX末两步/VAE传回index48 latent。
候选接受/停止只看预测energy；GT仅在完整推理结束后算评价指标。
预测安全不等于实际no-harm，必须如实报告退步，不能用GT挑回好图。

保持Noise step400/50步/512裁图/原seed/CFG1/reference time10/alpha=.30/双uint8。
原54,585条Noise training记录、1,755来源排除旧32来源后，冻结来源划分：
train1467、calibration128、engineering_evaluation128。资格仅用24新训练来源、
8新检查来源，校准池不读像素。检查来源只对energy训练分离，FLUX可能见过，
不是系统未见过测试集。最终1859开发集及official/sealed/internal/parent-test禁止。
训练缓存包含连续与实际量化候选、随机latent扰动及固定GT轨迹；检查图无训练轨迹。
全部源码/数据/权重绑定，旧V1/V2/V3/24图确认文件与结果不改。

25 optimizer steps，fresh AdamW lr1e-4，warmup5，global batch32候选对/64视图。
4卡每rank累积8对，8卡累积4对；完整512训练，不改科学口径。
同一energy引导五种方向混合、五种步幅，最多16轮；报告1/2/4/8/16，主终点16。
真正目标仍为相对匹配FLUX PSNR+.30dB、LPIPS下降9.928107050377874%。
25步及8图均为工程检查，不是正式科学结论；不据此挑配方或自动长训。
资格checkpoint明确不得seed正式训练；后续规模须由实测吞吐及独立方案决定。

已完成：46项CPU回归、两进程真实CPU DDP二阶反传/累积一致性、
Python/shell静态检查、4/8完整真实权重与数据元数据预检（不解码图片）。
GPU尚未启动；Codex节点无GPU，不把CPU通过当CUDA显存/性能/指标通过。
单节点同型号4/8×H100/H200，每卡至少75GiB；不混卡、不跨节点。
四卡H100粗估30–60分钟，2–4使用GPU-hours；独占八卡时另计4–8整机GPU-hours。
硬cap四卡5400秒/八卡2700秒，均6使用GPU-hours，另最多30秒清理。
八卡/H200没有本任务实测，不承诺精确加速；不覆盖已有run，不同时跑两套。

四卡启动：
bash smoke/start_foundir_flux_tail_energy_qualification_v1_tmux.sh qualification_4gpu_v1 4
入口显式传CUDA_VISIBLE_DEVICES；八卡须新ID及参数8。
阶段hardware→prepare→cache→train→evaluate→verify；无自动formal。
日志logs/foundir_flux_tail_energy_qualification_v1_<id>.log；
输出checkpoints/foundir_flux_tail_energy_qualification_v1/<id>/。
必须看到TAIL_ENERGY_RESULT_VERIFIED和TAIL_ENERGY_LAUNCH_END status=0；
tmux创建不等于训练开始。当前无新GPU结果或实测成本。
配置SHA256：22da9b3fb51f4705cc917b3d3c8bd1e1fea8bdd3193fff205e198c57ef462c71。
详见reports/foundir_flux_tail_energy_qualification_v1_readiness_2026-10-28.{md,json}。

## TERMINAL: fixed GT policy passes on 24 separate training sources — 2026-10-28

confirm24_4gpu_v1 已正常结束，不再启动/续跑。下面ACTIVE/readiness记录均为历史；
当前没有待完成GPU任务，不自动启动energy训练或恢复旧Restormer/MoE/EBT。
4×H100 80GB，2026-10-28T21:58:30Z–2026-10-28T22:17:00Z，launcher status=0，
CONFIRMATION_RESULT_VERIFIED及独立标准库重算均通过。
共1110秒/18分30秒/.3083333333普通小时/1.2333333333使用GPU-hours；
若独占八卡整机则2.4666666667整机GPU-hours（是否独占未确认）。
峰值allocated显存15.306301593780518GiB。仅本任务四卡H100实测通过。

24个来源、缓存索引8–31、7个报告项完整，24/24基线逐像素复现。
来源/GT/LQ哈希与旧八图分离；不把八图混入平均值，不在24图重新挑方案。
固定唯一主终点mixed_psnr_budget_step16：
原部署FLUX为40.669039726257324/.05249165142110238，
GT引导为41.655218760172524/.03592422348447144，
即PSNR+.9861790339151995dB、LPIPS下降31.56202460410801%。
超过+.30dB/LPIPS下降9.928107050377874%的原平均双目标。

22/24单图通过，24/24两项均未比历史基线变差，但不是每张有充足余量。
索引24接受0次，仅FP32无引导精度差异（+.0002479553dB/LPIPS降.0151213%），
不能归为引导收益；索引27接受1次，仅+.0255088806dB/降.0884384%，也未过。
索引17 PSNR仅比目标高.0007125854dB，索引18 LPIPS降10.7094849%，边际仍小。
4/8轮平均也过，但只是预定描述性报告，主终点仍16轮，不事后挑最优。

独立核验334次尝试、8350候选、327次接受、8023次拒绝，逐次选择/停止、
全部均值、源码配置哈希、sample/rank及trace一致。V1/V2/V3绑定文件未改。
policy含18704次FLUX forward（包括checkpoint重算）、668次目标backward，
不是廉价16次前向。成本和全部逐图结果均保留。
仍是训练来源GT确认：所有方向/接受/停止用GT，没有训练energy，没有VLM引导，
不是部署、泛化或论文创新成功。这24来源曾参加旧32图工程资格，不是未用过的测试集。
没有读取保护/最终1859条开发集像素，没有恢复图像定性审核。
后续energy学习需独立冻结方案和用户决策；禁止把本GT成绩当可部署收益。

结果SHA256：37f13289195eb6abd4fc4127f797008f7e6843ebd11957123b7245fcfa735137。
报告：reports/foundir_flux_tail_confirmation_v1_terminal_2026-10-28.{md,json}。

## HISTORICAL PLAN: fixed-policy GT confirmation on 24 other training sources — 2026-10-28

用户已授权执行。当前唯一新任务是固定 V3 的 mixed_psnr_budget、最多16次更新，
在原缓存索引8–31的24个训练来源上确认；此前V3终态/历史计划不再是待执行任务。
不重新挑方向/步幅/窗口/检查点，不恢复旧Restormer/MoE/EBT或最终像素下降，
不训练energy、不自动长训。V1/V2/V3绑定源码、配置、结果保持不动。

复用V3原始diagnose_record及trajectory核验，不改梯度、rollout、量化或接受规则。
保持Noise step400、50步、512裁图/seed、CFG1、reference time10、alpha=.30、
双uint8量化、index48真实两步rollout、BF16模型和FP32 latent。
只保留mixed五方向与psnr_budget一条轨迹，仍按原25候选/轮、最多16轮；
报告1/2/4/8/16及两个对照共7项，主终点唯一mixed_psnr_budget_step16。
目标相对这24图自己的原部署基线：PSNR+.30dB、LPIPS下降9.928107050377874%。
不沿用八图绝对数值、不把八图混入均值、不逐图挑最佳检查点。
未接受修正仍保留FP32无引导结果，禁止GT事后换回历史图美化指标。

24来源与原八图的source、GT/LQ哈希、record ID均分离；元数据预检不解码图片。
这些来源曾参加旧32图工程资格，不是未用过的正式测试集。
全部方向/候选/停止仍依赖GT；只是训练来源机制确认，不是energy学会、
VLM贡献、部署、泛化或论文成功。全部测试/封存像素及最终1859条开发集仍禁止。
成功不自动授权后续训练；全部候选及拒绝/forward/backward成本继续记录，
不能将16轮更新说成只多16次前向，也不能保证LPIPS提升等于视觉自然。

单节点同型号4/8×H100/H200，每卡至少75GiB，4卡每rank六张/8卡每rank三张。
34项CPU回归、Python/shell静态检查、真实权重与全部24缓存的4/8卡预检均通过；
新任务GPU未启动，Codex无GPU。仅旧V3四卡H100实测，不扩称本任务GPU验收。
暂估四卡H100 15–30分钟（.25–.50普通小时，1–2使用GPU-hours），
若独占八卡整机仅用四卡另计2–4整机GPU-hours；这是粗估，不是实测。
硬cap：4卡1小时/8卡.5小时，均4使用GPU-hours，另最多30秒清理。
独占八卡只用四卡时，cap对应8整机GPU-hours；八卡/H200未有本任务实测ETA。
四卡命令：bash smoke/start_foundir_flux_tail_confirmation_v1_tmux.sh confirm24_4gpu_v1 4
八卡可用独立ID confirm24_8gpu_v1 8；二选一，不重复同时跑。
日志logs/foundir_flux_tail_confirmation_v1_<id>.log；
结果checkpoints/foundir_flux_tail_confirmation_v1/<id>/result.json。
必须核验CONFIRMATION_RESULT_VERIFIED及CONFIRMATION_LAUNCH_END status=0，
tmux创建不等于GPU开始。全部结束前不自动扩大任务。

配置SHA256：c3ac68f96f2231a434d90b5fa733948d3c9dbe3bb08bfc8f86f839e4b97d3dcf。
V3父结果SHA256：06eed91ab7c2c0942b221a14d4c295d1e269bf57b17a8d79354ec9be074d24e4。
详见reports/foundir_flux_tail_confirmation_v1_readiness_2026-10-28.{md,json}。

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

## TERMINAL: V3 GT mean joint targets PASS on eight training sources — 2026-10-28

budget_4gpu_v1 已正常结束，launcher status=0，BUDGET_RESULT_VERIFIED。
4×H100 80GB，2026-10-28T21:11:43Z–21:26:53Z；总910秒/15分10秒，
.2527777778普通小时、1.0111111111使用GPU-hours。
若独占整台八卡则2.0222222222整机GPU-hours（是否独占未确认）。
峰值allocated显存15.307990GiB。八来源×22报告项、8/8基线逐像素复现、
sample/rank/trace、源码配置哈希及独立均值/选步重算均通过；
独立核验367次尝试更新、6115候选、352次接受。仅四卡H100实测通过。

原部署基线39.64431142807007/.07384782587178051。
固定 mixed_psnr_budget_step16 为40.34524202346802/.053707223385572433，
相对原基线PSNR+.7009305953979492dB，LPIPS下降27.27311501516311%。
超过原+.30dB/LPIPS下降9.928107050377874%的平均目标，确实有平均余量。
不能继续描述为“GT平均双目标尚未通过”，也不能扩称energy/泛化成功。

同16次报告：balanced_monotone 为+1.1016478539dB/LPIPS降9.7295917357%，
仍未通过；balanced_psnr_budget 为+.7904086113dB/降13.9110398133%，通过；
mixed_monotone 为+1.4849891663dB/降14.7461070083%，通过。
这是各自同一个方案的两项指标，禁止拼接不同方案的最好值。
mixed_psnr_budget 的4/8次报告为+1.1330642700dB/降11.1080559356%、
+.9448838234dB/降18.8657205717%，也通过；未逐图选最佳检查点。

主方案7/8张单独双目标通过，8/8两项优于原基线。
样本0仍只有3.362307% LPIPS降幅，接受3次后停；其余七张到16次上限。
样本1/3 PSNR只比+.30目标多.0383522/.0071556dB，平均余量不等于逐图余量。
全部指导、候选比较、接受/停止使用训练GT，没有训练energy、没有VLM引导、
没有部署选择器/新泛化结果或论文创新通过。八图是多轮机制诊断来源。

四组共享新排序和更新上限，但多方向有额外计算，不能宣称等算力优势。
主方案共2900候选、2785次拒绝、6496次FLUX前向计数、232次目标反传，
还不含共同基线复现/前缀/加载。不是只多16次前向的廉价部署TTS。
本轮没有恢复图像的定性审核，不可用LPIPS直接保证视觉纹理自然。

无自动后续任务；旧ACTIVE/未启动记录均为历史。下一步应先冻结单个方案
和预算，在缓存另外24个训练来源索引8–31做确认，再决定energy学习；
V3未读取这些像素，不自动扩展/重跑/加步/长训，不恢复旧分支。
24来源仍是训练来源机制确认，不是最终held-out泛化，也不保证正式训练必过。
报告及全22项均值：reports/foundir_flux_tail_budget_v3_terminal_2026-10-28.{md,json}。
原始result SHA256：06eed91ab7c2c0942b221a14d4c295d1e269bf57b17a8d79354ec9be074d24e4。

## HISTORICAL PLAN: GT-only PSNR-budget / direction diagnosis V3 — 2026-10-28

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

## Latest Rain qualification update — 2026-10-25

The 25-step FP32 precision qualification completed successfully at
2026-10-25T20:39:00Z on 8xH100, status 0. Development improved from
32.08470176034529/.0888220098754391 to
32.61941190970921/.08720199926523492, still failing the 34.2/.08 gate.
The user authorized the independent 300-step fit diagnostic, not full-data
training. Initialize from the original step-2,000 LoRA with fresh optimizer,
never the qualification checkpoint. Command on the GPU node:
`bash smoke/start_foundir_rain_flux2_precision_fit_v1_tmux.sh fit fit_v1`.
Estimate 15-20 wall minutes / 2.0-2.67 allocated GPU-hours. Fit has no launch
log at this update; this Codex environment has no GPU. Keep Rain gate first,
image-only MoE second, VLM router third; no pixel refiner.
See `reports/foundir_rain_precision_qualification_2026-10-25.md` for evidence.

## Active workspace and runtime

- The sole active project root is `/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang` (migration decision 2026-10-30).
- Use the validated project Python environment; its last known location and the required new-node verification are stated in the CURRENT AUTHORITY section above. Do not silently substitute system Python.
- Preserve all dirty and untracked project files. Do not reset, clean, or overwrite unrelated work.
- Keep prior workspaces frozen; no execution or automatic synchronization back to them. Historical paths below are provenance, not active workspace choices.
- Never use official, sealed, parent-test, or internal-test pixels for training, hyperparameter selection, checkpoint selection, or early stopping.

## ACTIVE: Rain FLUX precision repair and training-fit diagnosis — 2026-10-25

The user rejected the proposed small pixel refiner and authorized the
precision/fit diagnosis. This supersedes the refiner suggestion below.
Keep the sequence Rain DiT >=34.2 PSNR / <=.08 LPIPS, then working image-only
MoE, then VLM tokens. Do not train the observation-anchored pixel head.

The prior run stored trainable LoRA weights and both Adam moments in BF16
at learning rate 1e-5. At step 3,000, 64.2164% of LoRA A entries were exactly
unchanged from its parent. This is evidence to investigate update rounding,
not proof that it explains the entire failure. Training kept the encoded
observation FP32 for subtraction while inference cast it to BF16 first.

The new isolated implementation is configured by
`configs/foundir_rain_flux2_precision_fit_v1.json`: FP32 LoRA parameters and
Adam states, BF16 CUDA compute, and one shared FP32 endpoint subtraction in
training/evaluation. Start independently from the hash-bound best Rain
direct-endpoint step-2,000 LoRA with a fresh optimizer. Retain the eight
parent loss terms and weights. Use 32 deterministic training source groups
with fixed 512 crops; evaluate the exact same training inputs and the
unchanged 100-source development gate at 512. The initial evaluation also
reports the legacy subtraction path on the same checkpoint.

Run a 25-step qualification first on one homogeneous 8xH100/H200 node. Its
checkpoint cannot seed the independent 300-step fit diagnostic. Fit requires
a passing qualification with matching config, source and sample hashes.
Qualification/fit milestones are [0,25]/[0,100,300]. No automatic full-data
training follows. Passing the 32-image numeric target is evidence of fitting
ability only; development PSNR/LPIPS remain separate. This is not a matched
long-training BF16-versus-FP32 ablation and cannot attribute all gains to
precision alone. Both launchers preserve existing run directories/logs.
Read `reports/foundir_rain_flux2_precision_fit_v1_readiness_2026-10-25.md`.

## ACTIVE: Rain DiT gate, then image-only MoE, then VLM router — 2026-10-25

This user decision supersedes the older MoE-first scheduling text below. The
order is strict: (1) repair the Rain FLUX/DiT expert until it passes the frozen
Rain development gate (`PSNR >=34.2`, `LPIPS <=0.08`); (2) build and validate
an ordinary working image-only restoration DiT-MoE using successful experts;
(3) only then add pooled VLM output tokens through a small projector to
adjust that same router. Blur DiT has already passed its SOW gate; do not
repeat Blur training. The failed 1,000-step image-only MoE is not a usable
baseline, and its VLM-assisted arm must not be launched.

The full-data direct-endpoint Rain FLUX formal run completed on 2026-10-21 at
3,000/3,000 steps (8xH100, `6.470578311735836` wall hours,
`51.76462649388669` allocated GPU-hours). Its frozen 100-record development
metrics were `31.900863136522702/.08921627756208181`, failing both gates.
The best predeclared checkpoint was step 2,000 at
`32.085315339143946/.08882607079111039`; 3,000 regressed. Do not repeat or
extend the same direct-endpoint recipe. On the 3,000-step result, all 24
records with input PSNR >=34 dB lost PSNR after restoration; 14 of the other
76 records also lost PSNR. A no-training, fixed 0.75 blend of input and the
step-2,000 output reaches only about 32.545 dB PSNR on saved development
triptychs. Even a non-deployable GT-only best blend per image reaches only
33.444 dB. These are quantized-export exploratory calculations, not official
model results. Simple strength control cannot repair the gap. Diagnose the
Rain action/representation and verify a distinct fix before another GPU run.
See `reports/foundir_rain_dit_gate_priority_and_failure_audit_2026-10-25.md`.
An observation-anchored correction head now exists and passes three CPU unit
tests, but it has no Rain trainer or GPU result. Do not call the gate passed.

## ACTIVE: advisor-approved minimal VLM-token-to-MoE-router pilot — 2026-10-25

This section supersedes every older Rain, EBT, energy, specialist, and
explicit-VLM scheduling priority below. The user reported that the advisor
approved the minimal architecture and explicitly authorized execution. Do not
restart an older branch unless the user explicitly returns to it.

Implement and test only the minimal advisor proposal:

1. start from one ordinary restoration DiT-MoE with one expert bank;
2. retain its image-feature router and restoration experts;
3. pool frozen VLM output tokens/features, pass them through one small MLP,
   and use the result to adjust the original router's expert logits;
4. select/weight the same MoE experts and combine their outputs normally.

There must not be a second MoE, a bank of LoRA experts, EBT/energy refinement,
atomic-token composition, counterfactual routing, hard diagnosis priors, or
block/timestep scheduling in this first experiment. The matched comparison is
image-only routing versus the identical MoE with the VLM score branch enabled.
Use the same backbone, expert bank, data, optimization, and inference budget.

CPU/static implementation and a training-only data preflight are complete.
The frozen 25-step eight-GPU qualification is defined by
`configs/foundir_vlm_token_moe_minimal_qualification_v1.json`. Run the
`image_only` and `vlm_assisted` arms separately; neither qualification may seed
formal training or auto-launch later work. Keep official, sealed, parent-test,
and internal-test pixels out of development. Report wall-clock hours and
allocated GPU-hours for every GPU run.

The `image_only` qualification completed successfully on 8xH100 at
2026-10-25T15:38:24Z. It ran 25/25 steps in `.0563902` wall-clock hours and
`.451122` allocated GPU-hours, with finite training, no OOM/NCCL/runtime
failure, and `12,653,083,136` bytes peak allocated memory per rank. The VLM
logit adjustment was exactly zero, expert weights summed to one within
`1.1921e-7`, and the checkpoint hash was verified. This is engineering
qualification only, not a restoration-quality result, and its checkpoint may
not seed formal training. The next active action is the fresh matched
`vlm_assisted` 25-step qualification; do not launch formal training yet.

The independent `vlm_assisted` qualification also completed successfully on
8xH100 at 2026-10-25T15:53:30Z. It ran 25/25 steps in `.0460070` wall-clock
hours and `.368056` allocated GPU-hours, with no OOM/NCCL/runtime failure and
`12,669,009,408` bytes peak allocated memory per rank. Its VLM branch was
active: maximum logit adjustment was `.75` and the output-head L1 update was
`.78411865234375`. Expert weights remained normalized within `1.1921e-7`.
Both engineering arms have therefore passed. Their last-five training losses
(`1.1534720` image-only versus `1.1534493` VLM-assisted) are not restoration
metrics and must not be presented as scientific improvement. Neither
qualification checkpoint may seed formal work. The next step is to freeze a
matched scientific train/evaluation protocol before any longer GPU run.

That matched protocol is now frozen and CPU-qualified in
`configs/foundir_vlm_token_moe_matched_formal_v1.json`. Each arm starts
independently from the same pretrained FLUX.2 weights, trains for 1,000
optimizer steps (32,000 fit presentations, about 14.2 fit-set exposures), and
uses the same data/order/objective/compute. Calibration is evaluated at steps
500 and 1,000; the source-disjoint 450-record holdout is read exactly once at
the fixed final step. Metrics are PSNR, AlexNet LPIPS, per-degradation results,
and routing traces. Four protocol unit tests and both arm-specific CPU/data
preflights pass. No protected data was accessed. Run the arms separately with
`smoke/run_foundir_vlm_token_moe_matched_formal_v1_8gpu.sh`; start with
`image_only`. The launcher is resume-safe within the same arm but forbids all
qualification checkpoints and never launches the other arm automatically.
Estimated cost per arm is `1.3-2.5` wall hours and `10.4-20.0` allocated
GPU-hours. Do not claim scientific success until both final holdouts have been
paired by `smoke/compare_foundir_vlm_token_moe_matched_formal_v1.py`.

The `image_only` formal arm completed normally at 2026-10-25T16:53:00Z:
1,000/1,000 steps, `.643642` wall hours, `5.14914` allocated H100 GPU-hours,
finite training, and no OOM/NCCL/runtime failure. It failed the scientific
prerequisite that the ordinary MoE be usable. On the fixed 450-record holdout
it reached only `15.634318159133873` PSNR and `.34884399509264363` LPIPS,
versus the unprocessed input at `23.71294072670594/.3023991875624698`.
Routing collapsed almost completely onto expert 1: mean weights are about
`5.8873e-7/0.99999941`, with entropy `8.6261e-6`. Step 500 showed the same
failure and step 1,000 did not repair it. This is a valid terminal result for
the current image-only construction, not a compute/runtime failure.

Do not call this a working restoration MoE or use it as paper evidence. Do not
automatically launch `vlm_assisted`: an additive VLM score cannot by itself
repair a restoration backbone/expert bank whose image-only output is already
catastrophically below the input. The next design must begin from an actually
working restoration model/MoE and explicitly prevent router collapse; obtain
the user's decision before replacing the frozen matched protocol.

## MANDATORY: regular focused GitHub progress tracking

GitHub is the durable project history. Do not allow material work to remain
only in the dirty local workspace for long periods. The following is a hard
operating rule for all later agents, not an optional cleanup suggestion.

Create a focused commit and push it to `origin/main` after each coherent,
material milestone, including:

- a new or materially changed model, training script, evaluation script,
  configuration, launcher, or reproducibility contract;
- a code failure that required a substantive fix once the corrected code has
  passed the available static/CPU/GPU qualification;
- an advisor/user research-priority decision that changes the active plan;
- a completed qualification, pass-level validation, selected checkpoint, or
  terminal experiment result after its report/handoff has been updated;
- before leaving a long-running experiment unattended, provided its launch
  code/config and current handoff form a coherent reproducible snapshot.

For an active run, do not commit every log line or optimizer step. Commit the
launch-ready code before or immediately after launch, then commit again at
meaningful validated milestones and terminal completion. A running job is not
an excuse to leave its source/config untracked.

Every push must follow this safety procedure:

1. Run `git fetch origin` and inspect local/remote divergence.
2. Preserve the dirty workspace. Never use `git reset --hard`, `git clean`, or
   broad checkout commands to prepare a push.
3. Stage an explicit allowlist of relevant source/config/report files. Never
   use `git add -A`, `git add .`, or a directory-wide add in this repository.
4. Exclude checkpoints, model weights, raw/generated images, datasets, caches,
   telemetry, live logs, temporary files, and large generated manifests. Keep
   data identity through small metadata summaries and SHA-256 bindings.
5. Scan the staged files for credentials, access tokens, private keys, and
   unintended machine-specific secrets. Never commit the previously exposed
   GitHub token or any replacement credential.
6. Run syntax/JSON checks and `git diff --cached --check`; inspect the staged
   file list and size before committing.
7. If the main workspace is heavily dirty or an untracked file conflicts with
   an incoming remote path, create an isolated temporary Git worktree from
   `origin/main`, copy only the explicit allowlist, commit there, and push from
   there. Do not disturb the live training workspace.
8. Push the focused commit to `origin/main`, verify the remote commit hash, and
   report the commit hash and included scope to the user.

If authentication, networking, a remote conflict, or a required secret scan
blocks the push, report that immediately and preserve the ready commit. Do not
silently defer Git tracking across later milestones. Do not claim the work is
backed up until the remote hash has been verified.

## TERMINAL: residual-basis formal run complete — 2026-10-17

The user explicitly authorized addressing the oracle-proven lack of corrective
information in the V2 direct residual. The next experiment is not another
energy/gate run. Freeze the existing Noise proposal and train exactly one
restoration-native residual-basis model on the same 54,585-record training
split. It receives the degraded image, proposal, and their difference and
produces two independent directions: fidelity trained explicitly with MSE and
perception trained with stronger LPIPS while retaining MSE/local no-harm.

This is a causal prerequisite for later energy work. The fresh 25-step
eight-GPU engineering qualification completed successfully on `8 x H100
80GB` with status `0`, finite training, checkpoint and matched 256-record
development validation. It consumed about `.0217` wall-clock hours and `.173`
allocated GPU-hours. At step 25 the fidelity/perceptual heads improved PSNR by
about `+.0116/+.0114 dB` but worsened LPIPS by about `.00020`; this is only an
engineering check, not scientific evidence or a reason to alter the frozen
recipe.

The next action is a fresh formal run ID for the predeclared five-pass/8,530-
step schedule, validating after every effective pass. Never initialize it from
`qualification_v1`. Do not restart V2, train an energy/gate, use protected
data, or claim novelty success until the new residual basis itself
demonstrates adequate headroom on the fixed 256-record development protocol.
Formal runtime is conservatively estimated at `3.5–6.0` wall-clock hours on
one eight-GPU node (`28–48` allocated GPU-hours); update this estimate from
the measured pass-1 runtime.
Implementation contract:
`configs/foundir_proposal_residual_basis_noise_v1.json` and
`reports/foundir_proposal_residual_basis_noise_v1_readiness_2026-10-17.md`.

`formal1` is now live on `8 x H100 80GB` and passed its first effective pass
at step `1,706`. Do not restart, duplicate, retune, or interrupt it. The
fidelity head reached `43.16839203238487/.0400125195938017`, simultaneously
improving over the frozen proposal by `+.044231854379177094 dB` and
`-.010349548166232125` LPIPS. This already exceeds the old direct residual's
GT-only oracle in both metrics, so the restoration-native basis has resolved
the earlier lack-of-information bottleneck at least for LPIPS/headroom. It has
not passed the full `+.30 dB/-.005 LPIPS` gate because PSNR gain is still only
`.0442 dB`. Continue the frozen five-pass run and judge the next validations
without changing the recipe. Full pass-1 record:
`reports/foundir_proposal_residual_basis_noise_v1_pass1_2026-10-17.md`.

Pass 2 at step `3,412` improved further. Fidelity is now
`43.22963780909777/.03339703518213355`, or `+.10547763109207153 dB` and
`-.016965032577900274` LPIPS versus proposal. This lands inside the
predeclared `+.09–.12 dB` pass-2 hopeful trend range. It has not passed the
`+.30 dB` PSNR gate, but the evidence now supports completing the unchanged
five-pass schedule. Do not extrapolate linearly or retune from this interim
result. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass2_2026-10-17.md`.

Pass 3 at step `5,118` strengthened the result materially. Fidelity reached
`43.354813657701015/.030396450317311974`, equal to
`+.23065347969532013 dB` and `-.01996561744272185` LPIPS versus proposal. It
now needs only `.06934652030467986 dB` more PSNR to pass the joint target, and
the pass-2 to pass-3 PSNR increment was `+.1251758486032486 dB`. Continue the
unchanged run through passes 4 and 5; do not stop early or alter the schedule.
Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass3_2026-10-17.md`.

Pass 4 at step `6,824` did not cross the gate. Fidelity is
`43.365845650434494/.02899720575715037`, or `+.24168547242879868 dB` and
`-.021364862002883456` LPIPS. The PSNR gain increased only
`+.011031992733478546 dB` from pass 3 and remains `.05831452757120131 dB`
short, showing clear late-stage saturation after the 256-crop transition and
with the learning rate already near `6e-6`. Complete the frozen fifth pass,
but do not claim likely gate success or modify the live recipe. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass4_2026-10-17.md`.

`formal1` completed successfully at step `8,530` with launcher status `0`.
Final fidelity is `43.39715355634689/.028955043449059303`, equal to
`+.27299337834119797 dB` and `-.02140702431097452` LPIPS versus proposal. It
therefore misses the PSNR gate by only `.02700662165880202 dB` while passing
the LPIPS gate widely. The joint gate is terminally **not passed** for this
run. Do not resume or silently extend it. The next authorized reasoning step
is a no-training GT-only amplitude/local feasibility audit of the frozen
step-8,530 fidelity residual; no energy/gate or new training may start before
that evidence is reviewed. Terminal report:
`reports/foundir_proposal_residual_basis_noise_v1_terminal_2026-10-17.md`.

## TERMINAL: final fidelity-residual oracle passes local headroom — 2026-10-17

The user authorized the next feasibility audit. Run exactly one no-training,
GT-only diagnostic on the fixed 256-record source-disjoint development subset
using the hash-bound step-8,530 checkpoint. It evaluates global fidelity-
residual amplitudes from `0` to `2` in `.025` increments plus analytic local
8/16/32/64-patch amplitudes, and reports fixed-candidate and per-image Pareto
upper bounds. It must reproduce both proposal and alpha-1 fidelity metrics
within the frozen tolerance before writing a result. This is not deployable
evidence and may not access protected data. Code and CPU/unit qualification
are complete; the Codex container exposes no GPU, so the next action is the
single-node eight-GPU launcher:
`smoke/run_foundir_proposal_residual_basis_noise_v1_oracle_8gpu.sh`.
The audit completed successfully in `72` seconds (`.020` wall-clock hours,
`.160` allocated GPU-hours), with exact proposal and fidelity metric
reproduction and no protected-data access. No global amplitude crosses the
joint gate; alpha `1.000` remains the maximum-PSNR global setting. In contrast,
the GT-derived local 8x8 amplitude upper bound reaches
`43.65973751991987/.03243445468615391`, or `+.5355773419141769 dB` and
`-.01792761307387991` LPIPS, and all tested local patch families cross the
gate. Therefore the frozen residual direction contains ample local corrective
information; the remaining problem is learning a deployable spatial amplitude
map without GT. Do not report any patch-oracle metric as model performance.
Result SHA-256:
`ce6ab400c5003a630f3e05fed60a5e1fe9e6d6514af86cddc1fed6b9e65e2f66`.
Full report:
`reports/foundir_proposal_residual_basis_noise_v1_oracle_terminal_2026-10-17.md`.

## LIVE: deployable 8x8 spatial amplitude predictor ready — 2026-10-17

The user authorized the minimal causal follow-up to the local oracle. Freeze
both the proposal and step-8,530 residual basis and train only the `223,393`-
parameter spatial amplitude predictor. It receives observation, proposal,
fidelity and fidelity-minus-proposal, outputs one `[0,2]` amplitude per 8x8
patch, and starts at exact alpha `1` so step zero reproduces fidelity. Training
GT provides analytic patch-amplitude supervision; inference and validation
must never use GT to select amplitudes. The loss contains restored-image MSE,
residual-energy-weighted teacher regression, weak LPIPS and local no-harm.

JSON, Python, Bash, focused unit tests, strict frozen-checkpoint loading and
diff checks pass. The next action is a fresh 25-step engineering qualification
on one homogeneous 8xH100/H200 node. It must not seed a formal run. Do not
train energy, use protected data, or report the GT oracle as model quality.
Contract:
`configs/foundir_spatial_residual_amplitude_noise_v1.json` and
`reports/foundir_spatial_residual_amplitude_noise_v1_readiness_2026-10-17.md`.

The fresh `qualification_v1` run completed successfully on `8 x H100 80GB`
with status `0`, finite training, checkpoint/validation output and no protected
data access. At step 25, predicted amplitude is still near its exact-identity
initialization (`mean 1.0000218451`, standard deviation `.0004146491`), so
spatial output reproduces fidelity within `+.0000015423 dB` and
`-.0000000773` LPIPS. This is an engineering pass, not a scientific result.
The next action is a fresh-ID, from-zero five-pass/8,530-step formal run; never
seed it from qualification. Estimated H100 cost is `2.5–3.5` wall-clock hours
and `20–28` allocated GPU-hours, to be updated from pass-1 throughput.

`formal1` is live and completed pass 1 at step `1,706`. Spatial output is
`43.395900413393974/.029359601011492487`, which is worse than frozen fidelity
by `.00125314295291574 dB` PSNR and `.000404557562433184` LPIPS. The amplitude
map has moved materially (`mean 1.0358318537`, standard deviation `.1107148176`),
so this is an early generalization miss rather than a no-learning failure.
Continue the unchanged five-pass schedule; do not retune or stop from pass 1.
Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass1_2026-10-17.md`.

Pass 2 at step `3,412` is mixed but slightly more encouraging. Spatial output
is `43.3995122089982/.033109975053776`, which improves PSNR over frozen
fidelity by `+.00235865265130997 dB` but worsens LPIPS by
`.004154931604716697`. Relative to proposal it remains
`+.27535203099250793 dB/-.017252092706257827 LPIPS`, so LPIPS still passes the
joint target while PSNR is `.02464796900749206 dB` short. Continue unchanged;
do not claim oracle transfer or retune from this interim result. Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass2_2026-10-17.md`.

Pass 3 at step `5,118` remains a small positive PSNR result but is flattening.
Spatial is `43.400297187268734/.0317960577121994`, or
`+.0031436309218444425 dB` PSNR and `.0028410142631400966` worse LPIPS versus
frozen fidelity. Relative to proposal it is
`+.27613700926303864 dB/-.018566010047834425 LPIPS`, leaving
`.023862990736961353 dB` to the PSNR gate. Continue passes 4 and 5 unchanged;
current evidence does not support likely gate success or early retuning. Full
record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass3_2026-10-17.md`.

## STOP: V2 and direct-residual oracle are terminal — 2026-10-17

The user stopped `noise_energy_v2_formal1` externally after its last logged
step `6,655`; do not resume it. The last complete checkpoint with a matched
validation is step `5,118`. Its GT-only direct-residual feasibility diagnostic
finished on the same fixed 256-record source-disjoint development set. Even
the target-selecting upper bound failed the `+.30 dB/-.005 LPIPS` joint gate.
The maximum-PSNR oracle point was only `+.042452 dB/-.003946 LPIPS`; when the
LPIPS requirement was enforced, the best PSNR gain was only `+.037468 dB`.
Therefore the learned direct residual does not contain enough corrective
information, and a learned gate over these candidates cannot meet the target.
Do not resume V2, train a gate over this frozen residual, or claim the oracle
as deployable evidence. No V3 or fallback is authorized until the user makes
the next research decision. Full result:
`reports/foundir_proposal_energy_v2_direct_residual_oracle_terminal_2026-10-17.md`.

## STOP: max-complete PPT energy V2 only — 2026-10-15

This section supersedes the V1/fallback sequence below wherever they differ.
The proposal-relative energy V1 must finish its already-running frozen
five-pass endpoint and then stop; do not extend it to ten passes. Its observed
failure mode is higher PSNR with worse LPIPS, poor gradient-direction
alignment, and multi-step drift. Preserve V1 as causal evidence.

Before switching to the DiT-MoE fallback, implement and test exactly one
max-complete V2 of the advisor-approved PPT energy idea. V2 must contain all
and only these six missing parts:

1. LPIPS/perceptual-aware training in addition to pixel/edge fidelity;
2. actual frozen-model outputs for wrong-expert, under-restored,
   over-restored, and detail-damaged contrastive candidates;
3. directly unrolled and trained 1/2/3-step energy trajectories;
4. per-step no-harm constraints for both pixel fidelity and LPIPS, with the
   learned energy also allowed to select the unchanged proposal;
5. per-image VLM degradation severity from the existing frozen Qwen manifest,
   not a constant Noise one-hot vector;
6. spatially preserved/local energy supervision rather than reducing all
   quality evidence to one early global scalar.

Do not add another backbone, dataset, degradation task, specialist, routing
mechanism, MoE, compositional branch, or unrelated loss/idea to V2. Keep the
frozen Noise proposal, the 54,585-record source-disjoint training split, the
fixed 256-record development validation, 1/2/3-step reporting, protected-data
prohibitions, and the existing minimum `+.30 dB` and `-.005 LPIPS` gate.
Engineering qualification and the adequately scaled scientific run are
allowed only to validate this exact V2; they are not extra research branches.
Do not claim the PPT approach works until both metrics improve on the frozen
development protocol.

Current status: V1 completed `8,530` steps and failed the gate because its best
energy arm improved PSNR only `+.10285484790802033 dB` while worsening LPIPS
by `+.006770727247385367`. V1 is terminal and must not resume. The exact-six-
change V2 code passed its fresh 25-step qualification on `8 x H100 80GB` with
status `0`, finite saved weights, optimizer state for every trainable entry,
successful validation/checkpointing, and no protected-data access. The next
action is a new-run-ID, from-zero `8,530`-step formal V2 run. The 25-step
checkpoint must never seed the formal run.

## DEFERRED: advisor-approved two-stage novelty plan — 2026-10-15

This section supersedes the 2026-10-14 energy-TTS decision below wherever the
two differ. The phrase "PPT direction" means the restoration-energy method
already presented in the advisor slides; it does not mean making more slides.

The advisor's decision is:

1. **Try the PPT restoration-energy idea first.** Test whether a lightweight
   action-contrastive / proposal-relative energy can improve the already
   frozen FLUX.2/Restormer output. The energy is conditioned on the degraded
   input and explicit VLM degradation information. It should learn to
   distinguish a correct restoration from under-restoration, over-restoration,
   a wrong expert action, or newly damaged details, and then use that signal
   for a small number of candidate-image correction steps. This is currently a
   hypothesis, not an established working result.
2. **Use a staged but adequately scaled causal test.** Start from Noise on
   source-disjoint development data and compare against both the frozen
   proposal and a matched feed-forward refiner. Preserve the numerical gates
   and data restrictions in the 2026-10-14 section below. The first 500 steps
   are an engineering qualification only: they test correctness, memory,
   throughput, checkpoint/resume, and whether gradients and validation are
   finite. They are not enough to declare the research idea dead. After that
   qualification, train on the complete 54,585-record Noise training split.
   The first scientific decision point must include at least one complete
   effective pass over those records; if the validation trend is still
   improving, continue through a predeclared multi-pass schedule rather than
   stopping because the model is initially undertrained. Derive the exact
   optimizer-step schedule and GPU-hours from measured throughput before the
   formal launch. Do not claim success merely because learned energy decreases;
   PSNR/LPIPS and candidate ranking/refinement must improve.
3. **Only if the PPT energy idea does not work, switch to the advisor's DiT-MoE
   fallback.** The fallback is a FLUX/DiT restoration model whose internal
   experts are controlled using explicit VLM predictions of degradation type,
   severity, and composition. Do not begin this fallback in parallel with the
   first energy test unless the user explicitly changes the sequence.

Novelty boundaries for the fallback must be stated honestly:

- FoundIR-v2 already uses MoE, although its backbone is SDXL rather than DiT.
- MiM-DiT already uses MoE inside a diffusion transformer for All-in-One image
  restoration. Therefore **"MoE + DiT" is not a new contribution**.
- The remaining hypothesis is narrower: MiM-DiT directly looks at degraded
  image features to choose experts, while the proposed fallback first asks a
  VLM to state the degradation type, severity, and composition and then uses
  that explicit diagnosis to control DiT experts.
- "We added a VLM to the router" is not automatically enough novelty. It only
  becomes defensible if a matched comparison shows that explicit VLM guidance
  improves mixed/OOD degradation restoration or expert composition over an
  otherwise identical image-feature router.

Advisor-chat provenance, in plain language: the advisor said the PPT ideas are
reasonable but may not work and asked whether any already work. The advisor
then suggested MoE, initially noting FoundIR-v2 and its lack of this DiT
combination. After learning that MiM-DiT already covers the broad MoE+DiT idea
but does not explicitly output degradation type/severity through a VLM, the
advisor approved the sequence: **try the project's own PPT energy idea first;
if it fails, the VLM-guided DiT-MoE fallback is still acceptable because it is
not a complete overlap.** No method-success claim follows from that approval.

### Frozen scale and hardware plan for the first energy experiment

Only one homogeneous eight-GPU node may be used for this experiment. It may be
**8 x NVIDIA H200** or **8 x NVIDIA H100 80GB**, but never a mixed node and
never multiple nodes. Do not create or propose a cross-node launch. All
training, validation, checkpointing, and resume logic must work on either
supported single-node profile while preserving global batch 32.

- H200 profile: micro-batch 4/GPU, gradient accumulation 1.
- H100 profile: micro-batch 1/GPU, gradient accumulation 4.

These profiles have the same eight GPUs, global batch, optimizer-step schedule,
and training-record sampler. The smaller H100 micro-batch is only a memory
compatibility change; it does not authorize fewer data passes or different
method settings.

The Noise training split contains 54,585 paired records. Use global batch 32
unless the measured H200 memory/throughput qualification proves a different
batch is materially better. At global batch 32, one complete effective data
pass is `ceil(54,585 / 32) = 1,706` optimizer steps.

Use this frozen staged scale:

1. **Systems qualification:** 500 optimizer steps. This checks correctness,
   finite second-order gradients, memory, throughput, validation, checkpoint,
   and exact resume. It is not a scientific failure decision.
2. **First scientific checkpoint:** 1 complete effective pass, 1,706 total
   optimizer steps. Evaluate the frozen proposal, matched feed-forward refiner,
   and 1/2/3-step energy refinement on the fixed source-disjoint development
   validation set.
3. **Primary adequately scaled run:** 5 complete effective passes, 8,530 total
   optimizer steps. Save and validate at the end of every pass. This is the
   minimum scale for the main method judgment unless an earlier run has a
   numerical/code failure rather than merely weak metrics.
4. **Pre-authorized continuation:** extend the same run to at most 10 complete
   effective passes, 17,060 total optimizer steps, only when pass-4 to pass-5
   source-disjoint validation still shows a meaningful improving trend and no
   overfitting. Do not change the validation set, loss, candidate steps, or
   checkpoint rule based on protected reporting data.

Do not launch 100,000 steps by default: that would be approximately 58.6
passes over this split and is not currently justified. The 10-pass cap may be
changed only by a later explicit user decision backed by the pass-by-pass
development curve.

Exact wall-clock and allocated GPU-hours cannot be claimed before the H200
qualification measures seconds per optimizer step. After the 500-step run,
compute and report:

- `wall hours = remaining_steps * measured_seconds_per_step / 3600`;
- `allocated GPU-hours = wall hours * 8`.

Never transfer a timing estimate between H100 and H200, and never substitute
an old multi-node EBT benchmark. Measure seconds/step on the exact single-node
GPU family used for the run.

## Mandatory reporting style

All future user-facing reports must be easy to understand on the first read:

- Start with the practical answer or current outcome, then explain why.
- Use short, direct Mandarin sentences; retain English technical terms only
  where they prevent ambiguity.
- Translate each necessary term immediately into plain language. For example,
  say "模型直接看图选专家" before or alongside "latent-feature routing".
- Clearly separate **confirmed fact**, **our hypothesis**, and **next test**.
- For every comparison, name both sides, the dataset/protocol, and what each
  number means. Never use an unexplained phrase such as "coverage", "gate", or
  "better".
- Do not present an idea, advisor approval, lower energy, or a successful code
  run as evidence that image quality improved. State "not tested yet" when no
  metric result exists.
- Give time estimates in normal wall-clock hours and also report allocated
  GPU-hours for every proposed or completed GPU experiment.
- Prefer one concrete example over dense jargon. Add equations or internal
  implementation details only after the plain-language explanation.

## STOP: frozen FLUX/Restormer proposal + energy-TTS decision — 2026-10-14

This section supersedes the older Generalist-first and live-run sections below.
The from-scratch EBT Generalist is paused and must not be resumed, duplicated,
reconfigured, evaluated on the protected 660 endpoint, or used to select a
checkpoint. Its existing artifacts and failure logs remain historical evidence.

The only active restoration-model research objective is a lean test of
EBT-style test-time optimization applied to the already successful frozen
FLUX.2/Restormer routed system. FLUX.2/Restormer produces an immutable initial
proposal. A lightweight observation- and VLM-conditioned scalar energy
`E(y; x, c)` may then refine candidate `y` by a small number of gradients with
respect to the candidate image. Do not retrain or alter the frozen proposal
models during the first causal test.

Work source-disjoint development data only. Never use the official, sealed,
parent-test, internal-test, or frozen 660 reporting pixels for training,
hyperparameter selection, checkpoint selection, early stopping, or deciding
the number of test-time steps.

Proceed in this order:

1. Start with the Noise singleton, because the frozen routed result
   `40.133486/.096746` is behind Full FoundIR `42.655305/.067910` in both PSNR
   and LPIPS.
2. Compare the frozen proposal against a matched-parameter feed-forward bounded
   refiner and fixed 1/2/3-step candidate-gradient refinement. Require extra
   steps to improve restoration rather than merely lower an uncalibrated
   energy.
3. A minimum technical pass is at least `+.30 dB` and `-.005` LPIPS relative
   to the frozen proposal on protected source-disjoint development validation.
   Paper-worthy evidence is at least `+.50--.75 dB` and `-.010` LPIPS, a useful
   compute-quality curve, and superiority to the matched feed-forward refiner.
4. Only after the Noise gate passes may the same frozen mechanism be tested on
   the remaining singleton categories. A convincing singleton aggregate target
   is at least `29.2 dB` with LPIPS no worse than `.135`, with gains across
   multiple categories rather than one category dominating the mean.
5. Only after multi-category singleton evidence may the unchanged mechanism be
   evaluated on Double compositions. The frozen Double baseline is
   `24.321537/.192313`; an indicative paper-strength target is at least
   `24.8/.180`, subject to per-category no-harm evidence.
6. Do not reopen Triple, specialists, compositional EBT, or another from-scratch
   Generalist until this lean mechanism succeeds or fails under its frozen
   causal comparison.

The working novelty hypothesis is not simply "EBM", "more inference steps",
or "VLM + EBT". It is candidate-dependent, differentiable test-time refinement
of frozen heterogeneous AiOIR proposals under an observation- and structured
VLM-conditioned scalar energy, with the same energy used for iterative
restoration and stopping/verification. Treat "first" as unverified until the
targeted prior-art audit is complete. Current audit and exact claim boundary:
`reports/foundir_flux_restormer_energy_tts_novelty_audit_2026-10-14.md`.

The first GPU qualification, if the implementation and protocol pass CPU
review, is capped at 500 optimizer steps on one 8-GPU node. Its duration is
measured rather than constrained by the obsolete `.5` wall-hour estimate,
because second-order candidate gradients may take longer on H100 than H200.
**This is a systems check, not a method-failure gate.** After it passes, use
its measured throughput to freeze
an adequately scaled schedule over all 54,585 Noise training records, with at
least one complete effective data pass and further predeclared passes whenever
the source-disjoint validation trend is still improving. Report both
wall-clock hours and allocated GPU-hours. Stop only for a code/numerical
failure, a clearly saturated or worsening development curve after adequate
exposure, or completion of the frozen scale schedule—not merely because a
small run misses the final metric target.

## DEFERRED: Generalist-first research decision — 2026-10-11

This was the active objective on 2026-10-11 but is superseded by the
2026-10-14 decision above. Preserve it as historical protocol context only.

Do not design, implement, train, tune, benchmark, or launch any low-light,
weather, trajectory-energy, residual-energy, or other specialist. Preserve the
specialist prior-art/design notes as deferred research only. Do not use a
specialist result to claim that the Generalist gate passed.

The decisive comparison is Generalist versus Generalist on the existing 660
full-resolution whole-image composition set, using identical records and the
frozen metric protocol (`full_rgb_uint8_image` PSNR and
`area_weighted_nonoverlap_512_rgb_uint8` LPIPS). Neither side may run a
specialist. The released FoundIR Generalist reference is hash-bound in
`checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json`.

For an unambiguous pass, the EBT Generalist must beat the released FoundIR
Generalist in both PSNR (higher) and LPIPS (lower) for every composition tier,
not merely in a weighted aggregate:

| Tier | Released FoundIR Generalist PSNR | Released FoundIR Generalist LPIPS |
|---|---:|---:|
| Singleton, 300 images | 27.2323939564508 | 0.1744119563086327 |
| Double, 300 images | 22.245453630403492 | 0.2066620594635684 |
| Triple, 60 images | 24.9455711880497 | 0.03775338436922618 |
| Overall, 660 images (reporting only; not a substitute for tier gates) | 24.757709920211013 | 0.17664758756638468 |

## HISTORICAL FAILED RUN: real-data Generalist and pass-1 diagnostic — 2026-10-12

`real_generalist_4node_mixed_hopper_v1` ran on 8 x H200 plus 24 x H100. It used
global batch 64, 10,233 steps/pass, three planned passes, EMA `.9999`, and the
frozen 654,881-pair real training split. It failed during pass 2 at pass step
1,300 / global step 11,533 after a 600-second NCCL ALLREDUCE watchdog timeout.
Only pass 1 is complete. Do not resume or evaluate this run while the
2026-10-14 decision is active. Its 16,461-record 256x256-crop validation is
diagnostic only and is not comparable to the decisive 660-image protocol.

Pass-1 raw tier metrics were Singleton `28.829671/.172490`, Double
`23.916114/.242836`, and Triple `33.070977/.025842`. The apparent Double/Triple
inversion is not contradictory:

- Triple contains only `blur+low_light+noise` (579 validation records), whereas
  Double averages five heterogeneous categories.
- Double raw per-category LPIPS is `blur+noise .017810`,
  `haze+low_light .196602`, `low_light+noise .242655`, `haze+rain .277549`,
  and `blur+low_light .312665`. The last three drive the Double weakness.
- The sampler assigns one third of presentations to each composition tier. The
  sole Triple category therefore receives approximately five times the
  category-level exposure of each of the five Double categories.
- LPIPS has weight `.05` and is included only every four optimizer steps, so
  PSNR can improve before hard Double perceptual artifacts are resolved.

The former instruction to complete all three passes and run a matched 660
evaluation is deferred. Do not access the 660 endpoint for this failed run
while the 2026-10-14 decision is active.

## Historical FLUX/Restormer gap record for the monolithic/generalist route

The old routed FLUX.2/Restormer system's Triple gap to FoundIR was not a simple
parameter-capacity problem. Preserve these established causes. They motivate
the bounded energy-refinement test above but do not authorize specialist or
Triple work:

1. Most FLUX work used synthetic/task-specific data derived from roughly 1,600
   clean sources, versus FoundIR metadata covering 1,010,064 pairs and the
   frozen 654,881-pair real training split now used by EBT.
2. FLUX experts received hundreds to roughly 10,000 task-specific steps,
   whereas the released FoundIR Generalist is at step 1,300,000. Raw steps are
   not batch-normalized, but the exposure gap is material.
3. FLUX/Restormer was a heterogeneous routed expert system, not one jointly
   trained Generalist over all singleton/double/triple categories.
4. Sequential experts have measured inter-expert domain shift: on 53 Triple
   records, Noise was nearly inert (`+.069859 dB`), Low-light overshot GT luma
   by `+29.503`, and Blur worsened 49/53 records and lost `.272308 dB`.
5. Synthetic operator order/strength did not fully match FoundIR, including the
   documented Noise/JPEG visibility mismatch.
6. Short FLUX/Restormer selections lacked a mature EMA selection path; current
   EBT uses EMA `.9999` and validates raw and EMA after every pass.
7. FLUX is a large pretrained generative DiT, while FoundIR's approximately
   36.3M-parameter UNetRes is restoration-native. More FLUX parameters did not
   substitute for paired restoration alignment.
8. Full FoundIR also has Low-light and Weather specialist refinement. That is a
   separate full-system advantage and must not be mixed into the frozen
   Generalist-versus-Generalist gate. Specialist work remains paused.

After the active real-data run completes, the remaining path is:

1. Audit all pass/checkpoint/validation artifacts and hashes.
2. Select and freeze raw or EMA weights without protected-test access.
3. Run one matched 660 whole-image Generalist-versus-Generalist evaluation.
4. Only after all tier gates pass may specialist research be reopened.

Report wall-clock hours and allocated GPU-hours for every proposed or completed
GPU experiment.

## DEFERRED: potential SOTA follow-up after the Generalist gate — 2026-10-12

This records a candidate paper-evaluation program. It does not change the
active Generalist-first decision and does not authorize another training or
evaluation while `real_generalist_4node_mixed_hopper_v1` is running. First
finish all three frozen passes, select raw versus EMA using only the frozen
16,461-record validation evidence, and run exactly one matched 660-image gate.
Do not tune from that result. The project is not yet demonstrably SOTA:
pass-1 crop validation is encouraging, but Double LPIPS remains worse than the
released FoundIR Generalist and is not directly comparable to whole-image
metrics.

If and only if EBT beats the released FoundIR Generalist in both PSNR and LPIPS
for Singleton, Double, and Triple on the frozen 660 protocol, the lean
potential-SOTA program is:

1. Train/evaluate one matched CDD-11 model and report all eleven categories
   plus Singleton/Double/Triple aggregates with PSNR/SSIM. Main composite
   targets are AllRestorer `33.72/.9436`, DAR-Net `29.73/.888`, ClearAIR
   `29.34/.886`, MoCE-IR-S `29.05/.881`, and OneRestore `28.47/.878`.
   AllRestorer lacks a verified reproducible public implementation, but its
   published number cannot be omitted from an unrestricted SOTA claim.
2. Train/evaluate one matched standard 5D model on SOTS, Rain100L, BSD68 sigma
   25, GoPro, and LOL. Important average PSNR/SSIM targets include D3Net
   `32.36/.901`, DAR-Net `31.02/.921`, DPC-Net `31.00/.923`, DFPIR
   `30.64/.913`, MoCE-IR `30.58/.919`, and CyclicPrompt `30.52/.935`.
   Disclose metric/protocol tradeoffs and never compare unmatched numbers.
3. Run two matched retraining ablations: no VLM conditioning and no learned
   energy/direct restoration. Use the frozen full checkpoint for evaluation-
   only adaptive-TTS fixed-step sweeps and self-verification on/off tests.
4. Report parameters, FLOPs, peak memory, latency, and actual adaptive descent
   steps. Cite and discuss FoundIR-v2, VLU-Net, DFPIR, InstructIR, Perceive-IR,
   ClearAIR, and EBT even where benchmarks are not directly matched.

The minimum defensible follow-up has eight blocks: frozen 660 evaluation;
matched CDD-11 train/eval; matched 5D train/eval; two retraining ablations;
adaptive-TTS evaluation; self-verification evaluation; and efficiency
evaluation. This means four new training runs and roughly six to eight
evaluation jobs after the current gate. A broad AiOIR claim additionally needs
separately trained 3D, real-world, and robustness/OOD evidence, raising the
program to roughly 12--15 blocks and six to eight training runs.

The current 30,699-step 32-GPU schedule measures about 8.97 wall hours/pass,
or about 27 wall hours and 864 allocated heterogeneous GPU-hours for three
passes. Four mechanical repeats would be an unapproved conservative ceiling of
108 serial wall hours and 3,456 GPU-hours. Before any CDD-11, 5D, or ablation
launch, run a protocol-specific 500-step throughput qualification (target at
most 0.5 wall hour and 4 allocated GPU-hours on eight GPUs), then size training
by matched epochs and unique-sample exposure. Narrow the intended claim to
composite/mixed Generalist SOTA or best matched average; winning the
project-specific 660 comparison alone is not universal AiOIR SOTA. Specialist
work remains paused.
