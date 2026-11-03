# Rain：同配方从step600真正续训到800

## 当前授权

用户已授权新增200步；本节取代上一轮“完成后没有新任务”的状态，不恢复其它路线。
父任务 `step400_onpolicy200_4x2_v1` 已在4节点×2H100正常完成到全局600，
开发100图为33.99953558/.06457076；相对step400提高.87043054dB、LPIPS降低14.32%。
距34.2仍差.20046442dB，LPIPS已达标；14张PSNR退步、10张LPIPS退步，78张两项不差。
父轮耗时1小时12分32秒、9.64777778使用GPU-hours。这笔历史成本保留。

这些结果支持先继续优化当前模型，不保证再200步能补齐缺口。训练探针后100步增长
已经放慢，开发缺少中途趋势，所以本轮固定700/800两个检查点再判断，不自动堆几千步。

## 配方保持，真正恢复状态

- 父目录是 `checkpoints/foundir_rain_onpolicy_midpoint_v1/step400_onpolicy200_4x2_v1`。
  **不是旧native50任务的step600，更不是它的step2000。**父step600完整状态SHA：
  `f062a04bc0cbf5d1b1541ec17721c34f8138aff1cb28b7fff93e77bfa4b58b54`。
  恢复LoRA、Adam全部状态、8rank RNG、600条历史和next_global_slot=19200。
  不fresh optimizer、不从最终LoRA文件代替完整恢复、不重复前200步。
- 新独立目录保存初始600和每步601..800，共201份完整状态；旧目录/绑定源码不修改。
  用户说的“再100/200步”对应全局700/800，累计本路线on-policy更新300/400步。
- 完全复用已跑通的 `rain_onpolicy_midpoint` 核心：当前FLUX每次重新生成原50步
  时间表前25次，截断前缀导数，第26次预测 `x_t-t*v` 后用原生VAE解码反传。
  仍不是全26次端到端优化，也不是继续跑满50步的输出；不宣称已完成MoE兼容。
- GT RGB MSE×10、GT LPIPS×.05、lr1e-5、global32、micro1/每rank累积4次、
  BF16 FLUX/FP32 LoRA与Adam、CFG1算式、reference time10、512、alpha1、单uint8不变。
  只训练FLUX LoRA；无CNN/Restormer恢复器、无新teacher推理、无其它新增机制。
- 继续原38334记录/5780来源训练池source-uniform计划，槽位19200..25599，
  新6400次呈现；LQ/GT缓存已由旧native任务生成，读取核验后复用，不重生成。
  缓存里旧teacher/latent不进入当前目标。生成前缀只收到LQ/prompt/seed，不收到GT。
- 不读官方、封存或最终测试；不换损失、学习率、采样位置、读出位置或阈值。

## 两次评价与停止边界

700、800各评价同32张固定训练探针（原槽位12800..12831）和原100开发图
（64032..64131），严格分开平均。训练探针曾参加上一轮更新，不是泛化数据；
开发集也已使用，不是新盲测。新训练槽位检查在缓存读取之前拒绝这些评价索引。

在700检查时重算一次同环境冻结step600的132张对照，以便H100/H200或节点迁移时
有真正匹配的起点；800复用已保存对照，不重新生成完整一遍。
用同一个FLUX实例暂时载入step600 LoRA，不修改Adam；无论评价成功或异常都恢复
当前训练权重。评价保存/恢复RNG，不改变随后训练的数据或随机进度。
保留历史step600均值，同时报告新环境对照与历史差异，不以换弱分母美化增益。

每个检查点独立报告两项指标、相对匹配step600增量、逐图退步。
必须**同一个检查点PSNR≥34.2且LPIPS≤.08**才过，不能拼接700的PSNR与800的LPIPS。
结果的 `development_gate_pass` 指最终800；`qualifying_checkpoint_steps` 单列700/800
各自达标者。最终导出固定是800，不把它冒充指标最好的检查点；700完整状态也保留。

训练最多到800，之后人工重新决策。700结果仅用于观察，不自动改配方/追加预算；
严重退步需回报主代理/用户决定，不在后台静默“修复”。非有限值、运行或保存错误
立即失败退出并保留最后完整步，不自动回滚到较好指标或重启。
不自动MoE/energy、其它专家、长训或部署。

## 资源、保存与完成检查

仍4节点×每节点2张H100/H200，总8卡；单节点两卡同型号、每卡≥75GiB。
各节点软件/驱动release/指标版本一致，运行内不静默换环境。跨节点混型号是代码支持，
不是这轮已实测；旧训练核心已在8H100跑通，峰值allocated15.13GiB/卡。
新的续训入口和真实GPU中断恢复仍待本轮验证，CPU精确恢复不能替代CUDA保证。
Hopper激活CPU转存、原生activation checkpoint、VAE query256分块与工作区策略不变。

按上一轮每步19.182秒＋保存1.016秒估计：全程75–90普通分钟/10–12使用GPU-hours，
首700检查约35–45分钟。是基于8H100的估计，不保证H200或共享盘负载下一定相同。
每节点cap3小时，合计24GPU-hours加清理，不自动扩预算；每次恢复耗时也计入累计。
首次要求100GiB空闲，预计新完整快照约73GiB并留输出余量，不删除历史资产。

每完整步原子保存模型/Adam/RNG/采样位置及哈希；每张probe独立保存。
相同run_id＋新的共同launch_id恢复；若700评价中断，先补完评价再进入701。
不支持换world_size的严格恢复；不换ID掩盖丢失进度，不覆盖旧checkpoint或修改绑定代码。
终态只核对退出、里程碑、关键step800完整状态与最终导出；不全扫201份tensor、不重算LPIPS。

## 目的地执行

主代理在代码工作树实现/测试/推GitHub，无目的地GPU挂载或远程进程控制；不能声称已启动。
目的地安全同步最新main，保留无关修改和全部模型/数据/缓存。
仅在 `/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang` 执行。

```bash
.conda/gpu-train-stable/bin/python smoke/train_foundir_rain_onpolicy_continue_v1.py preflight
```

预检只读元数据，检查父终态/完整状态receipt、缓存receipt和磁盘；大父状态在实际恢复时验SHA。
不重跑旧工程资格，不重新生成teacher/cache；缺资产时报告，不猜路径或修改绑定receipt。
目的地确认当前node0 IP、四节点都空闲后，给用户四条启动命令（rank分别0/1/2/3）：

```text
bash smoke/start_foundir_rain_onpolicy_continue_v1_tmux.sh <node_rank> <当前node0地址> continue600_to800_4x2_v1 launch_continue600_to800_v1
```

端口29698；四节点相同run/launch，恢复用fresh共同launch。tmux创建不等于训练正常。
保留原节点共享锁和GPU空闲检查，不杀其它任务。
输出 `checkpoints/foundir_rain_onpolicy_continue_v1/continue600_to800_4x2_v1/`；
日志 `logs/foundir_rain_onpolicy_continue_v1_continue600_to800_4x2_v1_node<N>.log`。
完成须 `RAIN_ONPOLICY_CONTINUE_RESULT_VERIFIED` 和四节点 `RAIN_ONPOLICY_CONTINUE_NODE_END status=0`。
`result.json`与verified标记引用一致，避免旧收尾文件名错误。

请在首次确认601..605、700检查和800终态及时推小报告：最后完整全局步、新增步、
指标、退步计数、步/保存耗时、显存、每节点状态和实际GPU-hours。不要又等结束才说
已经检查过首5步；如延迟回传，明确是依据保存记录补发，不伪造当时发布记录。
不上传权重、tensor、图片、凭据或实时日志。分析与代码修改仍由主代理负责。

## 本地验证

8项CPU回归全部通过：配方/预算冻结、下一采样位置及完整评价分片、训练像素边界、
真实600次小模型Adam更新后的600→601精确恢复、拒绝fresh/超预算状态、临时基线
权重在正常和异常路径下均恢复且Adam不变、生成/loss/反传代码段与父任务一致、
父报告身份和启动/结果契约。另通过两进程Gloo真实梯度累积及Adam/RNG恢复测试，
恢复后下一步参数和Adam状态逐元素一致；DDP边界用一次初始化后设置计数600，
不冒称完成600步FLUX多节点复现。Python AST、bash语法和focused密钥扫描通过。
本地主代理没有目的地大tensor或GPU挂载，因此目的地资产预检/真实续训尚未执行。
