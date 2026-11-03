# Rain从step200真续训到最多2000

## 决策与不变项

用户在已完成原生50步200步试验后授权继续训练，并把总上限提高到2000。
这不是再做资格实验，不是从旧native step2000权重fresh Adam重启，也不是恢复endpoint转flow。
父运行native50_stage1_4x2_v1的开发结果31.0840997/.08360457，工程完成但未达34.2/.08。
从它的step200恢复完整状态，新增最多1800步，固定400/700/1000/1500/2000评价。
没有变更数据、损失、学习率、恢复结构或推理配方；这轮只检验更多训练是否继续改善。

训练池38334记录/5780来源，均匀来源再选记录；512随机裁图/D4，小图reflect padding。
global32=8rank×micro1×累积4，lr1e-5，AdamW weight_decay0，clip1；所有五项loss系数不变。
FLUX base/VAE冻结，LoRA和Adam FP32、模型BF16；GT主监督、一步Rain FLUX弱RGB示范。
没有CNN/Restormer恢复器，没有新energy/router/MoE训练，不改另外四个原专家或另一代理energy。
32训练诊断和100开发图固定，评价仍50步/CFG1/ref10/alpha1/单uint8，逐图seed与父轮一致。
开发GT不用于训练，官方/封存像素仍禁止。开发均值与训练诊断分开，不包装为新盲测。

## 真续训的证据要求

父根checkpoints/foundir_rain_native50_warmstart_v1/native50_stage1_4x2_v1。
父receipt SHA ac284c0ed2ffe1d7291c7c4462f31632187036f1885d29d221f41c9333be61c2。
父step200 receipt SHA 23e62b72b004e8b76714f2114b565d8c10d641c4e349ee307f6b3b53c90fcfcd。
父step200 tensor state SHA 8fdd6a0d5c6682ab616ba65580477329f3155aecab9da94ec6f1f943720cedcb。
新目录checkpoints/foundir_rain_native50_continue_v1/continue2000_4x2_v1。
第一次载入旧LoRA/Adam/200条历史/8rank RNG/next_global_slot6400，再以新receipt存200号状态。
不是把旧receipt改哈希：旧文件只读，新目录明确绑定父身份；模型和优化器数值不变。
下一步是201、全局样本位置6400..6431，noise/t也按原绝对位置规则生成。
新计划前6400行必须与父轮完全一致，固定132评价行也完全一致；不会只重复旧6400曝光。

每完整步均原子保存，最多200..2000共1801份新快照。旧201份留在父目录。
若中断，同run_id和fresh共享launch_id继续，读取最高完整检查点；损坏不偷偷回退。
换world/runtime或修改绑定源码不支持严格恢复；不声称跨环境CUDA逐比特等价。
在评价中断时先补完该里程碑，不先生成下一段缓存；最终导出中断可从step2000补齐。

## 缓存与质量审查

旧132评价缓存继续引用原producer。新的57600训练曝光包含新记录/裁图，需要新缓存，
不是上一轮6532份可以全部复用。分段200→400→700→1000→1500→2000按需生成；
每个rank只准备自己的分片，逐张保存、核验后可跳过；原一步教师释放后才加载学生。
载入学生时恢复保存的RNG，所以准备缓存或重新加载模型不应改变下一步训练随机流。
累计曝光64000，不等于按记录遍历了1.67轮，也不保证2000步补齐约3.12dB差距。

每个检查点报告同一100开发图的PSNR、LPIPS及相对上个检查点变化。
若连续两次PSNR不升且LPIPS不降，进入paused_for_quality_review并退出，等待人工方向；
一次波动不停止。任一指标改善即清零该连续计数；不靠调阈值追求好看结果。
这条暂停规则只是避免盲目追加，不是质量安全保证。出现NaN/Inf或运行异常保留状态并报错。
同checkpoint必须PSNR>=34.2且LPIPS<=.08，不拼接不同步最佳值；到2000不自动继续。

## 资源与时长

4节点×2H200=8卡，每卡>=139GiB；继承已运行节点/runtime，H10080GB未资格。
新缓存与逐步快照估计1.3–1.6TiB；首次预检至少1800GiB空闲，不能为腾空间删除旧产物。
恢复入口按已有提交数量扣除估算已写部分，保留余量；实际共享磁盘争用仍可能影响空间和速度。
全程粗估3–6小时，即24–48使用GPU-hours；硬cap每节点8小时、合计64GPU-hours加清理。
第一关400步暂估30–70分钟，未经本轮实测；根据缓存和训练速度更新，不按旧24分钟保证。
历史累计11.972222222GPU-hours另计，不因新ID清零。
退出后默认轻量核对，不再占用额外时间运行全量CPU扫描；不能把轻量通过说成全量核验。

## 目的地代理操作

安全同步origin/main，保留所有脏文件和运行资产。唯一执行根仍是：
/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。

先执行CPU预检（只核对所需父step200状态、计划和资产，不启动GPU，不扫描所有历史tensor）：

```bash
.conda/gpu-train-stable/bin/python smoke/train_foundir_rain_native50_continue_v1.py preflight --require-parent --report reports/foundir_rain_native50_continue_destination_preflight_v1.json
```

通过后给用户四节点命令，入口smoke/start_foundir_rain_native50_continue_v1_tmux.sh。
参数为node_rank 0/1/2/3、当前node0地址、continue2000_4x2_v1、launch_continue2000_v1。
四节点用同run/launch，端口29690。不要沿用历史IP，也不要重启旧200步或25步资格。
工程中断恢复仍用同run_id但fresh共享launch_id；质量暂停不能用重启绕过。
确认各rank的RESUMED日志含step=200 next_global_slot=6400；第一条
RAIN_NATIVE50_CONTINUE_TRAIN日志的JSON字段step应为201。
在400/700/1000/1500/2000回传短报告，终态无论完成、暂停、异常或超时都推回GitHub。

入口结束时读取节点状态、最新完整状态、指标摘要、最终导出等关键文件，输出light_result.json。
RAIN_NATIVE50_CONTINUE_LIGHT_CHECK_PASS与四节点status0只表示有界任务正常结束及轻量核对通过；
据status/last_step区分完成2000和质量暂停。full_tensor_audit_performed=false，LPIPS未独立重算。
不另跑CPU全量核验，不上传数据/权重/tensor/实时日志，不自动训练新任务。

## 本地主代理验收范围

代码、CPU回归与两进程状态迁移检查结果见同名JSON。源资产只读预检不等于目的地检查。
11项CPU回归通过；真实两进程Gloo小模型从旧step200迁移后，step201及重启后step202
均与不中断路径逐元素相同。Python AST和shell语法检查通过；这不是八卡FLUX CUDA验收。
完整采样计划64000次曝光实际覆盖23494条记录、全部5780个训练来源；不等于遍历全部记录。
主代理无法访问目的地step200状态文件或GPU节点，因此目的地--require-parent仍需执行。
没有本轮GPU启动或新质量指标，CPU检查不替代CUDA/NCCL实测。
