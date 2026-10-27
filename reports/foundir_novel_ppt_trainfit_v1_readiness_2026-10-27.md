# PPT能量：固定32个训练裁块的300步学习能力检查

## 决定与边界

用户在审阅25步资格几乎没有画质变化、完整8530步约需52小时的事实后，
明确授权本次检查。不是重新做一个小规模正式效果实验，也不是自动续训资格checkpoint。
本次从零初始化energy，固定32个训练来源的32个256裁块，连续训练300步，
检查同一批训练样本能否被有效拟合。诊断权重永不初始化正式训练。
不读取开发、official、sealed、internal、parent-test图片；
继承来源隔离与clean hash排重检查时只读held-out split元信息。

如果训练样本也学不动，应进一步查优化、目标或容量，不能把不动简单归因于数据少。
如果训练样本学得动，仍不能声称泛化或论文创新成立；下一阶段的完整数据训练
需要另行决定。本脚本没有自动启动1706步或8530步的路径。

## 保持什么不变

复用已通过GPU资格的NovelPPTEnergy和原objective、冻结proposal/basis/外部模型、
634032参数规模、完整三步梯度轨迹、每图VLM等级、所有损失权重、
AdamW、gradient clipping、100步warmup与8530步余弦学习率跨度。
不因总步数变成300而把余弦周期压缩到300。
global batch仍为32：8卡、每卡micro1、累积4；单节点同型号H100/H200。
使用新ID与独立config/trainer/launcher，不修改原资格或MoE诊断的绑定文件。

刻意改变的数据条件是：每步恰好看同一32个裁块各一次，共300次曝光/裁块。
来源与记录通过固定seed、哈希排序选择，不依据恢复成绩筛图；
每个记录的crop和D4变换seed固定，不随step变化。
每rank持有固定4个裁块，每步只改变本地累积次序。
这不是用32个样本估算全数据泛化表现，而是有意做训练拟合诊断。

固定的冻结模型输出只生成一次并缓存于GPU，包括proposal、Fidelity、perceptual、
三个外部恢复结果及重复恢复结果；不缓存可训练energy或GT派生的选择答案。
冻结模型保持eval/no-grad，使用原BF16冻结推理与FP32结果加法约定。
保存每个裁块、条件及缓存输出的hash，检查同一记录/同一裁块的可重复性。
缓存避免每一步重复计算冻结模型，不改变训练候选、损失或更新规则。

## 观察什么

第0、100、200、300步，对完全相同的32个训练裁块评估并保存逐图结果：

1. proposal、Fidelity、energy一步/两步/三步、自主选择的MSE、PSNR、LPIPS。
2. 相对Fidelity的MSE下降比例、PSNR变化、LPIPS变化。
3. 修正梯度与真实残差的方向一致性、平均修改幅度。
4. 固定相近候选的能量排序与真实质量排序是否一致。

相近候选为Fidelity、perceptual、半幅/1.25倍Fidelity修正、
75/25和50/50的Fidelity/perceptual混合。
前四种已在原损失中；两种混合只用于本次诊断读数，不增添训练目标。
质量按原归一化MSE+2*LPIPS计算。报告全部非平局候选对，
并单列质量差不超过Fidelity cost的10%的“近质量”候选对。
这个10%仅定义诊断分组，不用于训练、推理选择或阈值调参。
GT质量差<=1e-6不计入排序统计；预测能量平局给0.5分；
若近质量对数为0，准确率为null，不伪造50%或100%。
必须同时报告参与对数、step0基线和后续变化，不能仅看排序loss下降。

推理式选择仍只看同一energy，在Fidelity与1/2/3步之间选，不读GT。
训练拟合评估当然使用训练GT测误差/排序；不得把这称为无监督评价。
能量下降不等于画质提升；32图都选最后一步也不构成自验证成功。
不新增随意的“通过”数值阈值；保存分项事实，由效果幅度、排序支持数及
方向变化共同判断是否值得全数据试验，不能自动发起下一轮。

## 成本与运行

原25步实测21.83572708698921秒/step，300步不含额外检查约1.82小时、
14.56 allocated H100 GPU-hours。当前规划含启动/缓存/四次检查为
2.0–2.5普通小时、16–20 GPU-hours；缓存可能加快，须用新运行实测更新。
首个训练后检查为step100，按旧速度单训练约36.4分钟，另加启动/检查。
没有承诺指标提升或给出未经验证的加速倍数。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_novel_ppt_trainfit_v1_tmux.sh trainfit_v1
```

日志：logs/foundir_dual_retry_v1_ppt_trainfit_v1.log。
结果：checkpoints/foundir_dual_retry_v1/ppt/trainfit_v1。
启动器复用节点锁、八卡空闲检查与180秒有界通信检查。
正常标记：PPT_TRAINFIT_CACHE_READY、PPT_TRAINFIT_TRAIN、
PPT_TRAINFIT_EVALUATION、PPT_TRAINFIT_COMPLETE及DUAL_RETRY_END status=0。
创建tmux不代表训练已成功启动。Codex无GPU执行通道，本轮未代为启动。

## 验证

5项CPU回归测试全部通过：固定来源/裁块选择不受元信息行顺序影响、
排序平局及近质量对统计、保留原学习率跨度、缓存候选重复完整三步反传
且缓存不被修改、训练诊断不得冒充正式/测试结果。
Python语法、JSON和Bash语法检查通过。
真实预检查通过：32条/32来源，原成功资格receipt及checkpoint、
原配置、冻结proposal/basis与来源绑定一致，images_opened=0。
本次没有重新访问图片或运行GPU。新脚本尚未经过实际300步GPU执行；
复用了已通过25步GPU资格的模型/损失，不能把这混称为新任务已经完成。
源代码只新增，不覆盖原已完成或可能运行的绑定文件。

配置SHA-256：dd3a7748f6c821bc870e685c1cd8613ffc113877b64f6a8773f5279a29326cb5。
固定32记录/来源/crop seed及像素文件hash身份摘要：
c7ec30985a5238788c1752986b8c1618501bfd79dfbedb5de39dab9131aee6cf。
trainer SHA-256：293e9f05dc638d34a6e1e093caf286d575f7e400d95115320b15de44b0cb7469。
输出另记录裁块实际tensor hash，不能把元信息身份检查说成图片已经解码检查。
