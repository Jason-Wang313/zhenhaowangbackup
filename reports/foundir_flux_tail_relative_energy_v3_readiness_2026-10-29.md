# FLUX-tail relative energy V3：第一阶段实施与验收准备

## 当前状态

用户已授权实施修复。代码、CPU测试、来源/缓存预检已完成；没有GPU可用于本会话，
尚未声称CUDA资格、速度或新指标通过。旧V2全部源码、配置、权重和结果保持不动。

本轮先处理“训练侧是否能学会一次有效修正”，不直接做正式论文训练。
没有增加Restormer、proposal/refiner网络或恢复MoE；FLUX和原能量网络架构保持不变。

## 第一性原理证据与改动

旧V2的25步模型在8张检查图上连续压暗图片。对已保存输出的只读分析显示，
最终修改的77%–98%平方幅度来自全局RGB偏移，各通道均值降低约18–33/255。
这些统计是诊断，不作为新训练输入；这8张图不回灌。

当前只读原24训练来源，保持512、原seed、Noise step400、50步和双uint8部署口径。
最终目标仍为energy相对原FLUX +.30dB、LPIPS下降9.928107050377874%。

具体改变：

1. 保留旧GT轨迹/随机候选，增加energy自行产生的4步压力轨迹。
   初始生成器是冻结旧V2 step25，后续生成器是当前训练checkpoint。
   GT不进入生成方向或接受判定，只在生成后标注训练候选的真实质量。
2. 固定候选的log MSE/log LPIPS和对应RGB梯度一次性缓存为FP32。
   不每次训练重新运行LPIPS标签反传；是否整体加速必须由GPU测量，不能预先承诺。
3. 强化同一图两个候选的数值质量差分，而非只做先后排序。
   loss为relative1 + absolute.1 + normalized-gradient1 + direction1；
   relative SmoothL1 beta=.05。方向项保护零目标导数和近零预测梯度。
   50%的pair包含当前self-mined候选；其余覆盖proposal/GT轨迹与原候选。
4. 每个训练chunk开始记录各项loss对参数的加权梯度范数，
   避免仅看loss数值就断言某个目标有效。
5. 真实FLUX尾段只检查一次修正，同时保留原始建议与受限建议两个报告项。
   受限版相对原proposal限制RGB RMS和通道均值偏移；
   限制来自训练GT轨迹首步的95分位×1.25，最小1/255。
   此限制未经独立校准，不能称安全保证。拒绝全部候选必须报告零收益。

## 执行阶段

| 阶段 | optimizer steps | 自身反例刷新 | 评价身份 |
| --- | ---: | --- | --- |
| qualification | 8 | step4 | 8张训练图，一次修正 |
| trainfit | 200 | step50/100/150 | 同8张训练图，一次修正 |

两个阶段都fresh初始化；资格权重不用于200步训练。旧失败权重只生成训练反例，
不是初始化或教师恢复模型。200步阶段需要已完成且源码哈希匹配的工程验收结果，
但启动器不自动启动它。

当前并未实现独立来源校准和正式泛化评价：应先证明训练侧可学习，再冻结后续方案。
训练probe不能当unseen evidence；旧8张检查图、calibration池和最终1859开发集均不读。
一次收益也不能自动证明多步TTS或VLM贡献。全部拒绝、仅限幅减少伤害不是达标。

## 保存、恢复与资源

- 默认8张同型号H100/H200（每卡至少75GiB）；支持4卡，不跨节点。
- global batch32候选对：8卡每rank4对，4卡每rank8对。
- 每完整step（含step0）原子保存模型、Adam、warmup进度、每rank RNG、
  next_global_slot和完整日志；哈希错误不静默回退。
- 恢复目前要求同world_size。候选缓存和probe逐图提交，不覆盖历史。
- 共享cache和节点锁防止重复写入，不杀任何无关进程，不自动重启/扩预算。
- 每次尝试均写开始/结束和GPU-seconds；历史旧成本继续保留。
- 固定标签、导数、当前反例和200步全量快照预计25–40GiB，不加入Git。

八卡资格粗估35–75分钟、4.67–10使用GPU-hours；硬上限90分钟/12 GPU-hours。
四卡暂估70–150分钟，硬上限180分钟，使用GPU-hours同上；未实测，不承诺倍速。
独占八卡却仅用四卡时，还须按8×wall time另计整机成本。
八卡200步暂估7–10小时/56–80 GPU-hours，硬上限10小时；四卡上限20小时。
这些不是新代码实测值，200步启动前需用资格速度更新。每次上限外另有30秒清理宽限。

## 已完成验证

- 67项CPU单元/回归测试全部通过（含15项新测试）。
- 两进程真实二阶DDP：参数一致，恢复后下一步与不中断路径逐元素相同。
- 相同global slots下4/8卡候选采样一致；训练来源限制、角色误用拒绝通过。
- FP32缓存质量与导数与原计算逐元素相同（合成数据测试）。
- 真实24训练来源旧缓存哈希、旧结果/反例权重身份已预检，不解码检查图。
- Python/JSON/shell静态检查通过；CUDA尚未验收。
- 早期只有两个CPU预检JSON曾在兼容入口补齐后归档为pre_gpu_revision1文件，
  没有训练缓存/模型被删除或覆盖。

测试命令：

```bash
OMP_NUM_THREADS=1 /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python -m unittest discover -s tests -p 'test_flux_tail*.py' -v
OMP_NUM_THREADS=1 /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 smoke/smoke_flux_tail_relative_energy_v3_cpu_ddp.py
```

## 用户GPU节点入口

默认八卡资格（H100/H200自动检查；同命令可恢复）：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh
```

四卡资格须使用独立ID：

```bash
bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh qualification qualification_4gpu_v1 4
```

未来200步入口（不是当前自动执行的命令；先检查真实资格结果与预算）：

```bash
bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh trainfit trainfit_v1 8 qualification_v1
```

四卡trainfit可用最后参数qualification_4gpu_v1明确指定自己的资格结果。
GPU日志：logs/foundir_flux_tail_relative_energy_v3_qualification_v1.log。
必须同时看到TAIL_RELATIVE_V3_RESULT_VERIFIED和TAIL_RELATIVE_V3_END status=0。
tmux创建不等于CUDA/训练成功；工程通过也不等于修正方向学会。
