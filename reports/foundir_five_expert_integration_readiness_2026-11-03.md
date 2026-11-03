# 五专家接入：保留四类原能力＋新 Rain step800

## 当前结论与授权

Rain step800已在原100张开发图达到34.3011809432 / .06186743590；四节点正常结束。
用户授权下一阶段：接入新Rain、保留四类原专家，明确混合规则，接入通过后冻结专家训练router。
本次交付该序列的第一项GPU入口；不是重新训练五专家，不自动长训。
下一阶段router仍按导师方向：图像分数＋VLM token小投影的独立分数修正，VLM修正零初始化。
当前工程固定比例不是VLM输出，也不构成VLM贡献证据；不把旧FFN-MoE当作本五LoRA bank。

## 官方测试权限：纠正此前过度限制

已完整查阅 `reports/advisor_governance/foundir_official_sow600_qwen_authorization_2026-09-21.json`。
历史记录明确允许五类官方SoW600张反复测、重新训练后再测、调参，并要求保留全部轮次和修改。
此前代理说“只能等整个MoE冻结后才准看官方测试”，忽略了该授权，应纠正。
用户本轮重新确认此约定，并将当前MoE纳入下一阶段工作；详见新的授权补充JSON。

- 仅Blur150、Noise100、Haze200、Rain100、Low-light50，共600；不扩到其它封存集。
- 允许根据公开benchmark结果改进方案；所有测试轮次和修改必须记录，不能宣称独立盲测。
- 官方LQ/GT不得成为训练样例；官方GT不能用于loss、反向传播、逐图挑输出或GT路由。
- 后续可安排五类官方评测，不把五类开发集过线当成额外硬门槛。
- 保留源分离开发数据用于工程和方法诊断；如需独立泛化论证，另保留不参与调参的集合。
- 官方协议是原分辨率及对应pyiqa评分；当前512裁图/torchmetrics开发分数不能直接冒充官方成绩。
  官方入口还须冻结分块拼接、router输入、原尺寸还原、量化及指标，不能直接拿此资格脚本替代。

本轮不打开官方像素，原因是当前测试的是接入前后同一输出能否保持，而不是追加一道科学门槛。

## 单条生成轨迹的明确规则

五份LoRA共同使用原FLUX底座/VAE和50个相同Linear位置，计算各LoRA增量后加权；
不是平均A/B因子，也不是分别生成五张图后融合。无新增CNN、Restormer或RGB refiner。
四类原BF16 LoRA和Rain step800 FP32 LoRA保留；共享底座和VAE冻结。

1. 由权重最大者固定本张图的主专家（平局按Blur/Noise/Haze/Low-light/Rain顺序）。
2. 主专家决定原提示词、输出alpha、量化和读出方式；不使用GT决定这些参数。
3. 主专家是Rain：沿原50步shifted时间表，前25次生成状态，第26次作x_t−t*v端点预测，原生VAE解码。
4. 主专家非Rain：保留完整50次生成及其原输出协议。
5. 内部比例作用于同一轨迹；第0..25次允许五类，26..49次把Rain权重置0，其余四类重新归一化。
   这是固定的兼容候选，不是已验证最优机制。Rain step800没有完整50步速度场资格。

此方案保住单选端点的必要条件，但主专家切换不连续、后半程比例变化也可能伤害混合退化。
不承诺soft无损。若固定soft失败，不自动否定可学习router；如单选失败，先修接入而非训练掩盖。
不为追求形式统一把四类截到26次，也不把新Rain强行推至50次。

## 本轮范围、数据与验收

新入口 `smoke/qualify_foundir_five_expert_integration_v1.py`，配置同名。
新模块 `lowlight_restoration/five_expert_flux.py`；旧绑定文件全部不动。

- 四类各取原replay固定前2张训练图，共8；Rain取原训练槽位12800..12807共8。
- Rain另复用原100张开发图64032..64131，检查接入后是否完整复现step800成绩。
- 共116输入；全部跑原独立入口和五库one-hot。仅16训练图再跑.9主专家＋其它各.025。
  总248份输出，不新增训练/教师缓存，不扫新图，不挑好图。
- 原入口与bank在本轮同一软件/显存策略下比较。decoded最大差≤1e-6且uint8完全一致。
- Rain原100张与历史step800逐图uint8比较，报告两组均值及历史差异；保留原分母。
  完整历史复现且Rain依然>=34.2/<=.08，才标记可进入router方案审核。
- 四类旧H200原输出也保留并报告逐像素差异；新Hopper VAE分块若改变舍入，不伪称历史复现。
- soft仅看能否执行、有限值及完整PSNR/LPIPS变化/逐图退步；不包装科学成功、自动接受或GT选择。
- 输出保存后才评分；Rain旧缓存虽包含GT/teacher字段，生成接口只接LQ，从不生成新teacher。
- 所有专家冻结，未构建优化器、无参数更新。当前适配器明确拒绝autograd训练。
  将来训练router须显式传入call-index/route，不能把本推理计数器用于checkpoint重算。

## 资源与恢复

4节点×每节点2张H100/H200，共8卡；单节点两卡同型号、至少75GiB。
使用Rain已验收Hopper VAE query256分块（全部key/value保留）和cuDNN workspace策略；
同一原入口/bank一起使用该策略，历史四类H200全精度路径不改。
旧四库入口曾峰值137.73GiB，不能把那一入口直接称为H100可用；本次五库需实测。
源码支持H100/H200，但五库CUDA峰值、完整单选和soft尚未验收。

预计15–35分钟、2–4.67使用GPU-hours，仅按Rain推理实测作粗估；硬cap每节点1小时，
四节点合计最多8使用GPU-hours＋清理。需5GiB空闲。没有自动扩预算。
按图/arm原子保存；同run_id＋新的共同launch_id恢复，SHA通过的图跳过。
保存期间中断的孤立tensor保留不覆盖。不杀已有任务；共用节点锁、四节点失败联动。
只复核本轮248份输出和所需旧图，不全扫历史201个checkpoint，也不重新训练或重跑LPIPS审计。

## 目的地执行

本地CPU：38项回归全部通过（测试主体1.051秒，不含共享环境启动/导入等待），覆盖五类
单选与原LoRALinear等价、Rain FP32/四类BF16保留、26/50调用边界、后半程Rain屏蔽、
真实prefix捕获、Rain原量化、116图完整分片、开发禁止soft和模拟目的地receipt结构。
旧算子的router梯度/显式checkpoint重算回归也通过，但新推理计数器仍禁止用于训练。
Python AST、两个bash语法、diff空白与定向凭据/大资产扫描通过。
没有本次真实目的地preflight、CUDA、峰值显存、完整FLUX图像结果或正式router训练证据。

主代理无法访问新节点挂载、控制GPU；CPU代码测试不是GPU通过。
目的地先安全同步origin/main，保留所有权重/数据/缓存和无关脏文件，确认旧Rain已退出。
使用已验证 `.conda/gpu-train-stable/bin/python`：

```
.conda/gpu-train-stable/bin/python smoke/qualify_foundir_five_expert_integration_v1.py preflight
```

预检通过后，目的地按当前node0地址生成四节点命令：
`bash smoke/start_foundir_five_expert_integration_v1_tmux.sh <rank0..3> <current_node0_IP> rain800_bank_4x2_v1 <fresh_shared_launch_id>`。
端口29701，不复用旧IP。所有GPU任务只在新workspace运行。
输出 `checkpoints/foundir_five_expert_integration_v1/rain800_bank_4x2_v1/`；
日志 `logs/foundir_five_expert_integration_v1_rain800_bank_4x2_v1_node<N>.log`。
终态要求 `FIVE_EXPERT_RESULT_VERIFIED`、四个 `FIVE_EXPERT_NODE_END status=0`；
同时检查result中的Rain历史复现及双门槛，退出0不自动代表soft质量好。
目的地首次正常输出及终态及时推小报告，不推原始日志/图/模型。

通过后下一项是冻结专家的router训练配方；正式训练入口尚未交付，不能运行历史五类FFN训练器替代。
官方600评测可按已授权协议安排，不必等所有消融完成，也不自动在本脚本末尾启动。
