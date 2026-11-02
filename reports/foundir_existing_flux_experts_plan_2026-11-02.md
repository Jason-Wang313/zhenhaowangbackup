# 已有 FLUX 专家复用：先继承四类，再适配 Rain

> 2026-11-02 资源纠正：下文单节点4/8卡入口已废止，不要执行旧命令。
> 用户当前要求四节点×两卡，总八卡；科学协议保持不变。最新执行说明见
> `foundir_existing_flux_replay_4node_readiness_2026-11-02.md`。旧CPU证据保留为历史，
> 不把旧单节点测试报告当四节点CUDA资格。本文件下文仅供方案沿革参考。

## 用户决策与本次交付

用户于 2026-11-02 明确授权：坚持右图内部路由，直接利用已有 LoRA，
保留四类多步能力、单独适配 Rain。本节覆盖旧五类 retention 正式1000步计划；
不启动该计划，也不续训失败的五类 step2025。另一代理的 FLUX-native energy 不动。

右图没有写 FFN，也没有规定50步。此前 FFN 是代码实现；旧PPT handoff 的
“no LoRA experts”是旧实现约束，不是图的要求。此次明确改为已有 LoRA expert bank，
不是继续“五块各两个FFN专家＋一份共享Rain LoRA”。不得再混淆这两种结构。

本次完成的是第一阶段 CPU 资产核验和内部 LoRA 混合算子，不是整套训练器：

- `configs/foundir_existing_flux_experts_v1.json`：五份权重来源与原协议索引。
- `smoke/audit_foundir_existing_flux_experts_v1.py`：只读哈希、真实张量尺寸/精度/
  finite 检查、Rain及四类历史声明训练池的跨任务来源/GT哈希隔离核验。
  不打开图像，不导入旧评估入口去访问开发像素。
- `lowlight_restoration/existing_flux_experts.py`：冻结共享 Linear 与独立 LoRA 分支，
  对各分支计算结果加权，不平均A/B因子；显式传入比例，避免梯度检查点重算时
  读到另一张图的路由状态。静态单专家包装器供原采样器重放，不是已训练router。
- `tests/test_existing_flux_experts.py`：算子单选等价、BF16、加权公式、router梯度、
  重算一致、加载失败不改原模型、拒绝endpoint专家等CPU测试。

**没有新GPU运行，没有新PSNR/LPIPS，没有完整模型重放通过结果；Rain训练器尚未交付。**
旧目录仅作为只读资产源；不在旧目录执行/恢复GPU任务。目的地资产仍须核验。

本地实测：12项CPU测试全部通过（60.653秒，不含Python/torch导入）；FP32与BF16
小矩阵单选结果均与原LoRALinear逐元素一致，checkpoint重算后的router梯度一致。
第二版真实资产报告为 `foundir_existing_flux_experts_source_audit_v2_2026-11-02.json`：
五份LoRA、共享base/VAE、Rain prompt缓存哈希通过，500个LoRA张量均有限。
声明训练池为Blur105057/7339来源、Noise16/16、Haze1580/1580、Low-light16/16、
Rain38334/5780；来源和GT哈希与当前跨任务非训练集合无交叠。
这只核对本地微调训练清单，不声称审计了FLUX底座预训练来源，也不声称目的地已通过。

## 我们保留什么

| 分支 | 权重 | 原恢复流程 | 原输出系数 |
| --- | --- | --- | ---: |
| Blur | full_v1 step10000 | 50步noise-to-clean flow | .75 |
| Noise | capacity_sanity noise step400 | 同类50步flow | .30 |
| Haze | source_diverse_continuation step3000 | 同类50步flow | 1 |
| Low-light | capacity_sanity lowlight step400 | 同类50步flow | .925 |
| Rain | precision_resume continue_v1 step4000 | 一步degraded endpoint | 原流程不改 |

精确路径、SHA和提示词见审计JSON，不凭表中的简称找权重。
四类共享FLUX.2 Klein Base4B与VAE、rank/alpha32、同50个LoRA位置。
每份LoRA 32,440,320参数；四类保存为BF16，Rain保存为FP32。保持原精度，
不能为了统一加载把Rain偷偷降成BF16。50是原采样设置，不是MoE必须运行的次数。
输出alpha不是LoRA alpha：前者在RGB端与LQ混合，后者是低秩增量缩放，不能互换。
原双uint8顺序按各任务原评估代码保留；原配置里的简写不取代真实实现。

Rain历史训练用过Restormer行为教师，这段历史不能抹掉；新阶段不加载/运行Restormer，
只允许已有Rain FLUX作为教师，恢复网络只有FLUX及其原生VAE。

## 依次执行什么，为什么

### A. 继承检查，不更新任何专家

先核验新目录与旧资产摘要一致。接着新建只选一个专家的内部bank，
对同一训练图片，用原提示词、原采样、种子、裁图、精度和输出处理，比较原单LoRA入口
与新bank入口。单选等价是工程问题，不是再要求所有任务过一次科学门槛。
初始建议四类各32个训练来源；这批记录必须先按元数据冻结，不能挑质量好的图片。
GPU完整重放入口还未交付，不运行旧full-development脚本替代。

同一任务同一图片同一设置的两条实现应一致。CPU小矩阵逐元素一致不代表CUDA完整
50步也已一致。若不一致先定位加载/精度/条件/量化，不用调训练掩盖。
原专家历史指标和新同环境原入口对照都保留，不悄悄更换较弱分母。

### B. 先解决共同条件，不假定四个50步专家自动能混

四类原提示词不同，输出系数也不同。单选继承通过后，在训练侧匹配比较共同提示词
对原专家的影响，再冻结混合时的提示词/输出协议。不得直接输入GT类别选择推理prompt，
也不把一个alpha加权平均值当作已经证明有效的混合策略。
这些接口问题尚未解决，因此不宣称完整四类或五类MoE已可训练。

### C. Rain多步适配，不破坏原一步Rain

仅创建独立的新Rain LoRA分支；原成功权重只读保留，四个多步专家冻结。
新分支计划复制成功Rain权重作为初始化，fresh Adam；不是随机底座，也不是严格续训。
初始化可用不等于它已经会处理带噪中间状态，需要验证step0并完成实际训练。

共同flow采用原四类的原生LQ reference、noise-to-clean定义：
`z_t=(1-t)*z_GT+t*epsilon`，目标速度为`epsilon-z_GT`，
预测干净latent为`z_t-t*v_theta`。t从噪声端1走向干净端0。
这是重新学习不同t下的速度，绝不是把一步Rain重复调用50次。

配对GT是主要监督；冻结一步Rain仅在训练裁图上提供辅助恢复示范。
它没有正确50步轨迹，不能拿一个endpoint残差当作所有t的速度标签。
教师输出可以离线生成以避免常驻第二个FLUX；每条示范须绑定教师SHA、原精度、
训练record/crop/augmentation和prompt，不能把某个裁图示范用于另一个裁图。
教师辅助项的精确定义/权重、prompt迁移和工程预算须在训练配置中明确后再启动，
本交付没有伪装成已冻结这些未实现部分。

从旧Rain失败学习：不能只看随机t的flow loss下降。工程阶段应同时检验真实50步
恢复、FP32训练状态、有限梯度、保存和恢复；科学验收仍是原Rain100图≥34.2/≤.08，
工程小样本不替代该门槛。先25步工程验收，独立正式规模由实测吞吐/结果确定，
不自动长训，也不认为25步失败就证明多步Rain不可能。

### D. 能力可用以后，接右图router

五类专家协议兼容后，先冻结base/专家，训练图像router和VLM小投影；
VLM分数修正从零开始，`softmax(image_scores + alpha*vlm_scores)`。
不生成五张完整图片再做像素平均，不另加CNN/Restormer，不把专家名称当推理GT输入。
单类保持后再现有混合退化。是否需要小学习率适应专家由结果决定，不预先解冻全部。
证明VLM贡献最终需要同一专家库/数据/预算的匹配image-only对照；当前暂不启动该消融。
LoRA-MoE或VLM路由本身不宣称首创，收益和创新仍待实验证明。

## 执行边界与下一位代理

- 唯一GPU根：`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
- 按用户本轮提供的最新资源规则：默认单节点8×同型号H100/H200，保留4卡兼容；
  不沿用旧4节点×2卡或旧IP。GPU入口未交付，不能据旧启动器宣称已兼容。
- 主代理实现/审查/推送；目的地CPU代理安全同步、核验资产并回传小报告。
  不覆盖数据/权重/缓存/脏文件，不重启energy或旧retention，不自行补写训练方案。
- 目的地用已核验Python执行 `smoke/audit_foundir_existing_flux_experts_v1.py`
  的 `--asset-root` 指向新根，`--output` 指向新建的临时目录内JSON。
  脚本输出必须在资产根外，防止写入冻结来源。将小报告focused提交，不上传大资产。
- 新目录本会话不可见：旧目录审计通过不是目的地通过，不伪称已启动训练。
- official/sealed/internal/parent-test及最终1859 Noise像素不得进入本次核验/训练；
  当前只读manifest元数据。将来训练与验证来源和相同GT哈希继续隔离。
- retention回传原始日志文本与标注SHA不一致仍待目的地核对；不抹掉该审计缺口，
  也不误说成新LoRA资产检查失败。

当前CPU交付消耗0 GPU-hours。完整重放、Rain资格和正式训练的ETA尚无本任务实测；
不复用旧一步MoE每步7秒去估50步方案，不给出无依据的训练结束时间。
