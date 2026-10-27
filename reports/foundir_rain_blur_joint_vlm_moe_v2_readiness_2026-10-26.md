# Rain／Blur 联合训练 MoE V2：image-only 与 VLM 对照

最终状态（2026-10-27）：两条formal_v1均已10000步正常结束、退出0，不要再执行下方启动命令。
Rain两条都达标；完整Blur2595两条PSNR达标但LPIPS未达标。VLM表现为小幅LPIPS改善伴随Blur PSNR下降。
结果与完整断点已核验，监控任务完成；不自动续训或启动新实验。
详见 `foundir_joint_moe_v2_terminal_2026-10-27.md` 和 `foundir_joint_moe_v2_final_comparison_2026-10-27.json`。
以下启动／运行状态仅为历史记录。

最新启动核验（2026-10-26 20:05:52 UTC）：两条正式训练均已实际开始。
image-only 在8×H100到480步，VLM在8×H200到60步，日志未见运行错误。
第1000步前尚无训练后质量验证。正式step0输出存在微小跨硬件数值差异，
不能称逐位一致或把该差异算作VLM收益。不要再次执行下面的启动命令。
详情见 `foundir_joint_moe_v2_parallel_formal_start_2026-10-26.md`。
以下“VLM未启动”或单节点顺序安排均为历史状态，以本段为准。

最新资源决定：用户另有一台空闲八卡节点，可与当前image-only并行运行VLM对照。
image-only已于19:45:55Z在8×H100启动，核查时正常推进至70步，峰值42.57 GiB，
无非有限损失／梯度或退出标志；全部29项代码hash和数据身份、初始评估一致。
新VLM正式日志／输出尚不存在。不要停止image-only，也不要修改已运行的代码／配置。
在另一台空闲同型号8×H100或8×H200节点运行下面的VLM正式命令；优先H100便于硬件匹配。
这是两条独立单机任务，不是跨节点16卡DDP。无需master IP／node rank；
每条仍使用独立全局batch32，相同10,000步／seed／数据／损失，独立从原Rain权重初始化。
输出按arm隔离，锁和tmux socket在各节点本地；不需要改启动器。
VLM单条仍暂估5.45–9.06小时／43.63–72.45 GPU-hours；两条并行总计算量仍87.26–144.89 GPU-hours，
但等待时间通常接近一条任务，受共享存储和另一节点硬件影响。以下顺序运行限制为旧的单节点安排。

最新授权：用户已批准正式阶段。先在单机8×H100／H200运行
`image_only/formal_v1` 10,000步，结束后再独立运行匹配的VLM臂10,000步。
不从25步验收权重续训，不改已验证的配置／代码。当前Codex没有GPU，
正式日志／输出尚不存在，需用户在GPU节点执行下面的正式命令。
实际训练器的正式资格检查已通过：两条验收记录、四个权重／Adam文件hash、
配置／源文件／数据身份一致；正式ID未占用，shell语法通过，未读取受保护像素。
每臂暂估5.45–9.06小时／43.63–72.45 GPU-hours；两臂顺序合计
10.91–18.11小时／87.26–144.89 GPU-hours。第1,000步首次训练后评估预计约35–50分钟。

立即启动第一臂：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh formal image_only formal_v1
```

在另一台空闲八卡节点上启动第二臂（现在可以并行；不要在image-only所在节点执行）：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh formal vlm_assisted formal_v1
```

tmux starter退出只代表创建会话；两节点分别启动，不要在同一节点连续启动两条。
训练仍为Rain／Blur两任务；完整Blur2595只在最终10,000步评估。
保存250步、每1,000步和预定评估节点的完整权重／Adam／八rank RNG；
后续如果中断，需按同臂完整checkpoint真续训，不能直接重用已占用ID。
以下旧的“正式尚未授权”语句仅为历史记录，以本段和AGENTS顶部为准。

最新终局：两条独立25步工程验收均已通过。VLM臂于19:34:33Z正常结束，
8×H100，278秒／.6177777777777778 GPU-hours。VLM网络实际更新并改变路由，
但与image-only的恢复指标差异极小且有好有坏，尚无明确效果优势。
原始指标、核验记录见 `foundir_joint_moe_v2_vlm_qualification_terminal_2026-10-26.md`。
下方两个qualification命令均已执行完，不要重复。
正式训练尚未授权或启动；每臂10,000步暂估5.45–9.06小时／43.63–72.45 GPU-hours，
单机八卡顺序完成两臂合计10.91–18.11小时／87.26–144.89 GPU-hours。

以下为按时间保留的授权／验收记录，以最新终局为准。

最新授权：用户已批准单独执行 `vlm_assisted/qualification_nvls_v2` 的25步验收。
代码／配置与image-only保持一致；从原始Rain step4000独立初始化，不从验收权重续训。
当前Codex无GPU，新VLM日志／输出尚不存在，需用户在八卡节点执行下方VLM命令。
暂估10–20分钟／1.3333–2.6667 GPU-hours，基于上一臂实测7.3分钟；
共享存储较慢时保留原.81小时／6.47 GPU-hours保守预算。没有启动正式训练。

最新终局：image_only/qualification_nvls_v2 已于19:20:40Z完成25步，8×H100，
工程验收通过、launcher状态0；总耗时438秒／.9733333333333334 GPU-hours。
Rain100为33.86460822453734/.07575711041688919，相比step0两个指标退步；
Blur100预览为24.26987270296453/.4072400273568928，相比step0两个指标改善，
但没有达到质量要求，也不是完整Blur2595评估。VLM臂与正式训练均未启动。
不要重复执行下方已完成的image-only命令；下一步VLM工程验收已获授权，尚待GPU节点启动。
详见 `foundir_joint_moe_v2_image_only_qualification_terminal_2026-10-26.md`。

以下启动失败／修复记录按时间保留，以本页最新终局和AGENTS顶部为准。

更新：image-only/qualification_v1 已在18:58:49Z因NCCL/CUDA401初始化失败而结束，
训练0步。下面的CPU验收记录仍成立，但不是GPU验收通过。
用户随后批准修复：实际启动器已强制关闭NVLS、保留P2P，并加入限时八卡通信检查。
GPU端尚未验证恢复。新ID为 `qualification_nvls_v2`，不要重复旧ID。
失败记录见 `foundir_joint_moe_v2_image_only_launch_failure_2026-10-26.md`；
修复与测试见 `foundir_joint_moe_v2_nccl_repair_2026-10-26.md`。

日期：2026-10-26。状态：CPU／真实资产预检通过；首次 GPU 工程验收在通信初始化时失败，训练0步。
当前没有新的恢复指标，不能宣称模型改善、论文创新成立或 SOTA。

## 用户决定

取消“新单步 Blur 单专家先达到全部指标，然后普通 MoE，再 VLM”的串行前置。
改为同一恢复骨架的两个匹配实验，专家和路由一起训练。image-only 的质量
失败不再禁止 VLM 对照；两者的工程验收仍必须通过，正式训练不自动启动。
本次限于 Rain／Blur 两任务，不扩展到五类或复合退化，不恢复旧失败实验。

## 我们发现什么问题，具体改什么

| 已有证据／限制 | V2 的具体处理 | 尚未证明的部分 |
|---|---|---|
| 去掉 router、只训练 Blur，仍约 26.0962/.32610 | 不再把主要差距归因于 router | 不能由此断言 VLM 联合训练无效 |
| 原方案冻结 Rain LoRA，只训练五个 FFN | 两条对照都开放原 50 个位置的共享 rank32 LoRA，同时训练 FFN 专家 | 开放空间交互相关参数是否足够，待 GPU 结果 |
| Blur 仍收到固定 Rain 文本 | 两条对照都采用已缓存的空文本条件 | 新初始化的 Rain 指标可能变化，不能冒充原 Rain 结果 |
| Rain 专用变化权重／保持输入规则可能不适合 Blur | 去掉这两项，去掉 Rain-only Restormer 教师；使用统一 GT 恢复损失 | 这是成组修复，不是各因素的单变量因果证明 |
| 原始 token 特征只够小规模子集 | 使用全部已有逐图 VLM 五类强度预测，经小 MLP 加到 router | 本次不是 raw-token 输入；中心448诊断与随机512恢复 crop 有差异 |
| 旧新分支只接触48,000次 Blur 样本 | 预备每臂10,000步、160,000次 Blur＋160,000次 Rain | 仍少于旧 Blur 320,000次呈现，不能声称训练量完全匹配旧模型 |

保留原成功 Rain、旧 Blur、失败 MoE／单 Blur 的所有结果和权重，不覆盖。
V2 仍是单步 t=1、零 anchor noise，不能把“一步已可行”当成结论。

## 两条对照唯一的区别

- `image_only`：图像特征产生两个专家的权重。
- `vlm_assisted`：相同图像 router，再加上 VLM 预测经 `5→256→2` MLP
  产生的两个分数，然后正常 softmax 合并同一组 FFN 专家的输出。

两者都从原 Rain step4000 独立初始化，不使用失败模型或25步验收权重。
共享 LoRA 是全模型唯一一份共享适配参数，不是第二个专家库。
每个双流 block 有两个普通完整 FFN；五个 block 均参与。

训练参数：

- 共享 LoRA：32,440,320。
- FFN 专家：849,346,560。
- 图像 router：3,966,730。
- image-only 合计：885,753,610。
- VLM 分支仅增加10,250个训练参数；VLM-assisted 合计885,763,860。

两臂都实例化相同模块；image-only 的 VLM 小网络冻结且不执行。
VLM 输出层零初始化，所以两臂初始恢复输出严格一致。
图像专家合并使用 `E0 + w1*(E1-E0)`，保留相同专家插入时的输出。
共享原模型参数本体冻结，仅共享 LoRA、FFN 专家、router 训练。
训练参数和 Adam 状态使用 FP32，矩阵计算 BF16。
所有双流／单流 transformer block 使用 activation checkpointing 节省显存；
CPU 双进程测试覆盖这条反向传播路径，GPU 显存是否安全仍待验收。

## 条件来源与数据边界

只读取 `qwen_levels_json`，绝不读取 `type_oracle_levels_json` 作为 VLM 输入。
五个值按 Blur、Noise、Rain、Haze、Low-light 排列，等级1–5映射为0–1。
例如预测 Rain 等级4对应第三个值0.75，不是直接强制选择 Rain 专家。
不使用 LayerNorm 抹掉绝对强度。不运行新的 Qwen／Qwen3，也不重新生成条件。

条件缓存是既有冻结 V8、中心448 crop 的逐类诊断。用于整张图的条件，
未对每次随机512 crop重新预测。这可能影响细粒度强度信号，应如实披露。
Qwen 诊断有额外推理成本，缓存不意味着最终部署时这部分免费。

- Rain：38,334训练记录，5,780来源组。
- Blur：105,057训练记录，7,339来源组。
- 每全局 batch32：16 Rain＋16 Blur；来源均衡抽样。
- 两臂同 seed、来源／记录／crop／D4次序。训练512 crop，不缩图。
- 评估沿用原 Rain100／Blur2595 固定512 crops；Blur100只作中途预览。
- 训练与开发来源／GT hash交叉排除结果不变，无新增交叉记录。
- 146,086个允许记录全部连接到真实 VLM 预测，按 ID／来源／LQ+GT hash核对。
- 大缓存含其他 split 的元数据；只保留允许 ID，没有读取受保护像素或其指标。
- 不接触 official／sealed／internal／parent-test 像素。

连接后预测字典 SHA-256：
`b10b91909b471ddafa42bd77b1f9433030ef06be495b187cd5cb8f4d9d9c8fc0`。

## 训练目标与对照公平性

两臂完全相同的目标：latent velocity MSE×.1、clean latent L1×.25、
解码 RGB MSE×10、Charbonnier×1、high-pass L1×.1、LPIPS×.05。
不增加 pixel refiner、energy、第二 MoE、额外专家种类或新迭代机制。
没有依据把 LPIPS 损失值占比直接当成梯度占比，因此本轮不盲目放大其系数。

训练初期同样使用真实任务标签 warmup，并保留相同 router CE×.01；
正式 hard250／transition-end500，验收 hard6／end12。
这些标签只用于训练监督，验证和部署只用图像及真正的 VLM 预测。
两个实验的共同修复可能改变相对旧模型的结果，因此不能把“V2对旧V1的改进”
直接归因于 VLM；只能用 V2 内两臂的匹配结果判断 VLM 的作用。

## 规模、费用与续训

当前只交付25步工程验收，两臂各独立一次，单机同型号8×H100／H200。
每臂暂估 .20–.75 普通小时、1.6–6.0 allocated GPU-hours；这不是已测速度。
新增无更新通信检查上限180秒，另有30秒强制退出宽限；最坏额外约.05833普通小时／.46667 GPU-hours。
验收内容：原模型到 MoE 的插入一致性、两臂初始化一致、梯度、各分支实际更新、
显存、归一化权重、FP32 checkpoint／Adam、固定开发输出。25步不用于质量结论。

预备正式训练每臂10,000步，即320,000样本呈现（不是唯一图片数），
每任务160,000次。正式时刻0／1000／3000／6000／10000评估；
完整 Blur2595 只在最终10000步评估，不挑选最好的开发 checkpoint。
正式的耗时／GPU-hours必须根据两臂验收速度重新估计并向用户报告；
在用户决定正式启动前不代为启动。不能使用验收 checkpoint 初始化正式训练。

第250步、每1000步和评估节点存完整权重／Adam／八 rank RNG。
两臂各约117 GB训练状态，另加输出图和指标；不自动删除旧文件。
断点续训只允许同配置／同源文件／同臂未完成正式状态，使用新的输出ID；
恢复优化器和绝对数据位置，不把重置优化器称为续训。

## 怎样判断结果

固定最终节点逐图匹配比较 image-only／VLM。分别报告 Rain 与 Blur：
PSNR、LPIPS、获益比例、按来源组的配对 bootstrap 区间、专家权重。
还记录恒定专家0／1，以及 VLM 模型关闭 VLM 分支的预览诊断。
bootstrap只反映这次训练的样本差异，不替代多 seed 不确定性。

绝对目标保留：Rain >=34.2／<=.08；Blur >=25.9／<=.18。
VLM 相对改善与两任务绝对目标是两件事；平均值也不能隐藏逐任务退步。
在坏 baseline 上获得微小改善，仍不能宣布论文成立。
沿用相关工作审计，禁止宣称首个 VLM-guided MoE／MoE+DiT。
当前反复使用的 development 不是新的独立测试，不能冒充正式最终测试成绩。

## 验收记录与下一条命令

- 新测试套件18项通过，包含4项复用数据测试与14项联合 MoE 测试。
- 覆盖2-rank CPU DDP、整块重计算与普通反传一致、同臂真续训等价。
- meta-device 实例化真实 FLUX，核对50个 LoRA位置和上述完整参数量。
- Python／shell语法通过。
- 真实模型、原协议、条件缓存和全部开发图片 hash／训练路径预检通过。
- GPU更新：image-only和VLM两臂25步工程验收均通过；正式训练／完整Blur2595质量结果尚未产生。

以下image-only命令已完成，仅保留为复现记录，不要再次执行：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh qualification image_only qualification_nvls_v2
```

以下第二臂命令也已完成，仅保留复现记录，不要再次执行：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh qualification vlm_assisted qualification_nvls_v2
```

两条命令不能在同一节点同时执行。启动器有同节点锁，tmux创建不等于训练成功。
先确认 `CURE_NCCL_DDP_REHEARSAL_PASS`／`JOINT_MOE_COMM_CHECK_PASS`，再看实际训练及最终 `engineering_pass`。
无自动其他臂／正式训练。后续正式比较工具：
`smoke/compare_foundir_rain_blur_joint_vlm_moe_v2.py`。

配置：`configs/foundir_rain_blur_joint_vlm_moe_v2.json`
SHA-256：`3bdd59b3fb3c010cc7a93ab299f3f67fe85cc5eb5a6f380f610d4338398baebe`。
