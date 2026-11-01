# VLM-token MoE：保持配方，正式step1000→2000真续训

## 用户决定与本轮问题

用户选择先改善当前VLM-token版本，暂不做image-only匹配对照。
本轮只问：在现有一步统一MoE上，继续原优化是否能补齐Blur LPIPS且守住Rain？
不增加结构、不改变损失权重或学习率、不去重新训练独立Blur专家。
依据是正式1000步的完整Blur两指标改善，而不是25步资格或小预览。
这支持有限续训，不证明继续训练一定过线；Rain PSNR余量只有.073686dB。
1000步上限之外不自动继续；下一决策需看新完整结果。

主代理本次只实现、测试和push；没有新GPU挂载，不声称已启动。
目的地仍为
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
四个不同节点，每节点两张H100/H200；节点内同型号，节点间允许混用，共八卡。
另一代理的FLUX-native energy显存工作不修改、不停止。

## 必须保留的三个参照

| 系统 | 完整Blur PSNR | LPIPS | 解释 |
|---|---:|---:|---|
| 历史独立Blur FLUX | 26.559141 | .151993 | 50步＋固定.75输出混合，已过SOW平均指标 |
| image-only MoE原历史起点 | 26.619521 | .188334 | 一步统一Rain/Blur，未过LPIPS |
| 当前VLM-token step1000 | 26.705142 | .183430 | 比本轮step0改善，但仍未过LPIPS |

历史独立Blur另有最差单图退步安全检查失败，不等于SOW平均指标没过。
旧独立结果出处：`foundir_blur_only_flux2_scalar_residual_full_development_terminal_result_2026-09-16.md`。
统一MoE没有加载旧50步Blur LoRA，曾在不同的一步端点上重新学习Blur。
故不能声称“在已达标的Blur专家上仅加router”或“当前已经全面优于旧Blur”。
历史结果不是本次同环境逐图配对对照；推理成本也不同，不能声称等算力胜出。

本次parent同环境step0为Rain34.417020/.064384、Blur26.619362/.188329；
step1000为Rain34.273686/.062162、Blur26.705142/.183430。
最低门槛仍Rain≥34.2/≤.08、Blur≥25.9/≤.18。
Blur LPIPS差.0034295；跨过.18只是过最低线，不是追平独立专家.151993。
没有匹配消融，VLM贡献仍未证明，不用旧专家或image-only收益包装VLM创新。

## 严格续训，不重置学习状态

配置：`configs/foundir_token_residual_moe_continue_v1.json`。
自动起点唯一绑定为
`checkpoints/foundir_token_residual_moe_formal_v1/formal_4x2_v1/step_001000.resume.json`，
SHA `e8d2f068eb4f436bb7420d8eeb4b973372bbb22c234a51f689cc5a8afeb9af35`。
父终态GitHub证据为c9336a7提交的同日formal terminal JSON，SHA
`8ddd2e58d7b4fdda424a39f4dfd99a75b804dd9b8ed7682b3ecdf29921d87923`。

先核验父报告、配置、源码、launch/result/baseline/收据，再恢复原权重、完整Adam
和8rank RNG；CPU preflight还核验模型/Adam大文件SHA。加载时再次核验文件身份。
只允许helper/trainer编排变化，模型、损失、采样、FLUX/Qwen等仍必须匹配父运行。
旧源码/配置的收据不伪装成新绑定：只对这个获授权、hash-bound父终态开放一次过渡。
其后每份新收据均绑定本轮代码/配置/数据，中断恢复仍严格匹配，不普遍放宽断言。
原image-only step11000权重仅用于构建原结构和零接入检查，正式更新前必恢复step1000；
VLM已训练权重和Adam不能被零初始化/fresh Adam覆盖。

全局优化步1001–2000；训练曝光32000次，从绝对采样位置384000到415999。
不重放之前352000–383999的采样位置；源均衡取样可能再次看到同来源，这是正常续训。
Rain/Blur全池、512裁图/D4、batch32、每步16/16、原loss与LR均不变。
没有旧Restormer teacher/CNN新调用；Qwen冻结视觉特征、FLUX原生VAE和固定LPIPS照旧。
不加载25步资格权重、不启动image-only/拼接/energy任务。

## 评估、保存、恢复

- 起点1000重算完整Rain100＋Blur2595，记录实际新环境的恢复结果。
- step1500、2000均评估完整Rain100＋Blur2595，不能用预览替代。
- 新结果保留原formal step0基线SHA，并单独保存本次恢复后的step1000基线。
  终态同时输出相对原step0和本次step1000的变化；报告原parent step1000数值，
  不隐藏跨环境起点差异，不把新分母变弱算作进步。
- step1100、1200…2000保存十组完整checkpoint，Adam计数与全局步一致，receipt最后写出。
- 只用完整收据恢复，partial不能使用；旧结果/日志/缓存全部保留。
- 首次用新入口时不要传第四参数，它自动加载唯一父step1000。
  本次中断后才使用新run ID和第四参数，指定本次step1100及之后的收据。
  若最后step2000权重已存、评估中断，可只补评估，不增加训练步数。
- 1500是固定中途检查，2000为固定结束；脚本不会因指标好看自动加步数或启动新方向。
  不保证Rain不退步；完整公布每次两类指标，不用Blur改善掩盖Rain损失。

新缓存仅包含本轮32000训练裁图＋2695开发裁图，共34695条。
同一开发preview与full使用canonical特征；配置身份改变后重新提取，不冒充旧cache。
Qwen提取和FLUX训练分进程，前者释放后再加载后者。本轮绑定一致的已完成缓存可重试复用。

## 资源预算和工程检查

实测上一轮四节点8×H200为2.6197小时/20.9517 GPU-hours，中位6.025秒/训练步。
本轮多一次完整中途评估，预算2.5–3.5小时/20–28 allocated GPU-hours，不是保证。
单次启动四小时硬cap（含等待、提取、训练、评估与存盘），不自动延期。
H100或混用以最慢rank/网络为准；根据日志更新预估。至少160GiB新空间，不删旧资产。

CPU元数据检查通过：真实旧manifest和父报告绑定一致；34695份spec，
训练首末key是train_000384000 / train_000415999，完整开发2695条。
未读取训练/开发图片像素，不代表目的地模型文件/磁盘空间已核验。
本次测试包括真实tiny模型1000次Adam更新后恢复、逐位权重/Adam/RNG一致、
继续到1001与未中断对照一致、新绑定再次恢复；另检查配方篡改/失败父报告/错误收据拒绝。
23项CPU测试全部通过；八进程Gloo batch32恢复回归通过，继续计算最大参数差
7.450580596923828e-09，满足原atol1e-7/rtol1e-6容差。
Python编译、JSON解析、shell语法和diff空白检查通过。
CPU测试不等于本次CUDA已运行，跨硬件继续计算也不保证逐位一致。

## 目的地代理简短交接

从origin/main安全同步白名单；保留本地dirty、energy工作、所有权重/缓存/日志。
不要reset/clean/整仓覆盖；遇到冲突逐项合并。

- `configs/foundir_token_residual_moe_continue_v1.json`
- `smoke/token_residual_moe_v1.py`
- `smoke/token_residual_moe_continuation_v1.py`
- `smoke/train_foundir_token_residual_moe_v1.py`
- `smoke/run_foundir_token_residual_moe_continue_v1_4x2.sh`
- `tests/test_token_residual_moe_continuation_v1.py`
- 本报告、AGENTS.md、agents.md、current handoff顶部新节。

确认父终态报告已同步。用已验证的项目Python运行：

```text
smoke/train_foundir_token_residual_moe_v1.py --config configs/foundir_token_residual_moe_continue_v1.json --phase preflight --run-id continue_4x2_v1
```

这是跟在Python解释器后的参数，不是直接可执行shell命令；不要改用系统Python。
通过后获取当前四节点地址、空闲GPU、网络，给用户tmux命令，不自行启动GPU。
入口参数依次node_rank0/1/2/3、当前node0 IP、相同新run ID、可选本轮恢复收据。
默认端口29763。PROJECT_PYTHON须指向各节点相同版本的已验证环境。
保留旧正式/资格入口和配置，不把它们改成2000步，不重跑25步资格。
1500和终态回传小JSON/Markdown报告到GitHub，含完整两类指标、逐类增益、
原/新基线核验、运行起止、四节点状态、恢复收据与耗时；大文件留共享目录。
