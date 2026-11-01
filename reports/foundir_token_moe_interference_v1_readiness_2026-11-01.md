# 五类扩展退步：只读交叉检查准备 — 2026-11-01

## 当前结论

用户授权定位五类25步验收的退步原因，不是授权照原配方正式长训。
本地已执行回传报告的CPU逐图审计，并实现零训练更新的四组交叉检查。
本会话看不到迁移后的GPU目录，**GPU诊断尚未运行**，不能声称已找到唯一根因或已修复画质。
不重跑25步，不增加专家、不加入CNN/Restormer/energy、不启动image-only消融。

证据来自6fbb32a回传的
`reports/foundir_token_moe_five_single_v1_terminal_2026-11-01.json`，
SHA `27430079d43e176d8ed1f09758f343a9411f68298f33dfd9b7306137ba816f83`。
CPU重算结果保存在 `reports/foundir_token_moe_interference_cpu_audit_2026-11-01.json`。
核对了相同500张图的record/source/split和两次逐类均值，没有读取图片像素。

已经确认：

- Rain有99%、Blur有96%、Noise有74%的预览图PSNR下降，不是少数坏图拖低平均值。
- Rain后四个MoE块训练后仍有约99.998%以上平均权重给原专家0；首块则从.913降到.788。
  因此不能仅说“所有层都选错专家”。但首块及共享特征变化仍可能重要，不能仅凭后四层就排除路由影响。
- Blur后几个块的原专家1占比从约99.5–100%降至约74–86%，路由确实也发生了明显变化。
- 起点Low-light/Haze的平均RGB像素MSE分别约为Rain的109.52/34.19倍。
  **这只是开发图误差尺度，不是训练梯度测量**；支持调查任务间更新失衡，但不能证明它就是根因。
- Noise平均RGB MSE从.003393降至.001212，但平均PSNR下降1.540dB、LPIPS变差.022325。
  说明改善少数高误差图可以降低平均MSE，同时伤害多数图片的指标；不能用总体loss下降代替逐类质量。
  MSE按每张PSNR分别还原后求均值，没有错误地从平均PSNR反推平均MSE。

静态代码核对：五类仍共享一套LoRA和每块两个专家，同时更新image/VLM路由；
原二类CE已按计划取消，未给两个专家输入五分类标签。每个优化步含五类中的四类，
五步周期才完全均衡，各类25步合计160次曝光。当前总loss没有逐类梯度分解，
不能从起止两个不同任务组合的loss直接推断优化是否改善。

## 为什么做这四组

仅换回旧router参数还不够：新的前层权重会改变图像特征，即使用旧router也可能得到新比例。
本检查先分别运行step2000和2025，记录**同一张图在每个块真正得到的两个专家比例**，
然后固定这些比例重放，改变修图权重。不是使用全数据平均比例，也不看GT选择比例。

| 组别 | FLUX修图权重 | 每图每块专家比例 | 用途 |
|---|---|---|---|
| old_body_old_gates | step2000 | step2000自然产生 | 同环境旧起点 |
| old_body_new_gates | step2000 | step2025自然产生 | 仅改变实际专家选择 |
| new_body_old_gates | step2025 | step2000自然产生 | 固定旧选择，看修图权重变化 |
| new_body_new_gates | step2025 | step2025自然产生 | 同环境退步终点 |

“修图权重”包含FFN专家及共享LoRA；冻结基础FLUX/VAE不变。
重放时绕过image和VLM两个分数分支，使用已记录的softmax比例，原两个专家及融合公式不变。
诊断override仅存在于独立脚本的临时eval上下文，离开后恢复原forward；
生产模型、旧训练器、旧配置以及另一代理energy源码均不修改。

每个原模型还须用自己的比例重放全部500张：浮点输出最大差≤1e-5且uint8图完全相同。
不通过就停止，不能用错误重放结果分析原因。总计四组指标＋两组自重放，3,000次单图前向，
每卡只驻留一个FLUX，按旧→新→旧依次加载，不同时放两个4B模型。
复用完整资格运行的500份开发Qwen特征，不重新加载Qwen提取、不打开训练/混合图。

这些交叉结果是诊断，不是可部署提升，不用GT择优，也不拿最佳交叉组包装论文指标。
报告四组完整逐类PSNR/LPIPS、逐图结果、与历史两个自然模型的差异，以及非加性交互项。
新环境的自然结果单独列出，不能只展示交叉组或隐藏与旧环境的差别。

## 如何根据结果决定修哪里

- 新修图权重＋旧比例仍严重退步：共享LoRA/专家变化不能忽略，单纯修router不足。
- 旧修图权重＋新比例就明显退步，而固定旧比例能救回：优先处理路由漂移。
- 两种单独改变都损害或交互很强：需要同时约束学习，不能用一个“错误百分比”强行归因。

这些是在两个固定checkpoint和同一批图上的功能干预，不直接测训练梯度冲突。
即便“比例变化”主导，也不能直接归罪于VLM：图像router和上游特征也参与自然比例生成。
本次尚不把共享LoRA与每个专家逐一拆开，也不自动决定降学习率、加专家或新损失。
先依据结果提出一个针对性的训练修订，再冻结训练规模。25步不足以宣布五类路线失败。

## 源码、状态与安全

配置：`configs/foundir_token_moe_five_single_interference_v1.json`。
入口：`smoke/run_foundir_token_moe_five_single_interference_v1_4x2.sh`。
程序：`smoke/diagnose_foundir_token_moe_five_single_interference_v1.py`。
只读父成功step2000与资格step2025权重；验证父/子报告、配置、31项科学源码、
数据及特征绑定、权重SHA。CPU preflight另输出共享LoRA/专家0/专家1/image router/VLM router
各组参数变化统计；参数变化大小本身不当作质量影响证据。

不创建optimizer、不反传、不写模型/Adam、不删除任何文件；成功父step2000保持不动。
新结果写入独立 `checkpoints/foundir_token_moe_five_single_interference_v1/<run_id>`。
各rank每完成一个阶段即保存小JSON；中断保留它们，使用新run ID重做只读评估，
没有训练resume参数、不触碰旧运行目录、不自动重试或接正式训练。
所有开发图沿用资格100×5预览（Noise/Low-light仅34/21来源）；不是完整五类门槛。
GT仅用于评分，official/sealed/internal/parent-test像素不读取。

资源仍4节点×每节点2张H100/H200，共8卡；节点内同型号，跨节点可混用。
共享根仍 `/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
预计.2–.5小时（12–30分钟）、1.6–4.0 allocated GPU-hours，单次硬cap1小时，
至少5GiB新空间。估计包含只读评估和加载，不保证新节点/网络达到该速度。
记录所有节点墙钟与GPU-hours；不只报告模型forward时间。

## CPU检查和目的地交接

新单元检查覆盖原比例重放、禁止训练、无GT条件、不同图片不同soft比例、
两层网络上游特征变化时仍固定下游比例、异常后恢复forward、权重不变、
四组配对和交互计算、CPU权重差统计、真实31项源码/元数据身份。
最终44项CPU unittest（原31＋新13）全部通过，30.043秒；语法/JSON/shell检查通过。
新/旧真实大权重差统计仍需目的地CPU预检，这里只完成小张量逻辑测试与回传数据审计。
原MoE续训/资格回归也保留。真实4B CUDA的自重放及四组结果仍待运行。

目的地代理安全同步本次提交，保留本地修改、energy工作、全部权重/数据/特征/日志。
白名单：新诊断config、`smoke/token_moe_interference_v1.py`、新诊断程序和启动器、
`tests/test_token_moe_interference_v1.py`、本报告及CPU audit JSON、三份交接文件顶部新节。
本次不修改e275292资格代码，不重跑25步。需已有6fbb32a终态报告与对应共享资产。

在已验证项目Python后接以下参数，先CPU预检：

```text
smoke/diagnose_foundir_token_moe_five_single_interference_v1.py --phase preflight --run-id diagnosis_4x2_v1
```

保存预检输出作为小报告。通过后确认当前四节点地址/空闲卡，给用户tmux启动命令；
参数依次rank0/1/2/3、当前node0 IP、共同新run ID，端口默认29765，不复用历史IP。
不自行启动GPU；不启动formal/mixed/energy。若只需重算现有回传数据，用`--phase report`，不需要GPU。
终态回传全部四组逐类/逐图结果、1000次自重放检查、历史原模型差异、状态/耗时/峰值显存；
推送小报告，权重与原特征留共享目录。诊断完成后先解释证据，再改训练，不自动扩实验。
