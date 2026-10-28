# FLUX 生成末段 energy 引导：原文、公式和实现复核

日期：2026-10-27 至 2026-10-28 UTC。用户要求先仔细核查，再重新解释方案。
本轮没有修改训练/推理实现，没有启动 GPU，没有读取训练或受保护测试像素。
GPU-hours = 0。以下实施步骤是建议，不是已完成实验或长训授权。

## 1. 结论

方向有技术依据：冻结 FLUX，在生成过程中评价预计成品，把梯度传回当前 latent，
再让 FLUX 继续生成。它确实不同于最终图片上直接改像素。

但上一轮建议需要收紧：

- 中间态适配不是需要从零发明的问题；已有多种处理方法。
- 不能由此推出旧 energy 已可靠、末尾两三步足够、或双指标必定达标。
- 先分开验证“FLUX 末段能否提供有效修正”和“学到的 energy 能否找到这种修正”。
- 能正确排序的 energy 不一定有正确局部梯度；预览改善也不等于实际终图改善。
- 不直接复制 DDPM 系数、FlowChef 的正号、MCMC 接受公式或旧 pixel-residual 的尺度。
- 不从零再训大 EBT，不增加 Restormer，不恢复 MoE；先 Noise，其他退化后续另定。

## 2. 阅读范围与不能声称完成的工作

本次对下列 12 篇指定版本，连续分段阅读了正文、方法公式、实验/局限及 HTML 中的
实质附录文字/公式和表格，不是只看摘要或搜索片段。参考文献清单不列为逐字审查对象。
数学读取保留 MathML 的原始 LaTeX；未逐页目视比对所有 PDF 图像，也没有运行作者实验。
因此这是细读和静态代码审查，不是完整复现，更不是“保证不存在任何错误”。

| 论文及原始版本 | 关键核查位置 | 对本方案的实际影响 |
| --- | --- | --- |
| [IRED](https://arxiv.org/html/2406.11179v1) | §3、算法1/2、实验及附录 | 阶段匹配、梯度方向监督加对比排序；不等于无需二阶导。其主要验证是推理/规划，不是 FLUX 图像恢复。 |
| [Reduce, Reuse, Recycle](https://arxiv.org/html/2302.11552v6) | §2.3–4.2、附录B–F | 噪声阶段的组合分布与简单相加不同；MCMC 接受是分布采样规则，不是 GT 质量验证。 |
| [CEP](https://arxiv.org/html/2304.12824v2) | §3、式12/13、附录证明/实现 | 终态 energy 到中间 energy 有严格定义；有限候选 softmax 实用目标仍有偏，不能把我们的排序损失称 exact CEP。 |
| [DPS](https://arxiv.org/html/2209.14687v4) | §3、算法1/2、附录A–E | 对 predicted-clean 的观测损失求当前状态梯度；真实退化算子未知时不能直接借它的 data-consistency loss。 |
| [Universal Guidance](https://arxiv.org/html/2302.07121v1) | §3、算法、实验/附录 | forward guidance 穿过生成器；backward guidance 优化预计成品；反复重采样有额外成本。 |
| [FreeDoM](https://arxiv.org/html/2303.09833v1) | §4、time travel 分析、附录 | 干净图能量经预览连接 noisy state；部分任务中段比末段更有用。 |
| [On the Guidance of Flow Matching](https://arxiv.org/html/2502.02150v3) | §3–4、附录推导/实验 | 区分精确分布引导、局部近似、经过模型的目标梯度；其时间方向与本项目相反。 |
| [D-Flow](https://arxiv.org/html/2402.14017v2) | 算法1、§3–4、附录A–D | 通过实际生成 ODE 求导能利用生成器响应，但较贵；不能把其理想化流理论当本 FLUX LoRA 的保证。 |
| [FlowChef](https://arxiv.org/html/2412.00100v1) | §4、算法1、表3、附录证明/局限 | 跳过 vector-field Jacobian 是加速近似；不是完全没有梯度，也不是只有最后几步引导。 |
| [Trust Sampling](https://arxiv.org/html/2411.10932v1) | §3、算法1、§5.3、附录A/B | 限制在不可靠预览上过度优化；其噪声范数阈值是模型相关启发式，不能照搬到 FLUX。 |
| [KDS](https://arxiv.org/html/2507.05604v2) | §3–6、附录A和算法1–5 | 多粒子预计成品的局部共识，不是 learned quality energy；附录式11关闭 t/T<.3 的引导。 |
| [ResFlow-Tuner](https://arxiv.org/html/2603.22027v1) | §3、§4.1、§5.3、后续补充实验 | VLM/FLUX 恢复的直接 prior；方法是扰动、rollout、外部评分、筛选，不是 energy-gradient descent。 |

Classifier Guidance / Composable Diffusion 两篇仅作为上轮背景方法核查，
不冒称本轮又做了全文逐字重读。历史 EBT/恢复论文也不属于本轮 12 篇阅读清单。

### 官方代码覆盖

逐段核查与当前决策相关的函数/调用链，不宣称审查过各仓库每个文件。
只获取源码，没有运行外部程序。Git 部分 blob 拉取超时后，使用同一 commit 的
raw.githubusercontent.com 源文件核对，避免将网络失败写成代码已读。

| 官方仓库 / 固定 commit | 已核查实现 |
| --- | --- |
| `yilundu/ired_code_release` / `3d74b85fab7fcf5e28aaf15e9ed3bf51c1a1d545` | `models.py:789` DiffusionWrapper；`diffusion_lib/denoising_diffusion_pytorch_1d.py:373` opt_step 与 p_losses；train.py 的监督开关、pred_noise、训练配置 |
| `yilundu/reduce_reuse_recycle` / `81177d99bea601e53396a9d35e37f14ab2ee7a98` | `anneal_samplers.py` 的 ULA/MALA/HMC 更新与接受计算 |
| `FlowChef/flowchef` / `9b705fcd91cb3cb8bf9d485ef6badc539201651c` | `src/fluxcombined.py` 的 no_grad/enable_grad、velocity 缓存和引导窗口；`src/scheduling_flow_match_euler_discrete.py` 的 step_final |
| `AI4Science-WestlakeU/flow_guidance` / `b47872e9f72c0b8360c6232fa8ae45f64159bdae` | `image/gflow_img/inverse/inverse_problems.py` 的 guide_fn、wrap_grad_model、solve_ip_grad |
| `thu-ml/CEP-energy-guided-diffusion` / `4cc0622fbe6a032df3e2fd661c5adcdc69cf4b0f` | `images/scripts/classifier_train.py` 的按 batch 归一化对比训练；classifier_sample_color.py 的 cond_fn；另查 RL score loss，未把它误称 CEP 专属 loss |
| `arpitbansal297/Universal-Guided-Diffusion` / `ff82f88072b7f53884c32fc65755fe223e09d291` | `stable-diffusion-guided/ldm/models/diffusion/ddim_with_grad.py` 的模型→pred_x0→可微decode→loss→当前latent梯度及重采样 |
| `vvictoryuki/FreeDoM` / `1394b1dc5807fb01db6a26d2dc42ca05e3d2eaf5` | `Face-GD/functions/denoising.py`；`SD_style/ldm/models/diffusion/ddim.py` 的真实窗口、time travel、梯度归一化 |
| `DPS2022/diffusion-posterior-sampling` / `effbde7325b22ce8dc3e2c06c160c021e743a12d` | `guided_diffusion/condition_methods.py` 和 gaussian_diffusion.py 的 x_prev→pred_xstart→measurement loss→gradient→detach |
| `will-s-h/trust-sampling` / `04fc14452b512426fd09e00c2047b3855f6edb1f` | `diffusion/diffusion.py:387` 的 trust_sample、梯度步、范数阈值及再次模型预测 |

ResFlow-Tuner 原文写将公开代码/模型，本轮未找到它的官方方法实现；不能声称已经核查其代码。
D-Flow、KDS 本轮依据原文/附录算法；没有完成它们独立官方方法代码的核验。
这些缺口已明确保留，不能把“论文可实现”写成“作者代码已复现”。

## 3. 本次确认和修正的重点

### 3.1 三种梯度不是一回事

当前 FLUX 用 t=1 噪声到 t=0 成品，Euler 更新是：

`z_next = z_t + (t_next - t) * v(z_t, t)`。

单步预计成品：`z0_hat = z_t - t * v(z_t,t)`。
其输入导数为 `I - t * J_v`，不是照搬论文另一种约定下的加号。

需要区分：

1. **预览完整梯度**：energy→连续blend→VAE→预计clean latent→FLUX→当前latent。
   是所选预览目标的正确导数，不是实际剩余轨迹终点的精确导数。
2. **预览近似梯度**：把 v 当常量，省略 J_v；VAE/energy 仍要反传。
3. **实际终点梯度**：经过真实剩余若干生成步，再评价终图；计算和存储更贵。

FlowChef 的“Jacobian 局部变化慢”本身不能推出 J_v 很小或 I-tJ_v 接近 I。
其官方 FLUX 路径先计算 velocity，再更新 latent，并复用原 velocity。
本项目第一版建议以完整预览导数为参照，修改 latent 后重新计算 FLUX velocity；
之后才将 skip-Jacobian / 旧 velocity 作为明确的近似比较，计入各自成本。

### 3.2 阶段适配有解法，但不等于原 energy 直接可用

IRED 的能量梯度训练实际使用 `create_graph=True`，有混合二阶导数。
避免沿整条长链反传，并不等于没有二阶导。其 train.py 默认对比监督开关为 false，
不能只抄默认命令便说复现了完整 IRED。

CEP 的阶段能量是负对数的条件指数期望，不是简单的 clean-energy 平均值，
也不是 clean-energy(平均成品)。实用有限候选 softmax 目标与精确理想目标仍有区别。

我们应训练实际 FLUX 预览分布上的条件质量目标，不复用只训练25步的终图 energy，
也不把“给GT加高斯噪声”当作已经覆盖了 FLUX LoRA 真实轨迹。

### 3.3 更低 energy 不是可靠性证书

IRED 报告过不合法 Sudoku 解比 GT 能量更低的失败案例。
D-Flow 附录 denoising 的 LPIPS 可以更好而 PSNR 不如一个对照；DPS 也明确讨论感知/失真权衡。
这些都不支持“同一energy既修正又评分，所以自然保证恢复更好”。

同一时间点比较候选、预测实际质量变化、真实输出双指标，是三个必须分别核验的环节。
跨不同 t 直接取最小 energy 做停止尤其没有天然依据。

### 3.4 不能把“最后几步”当文献已证实的最佳窗口

- FlowChef 公开 FLUX 分支的条件是 `i < max_steps`，是前面的窗口。
- FreeDoM 的 SD_style 代码显式控制中段窗口，并在那里重复采样。
- KDS 附录A.1.3/式11在 `t/T < .3` 时关闭 steering，原因是后段敏感。
- RRR 的 Image Tapestry 实验也没有在最后16个 ancestral steps 做 LA。

这些不是证明我们末段不能成功，而是说明窗口需用本模型诊断，不能默认2–4步必够。

### 3.5 TTS 效果和成本不能从别的任务直接移植

ResFlow-Tuner §5.3 默认配置报告15倍基线计算预算；它优化的是外部美学/语义reward组合，
并承认 PSNR/SSIM 较低。不能用它保证我们低成本同时涨 PSNR、降 LPIPS。
其默认 sampler 是 SDE 的 FlowDPMSolverMultistepScheduler，不是本仓库的纯 Euler 路径。

KDS 在恢复任务上确有双指标改善，证明这类改善是可能的；但通常使用多粒子，
不代表单条 FLUX 轨迹最后几步能直接获得同等收益。论文提升不能当本项目预测值。

### 3.6 官方代码也不能免检照搬

RRR 固定 commit 的 `AnnealedMALASampler.sample_step` 用 `sqrt(2*ss)` 生成噪声，
但计算 proposal log-density 时分母写为 `2*ss**2`。
按这个函数自身的采样定义，协方差为 `2*ss*I`，对应二次项分母应为 `4*ss`。
这是该函数的静态一致性问题，不是对论文全部结果的否定，也未在本轮运行重现实验。
我们当前不采用 MALA；若未来采用，应独立验证公式，不能因“官方实现”就直接拷入。

## 4. 本项目代码核查

检查的是当前实际文件，而不是仅看报告或根据历史对话猜测。

| 文件/入口 | 确认事实及下一版约束 |
| --- | --- |
| `smoke/qualify_foundir_flux_energy_v1.py:restore` | 整段包在 inference_mode；不能只加 requires_grad。需单独可微入口，前缀缓存离开 inference_mode 后建立普通可求导tensor。 |
| `third_party/flux2/src/flux2/sampling.py:get_schedule` | schedule 按token数/步数做shift，不能使用 step_index/50 作为实际 t。 |
| 同文件 `denoise_cfg` | 每步拼接reference tokens；CFG双batch后返回一半。即使guidance=1也不能未经一致性检查擅自改计算路径。 |
| `lowlight_restoration/flow.py` | reference flow 与 degraded-anchor flow 都存在；Noise 应用原参考图/纯噪声路径，不能误接 Rain 的 anchored sampler。 |
| `third_party/flux2/src/flux2/autoencoder.py:decode` | 本函数没有 no_grad 装饰，包含inv-normalize和2×2 channel rearrange；需保留完整解码定义。 |
| `lowlight_restoration/scalar_residual.py` 及 deployed_proposal | raw FLUX先uint8取整，再固定alpha=.30混合LQ，再取整。连续代理不等于真实部署函数。 |
| `smoke/flux_energy_qualification_contract_v1.py:descriptor` | 读每条记录的qwen_levels_json，5个退化等级1–5映射到0–1；不是每图在线VLM token，也不是固定Noise标签。 |
| `lowlight_restoration/novel_ppt_energy.py` | 旧模型在归一化RGB residual u上工作、无diffusion-time输入、固定1/2/3步，排序/方向/轨迹loss都有；不能把它直接称latent-tail energy。 |

额外注意：旧loss按 `residual_scale=.02` 缩放MSE，而LPIPS使用另一量纲。
换成latent/preview目标必须重新明确尺度，不能把旧loss权重和步长直接搬来。
这个静态事实本身不证明它就是旧实验失败的唯一原因。

源码SHA-256（本轮只读，便于之后确认是否换过实现）：

```text
sampling.py f1a1e89651d450c05ab87840ed65011a2d25b83d811c61853ec6ed38e938d640
autoencoder.py bb578f78d097a88f6e7e8aa01032c10f55659729b85203ae2d5852a7889ac96a
flow.py 000e6549a67015d61e7ff0108fd8024afc7004124920fb354a74a616529cae10
novel_ppt_energy.py 66d96b9bde3f78b9e2bdc79049330baa4f1d698459d8f231a96de14ef94b5ea6
scalar_residual.py a61ae2ae382a25848e189c962910780f79beaa9cc5508ee99df56c0434410801
qualify_foundir_flux_energy_v1.py 2528ca4afe7de5f1517b23accbadc41d55f7a680c4aa03c6e13be3aac162301b
flux_energy_qualification_contract_v1.py ea84b0ec679ccf274ed4366ebfacb4013499aad715acbe9461665583b9d5bb70
```

## 5. 重新建议的最小方案

### A. 先确认接线和可改进空间，不训练新恢复大模型

1. 冻结 Noise step400、VAE、reference/text、seed、50步及alpha=.30。
   建独立可微末段入口；零引导必须复现原量化结果。
2. 做输入梯度检查，区分完整预览梯度、skip近似和真实短尾段梯度。
   先验证符号、尺度、有限值、有限差分方向；没有通过前不以模型容量解释失败。
3. **只在训练来源上**用GT的MSE/LPIPS临时当已知目标，诊断“预览改善能否传到真实终图”。
   对有限的几个末段窗口/步长检查实际完成生成并量化后的结果。
   这是非部署oracle诊断，不是模型成绩；不能把GT放进最终推理。
4. 若单步预览方向与真实终图明显不一致，再用短段真实rollout替代预览验证原因。
   一次小范围oracle失败不证明所有FLUX引导都不可能；oracle成功也不证明energy一定能学会。

### B. 训练真正适配这个过程的条件 energy

输入：原始LQ、FLUX当前预计成品、已有逐图VLM等级、实际时间t。
输出：当前候选的恢复质量代价；不是另一个直接输出修正图片的Restormer。

训练样本来自合法training sources的真实末段预览，以及局部latent扰动/引导后的预览，
同时包含实际终图；优先学习接近可部署质量的细微差别，不让明显错误专家负样本主导训练。
不用32张/25步当科学规模；工程资格通过后，依据缓存吞吐冻结覆盖足够来源的正式预算。

借鉴 IRED 的“排序+方向”原则，但不照搬其高斯score目标：

- 利用训练GT计算候选MSE/LPIPS代价和相对优劣；各量纲/权重仅在training/calibration冻结。
- 对energy的图像输入梯度提供局部质量方向监督，并实测它引起的真实rollout变化。
- 可先缓存预览及GT目标梯度，方向训练的二阶导限制在小energy网络；
  不默认每个训练step对50步FLUX反传。是否需要短轨迹监督由diagnostic决定。
- 保留source-disjoint训练/校准/评价；校准窗口或步长不得反复看最终1859条gate。

以上是待实现的训练原则，网络宽度、学习率、步数/窗口尚未冻结，不冒充已经launch-ready。

### C. 推理和判定

`FLUX生成到指定阶段 → 预计成品 → 条件energy → 梯度改当前latent → 重新算FLUX方向 → 继续生成`。

推理不读GT，FLUX权重不更新。先采用固定预算，避免未校准的跨t能量提前停止。
PPT所说的自验证要另证实：在同一最终评分条件下，energy能否拒绝真实退步，
不能仅因为使用同一energy，就宣布自验证成立。

最终保持原uint8处理和匹配协议：1859条、34 source groups、512 crop。
energy独立目标仍为PSNR增加至少.30 dB，LPIPS相对降低至少
`0.09928107050377874`。旧Restormer residual提升不得抵扣。

比较原50步baseline和相同总计算预算的无energy控制。统计全部FLUX前向/输入反向、
VAE、energy、额外rollout/筛选及fallback成本；不只数采样步。
若为了小幅更新采用FP32 latent累积，也要设置相同精度的无energy控制，
避免把精度变化的收益算成energy贡献。
按source group报告配对变化和不确定性，不能把同源记录当独立样本夸大证据。

## 6. 交付边界

技术接入具备现成依据；有效恢复目标及泛化梯度仍待证明。
“VLM+FLUX+energy/TTS”不能称首次，FlowChef、FreeDoM、ResFlow-Tuner等必须引用。
目前没有证据可以承诺+.30dB/9.93%一定通过，也不能凭旧25步零收益判整条方向无效。

本轮仅完成审查/文档。尚未实现新采样器、未运行oracle、未启动新训练。
下一次若用户授权实施：先冻结上述diagnostic的split、窗口清单与预算，完成静态/单元检查，
再由单节点同型号8×H100/H200进行工程资格；普通小时与allocated GPU-hours以实测报价。
