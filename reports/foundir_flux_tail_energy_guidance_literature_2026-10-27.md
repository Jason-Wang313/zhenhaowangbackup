# FLUX末段energy引导：中间态适配文献与代码核查

日期：2026-10-27。范围：公开原文方法/关键假设、作者页面、三个官方代码仓库、当前项目接入点。
不是所有论文逐字通读或穷尽检索；不将摘要发现等同于完整复现。未运行任何下载的代码、训练或评估。

## 结论与用户决定

用户停止当前最终输出pixel descent方向，优先研究FLUX末段采样引导。
先完成这次文献审查，不自动开GPU任务；不复用未充分训练的25步energy作为可靠引导器。
过去“先证明pixel energy可靠且纹理被破坏，才能探索采样引导”的前置条件已由用户取消。
旧25步工程结果保留：科学收益未建立，不能说已证明所有pixel energy方法不可能有效。

用户记忆基本正确：中间态分布与最终图评分器不匹配是已研究的问题。
IRED、CEP直接研究阶段相关energy的学习；DPS、Universal Guidance、FreeDoM通过
predicted-clean估计连接最终图目标与中间态；Flow Matching有自己的guidance推导和代码。
不能把这些已知技术问题当成从零创新，也不能把“有标准处理方法”说成“本项目指标必然达标”。

## 核查工作与证据

| 原始工作 | 本次核查重点 | 可借鉴机制 | 不能直接推出的结论 |
| --- | --- | --- | --- |
| [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) | §4 noisy classifier训练及引导 | 在同一噪声分布训练带t的classifier，梯度引导生成 | 干净图classifier无需适配即可用于所有t；FLUX等于DDPM |
| [Compositional Visual Generation with Composable Diffusion Models](https://arxiv.org/abs/2206.01714) | 背景/组合score框架 | 组合阶段相关生成信息 | 任意终图energy可无条件直接加到flow |
| [Reduce, Reuse, Recycle](https://arxiv.org/abs/2302.11552), Yilun Du等 | §2.3–2.4、§4.1–4.2、算法1 | 不同噪声阶段energy/score，annealed MCMC；修复错误的组合采样 | 失败一定只在sampler，不在我们的quality energy；MCMC接受就等于GT质量提高 |
| [Learning Iterative Reasoning through Energy Diffusion / IRED](https://arxiv.org/abs/2406.11179), Yilun Du等 | §3.1–3.4、算法1/2及官方训练代码 | 一系列噪声阶段energy，梯度去噪监督+对比energy监督 | 已验证FLUX图像恢复；只加t embedding无需匹配训练；不用任何二阶导数 |
| [Contrastive Energy Prediction / CEP](https://arxiv.org/abs/2304.12824), Lu等 | §3.1–3.2、式9/12、定理3.2 | 从终态energy构建中间energy并通过对比预测学习guidance | 是Yilun Du论文；有限数据下保证精确；照搬即可适用于任意flow |
| [Diffusion Posterior Sampling / DPS](https://arxiv.org/abs/2209.14687) | clean estimate及posterior近似 | 在预测干净图上计算观测一致性，再向当前state求导 | 真实复合退化的算子/噪声参数已知；把LQ直接当clean target就可恢复 |
| [Universal Guidance for Diffusion Models](https://arxiv.org/abs/2302.07121) | §3.1–3.3、式6、算法1 | 干净图模型评价predicted-clean；forward/backward guidance；self-recurrence | 一步clean estimate是准确最终输出；增强引导总是安全 |
| [FreeDoM](https://arxiv.org/abs/2303.09833) | §4.1–4.2、式7–10 | time-independent clean energy经clean estimate指导noisy state；time travel | 不需要可靠energy；只改最后几步必然最好 |
| [On the Guidance of Flow Matching](https://arxiv.org/abs/2502.02150) | §3.3–3.5、式5–7、inverse-problem结果与代码 | flow专属guidance、endpoint estimate、Jacobian/协方差近似或训练guidance | diffusion梯度可无系数/无假设直接加到任意velocity |
| [D-Flow](https://arxiv.org/abs/2402.14017) | 算法1、§3及计算代价讨论 | 通过生成ODE求导，优化源状态，真正利用生成器的响应 | 完整rollout反传便宜；我们只改末段就继承其全部理论保证 |
| [FlowChef](https://arxiv.org/abs/2412.00100) | §3–4、Jacobian近似、官方FLUX代码 | predicted-clean目标梯度、跳过vector-field反传来省算力 | 完全没有梯度；本项目FLUX.2/LoRA下近似必然准；其代码只引导最后几步 |
| [Constrained Diffusion with Trust Sampling](https://arxiv.org/abs/2411.10932) | §3.1–3.2 | 按proxy可信度/噪声阶段限制每次优化，不让梯度把state推离有效区域 | 自动保证每图PSNR/LPIPS不下降 |
| [Kernel Density Steering / KDS](https://arxiv.org/abs/2507.05604) | 方法框架、clean particle预测及KDE引导 | 在predicted-clean ensemble上mode seeking，改善恢复的失真/感知权衡 | 学过VLM条件quality energy；单粒子直接继承其效果 |
| [ResFlow-Tuner](https://arxiv.org/abs/2603.22027) | §3.1–3.2、MSPDE和verifier ensemble | FLUX恢复、VLM语义条件、中间态扰动/短rollout/预览评分/选择 | 它等同于我们的energy-gradient更新；简单FLUX+VLM+TTS可以称首次 |

检索使用arXiv API、上述原文HTML、[Yilun Du作者主页](https://yilundu.github.io/)和项目页。
GitHub API遇到限流，改用官方仓库只读浅克隆获取源码；未执行外部代码。

## 三种标准解法

### A. 让energy在训练时就认识各个阶段

IRED用E(input, candidate, t)，学习不同噪声阶段的energy landscape。
训练同时约束energy梯度的去噪方向，以及正确/错误候选的energy排序；同一组对比候选
使用相同噪声以减小监督方差。其主要验证是推理/规划任务，不是图像恢复。
它不需要沿整条长优化轨迹反传，但对energy梯度做监督仍涉及混合二阶导数。

CEP更直接形式化“终图目标如何变成中间态引导”。在其扩散设置下：

\[
E_t(z_t)=-\log\mathbb E_{q(z_0\mid z_t)}[\exp(-\beta E(z_0))].
\]

它一般不等于E(z_t)，也不等于E(预期z0)，甚至不等于简单的期望energy。
CEP用终图energy的soft label训练阶段相关对比预测，精确性结论依赖无限容量/样本等条件。
我们的条件restoration flow还需对应其概率路径/目标；不能把CEP的扩散定理冒充本方法保证。
IRED学到的完整conditional score也不能不加区分地叠加到已条件化FLUX，避免重复计入先验。

### B. 不给energy看半成品，而给它看“预计成品”

Universal Guidance §3.1明确指出：clean classifier直接读noisy state通常失败。
它与FreeDoM/DPS的关键是先用生成器估计干净结果，再计算loss/energy并把梯度传回当前state。
因此可以复用成熟clean-image目标，不一定另训覆盖所有t的大模型。
但“training-free guidance”是相对生成器/已有目标，不表示我们自定义quality energy无需训练。
预测干净图仍可能偏离真实最终图，所以需要评估这种代理的可信度。

### C. 真的向前生成一小段，再评价更可信的结果

D-Flow是对生成过程求导的代表；ResFlow-Tuner用多步partial denoising/lookahead评估候选。
对末段可缓存前缀，仅对短尾段重放/求导，避免每次从头跑50步。
这是合理工程候选，非本轮已实现方法。多次rollout、VAE和反传都必须计入预算。
ResFlow-Tuner的具体TTS是扰动—rollout—外部reward ensemble排序/选择，不是直接energy gradient。
因此也应纠正把其概括成“直接梯度引导”的过宽描述。

## 公开实现实际可借鉴的内容

- IRED官方仓库 `yilundu/ired_code_release`，检查commit
  `3d74b85fab7fcf5e28aaf15e9ed3bf51c1a1d545`。
  `diffusion_lib/denoising_diffusion_pytorch_1d.py`中p_losses在t上q_sample，
  以噪声/状态目标监督，另计算真实/负样本energy的交叉熵；models.py以autograd.grad
  产生候选梯度。可借鉴阶段匹配及方向+排序监督，不是可直接装进FLUX的恢复checkpoint。
- FlowChef官方仓库 `FlowChef/flowchef`，检查commit
  `9b705fcd91cb3cb8bf9d485ef6badc539201651c`。
  [src/fluxcombined.py](https://github.com/FlowChef/flowchef/blob/9b705fcd91cb3cb8bf9d485ef6badc539201651c/src/fluxcombined.py#L899)
  先计算transformer noise_pred，再在enable_grad块内对opt_latents用step_final预览、
  loss、autograd.grad更新。transformer预测在外层no_grad中，故确实省略其输入Jacobian。
  该段用已知mask/reference约束，不能把reference target换成真实Noise的未知GT。
  控制条件i<max_steps也不是“只在最后几步”；末段窗口是本项目待验证选择。
- flow_guidance官方仓库 `AI4Science-WestlakeU/flow_guidance`，检查commit
  `b47872e9f72c0b8360c6232fa8ae45f64159bdae`。
  `image/gflow_img/inverse/inverse_problems.py`明确区分
  `nabla_x1_J_x1`（预测干净图梯度）和`nabla_xt_J_x1`（经过flow模型的当前state梯度），
  同时有PiGDM/其他近似。可借鉴准确梯度与省算版本的匹配比较，不能互相冒名。

## 本仓库的具体含义与推荐

本仓库lowlight_restoration/flow.py定义z_t=(1-t)z_clean+t*noise，target_velocity=noise-clean。
sampling.py由t=1降到0，且使用shifted schedule。相应单步clean latent估计为：

\[
\hat z_0=z_t-t\,v_\theta(z_t,t;\mathrm{reference},\mathrm{text}).
\]

这是endpoint估计，不等于完整余下轨迹的精确终点。不要照抄noise→data记号论文中的加号，
也不要用step_index/50代替真实t。原restore在inference_mode中，新求导入口须独立实现，
不能只加requires_grad便声称梯度穿过了FLUX。冻结模型权重不等于禁止对输入求导。

建议第一版设计为：

> 冻结FLUX生成到末段 → 预测干净latent并解码预览 → 条件quality energy评估 →
> 梯度修正当前latent → FLUX继续生成最终图。

这让FLUX参与后续生成，不是最终图片完成后由小网络独自改像素。
新energy的训练样本应覆盖真实末段预览、以及小幅修改state后产生的候选，保留训练GT监督、
LQ/VLM条件；不直接用旧25步权重、不仅靠随意给GT加高斯噪声声称分布已对齐。
实际训练目标、规模、时间点和窗口在本轮尚未冻结；这是一条有文献依据的设计建议，非实验成功。

先以保留必要输入Jacobian的局部梯度作正确性参照，再决定是否采用FlowChef式近似省算。
省略Jacobian不保证等价：即使J局部变化慢，也不意味着其范数小；末段t趋零才可能减小tJ项，
仍需本模型实测。不同t的energy绝对值未必可直接比较，不能顺手当作已校准停止指标。

Noise部署baseline仍是step400/50步/固定alpha=.30，不换弱raw FLUX。
energy所评价的预览应对齐最终算术blend，最终评价保持真实uint8量化。
优化可用明确标注的连续代理，但不能对hard rounding悄悄当作可微，也不能用float评分替换原gate。
GT只可用于training和允许的独立评估，不得进入推理引导；真实noise level/算子未知时
不得把DPS的已知观测模型当作我们已有条件。

## 最低比较与风险边界

1. 新入口guidance=0必须复现原冻结FLUX流程；相同输入/seed/crop/alpha/量化。
2. 对比原50步baseline，同时匹配总计算预算的无energy方案；计算包括模型前向、反向、VAE、
   energy、额外候选/rollout。不能只对齐FLUX前向步数就宣称公平。
3. energy学会排序、梯度方向有用、修正后PSNR/LPIPS提升是不同事实，分别验证。
4. 仅末段引导是降低错位/成本的合理假设，非保证。FreeDoM的time-travel分析甚至发现
   某些生成任务的中间语义阶段比最后细化阶段更有用，不能机械外推“越晚越好”。
5. 原始输入保真项可能退化成identity，纯感知目标可能幻觉/降低PSNR；把更新移入FLUX不消除该矛盾。

保留energy自身+.30dB / LPIPS相对下降9.928107050377874%的目标，不许把先前基线收益抵扣。
这一技术障碍有成熟解法；我们的未解问题是restoration目标和guidance是否在独立数据上有效。
普通VLM+FLUX+TTS或energy guidance不构成首次，FlowChef/ResFlow-Tuner等是必须面对的prior。
本轮没有新GPU作业，GPU-hours=0；未来预算待具体工程设计与实测，不凭现有论文编造耗时/提升。
