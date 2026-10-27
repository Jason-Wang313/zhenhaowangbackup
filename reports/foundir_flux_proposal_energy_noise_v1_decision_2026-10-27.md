# FLUX proposal + energy：用户目标更正与启动前约定

日期：2026-10-27。状态：协议及增益检查已实现，GPU流水线尚未就绪，未启动新训练。

## 用户决定

主实验改为冻结FLUX输出，再由VLM条件energy判断候选并通过同一energy的梯度修图。
不得在主链路加入额外residual Restormer、可训练直接refiner，或把它们已有的收益归给energy。
用户并非禁止Restormer作为其他独立实验的基线；禁止的是用基线已有收益抵扣energy目标。

本次先保持Noise任务，不自动扩成五类或MoE。旧Restormer-based energy任务及其初始化
修复草稿均不是当前FLUX主实验。保留旧产物，不修改活跃任务绑定源码，不擅自杀进程；
切换前须在实际GPU节点确认旧任务结束或由用户停止，避免争卡。

## 增益合同

- PSNR：相对同协议冻结FLUX基线提高至少0.30 dB。
- LPIPS：相对该FLUX基线降低至少0.005 / 0.050362067760033824
  = 0.09928107050377874，约9.93%。
- 必须同时满足；不是固定降低0.005，不是PSNR数值提高0.3%，也不继承旧绝对门槛。
- 相同记录、来源组、输入像素哈希、裁块、量化、LPIPS定义和模型/推理身份。
- energy不变输出绝不能因为强基线已通过旧门槛而被判成功。
- energy选择1/2/3步相对固定3步的收益独立报告；画质过线不自动证明自验证或novelty。

实现：lowlight_restoration/flux_energy_gate.py。
协议：configs/foundir_flux_proposal_energy_noise_v1_plan.json。
配对检查拒绝重复/缺失样本、不同裁块/输入哈希、不同协议以及非有限指标。

## 基线必须是当前可用FLUX，而不是更弱的原始输出

绑定Noise step400 LoRA SHA：
1b4ed05ad5ad403a3d7fcc633eca97cdb69d00454e357da0c3d3131ac2c3ff1a。

继承configs/foundir_noise_flux2_scalar_residual_full_development_v1.json，
SHA 5e7a356ea89d21f7b60b4b736f851ef26a0216d8555d73846f527b958b5b7dd6。
保留50步、固定种子/命名空间、reference time 10和alpha=.30的原量化顺序。
这里scalar residual只是LQ + .30*(FLUX-LQ)算术，不是额外的Restormer网络；
不得移除它，用更弱的alpha=1结果制造energy增益。

既有完整1859条/34来源、512 crop协议结果：

|输出|PSNR|LPIPS|
|---|---:|---:|
|输入|40.4741102620828|0.10939864525867043|
|原始FLUX alpha=1|35.31568790674834|0.05249364261383681|
|部署FLUX alpha=.30，主基线|40.150553492200345|0.06888036001205641|

证据：checkpoints/foundir_noise_flux2_scalar_residual_full_development_v1/result.json。
如果严格重现同一协议基线，对应目标为PSNR >=40.45055349220034、
LPIPS <=0.062041844133373776。这些是换算门槛，不是新实验结果。
必须实际核对匹配FLUX输出身份；不得把旧256条/256 crop的43.x Restormer值搬来比较。

## 第一性原理与信心边界

FLUX是rectified-flow DiT；学习的时间相关速度场不能自动当成当前候选的标量
restoration-quality energy梯度。本方案是在冻结FLUX之后新学energy，不继承所谓
EBM自验证或质量改善保证。关于score/diffusion/energy的联系不能替代实验。

尝试的实际依据是：Noise FLUX相对输入PSNR低.32355676988244864 dB，
但LPIPS显著更好，输入可能保留被生成过程损伤的细节。energy若能利用原始观察、
保留降噪收益并修复失真，就可能同时提升。直接返回输入可提高PSNR却严重损害LPIPS，
所以不能称双门槛容易通过。相比强Restormer有更多可修错误是合理假设，不是已证明结论；
协议不同，也不能用两组原始均值断言更容易。

没有基于证据的数值成功率。当前判断为值得检验、谨慎乐观，不是高把握或必过。
已有fixed32 energy拟合基于Restormer，不是FLUX成功证据。

## 下一实施步骤与成本

先做真实训练来源的FLUX缓存/加载/显存吞吐资格，不直接开全量长训。
现有完整开发目录主要保存逐图指标与16张triptych，不是完整训练proposal缓存；
不能从截图或GT重建假FLUX输出。

需要核对冻结Noise prompt上下文，避免重演GPU端FP8 kernels联网故障。
冻结FLUX候选可预生成一次，energy训练时不每步重跑50步FLUX；缓存需来源、裁块、
模型、prompt、采样种子及像素哈希绑定，训练与开发严格分离。
完整candidate bank的实际wrong-expert输出及三步反传仍需接通和资格验证，
不得把简化幅度选择器冒充PPT方法。初始化修复只保留为待验证改动，不声称已过GPU。

历史8xH100全开发评估1859条耗时1611.2560072094202秒。
按条数线性外推54585条约13.1418普通小时/105.13 allocated GPU-hours；
旧评估还含加载、指标等开销，该数字不是新缓存实测，也不适用于H200速度承诺。
因此全量导出/训练预算必须先由小规模实测给出，再决定，不能沿用旧Restormer约22秒/步。
本次没有使用GPU，未读取数据集图片。旧任务是否停止仍需节点端确认。

## 未完成项

- FLUX缓存导出和loader、冻结prompt核验、完整候选生成、训练/评估入口；
- 初始化修复的完整CPU/DDP/GPU资格；
- 任何FLUX + energy的训练、开发结果与科学结论。

不可将本协议报告当成launch-ready报告。

## 本轮验证

6项CPU单元测试通过：强基线原样输出拒绝、相对LPIPS降幅、双指标同时通过、
协议不匹配拒绝、重复/缺失/不同输入身份拒绝、非有限指标拒绝。
专用HDD3 Python运行，未使用GPU或数据集图片；源码AST/JSON检查和staged凭证扫描通过。
这些只验证指标归属和配对检查，不证明FLUX生成、energy训练或画质改善。
