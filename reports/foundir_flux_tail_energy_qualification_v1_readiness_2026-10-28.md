# 下一步：让学到的energy接替GT，指导冻结FLUX末段

## 当前决定

用户在24来源GT确认通过后明确授权执行下一步。本轮实现独立的25-step工程资格：
训练一个条件质量energy，通过它对当前候选的梯度修改FLUX latent，再实际运行
剩余两步生成。不是再做GT oracle，不是最终RGB直接下降，也不是另一个Restormer。
FLUX/LoRA、VAE、LPIPS全部冻结。旧V1/V2/V3/24图确认的绑定代码与结果不修改。

GT已证明有限搜索有改善空间，但不能证明一个不知道答案的模型能找到它。
这一轮先验收训练、二阶梯度、数据隔离、checkpoint及真实无GT推理是否接对。
25步成绩无论好坏都不是正式科学结论；不自动formal，不从资格权重续训正式实验。

## 为什么这样学

energy需要同时学会“哪个候选好”和“朝哪个方向改”，仅学分数排序不能保证梯度正确。
因此共享一个多尺度网络，输出两个代价：log MSE、log LPIPS。
这两个输出分别描述像素保真和感知质量，避免用一个固定加权总分掩盖两者冲突。
方向搜索沿用已验证的五种梯度混合及五种步幅，允许使用超过+.30dB的PSNR余量。

网络输入为原LQ、原部署FLUX结果、当前候选、候选与前两者的差分、
该图已有Qwen预测的五类退化等级，以及实际shifted时间t。不是GT退化标签，
不在线重跑Qwen、不下载FP8 kernel。网络输出质量代价，不输出RGB或residual。
网络有1,010,274个可训练参数；承担生成的FLUX.2约4B参数仍保留且冻结。
小energy与“小型从零恢复模型”不是同一件事，但容量是否足够仍须实验。

本网络复用现有多尺度energy组件，不沿用旧权重，不含旧Restormer residual basis，
不含旧u坐标的固定二次势阱。非零head初始化提供非零候选输入梯度，已做单元测试。
不能仅由复用组件/两头设计声称论文创新成立。

训练监督：

- 分数：真实log MSE和log LPIPS的Smooth-L1。
- 方向：同一候选上真实两种质量函数的图像梯度，监督预测energy的图像梯度；
  各按真实梯度RMS归一化。真实梯度为零时保留绝对零拟合，不把误差放大1e10倍。
- 同来源候选对：按真实差距给排序margin；log代价差小于.01视为平局。
- 总loss为分数 + 方向 + .25×候选对排序。
- 方向loss有混合二阶导，但只穿过小energy网络；每个训练step不反传50步FLUX。
  正式推理的输入梯度会穿过真实FLUX两步与VAE，这部分算力必须如实计入。

目标和方法依据延续已完成的IRED/CEP/flow-guidance原文代码审查：
`reports/foundir_flux_tail_energy_guidance_deep_audit_2026-10-28.md`。
这是项目实现，不是复现了那些论文，也不是首次energy guidance的证据。

## 新数据与缓存

从已哈希绑定的54,585条Noise training记录/1,755个来源中选，排除旧32来源，
同时按旧GT/LQ哈希排除别名来源。按固定hash顺序预留128来源作校准、
128来源作开发检查，其余1,467来源作为energy训练池。
检查训练/校准/检查之间的source、GT/LQ内容身份，以及全部protected来源/GT排除。

本次工程资格固定24个新训练来源、8个新检查来源，互不重叠。
128来源校准池本轮不读取像素。检查图仍来自原始training pool，FLUX可能见过它们；
它们仅对新energy训练来源分离，不能称整个系统从未见过的正式测试集。
最终1,859条开发集和全部official/sealed/internal/parent-test像素仍禁止访问。

缓存阶段：

- 每张先按原Noise step400、50步、CFG1、reference time10、512裁图/原seed、
  alpha=.30及两次uint8量化复现基线；新可微tail必须逐像素一致。
- 训练来源缓存：FP32无引导、原部署结果、两种幅度的正负随机latent扰动，
  以及冻结GT策略最多16轮中每轮接受的候选和局部LPIPS最差候选。
  没有合格候选则按原规则停止，不伪造继续改进；不引入明显错误专家主导训练。
- 保存连续blend及对应实际双量化图像。每种图像分别用真实质量函数计算标签
  和图像梯度；不把round的导数当真，也不拿连续图的指标冒充量化结果。
- 每个训练来源最多38个状态×两种表示。缓存保存float32连续图和uint8部署图，
  不先压成JPEG，不改采样crop，也不缓存一套错误的低精度梯度。
- 8个检查来源只生成基线，不生成GT轨迹用于训练或挑参数。

使用实际生成轨迹比只给GT加噪声更接近部署，但学习后的轨迹仍可能偏离GT缓存；
这是尚待检查的分布偏移，不保证已有训练覆盖所有推理状态。

## 训练与推理协议

新模型、AdamW lr=1e-4、5-step warmup、weight decay=1e-4、gradient clip=1，
训练25个optimizer steps。global batch=32个同来源候选对，即64个候选视图。
四卡每rank累积8对，八卡累积4对；全局样本槽/随机选择与卡数无关。
训练完整512图，无额外crop或增强，以保持图像梯度标签含义一致。

训练后只评估固定step25，不挑最优checkpoint。推理函数没有GT参数：
从候选连续图的两个energy梯度，经真实两步FLUX/VAE传回index48 latent；
分别归一化，再试LPIPS权重0/.25/.5/.75/1、步幅.0005/.001/.002/.005/.01。
实际双量化候选由同一energy评分，预测log MSE转成预测PSNR、预测log LPIPS转成预测LPIPS，
再应用原psnr_budget排序与接受规则。这里是**预测**安全，不是实际无损保证。
真实GT在全部修正/接受/停止结束之后才读入指标计算，不能进入选择器。

无合格预测候选或可用梯度接近零时保留当前状态；不因预测达标而提前停止。
报告1/2/4/8/16轮，固定16轮为主终点，同时保留原部署及FP32无引导对照。
不接受更新时使用FP32无引导结果，不借GT换回较好的历史图。
终态保存实际量化结果tensor供以后审核，但工程通过不等于已经做了视觉审查。

真正目标不变：相对匹配FLUX基线，energy独立增加至少+.30dB，LPIPS下降至少
9.928107050377874%。不扣除/抵用旧Restormer成绩，不用8图工程均值冒充正式gate。
后续科学实验还需要足够来源、训练曝光、来源分离的校准/评价和等总计算量对照。
根据本轮缓存/训练/推理吞吐再冻结正式规模；不能从25步零收益直接终判整条路线失败，
也不能因GT通过便盲目启动100k步。当前入口不提供formal模式。

## 硬件、预计成本与命令

单节点同型号4/8×H100/H200，每卡至少75GiB。禁止混卡、跨节点和重复占用。
有空闲硬件检查、节点锁、fresh ID、源码/权重/数据绑定、总超时及阶段终态核验。
全部阶段顺序运行：预检→缓存→25步训练→无GT推理检查→结果核验。
旧Qwen/旧训练和旧qualification权重不参与新初始化。

四卡H100暂估.5–1.0普通小时、2–4使用GPU-hours；这是实现前测量基础上的粗估，
尚无本任务GPU实测。若独占八卡整机仅用四卡，另计4–8整机GPU-hours。
硬cap为四卡1.5小时或八卡.75小时，均6使用GPU-hours，另最多30秒退出清理。
独占八卡仅用四卡时对应12整机GPU-hours上限；不承诺H200或八卡精确翻倍加速。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/start_foundir_flux_tail_energy_qualification_v1_tmux.sh qualification_4gpu_v1 4
```

八卡另用新ID和显式0–7卡、参数8；二选一，不重复运行两套。
日志：`logs/foundir_flux_tail_energy_qualification_v1_<id>.log`。
输出：`checkpoints/foundir_flux_tail_energy_qualification_v1/<id>/`。
需看到`TAIL_ENERGY_RESULT_VERIFIED`和`TAIL_ENERGY_LAUNCH_END status=0`才算正常完成。
tmux创建不等于缓存/训练已开始。正式训练不得以本25步checkpoint初始化。

## 已完成验证与尚未完成

46项CPU回归通过，包括非零输入梯度、score正确而gradient错误时仍受罚、
完整16轮预测选择、零梯度停止、真实safetensor读取、来源/GT隔离及4/8分片口径。
两进程Gloo的真实二阶反传/梯度累积通过，最终全参数逐元素相同。
Python/shell静态检查、四卡/八卡完整权重及数据元数据预检通过；未解码原图。
参数量：1,010,274。配置SHA256：
`22da9b3fb51f4705cc917b3d3c8bd1e1fea8bdd3193fff205e198c57ef462c71`。
当前Codex无GPU，GPU-hours=0；尚未进行真实GPU资格，不宣称已经训练或改善指标。
