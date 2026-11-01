# 五类 MoE：保护已有能力的训练修订 — 2026-11-01

## 当前决定与边界

用户已授权执行五类训练修订。当前交付是独立的25步工程资格入口，
从成功的VLM-token step2000模型权重开始，fresh Adam，不续训退步step2025。
本会话不能访问迁移后的GPU共享目录，尚未启动GPU，也没有新的画质结果。
主代理实现、测试和推送；目的地CPU代理安全同步、核对资产，再给用户GPU命令。

保持右图③：image-router分数＋冻结VLM视觉token经小MLP产生的修正分数，
softmax混合FLUX内部FFN专家。仍是五个块、每块两个专家、一套共享LoRA；
不是每个退化一个专家，不是多LoRA专家库。不增加CNN、Restormer、energy或新的恢复模块。
FLUX原生VAE和冻结LPIPS网络照旧。独立energy工作线、旧代码/配置/权重不改。

## 已知原因与未知部分

成功step2000本来已有VLM，不是加上VLM才突然退步。五类资格新增Noise、Haze、Low-light后，
Rain/Blur/Noise两项变差，Haze/Low-light两项改善。
交叉检查固定旧专家比例、只替换训练后的修图权重，Rain仍掉7.173306dB、
Blur预览掉3.583147dB。因此单改router不足，必须保护修图权重更新。
Blur的路由变化也伤害LPIPS，不能说路由完全没问题。

“修图权重”包括共享LoRA和FFN专家，尚未隔离两者各自责任。
开发图Low-light/Haze平均MSE分别约为Rain的109.52/34.19倍，支持调查误差尺度失衡；
这不是实际训练梯度主导/冲突的证明。新配方是针对已知功能退步的可检验修复，
不是已证实的唯一根因或成功保证。也不能把五类性能改善归为VLM独立贡献。

## 精确改动

1. 严格加载原成功step2000全部可训练权重，包括已有VLM投影；不重置VLM。
   父模型身份沿用旧五类config内的报告、收据及SHA。没有读取step2025进行初始化。
2. 这是新优化阶段：Adam重新初始化，阶段步数0→25，不冒称原batch32优化轨迹精确续训。
3. global batch从32改为40：8rank，每rank累积5次，每次一类。
   每个优化步五类各8张；原来每步四类、五步才平衡。25步每类200次，共1000次曝光。
   仍在311,921条训练池按来源均衡采样，同一确定性512裁图和D4。
4. 共享LoRA和FFN专家学习率1e-5→1e-6；image router/VLM投影仍1e-4。
   全部保持可训练。全冻结修图权重不适合本次目标，Haze/Low-light需要适应。
5. 六项原损失保留，但改为每张训练裁图相对旧模型误差归一化，并分别惩罚MSE/LPIPS退步。
   不蒸馏旧专家比例：Noise的交叉检查显示新比例可能有益。

具体公式：对同一训练裁图，旧模型六项损失为常数 `r[k]`，当前为 `l[k]`。
权重仍为flow MSE=.1、latent L1=.25、RGB MSE=10、Charbonnier=1、highpass=.1；
LPIPS Rain/Noise/Haze/Low-light=.05、Blur=.20。

```text
R  = sum(w[k] * l[k]) / max(sum(w[k] * r[k]), 1e-4)
HM = relu(l[rgb_mse] - r[rgb_mse]) / max(r[rgb_mse], 4e-6)
HP = relu(l[lpips]   - r[lpips])   / max(r[lpips],   1e-3)
loss = R + 5 * (HM + HP)
```

这里归一化的是每张图六项加权总误差，并不是给六项分别除参照，也不是直接对dB取比例。
RGB训练范围[-1,1]；4e-6只用于分母数值保护，对应[0,1] MSE约1e-6。
惩罚仍相对真实旧误差计算，分母下限不提供免费的退步余量。
系数5是本次预先固定的工程选择，不声称来自理论最优或调参结果。
旧表现较差的新任务仍允许大幅改善；不要求它们模仿旧模型的错误输出。
软惩罚不保证训练中、尤其未见开发图片上绝不退步，需实测。

## 参照缓存与数据安全

入口依次运行三个阶段：

- features：冻结Qwen2.5-VL＋既有V8 adapter，只编码对应恢复裁图的LQ视觉tokens。
- reference：加载冻结的step2000 FLUX，只对1000个训练裁图计算六项误差标量。
  不保存教师图像，不同时驻留第二个FLUX，不训练任何网络。
- train：同样从step2000加载，训练25步；每次读取该裁图的冻结误差作为尺度和保护参照。

缓存绑定config、科学源码、父权重、record、来源、裁图位置、D4种子等身份。
新缓存独立目录，不能把旧缓存改名复用。中断已完成的特征/参照保留并核验后复用。
参照生成明确拒绝validation/internal/official路径；只有训练GT可计算训练损失。
开发GT只用于算指标，不输入模型/router，不参与反传或训练参照缓存。

验证沿用之前每类100张固定来源隔离预览，共500张，在阶段0、5、25评估。
Noise/Low-light各100张只有34/21个来源，不能说各100独立来源。
Blur预览不是完整2595张，因此不把预览是否超过25.9/.18当完整Blur门槛结果。
训练/开发按source group和相同GT哈希排除重叠；官方、sealed、internal、parent-test像素不读。
即使数据路径为MillionIRData/Train，开发图片也不在本次或父模型训练池。
这是反复用于决策的开发集，不是最终独立测试集。官方测试留待方案、超参及checkpoint冻结后。

## 如何判断结果和下一训练规模

工程PASS只指有限loss/梯度、每rank实际更新VLM、完整权重/Adam/RNG收据、合法评估；
不等于科学成功，不因PASS自动formal。分别报告五类相对同环境step2000的PSNR/LPIPS变化，
不能平均成一个分数掩盖某类退步。保留父历史完整结果与本次同图预览，不换弱分母。

先看Rain/Blur/Noise是否仍明显退步，以及Haze/Low-light是否在学。
若仍复现数dB遗忘，不照原样延长；进一步区分共享LoRA/FFN专家责任，
或分析每类损失/参数组梯度记录。不能凭当前损失大小宣称已经测出梯度冲突。
若保持已有能力且新增任务能学，下一计划为独立正式1000步、batch40，
40,000次曝光、每类8,000次；每250步预览，0/1000做完整五类开发评估。
这不是只用25步判断最终能力，也不是保证1000步够达标；预算和入口需根据资格结果确认。
正式仍从原step2000重新起，不加载25步资格权重。当前formal入口明确禁用。
五类稳定后才进入已批准的现有5种double＋1种triple，不擅自扩大到所有数学组合。

## 资源、保存和恢复

4节点×每节点2张H100/H200，共8卡；节点内同型号，跨节点可混用，软件版本必须匹配。
共享GPU目录：`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
绝不在旧agent-driven-world-model目录启动；不复用历史IP。
Python使用已核验项目环境，默认仍指向原`.envs/gpu-train-stable-hdd3/bin/python`；
若目的地该环境不可见，先核验相同依赖版本再显式设置PROJECT_PYTHON，不能悄悄换环境。

资格预计.3–.8普通墙钟小时、2.4–6.4 allocated GPU-hours，包含特征、参照和评估；
每次启动硬cap1.5小时，至少80GiB新空间。不是已测速度，不保证新节点/网络性能。
正式1000步目前仅粗估3–6小时、24–48 GPU-hours，必须用资格实测更新，包含缓存及完整验证。
逐节点日志记录总耗时；result里的train阶段耗时不替代整个三阶段launcher成本。

每5步完整保存：FLUX可训练权重、Adam、八rank Python/NumPy/PyTorch/CUDA RNG、
新阶段step与采样位置 `next_sample_position=step*40`。旧batch32收据一律拒绝。
模型/Adam原子发布、收据最后写入；残留partial不是可恢复状态，保留不删除。
中断用新run ID，第四参数仅指定本配方完整收据；不覆盖旧目录/日志/部分文件。
最后step25训练已保存但评估中断时，可从25恢复只补评估，不额外训练。

## 文件与目的地代理操作

新配置 `configs/foundir_token_moe_retention_qualification_v1.json`；
helper `smoke/token_moe_retention_v1.py`；
训练器 `smoke/train_foundir_token_moe_retention_v1.py`；
状态模块 `lowlight_restoration/token_moe_retention_state.py`；
启动器 `smoke/run_foundir_token_moe_retention_v1_4x2.sh`。
旧31项科学源码应与五类资格报告逐项一致；不修改energy源码。

目的地安全同步本次focused commit及依赖，保留dirty、全部模型/缓存/数据和energy工作。
先使用核验项目Python运行以下CPU入口参数（不是GPU启动命令）：

```text
smoke/train_foundir_token_moe_retention_v1.py --phase preflight --run-id qualification_4x2_v1
```

预检通过后，获取当前node0 IP，给用户四条GPU命令；不要自行启动。
启动器参数为node rank、当前node0 IP、共同新run ID、可选同阶段恢复收据；默认端口29768。
不要复用旧五类25步启动器，它会加载旧Adam并使用失败配方。
不要启动正式、混合或另一条energy工作。

完成后提交小报告：四节点原始日志/退出状态、三阶段与总墙钟/GPU-hours、
0/5/25各500张逐图指标与均值、相对起点变化、逐类相对loss/退步项、组梯度、显存、
完整收据与最终权重/Adam SHA、源码与数据绑定。大模型、图片和缓存留共享盘。
本地CPU测试结果见本报告后续验收小节；真实GPU验证尚待运行。

## 已完成CPU验收（不是GPU成功）

- 最终56项单元/回归测试全部通过，41.159秒（不含导入启动时间）。包括新增12项、
  旧五类/交叉诊断/正式与续训测试，以及原token-router的11项函数测试。
- 八进程真实CPU/Gloo batch40测试通过：每图相对损失的DDP梯度与串行40样本一致；
  权重、Adam状态和Python/NumPy/PyTorch RNG恢复逐值相同；继续更新后与不中断版本
  最大参数差4.656612873077393e-10，容许FP32归约次序误差。
- 训练器CLI导入、5个新Python文件编译、launcher bash语法检查通过。
- 实际311,921条训练元数据、来源/相同GT隔离、冻结预览身份、父step2000报告，
  旧五类资格绑定的31项科学源码逐项核验通过；没有打开保护像素。
- 测试覆盖训练参照拒绝validation/internal、缓存身份、零误差分母、双指标退步梯度、
  每步五类平衡、已有VLM权重严格加载、batch32/异配方/资格转正式恢复拒绝。

八进程可复现入口：smoke/smoke_token_moe_retention_cpu_ddp_v1.py。
单元文件：tests/test_token_moe_retention_v1.py，并回归原token_moe相关测试。
CPU使用指定项目Python及单线程环境；本地-S测试显式提供项目site-packages。
本会话未完成目的地大资产预检、CUDA前反传、H100/H200实际显存/画质或混型号互联验收；
不能把CPU通过说成这些已通过。目的地预检及25步GPU资格仍是下一步。
