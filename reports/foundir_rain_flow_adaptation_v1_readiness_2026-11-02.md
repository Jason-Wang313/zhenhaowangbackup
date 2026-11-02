# Rain一步能力迁移到多步：25步工程资格

## 结论和动机

前一轮已经把四个已有FLUX专家接进同一内部LoRA库：8/8单选与原图逐像素一致。
但固定90%主专家的soft混合损害了六张图，只有两张Low-light双指标改善。
这说明原能力确实继承了，也说明比例不能不看图就固定；不是完整MoE或VLM已成功。
用户授权下一步Rain适配。本次只训练独立Rain分支，不修改四类原专家，不训练router。

Rain原成功模型是一步“从LQ修到结果”，其他四类是多步“从噪声逐渐生成结果”。
二者权重形状相同，不代表输出含义相同。新分支必须学会各个噪声时刻应往哪里走；
不能把原一步残差当作所有时刻速度，也不能只重复调用50次。

## 从哪里开始，保留什么

- 新分支复制成功Rain `precision_resume_v1/continue_v1/step_004000.safetensors`，
  SHA256 `020113ce209728452917d49859da7be220f638f0d27f6f87f182ffdab0a8fc2e`。
  原32,440,320个LoRA参数保持FP32，fresh Adam；不是随机初始化，也不是严格续训。
- 原一步Rain FLUX冻结，在这32张训练裁图上提供恢复示范，缓存之后释放。
  同一进程先教师、后学生，训练时没有第二个常驻FLUX。
- 原模型历史用过Restormer蒸馏，这个历史事实保留；本轮完全不加载Restormer。
- 四份已有多步LoRA、共享FLUX base和VAE不更新。仅独立新Rain的原50处LoRA可训练。
- 复用原Rain prompt tensor，原生LQ reference、ref time10、CFG1。
  新学生推理走已有四类相同的50步原生入口；Rain输出一次uint8、alpha1，不混LQ。
  不加任何CNN/Restormer恢复网络、energy、像素修正器、额外VLM或router模块。

## 训练到底学什么

配对GT提供主要答案。令编码后的GT为z_GT，随机噪声为epsilon：

`z_t = (1-t) * z_GT + t * epsilon`

`v_target = epsilon - z_GT`

`z_clean_pred = z_t - t * v_model`

每次同时把原LQ编码作为reference输入。t均匀采样于[.0001,.9999]，每个样本/位置
有独立固定种子；不把端点专用的`degraded-v`公式误用于中间t。
路径构造与clean减法用FP32，模型token/矩阵计算用BF16，LoRA与Adam状态FP32。

固定loss：

| 监督 | 系数 | 依据 |
|---|---:|---|
| GT flow速度MSE | 1 | 学兼容原生采样的速度 |
| 预测clean latent与GT的L1 | .25 | 约束端点保真 |
| 解码RGB与GT的MSE，[-1,1]范围 | 10 | 直接约束像素误差 |
| 解码RGB与GT的LPIPS，[0,1]范围 | .05 | 保留感知质量目标 |
| 解码RGB与冻结Rain输出的L1，[-1,1]范围 | .1 | 低权重保留已有去雨行为 |

教师不是无误真值，不代替GT，不提供虚构的多步轨迹或速度标签。
这些系数是待验收的初始配方，不是文献保证或已经调到最优，不自动搜索。
当前仍然存在教师偏差、endpoint初始化不适应噪声状态、随机t与真实rollout差异等风险。
即使loss下降，50步结果也可能不好，所以必须实际采样比较，不能仅报告训练loss。

## 数据、规模和科学边界

仅使用原Rain训练manifest的38334条/5780来源作为选择池。逐项核对主manifest身份，
排除与全部非训练source或GT哈希的重叠；不解码开发/官方/封存/parent-test等保护图。
按固定哈希顺序取32个不同来源，各选一条≥512的记录，固定512中心裁图，无增强。
Teacher、GT、reference和训练输入缓存绑定同一record/crop/hash；不能复用错裁图示范。

每个optimizer step将32张各用一次，8个rank各micro1×累积4；每步固定种子打乱顺序。
25步共800次曝光，样本重复用于验收，不冒称扩大到全38334训练。
step0与step25在这32张中的固定前8张跑完整50步，各rank一张；同时报告原一步教师。
这些是训练拟合指标，与原100张开发均值不能直接比较。当前不读开发GT、不选checkpoint，
不根据结果改alpha/prompt，也不宣称泛化或VLM贡献。
正式Rain仍需在冻结100来源开发协议同时达到PSNR≥34.2、LPIPS≤.08；本轮不评该门槛。

## 显存、恢复和预算

当前FLUX的实际forward调用`block.forward_kv_extract`，不是普通`block.forward`。
新包装器在这个真实入口做非重入activation checkpoint，保留state-dict名字，
推理或无梯度时走原实现；VAE解码也重算。冻结参数不等于可以切断从RGB到LoRA的梯度。
LPIPS AlexNet只用于loss/评分，参数冻结，但训练时保留输入梯度。

4节点×2张H200，总8卡；每卡≥139GiB。上轮推理峰值137.73GiB不代表本轮反传峰值，
本轮CUDA显存尚未实测，不谎称H100已支持。四节点运行环境须一致；同run恢复禁止换环境。
预计10–30分钟/1.33–4 GPU-hours，非实测。每节点硬cap3600秒，总8 GPU-hours加清理。
超时保存已完成步，不自动重启/扩预算/缩裁图。每节点按两卡计成本，四节点相加。
预留至少20GiB：26份FP32 LoRA+Adam快照约10GiB，加缓存、评分及临时文件；不删除旧资产。

每完整step（含0）发布不可变LoRA、Adam、各rank Python/torch/CUDA RNG、next_global_slot、
训练日志和SHA。只保存LoRA，不保存4B底座。先fsync数据再发布收据；没有可滞后的latest依赖。
同run_id＋fresh共用launch_id恢复最高完整步，保留全部旧快照及未提交孤立文件。
文件损坏、收据断档、world/源代码/协议变化立即报错，不悄悄回退或假称严格续训。
未落盘的一步可能重算；本轮资格权重绝不能拿去初始化正式训练。
step0/25跨rank核对权重一致，每张缓存与probe独立保存并验证后复用。

完整工程通过要求：25步有限loss/梯度/FP32 Adam、LoRA真实变化、26份完整快照、
32份教师缓存、step0/25各8份完整50步输出、最终导出与step25状态一致、四节点退出0。
汇总重读tensor核验SHA/finite/量化；不独立重跑LPIPS。出现质量退步照实报告。
工程通过不是“Rain质量通过”，也不是允许自动长训。

## 目的地代理接手

本地CPU共36项通过（35项回归＋1项新增跨节点环境检查），另有两进程Gloo DDP恢复通过：
micro4×累积4×2rank=global32，恢复后下一步LoRA与Adam均逐元素相同，rank间参数一致。
实际Flux2小尺寸模型的forward_kv_extract重算与原路径输出、梯度也完全一致；
这不是4B CUDA反传或四节点NCCL资格。7个Python文件语法、2个shell语法检查通过。
真实源资产/来源隔离预检通过，最终44份源码绑定复核；不读真实图像。
32来源选择摘要：`811826667128ba780e79614ea0cd761c73d4a110890dfd4ac3ecfe0758c0c165`。
目的地挂载当前不可见，源预检不代替新节点资产核验，GPU仍未启动。

主代理在隔离worktree实现并focused push；目的地安全同步`origin/main`，保留脏文件、
模型、数据、缓存和另一代理energy。唯一GPU根仍为：
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
不要在旧agent-driven-world-model目录执行GPU，不能拿旧Rain启动器代替新入口。

使用已验证环境先跑CPU资产预检，不打开图像：

```bash
.conda/gpu-train-stable/bin/python smoke/train_foundir_rain_flow_adaptation_v1.py preflight --report reports/foundir_rain_flow_destination_preflight_v1.json
```

与前轮不同：本轮不复用父输出tensor，只依赖已绑定父终态报告和原权重；
所以不需要`--require-parent`，不要为不存在的参数额外改脚本。
通过后取得当前node0地址，为用户提供四条分节点命令，入口：
`smoke/start_foundir_rain_flow_adaptation_v1_tmux.sh`，参数node_rank、当前node0地址、run_id、
fresh共用launch_id。默认端口29684，每节点只暴露0,1两卡，不复用历史IP。
最后须`RAIN_FLOW_RESULT_VERIFIED`和四份`RAIN_FLOW_NODE_END status=0`。
不自动启动正式Rain、MoE/router、mixed训练或energy；只回传小报告，不上传权重/tensor/live日志。

GPU尚未启动；CPU验收记录在同名JSON，不能把CPU合成测试中的指标当真实恢复指标。
