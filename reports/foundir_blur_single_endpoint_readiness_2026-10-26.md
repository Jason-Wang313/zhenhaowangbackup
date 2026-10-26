# Blur 单专家对照：先确认这条恢复分支能否独立工作

状态：用户已授权实现和先做工程验收。GPU 尚未启动，无新质量结果。

## 为什么做

上一轮 3,000 步 Rain+Blur MoE 已正常完成，但没有同时做好两个任务。
Rain100 为 `34.36398316275458 / .07267702806275338`，达到要求；
完整 Blur2595 为 `26.10596660902048 / .3271988157297777`，LPIPS 未达到 `.18`。
Blur 输入本身为 `25.849544629199475 / .29110622298011024`，输出的 LPIPS 反而更差。

同一组 Blur100：模型自己选专家时为
`24.966814240094454 / .3831212885398418`；固定使用专家1时为
`25.041084236119197 / .38290703640086576`。固定专家只帮了一点，
不能把主要差距都归咎于 router。

旧的成功 Blur 路线使用另一套 LoRA、Blur 提示词和 50 步生成流程。
上一轮的新 MoE 没有继承它，不能说“把已经成功的 Blur 直接装进 MoE 后变差”。

这次问：**不经过路由、不混入 Rain 训练，同样的一步 Blur 分支能否学好？**

## 冻结对照

| 项目 | 本次设置 |
|---|---|
| 初始化 | 原始成功 Rain step4,000 LoRA；不是失败 MoE，也不是验收权重 |
| 可训练部分 | 同样的五个 double-stream block，各一个完整图像 FFN，共 424,673,280 参数 |
| 冻结部分 | Rain LoRA、其余 FLUX.2 Klein4B、AE、文本缓存 |
| 不使用 | router、VLM、Rain 训练样本、Restormer teacher、pixel refiner、新损失 |
| 恢复方式 | 原来的一步、t=1、零噪声、FP32 latent subtraction |
| 提示词 | 原来的固定 Rain 文本缓存；这次故意不同时换提示词 |
| 数据池 | 原来的 105,057 条 Blur / 7,339 个来源，按来源均衡采样 |
| 裁剪 | 训练512随机裁剪+D4；开发集原有固定512裁剪 |
| 正式规模 | 3,000 次优化，每次16张 Blur；48,000次样本呈现，不是遍历完整数据池 |
| 对应关系 | 与旧 MoE 每次32张中那16张 Blur 的记录、裁剪、增强顺序逐一相同 |
| 优化 | AdamW，学习率1e-5，weight decay0，梯度裁剪1，FP32可训练权重和Adam，BF16计算 |
| 损失 | 继承原 Blur 项及系数；teacher权重仍为0 |
| 整体权重 | Blur平均损失乘0.5，保留它在原16Blur+16Rain目标中的权重；移除Rain项但不重新放大Blur项 |
| 中途验证 | 0/1,000/2,000/3,000；中途100张预览，正式最后用完整2,595张 |
| 最终数值要求 | 完整 Blur PSNR≥25.9、LPIPS≤.18；另外报告逐图退化情况 |

这是“单任务、无路由”对“混合任务、有路由”的诊断，不是只改变路由一个变量的严格因果消融。
单分支容量匹配，整个模型的可训练参数量/计算量不匹配；后期旧 MoE 的软路由也不能与此完全等价。

如果独立分支明显变好，说明路由/混合训练值得进一步定位；
如果仍差，只能说明**这套固定初始化、训练范围、提示和损失下**的一步分支不够好，
不能推出“一步 FLUX 不可能去模糊”，也不能立即认定是哪一个损失项导致。
不会因通过数字门槛就宣称 MoE 或论文创新成功。

## 先工程验收，再正式训练

先用全新的 `qualification_v1` 做25步，不自动启动正式训练。
更新前先检查 Rain100 和 Blur100 的逐图结果是否与旧初始化一致；
要求有限梯度、真实参数更新、无OOM、可保存FP32权重和Adam、有效的同协议评估。
25步结果仅代表代码可运行，不代表 Blur 已做好。

正式训练必须从原始成功 Rain 重新初始化，绝不能接着验收的25步训练。
正式训练每250步保存一次模型、Adam、绝对采样位置和各rank的随机状态；
只有最后写完的 `.resume.json` 完成凭据可用于续训。
续训使用新run ID，不覆盖旧目录；校验同配置、同代码、权重/Adam哈希和步数，拒绝验收权重。
保存间隔缩短不增加验证次数，也不改变训练目标。
保存完整状态预计约5.1GB/次，正式12次约61GB，另有日志和图片；不自动删除旧结果。

GPU：单机同型号8×H100或8×H200；无跨节点。
验收暂估 **0.15–0.50小时 / 1.2–4.0 GPU-hours**。
正式暂估 **1.0–2.5小时 / 8–20 GPU-hours**，需用验收实测速率重估；
包括完整开发集评估和较频繁保存造成的不确定性，不承诺通过质量门槛。

## 可复现入口

- 配置：`configs/foundir_blur_single_endpoint_v1.json`
- 配置 SHA-256：`a273b55ccfd9ca128b976ea5a6fa314ce862d77273a4127f0e5836e58957eefb`
- 训练器：`smoke/train_foundir_blur_single_endpoint_v1_ddp.py`
- 数据/恢复助手：`lowlight_restoration/blur_single_endpoint.py`
- 测试：`tests/test_foundir_blur_single_endpoint.py`
- tmux：`smoke/start_foundir_blur_single_endpoint_v1_tmux.sh`
- 直接启动器：`smoke/run_foundir_blur_single_endpoint_v1_8gpu.sh`
- 输出：`checkpoints/foundir_blur_single_endpoint_v1/<run_id>/`
- 日志：`logs/foundir_blur_single_endpoint_v1_<run_id>.log`

```bash
bash smoke/start_foundir_blur_single_endpoint_v1_tmux.sh qualification qualification_v1
```

待验收通过、确认预算后，正式入口是 `formal formal_v1`，不会自动执行。
意外中断的正式训练可以使用：

```bash
bash smoke/start_foundir_blur_single_endpoint_v1_tmux.sh formal continue_v1 checkpoints/foundir_blur_single_endpoint_v1/qualification_v1/result.json checkpoints/foundir_blur_single_endpoint_v1/formal_v1/step_001000.resume.json
```

上面的续训例子只在对应文件确实存在、旧进程已停止时有效。
锁只能防同节点并发，不能证明其他节点没有旧进程。
不读取官方、sealed、internal、parent-test像素；数据/裁剪继承已审计的父协议。

## 本地验收状态

静态语法/JSON检查通过。新诊断11项测试、原MoE17项回归测试共28项通过，
包括两套双rank CPU DDP测试、48,000次采样位置对应、实际来源隔离、
真实初始化不变、训练参数量检查，以及保存后续训的权重/Adam/RNG精确一致。
完整实际资产只读预检通过（包括父配置、初始权重、数据路径、开发集文件哈希）。
训练图片在实际读取前按哈希校验，不谎称已经完整解码全部训练集。
GPU验收仍待用户节点执行；CPU检查不能证明实际显存安全或图像质量达标。
原 MoE 的代码、配置和成功 Rain/旧 Blur 权重保持不变。
