# 两条 MoE 正式训练：并行启动核验

检查时间：2026-10-26 20:05:52 UTC。仅检查共享日志／已生成记录，未访问远端实时 GPU 状态，未启动、停止或修改训练。

## 当前结论

两条都已实际进入训练，不再停留在预检／模型加载。以下为检查时刻快照，不是最终结果。

| 实验 | 硬件 | 启动 UTC | 最新观察步数 | 日志显存峰值 GiB |
|---|---|---|---:|---:|
| image_only/formal_v1 | 8×H100 80GB | 19:45:55 | 480/10000 | 42.56962299346924 |
| vlm_assisted/formal_v1 | 8×H200 | 19:54:58 | 60/10000 | 42.57342576980591 |

两条损失／梯度记录有限，日志继续推进；未发现 Traceback、OOM、NCCL fatal、RuntimeError 或 launcher END。
H200 本次已通过通信、模型加载和实际训练；不能据此宣称此前 CUDA401 的唯一根因已证明。
训练后首次质量验证在1000步。目前不能宣称 VLM 优势、Rain／Blur 达标或论文创新成立。

## 可比性与硬件差异

两条配置、29项绑定源文件、数据身份一致，独立从原 Rain step4000 初始化。
配置 SHA-256：`3bdd59b3fb3c010cc7a93ab299f3f67fe85cc5eb5a6f380f610d4338398baebe`。
逐图 ID、来源组、input PSNR／LPIPS 完全相同，通过现有比较器的输入差异门槛。
但是 H100／H200 正式 step0 的输出有微小数值差异：

| step0 协议 | image-only PSNR | image-only LPIPS | VLM PSNR | VLM LPIPS |
|---|---:|---:|---:|---:|
| Rain100 | 34.17715539550872 | .07015681236982346 | 34.17745915897161 | .07014857136178762 |
| Blur100 预览 | 23.8596342447602 | .4188850187882781 | 23.86055132991541 | .4188507604598999 |

VLM减image：Rain `+.0003037634628881847 dB / -.000008241008035841935 LPIPS`；
Blur `+.0009170851552084969 dB / -.000034258328378222735 LPIPS`。
这不是 VLM 改善证据。正式两臂不能称输出逐位一致，也不能把运行速度当作同硬件方法比较。
此前两次 H100 验收起点完全相同仍成立；不要混淆验收与正式阶段。
若最终方法差异同样微小，仅靠这次跨硬件结果不能可靠归因给 VLM。

## image-only 第250步完整断点

目录：`checkpoints/foundir_rain_blur_joint_vlm_moe_v2/image_only/formal_v1/`。

| 文件 | 独立重算 SHA-256 |
|---|---|
| step_000250.safetensors | 16ccb712af15f2f00a36e8fc1ca7e8a4bc7bfe1a4ed5ff18c65e0f772efaceb0 |
| step_000250.optimizer.pt | 3b7eb0969789010bfc666d397d6b43ec54081cedb6dbfd3d1948585822cb741b |
| step_000250.resume.json | 251fa6de8692ae363100d2a8caccb2aa9265aabc8da13e4c74c8da4f6b96815c |

权重／Adam hash与receipt一致；step250、formal、world8、next_sample_position8000，
绑定源文件与正式receipt一致。权重／Adam有限性由运行时保存逻辑检查；
本轮独立核验文件hash，不宣称另行加载检查了所有张量。

## 保持既定计划

每臂10000步、全局batch32，Rain／Blur各16；每臂320000次呈现，不是唯一图片数。
验证在0/1000/3000/6000/10000，完整Blur2595只在10000。固定最终checkpoint，不中途调参或挑权重。
目标仍为Rain PSNR≥34.2／LPIPS≤.08，Blur≥25.9／≤.18；现阶段没有通过结论。
两条是独立单机八卡作业，不是跨节点16卡DDP。不能在任一忙碌节点重复启动。

原预算每臂5.45–9.06正常小时／43.63–72.45分配GPU-hours；两臂合计87.26–144.89 GPU-hours。
截至检查时，从launcher启动计，image-only已1197秒（.3325小时／2.66 GPU-hours），
VLM已654秒（.18166666666666667小时／1.4533333333333334 GPU-hours）；包含启动和初始验证。
这些不是最终耗时，也不是利用率测量。并行只缩短等待，不减少总计算量。

证据：`logs/foundir_rain_blur_joint_vlm_moe_v2_{image_only,vlm_assisted}_formal_v1.log`
及对应checkpoint目录的receipt／step0指标记录。原始日志、权重、图片不提交Git。
本轮没有读取受保护像素、重跑评估或修改hash绑定的训练源文件／配置。
