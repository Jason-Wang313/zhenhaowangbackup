# 官方V4恢复首轮失败 — 2026-11-05

四节点均在第一块生成前退出，错误一致：`ValueError: official metric weights changed`。8worker没有OFFICIAL_VLM_V4_TILE，落盘恢复块0/21828、整图指标0/600。没有新的PSNR/LPIPS/SSIM结果。

| 节点 | 秒 | 退出码 | 使用GPU秒 |
|---|---:|---:|---:|
| 0 | 419 | 1 | 838 |
| 1 | 421 | 1 | 842 |
| 2 | 415 | 1 | 830 |
| 3 | 399 | 1 | 798 |

失败消耗0.918888889 GPU-hours；本轮官方累计含route为2.658888889 GPU-hours。历史V4的2.473888889 GPU-hours另计。错误发生于原计划核验/模型加载后、恢复循环之前，日志没有报告OOM。

只读CPU核对：当前pyiqa默认三项模型的CPU身份与原receipt完全一致。发现pyiqa InferenceModel用`Parameter(empty).to(device)`建立dummy_param：CPU保持注册Parameter，跨到meta后成为普通Tensor、从state_dict消失。PSNR底层net的state_dict均为空；CPU包装器hash为131310…，空字典hash为e3b0c4…。这证明包装器身份可随设备变化而拒绝未变的评分网络。此为CPU/meta机制复现；原GPU日志没有打印实际各项哈希，不能宣称已完全隔离CUDA根因或证明全部权重文件未变。

未修改绑定producer/receipt/plan、缓存或环境，未解码任何测试像素、未重启任务。全部失败账目和日志保留。本次只推小故障JSON与简报。后续需要修复设备相关的评分身份检查，并继续核验真实网络参数；不能直接删除检查或覆盖原receipt。修复需兼容原路由缓存，不能重跑训练或route。
