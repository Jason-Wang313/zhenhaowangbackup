# CPU路由特征覆盖审计终态 — 2026-11-04

已按bc25869交接运行指定CPU shell入口，ROUTER_FEATURE_COVERAGE_PASS，ROUTER_GENERALIZATION_AUDIT_END status=0。整体wall耗时40秒；报告内部计算段32.277631秒。CPU8线程，0 GPU-hours，一次执行，无重试、重训或配方修改。

全部680记录（640训练/40开发）身份、标签映射、crop边界与同图同crop特征链通过；VLM复制逐元素相等。训练/开发source_group、GT SHA、LQ SHA交叉数均为0，每条训练记录曝光20次。仅核验元数据标签一致性，没有视觉审核标签。

固定去均值余弦距离，报告全部40开发图的图像/VLM前5训练近邻、各类最近邻和排除同来源训练参考分位：

- Blur错例0007534：同类别最近距离在图像参考43.75%分位、VLM参考71.875%分位；图像和VLM最近邻均为Blur。
- Lowlight错例0479048：图像参考83.59375%分位、VLM参考99.21875%分位；图像前5训练近邻全为Noise，VLM前5为Noise/Noise/Noise/Haze/Noise。

以上是描述性覆盖证据，不是校准OOD阈值、唯一根因证明或“增加数据必然解决”的证据。没有扫系数、改标签、开发回灌、挑checkpoint或新增恢复输出。

只读现成小特征与JSON，未读GT/LQ像素、reference latent或大模型，未使用GPU。完整40图结果见同名JSON；只上传小报告及本耗时简报，缓存/模型/日志保留共享目录。不启动后续任务。

日志：logs/foundir_router_generalization_audit_v1.log
日志 SHA256：6b1e80e1e56ef1898c4c4b4b742594acbf0a48e6e44537c25772009625c33bff
