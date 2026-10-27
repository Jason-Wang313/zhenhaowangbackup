# Novel PPT Approach — VLM-Conditioned Restoration Energy

[`Novel_PPT_Approach_VLM_Restoration_Energy.pptx`](Novel_PPT_Approach_VLM_Restoration_Energy.pptx) is Jason's original, advisor-approved proposal. Renamed from `original_attempt_retry` at Jason's request on 2026-10-27. The PPTX bytes are unchanged.

The approach was tried before without establishing the intended energy-specific quality improvement. Jason has now authorized a careful retry alongside the VLM/MoE diagnosis. **"Novel PPT Approach" is the requested project name, not a verified novelty or success claim.**

核心思路：先固定 FLUX.2 / Restormer 的初步恢复结果，再训练一个结合原图和 VLM 退化判断的 energy。它既要分清不同恢复结果的好坏，也要通过对候选图求梯度来真正修图。不能把普通 refiner 的提升、GT 帮忙选结果的上限，或离散选择器的总指标，算成 energy 修图成功。

旧 V2 已实现真实候选、排序和 1/2/3 步梯度修图，但效果失败；后续安全选择器不是完整的梯度修图实现。新重试应先定位失败原因，保留旧结果，不能直接重复失败配方。

审计与下一步：[双路线瓶颈与重试记录](../../reports/foundir_vlm_and_novel_ppt_retry_audit_2026-10-27.md)。

Source PPTX SHA-256: `CDB8D5E0FE682D6CF30F0BA217D64C426E682FBBC7C70BD51E950ED38EDACEFB`.
