# 四图模型自选专家单选补测终态 — 2026-11-04

predicted_hard_four_v1 已完成，固定4/4图，无训练、无额外图像。CPU summarize再次返回 ROUTER_HARD_RESULT_VERIFIED，launcher status=0。

本轮单节点2×H200，实际耗时77秒（1分17秒），154 GPU-seconds = 0.042777778 GPU-hours；共一次attempt，无失败尝试成本。如独占更多GPU，整机资源占用成本另计，未确认独占范围。

四张错误路由图平均：相对正确类别单选，模型自选专家hard输出PSNR低11.193566dB、LPIPS高0.137361。原soft相对这个hard平均PSNR高0.024818dB、LPIPS低0.020667。因此这4张的巨大差距主要已存在于选错专家及其配套推理协议；直接改hard不能解决，且平均略差。该拆分不是纯分类因果估计，也不能把4张错例均值当40张总体指标。

逐图有取舍：0144583 hard比soft PSNR高1.405131dB，但LPIPS更差；另外3张soft的PSNR/LPIPS都优于hard。结合原36张选对组仍有混合损害，不能推断全局hard/soft任一种已经解决问题，不进行GT逐图择优。

记录峰值allocated约137.933GiB/卡；本次也不能作为H10080GB显存验收通过证据。仅保留实测，不在本轮修改加载/显存策略或科学配方。

完整逐图指标、身份与输出SHA见同名JSON。仅上传小报告，图片/tensor/权重/原日志保留共享目录。未启动后续GPU或训练。

日志：logs/foundir_router_hard_counterfactual_v1.log
日志 SHA256：88c27b783c2f0cb8a58aee7459fcd67b711113f183b03dd7e43a4b5b2e70a787
