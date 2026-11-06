# 纯FLUX专家为何开发通过、官方未通过 — 2026-11-06

用户明确指替代Restormer后的纯FLUX专家。此前回答转向旧异构v8系统，回答错了对象；旧v8真实性核对并不能解释这里的开发→官方落差。

|专家|当时通过的集合|当时PSNR / LPIPS|当前官方PSNR / LPIPS|
|---|---|---:|---:|
|Blur step10000，alpha .75|2595条开发记录，每条确定性512裁图|26.559141 / .151993|150图：21.546762 / .202622|
|Rain step800|100张开发裁图|34.301181 / .061867|100图：32.034930 / .075014|

Blur报告明确是full-development结果，双均值通过但单图最差退步安全门槛未过；不是所有门槛通过。
Rain报告明确“官方/封存测试未访问”，且其PSNR开发余量只有.101181 dB。
五专家接入116输入/248输出PASS，证明同环境bank单选与独立入口一致，Rain开发100图复现。没有证明官方600恢复指标通过。

所以当时没有做官方同口径确认，确实让能力差距晚暴露。不能把“不测官方”说成直接导致像素或模型变坏。
开发与官方在来源、裁图/完整尺寸、拼接、seed、评分实现上不同；现有证据不能分配各因素的因果比例，也不能未经隔离就把差距全归为过拟合或H100。
完整官方评价使用原分辨率、512块/128重叠拼接、pyiqa0.1.16；不能把开发512裁图的torchmetrics数值直接继承为官方通过。

后续只按用户已明确要求的官方测试集做效果验收，工程与合成张量测试单独标注。合法训练来源仍可训练，官方GT只评分/诊断，不训练或逐图选输出。
当前已知正确类别专家参考也没达标，因此问题不只在router。当前LQ起点对照是待实施的固定机制检验，不是已经证明的修复。

证据：
- reports/foundir_blur_only_flux2_scalar_residual_full_development_terminal_result_2026-09-16.md
- configs/foundir_blur_only_flux2_scalar_residual_full_development_v1.json
- reports/foundir_rain_onpolicy_continue_terminal_2026-11-03.md
- reports/foundir_five_expert_integration_terminal_2026-11-04.md
- reports/foundir_five_expert_integration_readiness_2026-11-03.md
- reports/foundir_official_vlm_router_v4_terminal_4x2_2026-11-05.json
