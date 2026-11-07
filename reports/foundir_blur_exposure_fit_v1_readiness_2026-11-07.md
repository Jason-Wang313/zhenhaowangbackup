# Blur最终1280权重：同64训练图只读诊断

官方最终1280已完整失败：20.936988/0.196516；不得重跑该训练或官方评价。下一项仅检查最终模型在同一历史64训练输入上是否真正改善，不新增测试集或质量门槛。

复用smoke/diagnose_foundir_blur_coverage_fit_v2.py的原64图推理、量化、评分及前后参数SHA核验函数；新wrapper只绑定最终1280导出权重和新输出根。原训练缓存、裁图、指标、64项顺序不变。不创建optimizer、不更新参数、每图只有一个输出，不用GT选择。比较原step200固定模型64图与当前step1280，避免把online训练曲线当固定模型评价。

目的地CPU预检通过，6.694160秒/0GPU-hours，旧64项身份、缓存、最终权重SHA/parameter_hash、官方终态绑定通过；Python与shell语法通过。实际最终权重GPU路径须待新job核验。

只使用物理node3两张H100；预估1–3分钟/0.033–0.10使用GPU-hours，依据旧同64诊断48秒而留加载余量，不保证。固定64图，故障停、无自动重试或新训练。新job round15_blur_exposure_fit_v1。

发布源码后通过现有node3常驻worker派发：

```bash
.conda/gpu-train-stable/bin/python -S smoke/foundir_fullstack_node3_extension_v1.py submit --job-id round15_blur_exposure_fit_v1 --entry smoke/run_foundir_fullstack_blur_exposure_fit_v1.sh --commit <verified_release_commit> --purpose 'Read-only final1280 on the same historical training64; no updates or new gate' --sources smoke/diagnose_foundir_blur_exposure_fit_v1.py smoke/diagnose_foundir_blur_coverage_fit_v2.py smoke/evaluate_foundir_blur_exposure_official_2x2_v2.py lowlight_restoration/blur_delta_velocity_v1.py
```

本轮无独立持久CPU调度器；接手先检查round15队列与node3心跳，不重复提交。GPU两rank前后权重hash相同且launcher退出0后，在CPU执行新wrapper的finish，生成result与逐张comparison并推送小报告/耗时。

新目录checkpoints/foundir_blur_exposure_fit_v1；入口smoke/diagnose_foundir_blur_exposure_fit_v1.py prepare|run|finish。仍只攻Blur，官方150的25.9/.18通过前不推进其它类别。所有旧资产保留。
