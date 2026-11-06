# 固定LQ起点官方对照 — 2026-11-06

## 原理及边界

原模型训练是x_t=(1-t)*GT_latent+t*noise，速度目标noise-GT_latent；推理从纯noise t=1下降到0。
LQ通过reference token进入，训练/推理的符号及reference_time=10一致；未发现这里的明显接线错误。
当前Blur残差幅度四组均未达标，证明已试幅度不足，不证明全部FLUX路线不可能。

本轮检验：恢复时直接保留部分LQ空间内容，而非全部从噪声重新生成，是否改善官方像素保真。
只用一个预定起点：原50步shifted时间表第一个t<=.5，即index43/t=.4716968238353729，沿剩余7次真实调用到0。
起点(1-t)*encoded_LQ+t*原seed噪声，原LQ参考token仍存在；本轮不训练或优化官方像素。
该状态与训练clean+noise分布不同，属于未验证推理干预，可能欠修复或退化。结果不保证成功，也不自动扩展强度扫描。

## 固定的其余设置

同一父receipt/LoRA/底座/VAE/提示词，冻结原V4路由，只有预测为Blur时启用；不使用GT类别选路。
保留原CFG1的双分支BF16算术、原reference_time10、相同每块seed、alpha=.75及双uint8。
VAE沿用FP32 native convolution和query256策略。512图块/128重叠、原尺寸拼接与pyiqa0.1.16均保留。
150张Blur/2250块重新生成；另外450张按已绑定的原部署图像和指标复用，每轮完整600图汇总，十项门槛不变。
新候选输出先保存，随后才加载GT评分。不是独立盲测，不用GT挑输出，不把已知类别参考作为实际路由。

## 执行与核验

新模块lowlight_restoration/flux_lq_start_v1.py；入口smoke/evaluate_foundir_fullstack_lq_start_v1.py；shell run_foundir_fullstack_lq_start_v1.sh。
4项核心CPU测试通过：原时间表suffix不重采样、插值端点、非法输入拒绝、真实sampler的参考token/seed/无梯度/调用数。
600图六分片无遗漏重复，每rank100条/25张Blur/375块。底座、VAE、五专家和提示缓存完整SHA预检通过，未读官方GT像素。
总15750次native CFG调用（每次包含两个分支），远重于只评分的上一轮，故用3节点×2H100；各节点standalone无跨节点通信。
粗估20–40分钟/2–4 GPU-hours，非测速保证；不能假定7步比50步整体精确快50/7倍，加载/VAE/评分/I/O另计。

新run round2_blur_lq_start_v1，输出checkpoints/foundir_fullstack_lq_start_v1。旧run/source不修改。
每图块PNG+identity receipt、每图完整输出/指标独立保存；进程失败保留所有资产，不自动重试。
核验每rank实际torch/CUDA/VAE数值策略，6worker完整性、逐图/逐块SHA和完整600均值。
GPU结束后CPU执行 `python -S smoke/evaluate_foundir_fullstack_lq_start_v1.py finish`（共享环境Python），推送小报告、退出码及实际GPU-hours。
