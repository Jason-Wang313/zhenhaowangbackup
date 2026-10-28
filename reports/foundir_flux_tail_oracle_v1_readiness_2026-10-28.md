# Noise FLUX 生成过程：最小机制诊断

## 当前决定与边界

2026-10-28 用户授权执行。先检查接线、梯度和冻结FLUX的可修正空间，
不完整复现多篇作者实验，不再重复泛泛文献审查，不训练新energy。
参考已完成的 `foundir_flux_tail_energy_guidance_deep_audit_2026-10-28.md`。

不是验证“energy已经成功”，而是分离以下问题：

1. 新入口是否改变原基线？
2. 当前latent的梯度是否实现正确？
3. 预计成品变好后，实际完成生成的成品是否也变好？
4. 在有限窗口/幅度下，GT告诉我们方向时是否有可利用空间？

当前GPU未启动。Codex无`/dev/nvidia0`或`/dev/nvidiactl`。
CPU/static/元数据结果不等同GPU实际资格。实测GPU耗时和质量结果均待运行。

## 冻结输入

- 配置：`configs/foundir_flux_tail_oracle_v1.json`。
- 复用`checkpoints/foundir_flux_energy_qualification_v1/qualification_v1`中
  索引0–7的8条/8训练来源缓存，固定512裁图，不重新选“容易样本”。
- 父preflight SHA256：`b09c57e2af4b7732504944e39222fce73aa4011ecab2dfbff47c5cdd57065cff`。
- 父prompt receipt SHA256：`fde1ed749b82c58722844dd284dce2bde720804cddcc49344510129448d4b625`。
- 父receipt绑定完整685,495条manifest、训练/heldout来源及GT哈希排除检查。
  新预检核验父receipt、原源码/配置、每条缓存元数据/哈希和本轮权重。
  不打开任何原始验证或测试图片，不读取最终1859条开发像素。
- Noise step400 SHA256：`1b4ed05ad5ad403a3d7fcc633eca97cdb69d00454e357da0c3d3131ac2c3ff1a`。
- 原50步、CFG1、reference time10、逐记录seed、真实shifted schedule。
- 原部署处理：raw先uint8 round，再与LQ固定alpha=.30混合，再round。
  不能使用raw FLUX较弱基线，也不加载额外Restormer。
- 只加载冻结Noise FLUX/VAE/LPIPS与已有Noise文本缓存；无在线Qwen、Haze模型或旧energy。
  父缓存schema仍包含训练来源wrong-Haze图，但本诊断不使用它指导或评分。

## 检查顺序

### 1. 零引导一致性

同一图片/seed运行原`restore()`以及新的可微入口。
量化输出必须同时逐像素等于原入口和已有缓存proposal。
新入口使用`no_grad`保存前缀，绝不把inference tensor直接拿来求导。
冻结权重；整模型及block的non-reentrant checkpoint只用于节省激活显存。

### 2. 梯度数值核验

在零基index48，分别检查完整预览与真实两步rollout的输入梯度。
`z0_hat = z_t - t*v(z_t,t)`；真实rollout逐步Euler到t=0。
数值核验临时FP32模型/latent、关闭TF32，避免BF16量化破坏有限差分。
沿RMS归一化的分析梯度，用epsilon .001/.003/.01做中心差分；
每种梯度至少2个epsilon相对误差<=.20且方向为正才通过。
这是数值工程容差，不是论文精确性证明；失败保存诊断并停止，不自动松门槛。
恢复各参数/buffer原dtype及TF32后才运行混合精度oracle arms。

### 3. 有限GT引导诊断

| 项目 | 固定设置 |
| --- | --- |
| 预览窗口 | index40/46/48，即尚余10/4/2个Euler transitions，记录真实t |
| 梯度 | 完整预览；skip-J预览近似；index48额外真实两步终点梯度 |
| 修正次数 | 每个arm只改一次当前latent，然后重新算velocity、完成剩余生成 |
| latent修正RMS | .005 / .02 / .08；不是把旧pixel-energy步长搬过来 |
| 目标 | MSE / 同窗口无引导连续终图MSE；或再加LPIPS / 对应LPIPS |
| 连续代理 | 保留alpha=.30和clamp，但去掉两次round；不声称等于真实量化目标 |
| 最终评价 | 恢复原两次round，PSNR和Alex LPIPS分别报告 |

总共42个GT引导arm，加原BF16基线及3个FP32 latent无引导control。
所有arm和负收益保留；不依赖GT选优后的均值隐藏失败。
同时报告相对历史基线和相对同窗口精度control的增益。
本轮不是同算力TTS科学比较：将来energy需要正式同算力无energy对照。

直接GT目标替代尚未学出的energy，因此VLM等级只记录身份，未用于引导。
不能把本轮结果称作VLM贡献、energy学习、泛化、自验证或最终科学gate通过。
目标+.30dB及LPIPS相对下降.09928107050377874仅检查训练诊断空间。
这只是有限搜索，不是真正全局上界。成功不保证能学；失败不证伪全部路径。

## 运行与预算

更新：单节点同型号4/8×H100/H200。固定8张，4卡每rank顺序两张，8卡每rank一张。
每张使用独立模型生命周期，结束后回收显存；不增大单卡batch，无跨节点。
预算cap为4卡3普通小时、8卡1.5普通小时，均12 GPU-hours，额外最多30秒清理。
此处没有编造吞吐/ETA；完成后用launcher实际秒数和峰值显存替换估计。
报告模型forward、物理block调用（含checkpoint重算）、目标backward与VAE解码，
原始restore额外50个forward和encode/decode单列。LPIPS/其他操作的成本包含在实测时间内。
不自动启动energy或长训、不覆盖旧结果、不杀已有其他任务。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/start_foundir_flux_tail_oracle_v1_tmux.sh mechanism_4gpu_v1 4
```

八卡可用`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`、fresh ID和第二参数`8`。
通用run入口为`smoke/run_foundir_flux_tail_oracle_v1_gpu.sh`；旧`_8gpu.sh`仅兼容wrapper。
日志：`logs/foundir_flux_tail_oracle_v1_mechanism_4gpu_v1.log`。
输出：`checkpoints/foundir_flux_tail_oracle_v1/mechanism_4gpu_v1/`。
每条保存独立sample000–007.json，rank文件仅记录分片/耗时，不能把rank数当样本数。
结束须有完整8条、全部46个arms、梯度/基线通过及`TAIL_ORACLE_RESULT_VERIFIED`，
不能仅凭tmux创建或退出码0宣布成功。中断保留partial，重试必须fresh ID。

## 本地验证

- 权重/缓存/源码/训练来源元数据预检已通过，images_decoded=0。
- Python语法与shell语法检查已通过。
- 当前专用环境没有pytest；测试采用stdlib unittest，不更改环境安装依赖。
- 8项CPU测试通过（1.769秒，不含Python冷启动）：官方CFG零引导/切段一致性、
  时间符号与完整/skip Jacobian、真实rollout梯度、有限差分/下降方向、alpha梯度、
  真正FLUX类小尺寸双流/单流blocks的checkpoint前后输出及梯度一致性、
  配置防越界、报告保留失败arm及独立精度对照。测试没有用真实图像。
- GPU一致性、有限差分和显存验收仍待实际节点运行，不能由CPU测试代替。
