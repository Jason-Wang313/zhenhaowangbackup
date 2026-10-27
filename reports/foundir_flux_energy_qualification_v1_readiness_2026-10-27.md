# FLUX proposal → energy-only：25步工程验收入口

## 结论

代码、CPU测试和真实文件身份预检完成，**GPU验收尚未运行**。
这是新FLUX起点，不是旧Restormer实验的续训，也不是已验证有效的energy方法。
只有未来开发集证明energy判断可靠、像素修正损坏纹理限制收益，才考虑FLUX末段采样引导。
未来该对照需匹配总计算预算，计入energy反传与VAE开销；只匹配采样步数不够。
当前不实现或启动采样引导，不启动MoE，不自动formal。

## 本轮实际做什么

1. 验证单节点同型号8×H100/H200、空闲显存、同节点任务锁、没有旧PPT训练进程。
2. 单GPU离线BF16解量化现有Qwen3 FP8文本编码器，生成精确Noise/Haze提示词缓存。
   这不是训练Qwen，也不联网拉取FP8 kernel；Rain提示词缓存不能替代Noise。
3. 文本编码器进程退出后，八卡各处理四个训练来源，生成512×512的真实FLUX输出。
   Noise step400为baseline；Haze step3000为真实错误专家负例。两者用同一Noise输入。
4. 所有FLUX/AE进程退出后，仅训练energy。32个来源、每步global batch32、25步。
   每步从缓存同步裁256并做D4变换，最终在原512裁图上检查推理和指标。
5. 保存并回读energy/Adam checkpoint，检查有限梯度、八rank全部参数一致、512推理、完整结果。
   launcher不仅检查进程退出码，还要求25步结果文件和engineering_pass。

训练候选包括GT、原始degraded observation、冻结Noise FLUX proposal、
不足恢复、过度恢复、人工低通detail-damaged，以及真实wrong-Haze output。
GT仅用于训练监督/离线评分；推理trajectory和selector签名不接收GT。
同一个energy负责完整1/2/3步梯度修图和选择起点/三步结果。
没有额外residual Restormer、直接refiner或七选一alpha selector。

## 固定基线与指标归属

- FLUX Noise LoRA：step400，SHA-256
  `1b4ed05ad5ad403a3d7fcc633eca97cdb69d00454e357da0c3d3131ac2c3ff1a`。
- 保留50步、guidance1、reference time10、原seed命名空间；先把raw FLUX量化，
  再固定alpha=.30混合并再次量化。该算术处理不是额外Restormer。
- Haze训练负例SHA-256：
  `4d94ecf6dd902704345e28ae0bf3447d7e172235158c2b2177933794fd973e00`。
- 新入口与旧原生参考图推理函数在合成替身上逐项一致，包括随机输入、context、schedule、输出；
  这不等于已经在GPU复现历史1859条指标。BF16文本编码和batch-one数值也需实测。
- 本轮两臂使用同一真实缓存，指标都先量化uint8。额外报告observation-only，防止隐藏简单identity对照。
- 最终科学门槛仍是energy独立增加PSNR至少0.30dB，LPIPS至少降低
  `0.09928107050377874`。不能把旧Restormer收益算给energy。
- 本轮只报告训练集工程检查，不评估科学门槛。正式开发比较仍要求1859条/34来源/512匹配协议。
  本轮缓存/权重均不得拿去充当正式泛化结果，qualification权重不能seed formal。

## 数据与可复现性

VLM导出表没有尺寸；入口与原始pairs表逐条核对record、split、source、路径、输入/目标hash，
再补width/height。元数据预检确认54585条Noise训练记录、1755个来源全部满足512尺寸。
按固定哈希排序选32个不同训练来源，每来源一条，不按GT质量挑选，不resize。
所有训练来源及GT hash与heldout元数据排重；预检没有打开任何图片。
GPU阶段只允许selected training Noise记录的像素进入缓存；读图前验证hash和原始尺寸。
缓存绑定原图hash、位置、seed、LoRA、提示词、配置，后续载入再次验证。
每次运行保存本地源码依赖、配置和权重身份；不覆盖已有run ID。

配置：`configs/foundir_flux_energy_qualification_v1.json`。
当前配置SHA-256：`0b5465cebc289f5cb4b59c87120e838fe77393aed92aca71b4d7f05508a3debf`。
推理/训练：`smoke/qualify_foundir_flux_energy_v1.py`。
身份与数据隔离：`smoke/flux_energy_qualification_contract_v1.py`。

初始化修正只将内部NAF beta/gamma由0设为.1，打开非线性特征路径；最终head仍为0。
CPU测试证实无额外随机数消耗、无参数结构变化、初始输出严格identity、
head混合导数依赖输入。不能据此声称真实画质会变好；VLM调制也仍需要学习。
损失中的no-harm是惩罚项，不是逐图PSNR/LPIPS永不下降的保证。

## 已验证与未验证

- 15项CPU单测通过：相对门槛、配对身份、来源隔离、缓存拒绝、同步裁图、
  双重量化、原推理函数等价替身、零输出初始化、非线性启动、完整三步反传。
- 合成CPU双rank Gloo通过：2步、每rank累积2次、二阶梯度有限、所有训练参数跨rank完全相同。
- 真实模型/manifest/LoRA文件hash和元数据预检通过；未解码真实图像。
- shell语法通过。没有GPU训练结果、没有质量改善证据、没有energy可靠性证据。
- 不确定项：实际Qwen/FLUX环境兼容、GPU峰值显存、候选生成吞吐和完整LPIPS训练成本。
  这些是本次GPU工程验收要回答的问题，不可提前写成通过。

## 时间和启动

单节点8×H100/H200，保守初估0.5–1.5普通小时，4–12 allocated GPU-hours。
这是新流程估计，非实测；启动/文本缓存/模型加载和能量训练都计时。
完整缓存与正式训练预算待本轮吞吐确定，不能沿用旧Restormer速度，不能直接批准100k。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_flux_energy_qualification_v1_tmux.sh qualification_v1
```

日志：`logs/foundir_flux_energy_qualification_v1_qualification_v1.log`。
结果：`checkpoints/foundir_flux_energy_qualification_v1/qualification_v1/result.json`。
tmux session创建不代表训练已开始；最终必须见`FLUX_ENERGY_RESULT_VERIFIED`。
若旧进程/显存占用检查失败，不擅自kill，由用户确认后处理。

## 旧任务状态更正

旧`stage1_retry1`最新只记录到580/1706步，随后在2026-10-27T21:55:21Z
写`PPT_FULL_DATA_LAUNCH_END status=0 elapsed_seconds=13182`。
没有1706完成标记或最终验证，因此只能记为未完成，不算科学成功。
launcher时间3.661667普通小时、29.293333 allocated GPU-hours；停止原因未知。
未核验旧GPU PID，未替用户终止或恢复旧任务。保留历史数据与被替代草稿。
