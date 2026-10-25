# Rain FLUX：从 step 3,000 真正续训到 4,000

## 决策与边界

用户已明确授权增加 1,000 步，数据、损失和学习率全部不变。
原正式训练完成，开发集为 **34.00838778684664 PSNR / .07343229320365935 LPIPS**。
PSNR 距 34.2 还差 .19161221315336263 dB；LPIPS 已低于 .08。
最后 1,000 步仍提升 .384159465303874 dB，因此本次只检验继续优化是否有效，
不新增结构、不换损失，也不承诺一定过线。

当前状态：代码、CPU 测试和真实资产预检完成；Codex 无 GPU，尚未启动续训。

## 实际恢复了什么

- 恢复同一个 full-data `formal_v1` 的 step-3,000 LoRA：100 个 FP32 张量。
- 恢复全部 100 组 Adam 一阶／二阶动量及 step=3000；学习率仍为 1e-5。
  不以新优化器冒充续训。参数顺序由冻结的原 FLUX 架构及 LoRA 注入代码重建，
  检查名称、形状、状态计数、数值有效性及原源码身份。
- 数据从零起算位置 96,000 接到 127,999：新增 32,000 次呈现，累计 128,000。
  相同随机种子与绝对位置决定 source、record、裁剪及 D4 增强，不重启采样流。
- 保留 38,334 条训练记录／5,780 个来源、512 crop、batch32、四次梯度累积、
  两个预取 worker／卡、FP32 LoRA/Adam 与端点减法、BF16 模型运算。
- 直接复用原八项损失及系数、冻结 teacher、推理函数和评估函数。
  部署仍只有 FLUX LoRA，不增加 Restormer 推理支路。
- 只在全局 step 3,500 和 4,000 保存并验证：相同 100 条开发记录和 32 条训练探针。
  不重复 step-0 验证，不另跑短训练，不自动继续到 4,000 之后。
- 新目录保存本次结果，旧配置、训练器、数据辅助模块、checkpoint 和日志全部不变。
  报告中的 unique counts 仅统计本次新增曝光，不能冒充累计去重数量。

旧 checkpoint 没保存全局 RNG 状态。当前训练没有随机 anchor noise，LoRA dropout=0，
VAE 编码取 mean，数据增强使用由绝对位置构造的局部随机数生成器。
因此可恢复所需的数据轨迹和优化器状态，但**不宣称不同 GPU／不同启动之间逐位等同于不中断运行**。

## 验证记录

指定项目 Python 环境下执行：

```bash
OMP_NUM_THREADS=2 /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python -m unittest tests.test_foundir_rain_precision_resume tests.test_foundir_rain_precision_full tests.test_foundir_rain_precision_fit -q
```

最终版本 **15 tests passed，12.361 秒**。覆盖：

- 小型 LoRA+Adam 保存恢复后，后续三次 CPU 更新与不中断训练逐张量完全相同。
- 真实 100 组 Adam 状态加载后逐张量相等；step、LR、所有动量保持不变。
- 错误配置／step／LR、缺失动量、非有限数、错误形状和错误参数索引被拒绝。
- 新增全部 32,000 个分布式数据位置与原序列切片相同；无缺漏、无位置重复。
- 双 worker spawn 加载与不中断序列的 crop／augmentation 张量完全一致。
- 原数据划分、FP32 参数更新、完整损失反传及训练／推理端点测试仍通过。

完整只读 `--preflight` 返回 `rain_precision_resume_state_pass`：原模型资产、
全训练文件路径、固定探针／开发集图片校验、LoRA、Adam、父结果及代码身份全部通过。
训练图像 SHA 沿用原协议：首次使用或文件状态改变时校验，不声称已逐一解码全训练集。
两个启动脚本通过 `bash -n`。未使用任何 official／sealed／internal／parent-test pixels。

## 启动与耗时

单机、同型号 **8×H100 或 8×H200**，不跨节点：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_flux2_precision_resume_v1_tmux.sh continue_v1
```

- tmux：`rain_resume_continue_v1`，server `zhenhaowang_jobs`。
- 日志：`logs/foundir_rain_flux2_precision_resume_v1_continue_v1.log`。
- 输出：`checkpoints/foundir_rain_flux2_precision_resume_v1/continue_v1/`。
- 看到 `RAIN_RESUME_STATE_RESTORED` 后，还要确认 `RAIN_RESUME_TRAIN` 的 step 3001；
  仅创建 tmux 不等于训练成功。重复启动会拒绝已有日志／目录；节点锁阻止同任务并发。
- 8×H100 估计 **40–50 分钟，5.3–6.7 allocated GPU-hours**，包括启动和两次验证。
  以实际吞吐更新，H200 不保证固定加速比。
- 结果仍用原联合门槛：PSNR >=34.2 且 LPIPS <=.08；工程结束不等于科学过线。
  没有后续用户决定，不自动延长、不进入 MoE。

## 冻结身份

| 资产 | SHA-256 |
|---|---|
| 本次配置 | `962ed5663e49ca59827f28f37260a75229085f8b49cb9e63ba03748a922638b9` |
| 本次训练器 | `f1b4dc375532af8e716e98ce362f0c691b448478881b50ccf8afe57d8fdad577` |
| 续训状态／数据辅助模块 | `f9d135a68faebace8464373bf3042bec4bf51d38225b438db46fa7b92dd1845b` |
| 父 full-data 配置 | `382bb767dfc6dae743f8c70d690f2307b640acc1eb1ff06d25ce557aef981237` |
| 父 step-3,000 LoRA | `336d412299cd7cd9ff919d8983b2b4c435105161496510c0cec8748785c15904` |
| 父 step-3,000 Adam | `8f45db7a0f700410bdef910898ce2b7588c8f6ec70aefa74aa099ddcf99bd711` |
| 父完成结果 | `59849092b5b4e609aaa5aef703537807b219a579142d86b18bdfc5c4e2875972` |
| 训练记录 digest | `0207fd65e5a37628023393a93bc9e80d1f1df85708d6677a94cc656dffea9fb0` |
| 开发集 digest | `1652d4e46634c1ee248e1984ea2aeaec271d8491a67dcc3acf12838cb181d783` |
| 训练探针 digest | `cfd29a617d50b25c87ba8d3e246566cd789fae412b057a677d2a6c43afdd969a` |
