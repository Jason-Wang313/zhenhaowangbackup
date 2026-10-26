# Rain＋Blur 普通 image-only DiT-MoE：启动前协议

## 当前结论

用户已批准执行先前讨论的普通 MoE 路线。实现、数据隔离、CPU 测试及真实文件预检已完成；
**尚未启动 GPU，也未证明 MoE 可以同时完成两类恢复。**
下一步只运行 25-step 工程资格检查，不自动正式训练，不接 VLM。

已有 Rain checkpoint 保留：step 4,000，固定开发集
34.22743253549469/.07367917290423065。旧 Blur expert 及其 50-step 推理完全保留。
旧 raw-FLUX 单 MLP MoE 已失败，不再使用其权重或重复其训练配方。

## 从哪里开始、具体改哪里

1. 加载同一个 FLUX.2 Klein 4B 和已通过门槛的 Rain LoRA，冻结它们。
2. 在五个 double-stream block 的图像 MLP 位置，每层放两个普通完整 FFN expert。
   每个 expert 从该层原模块复制；不是两个完整 FLUX，不是 LoRA expert bank。
3. 每层 router 只读取当前图像 token 的均值特征，输出两个 softmax 权重。
   在该层内部按权重合并 expert 输出。没有另一个外部 router 或第二套 MoE。
4. 所有新 expert/router 参数与 Adam 状态用 FP32，模型矩阵运算使用 BF16。
   对 expert FFN 做 non-reentrant activation checkpointing，控制激活显存。
5. 保持同一个一步恢复端点：t=1，anchor noise=0，FP32 latent subtraction，直接 RGB 监督。

初始两个 expert 完全相同，soft mixture 必须保留原 Rain 输出。实现用
`E0 + w1 * (E1 - E0)`，与归一化加权和等价，也避免初始相同分支因求和舍入造成偏差。
CPU FP32/BF16 精确等价已通过；真实 8-GPU 上的原 Rain 逐图复现仍待资格检查。

**容量不再是旧单层小实验：**MoE 可训练参数 853,313,290，dense 对照 424,673,280。
其余共享 FLUX/Rain LoRA 固定。checkpoint 含完整专家，不只是小 LoRA：
MoE FP32 权重约 3.41 GB，Adam 两组动量约 6.83 GB（十进制近似），
文件写入与哈希时间必须纳入预算。显存是否安全需要在实际 H100/H200 验收。

## 两个容易混淆的地方

**这不是直接混合旧 Blur 和新 Rain 的 LoRA。**旧 Blur 学的是 50-step 推理，并有 .75
输出修正；新 MoE 采用统一的一步恢复，必须在新的专家模块里学习 Blur。
本轮不加载旧 Blur LoRA、不重训旧 Blur 模型、不把旧 Blur 的成绩写成新 MoE 成绩。
Blur 直接用允许训练的数据 GT 监督，不额外生成 50-step teacher 数据。

**固定 prompt 不是每图的退化标签。**为了保留 Rain 起点，两类图都用同一个已冻结的
Rain prompt context。不会根据类别切 prompt，模型不会通过 prompt 得知真实任务。
这意味着保留了固定 Rain 文本先验，是当前基线的限制，不应隐瞒。
未来 VLM arm 必须用同一个固定 context，才是匹配比较。运行时不加载 Qwen 编码器。

## 如何让专家和 router 学起来

- 训练初期用已有训练标签：Rain 更新 expert 0，Blur 更新 expert 1；同时用分类损失训练
  图像 router。两个 expert 都参与计算，未选分支有零梯度，避免 DDP unused-parameter 问题。
- 之后从标签指定权重平滑切到图像 router 权重，再进行联合训练。
- 正式 hard warmup：前 250 步；到全局 step 500 后只使用图像预测权重。
- 25-step 资格检查采用 hard6/transition-end12，短程覆盖三个工程阶段；这不是正式效果实验。
- router 辅助分类权重 .01；训练标签仅用于训练监督，所有验证/部署路径禁用训练标签。
- 不强迫每张图 50/50 使用两个专家。记录每类图、每层的权重，再做固定 expert 0/1 的输出诊断。
  这些诊断不使用 GT 选择结果，不可把它们的逐图最佳值当模型性能。

学习率：expert 1e-5，router 1e-4；AdamW weight decay=0，梯度裁剪 1。
恢复目标沿用成功 Rain 的八项损失及系数，唯一任务限定为：
**冻结 Restormer 辅助 teacher 只用于 Rain，Blur 的 teacher 权重为 0。**
不会把 Rain teacher 的错误去雨行为施加到 Blur。两类仍有 RGB MSE、Charbonnier、LPIPS、
latent、细节与未变化区域约束；没有额外 pixel refiner。

## 数据与验证

| 项目 | Rain | Blur |
|---|---:|---:|
| 可用训练记录 | 38,334 | 105,057 |
| 训练来源组 | 5,780 | 7,339 |
| 每个 optimizer step 的样本数 | 16 | 16 |
| 正式 3,000 步的样本呈现次数 | 48,000 | 48,000 |
| 中间开发评估 | 原固定 100 条 | 固定选取 100 个来源的预览 |
| 正式最终开发评估 | 原固定 100 条 | 原完整 2,595 条 |

两类使用全部允许训练池、按来源均衡采样，随机对齐 512 裁剪和 D4 增强，不缩放。
尺寸不足用原 helper 的反射 padding。96,000 次呈现不是完整遍历 143,391 条记录。
所有训练数据与两类开发集检查 source group 和 GT SHA-256 隔离，交叉排除数为 0。
训练图像首次解码及 stat 改变后校验 SHA；不缓存解码后的全训练集。
两个 spawn worker／卡，prefetch=2，global batch32，四次累积。

Blur 的所有 2,595 条评估裁剪沿用原 `2026091111:full_development_crop:record_id` 规则；
Rain 沿用原 manifest 的固定坐标。指标在 round/clip uint8 等价结果上计算。
没有 official、sealed、internal、parent-test 像素参与。

## 必须通过的工程检查

更新参数前，重跑原 100 张 Rain，逐图对比 step-4,000 原结果：

- PSNR 最大误差 <=.0001 dB；LPIPS 最大误差 <=.00001。
- 平均指标依然满足 34.2/.08。
- 失败则直接退出，不在已损坏起点上训练。

25 步后：有限 loss/gradient、两类 expert/router 实际更新、FP32 权重和 Adam 文件、
开发集指标、正常退出及显存/速度记录。25 步只验工程，不要求 Blur 此时过线。

真实 GPU 初始化复现、DDP、峰值显存和学习效果目前都 **未验证**。
CPU 已完成 24 项测试（16 项新 MoE 测试＋8 项原精度/数据回归），30.212 秒，全部通过。
另行运行的双进程 CPU DDP 测试也通过，用时 9.200 秒：覆盖梯度累积、activation
checkpointing、固定到自动路由切换，以及两进程更新后参数一致性；不替代真实 GPU 验收。
涵盖初始化、checkpointing 梯度等价、训练/推理标签隔离、分支更新、权重重载、
平衡采样、spawn 加载、跨任务隔离、原 Blur 裁剪、Rain parity 拒绝退步和比较器。
完整真实资产/开发集字节预检返回 `rain_blur_image_moe_preflight_pass`。
Python 编译、配置 JSON 和两个启动脚本的 shell 语法检查均已通过。

## 正式实验及科学判断

拟定且已配置：fresh 3,000 optimizer steps；在 0/1,000/2,000/3,000 验证。
正式必须读取同 arm 已通过的资格检查结果，并重新从成功 Rain checkpoint 初始化；
不使用资格检查的专家权重或 Adam。MoE 和 dense arm 独立运行，各需自己的资格检查。
没有自动继续、自动另一 arm、自动 VLM。当前启动器不自动恢复部分正式 run；保留中断产物，
需显式审查后实现同协议恢复，不能改 run ID 把重新开始说成续训。

数值门槛：Rain >=34.2/<=.08；完整 Blur >=25.9/<=.18。
逐图与尾部退步需一起报告，不能复用旧 Blur “平均过线”等同“尾部安全”的误述。
只有普通 MoE 有效后，才考虑相同 MoE 加 VLM。

对照：同 Rain 初始化、数据、步数、端点、恢复损失训练一个每层只有单 FFN 的 dense 模型。
比较器 `smoke/compare_foundir_rain_blur_image_moe_v1.py` 检查逐图对齐后分别报告两类结果。
**此对照没有匹配参数量/FLOPs，不能把胜出单独归因于 routing，更不能据此声称论文创新。**
还必须检查 router 是否区别任务、固定 expert 诊断是否显示实际分工。

## 启动命令与预算

在一个同型号 8×H100/H200 节点，只启动 MoE 资格检查：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_blur_image_moe_v1_tmux.sh qualification moe qualification_v1
```

tmux：`rain_blur_moe_qualification_v1`，server `zhenhaowang_jobs`。
日志：`logs/foundir_rain_blur_image_moe_v1_moe_qualification_v1.log`。
输出：`checkpoints/foundir_rain_blur_image_moe_v1/moe/qualification_v1/`。
同节点锁与日志独占创建防止重复运行；发现已有 run 必须保留并检查，不覆盖。

资格检查暂估 **.25–1.0 墙钟小时 / 2–8 allocated GPU-hours**。
这包含资产预检、完整 Rain 复现、25 步、较大 checkpoint 写入和评估诊断，非纯训练耗时。
正式时间必须据资格检查实际速度、显存和 checkpoint I/O 重估；不沿用 Rain LoRA 的 31 分钟。
Codex 环境没有 GPU，本轮没有代为启动训练。

配置 SHA-256：`8988e2374ad736f3e406945f1d9db83ae14587d0aba28a04dbf9cdc900ab5f12`。
父 Rain checkpoint SHA-256：`020113ce209728452917d49859da7be220f638f0d27f6f87f182ffdab0a8fc2e`。
数据 digests、代码身份与全部配置会写入每个 run 的 receipt；权重与优化器留在项目磁盘，不上传 GitHub。
