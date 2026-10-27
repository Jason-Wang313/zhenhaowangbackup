# MoE：同节点、单变量的 Blur 感知损失续训对照

## 用户决定与当前状态

用户批准各续训1,000步的对照，随后明确要求两组在同一节点。
采用**同一节点、顺序执行、每组独占8卡**，不是两组各占4卡，也不是跨节点。
先control，再blur_lpips4x；第二组重新加载原step10000，不继承第一组输出。
本轮不启动PPT、不改VLM、不重训10,000步、不自动扩展。
Codex环境没有CUDA设备和GPU远程执行通道；准备完毕不等于GPU已启动。

## 为什么值得做，但不能保证成功

旧普通MoE已完成10,000步。Rain100为
`34.73973430724847 / .06654019097797573`；完整Blur2595为
`26.655435556096634 / .23083898932184047`。
Rain通过34.2/.08；Blur通过PSNR25.9，但LPIPS没有达到.18。
直接固定正确Blur专家也不能明显修复感知质量，所以本轮不再单独改router。

已完成诊断对每臂16个Blur训练来源、三个代表性共享/Blur专家张量的
逐样本加权RGB-MSE/LPIPS梯度范数比值取中位数：

| 参数 | 普通MoE | VLM版 |
|---|---:|---:|
| block0共享attention LoRA-B | 11.577761575667346 | 10.49906262468982 |
| block0 Blur专家输出矩阵 | 12.346233757934643 | 10.53005614372194 |
| block4 Blur专家输出矩阵 | 11.485799984847686 | 9.979495521884944 |

这是抽样参数的原始梯度，不是全网络Adam实际更新占比。
高通项与LPIPS方向经常相反，但高通项梯度本身较小，不能宣布它是唯一原因。
四倍LPIPS是预先冻结的保守干预，不是由开发集搜索出的最佳权重，
更不是“梯度相差12倍就必须把权重乘12”的推论。

## 唯一改动

| 配方 | Rain LPIPS权重 | Blur LPIPS权重 |
|---|---:|---:|
| control | .05 | .05 |
| blur_lpips4x | .05 | .20 |

两组都保留原架构、image-only router、共享LoRA、训练数据、裁块、D4变换、
学习率、AdamW设置、其他损失及每步16Rain+16Blur。VLM投影仍不启用。
不加教师、refiner、身份损失、梯度手术或新数据。
Rain损失没改，不代表共享参数不会受Blur影响，必须继续监测Rain。

## 真正续训，不是只加载权重

共同父状态：
`checkpoints/foundir_rain_blur_joint_vlm_moe_v2/image_only/formal_v1/step_010000.resume.json`

- receipt SHA：`d16c0d0a66508b5227eeaeaf378ea667bd330db688ae86dc4a2ccb1071b27b4d`
- 权重 SHA：`76c2accf2274099320b913e95c7898b076714335b32579f7c30be0b9074c730b`
- Adam SHA：`22f2a517d7b610c409667e76796c58a35579d18c2f8a2b1be65c57e80e3062b2`

严格恢复全部可训练参数、Adam一二阶动量、step、参数顺序、每rank随机状态。
从样本位置320000开始，结束于352000，各增加32,000次曝光、每任务16,000次。
不重置优化器、学习率或路由warmup。
开始评估/构建LPIPS后恢复训练随机状态；后续评估前后也保存/恢复随机状态。
采样记录逐rank计算摘要，两组结束时逐一核对。
训练字节继续使用原loader在解码前核验；GPU初始化还核验全部开发图像SHA。
受保护像素始终禁止，CPU预检查只读元信息和模型文件、不打开图片。

## 冻结验证与判断

- 开始：同一父状态的Rain100/Blur100预览，逐图指标核对一致。
- 第10500与11000步：完整Rain100、完整Blur2595；原512裁块、量化和LPIPS不变。
- 固定11000为最终判断，不挑中期最优，不用开发集改变权重或阈值。
- 逐图配对比较，按来源组bootstrap报告区间；这是单seed数据不确定性，不是多seed训练不确定性。
- 继续研究的信号：修改版Blur LPIPS优于control且95%来源bootstrap区间上界<0，
  同时Blur PSNR>=25.9、Rain PSNR>=34.2且LPIPS<=.08。
- 完整双任务达标另报：还要求Blur LPIPS<=.18。改善但未到.18不能说达标。
- 没有“自动扩大训练”；最终结果需要用户判断。这一实验也不证明VLM必要性或论文创新。

## 入口、资源与产物

一台空闲同型号8×H100/H200，预计每组1.0–1.5普通小时/8–12 GPU-hours，
顺序合计2.0–3.0小时/16–24 GPU-hours（参考原2.57秒/步，包含启动与验证的暂估）。
两组全程持有同一节点锁，禁止与PPT/旧Rain/Blur任务抢卡。
绑定hostname及8张实体GPU UUID；检测到换节点或换卡就拒绝第二组。
第一组发生工程故障就不启动第二组；画质未达标本身不会取消预先确定的对照。
使用有界NCCL通信检查、空闲显存检查、tmux及新ID防覆盖。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_moe_blur_lpips_pair_v1_tmux.sh pair_v1
```

日志：`logs/foundir_moe_blur_lpips_pair_v1_pair_v1.log`

结果根：`checkpoints/foundir_moe_blur_lpips_continuation_v1/pair_v1/`
分别保存`control`/`blur_lpips4x`的10500/11000权重、Adam与恢复凭证，
每个里程碑逐图结果、初始化指标、来源与采样摘要；最后生成`comparison.json`。
新配置SHA：`f9840280eabc089567168ab55d7ce8f6431772f6fcb94ddbd208a3ff16449d20`。
全部新代码独立新增，旧MoE/PPT绑定源文件没有修改。

## 验收记录

7项CPU测试通过：只修改Blur LPIPS、实际损失梯度差、完整Adam/RNG恢复后下一步与
不中断训练完全一致、采样位置、顺序八卡入口、信号与达标区别、不同节点/状态拒绝。
Python AST/JSON及两个shell脚本语法检查通过。
真实父权重/Adam/29项旧来源绑定、冻结FLUX/VAE/context、训练/开发元信息预检查通过。
预检查未打开图片；训练池Rain38,334条/5,780来源，Blur105,057条/7,339来源，
仍只续训32,000次总曝光，不能说这一轮遍历了全部训练数据。
CUDA实际运行尚未验证；没有产生新画质结果，本轮GPU消耗为0。
