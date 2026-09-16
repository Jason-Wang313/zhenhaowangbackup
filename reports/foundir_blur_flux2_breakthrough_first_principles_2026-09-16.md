# Blur 突破复盘：为什么复杂训练连续失败，而固定 residual blend 成功

## 最终突破

完整 2,595 张 source-group-disjoint Blur development 结果：

| 系统 | PSNR | LPIPS |
|---|---:|---:|
| 原始输入 LQ | 约 25.85 | 约 .291 |
| FLUX.2 step-10k | 25.730036 | .145439 |
| `LQ + .75 * (FLUX - LQ)` | **26.559141** | **.151993** |
| SOW 目标 | >=25.9 | <=.18 |

固定 alpha=.75 的零参数 correction 相对 FLUX 提升 `.829105 dB`，LPIPS
仅回退 `.006554`，89.56% records 的 PSNR 提升。两项 SOW 平均指标和 16
张固定视觉审计均通过。它也进入此前希望 teacher 达到的约
`26.4--26.9 / .14--.16` 区间。

## 真正的 epiphany

突破不是来自更复杂的 loss 或更大的训练，而是重新阅读两个已经存在的端点：

- LQ 的 PSNR `~25.85`，离 25.9 只差约 `.05 dB`，但 LPIPS `.291` 很差；
- FLUX 的 LPIPS `.145` 有巨大余量，但 PSNR `25.734` 反而低于 LQ；
- 目标 `25.9/.18` 位于两种系统能力之间，而不是要求从零创造一种新能力。

因此问题被从“怎样继续训练 FLUX 同时提高两个冲突指标”改写为：

> 是否可以在已经像素对齐的 LQ fidelity anchor 与 FLUX perceptual
> restoration 之间找到一个固定 operating point？

最小可证伪公式是：

```text
Output = LQ + alpha * (FLUX - LQ)
```

它没有新参数、没有 optimizer、没有新的 hallucination source，也不会改变原
step-10k checkpoint。alpha=0 是输入，alpha=1 是 FLUX；中间值直接探索两者的
Pareto trade-off。最终 alpha=.75 在完整 development 上同时跨过两个目标。

## 用户 prompt 如何触发这次改向

这不是某一句“魔法 prompt”，而是一串持续逼迫实验回到 first principles 的追问。
关键催化包括：

1. `how to make sure this can be achieved, ffs, i'm hopeless`
   - 迫使方案从开放式 architecture search 收缩为 fail-fast、可回退、最多两轮
     得出结论的验证。
2. `won't this be more risky?`
   - 迫使 correction 采用保留 step-10k fallback 的 identity/zero-risk 设计，而
     不是立即训练新的 restoration branch。
3. `so all the exp we did today and yesterday weren't even good enough ... what
   was the secrets of the first flux training`
   - 这是最关键的重新框定：不再默认“下一步一定还要训练”，而是审计为何最简单的
     原始 full-data flow 模型一直最好，并把 LQ 与 FLUX 的互补指标并排比较。
4. 用户反复要求 `first principles`、不要害怕执行，但也要求 brutally honest。
   - 促成了先做零参数 scalar feasibility、通过后只进行一次冻结 alpha 的全量
     confirmation，而不是再发明一个复杂可训练模块。

用户的贡献不是指定 alpha=.75，而是不断挑战“复杂度等于进步”的默认假设，并要求
解释为何已经花费多轮仍未越过 `.166 dB`。这个压力暴露了最简单而此前被忽略的事实：
我们已经分别拥有高 PSNR anchor 和低 LPIPS restoration，只需要组合它们。

## 为什么之前的复杂方法失败

### 1. 继续训练已经处于峰值的 formulation

完整数据 continuation 从 step-10k 的 `25.734236/.145406` 到 step-12k 的
`25.709168/.145691`，两项都略退化。原始 pure-flow full-data recipe 已接近其局部
最优；更多 steps 或 LR sweep 没有新信息。

### 2. Tiny-overfit 证明了 capacity，却没有证明 generalization

16-image RGB/trajectory gates 能明显提高 train PSNR，说明梯度链和模型 capacity
存在；但 source-disjoint development 出现 sharp-but-wrong 的边缘、文字和物体位置。
它们优化的是局部样本，而不是稳健恢复规则。

### 3. 更复杂 endpoint/spatial losses 改变了强模型，而不是保护它

512-source spatial-anchor candidate 相对同样被 continuation 损伤的 control 有
`+.678 dB`，但相对 matched parent 只有 `+.073 dB`，LPIPS 反而差 `.0127`，并有
`8.13 dB` worst-record regression。额外 loss 大部分只是在抵消继续训练造成的损伤。

### 4. 训练引入了不必要的自由度

当目标仅差 `.166 dB` 时，新的 LoRA updates、fresh optimizer、多步 trajectory
gradient 和多项 loss 都提供了足够自由度去破坏原本正确的像素。问题需要的是约束，
而不是更多 capacity。

## 为什么小 trick 有效

1. **直接利用互补端点。** LQ 提供几何和像素锚点，FLUX 提供感知清晰度。
2. **不产生新内容。** 输出严格位于两个已有图像之间，不会创造第三种生成结果。
3. **无灾难性遗忘。** 没有权重更新，原始 checkpoint 完全保留。
4. **与真实 inference 一致。** correction 直接作用于最终 50-step 输出，没有
   train/inference trajectory mismatch。
5. **问题维度从数十亿降到一个。** 唯一自由度 alpha 在 development 之前冻结，
   因果解释清晰。
6. **LPIPS margin 足够大。** FLUX 的 `.145` 距 `.18` 有余量，可以换取 PSNR。

## 不能抹去的 caveat

完整预声明 scientific gate 仍然失败：3/2,595 records 的 PSNR regression 超过
2 dB，最坏为 `12.569169 dB`。最坏 candidate 本身仍是很好的
`31.479966/.016354`，但 base 异常高达 `44.049135/.002379`；固定 blending 会损伤
这种 FLUX 已近乎完美的样本。

因此必须同时保留两个结论：

- **成功：** Blur development 平均 SOW 指标和视觉 gate 通过，达到理想 teacher
  平均指标；
- **未完全解决：** 固定 scalar correction 不是 per-image Pareto-safe，严格 tail
  gate 未通过。

不得用 development GT 继续调 alpha 来掩盖 tail failure。若未来获批处理 tail，方向
应是无需 GT 的保守 confidence/fallback 机制，而不是重新开始大规模 loss search。

## 可复用原则

在下一种退化或 All-in-One joint training 中，每次准备增加 architecture/loss 前，先问：

1. 输入、base model、目标分别在哪个 metric 上有余量？
2. 目标是否位于已有系统的可组合 Pareto 区域？
3. 能否先用零参数或 identity-preserving transformation 证伪？
4. 小样本提升是相对原 parent，还是只相对一个已经退化的 control？
5. 平均指标之外，worst-case 是否只是 delta 大，还是输出本身真的坏？
6. 最小实验失败后是否有明确停止条件？

本次最重要的教训不是“alpha=.75 永远有效”，而是：当强 baseline 只差很小 margin
时，先利用已有正确性并减少自由度，往往比继续训练一个生成模型更可靠。
