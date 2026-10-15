# FoundIR 五类 restoration：advisor status（2026-09-10）

> **SUPERSEDED / 非 FLUX.2 主线结果（2026-09-11）：** 本报告描述的是 released
> FoundIR generalist/specialists + Qwen routing diagnostic，不是 FLUX.2。
> 它不得作为当前主系统、FLUX.2 final metrics 或 SOW final result 对外汇报。
> 当前主线状态以 `reports/advisor_governance/` 中由 FLUX.2 sealed failure
> 生成的 advisor gate report 为准。

## 可直接发送的简版

老师您好，我们完成了 FoundIR 五类 restoration 的大规模 development audit。当前系统采用 Qwen 输出的 degradation level 做 category-blind routing，ground-truth task label 不参与路由；后端组合 released FoundIR generalist、Blur specialization、Low-light/Weather specialist，并加入了针对 Qwen Low-light 漏检的 image-luma rescue rule。

在 source-group-disjoint development 上，五类 isolated records 共 8,269 张（495 个 source groups），按 SOW 的 mean PSNR/LPIPS 口径目前 10/10 checks 全通过：

| Task | Records / groups | SOW target（PSNR / LPIPS） | 当前 integrated development | Margin | SOW |
|---|---:|---:|---:|---:|---:|
| Blur | 2,595 / 165 | >=25.9 / <=0.18 | 28.4863 / 0.17118 | +2.5863 / 0.00882 | 2/2 |
| Noise | 1,859 / 34 | >=38.9 / <=0.10 | 42.0503 / 0.05749 | +3.1503 / 0.04251 | 2/2 |
| Haze | 1,580 / 158 | >=22.1 / <=0.19 | 23.3862 / 0.14758 | +1.2862 / 0.04242 | 2/2 |
| Rain | 835 / 117 | >=34.2 / <=0.08 | 38.5524 / 0.07066 | +4.3524 / 0.00934 | 2/2 |
| Low-light | 1,400 / 21 | >=18.9 / <=0.28 | 26.1995 / 0.11610 | +7.2995 / 0.16390 | 2/2 |

结果演进为：最初 660-record Qwen-routed v1 仅通过 5/10 SOW checks；修正 Blur/Low-light/Weather routing 后，v2 通过 8/10；当前大规模 development candidate 达到 10/10。当前结果仍不是 final test：luma threshold 是在 development 上选择的，sealed internal 与 protected official test 均未访问。

我们另外设置了比 SOW 更严格的 internal safety gate：mean target 比 SOW 提高约 1 dB、LPIPS 收紧 0.02，并要求 95% source-group bootstrap bound。该 gate 目前只有 Haze 全通过。主要风险是 Blur 和 Low-light 的跨 source-group tail，而不是 SOW mean：Blur 的 group-balanced bootstrap 为 PSNR lower 24.3170、LPIPS upper 0.31677；Low-light 为 17.6946、0.28477。Noise task-native checkpoint 已通过全部 strong gates（42.5805 / 0.06530；bootstrap 40.8793 / 0.06979）；Rain task-native checkpoint 为 42.2843 / 0.04745，只有 bootstrap LPIPS upper 0.07038 比 0.07000 高 0.00038。这两个 task-native checkpoint 尚未接入 final deployable router。

想请您建议三个决策：

1. SOW mean 已在大规模 development 上 10/10 通过的情况下，是否应现在冻结 architecture/router/checkpoints，并只运行一次 sealed internal；还是先要求所有 task 通过我们额外设置的 bootstrap safety gate？
2. Blur 虽通过 SOW mean，但 source-group tail 较弱；是否值得继续投入 task-native deblurring capacity，还是先以当前结果完成一次冻结评估？
3. Noise/Rain isolated specialist 明显更强，但接入会增加 routing complexity；您倾向先接入并重新做完整 development，还是保留当前已 10/10 通过的较简单系统？

## 实验设置摘要

- Evaluation：只用 FoundIR-training-derived、train/source-group-disjoint development；deterministic 512 crop；保存后重新读取 uint8 RGB；报告 PSNR 与 offline AlexNet LPIPS；bootstrap 以 source group 为 resampling unit，5,000 次、95% interval。
- Routing：使用 frozen actual Qwen degradation levels；ground-truth category 只用于 inference 后 aggregation，不控制 route。当前 base 为 Blur `L4+` 时选择 Blur specialization，否则 generalist；Low-light `L3+` 优先触发 Low-light specialist；Haze `L2+` 触发 Weather specialist；Rain 本身不触发 Weather。新增 rescue 为 category-blind rule：Qwen Low-light `L2` 且 crop mean luma `<=0.20`，以及极暗 `L1` fallback。
- Blur：released FoundIR residual-diffusion U-Net specialization，8 GPUs、global batch 32，development 选择 step 3,000。后续 Weather-Restormer-initialized Blur refinements 虽能降低 LPIPS，但牺牲 PSNR，未被采用；failure audit 指向 task/domain mismatch，而非单纯 loss weight 不够。
- Noise：Restormer，以 released real-denoising weights 初始化；4 GPUs、global batch 32、AdamW、LR `5e-5`、RGB L1、BF16、progressive crops `128->192->256`，训练到 3,000 steps，按 worst normalized strong-gate margin 选择 step 250。
- Rain：Restormer，以 released deraining weights 初始化；从 4 GPUs step 1,500 无损迁移到 8 GPUs，global batch 始终为 32；AdamW、LR `1e-4`、RGB L1、BF16、progressive crops `128->192->256`，训练到并选择 step 5,000。
- Low-light rescue gate：在 16,461-record development union 上复用 immutable baseline outputs，只对预声明的 1,076 个 counterfactual candidates 重新 inference；最佳 rule 改变 934 routes。该 rule 提升 Low-light tail，但尚未通过额外 strong bootstrap gate。
- Data/seed policy、optimizer state、LR absolute-step schedule、loss、checkpoint selection 与 evaluation protocol 均在运行前冻结；4-to-8 GPU Rain migration 未改变 global batch 或 optimizer semantics。

## 口径与 provenance

- 上表是当前可部署形式的 development-selected routing candidate，不应称作 official/final test result。
- Noise/Rain task-native 数字是 isolated capacity evidence，不应与尚未接入它们的 integrated routing 数字混写。
- 当前 development candidate result SHA-256：`f98137a767b6991c72cc064ceaaa678ca001c19ca0e1d239f8459b5410acc9c0`。
- 13,767-record Blur/Haze/Low-light audit SHA-256：`ec308c3ee4bd8e74a9ad0eb16f96ed5bebba186356ce8e3aa084e5a99210a7d3`。
- Noise selection SHA-256：`e03c9e58e7873e3bfd65d210964f66466f547b4f05aa0421fcef1d2ec0c84754`。
- Rain selection SHA-256：`924f64288198d6ebc1aa8b3ef8b744865f260d679b989633b3544a43dce70bc4`。
- sealed internal access：0；protected official FoundIR test access：0。
