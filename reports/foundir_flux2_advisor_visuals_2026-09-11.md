# FLUX.2 sealed internal advisor visual package（2026-09-11）

- 内容：Blur、Noise、Haze、Rain、Low-light 各 2 张，共 10 张 triptychs。
- 排列：左侧 Input (LQ)，中间 FLUX.2 output，右侧 Ground Truth。
- 来源：FoundIR training archive 划出的 source-group-disjoint sealed internal；不是 official FoundIR test。
- 复用现有 outputs；没有重新 inference，也没有访问 official test。
- 选样：每类 60 张按 output PSNR 排序，固定选择约 1/3 与 2/3 分位，并要求不同 source group；不挑 best cases。
- 输出目录：`reports/advisor_visuals/flux2_v3_sealed_triptychs_2026-09-11/`
- 完整 provenance 与每张指标：输出目录内 `manifest.json`。
