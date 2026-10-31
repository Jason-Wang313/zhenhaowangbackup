# Novel PPT Approach — FLUX-only 修订版（2026-10-31）

[`Novel_PPT_Approach_VLM_Restoration_Energy.pptx`](Novel_PPT_Approach_VLM_Restoration_Energy.pptx)
现为用户明确要求的 **FLUX-only重做方案**，不是仍未改动的原导师确认PPT，也不表示
导师已批准这一版具体结构。新25步入口已实现、16项CPU测试通过，尚未GPU启动/验收。
实施状态见 [GPU验收交接](../../reports/foundir_flux_native_energy_v1_readiness_2026-10-31.md)。

最新资源改为**默认3节点×每节点2张H100/H200，总6卡，允许节点间混用**。
当前启动与例外规则见 [3×2交接](../../reports/foundir_flux_native_energy_3x2_readiness_2026-10-31.md)。
全局batch8/512/三次更新不变；三节点CUDA尚未运行，不把CPU测试当作GPU通过。

**禁止外挂CNN energy、SpatialConditionalEnergy、TailQualityEnergy、Restormer、
residual refiner及其蒸馏。** 可训练图像主干只用预训练FLUX；energy来自FLUX内部
tokens＋energy LoRA＋线性读出，VLM等级经线性投影输入。原生冻结VAE和固定LPIPS
保留，不把它们说成额外恢复器，也不声称全计算图没有任何卷积。

固定FLUX生成路径，直接训练实际1/2/3次修后图像的MSE/LPIPS和动作排序。
目标仍是energy独立+.30dB及LPIPS相对降9.92810705%；旧CNN没有通过联合目标，
更大FLUX不保证成功。当前实验先Noise，之后才独立来源与其他退化。

完整方案与交接：[FLUX-only energy 重做](../../reports/foundir_flux_only_energy_redesign_2026-10-31.md)。
可复现PPT源码：`build_flux_only_deck.py`（Python标准库；沿用现有PPT的Office主题/布局，
重新生成全部正文、图形；不保留旧图片或笔记，不修改训练环境）。
本版10页，SHA-256：`386349ba2a7999a21dae31eb164f38b5504fa853d8ec76ebd20d7a2184213734`。
已核对XML/内部关系/文字/图形边界/ZIP CRC；未做PowerPoint或LibreOffice逐页渲染验收。

原导师讨论版本没有删除，保存在Git历史：
[原8页PPT，修订前版本](https://github.com/Jason-Wang313/zhenhaowangbackup/blob/610d23ac9df682376743fe96b44fd753992a42c4/presentations/novel_ppt_approach/Novel_PPT_Approach_VLM_Restoration_Energy.pptx)。
原文件SHA-256：`cdb8d5e0fe682d6cf30f0ba217d64c426e682fbbc7c70bd51e950ed38edacefb`。
项目名“Novel PPT Approach”不是已验证的创新性或成功声明。
