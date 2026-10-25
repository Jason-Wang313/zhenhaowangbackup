# Rain FLUX 3,000 → 4,000 续训进展

## 当前状态：运行中

用户已在 8×H100 节点启动 `continue_v1`，启动时间
`2026-10-25T23:20:16Z`。用户要求持续监控直到结束；不修改冻结配方。
代码及恢复协议备份：`c9dcdeaabcd989fce088aa75d9fc0e293252b063`。

日志已确认：LoRA step=3000、Adam step=3000、学习率 1e-5、
`optimizer_state_exactly_restored=true`、首个数据位置 96000。
第 3,001 步已成功完成，后续 step 3,020 损失／梯度有限，
LoRA A/B 均实际更新。预热后约 1.5 秒／步，峰值已分配显存约 24.86 GiB。

此时尚无 3,500／4,000 验证结果，不预测达标。
原 step-3,000 开发集基线为 34.00838778684664/.07343229320365935。
联合目标仍为 PSNR >=34.2、LPIPS <=.08。
启动预算为 40–50 分钟／5.3–6.7 allocated GPU-hours，结束时记录实际成本。

日志：`logs/foundir_rain_flux2_precision_resume_v1_continue_v1.log`。
输出：`checkpoints/foundir_rain_flux2_precision_resume_v1/continue_v1/`。
保持原始历史文件，不启动 MoE 或下一轮实验。
