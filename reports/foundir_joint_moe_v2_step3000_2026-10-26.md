# 联合 MoE 第3,000步中期检查

2026-10-26 21:34 UTC：普通版结果已生成并恢复训练，VLM版尚未到第3000步。

| 实验／协议 | PSNR | LPIPS | 本轮判断 |
|---|---:|---:|---|
| image-only Rain100 | 34.32609266908536 | .07442104332614691 | 本检查点两项满足34.2／.08 |
| image-only Blur100预览 | 25.399459516822663 | .31455456842668356 | 仍未满足25.9／.18 |
| VLM第3000步 | 尚未生成 | 尚未生成 | 不做不同步数的优劣判断 |

相比普通版第1000步，Rain PSNR上升、LPIPS略变差但仍低于.08；
Blur两个指标均改善，但LPIPS距离.18仍有约.13455。
这说明联合训练并非完全无效，但没有证明两任务均达标，更没有证明VLM有效。

本轮是中期观察，不选checkpoint、不改损失或学习率、不提前停止。
完整Blur2595只在最终10000步；必须等最终对照才能评价VLM增益。
两臂硬件仍分别8×H100与8×H200，保留已记录的初始数值差异说明。
固定正式预算每臂5.45–9.06正常小时／43.63–72.45分配GPU-hours；
这不是最终已测耗时。

## 存档核验

普通版完整step3000权重／Adam／resume记录已保存，
mode=formal，world_size=8，next_sample_position=96000。
小文件独立SHA-256（路径相对image_only/formal_v1）：

| 文件 | SHA-256 |
|---|---|
| metrics/rain_003000_learned.json | 09884c3522cf82834e28ac07d27d8153a037f718d4b17045dafd135c4cd7ef0f |
| metrics/blur_preview_003000_learned.json | f2c43d8cacae0e0575241b7b64cf5fe950c07876d61a44fd74245f578eb3a6fc |
| step_003000.resume.json | 22af4e53f547363c6c45b07c1e8cdd3006ad37f94808e77e40853aab9bea49c4 |

以下大文件hash来自训练器保存的receipt，本轮未独立重读全部大文件：
权重 `691d6a86c131507001d0755345715aa46962a74ca2b05a50a2f4279a1ef0d4e1`；
Adam `7379e9ac78581a1a221d794e5c9db4cc255800d3100e102f623a56c7f34ccffe`。
监控继续，不占用额外GPU，不读取受保护像素。
