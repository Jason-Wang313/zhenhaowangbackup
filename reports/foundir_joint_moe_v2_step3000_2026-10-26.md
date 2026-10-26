# 联合 MoE 第3,000步中期检查

2026-10-26 21:57 UTC更新：两条第3000步结果均已生成。下面先前普通版观察保留；
两条继续既定10000步训练，不进行中期checkpoint选择。

| 实验／协议 | PSNR | LPIPS | 本轮判断 |
|---|---:|---:|---|
| image-only Rain100 | 34.32609266908536 | .07442104332614691 | 本检查点两项满足34.2／.08 |
| image-only Blur100预览 | 25.399459516822663 | .31455456842668356 | 仍未满足25.9／.18 |
| VLM Rain100 | 34.32999647989338 | .0712783659575507 | 本检查点两项满足34.2／.08 |
| VLM Blur100预览 | 25.38343235445578 | .30998902085004376 | 仍未满足25.9／.18 |

相同步数VLM减普通版：Rain PSNR +.003903810808019159 dB，
LPIPS -.0031426773685962056；Blur PSNR -.016027162366881953 dB，
LPIPS -.004565547576639795。两类LPIPS均降低，但Blur PSNR略差；
尚无两指标均稳定获益、统计显著性或论文创新成立的结论。
两臂Rain本检查点都达标，Blur仍有明显LPIPS差距。

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

VLM完整step3000也已保存，world_size8、next_sample_position96000。
小文件独立SHA-256（相对vlm_assisted/formal_v1）：

| 文件 | SHA-256 |
|---|---|
| metrics/rain_003000_learned.json | 0de30c43b78c327b2fdf50fbc73b5fd84b36fb4ec9f10b2d8fec3860a3da3bb7 |
| metrics/blur_preview_003000_learned.json | 530fbe34c834714429d218616c1155e80fefeff6e4129c054ab7b96ca70294c6 |
| step_003000.resume.json | 129787e581366fd80086ab9ad2ce192931ce27cd5738f15083fe6c798ad35d6e |

VLM大文件hash来自receipt，本轮未独立重读：
权重 `90ef3e0481d058cf7b0c8fa9ef2406c85466a090ec0dccda07ade0d82f344818`；
Adam `1237dcb245aa25fc80b055685c76cfe0d2df13804133cfa6cb87352eb86b9930`。
