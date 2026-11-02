# 已有FLUX专家四节点完整重放终态 — 2026-11-02

`replay_4x2_v1` / `launch_3d22527_v1`正常完成，四节点×每节点2张H200，
四份done和node0 verified标记齐全，四节点`EXPERT_REPLAY_NODE_END status=0`。

| 专家 | 配对图数 | decoded RGB最大差 | uint8逐像素一致 |
|---|---:|---:|---:|
| blur | 32 | 0.0 | 32/32 |
| haze | 32 | 0.0 | 32/32 |
| lowlight | 32 | 0.0 | 32/32 |
| noise | 32 | 0.0 | 32/32 |

目的地CPU另行读取256份保存张量，重算128对差异、tensor SHA、dtype/shape/finite及uint8一致，全部通过。
32份生产源码绑定、共同receipt/prompt、四份runtime和verified结果SHA均通过；26份energy源码未变。
每图仍为50步原入口与50步bank固定单选，不把这个结果包装成动态/soft路由或质量提升。
没有读取GT或开发/保护像素，没有新PSNR/LPIPS，没有Rain适配或训练。

各节点墙钟592/588/584/584秒；node0 9分52秒（.164444普通小时）。
四份每节点2×耗时合计4696 GPU-seconds = 1.304444 allocated GPU-hours，包含准备和等待。
所有卡实测H200。峰值allocated137.734817GiB；H10080GB未验收，不能仅凭原≥75GiB条件声称兼容。
下一次涉及H100必须先处理该显存实测限制；本轮不擅自优化或重跑。

同名JSON回传小汇总、128份比较收据身份、四节点成本及原日志SHA；不上传实时日志或tensor。
大资产与原始日志留共享盘。没有启动任何后续GPU任务。
下一阶段仍需交付共同prompt/输出协议核验与Rain多步适配；静态单选通过不等于五类MoE就绪。
