# 联合 MoE V2 image-only：25步工程验收完成

结论：工程通过，恢复质量没有通过。本任务已正常结束；没有自动启动VLM臂或正式训练。

## 运行事实

- 实验：`image_only/qualification_nvls_v2`，独立25步验收，不是正式训练。
- 硬件：8×NVIDIA H100 80GB HBM3，节点 `eight-h200-copy-copy-copy--daf90a2bdaa1-k3xjvzod7q`。
- 开始：2026-10-26T19:13:22Z；结束：2026-10-26T19:20:40Z；launcher状态0。
- 总耗时438秒，即7分18秒、0.12166666666666667普通小时、0.9733333333333334 allocated H100 GPU-hours。
- 训练器内部计时0.08841048486417599小时／0.7072838789134079 GPU-hours，不含完整启动开销，不能替代上述总费用。
- 中位训练步耗时1.6934155019698665秒；所有rank显存峰值42.57999897003174 GiB。
- 800次样本呈现，Rain／Blur各400条不同记录，不是完整数据集训练。

本次节点实际是H100，上一轮CUDA401节点是H200。修复后的通信检查和训练在
本H100节点通过，但不能据此声称已验证原H200节点恢复，或唯一证明旧错误根因。

## 工程检查

- `NCCL_NVLS_ENABLE=0`、`NCCL_P2P_DISABLE=0`已在实际任务内生效。
- 八卡scalar all-reduce和小模型DDP反向传播检查通过；随后才启动训练。
- `engineering_pass=true`，初始MoE插入误差为0。
- 所有八个rank都记录两个专家、image router、共享LoRA A/B的实际更新。
- VLM分支保持不执行／无更新，`max_vlm_adjustment=0.0`，符合image-only对照。
- 专家权重和最大误差 `1.1920928955078125e-07`。
- 无NaN、OOM、NCCL或运行异常；完整step25权重、Adam、八rank RNG和评估结果已保存。
- 所有29个绑定源文件、配置、receipt、结果history与4份指标JSON一致。
- 独立重算权重和optimizer文件SHA-256，与resume receipt一致。
- 逐图核对step0／25记录、来源组和输入指标一致，独立重算指标均值通过。
- 未读取official／sealed／internal／parent-test像素。

权重／Adam有限性和完整状态由保存函数在运行时检查；本轮独立复核文件hash，
没有重新运行训练或评估，也没有把旧模型指标当成本次结果。

## 相同开发图片上，训练前后怎样变化

PSNR越高越好，LPIPS越低越好。以下全部是本V2、相同空文本条件下的匹配比较。

| 开发协议 | step0 PSNR | step25 PSNR | step0 LPIPS | step25 LPIPS |
|---|---:|---:|---:|---:|
| Rain100 | 34.17715539550872 | 33.86460822453734 | 0.07015681236982346 | 0.07575711041688919 |
| Blur100预览 | 23.8596342447602 | 24.26987270296453 | 0.4188850187882781 | 0.4072400273568928 |

- Rain：PSNR变化 `-0.31254717097137785 dB`，LPIPS变化 `+0.00560029804706573`，两个指标都退步。100张里24张PSNR提高、32张LPIPS降低。最终PSNR低于34.2；LPIPS仍低于.08。联合目标不通过。
- Blur预览：PSNR变化 `+0.41023845820432925 dB`，LPIPS变化 `-0.01164499143138531`，两个指标都改善。100张里58张PSNR提高、75张LPIPS降低。
- Blur原输入均值 `23.87114107616899/.37266220796853305`；恢复后LPIPS仍差于原输入，不能只强调相对初始化的改善。
- 没有跑完整Blur2595；预览不能冒充正式Blur门槛结果，也不能直接与旧2595均值比较。

训练25步只回答“能否正常训练／保存／评估”。它既不能证明联合任务已经做好，
也不足以据此拒绝整个架构或修改冻结配方。Rain退步是需要持续关注的现象，
但本实验没有单独证明其具体原因。没有VLM臂结果，不能声称VLM有效或创新成立。

## 下一步边界

下一计划是相同方案下VLM辅助臂的独立25步工程验收：
`bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh qualification vlm_assisted qualification_nvls_v2`。
本轮仅监控，未自动执行该命令。等待用户启动指示。
VLM工程验收仍暂估.20–.81普通小时／1.6–6.47 allocated GPU-hours，包含通信检查；这不是其已测用时。

两臂工程验收均通过后，才用两者实测速度重新报告10,000步正式训练预算并等用户决定。
image-only质量没达标不阻止匹配的VLM工程对照；任何验收权重均不得初始化正式训练。
不要延长本次25步、修改损失、覆盖旧失败记录或选择开发最优checkpoint。

## 复现身份与原始结果

输出根：`checkpoints/foundir_rain_blur_joint_vlm_moe_v2/image_only/qualification_nvls_v2/`。
启动修复提交：`75ff916bfe5b7dbf7b2f82caf95eeff968c608e3`。

| 文件 | SHA-256 |
|---|---|
| config | `3bdd59b3fb3c010cc7a93ab299f3f67fe85cc5eb5a6f380f610d4338398baebe` |
| result.json | `249071478a7f68d3d3bc8d812e9736497fee3d19a2ca5d3f60147c0c6ab5d278` |
| receipt.json | `cd817f26fef2b1340186d1788f90ceda6ac78ff6a2871b587d1291416f58c1bf` |
| step_000025.resume.json | `b1f8c96307a3a411c2c25e8439ea28a4e0c354947b81382ca80d6fb8ad6196b3` |
| step_000025.safetensors | `2bcec201bfdaedbe737b51eeb81a31904e2f3216ed6ffd01b3360a175d346fce` |
| step_000025.optimizer.pt | `365b4a45889c36ca51ef4895bfe82256d831d62b56097aa4be851d27f9cc1819` |
| rain_000000_learned.json | `9064967b73800f75869e94cb2d11df772cd499117f1af74ab9576e5b1123f268` |
| rain_000025_learned.json | `364960651ee6c7362d1601dd70a1aadedf8d05d8432bae2bb1699925ca64e7f9` |
| blur_preview_000000_learned.json | `eac7d6d57c5b5f1f837bc3be989fe58be22c4ce04635ec6899672742de51ae11` |
| blur_preview_000025_learned.json | `acdc14ca0b53cd974b92caa729c9be897e842ee2d5a3fbc80c780ab497413b7e` |
| launcher log | `90d8a2f22472e5ecb74b907553f093987f4af7e0085e47dbd13237e3e633b3e1` |

日志：`logs/foundir_rain_blur_joint_vlm_moe_v2_image_only_qualification_nvls_v2.log`。
Git仅收录报告／交接，不提交实时日志、图片或大型权重。
