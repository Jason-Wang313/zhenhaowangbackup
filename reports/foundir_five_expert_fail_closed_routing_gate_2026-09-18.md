# 五个冻结专家 fail-closed routing gate

日期：2026-09-18

## 结论

确定性单退化 routing/integration contract 已完成并通过 CPU gate。

- 配置：`configs/foundir_five_expert_routing_v1.json`
- 配置 SHA-256：
  `1a9b05c2d89ebf8cf371f82b4e98d7f25c0526fea1df51dd9a964f0a74ecbee8`
- 测试：
  `tests/test_foundir_five_expert_router.py` 与
  `tests/test_foundir_rain_expert_deployment.py`
- 结果：48/48 passed
- GPU：未使用
- sealed internal / official FoundIR test：未访问

## 路由边界

只接受以下五个精确单例标签：

| 标签 | 冻结 backend |
|---|---|
| blur | blur_flux2_step10000_alpha075 |
| noise | noise_flux2_step400_alpha030 |
| haze | haze_flux2_step3000 |
| rain | rain_restormer_step5000 |
| low_light | lowlight_flux2_step400_alpha0925 |

以下情况全部抛出 `RoutingRefusal`，不猜测、不选择第一个标签、不静默回退：

- 空标签；
- 未知或非规范标签；
- 重复标签；
- 任意两个或以上标签；
- mixed category；
- `composition_size != 1`。

## 已验证内容

1. 五个单例分别只映射到自身 backend。
2. 所有 2--5 类组合均拒绝。
3. 三类标签的全部顺序置换均不能绕过 mixed 拒绝。
4. checkpoint path/SHA 与冻结 development registry 完全一致。
5. Blur/Noise/Low-light 的 alpha 分别锁定为 .75/.30/.925。
6. Haze/Rain 仅执行 identity quantization，不附加其他后处理。
7. 五条路径统一返回最终 uint8 像素及由同一 uint8 反归一化的 [0,1] 表示。
8. 配置扩大到 Qwen、mixed、错误 backend 或改变 alpha 时均 fail closed。

## 尚未包含

本 gate 不包含 Qwen 分类。Qwen 必须作为独立上游层评估分类混淆、置信度与
abstention；不得把 Qwen 误分类归因于专家本身。Mixed degradation 仍明确禁止，
只会在 Qwen 单退化路由完成后另行建立协议。
