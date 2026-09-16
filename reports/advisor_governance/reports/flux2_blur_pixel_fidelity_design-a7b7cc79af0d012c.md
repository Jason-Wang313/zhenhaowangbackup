# Advisor 阶段报告：flux2_blur_pixel_fidelity_design

- 结论：`information`
- Exact model：`FLUX.2 Klein Base 4B isolated-Blur step-10000 parent; paired DiT-LoRA flow-only versus flow-plus-frozen-VAE RGB-MSE tiny ablation`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_blur_only_flux2_pixel_fidelity_gate_v1.json` (`9219c0f9b1df8f871efb2cf87f2a95818a1ea73dbf9844b64cef57ec9e85c1e4`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`16 frozen isolated-Blur training crops plus 16 source-group-disjoint descriptive development crops; no protected split`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/flux2_blur_pixel_fidelity_authorization_v1.json` (`35976edd6b31cac486504618db67db434cb11f1e750fe6cd0374efcbc2c33139`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Step-12000 continuation regressed; historical five-type RGB path was confounded by anchored flow, mixed tasks, six-step solver, sparse auxiliary application, and a jointly trained pixel refiner. This design isolates direct RGB gradient on the best step-10000 pure-flow Blur parent.
- 请求 advisor 决策：是否按用户最新明确指令执行已冻结的 paired 16-pair pixel-fidelity tiny gate；仅 gate 明确胜过同轨 flow control 后再讨论 full scale？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
