# Advisor 阶段报告：zimage_conditional_teacher_full_scale_authorization

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image Base 6B conditional teacher, full Blur scale continuation`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_conditional_teacher_full_v1.json` (`d3fd35437abd6239c450f42bbb9674133aeb84f8d929dce00ced000f5b9ec8d7`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`105057 Blur training records / 7339 groups; 2595 source-disjoint development records / 165 groups`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/zimage_conditional_teacher_full_scale_authorization_v1.json` (`56257204db0239315301e0b45db1469a7dcb7e3aea8d13eeb1805847f3ca4eb8`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：User explicitly authorizes the maximum single-instance scaling plan after reviewing the failed but improving 2048-source pilot: exact pilot step2000 initialization, one homogeneous 8xH100/H200 instance, 20000 additional optimizer steps, full Blur training archive, protected tests untouched.
- 请求 advisor 决策：Record this exact explicit user instruction as authorization; no new advisor reply requested.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
