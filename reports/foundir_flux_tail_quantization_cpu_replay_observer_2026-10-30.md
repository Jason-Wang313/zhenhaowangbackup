# 新节点 CPU 量化重放回报 — 2026-10-30

采集UTC：2026-10-30T19:22:03.494002+00:00。
按主执行代理 `7ce6cef` 同步8个指定文件并执行原命令；7个文件与远端逐字节一致，
`agents.md` 三方合并保留旧本地修改。未修改绑定训练代码/缓存，未启动GPU或训练。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .conda/gpu-train-stable/bin/python -u smoke/replay_flux_tail_quantization_cpu_v1.py --output reports/foundir_flux_tail_quantization_cpu_replay_2026-10-30.json
```

退出码0。`kind=saved_tensor_cpu_replay`，`cuda_initialized=false`，GPU-hours=0。
脚本记录耗时1.873017秒（不含解释器启动前置导入）。
完整标准输出与[原始JSON](foundir_flux_tail_quantization_cpu_replay_2026-10-30.json)一致，
包含全部比较、证据SHA及synthetic结果，未删减。日志仅另有`pkg_resources is deprecated`
警告，无Traceback/执行错误。JSON SHA256：`02b4990d0be484ce8dd72e7641153375ea5d5059208db26d7714f506539d40a8`。

- 输入归一化：CPU与保存GPU值有786429/786432元素不同，最大差1.1920928955078125e-07。
- 第一次取整：重算与保存raw_uint8差异0。第二次取整及完整重放：与保存GPU输出差异0。
- 保存GPU与历史图：3630通道元素/3591 RGB像素不同，最大1灰度级。
- 其中1214通道差异在精确半整数点，2416不在半整数点。
- 合成65536组检查：FP32与精确有理数差1422组，均在半整数点（不是实际图像差异计数）。

下表均为不同通道元素数；“匹配旧差异”与“新增差异”按脚本原字段回报。

| 控制 | 与保存GPU不同 | 与历史不同 | 匹配旧差异 | 原本相等处新增差异 |
| --- | ---: | ---: | ---: | ---: |
| cpu_full_replay | 0 | 3630 | 0 | 0 |
| cpu_second_cpu_observation | 0 | 3630 | 0 | 0 |
| cpu_second_saved_observation | 0 | 3630 | 0 | 0 |
| diagnostic_exact_rational_second_only | 46143 | 48541 | 616 | 45527 |
| diagnostic_float64_second_only | 26579 | 29407 | 401 | 26178 |

所有控制都未完全匹配历史图。原始JSON报告`root_cause=not_established`、
`baseline_changed=false`、`gpu_qualification_pass=false`。这里只回传实际输出，
分析、修复及下一步由主执行代理负责。本次未重复其已通过的10项单元测试。
