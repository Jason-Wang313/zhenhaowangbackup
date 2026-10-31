# 独立200步trainfit终态日志回报

采集UTC：2026-10-31T10:57:04.407138+00:00；run：`trainfit_3x2_direct_v1`；protocol `c47620eb566c525d69f3546dd42e5d04b01ed0db355fa291e02d9d7343addda1`。
三节点六H200，均于2026-10-31 07:59:06 UTC以status0结束，有TRAIN_COMPLETE、
PROBE_COMPLETE、RESULT_VERIFIED。200/200完整训练步，8/8训练来源probe完成。
bootstrap及step50/100/150反例刷新均24/24完成；日志包含step0–200共201次保存标记。
preflight记录qualification_checkpoint_loaded=false，本轮独立fresh初始化。

| 报告项 | PSNR | LPIPS | 相对当前original PSNR | LPIPS相对降低 |
| --- | ---: | ---: | ---: | ---: |
| original | 38.176074505 | 0.080517740 | +0.000000000 | +0.000000% |
| historical | 38.172442913 | 0.080524399 | -0.003631592 | -0.008271% |
| fp32_control | 38.176177502 | 0.080520860 | +0.000102997 | -0.003876% |
| raw_step1 | 39.261043072 | 0.078503689 | +1.084968567 | +2.501376% |
| bounded_step1 | 39.236412048 | 0.078516108 | +1.060337543 | +2.485951% |

raw/bounded均未达PSNR+.30dB且LPIPS下降9.928107050377874%的联合目标。
bounded有8/8图PSNR改善，5/8图LPIPS改善，3/8图LPIPS变差。
仅training_fit_only，未建立泛化；这里只报告实际数值，不提出训练修改或新实验。

| 节点 | elapsed秒 | 使用GPU秒 | 结束UTC | status |
| --- | ---: | ---: | --- | ---: |
| 0 | 38712 | 77424 | 2026-10-31T07:59:06Z | 0 |
| 1 | 38711 | 77422 | 2026-10-31T07:59:06Z | 0 |
| 2 | 38710 | 77420 | 2026-10-31T07:59:06Z | 0 |

node0耗时38712秒=10小时45分12秒（10.753333普通小时）；三节点合计
232266使用GPU秒=64.518333333 GPU-hours。本轮成本不替代以前各轮记录。
训练平均186.639635秒/步；六卡峰值allocated均
31.848424911GiB。整轮耗时含启动、缓存、刷新及probe。

最终step200 checkpoint文件SHA已核对：`a414218997c4076e78ed7fab650fbffa21e1f7b892408dd41cae944129258339`。
原始result SHA：`0d020a4f1f7703f84f15b02a4ca4387f490ffb8b71da80b374aa824bf7d859f8`；8图均值独立重算匹配。
[同名JSON](foundir_flux_tail_v3_direct_trainfit_terminal_2026-10-31.json)完整保留原result字段、全部8图候选/选择/metrics、
200步训练数值、201次checkpoint日志、参数梯度审计及节点终态；不上传tensor/权重/活日志。
现有pkg_resources弃用警告照常出现，未见Traceback/RuntimeError/NCCL error，三台正常退出。
本次只读日志和保存记录，没有改代码、重启、扩预算或启动后续任务。下一步由主执行代理决定。
