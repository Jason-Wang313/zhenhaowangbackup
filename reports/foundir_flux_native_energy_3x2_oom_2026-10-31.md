# FLUX-native 六卡资格：首步显存不足退出

2026-10-31，源码69a6a10，run=`qualification_3x2_v1`，3节点各2张H200。
17:23:20–17:28:01 UTC，三节点均END status=1；不是完成，也不是超时。

硬件、资产身份、六rank通信和共享目录检查通过，24/24候选缓存生成完成。
50个energy adapter、16,269,312可训练参数；step0已保存且文件SHA256独立核对通过。
没有完成任何optimizer step，没有probe/result.json或质量指标。

失败栈在训练unroll的torch.autograd.grad(score.sum(), state, create_graph=True)，
lowlight_restoration/flux_native_energy.py:198。rank0日志显示申请1.15GiB失败：
单卡总139.81GiB，空闲850MiB，进程占138.97GiB，PyTorch allocated135.93GiB、
reserved但未allocated1.91GiB。六卡数据并行没有降低单卡图的显存要求。
这仅记录实际OOM，不据此判断科学路线，也未验证任何修复。

各节点wall_seconds为281/280/278，local_gpu_seconds为562/560/556；
合计1678 GPU秒，即0.466111 GPU-hours（含启动/CPU核对，按launcher记录）。
完整三节点日志存于同名JSON。原日志、缓存、step0均保留；未改配方、未重启GPU。
分析和修复交由主执行代理。
