# 冻结五专家router：跨节点混合驱动兼容

用户明确要求支持当前6张H100+2张H200，以及节点0/1驱动570.124.06、节点2/3驱动595.58.03（proprietary/open module）。只放宽节点间驱动release相同限制；仍严格匹配PyTorch/CUDA/cuDNN/transformers、评分和显存策略。原始驱动记录保留，同一run每节点runtime仍不可改变。CUDA/NCCL/parity/有限梯度必须真实验收，不声称混驱动GPU已跑通。

原四份v1启动/训练/contract绑定源码未改，旧目录和receipt未改；当前旧任务没有训练快照，但还有node0:29703 rendezvous超时记录。驱动修改不修复网络，必须先确认当前node0地址/端口可达。新入口不自动杀旧任务、启动GPU或扩预算。

新入口：smoke/start_foundir_frozen_expert_router_mixed_driver_v1_tmux.sh。
参数仍rank/current_node0_IP/run_id/fresh_launch_id；建议run=qualification_mixed_driver_4x2_v1、launch=launch_frozen_router_mixed_driver_v1、port29705。
结果仍在checkpoints/foundir_frozen_expert_router_v1/<新run_id>，独立身份防止复用旧绑定源码。不是丢弃已保存训练状态的续训：检查时旧run尚无step。

先CPU预检：
```
.conda/gpu-train-stable/bin/python -u smoke/train_foundir_frozen_expert_router_mixed_driver_v1.py preflight
```

目的地预检已通过，pixels_read=false、GPU_used=false。4项兼容回归通过：当前两种驱动/两种GPU组合接受、错误软件/指标/显存策略/拓扑拒绝、非法型号/驱动/节点缺失拒绝、训练主体AST与父入口相同。当前四份真实runtime记录已通过新策略。另验证bash/AST以及旧任务未退出拒绝、剩余预算扣除。

新启动脚本要求旧四节点退出收据齐全，并按节点扣除旧尝试和新run历史尝试秒数：remaining=10800-spent。原3小时/节点及总24 GPU-hours预算不重置，耗尽拒绝。科学配方不变：400轻量初始化＋8完整路径质量步，global32，原loss/数据/裁图/量化；未自动增加资格或正式训练。

启动命令（四节点分别rank0/1/2/3，必须使用当轮可达地址）：
```
bash smoke/start_foundir_frozen_expert_router_mixed_driver_v1_tmux.sh <rank> <current_node0_IP> qualification_mixed_driver_4x2_v1 launch_frozen_router_mixed_driver_v1
```

旧任务如仍占用节点锁，不重复启动。用户可在各节点原会话内Ctrl-C结束旧尝试，等待四节点退出收据后再运行；不删除日志/快照/孤立文件。终态仍FROZEN_ROUTER_RESULT_VERIFIED＋四节点status0，只做轻量报告，不自动formal。GPU兼容性/通信/显存/吞吐仍待真实验证。
