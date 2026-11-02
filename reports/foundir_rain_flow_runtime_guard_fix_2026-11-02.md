# Rain资格：修正驱动构建文本误拦截

用户授权修复；不修改模型、训练或评价方案。

## 故障与修复范围

GitHub d6a3142回传的四份runtime只有nvidia_driver字段不同：实际驱动均为
570.124.06，GCC描述分别以22.04与22.04.2结尾。旧代码直接比较完整字典，
因此在NCCL初始化及缓存生成前退出，0/25步，没有质量指标；不是OOM。
失败尝试累计.187777778使用GPU-hours，继续计入历史成本。

仅更改smoke/train_foundir_rain_flow_adaptation_v1.py的跨节点比较：
从NVRM行提取驱动release，比较release及全部其他原字段。未知/缺失描述拒绝，
不是删除driver检查。完整原始runtime记录不修改、不覆写；同节点断点恢复依然
比较完整原始字典。PyTorch/CUDA/cuDNN/transformers/GPU拓扑和评分环境继续严格比较。
此处放行编译器构建描述差异，不等于已验证NCCL、CUDA反传或确定性。

## 同步与重试

主代理无法访问新GPU挂载，未启动GPU。目的地代理安全同步origin/main，保留本地修改、
数据与权重。仅在新的multimodal-discrete-diffusion项目根执行：

```bash
.conda/gpu-train-stable/bin/python smoke/train_foundir_rain_flow_adaptation_v1.py preflight --report reports/foundir_rain_flow_destination_driverfix_preflight_v1.json
```

预检通过后，给用户原tmux入口的四节点命令，使用当前node0地址；四节点统一：

- run_id：qualification_4x2_driverfix_v1
- launch_id：launch_driverfix_v1（首次重试；后续同run恢复须新launch ID）
- 4节点×2H200，25步，原预算/损失/32训练来源/8图step0和step25评价均不变。

旧qualification_4x2_v1目录、日志和receipt原样保留。源码哈希已变化，
不能复用旧receipt或手改哈希；旧任务0步且无缓存，新ID不是丢弃训练进度或声称续训。
不自动正式长训、router或energy。结束后核验四节点退出码、保存及指标，回传小报告到GitHub。

## 回归覆盖

新增四项测试：实际故障的四份runtime可通过且收据不变；实际驱动或其他环境字段
变化仍拒绝；未知/缺失驱动格式拒绝；同节点resume仍拒绝原始runtime变化。
原有训练/保存/梯度测试一并运行：本套14项测试全部通过（15.629秒，不含共享盘依赖加载）。
包括真实小尺寸FLUX重算梯度一致性及单进程保存恢复；本次没有重跑两进程DDP测试，
该项先前通过记录保留，不冒称本次新验收。另以标准库提取实际函数重放四份报告，
确认构建后缀差异放行、真实驱动版本差异拒绝。Python语法、shell语法、diff检查通过。
测试命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:third_party/flux2/src <已验证Python> -m unittest discover -s tests -p test_rain_flow_adaptation_v1.py -v
```

本地使用此前已验证的gpu-train-stable-hdd3 Python；目的地仍需自己的资产预检与GPU验收。
