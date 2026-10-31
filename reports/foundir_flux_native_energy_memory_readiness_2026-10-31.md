# FLUX-native 导数图重计算：CPU通过，待GPU显存实测

日期：2026-10-31。用户在OOM分析后明确要求执行内存优化。
本次实际实现：对energy分数及其输入梯度整体使用非reentrant checkpoint，
保留已有逐层checkpoint。完整三步、数学attention、参数/损失/512图/全局batch8不变。
不detach更新方向，不切换CNN，不引入CPU offload或模型分片，不承诺速度提升。

## 已观察到的问题

旧qualification_3x2_v1的三节点6H200于17:28 UTC全部status1退出。
24候选缓存和step0已保存；0/25训练步，通信/共享目录通过。完整日志见
foundir_flux_native_energy_3x2_oom_2026-10-31.{md,json}。旧资产全部保留。
失败位置为unroll里的autograd.grad(create_graph=True)。rank0进程138.97GiB，
另申请1.15GiB失败。仅增加数据并行卡数不能分摊这份单样本计算图。

只读取缓存safetensors头部确定：候选/输入参考/proposal各1024 tokens，
文字512加1个条件token，共3585；24heads的FP32完整attention矩阵约1.15GiB。
这是与失败申请大小吻合的结构性证据，未做真实GPU算子分配归因。
逐层前向checkpoint不能消除create_graph计算导数时新保存的图，故增加外层导数重计算。

## 已测结果与限制

19项CPU测试通过：9项原生小FLUX回归、3项内存改动测试、7项拓扑测试。
包含完整三步修后loss、全部可训练参数梯度和Adam更新的前后对比、
输入梯度及更高阶反传对比、生成路径不变、六rank样本加权和Adam/RNG恢复。
CPU双精度输出逐元素相同，梯度/Adam比较rtol1e-9、atol1e-11通过。
通过saved_tensors_hooks观察的一次energy+输入梯度调用所保存的storage，
原590280 bytes，外层checkpoint后256 bytes。此数不包括模型/输出和重计算瞬时内存，
不能作为真实4B CUDA显存降幅或H100可运行的保证。正式CUDA/BF16仍未验证。
完整CPU测试日志附于同名JSON，不上传任何权重/图片/缓存。

## 下一步：一个GPU上的短容量诊断

唯一下一命令为run_foundir_flux_native_energy_memory_v1_gpu.sh。
此次明确使用单节点一张空闲H200，测单卡瓶颈，无需跨节点；这是容量测量例外，
不改变后续非轻量GPU工作的3×2默认。H200通过也不自动证明H100可用。
先核对原receipt、资产SHA和一个缓存文件；读取原训练流slot0对应的来源。
加载真实4B/VAE/LPIPS，fresh energy，在512图上做完整三次更新、三个真实尾段解码、
相同直接损失与两种排序loss，完整backward并检查有限非零LoRA梯度。
只算一个样本对global batch8的1/8贡献，不做Adam更新；不是一个完整训练步，
不输出可用于续训的权重，不出科学质量结论，不启动25/200步或正式训练。

每个阶段即时fsync记录显存/耗时。完整运行硬cap1200秒加最多30秒清理，
即最多20分钟/约0.3333使用GPU-hours加清理。不是预计耗时。
普通成功/异常写result.json，OOM另存allocator摘要；强制kill仍保留已写阶段日志。
输出checkpoints/foundir_flux_native_energy_memory_v1/memory_recompute_v1/。
旧缓存只读；旧qualification的源码身份已经变化，不能对旧ID直接resume。
通过标记NATIVE_ENERGY_MEMORY_BACKWARD_PASS与END status=0，只代表一个样本反传通过。
若失败按阶段结果定位，不自动重试、扩预算、减少三步或改配方。

在一个有空闲H200的节点执行一次：

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
export FLUX_NATIVE_PYTHON="$PWD/.conda/gpu-train-stable/bin/python"
export CUDA_VISIBLE_DEVICES=0
nohup bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_recompute_v1 > "logs/native_memory_bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
echo "launcher PID=$!"
```

目的地代理未启动GPU，等待用户节点实测。运行入口、诊断脚本、energy模块和本次测试
均须同步到共享根；本报告发布时会核对SHA，保留本地脏文件和备份。
