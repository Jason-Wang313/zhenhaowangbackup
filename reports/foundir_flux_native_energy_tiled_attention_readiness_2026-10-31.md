# energy attention分块：CPU通过，待GPU

GPU实测更新：本query分块版实际梯度差11.4771%，未过1%门槛，已停止。
最新候选与命令改读foundir_flux_native_energy_head_attention_readiness_2026-10-31.md。

2026-10-31。继续用户明确授权的内存优化，不增加训练任务或改变科学配方。

## 实测问题

第一版外层导数checkpoint在一个H200上完成三步更新及三个真实解码，
更新阶段峰值13.3024GiB，loss计算阶段峰值17.2524GiB。
最终backward重计算仍升到135.3005GiB并OOM，status1；不能称为修复完成。
完整结果与日志已推ddaa5f0，见foundir_flux_native_energy_memory_v1_terminal_2026-10-31。

## 实际改动

新模块lowlight_restoration/flux_native_attention.py执行完整softmax attention：
沿query切成128-token块，每个query仍使用全部keys，不裁剪/稀疏化或减少tokens。
前向每块用PyTorch math SDPA；自定义autograd backward按标准softmax解析VJP
重算该块概率，并对每块VJP非reentrant checkpoint，以保留完整高阶梯度而不同时
保存全部层的大attention导数矩阵。K/V块贡献先FP32累积再转回BF16，避免逐块取整。
FP64保留用于数值测试；未使用CPU offload或模型分片。

实际3585 tokens、24heads：一份完整FP32矩阵约1.15GiB，128-query块约42MiB。
这是矩阵大小，不是GPU峰值承诺；反传、其他激活和参数仍占空间。
现有外层导数checkpoint和逐层checkpoint保留，可能增加重算/启动开销。

仅energy模式走新分块实现；生成模式通过上下文分派调用原native函数。
不修改third_party文件、不覆盖torch全局SDPA、不换参数，checkpoint重算捕获模式。
源码binding新增attention文件。三步/512/global batch8/所有损失/模型/数据均保持。
旧缓存只读，保留旧结果、step0与代码备份；不同源码不能直接resume旧checkpoint。

## 已测数值与限制

24项CPU测试通过，完整日志在同名JSON：
- 原9项回归、7项六rank/恢复测试、3项导数重计算测试、5项attention测试。
- FP64 attention输出、一阶与二阶导数同原math SDPA在rtol1e-10/atol1e-11内一致。
- gradcheck及gradgradcheck通过；完整三步旧实现与新实现的输出/全参数梯度/Adam
  更新通过严格对比，生成路径保持原输出。
- 小尺寸BF16 attention输出和一阶梯度本例一致；高阶梯度相对L2差最大约0.4182%，
  小于预定1%检查门槛。不能声称BF16逐位等价，不能把CPU误差当真实4B误差。
- saved_tensors_hooks确认测试中的新高阶图不再保存完整Q×K矩阵；不代表CUDA验收。

## 唯一下一GPU任务

仍是一个空闲H200上的单样本容量诊断，新ID memory_tiled_v2，GPU未自动启动。
复用原cache的训练index17（global stream slot0）；真实4B、原生VAE、LPIPS、512图。
先在同一状态比较原math attention和新attention的分数及一阶latent梯度，
不构造旧的大高阶图；分数atol1e-5/rtol1%，梯度relative-L2上限1%，超出就报错停止，
不能事后放宽。之后执行完整三步、三次真实尾段渲染、原直接/排序loss和完整backward。
这里只计算一个样本对global batch8的1/8梯度贡献，无Adam更新、无训练步、无科学结论。
CPU小测试之外的BF16高阶数值一致性尚未完整实测，此单样本也不能代替多样本资格。

单卡诊断cap1200秒加30秒清理，约0.3333使用GPU-hours加清理；不是预计时长。
分阶段日志与result写新目录checkpoints/foundir_flux_native_energy_memory_v1/memory_tiled_v2/。
必须有ATTENTION_PARITY、MEMORY_BACKWARD_PASS与END status=0，才是本单样本检查通过。
H200通过不保证H100可用；后续分布式资格仍需实际optimizer/多步/保存验收，不自动开始。

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
export FLUX_NATIVE_PYTHON="$PWD/.conda/gpu-train-stable/bin/python"
export CUDA_VISIBLE_DEVICES=0
nohup bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_tiled_v2 > "logs/native_memory_tiled_bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
echo "launcher PID=$!"
```

不要在多个节点重复这条单卡命令，不复用memory_recompute_v1的输出目录。
