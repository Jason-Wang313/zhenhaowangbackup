# 局部VJP输入边界修复：CPU通过，GPU待测

GPU更新：本版一阶差0但首个高阶更新仍OOM，未解决显存；下一步为真实分配栈采集，
见foundir_flux_native_energy_allocation_trace_readiness_2026-10-31.md。

2026-10-31，继续已授权内存优化；下一步仅单H200单样本诊断，无自动训练。

v4已有实测：分数和一阶输入梯度与原生实现差均0，解决了分块版本的数值对照失败；
但首个create_graph更新仍OOM，峰值136.8115GiB，50秒status1，完整报告memory_v4_terminal。

## 具体复现与修复

Q/K/V来自共享投影和view图，尤其Q/K经过norm/旋转而V仍在共享分支上。
直接把这些非叶子输入作为内部autograd.grad的目标，会访问调用方上游图，
使本应局限于一个attention的重计算越过边界。
CPU hook实验：不隔离时访问q/k的上游各一次；为三个输入分别clone后，上游访问0次。
这说明真实边界问题存在，不独立证明已解释所有GPU显存峰值。

现在局部VJP对每个可微输入使用clone创建独立identity边；clone的导数为identity，
外层高阶反传仍连接原Q/K/V。只对本来不需梯度的常量建立临时叶子，活跃图不detach。
原native attention callable、布局、矩阵形状、math SDPA、原生autograd与数值开关保留。
局部图不再额外retain_graph=True；create_graph=True所需保留仍由原生autograd处理。
没有query/head分块、手写导数、CPU offload/模型分片或科学配方变更。

26项CPU检查全部通过，完整日志附JSON：旧25项全部保留，新增共享祖先回溯回归。
新回归同时确认内层不访问上游、外层高阶梯度确实仍传回且有限非零；
完整三步、全部参数梯度/Adam、数值梯度、生成不变、六rank加权/恢复继续通过。
这些是CPU证据，真实4B高阶CUDA容量仍未通过。

## 单卡GPU命令

一个空闲H200，新ID memory_isolated_vjp_v5，原cache训练index17，fresh energy。
先同状态原生/优化分数与一阶梯度对照，原1%门槛不变；通过后完整三步、
三个真实尾段/VAE/LPIPS、全部loss与最终反传。无Adam更新，无训练步或科学指标结论。
cap1200秒加30秒清理，不是预计时长；无自动重试/25步/长训。

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
export FLUX_NATIVE_PYTHON="$PWD/.conda/gpu-train-stable/bin/python"
export CUDA_VISIBLE_DEVICES=0
nohup bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_isolated_vjp_v5 > "logs/native_memory_isolated_bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
echo "launcher PID=$!"
```

仅一个节点执行一次。输出checkpoints/foundir_flux_native_energy_memory_v1/memory_isolated_vjp_v5/。
原始失败日志、缓存、step0、旧源码备份全部保留；不能resume旧源码checkpoint。
通过需ATTENTION_PARITY、MEMORY_BACKWARD_PASS和END status=0；即使通过，也不等于
H100、全部样本或分布式实际optimizer多步资格通过。目的地代理未自动启动GPU。
