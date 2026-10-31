# 原生attention VJP逐层重计算：CPU通过，GPU待测

2026-10-31，继续用户已授权的内存修复。当前仅新的单卡容量诊断，不是训练启动。

## 已测失败与边界

初始6H200资格首步OOM。外层导数checkpoint的单卡v1完成三步更新后，最终反传仍OOM。
query分块v2一阶梯度差11.4771%，head分块v3差8.1279%，均超过1%门槛，主动报错停止。
v3运行39秒/0.010833分配GPU-hours，完整日志315a09e；未执行三步/完整backward。
TF32=true/matmul=high已实测，但尚未隔离为差异原因，不据此放宽比较标准。

## 实际改动

删除query/head分块及手写softmax解析VJP。原native causal_attn_fn、拼接顺序、
矩阵形状、math SDPA前向和torch.autograd.grad导数原样执行。
自定义autograd前向只保存Q/K/V；每个attention的可微VJP用非reentrant checkpoint，
使最终反传只在需要时重算该attention的导数中间张量。原逐block和外层导数checkpoint保留。
关键区别是checkpoint边界下沉至每个attention的导数，而不是重建全部层的大导数图。
未改变数值全局开关，未加CPU offload/模型分片/新网络，未缩短三步/512/损失或global batch8。

内层autograd保留其共享Q/K/V上游图供外层反传使用；常量输入仅临时启用求导，
活跃可微输入不detach。CPU集成测试发现并修正了提前释放共享图的问题。
不再承诺分块矩阵大小；单attention临时矩阵仍可能1.15GiB，实际总峰值必须测。
生成模式直接调用原native函数，未更改third_party文件或原FLUX生成定义。

## 已测结果

25项CPU检查通过：原9项、7项拓扑/恢复、3项全三步/导数检查及6项attention检查。
包括原生autograd前后FP64的一/二阶梯度、gradcheck/gradgradcheck、完整三步旧新loss/
全参数梯度/Adam、原生成输出、六rank采样加权与恢复。完整日志见同名JSON。
24heads/head_dim128的CPU BF16测试输出和一阶梯度逐元素一致。
小BF16高阶梯度最大relative-L2约0.36971%，并非所有高阶结果逐位一致；不外推CUDA。
saved_tensors_hooks确认新高阶图未保存完整Q×K矩阵；不等于GPU峰值已通过。
参考PyTorch v2.7.0的原生math attention实现：
https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/transformers/attention.cpp
本候选直接重用运行环境里的native函数，不复制或替换其公式。

## 下一GPU命令

新ID memory_native_vjp_v4，单个空闲H200，原训练cache index17、fresh energy。
同状态比较原native与checkpointed native：分数atol1e-5/rtol1%、latent梯度relative-L2<=1%，
原门槛不变，通过才做完整三步、三个真实尾段/VAE/LPIPS、相同全部loss与最终backward。
无optimizer更新、无训练步，不自动资格或长训。cap1200秒加30秒清理；不是预计耗时。
即使通过也仅是一个H200/单样本检查，H100及分布式多步仍待验收。

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
export FLUX_NATIVE_PYTHON="$PWD/.conda/gpu-train-stable/bin/python"
export CUDA_VISIBLE_DEVICES=0
nohup bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_native_vjp_v4 > "logs/native_memory_vjp_bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
echo "launcher PID=$!"
```

仅一个节点运行一次；原目录全部保留。输出在
checkpoints/foundir_flux_native_energy_memory_v1/memory_native_vjp_v4/。
已准备并同步代码，未自动启动GPU；正式CUDA通过需ATTENTION_PARITY、MEMORY_BACKWARD_PASS
和END status=0三个证据，不能以CPU或tmux/nohup创建替代。
