# 按独立head分块：CPU通过，待4B CUDA同门槛检查

2026-10-31。继续用户已授权的内存修复，候选版本，不是GPU通过结论。
上版memory_tiled_v2的一阶梯度relative-L2差11.4771%，虽然其区间显存9.3973GiB，
仍按原1%门槛失败停止，未执行三步或最终反传。完整报告aa02a0c，失败结果不覆盖。

## 实际改动

lowlight_restoration/flux_native_attention.py改为每次一个attention head。
24个head相互独立，每个head保留全部3585个queries和keys，不拆query、不累加
不同query块的K/V梯度。其输出/梯度沿head维拼接。各head的解析VJP继续checkpoint，
全部高阶梯度保留；生成仍调用原native attention，源码/权重/数学配方边界不变。
一份FP32 head矩阵约49MiB，仍有其他激活/参数/反传开销，不承诺总峰值。
选择head分块的依据是保留每head原本的完整归约范围；尚未证明它解决实际11.48%差异。

25项CPU测试通过，完整日志附JSON；原数学等价/gradcheck/gradgradcheck、完整三步
旧新loss/参数梯度/Adam、生成不变和六rank恢复均通过。新增24heads、head_dim128、
较长序列的BF16测试：输出一致，一阶梯度relative-L2最大约0.004669%。
小尺寸BF16高阶梯度最大约0.36971%，不是逐位一致。CPU通过不等于CUDA通过。
旧小CPU测试没有覆盖真实4B长序列/多层误差，已由GPU失败证明不能过度外推。

## 下一诊断

一个空闲H200，新ID memory_heads_v3，旧cache训练index17、同512图、fresh energy。
先同状态比较旧math/new head分块的分数和一阶输入梯度：仍为分数atol1e-5/rtol1%、
梯度relative-L2<=1%，没有放宽。额外记录实际TF32/matmul/math-SDPA数值开关，只读不改。
通过才执行完整三步、真实FLUX尾段/VAE/LPIPS/全部loss和最终反传。
无optimizer更新、无训练步，不自动资格或长训。单卡cap1200秒+30秒清理。
结果根checkpoints/foundir_flux_native_energy_memory_v1/memory_heads_v3/。
即使单样本通过，H100、不同样本、实际Adam/分布式多步仍未验收。

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
export FLUX_NATIVE_PYTHON="$PWD/.conda/gpu-train-stable/bin/python"
export CUDA_VISIBLE_DEVICES=0
nohup bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_heads_v3 > "logs/native_memory_heads_bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
echo "launcher PID=$!"
```

只在一个节点执行一次。旧memory_tiled_v2/memory_recompute_v1/qualification目录不动。
本轮尚未GPU启动，等待用户执行；数值/显存错误如实记录，不自动重复或放宽标准。
