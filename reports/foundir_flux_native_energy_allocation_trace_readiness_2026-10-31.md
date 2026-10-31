# 保持v5运算，采集真实CUDA分配栈

2026-10-31。v5同状态一阶数值差0，但高阶更新仍OOM，真实显存问题未解决。
CPU小尺寸回归不足以解释CUDA张量生命周期。下一步取实际分配证据后再决定改法。

## 本次实际改动

只有诊断脚本和启动器增加可选--memory-trace/FLUX_NATIVE_MEMORY_TRACE=1。
attention、energy、训练器、配置保持8e78eb9字节不变，不再更换计算实现。
通过数值对照后、进入高阶unroll之前，启用PyTorch 2.7 allocator历史：
enabled=all，context=all，stacks=all，最多30000条alloc/free事件。
发生异常或完成时保存cuda_memory_snapshot.pickle和allocation_summary.json，
result记录快照SHA/大小；采集失败会明确记录，不能当作成功。
记录开启前的模型分配可能无栈，超过环形缓冲上限的历史可能被截断；
异常捕获时部分临时张量可能已释放，摘要明确为capture时存活量而不是OOM峰值。
原始历史保留OOM事件，后续CPU可核对具体栈及分配/释放关系。

当前静态/CLI/shell检查通过，新增2项CPU摘要计数检查通过（存活/等待释放/已释放及无栈分配）。
原计算的26项CPU回归和GPU失败证据仍保留，不把本次2项测试说成GPU通过。
实际快照工具仅在GPU可运行；本代理没有执行。大快照只留共享目录，不推GitHub。

## 一次单卡证据采集

同一空闲H200、同训练cache index17、同v5计算路径与1%数值门槛。
新ID memory_alloc_trace_v6；预计复现原OOM，这次不是新修复版本，也不是训练。
没有optimizer更新、无自动重试/资格/长训。cap1200秒+30秒清理，不是预计时长。

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
export FLUX_NATIVE_PYTHON="$PWD/.conda/gpu-train-stable/bin/python"
FLUX_NATIVE_MEMORY_TRACE=1 CUDA_VISIBLE_DEVICES=0 nohup bash smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh memory_alloc_trace_v6 > "logs/native_memory_trace_bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
echo "launcher PID=$!"
```

只在一个节点执行一次。输出目录：
checkpoints/foundir_flux_native_energy_memory_v1/memory_alloc_trace_v6/。
采集成功需NATIVE_ENERGY_MEMORY_SNAPSHOT_SAVED和快照文件SHA一致；
launcher status1仍表示计算失败，不是资格通过。旧目录全部保留。
