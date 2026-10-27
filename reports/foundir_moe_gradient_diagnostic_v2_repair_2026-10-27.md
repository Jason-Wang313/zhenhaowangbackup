# MoE梯度诊断：V1工程失败与隔离V2修复

## V1终局

MoE diagnostic_v1在8×H200正常通过GPU/通信检查，但首批梯度抽样触发CUDA
index out of bounds。启动器于2026-10-27T10:54:47Z以status=1退出，
总234秒=.065小时=.52 allocated H200 GPU-hours。
仅写出launch.json，没有result.json或完整样本诊断结果；零optimizer steps。
这是新诊断脚本的工程错误，不是模型科学失败、用户命令错误、OOM或NCCL失败。
旧日志与目录保留，不能原ID重跑，不能使用这次失败推断Blur瓶颈。

## 已复现的根因

实际专家输出矩阵shape=[3072,9216]，共有28311552个元素，
合法最大索引是28311551。V1使用默认FP32的
torch.linspace(0,n-1,16384).long()生成梯度抽样位置。
FP32不能精确表示这个末端整数，会把最后位置舍入到28311552，即数组外。
CPU按同一尺寸复现：一个越界索引。CUDA异步执行使异常在后续isfinite检查才显示。

此前小张量单元测试未覆盖这一边界；这项遗漏属于实现问题。

## V2的唯一实质修复

用整数算术生成位置：

```python
torch.div(torch.arange(k, dtype=torch.int64) * (n - 1),
          k - 1, rounding_mode="floor")
```

单元素情况单独返回0。保持相同来源、同一crop、冻结的两个step10000模型、
相同7项损失与代表性参数探针。仍然零optimizer steps、不访问开发/受保护像素。
固定坐标仍为抽样余弦，而非全网络的精确余弦；修复后的整数位置由明确公式定义，
不声称与错误FP32位置逐项完全相同。

为保护正在运行的PPT资格及其源代码hash，**不修改V1脚本、共用runtime、
配置、旧启动器或PPT训练代码**。新增隔离V2脚本与专用启动器：

- smoke/diagnose_foundir_joint_moe_train_gradients_v2.py
- smoke/run_foundir_moe_gradient_diagnostic_v2_8gpu.sh
- smoke/start_foundir_moe_gradient_diagnostic_v2_tmux.sh
- tests/test_moe_gradient_diagnostic_v2.py

V2来源绑定额外包含上述新诊断和两个启动器；继承原模型/数据/配置绑定。

## 验证与执行边界

4项CPU回归测试通过：实际矩阵边界重现与修复、小矩阵/单元素边界、
超过FP32精确整数范围的大尺寸、梯度诊断不改变参数且识别相反方向。
Bash语法通过。真实权重/来源预检查已通过：Rain/Blur各16来源、
原配置与全部冻结权重hash匹配，零图片读取。
CUDA修复后尚未实测；不能把CPU通过说成GPU任务已经成功。

用户在得知失败后要求重新给命令并明确“execute”，授权此工程修复及新ID重试。
Codex当前无CUDA或远端执行通道，无法替用户在GPU节点启动。
在之前运行MoE的空闲8×H100/H200节点执行，不要占用仍跑PPT的节点：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_moe_gradient_diagnostic_v2_tmux.sh diagnostic_v2
```

预计.25–.75普通小时、2–6 allocated GPU-hours，实耗以新日志为准。
日志：logs/foundir_dual_retry_v1_moe_diagnostic_v2.log。
结果目录：checkpoints/foundir_dual_retry_v1/moe/diagnostic_v2。
会检查空闲八卡、节点锁及有界通信测试；不自动启动任何训练。
必须看到MOE_TRAIN_GRADIENT_COMPLETE、result.json和DUAL_RETRY_END status=0，
才能认为诊断完成。创建tmux不代表成功。旧diagnostic_v1不得续跑或覆盖。

V2诊断SHA-256：08879b70ec5b22883b38a7a064ed89c88fe3115f5a83c45bd6b02a96909e6ec8。
V2 runner SHA-256：22261992c23801ebd0fb77030cc01a058a2e5c1d9c28d64b857aaedc89c9188a。
V2 tmux starter SHA-256：2fc7dd47fe3f8e4d2d4023f90a3ce149d21c1fcc1a0ee5780b92827623406802。
