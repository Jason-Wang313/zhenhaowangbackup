# Rain续训：H100 80GB / H200及跨节点混用

当前续训入口已实现H100/H200代码支持，实际GPU验收仍待用户节点执行。
**不能把CPU通过说成H100显存或混卡NCCL已通过。** 历史H200约138GiB峰值不能继续当作H100通过证据，
本次没有GPU，不对其根因作已隔离声明，也不承诺新峰值。

## 支持范围和内存路径

- 4节点×每节点2卡，总8卡；每节点两卡型号一致，至少75GiB可见显存。
- 可全H100 80GB、全H200，也可不同节点分别H100/H200。不接受同节点混卡或低显存型号。
- 只更新当前native50 continuation，不修改旧warmstart、其它四专家、energy或任何权重/缓存。
- VAE注意力只将query按256分块，保留完整keys/values，没有图像裁块、裁掉上下文或降低512裁图。
- backward保存的非叶CUDA激活同步转存CPU，叶权重保持GPU；恢复原device/dtype供反传。
- 进程启动前固定cuDNN卷积workspace请求为256MiB、plan cache为64。此设置不是显存总量硬上限，
  其CUDA/cuDNN实际效果仍需实测。原真实FLUX block activation checkpoint保持。
- source池、样本位置、loss、lr1e-5、global32/micro1/累积4、LoRA/Adam FP32、FLUX BF16、50步口径不变。

CPU转存占额外主机内存和PCIe带宽，可能显著降低吞吐。没有H100 ETA；保持每节点8小时cap，
不因慢而自动扩预算。OOM仍失败退出并保存已经提交的状态，不自动降裁图或改batch。

## 运行身份与迁移边界

原始完整runtime继续保存。跨节点只允许GPU型号在H100/H200之间不同，NVIDIA release、
PyTorch/CUDA/cuDNN/transformers、LPIPS及评分依赖仍严格一致。
允许从绑定的父H200 step200加载原LoRA、Adam、rank RNG、6400采样进度；不会fresh Adam。
这保留优化器/数据进度，但更换硬件和VAE分块可能导致浮点舍入变化，不声称跨卡逐比特一致。
新run内部恢复仍逐节点比较原始runtime，禁止换GPU或换软件后假称严格恢复。
父评价缓存只读并绑定旧producer；新增teacher缓存记录新run/runtime，不伪称原H200生成。

源码与配置身份已变。兼容版首次使用新run_id：`continue2000_hopper_4x2_v1`，
launch_id：`launch_continue2000_hopper_v1`，入口仍
`smoke/start_foundir_rain_native50_continue_v1_tmux.sh <rank> <当前node0 IP> <run_id> <launch_id>`，端口29690。
不要用原run_id覆盖旧receipt。如果原continue2000_4x2_v1已训练到201或更后，不要退回200重算：
先报告其最高完整步、制定明确状态迁移，再启动兼容版；本次不停止或改动任何现有GPU任务。

目的地先执行轻量父状态预检，不扫描全部历史tensor：

```bash
CUDA_VISIBLE_DEVICES= .conda/gpu-train-stable/bin/python smoke/train_foundir_rain_native50_continue_v1.py preflight --require-parent --report reports/foundir_rain_hopper_destination_preflight_v1.json
```

首次H100运行需据缓存生成、完整optimizer step及一次完整评价的实际显存/有限值判断是否可用。
本提交只交付实现，不自动启动GPU、不重做旧资格、不自动扩大预算。
各里程碑和退出时只做轻量状态/指标/关键文件检查并推简报，不追加全量CPU审计。

## 已验证及未验证

16项CPU回归通过：原11项续训回归＋5项显存/硬件策略测试。
VAE分块在小尺寸CPU FP64上输出、输入梯度和参数梯度与原实现一致；
训练循环AST比较仅允许新增激活存储context，其余采样/loss归一化/优化器节点保持一致。
测试了H100/H200混合runtime签名、真实软件差异拒绝、非Hopper拒绝、工作区设置和CPU梯度路径。
Python AST、shell语法、diff检查通过。未执行CUDA转存、4B H100反传/峰值、混卡NCCL或速度测试。
CPU测试不证明完整GPU数值逐比特一致或画质成功，部署/开发门槛仍由真实结果判断。
