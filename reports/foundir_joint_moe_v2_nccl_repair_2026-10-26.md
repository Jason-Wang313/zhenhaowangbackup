# 联合 MoE V2：H200 通信启动修复

状态：用户已批准修复；代码和CPU启动流程测试通过，GPU恢复尚待节点验证。
本轮没有启动训练或读取训练／开发／受保护图片。

## 修复什么，为什么

上一轮 `image_only/qualification_v1` 在八卡 NCCL 初始化报 CUDA401，0步退出。
项目历史在H200上记录过相同错误，并定位到NVLS；关闭NVLS、保留普通P2P后恢复。
新V2启动器遗漏了这项处理。这是有依据的兼容修复，不等于已证明当前节点的根因。

现在在真正执行的runner内强制设置：

```bash
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
```

这不依赖用户终端或旧tmux server的环境继承。启动日志打印实际值。
没有关闭普通GPU点对点通信，也没有更换CUDA、PyTorch、驱动或模型。

## 先检查，再训练

复用已有 `smoke/cure_nccl_rehearsal_v1.py`，不改这个共享检查器：

1. 创建八卡NCCL通信组，检查八个rank的数值求和等于36。
2. 对一个极小线性模型做DDP反向传播，检查梯度有限。
3. 不读图、不加载恢复权重、不创建optimizer、不更新参数。
4. 180秒超时，另给30秒终止宽限；非零退出或超时立即阻止训练。
5. 通过后才启动原25步工程验收，不自动启动VLM臂或正式训练。

检查单独进程运行，不消耗正式训练进程的随机数序列。
详细NCCL日志只在检查中启用，不强制带入训练进程。
日志包含runner与检查器SHA-256，失败也有最终退出状态。

## 验证与边界

新增5项CPU shell控制流测试全部通过：

- 即使继承了相反设置，也强制得到NVLS=0、P2P_DISABLE=0。
- 通信检查先于训练，详细日志不污染后续训练环境。
- 检查失败或模拟超时时，训练不会执行。
- 通信成功后若训练失败，最终失败码不被吞掉。
- 已有日志保持原样，不会覆盖。

这是模拟命令的CPU测试，不是实际GPU通信验收；shell语法检查通过。
原有18项联合MoE／数据CPU测试重新运行全部通过，包括双进程DDP、完整参数量和精确续训；合计23项通过。
原模型、数据、损失、优化器、步数、配对比较协议全部保持不变。
没有新的PSNR／LPIPS、显存或GPU吞吐数据。

## 新命令与费用

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh qualification image_only qualification_nvls_v2
```

先看 `JOINT_MOE_COMM_CHECK_PASS`，再确认实际训练和最终 `engineering_pass`。
仅出现 `JOINT_MOE_TMUX_CREATED` 不代表训练开始。
旧失败ID/log保留，不覆盖。默认两臂qualification receipt改指新ID；仍支持显式传入路径。

原25步验收暂估 .20–.75 普通小时／1.6–6.0 allocated GPU-hours。
新增检查最坏额外210秒，即 .058333普通小时／.466667 GPU-hours。
若通过后正常训练，合计保守上限约 .81小时／6.47 GPU-hours；仍须用实际日志更新。
当前Codex节点没有GPU，未代替用户在远端启动或宣称验证成功。

## 复现身份

- 科学配置未改：`3bdd59b3fb3c010cc7a93ab299f3f67fe85cc5eb5a6f380f610d4338398baebe`。
- 训练器未改：`bc55e9d120ee26830b9aa1a919ee776fbbef582b839668253029b58b8a01e5ee`。
- 修复runner：`65eef6c6cd83c4c97aca06a9bea551e9937563a7592831d663ad889d8930d8db`。
- tmux starter：`6a39d3d5d0659fcdad837ed976eee14c9fba32f22d41a093c1d42493bb347981`。
- 复用检查器：`15d316240578d776c070d9a8f8d29c9afe09a28528616363afe4c7887823586f`。
