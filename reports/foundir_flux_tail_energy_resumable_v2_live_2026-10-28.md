# FLUX-tail energy：八卡启动与逐步保存验收

## 结论

新任务 `qualification_8gpu_v2` 已在8×H100 80GB上真实训练。独立核验至step2：
模型、Adam、8份随机状态、采样进度和全部已提交检查点都完整。训练继续，25步资格
与后续评价尚未完成，不自动启动正式训练。这份记录不改变运行代码或科学方案。

## 切换与保留

- 新启动时间：2026-10-28T23:53:14Z。
- 旧任务 `qualification_4gpu_v1` 最后日志step7，于23:53:44Z主动停止，
  launcher status1是指定切换的结果，不是新任务报错。
- 旧32张图缓存全部复用；旧目录、日志及切换快照保留。
- 旧代码step25前不保存权重/Adam，因此旧7步内存状态无法恢复。
  新任务从相同初始化重算；不得声称旧训练进度无损迁移。
- 旧运行3376秒，.9377778普通小时、3.7511111使用GPU-hours；
  若按独占八卡整机计费，为7.5022222整机GPU-hours。

## 实际检查

八卡型号检查通过，rank0–7全部READY。每步保存前会汇集并核对八卡参数SHA；
任何rank缺失或参数不一致都会失败。step2独立CPU只读回读确认：

- step0、step1、step2文件SHA256与各自不可变receipt一致。
- 模型参数全有限，参数哈希已不同于初始化。
- Adam有236个参数状态，全部step=2，动量有限。
- next_global_slot=64；warmup状态与step2相符。
- 8份Python/CPU/CUDA RNG存在，训练日志完整。
- 八卡峰值allocated显存均31.891518115997314GiB。

step1为125.52943746501114秒，step2为124.69140653300565秒；
梯度范数分别.3372342884540558和.546185314655304，均有限。
相比旧四卡约246–250秒/步，接近两倍加速；没有改变global batch、数据或损失。

step2模型文件SHA256：
`9cec1b0b3c0c2452873688036d7268c43c87e8476f59a97cd014ea3a63fae4d9`。

## 时间、边界与下一检查点

核验时剩余23步，按前两步平均约47.96分钟，即.7993普通小时/6.3945使用
GPU-hours，另加评价时间。这不是全流程完成时间承诺。仍保留原每次90分钟/
12使用GPU-hours预算上限（清理宽限另计），不自动重启或扩预算。

训练中每个完整step保存，重启可读取最高完整receipt；尚未完整提交的一步
可能重算。CPU二阶DDP恢复测试通过，但此次真实CUDA中断重启尚未实测；
不为测试恢复而中断当前正常训练。

无OOM/NaN/NCCL失败证据，但这只证明工程上正常。25步训练和8图评价未完成；
早期方向cosine接近零，不能说energy已学会，更不能声称PSNR/LPIPS达标。
GT仅供训练监督及推理结束后计分；未读取保护数据或最终1859开发集像素。

下一终态须同时核验25步checkpoint、8图结果、`TAIL_ENERGY_V2_RESULT_VERIFIED`
和`TAIL_ENERGY_V2_LAUNCH_END status=0`。本次报告不授权formal。

启动代码快照：`d2c6f6bf8cbe82e82f58105bee4ca21652ae21e0`。
日志：`logs/foundir_flux_tail_energy_resumable_v2_qualification_8gpu_v2.log`。
