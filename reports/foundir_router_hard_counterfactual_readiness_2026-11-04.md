# 四张错误路由图：补模型自选专家的单选对照

## 结论与边界

CPU逐图报告已完成：GitHub f66143f。主代理独立重算了40图的分组均值、差距贡献和转变。
错误4张贡献整体平均PSNR损失1.116875dB，正确36张贡献.051591dB。
正确组LPIPS仍有21/36张退步，不能只修分类后就认为内部混合一定安全。
本轮只补缺失对照，不修改配方、不重新训练、不宣布VLM有效或失败。

## 为什么只需要四次生成

已有C：按正确类别使用独立专家的one-hot对照；已有S：step408模型的soft混合输出。
36张主专家正确，模型自己的hard与C相同，已有数据足以分析soft的增减。
剩余4张主专家错误，只缺H：模型自己已选主专家的one-hot。
固定这4张、固定已有argmax，不重新选路线。

| 开发图key末尾 | 真实类型（仅用于报告） | 模型已选专家 | 补测 |
|---|---|---|---|
| 0144583 | Noise | Blur | Blur one-hot |
| 0189233 | Noise | Blur | Blur one-hot |
| 0508782 | Low-light | Blur | Blur one-hot |
| 0479048 | Low-light | Noise | Noise one-hot |

所有4图使用所选专家自己的50步、prompt、alpha与量化，均不走Rain第26次端点。
使用已有五FLUX LoRA及原生VAE，不加CNN、Restormer、refiner或新权重；不是五张成图平均。
GT仅在完整生成后算同口径PSNR/LPIPS，不决定专家、输出或接受。

报告分开给出：选错专家及其协议的差距H-C，以及soft混合新增的差距S-H。
具体存储使用PSNR=C-H和H-S、LPIPS=H-C和S-H，使两项均正值表示变差。
有改善时保留负值，不截断、不用GT回退，不把四张错例均值当40张总成绩。
H-C仍包含prompt/alpha/生成协议变化，不是纯分类因果效应；也不是VLM消融。
驱动或Hopper型号差异必须保留，不能宣称跨硬件bitwise等价。

## 入口与运行条件

- 代码：smoke/diagnose_foundir_router_hard_counterfactual_v1.py。
- 固定输出：checkpoints/foundir_router_hard_counterfactual_v1/predicted_hard_four_v1。
- 固定输入报告SHA256：911c411710d96498dcdf6d8af2e1abe772c52b95bc3b79d948b485aba5bbf6c5。
- 旧receipt/result和120个评价JSON只做身份核对；不加载409份checkpoint、不重算旧40图。
- 一台节点2张同型号H100/H200、每卡至少75GiB。优先原H200环境。
  H100推理仅实现支持、尚未实测，不能推广为完整轨迹训练支持。
- 两个独立worker各两图，无DDP训练、无优化器，无额外Qwen特征计算。
- 保留原TF32关闭、VAE query chunk256、FP32 Rain/BF16其它LoRA和LPIPS实现。
- 当前任务累计硬上限30分钟/1使用GPU-hour，另最多30秒清理；不是ETA。
  若独占更多卡，应另报整机资源占用成本。失败尝试不清零，不自动扩预算。
- 旧文件不改；新receipt绑定源码。每图不可变保存uint8及SHA、指标、路线、显存、耗时。
  同命令恢复跳过有效完成项；损坏则报错，孤立tensor保留，禁止覆盖。
- 节点锁与共享run锁防止重复；不杀旧任务或占用中的无关进程。

目的地CPU代理先安全同步相关GitHub文件，保留脏文件和大资产，然后：

```bash
.conda/gpu-train-stable/bin/python -S smoke/diagnose_foundir_router_hard_counterfactual_v1.py prepare
```

预检仅标准库、只读元数据；写本任务新receipt，不用GPU。必须输出ROUTER_HARD_PREFLIGHT_PASS。
随后CPU代理给用户单节点启动命令（不是原四节点训练入口）：

```bash
bash smoke/start_foundir_router_hard_counterfactual_v1_tmux.sh
```

日志：logs/foundir_router_hard_counterfactual_v1.log。
每次尝试成本：新输出目录attempts/*.ended.json；若started缺ended，先查日志记账再继续。
终态：ROUTER_HARD_RESULT_VERIFIED及ROUTER_HARD_END status=0。
小报告：reports/foundir_router_hard_counterfactual_terminal_2026-11-04.json。
CPU代理只推小报告和简短成本/退出状态总结，不上传图片、tensor、模型、实时日志。

## 已验证与尚未验证

本地22项CPU测试通过（本轮11项＋原分析11项）、两shell入口语法通过。
覆盖真实报告固定四图、模型argmax不被GT替换、错seed拒绝、概率/指标有限、
两worker分片、正负误差分解、断点哈希、完整四图文件汇总及不可变输出。没有改变任何旧绑定源文件。
本机看不到目的地挂载，没有CUDA实验结果；目的地CPU预检及真实2卡推理仍待执行。
额外尝试用本机旧HDD3环境导入完整推理依赖，约3分钟仍处于文件系统I/O等待
（进程D状态/cxiWaitEventWait），已仅终止该本地导入检查，exit143；未得到导入通过结果，
也没有得到Python异常。不能把这个旧环境检查当作目的地运行环境验收通过或失败。
完成后先分析报告再决定是否修分类/稀疏混合，不自动启动正式训练或扩大任务。
