# FLUX-only 25步验收：代码已实现，CPU通过，尚未GPU启动

**同日资源更新：本文件中的单机4/8卡默认已被用户改为3节点×每节点2张H100/H200。
当前启动/同步请读 `foundir_flux_native_energy_3x2_readiness_2026-10-31.md`；下方旧命令
是初版记录，不再是默认入口。全局batch8、512、完整三次更新和25步不变。**

## 实际状态

用户已要求execute/start，允许启动首轮工程验收；不自动启动200步或正式训练。
主会话没有nvidia-smi/CUDA GPU，新执行根也未挂载，因此没有替用户在GPU节点开跑。
本次实现、测试、推送新入口；不得用旧CNN启动器替代。

- 9项CPU测试通过：真实FLUX块代码的小尺寸模型（不是4B权重）验证生成路径不变、
  三步二阶梯度经冻结FLUX尾段传到energy LoRA、梯度有限、有限差分、训练/推理更新一致、参数白名单、
  动作覆盖、Adam/RNG精确续训和原子文件保护。
- Python AST、JSON、两个shell的bash -n、4/8卡tmux命令dry-run通过。
- 读取已冻结训练来源metadata，核对24个训练来源及crop/role；没有读取图片。
- **没有运行真实4B+VAE+LPIPS的CUDA训练**；显存、性能、NCCL和实际质量均未验收。
  CPU测试的感知函数为测试用可微函数，不能冒称已验证真实LPIPS端到端CUDA链路。

## 代码实际做什么

`lowlight_restoration/flux_native_energy.py`：一个共享的预训练FLUX底座，原Noise
LoRA冻结；在现有50个投影位置加入独立rank16 energy LoRA。候选图像tokens在
final_layer之前线性读出标量energy，五个VLM等级经线性投影输入。没有外挂CNN。
生成模式禁用energy LoRA，energy模式启用；checkpoint重计算显式恢复对应模式，
不原地切换权重。energy用支持二阶导数的math attention；生成尾段只需要一阶
输入梯度，保留原attention后端，避免无故改变原FLUX基线或扩大生成成本。

原FLUX恢复模型、原生VAE、LPIPS全部冻结；只有energy LoRA、条件投影、线性读出
更新。真实训练路径为：三次latent energy梯度更新 → 每次真实FLUX尾两步 → VAE
解码 → 连续alpha=.30图像的相对MSE/LPIPS及分别变差惩罚。另有终态动作排序与
真实尾段轨迹排序。没有分数/导数缓存拟合替代直接修后损失，没有截断三步反传。

候选缓存是本轮新建：原FLUX、输入、训练GT正例、欠恢复、过恢复、固定细节损坏、
真正重复FLUX恢复、真正Haze FLUX错误专家。都不来自CNN/Restormer。
Haze/Noise权重只在无梯度的离线候选阶段顺序加载；训练前恢复并冻结原Noise LoRA。
旧V2 preflight只提供已审计的24来源、裁图、种子和基础模型身份；不读其CNN权重、
CNN反例或CNN生成的修正方向。GT正例只训练t=0评分，不能进入部署选择。

所有实际报告继续双uint8、alpha=.30；训练使用连续图像损失，因为取整不可微。
probe报告基线、FP32零引导控制、1/2/3步、无GT的energy自选及逐图结果。
FP32控制差异必须单独看，不能把它自动算成energy收益。8图均来自训练来源，
不是泛化。最终+.30dB/LPIPS相对下降9.92810705%的目标没有降低。

## 规模和成本边界

- 先25个优化步，512裁图，完整三次更新；不自动200步。
- 单机4或8张同型号H100/H200（至少75GiB显存）；默认8卡。
- 全局batch8、单卡microbatch1：4卡累积2，8卡累积1。
  这是新4B二阶图的明确工程设定，不冒充旧CNN全局batch32的完全匹配训练量。
- AdamW lr1e-4、warmup5、weight_decay1e-4、clip1；rank/alpha16。
- 每次latent RMS步长.005，累计半径.02；选择energy margin .001。
  这些是预先冻结的训练侧设置，不是可靠no-harm的证明。
- GPU阶段（含候选生成、训练、probe）硬上限2小时；这不是预计时长，也不含之前
  的CPU资产核对。对应8卡16、4卡8 GPU-hours上限，另报告启动器总分配时间。
  timeout/崩溃不等于完成，必须看result.json与launcher END status。
- 不扩大预算、不降裁图/步数、不换CNN；若实际OOM/不支持高阶导数，保留失败证据。

每完整step保存不可变的energy参数、Adam、每rank随机状态、采样进度、运行身份和
源码绑定，并更新latest指针；不保存4B底座副本。终局再次加载真实checkpoint核验。
已有缓存按记录安全续建，断点不清理/覆盖旧实验。resume要求同GPU映射/运行环境/
world-size/源码/config；节点迁移需另行明确，不能将换节点重新初始化冒充续训。

## 新节点同步和启动

执行根：`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
Python默认该根`.conda/gpu-train-stable/bin/python`；不安装或升级训练依赖。
主会话不能访问此目录，CPU代理先从GitHub同步明确文件，保留所有本地脏文件、
数据、权重、缓存及活日志；冲突逐项核对，不reset/clean/盲目pull。

新运行文件（必须全部同步）：

- `configs/foundir_flux_native_energy_v1.json`
- `lowlight_restoration/flux_native_energy.py`
- `smoke/flux_native_energy_contract_v1.py`
- `smoke/flux_native_energy_runtime_v1.py`
- `smoke/train_foundir_flux_native_energy_v1.py`
- `smoke/run_foundir_flux_native_energy_v1_gpu.sh`
- `smoke/start_foundir_flux_native_energy_v1_tmux.sh`
- `tests/test_flux_native_energy.py`

另同步AGENTS.md、agents.md、当前handoff、本报告和新版PPT/README/生成源码。
保留已存在的原生FLUX third_party源码、LoRA/量化工具、tmux helper与冻结资产；
新preflight会绑定实际运行源码并检查基础权重、prompt、来源receipt的SHA。
不得重新下载Qwen或在GPU运行时编码prompt；复用冻结Noise/Haze prompt缓存。

同步后，在有8张空闲卡的GPU节点执行这一整行：

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_flux_native_energy_v1_tmux.sh qualification_v1 8 fresh
```

只有4张空闲卡时将8改为4，并使用相应CUDA_VISIBLE_DEVICES；启动器会拒绝数目不符。
已有同ID任务确实中断且身份未变化时，把fresh改为resume；不要重复新建同ID训练。
tmux session是`flux_native_energy_qualification_v1`，日志为
`logs/flux_native_energy_qualification_v1_<UTC>.log`。

判断实际开始：先有HARDWARE_PASS / PREPARED，再有CACHE_READY，随后必须看到
`NATIVE_ENERGY_CHECKPOINT_SAVED step=1`和`NATIVE_ENERGY_TRAIN`。
`TMUX_CREATED`只表示会话创建，不表示模型已训练。终局看
`checkpoints/foundir_flux_native_energy_v1/qualification_v1/result.json`。

CPU观察代理仍只同步/回报，除非用户明确赋予它GPU访问与启动职责；不能假称已经
代主代理启动不可见节点。请回传小JSON/报告到GitHub，不提交权重/候选图/大缓存。
