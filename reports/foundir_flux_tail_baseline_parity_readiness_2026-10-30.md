# 单节点两卡：FLUX历史基线差异定位

用户已授权继续下一步，并要求短任务用单节点两卡，长任务才跨节点。
本次独立短诊断不改V3资格/训练的绑定源码，也不改旧缓存与实验身份。
当前代理节点无GPU/远程启动通道；下方入口交付后仍须在一台GPU节点执行。

## 问题与固定范围

三节点资格 `qualification_3x2_v1` 在训练前六rank均报历史基线变化。
本轮只问：同一原训练输入，当前两卡与重复推理是否一致；相对旧输出究竟差多少。
不训练energy、不做GT引导、不读GT tensor，不计算PSNR/LPIPS或宣称学习/泛化结论。

固定原24训练来源中的index0（foundir_train_0188849），直接复用旧缓存observation。
预检SHA绑定失败尝试preflight、parent preflight、66份已有源码/配置、缓存内容、
Noise prompt、FLUX base、AE和step400 LoRA；缓存包含GT字节，哈希检查不解码GT。
运行时只提取observation/proposal，原seed、50步、512、reference time10、alpha=.30、
双uint8和原restore调用均保留。不切换后端或精度来迎合基线。

每张卡独立加载冻结模型，对同一来源重复两次。两worker不用DDP/NCCL，
旧shell里的NNODES/MASTER_ADDR/NODE_RANK不参与这个入口，无需tmux。
两个设备必须是实际空闲、不同UUID的H100/H200 SM90且各>=75GiB；允许混用，
但本次用户已提供节点都是双H200，不把代码允许混用当作混用实测。

采集内容：

- 历史uint8差异：不同通道元素数、不同RGB像素数/比例、最大/平均绝对误差、RMS与有符号均值。
  误差以0–255强度单位计，先转浮点再相减，避免uint8溢出。
- 同卡重复与跨卡比较：按执行顺序报告第一个不同阶段及全部阶段的误差。
- 编码参考、50步模型输入/时间/输出、首次条件与位置、最终latent、解码/量化张量。
- 每worker运行时、NVIDIA driver、设备身份、torch/CUDA/cuDNN/依赖版本与后端开关；
  逐次文件SHA、峰值allocated显存与时长。文件留在新诊断目录，不上传GitHub。

阶段采集在原restore外加只读hooks/包装，保留同一算术；会增加同步、CPU复制和落盘开销，
因此不将该耗时当成V3训练吞吐。异常也移除hooks，所有部分文件保留，不覆盖重试。
旧运行没有中间态：阶段比较只能定位当前重复/设备差异。若四次彼此一致却都不同于旧图，
需继续查历史环境与推理路径，不能直接锁定H200或torch版本，更不能换基线/关闭相等检查。

## 预算、验证与状态

粗估3–8分钟（.05–.1333普通小时，.10–.2667使用GPU-hours），尚未实测。
整个入口含预检/模型加载/生成/汇总硬cap900秒，即.25普通小时/.5使用GPU-hours，
另最多30秒清理，清理时间也计入最终实际消耗。约数百MiB至低个位GiB诊断文件，
实际大小以落盘为准。完成四次生成或遇错误/超时即停，不自动接续任何任务。

已通过5项CPU测试：uint8差异不溢出、首分歧/非有限值检查、拒绝其他来源/角色、
50步观测不改模拟输出及异常时恢复hooks、单节点入口忽略旧跨节点设置并拒绝重复卡。
Python静态解析、shell语法及真实单训练来源资产SHA预检通过；无新GPU实测结果。
现有66个V3绑定文件保持一致。没有为此重跑历史训练或宣称两卡V3训练已支持。

## 只在一台GPU节点执行

建议在原node0（10.247.94.142）的终端粘贴整块；另两台不用输入任何命令。
项目是共享目录，代码已在此工作区，不需要git pull或覆盖本地文件。

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
mkdir -p logs
nohup env FLUX_TAIL_V3_PYTHON="$PWD/.conda/gpu-train-stable/bin/python" CUDA_VISIBLE_DEVICES=0,1   bash smoke/run_flux_tail_baseline_parity_2gpu_v1.sh baseline_parity_2gpu_v1   >> logs/baseline_parity_2gpu_v1_bootstrap.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -n 60 logs/baseline_parity_2gpu_v1_bootstrap.log
```

输出：`checkpoints/foundir_flux_tail_baseline_parity_v1/baseline_parity_2gpu_v1/`。
同一ID已有输出时拒绝覆盖；失败重试先检查原因，不自动重跑或递增ID。
`BASELINE_PARITY_DIAGNOSTIC_COMPLETE historical_parity=false` 加 `END status=2`
表示诊断已完成、历史基线仍不一致；result.json保留量化误差和阶段证据。
`historical_parity=true` 加 `END status=0`仅说明本来源四次基线相等，
不等于24来源或V3八步资格通过。其它异常/超时及无COMPLETE均视为诊断未完成。

新文件：`smoke/diagnose_flux_tail_baseline_parity_v1.py`、
`smoke/run_flux_tail_baseline_parity_2gpu_v1.sh`、
`tests/test_flux_tail_baseline_parity_v1.py`。设计依据见研究记录第12节；
后续收到真实结果再更新失败归因，不提前填写GPU指标。
