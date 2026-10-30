# V3 新目录迁移：CPU 验收通过，GPU 待验收

2026-10-30。已在新工作根完成迁移、资产身份核验和 CPU 测试，可以交付八卡
8 步工程验收入口。当前节点没有 CUDA / nvidia-smi，没有创建训练 tmux、启动 GPU
或开始 200 步训练。CPU 通过不代表 CUDA 显存、速度、基线复现或科学目标通过。

## 同步与保留

- 从 GitHub fetch 到 `280c490`，读取最新 AGENTS.md、迁移交接、inventory、当前
  handoff、V3 readiness 和旧 V2 terminal。
- 主工作树 HEAD 仍为 `9ff4522`；开始时 230 项已跟踪修改、1,112 项未跟踪文件。
  没有 pull/reset/clean。64 个绑定源码原本全部匹配远端/交接 SHA，无需覆盖。
- 新增两份远端迁移交接文档；小写 agents.md 前置当前权限说明，保留本地旧内容。
  无关本地修改、数据、模型、缓存及旧结果均保留。
- 本次原文件、原元数据、工作树状态和 tracked diff 保存在
  `checkpoints/foundir_workspace_migration_v3/20261030T131929Z/`，不上传 GitHub。

## 路径与运行环境

唯一工作根：
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。

两个 V3 shell 启动器改为从自身位置确定 ROOT，要求显式提供绝对路径
`FLUX_TAIL_V3_PYTHON`，不回退系统 Python。tmux 命令显式携带解释器、卡数、
CUDA_VISIBLE_DEVICES 和 run ID，清除旧 PYTHONHOME；运行阶段固定新根的
PYTHONPATH、TORCH_HOME、日志和输出目录。原父实验 V1/V2 绑定源码不变。

原 sibling 环境不存在。实际发现并完成 CPU 验收的解释器为：
`.conda/gpu-train-stable/bin/python`（新根下）。其复制文件权限原为 0644；仅恢复
这一个解释器至 0755，文件内容未改，没有安装、升级或覆盖依赖。
旧 `.venv` 配置仍指向更早目录，没有把它们当作已验收环境。

实测 Python 3.12.13、torch 2.7.0+cu128、torchvision 0.22.0+cu128、
torchmetrics 1.9.0、safetensors 0.8.0、transformers 5.14.1。
项目的 LPIPS 使用 torchmetrics；单独 lpips 包缺失不影响此调用链。
FLUX util 和实际 tail runtime 导入通过，离线真实 LPIPS CPU 前反传有限且非零。
此环境是新目录 CPU 验收结果，不声称与旧 sibling 环境完整逐包相同。
GPU 节点仍须能执行该解释器，并通过实际同型号 4/8 卡、每卡至少 75GiB、空闲检查。

## 哈希、缓存和新绑定

147 项 SHA 检查通过：原 64 个源码、FLUX base、AE、Noise step400 LoRA、
LPIPS AlexNet、两个 manifests、V2 result/preflight/step25、第三方 patch、24 份
父训练缓存、原 24 来源的 48 个 GT/LQ 文件，以及 Noise prompt context。
GT/LQ 仅按文件字节求哈希，没有解码。没有读取旧 8 检查图、校准或保护像素。
第三方 LICENSE 保留。

所需 24 份父训练缓存均为新根内的普通文件；不是指向旧目录的软链接，因而
没有改写缓存链接或重新生成缓存。所有已核验资产的最终路径都在新根内。

目的地 V3 原来只有 CPU 元数据，没有训练 tensor、checkpoint、probe 或 result。
持有 V3 缓存锁后，将三个活动准备 JSON 另名归档为
`pre_migration_20261030T131929Z_*`，保留已有 pre_gpu_revision1 文件；按新源码
重新生成共享 contract 和独立 4/8 卡 preflight，没有修改旧 JSON 内的哈希冒充续训。

64 个绑定文件仅两个 V3 shell 变化，另外 62 个保持原 SHA。前后 SHA 和两个
新 protocol SHA 见同名 JSON。配置、训练代码、energy 架构、8 步/step4 刷新、
global batch32、来源、阈值及预算均不变。没有 GPU 进度可迁移或声称精确续训。

## 验证

- 67/67 CPU 回归通过，无跳过、失败或错误。
- 两进程真实二阶 DDP 精确恢复通过，恢复后下一步参数逐元素相同。
- 64 个绑定文件 Python/JSON/bash 静态检查通过。
- 4/8 卡真实训练缓存/来源/SHA 预检及 receipt 重新读取通过。
- 隔离的 shell 测试验证跨 cwd、带空格路径和旧环境下的 tmux 参数传递；
  4/8 均正确，缺少解释器在创建任务前拒绝。测试只执行临时桩，不启动 GPU。
- CPU 回归文件、DDP 脚本和历史 readiness 共 10 文件与 origin/main 字节一致。
- 当前 CUDA available=false、device_count=0；本次消耗 0 GPU-hours。

CPU 检查可在新根用下列命令复核（不是训练命令）：

```bash
env -u PYTHONHOME PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 .conda/gpu-train-stable/bin/python -B -m unittest discover -s tests -p 'test_flux_tail*.py' -v
env -u PYTHONHOME PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 .conda/gpu-train-stable/bin/python -B -m torch.distributed.run --standalone --nproc_per_node=2 smoke/smoke_flux_tail_relative_energy_v3_cpu_ddp.py
```

## 下一步：用户 GPU 节点上的八卡工程验收

以下入口已迁移；本次仅准备，没有执行：

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
FLUX_TAIL_V3_PYTHON="$PWD/.conda/gpu-train-stable/bin/python" \
  bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh qualification qualification_v1 8
```

日志 `logs/foundir_flux_tail_relative_energy_v3_qualification_v1.log`。
需同时看到 `TAIL_RELATIVE_V3_RESULT_VERIFIED` 和 `TAIL_RELATIVE_V3_END status=0`。
同命令只支持同 world_size 恢复；若 GPU 节点缺解释器或绑定不匹配，应修复该项，
不能换系统 Python、跳过哈希或改 run ID 掩盖冲突。

八卡粗估 35–75 分钟、4.67–10 使用 GPU-hours，尚非新节点实测；硬 cap
5400 秒/12 使用 GPU-hours，另最多 30 秒清理。4 卡仍使用独立 ID
`qualification_4gpu_v1`，硬 cap 10800 秒。不能自动重启或扩预算。

所有 probe 都是 training_fit_only。8 步只检验工程链路；200 步须另行手动启动、
匹配 qualification 工程结果且 fresh 初始化，不复用资格权重。当前没有启动它。
最终目标仍为 energy 独立 PSNR +.30dB / LPIPS 相对下降 9.928107050377874%。
