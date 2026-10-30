# 新工作区交接：先完成迁移核验，再运行 V3

## 后续更新：本文是迁移前快照，不覆盖最新执行状态

迁移CPU验收与三节点执行适配后来已由新代理提交。实际当前计划以最新AGENTS.md、
current handoff及 `foundir_flux_tail_v3_cluster_readiness_2026-10-30.md` 为准；
下文“启动器仍写死旧目录”等是修复前事实，不要重复迁移或覆盖新代码。
研究依据另见 [设计演变与失败排查记录](foundir_energy_design_history_and_failure_playbook_2026-10-30.md)：
接手时除了命令，还须理解已证明的可达空间、尚未学会的方向，以及不能重复的失败。

日期：2026-10-30。此文与 AGENTS.md 顶部为当前操作依据；旧报告中的 ACTIVE/LIVE
只表示当时状态，不能覆盖本次迁移和最新 V3 任务。不要从历史 Rain / MoE / EBT 任务重新开始。

## 1. 一句话说明接下来做什么

**先从 GitHub 安全同步关键文件，核验新目录的运行环境、权重和缓存路径；
完成迁移验收后，运行 V3 的 8 步工程检查。通过后才考虑独立、从头初始化的
200 步训练拟合，不直接开始正式大训练。**

这不是更换研究方向。我们仍然训练一个小 energy（给候选打分并提供修改方向的模型），
引导冻结 FLUX 最后少量采样步骤。先解决它在训练图上都不会正确修正的问题。
不加入 Restormer，不恢复 MoE，不恢复从零训练的 EBT Generalist。

## 2. 唯一工作目录、GitHub 和 Python

- 新的唯一工作根：
  `/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
- 原目录：
  `/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang`。
  只作为历史迁移来源；不在那里继续实验，也不自动双向同步。
- GitHub：`https://github.com/Jason-Wang313/zhenhaowangbackup`，分支 `main`。
  用户已复制文件，之后旧目录的修改不会自动进入新目录。新代理必须读取远端新提交。
- GitHub 是代码、配置、决策和报告的权威记录，不是 TB 级数据、模型和缓存的备份。
  这些运行资产沿用已复制文件，并核验 SHA；不能因为 GitHub 没有就重新下载/生成整套数据。
- 原验证过的 Python 环境：
  `/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3`。
  它在项目文件夹外，复制 zhenhaowang 不等于复制此环境。新环境的实际路径尚未提供、
  也未验证。若原环境仍可见，可核验后只读使用；否则先确认可用等价环境及依赖，
  不凭空猜新路径、不退回系统 Python、不盲目 pip 升级覆盖原环境。
- 本次交接代理看不到新挂载目录，无法核验用户复制是否完整，也没有在新节点启动 GPU。
  这是交接完成，不是宣称新目录已经可以直接训练。

## 3. 新代理先从 GitHub 读什么

先执行只读同步检查：

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
git fetch origin
git status --short --branch
git rev-parse HEAD origin/main
git show origin/main:AGENTS.md
git show origin/main:reports/foundir_workspace_migration_handoff_2026-10-30.md
```

推荐阅读顺序：

1. 最新远端 `AGENTS.md` 的 CURRENT AUTHORITY。
2. 本文及 `reports/foundir_workspace_migration_inventory_2026-10-30.json`。
3. `reports/foundir_current_handoff_2026-09-23.md` 顶部当前迁移节和紧随其后的 V3 节。
4. `reports/foundir_flux_tail_relative_energy_v3_readiness_2026-10-29.md` 及同名 JSON。
5. `reports/foundir_flux_tail_energy_resumable_v2_terminal_2026-10-29.md`，了解旧失败。

注意仓库同时有大写 `AGENTS.md` 和小写 `agents.md`；前者是当前规则，
后者顶部已注明旧优先级失效。不要被旧文件要求“先完成 Generalist/Rain”等内容带偏。

源工作区 Git HEAD 落后远端，且有大量脏文件；很多本地 untracked 文件实际上已通过
隔离工作树提交到远端。不能据此断言未备份，更不能 `git add -A` 或 reset/clean。
目的地如果继承此状态：先比对远端与本地文件内容，保存有差异文件的备份/补丁，
再按明确文件清单合并。完全相同的文件不需要重复覆盖；冲突未解决不能启动 GPU。
必要时使用 origin/main 的独立临时 worktree 作为只读比对/提交来源。
不要盲目 git pull，不要将整个远端目录覆盖到含权重、缓存和本地修改的工作区。

## 4. 已经完成什么，尚未完成什么

上一版 V2 `qualification_8gpu_v2` 已正常结束，但图像质量失败，不能续训或部署：

- 8 张检查图原 FLUX：38.610993 dB / LPIPS .07752365。
- energy 一次修改：36.759309 / .07801964。
- 连续 16 次：19.787051 / .11268279。
- 主要现象是持续整体压暗，评分降低却不代表恢复质量提高。
- 这些是 V2 的检查图结果，不是新 V3 结果；检查图不回灌新训练。

V3 已实现：

- 原 24 个训练来源；旧 8 张检查图、校准池和最终 1859 条开发集不读像素。
- 冻结 Noise FLUX step400、50 步、512 裁图、alpha=.30、双 uint8 指标口径。
- 保留 1,010,274 参数 energy 架构，不增加 RGB 修复网络。
- 缓存候选的真实质量和图像导数，加强同图候选的相对质量/方向监督。
- 旧 V2 权重只生成训练反例；新模型随机初始化，训练中刷新自己产生的反例。
- 仅做一次真实 FLUX 尾段修改检查，同时报告原 FLUX、FP32 对照、原始建议、受限建议。
  限幅不是独立校准，不是“绝不损害画质”的保证；全部拒绝就是零收益。
- 每个完整 optimizer step 保存模型、Adam、RNG、采样位置和哈希；只支持同 world size 恢复。

迁移前已通过 67 项 CPU 回归测试和两进程二阶 DDP 精确恢复测试。
本次在旧目录仅找到 V3 的 CPU preflight / cache contract JSON；没有 V3 GPU 日志、
训练 checkpoint 或 result。新目录可能有新代理的更新，必须先检查，不能覆盖/重复启动。

**最终目标仍是 energy 相对匹配的原 FLUX 独立提高 PSNR .30 dB，
同时使 LPIPS 相对降低 9.928107050377874%。**
不能把旧 Restormer residual 的改善算进来，也不能把 GT oracle 当可部署效果。
V3 当前所有 probe 都是训练拟合，不是泛化或论文成功证据。

## 5. 最容易踩的迁移坑：不要直接运行旧命令

### 启动器仍写死旧目录

`smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh` 和
`smoke/run_foundir_flux_tail_relative_energy_v3_gpu.sh` 中的 ROOT 仍为旧目录，
后者 PYTHON 仍为旧环境。只是 cd 到新目录，不会改变它们的行为。

在目的地完成最小路径迁移：以脚本所在项目为根或显式确认新根；确认真实 Python；
确保日志、输出、TORCH_HOME、PYTHONPATH 和锁均使用新项目。此交接提交只改文档，
没有偷偷修改任何已绑定的训练源码或宣称迁移启动器已验收。

### 缓存有绝对软链接

例如旧 V2 的 `cache_000.safetensors` 指向旧 V1 目录的绝对路径。
普通复制可能保留这个旧链接；即使旧挂载仍可读，也可能违反代码的根目录边界检查。

在新目录核验所需链接的最终目标落在新工作根，且目标内容 SHA 与原记录一致。
保存原链接目标记录，针对确定的链接修复为新目录内目标或相对链接；
不要删除缓存，不要对整个项目做盲目全局替换。数据/模型路径也要检查是否越出新根。

### 源码与缓存绑定 SHA，不能为了“跑通”跳过核验

当前 V3 的 source binding 包含父实验源码及两个 V3 shell 启动器。
仅修改启动器也会改变 V3 身份；复制来的 `cache_v1/contract.json` 和
`qualification_v1/preflight.json` 可能因此不再匹配。

保存旧绑定、原文件与结果；记录旧根→新根、源码前后 SHA、资产内容 SHA、路径变化。
对明确只有 CPU 预检而无训练产物的 V3 准备记录，归档旧元数据后按迁移后的源码
重新生成准备记录；不可原地把旧 SHA 改成新 SHA 假装原实验未变。
如果目的地已有 GPU checkpoint/cache，则先审计恢复兼容性，不能按空任务处理。
仅换 run-ID 不能解决共享 cache contract 冲突，也不等于精确续训。
父 V1/V2 代码/配置尽量保持原字节不动；本轮不重跑它们。
迁移后必须重做 CPU/来源/缓存身份预检，不禁用失败即停止的检查。

## 6. 哪些东西已在 GitHub，哪些靠复制

代码基准提交：
`542e647c09f39d0d5769cde1402958698ed80581`。本次迁移提交在其之后，仅更新交接。

本次逐文件核对 V3 的 64 个 bound source 文件：

- 56 个项目源码/配置在 origin/main 中存在，并与源工作区逐字节一致。
- 8 个 third_party/flux2/src/flux2 Python 文件属于外部源码，未直接纳入本仓库。
  上游为 `https://github.com/black-forest-labs/flux2`，
  固定 commit `50fe5162777813d869182b139e83b10743caef15`。
  本地修改由已提交的 `patches/flux2-offline-smoke.patch` 保留，
  SHA-256 `b69370599196206c42c8fae418c2a790bbcfc5249e18edb791acc551a9e370d3`。
  当前绑定的本地 text_encoder 修改是 QWEN3_MODEL_PATH 环境变量支持。
  优先使用已复制且 SHA 一致的文件，不盲目 checkout 上游覆盖本地，不重复套补丁。
  第三方 LICENSE、运行环境依赖也须保留。
- 64 个文件都匹配迁移前 V3 preflight 记录；清单在同目录 inventory JSON。
  这不等于新节点实测，也不涵盖所有历史实验的本地脏文件。

必要的非 Git 资产至少包括：

- FLUX base、AE、Noise step400 LoRA、LPIPS AlexNet 权重。
- V2 `qualification_8gpu_v2` 的 result/preflight、step25 energy 权重和原24训练缓存；
  缓存链接指向的 V1 原始 tensor 文件也必须存在。
- 原训练数据、manifests、训练条件/缓存及身份记录。不要用另一套数据替换缺失文件。
- V3 已复制的准备文件，以及如果目的地已有运行则其缓存/逐步 checkpoint。
- 可执行 Python 环境和 tmux；`.tools` 复制成功不保证其动态库可在新节点运行。

精确路径/SHA 从 inventory、原配置及复制的 preflight 读取。不能为了检查资产
扫描或解码受保护图片。GitHub 不上传权重、图像、缓存、大 manifests、实时日志或凭据。

## 7. 迁移完成后如何继续

先在新根使用确认过的 Python，重做：

- Python / shell / JSON 静态检查。
- `tests/test_flux_tail*.py` 的 CPU 回归。
- `smoke/smoke_flux_tail_relative_energy_v3_cpu_ddp.py` 两进程恢复检查。
- 原24训练来源、父缓存、源码/模型哈希预检；不读取检查/校准/保护像素。
- 确认没有重复 GPU 作业，8 卡同型号 H100/H200、每卡至少75GiB，以及 tmux 可用。
- 将迁移实现和核验报告做 focused commit/push，再交付启动命令。

**以下是完成迁移验收后的入口语法，不是当前可以直接复制执行的启动许可：**

```bash
bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh qualification qualification_v1 8
```

工程资格：8 步，step4 刷新反例。检查有限梯度、显存、逐步保存、反例刷新与 8 张
训练图的一次实际修正。必须看到 `TAIL_RELATIVE_V3_RESULT_VERIFIED` 和
`TAIL_RELATIVE_V3_END status=0`；tmux 创建不等于训练正常。

迁移前预算估计：八卡35–75分钟 / 4.67–10 GPU-hours，硬上限90分钟 / 12 GPU-hours。
实际速度尚未验证；15 CPU 核可能够用，不承诺零影响，也不据此修改训练配方。

资格完成后，先报告实测速度和结果，再决定是否手动启动独立 fresh 200 步：

```bash
bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh trainfit trainfit_v1 8 qualification_v1
```

不得用资格权重初始化；200 步需匹配的 qualification 结果。粗估7–10小时 /
56–80 GPU-hours，八卡硬上限10小时；不能自动扩预算。
四卡兼容入口仍保留，使用独立 run-ID / 明确 qualification ID；默认有八卡就用八卡，
不跨节点、不改变 global batch32。恢复前核验同 world size 与迁移后的身份。

之后的独立来源校准、泛化评价和多步扩展未实现/未启动；等训练侧显示学会正确方向，
再冻结下一阶段协议，不自行加任务或读取最终开发集。

## 8. 可直接转交给新代理的任务

“请先 fetch origin/main，读取最新 AGENTS.md 和本迁移交接，而不是沿用复制目录中的
旧优先级。保留所有数据、权重和脏文件，在新的 multimodal-discrete-diffusion 根下
完成 V3 启动器、Python 和缓存链接的迁移/哈希核验。先检查是否已有新进度，避免重复。
通过 CPU/数据身份预检后，给我单节点八卡 H100/H200 的 V3 8 步工程验收命令。
不要恢复旧 V2、MoE、Restormer 或 EBT，也不要自动开始 200 步或正式长训。”
