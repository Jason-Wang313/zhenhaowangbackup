# Project agent instructions

## CURRENT AUTHORITY: short work on one two-GPU node — 2026-10-30

User now requests single-node TWO-GPU execution for short diagnostics/jobs;
reserve cross-node execution for long-running work. This overrides older
4/8-only and blanket 3x2-default rules for short work. Mixed H100/H200 remains
allowed with actual identity checks. Do not reduce global batch or silently
reinterpret six-rank training checkpoints as two-rank resumes. Current V3
qualification/trainfit launchers remain 3x2/1x4/1x8; this update adds a separate
short diagnostic, not an unverified two-rank training mode.

User said to continue with the next diagnostic: one original training source,
two repeats on each of two local GPUs, original frozen baseline only. Read
`reports/foundir_flux_tail_baseline_parity_readiness_2026-10-30.md`.
Entry: `smoke/run_flux_tail_baseline_parity_2gpu_v1.sh` with explicit qualified
FLUX_TAIL_V3_PYTHON, CUDA_VISIBLE_DEVICES=0,1. No tmux, torchrun, MASTER_ADDR,
NODE_RANK or inter-node networking required. Estimated 3–8 minutes (unmeasured),
cap900 seconds/.5 used GPU-hours plus30s cleanup. Fresh diagnostic output root;
all old V3 bound code/cache/contracts remain unchanged. No learning, GT tensor
decode, parity relaxation, baseline substitution or automatic qualification.
CPU observer cannot launch on GPU nodes; GPU diagnostic remains pending until
its actual log/results appear. Status2 with DIAGNOSTIC_COMPLETE means measured
historical mismatch, not missing diagnostic evidence. Partial/error runs remain.

## CURRENT TERMINAL: V3 first 3x2 attempt stopped at baseline parity — 2026-10-30

Read `reports/foundir_flux_tail_v3_cluster_baseline_failure_2026-10-30.{md,json}`.
qualification_3x2_v1 ended on all nodes at 18:00:14–15 UTC, status=1.
Six H200 ranks passed startup shared-storage/NCCL checks and completed 24 base
records (1706 frames), then all failed `cached historical baseline changed`
in initial mining, before energy-guided updates. Zero optimizer steps/probes,
no training checkpoint/result. This is not evidence of energy learning failure.
66 source hashes and 24 base metadata hashes/identities rechecked successfully;
tensor contents were not rehashed in this terminal status audit. Difference
magnitude was not saved; runtime/kernel/hardware/path cause remains unisolated.
Total used cost: 1420 GPU-seconds = .3944444 GPU-hours; wall span 239 seconds.
Preserve everything. No auto-retry, relaxed parity, cache replacement or trainfit.
Next proposal is one allowed training-source baseline-localization diagnostic,
with separately scoped command/budget; none started by this status audit.
This terminal snapshot supersedes older GPU-pending statements, not 3x2 authority.

## REQUIRED CONTEXT: design history and failure reasoning — 2026-10-30

Before interpreting V3 quality results or changing its scientific recipe, read
`reports/foundir_energy_design_history_and_failure_playbook_2026-10-30.md`.
It connects the original PPT, residual/selector attempts, FLUX-tail GT evidence,
failed learned V2 and current V3; it records what each result does and does not
establish and how to localize the next failure. Do not restart old experiments
or count residual/GT gains as learned-energy gains. Do not interrupt a live job
merely to read it. This context requirement does NOT override the current
hardware/runtime authority immediately below or authorize additional trials.
At each material milestone, append evidence and the next falsifiable decision
to the reasoning record as well as the runtime handoff; push focused updates.

## CURRENT AUTHORITY: user-authorized 3x2 cross-node V3, mixed Hopper allowed — 2026-10-30

The user explicitly changed the available allocation to three nodes with two
GPUs each and authorized cross-node H100/H200 mixing. This supersedes ALL
older single-node / homogeneous-only / default-eight-GPU restrictions for V3.
Latest supplied node evidence: all three currently expose two H200s, >=75GiB,
with the same GPFS project mount and executable project-local Python.
Map node ranks 0/1/2 to 10.247.94.142 / 10.246.88.49 / 10.244.58.183.
These are user-provided identities, not an agent-run CUDA/NCCL qualification.

Read `reports/foundir_flux_tail_v3_cluster_readiness_2026-10-30.{md,json}`.
V3 now defaults to 3x2=6 ranks, retaining 1x4 and 1x8; mixed H100/H200 SM90
with >=75GiB each is allowed, including within a node. All nodes require the
same code/runtime and shared project storage. No architecture/loss/metric,
source, crop, seed, threshold or training-step changes are authorized here.
Global batch remains 32: rank counts 6/6/5/5/5/5, EACH pair weighted world/32
before DDP's rank average. Never drop/pad pairs or average rank means equally.

Default run is qualification_3x2_v1, eight steps with mining refresh at four.
New output root is checkpoints/foundir_flux_tail_relative_energy_v3_cluster;
all prior V3 artifacts remain frozen in their old root. Five V3 bound files
changed and two helpers were added; 59 prior bound files, including all V1/V2
producers and the energy architecture/loss, retain their prior hashes.
One torchrun/group spans all stages; only global rank zero locks/writes shared
preparation metadata. Each node has its own locks and appended log. Fresh
per-rank shared-storage tokens and actual NCCL reductions precede data access.
Resume binds global/local ranks, host/GPU UUID/model and runtime identities;
changing the mapping requires separate audit, not a claimed exact resume.
Same-topology CPU resume is exact; mixed-GPU CUDA bitwise equivalence is NOT
claimed. Keep every complete-step checkpoint and all six rank RNG states.

75 CPU regressions and a real six-rank Gloo test via three two-worker torchrun
agents passed: serial 32-pair mean/gradient equivalence, durable model/Adam/RNG
exact resume, rank agreement and rejection of incompatible resume. That test
ran on ONE physical CPU host. 4/6/8 train-only metadata/hash preflights passed.
Actual three-node CUDA/NCCL, memory, throughput and FLUX parity remain pending;
no GPU job was started by this preparation, no training-fit metrics exist.

Launch the same run on all three user GPU nodes with explicit NODE_RANK=0/1/2,
NNODES=3, MASTER_ADDR=10.247.94.142, MASTER_PORT=29683 and the qualified
FLUX_TAIL_V3_PYTHON. Commands are in the readiness report. No automatic SSH,
remote job kills, retries, trainfit or formal training. Six-card qualification
hard cap is 7200 seconds / 12 total used GPU-hours (+30s cleanup), preserving
the prior GPU budget. Count SUM(2*node_elapsed) across three node logs, not
six times each log. Startup/rendezvous waiting counts against each node cap.
Do not promise the old single-node ETA. Independent/manual fresh trainfit is
still gated by matching qualification; six-card cap 48000 seconds / 80 GPU-hours.

## CURRENT VERIFIED: destination CPU migration complete; GPU pending — 2026-10-30

This section updates the migration prerequisites below. Read
`reports/foundir_flux_tail_v3_workspace_migration_2026-10-30.{md,json}`.
The new workspace was actually checked: 147 SHA checks, 67 CPU tests,
two-rank second-order exact resume, and regenerated 4/8-GPU source/cache
preflights passed. V3 had only CPU metadata; old metadata was archived,
not rewritten. No V3 GPU run/result exists from this session; CUDA count=0.

The two V3 launchers now derive ROOT from their script location and require
an explicit `FLUX_TAIL_V3_PYTHON`; they no longer hardcode the old workspace.
The destination's `.conda/gpu-train-stable/bin/python` passed CPU runtime
checks (Python 3.12.13, torch 2.7.0+cu128). Only this copied interpreter's
execute permission was restored, 0644→0755; dependencies were not changed.
Its CUDA execution and baseline parity are still unqualified. Do not use the
stale `.venv` configurations or silently substitute system Python.

On the user's homogeneous eight-H100/H200 node, from the new root, the
prepared engineering command is:
`FLUX_TAIL_V3_PYTHON="$PWD/.conda/gpu-train-stable/bin/python" bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh qualification qualification_v1 8`.
This session prepared the command only; no GPU task or long training started.
Keep the 5400-second / 12-used-GPU-hour qualification cap and all train-only
restrictions. No automatic trainfit/formal. Four-GPU compatibility remains.
Unrelated local changes and all runtime assets remain; the main checkout was
not reset/pulled, and focused updates are committed from an isolated worktree.

## CURRENT AUTHORITY: new-workspace migration and V3 handoff — 2026-10-30

- The sole active project root is now
  `/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`.
  This explicitly supersedes all older HDD3 / agent-driven-world-model workspace
  rules below. The old workspace is a frozen migration source, not a fallback
  execution directory. Do not launch or resume jobs there.
- GitHub `Jason-Wang313/zhenhaowangbackup`, `origin/main`, is the authoritative
  source for code, decisions and handoff updates. The user already copied the
  files to the new root; subsequent edits in the old root do NOT synchronize.
  Preserve copied weights/data/caches and all unrelated dirty files. GitHub
  does not contain those large runtime assets and cannot replace their copy.
- Read `reports/foundir_workspace_migration_handoff_2026-10-30.md` FIRST, then
  the top of `reports/foundir_current_handoff_2026-09-23.md` and the V3 readiness
  report. Lowercase `agents.md` and all older ACTIVE/LIVE headings are historical
  wherever they conflict with this section. Never revive their old priorities.
- Immediate next action: safely synchronize the relevant GitHub files, verify
  the copied assets, fix and qualify relocation of paths/runtime/cache links,
  then run the V3 **8-step engineering qualification** on one 8xH100/H200 node.
  Do NOT run the current launcher verbatim: its ROOT/PYTHON still reference the
  old workspace, and changing bound files requires a recorded migration.
- V3 is the train-only relative FLUX-tail energy repair, NOT the failed V2
  25-step run, old GT-only oracle, Restormer residual, MoE, Rain or EBT Generalist.
  Latest verified evidence at this handoff: 67 CPU tests and CPU DDP exact
  resume passed before migration; no V3 GPU log/result was found in the source
  workspace. Inspect the destination before assuming no newer run exists.
- Only after a matching qualification passes may a separately/manual-launched,
  fresh 200-step train-fit run be considered. No qualification-weight reuse,
  no automatic formal training, no automatic budget increase. Final energy-only
  goals remain +0.30 dB PSNR and 9.928107050377874% relative LPIPS reduction.
- Default: all eight GPUs on one homogeneous H100/H200 node; retain four-GPU
  compatibility. No cross-node jobs. 15 CPU cores / 500GB RAM is acceptable to
  qualify, not a measured zero-slowdown guarantee. Preserve global batch 32.
- The last validated Python environment was
  `/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3`.
  Its availability on the new node is UNKNOWN, and copying the project folder
  does not copy this sibling environment. Verify the interpreter/dependencies;
  do not guess a new environment path or silently use system Python. A missing
  runtime blocks GPU launch until an equivalent environment is qualified.
- Never use official/sealed/internal/parent-test or final-1859 development
  pixels for training, tuning or selection. V3 currently uses only its original
  24 training sources; all probes are training fit, not generalization evidence.
- Report in simple Mandarin: outcome first, facts versus hypotheses separated.
  Commit/push focused milestones using explicit file lists and secret scanning;
  never upload checkpoints, datasets, caches, credentials or live logs.

## ACTIVE: train-only relative FLUX-tail energy repair V3 — 2026-10-29

用户在第一性原理分析后明确授权execute。本节覆盖旧V2终态的“没有新任务授权”；
不恢复旧25步任务、Restormer、MoE或EBT Generalist。当前执行修复的第一阶段：
先确认energy在训练来源上能学会一次正确修改；不是直接上正式泛化训练。

保持冻结Noise FLUX step400、50步、512图、alpha=.30和双uint8口径。
energy architecture保持原1,010,274参数，不加RGB输出网络。仅原24训练来源可读，
旧8张检查图、calibration池、最终1859开发集及所有保护像素均不读。
所有训练集probe明确标记training_fit_only，不得包装为来源分离/泛化指标。
用户最终+.30dB和LPIPS下降9.928107050377874%的双目标不变。

新训练从fresh模型/Adam开始；旧V2 step25只作为冻结反例生成器，绝不seed新训练。
初始反例在24训练来源上由旧energy自行走4步生成，再用训练GT标注好坏；
不先假定所有生成候选都是坏图。学习期间也刷新当前模型自身产生的4步压力轨迹，
这些无边界轨迹只是训练反例采集，不是部署时允许连续走4步。
现有候选的log MSE/log LPIPS与真实RGB导数只计算一次，FP32缓存。
一半训练pair含self-mined候选；加强同图数值差分监督、保留低权重绝对分数、
梯度幅值拟合，并增加有零梯度保护的方向监督。记录各loss的真实参数梯度范数。

qualification_v1：8工程步，step4刷新当前模型反例，最后做8训练图的一次真实FLUX尾段probe。
这8步只验收链路、显存、保存、反例刷新；未改善不能当整条路线失败。
trainfit_v1：独立fresh200步，50/100/150刷新反例，最终做相同训练图probe。
代码已具备此入口，但必须先有hash匹配的qualification工程通过结果，并手动启动；
不能自动接续，不能从qualification权重初始化，不因工程通过就声称科学可行。

同时报告原FLUX、FP32无引导、原始一次建议及受限一次建议；受限版本最多一次，
限制相对原proposal的RGB RMS和通道均值偏移，界限从训练GT轨迹首步的95分位×1.25
导出并保留1/255量化下限。这只是训练侧防失控约束，未经独立校准，不是no-harm保证。
无候选时原样返回原FLUX；全部拒绝=没有收益。独立校准、来源分离确认与多步扩展
留到训练侧学会之后，当前没有实现/启动它们，不得暗中用检查图选阈值。

默认用满单节点8×H100/H200，同时支持同型号4卡；global batch32对保持不变。
每完整optimizer step原子保存模型/Adam/每rank RNG/采样位置/哈希；相同命令恢复，
不同world_size严格恢复不支持。每张候选缓存/probe独立落盘，可跳过已核验记录；
异常中断留下的孤立tensor文件保留，重试用新唯一文件名，不覆盖。旧目录不动。
67项CPU回归和两进程真实二阶DDP精确恢复已通过；真实训练缓存预检通过。
Codex没有GPU，尚未启动本轮GPU；4/8和H100/H200的CUDA显存/速度仍待实测。

八卡资格粗估35–75分钟（4.67–10使用GPU-hours），硬cap90分钟/12使用GPU-hours。
四卡暂估70–150分钟（同GPU-hours），硬cap180分钟；不是实测倍速保证。
八卡200步训练拟合粗估7–10小时/56–80使用GPU-hours，硬cap10小时；
四卡硬cap20小时/80使用GPU-hours。须按新资格实测更新，不能自动扩预算。
FP32标签/导数/反例/逐步checkpoint预计占25–40GiB，不能上传这些大文件至GitHub。

默认八卡资格命令：
bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh
四卡须独立ID：
bash smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh qualification qualification_4gpu_v1 4
未来trainfit第四参数可明确指定通过的qualification ID，不把四卡绑定八卡目录。
终态须TAIL_RELATIVE_V3_RESULT_VERIFIED和TAIL_RELATIVE_V3_END status=0。
详见reports/foundir_flux_tail_relative_energy_v3_readiness_2026-10-29.{md,json}。

## TERMINAL RUN: FLUX-tail qualification complete; learned guidance harms quality — 2026-10-29

本节覆盖下方LIVE状态：qualification_8gpu_v2已于2026-10-29T00:50:19Z正常结束，
8×H100，25/25训练步、8/8工程检查图、RESULT_VERIFIED、launcher status0。
3425秒=57分05秒=.95139普通小时/7.61111使用GPU-hours；旧四卡成本另计，
两次合计11.36222使用GPU-hours。没有待续训任务，不自动formal或扩大预算。

独立核验了step0–25共26份检查点文件SHA、全部8图结果文件SHA、源码绑定及7个
报告项的均值。每个完成步均保存；最终checkpoint SHA256为
2eebdafb0b6a30829de48792b8eced1037cbfcfc3e0114ee986f1f8b70da4ed1。
训练均值124.46875秒/步，八卡参数一致，峰值allocated显存31.8921GiB/卡。

工程通过不等于恢复有效。八图原FLUX为38.6109929085/.07752364664；
energy一次为36.7593092918/.07801964320；预定16次终点为
19.7870512009/.11268278910，即PSNR下降18.82394dB、LPIPS变差45.3528%。
无引导FP32对照38.6112513542/.07753595291，排除精度路径本身造成这种量级崩坏。
8/8图第一次PSNR就下降，16次后8/8两项都恶化；128次更新全被接受且全选.01最大步幅。
预测评分却认为质量改善：当前energy判断与真实质量脱节，自检未能拦住有害更新。

训练末5步梯度匹配loss均值1.000584、MSE/LPIPS方向cosine约-.051905/-.001187。
这些是训练采样诊断，不是评价图梯度测量；显示25步后尚无学会正确修正方向的证据。
推理将非零梯度归一化，因此微弱但错误的方向仍能触发大步更新；同一不准energy
再给更新打分，产生反复错误接受。完整根因尚未隔离，不能说已排除所有实现问题。

此次仅24来源/25步工程资格，不能据此宣判所有learned-energy路线不可能，也不能
因工程通过直接正式长训。资格权重不可seed formal。所有历史结果、缓存、模型保留。
后续应先确认训练侧的梯度/排序能否真正学会，再决定训练规模和部署约束；本次检查
未授权新GPU任务、调整阈值或改写已绑定实验。
报告：reports/foundir_flux_tail_energy_resumable_v2_terminal_2026-10-29.{md,json}。

## LIVE: eight-H100 training and per-step save verified — 2026-10-28

qualification_8gpu_v2 已在用户节点真实启动，不再是“待GPU运行”。
2026-10-28T23:53:14Z启动；八张卡均实测为H100 80GB，rank0–7全部参与。
旧四卡任务已精确停止：最后日志step7，23:53:44Z以status1结束，属主动切换，
不是新八卡训练崩溃。32图缓存全部复用，旧日志及快照保留；旧step1–7从未
保存optimizer checkpoint，不能恢复，当前是同初始化重算而不是无损续训。

已独立读取并核验step0/1/2不可变文件的SHA256；step2模型参数有限且已更新，
Adam共236个参数状态均为step2，next_global_slot=64，8份CPU/CUDA RNG齐全，
八卡参数一致。每卡峰值allocated显存31.8915GiB。前两步分别125.5294/124.6914秒，
接近旧四卡246–250秒的两倍速度；global batch/科学方案不变。
核验时2/25步，剩余训练约48分钟（约6.4使用GPU-hours），评价时间另计。
旧四卡已消耗3376秒/.9378普通小时/3.7511使用GPU-hours；若计整台八卡占用，
为7.5022整机GPU-hours。不得从历史账中删除这笔成本。

这是八卡启动、有限训练和逐步落盘验收，不是25步完成或科学成功。
GPU中断后真正重启尚未实测；仅CPU二阶DDP的精确恢复测试通过。
不要为测试恢复而打断正常任务，不改正在运行的绑定源码/配置。
同run-ID启动命令可恢复最后完整步，未完成的一步可能重算；不自动formal/扩预算。
当前未有新PSNR/LPIPS结果，早期gradient cosine接近零，不宣称energy已学会。
详见reports/foundir_flux_tail_energy_resumable_v2_live_2026-10-28.{md,json}。

## ACTIVE: use all eight GPUs; durable per-step FLUX-tail energy qualification — 2026-10-28

用户明确要求切换到八卡并保存进度。本节覆盖下面四卡首轮资格的执行优先级；
科学方案仍是同一个25-step learned-energy qualification，不扩训练规模、不自动formal。
默认单节点8×H100/H200，显式CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7。
仍保留4卡兼容入口，但兼容4卡不等于默认只用4卡；整机八卡空闲时默认用满八卡。

旧qualification_4gpu_v1已完成全部32图缓存（24训练/8检查），四卡训练实测
每步约246–250秒；原30–60分钟全程估算与90分钟旧超时不足，不能继续照此估算。
旧程序只在step25保存权重，当前中间optimizer状态只在GPU进程内，无法热补保存。
不要谎称换八卡是旧中间状态的无损续训。全部旧缓存/日志保留；新入口复用缓存，
从同初始化重算旧未保存训练步。切换时记录旧最终日志步数及日志快照，不包装重算。
Codex无GPU/远程进程控制通道；截至准备完成，尚未停止旧GPU进程或启动新八卡任务。

新任务：checkpoints/foundir_flux_tail_energy_resumable_v2/qualification_8gpu_v2。
已完成元数据与全部32缓存SHA核验，只链接原缓存，不重新生成、不读保护像素。
旧绑定源码/配置/缓存均不修改。新元数据同时记录原producer和新consumer代码身份。
global batch仍32候选对；八卡每rank累积4对，四卡累积8对。损失、lr、512裁图、
25步、来源划分、无GT推理和+.30dB/LPIPS下降9.928107050377874%目标不变。

每个完整optimizer step（含step0）原子保存模型、Adam、各rank Python/torchCPU/CUDA
RNG、显式warmup状态、next_global_slot、累计训练日志及哈希。fsync后发布不可变
checkpoint/receipt，再更新latest指针；旧步全保留。读取最高完整receipt，不因latest
指针滞后丢进度；损坏校验失败时禁止悄悄回退。未完成的一步可能需要重算，不承诺
断电瞬间未落盘计算可恢复。默认相同run-ID重复命令恢复最后完整步，不能新建ID
冒充续训。当前已验证同world_size恢复；换world_size恢复明确拒绝，不能冒称bitwise等价。
每张评价图完成后原子保存结果，重启跳过校验通过的已完成图；GT仍只在选择结束后计分。

已通过52项CPU回归，以及两进程真实二阶DDP的Adam/RNG恢复：
恢复后下一步参数与不中断路径逐元素相同。八卡缓存导入真实预检通过。
这不是八卡CUDA资格通过；新八卡训练、显存、吞吐、指标均待用户节点运行。
不因25步指标无收益就宣称路线终判，也不因CPU通过就声称科学成功。

八卡重算25步+评价粗估55–75分钟，约7.3–10.0使用GPU-hours（未实测八卡加速）；
旧四卡已消耗成本另计。新每次尝试硬cap：八卡5400秒、四卡10800秒，均12使用GPU-hours，
另最多30秒清理；超时保留所有已提交步骤，不自动重启/扩预算。若独占八卡仅用四卡，
须另计整机资源成本。整个历史累计成本不得被新run-ID清零。

用户GPU节点一条命令：
bash smoke/start_foundir_flux_tail_energy_resumable_v2_tmux.sh
默认run-ID qualification_8gpu_v2、8卡；同命令将来用于断点恢复。
启动器先准备/核验缓存，再通过精确脚本名、train phase、run-ID、4卡参数和cwd识别
旧进程，核对GPU没有无关任务，只SIGTERM该旧torchrun并等待退出，不使用宽泛pkill。
旧内存权重无法保存的事实会明确打印并记录；如果旧step25 checkpoint意外已经存在，
拒绝丢弃/重算，必须先处理其迁移。旧日志快照保存到新run目录，旧目录不删除。
随后获取节点锁、检查全部八卡空闲，执行train→evaluate→verify。
日志logs/foundir_flux_tail_energy_resumable_v2_qualification_8gpu_v2.log追加保存各次尝试。
终态需要TAIL_ENERGY_V2_RESULT_VERIFIED和TAIL_ENERGY_V2_LAUNCH_END status=0。
报告reports/foundir_flux_tail_energy_resumable_v2_readiness_2026-10-28.{md,json}。

## ACTIVE: learned FLUX-tail quality energy — 25-step qualification — 2026-10-28

用户在24来源GT确认完成后明确授权“下一步，执行”。本节覆盖下面全部历史
ACTIVE/TERMINAL中的优先级：当前只执行新energy的25步工程资格，不重跑GT诊断，
不恢复Restormer residual、MoE或旧EBT Generalist，不自动正式训练。
上一阶段GT平均+.98618dB/LPIPS下降31.562%只是可行性，不是energy已学会。

冻结FLUX/LoRA、VAE和LPIPS。新1,010,274参数条件quality energy预测log MSE与
log LPIPS，输入LQ、原FLUX结果、候选、差分、真实每图Qwen退化等级及shifted t。
同时监督分数、候选图像梯度和同来源候选排序；二阶训练只穿过小energy。
它不输出RGB，不加Restormer；推理梯度经真实FLUX末两步/VAE传回index48 latent。
候选接受/停止只看预测energy；GT仅在完整推理结束后算评价指标。
预测安全不等于实际no-harm，必须如实报告退步，不能用GT挑回好图。

保持Noise step400/50步/512裁图/原seed/CFG1/reference time10/alpha=.30/双uint8。
原54,585条Noise training记录、1,755来源排除旧32来源后，冻结来源划分：
train1467、calibration128、engineering_evaluation128。资格仅用24新训练来源、
8新检查来源，校准池不读像素。检查来源只对energy训练分离，FLUX可能见过，
不是系统未见过测试集。最终1859开发集及official/sealed/internal/parent-test禁止。
训练缓存包含连续与实际量化候选、随机latent扰动及固定GT轨迹；检查图无训练轨迹。
全部源码/数据/权重绑定，旧V1/V2/V3/24图确认文件与结果不改。

25 optimizer steps，fresh AdamW lr1e-4，warmup5，global batch32候选对/64视图。
4卡每rank累积8对，8卡累积4对；完整512训练，不改科学口径。
同一energy引导五种方向混合、五种步幅，最多16轮；报告1/2/4/8/16，主终点16。
真正目标仍为相对匹配FLUX PSNR+.30dB、LPIPS下降9.928107050377874%。
25步及8图均为工程检查，不是正式科学结论；不据此挑配方或自动长训。
资格checkpoint明确不得seed正式训练；后续规模须由实测吞吐及独立方案决定。

已完成：46项CPU回归、两进程真实CPU DDP二阶反传/累积一致性、
Python/shell静态检查、4/8完整真实权重与数据元数据预检（不解码图片）。
GPU尚未启动；Codex节点无GPU，不把CPU通过当CUDA显存/性能/指标通过。
单节点同型号4/8×H100/H200，每卡至少75GiB；不混卡、不跨节点。
四卡H100粗估30–60分钟，2–4使用GPU-hours；独占八卡时另计4–8整机GPU-hours。
硬cap四卡5400秒/八卡2700秒，均6使用GPU-hours，另最多30秒清理。
八卡/H200没有本任务实测，不承诺精确加速；不覆盖已有run，不同时跑两套。

四卡启动：
bash smoke/start_foundir_flux_tail_energy_qualification_v1_tmux.sh qualification_4gpu_v1 4
入口显式传CUDA_VISIBLE_DEVICES；八卡须新ID及参数8。
阶段hardware→prepare→cache→train→evaluate→verify；无自动formal。
日志logs/foundir_flux_tail_energy_qualification_v1_<id>.log；
输出checkpoints/foundir_flux_tail_energy_qualification_v1/<id>/。
必须看到TAIL_ENERGY_RESULT_VERIFIED和TAIL_ENERGY_LAUNCH_END status=0；
tmux创建不等于训练开始。当前无新GPU结果或实测成本。
配置SHA256：22da9b3fb51f4705cc917b3d3c8bd1e1fea8bdd3193fff205e198c57ef462c71。
详见reports/foundir_flux_tail_energy_qualification_v1_readiness_2026-10-28.{md,json}。

## TERMINAL: fixed GT policy passes on 24 separate training sources — 2026-10-28

confirm24_4gpu_v1 已正常结束，不再启动/续跑。下面ACTIVE/readiness记录均为历史；
当前没有待完成GPU任务，不自动启动energy训练或恢复旧Restormer/MoE/EBT。
4×H100 80GB，2026-10-28T21:58:30Z–2026-10-28T22:17:00Z，launcher status=0，
CONFIRMATION_RESULT_VERIFIED及独立标准库重算均通过。
共1110秒/18分30秒/.3083333333普通小时/1.2333333333使用GPU-hours；
若独占八卡整机则2.4666666667整机GPU-hours（是否独占未确认）。
峰值allocated显存15.306301593780518GiB。仅本任务四卡H100实测通过。

24个来源、缓存索引8–31、7个报告项完整，24/24基线逐像素复现。
来源/GT/LQ哈希与旧八图分离；不把八图混入平均值，不在24图重新挑方案。
固定唯一主终点mixed_psnr_budget_step16：
原部署FLUX为40.669039726257324/.05249165142110238，
GT引导为41.655218760172524/.03592422348447144，
即PSNR+.9861790339151995dB、LPIPS下降31.56202460410801%。
超过+.30dB/LPIPS下降9.928107050377874%的原平均双目标。

22/24单图通过，24/24两项均未比历史基线变差，但不是每张有充足余量。
索引24接受0次，仅FP32无引导精度差异（+.0002479553dB/LPIPS降.0151213%），
不能归为引导收益；索引27接受1次，仅+.0255088806dB/降.0884384%，也未过。
索引17 PSNR仅比目标高.0007125854dB，索引18 LPIPS降10.7094849%，边际仍小。
4/8轮平均也过，但只是预定描述性报告，主终点仍16轮，不事后挑最优。

独立核验334次尝试、8350候选、327次接受、8023次拒绝，逐次选择/停止、
全部均值、源码配置哈希、sample/rank及trace一致。V1/V2/V3绑定文件未改。
policy含18704次FLUX forward（包括checkpoint重算）、668次目标backward，
不是廉价16次前向。成本和全部逐图结果均保留。
仍是训练来源GT确认：所有方向/接受/停止用GT，没有训练energy，没有VLM引导，
不是部署、泛化或论文创新成功。这24来源曾参加旧32图工程资格，不是未用过的测试集。
没有读取保护/最终1859条开发集像素，没有恢复图像定性审核。
后续energy学习需独立冻结方案和用户决策；禁止把本GT成绩当可部署收益。

结果SHA256：37f13289195eb6abd4fc4127f797008f7e6843ebd11957123b7245fcfa735137。
报告：reports/foundir_flux_tail_confirmation_v1_terminal_2026-10-28.{md,json}。

## HISTORICAL PLAN: fixed-policy GT confirmation on 24 other training sources — 2026-10-28

用户已授权执行。当前唯一新任务是固定 V3 的 mixed_psnr_budget、最多16次更新，
在原缓存索引8–31的24个训练来源上确认；此前V3终态/历史计划不再是待执行任务。
不重新挑方向/步幅/窗口/检查点，不恢复旧Restormer/MoE/EBT或最终像素下降，
不训练energy、不自动长训。V1/V2/V3绑定源码、配置、结果保持不动。

复用V3原始diagnose_record及trajectory核验，不改梯度、rollout、量化或接受规则。
保持Noise step400、50步、512裁图/seed、CFG1、reference time10、alpha=.30、
双uint8量化、index48真实两步rollout、BF16模型和FP32 latent。
只保留mixed五方向与psnr_budget一条轨迹，仍按原25候选/轮、最多16轮；
报告1/2/4/8/16及两个对照共7项，主终点唯一mixed_psnr_budget_step16。
目标相对这24图自己的原部署基线：PSNR+.30dB、LPIPS下降9.928107050377874%。
不沿用八图绝对数值、不把八图混入均值、不逐图挑最佳检查点。
未接受修正仍保留FP32无引导结果，禁止GT事后换回历史图美化指标。

24来源与原八图的source、GT/LQ哈希、record ID均分离；元数据预检不解码图片。
这些来源曾参加旧32图工程资格，不是未用过的正式测试集。
全部方向/候选/停止仍依赖GT；只是训练来源机制确认，不是energy学会、
VLM贡献、部署、泛化或论文成功。全部测试/封存像素及最终1859条开发集仍禁止。
成功不自动授权后续训练；全部候选及拒绝/forward/backward成本继续记录，
不能将16轮更新说成只多16次前向，也不能保证LPIPS提升等于视觉自然。

单节点同型号4/8×H100/H200，每卡至少75GiB，4卡每rank六张/8卡每rank三张。
34项CPU回归、Python/shell静态检查、真实权重与全部24缓存的4/8卡预检均通过；
新任务GPU未启动，Codex无GPU。仅旧V3四卡H100实测，不扩称本任务GPU验收。
暂估四卡H100 15–30分钟（.25–.50普通小时，1–2使用GPU-hours），
若独占八卡整机仅用四卡另计2–4整机GPU-hours；这是粗估，不是实测。
硬cap：4卡1小时/8卡.5小时，均4使用GPU-hours，另最多30秒清理。
独占八卡只用四卡时，cap对应8整机GPU-hours；八卡/H200未有本任务实测ETA。
四卡命令：bash smoke/start_foundir_flux_tail_confirmation_v1_tmux.sh confirm24_4gpu_v1 4
八卡可用独立ID confirm24_8gpu_v1 8；二选一，不重复同时跑。
日志logs/foundir_flux_tail_confirmation_v1_<id>.log；
结果checkpoints/foundir_flux_tail_confirmation_v1/<id>/result.json。
必须核验CONFIRMATION_RESULT_VERIFIED及CONFIRMATION_LAUNCH_END status=0，
tmux创建不等于GPU开始。全部结束前不自动扩大任务。

配置SHA256：c3ac68f96f2231a434d90b5fa733948d3c9dbe3bb08bfc8f86f839e4b97d3dcf。
V3父结果SHA256：06eed91ab7c2c0942b221a14d4c295d1e269bf57b17a8d79354ec9be074d24e4。
详见reports/foundir_flux_tail_confirmation_v1_readiness_2026-10-28.{md,json}。

## MANDATORY: all future experiments support single-node 4/8 H100 or H200 — 2026-10-28

用户明确要求：从本次开始，所有后续实验、训练、评价、诊断和启动脚本都必须支持
单节点4×H100、4×H200、8×H100、8×H200。此规则覆盖历史“必须8卡”的操作限制，
但不是授权恢复旧实验或更改当前研究方向；历史脚本不因此自动视为已兼容。
每次只用同型号GPU，不混H100/H200，不跨节点。不得根据节点名推断型号。
当前实现以每卡至少75GiB可见显存为验收条件；其他显存规格须单独验证，不冒称已支持。

新agent必须遵守：
- 启动器显式接受4/8卡，检查实际CUDA可见数量、型号一致性和空闲状态；
  不把world_size、gather列表、样本分片、日志或GPU-hours硬编码成8。
  tmux必须显式传入卡数和CUDA_VISIBLE_DEVICES，不能依赖旧server环境。
- 训练用梯度累积保持原global batch、loss归一化、数据曝光量和optimizer-step口径；
  卡数减少不自动改学习率、裁图、loss、验证集、科学门槛或训练规模。
  若不能保持等价，必须先解释差异并取得用户确认，不能静默降低实验标准。
- 评价/诊断保持同一完整记录集合、crop、seed、指标、量化与权重；4卡顺序处理
  更多记录，不能丢掉一半数据。汇总防漏样本和重复计数。
- 换卡数续训必须核验optimizer/scheduler、RNG及sampler进度的可恢复性；
  不把重新初始化或数据顺序变化冒充严格续训，不覆盖已有run/artifacts。
- 每个新入口都测试4/8两种分片/汇总及CPU静态路径；分别记录实际GPU资格状态。
  “代码支持”不等于“4卡和8卡都已跑通”；无GPU时必须明确标待GPU验收。
- 分别报普通wall hours及实际使用卡数×耗时的GPU-hours。4卡可能更慢，
  不承诺精确2倍耗时；若实际占用整台8卡节点而只用4卡，还须另报8卡资源占用成本。
- 未来沿用通用命名/参数入口；历史_8gpu入口只能明确作兼容wrapper，
  不要让fresh agent误以为还必须8卡。保留历史结果，修改后focused commit/push。

历史Noise FLUX V1诊断已实现4/8支持，固定8训练来源、同46个arms、无新训练：
4卡每rank处理2张（先后独立加载、释放模型，不增加单卡batch），8卡每rank1张。
通用入口smoke/run_foundir_flux_tail_oracle_v1_gpu.sh <fresh_run_id> <4|8>；
tmux入口smoke/start_foundir_flux_tail_oracle_v1_tmux.sh <fresh_run_id> <4|8>。
V1历史预算cap：4卡3普通小时/12 GPU-hours；8卡1.5普通小时/12 GPU-hours，
另最多30秒清理。这是预算，不是实测ETA。4×H100已实测完成；8卡/H200尚未验收。
详细变更见reports/foundir_flux_tail_oracle_4_8gpu_compatibility_2026-10-28.md。

## TERMINAL: V3 GT mean joint targets PASS on eight training sources — 2026-10-28

budget_4gpu_v1 已正常结束，launcher status=0，BUDGET_RESULT_VERIFIED。
4×H100 80GB，2026-10-28T21:11:43Z–21:26:53Z；总910秒/15分10秒，
.2527777778普通小时、1.0111111111使用GPU-hours。
若独占整台八卡则2.0222222222整机GPU-hours（是否独占未确认）。
峰值allocated显存15.307990GiB。八来源×22报告项、8/8基线逐像素复现、
sample/rank/trace、源码配置哈希及独立均值/选步重算均通过；
独立核验367次尝试更新、6115候选、352次接受。仅四卡H100实测通过。

原部署基线39.64431142807007/.07384782587178051。
固定 mixed_psnr_budget_step16 为40.34524202346802/.053707223385572433，
相对原基线PSNR+.7009305953979492dB，LPIPS下降27.27311501516311%。
超过原+.30dB/LPIPS下降9.928107050377874%的平均目标，确实有平均余量。
不能继续描述为“GT平均双目标尚未通过”，也不能扩称energy/泛化成功。

同16次报告：balanced_monotone 为+1.1016478539dB/LPIPS降9.7295917357%，
仍未通过；balanced_psnr_budget 为+.7904086113dB/降13.9110398133%，通过；
mixed_monotone 为+1.4849891663dB/降14.7461070083%，通过。
这是各自同一个方案的两项指标，禁止拼接不同方案的最好值。
mixed_psnr_budget 的4/8次报告为+1.1330642700dB/降11.1080559356%、
+.9448838234dB/降18.8657205717%，也通过；未逐图选最佳检查点。

主方案7/8张单独双目标通过，8/8两项优于原基线。
样本0仍只有3.362307% LPIPS降幅，接受3次后停；其余七张到16次上限。
样本1/3 PSNR只比+.30目标多.0383522/.0071556dB，平均余量不等于逐图余量。
全部指导、候选比较、接受/停止使用训练GT，没有训练energy、没有VLM引导、
没有部署选择器/新泛化结果或论文创新通过。八图是多轮机制诊断来源。

四组共享新排序和更新上限，但多方向有额外计算，不能宣称等算力优势。
主方案共2900候选、2785次拒绝、6496次FLUX前向计数、232次目标反传，
还不含共同基线复现/前缀/加载。不是只多16次前向的廉价部署TTS。
本轮没有恢复图像的定性审核，不可用LPIPS直接保证视觉纹理自然。

无自动后续任务；旧ACTIVE/未启动记录均为历史。下一步应先冻结单个方案
和预算，在缓存另外24个训练来源索引8–31做确认，再决定energy学习；
V3未读取这些像素，不自动扩展/重跑/加步/长训，不恢复旧分支。
24来源仍是训练来源机制确认，不是最终held-out泛化，也不保证正式训练必过。
报告及全22项均值：reports/foundir_flux_tail_budget_v3_terminal_2026-10-28.{md,json}。
原始result SHA256：06eed91ab7c2c0942b221a14d4c295d1e269bf57b17a8d79354ec9be074d24e4。

## HISTORICAL PLAN: GT-only PSNR-budget / direction diagnosis V3 — 2026-10-28

用户明确授权：先解决 GT 诊断仍未通过双目标的问题，再检查更多来源；
不直接启动正式 energy 训练。此节覆盖下面终态/历史计划，不恢复其它分支。
V3 保持同八个 training sources、512裁图/seed、Noise step400、50步、
alpha=.30、双uint8量化；仅 index48 latent 真实两步rollout。
不只挑困难三图，不访问任何测试/封存像素及最终1859条开发集。
V1/V2绑定源码与结果不修改，新输出根 foundir_flux_tail_budget_v3。

四组匹配对照：balanced/mixed 方向 × monotone/psnr_budget 接受规则。
balanced 为 MSE/LPIPS 梯度各自RMS归一化后等权混合；mixed 同时试
LPIPS权重0/.25/.5/.75/1。共同步幅.0005/.001/.002/.005/.01。
monotone 不允许PSNR低于当前值；budget 达到原基线+.30dB后才允许
用掉超额PSNR，不能跌破目标；尚未达到时保持PSNR进度。
两组均不接受LPIPS恶化，并要求不损害原基线；先缩小PSNR目标缺口，
可行候选中优先最低量化LPIPS，再较高PSNR/较小步幅，固定顺序破平局。
最多16次尝试更新，1/2/4/8/16报告；达到目标不提前停，
无合法候选才停。共20轨迹报告项+2对照，全部尝试与拒绝保留。

四组共享排序及迭代上限；mixed每次25候选而balanced为5，
计算量不同，必须记录所有前向、重算、反向、解码和拒绝试步成本，
不能宣称等算力部署TTS优势。未接受更新时保留FP32 latent对照，
可能与历史BF16有极小差异，不用GT事后换回较好图来美化均值。
接受/停止直接用GT，仍不是可部署方法、energy学习或泛化证据。
原目标+.30dB与LPIPS相对下降9.928107050377874%不变，不承诺一定通过。

单节点同型号4/8 H100/H200；4卡顺序每rank两张，8卡每rank一张。
上限4卡1小时、8卡.5小时，均4 GPU-hours，另最多30秒清理；
这是cap不是ETA。独占八卡但用四卡须另报整机成本。
入口 smoke/start_foundir_flux_tail_budget_v3_tmux.sh <fresh_id> <4|8>；
建议四卡ID budget_4gpu_v1。Codex无GPU，不得把代码/CPU/preflight通过
说成GPU任务已经启动。无自动重训/数据扩展；不覆盖已有run。

更多来源计划：先检查八图双指标余量；通过后冻结单个方案，再检查
现有缓存中另外24个训练来源（索引8–31）。它们不是最终泛化测试，
本次V3不读取其像素，不自动扩展，也不逐图挑最佳方案冒充统一方法。
八图通过不能保证正式训练成功；学习和泛化仍须后续独立证据。
详见 reports/foundir_flux_tail_budget_v3_readiness_2026-10-28.md。

V3最终28项CPU测试、Python/shell静态检查和4/8卡权重/缓存元数据预检通过，
未解码原图。配置SHA256：
58ebc99b3bc92b4db4eaf80b3489bd3047bd0f6014e8ca26dbcdabe25a54ef9d。
GPU运行仍未启动，等待用户在GPU节点执行上述tmux入口。

## TERMINAL: V2 tail diagnostic complete; joint target still short — 2026-10-28

tradeoff_4gpu_v1 已正常结束，当前没有待续的 V2 任务，不自动启动 energy 训练。
4×H100 80GB，2026-10-28T20:44:07Z–20:47:38Z，launcher status=0。
总 211 秒 / .0586111111 普通小时 / .2344444444 使用 GPU-hours；
若独占整台八卡节点，对应 .4688888889 整机 GPU-hours（是否独占未确认）。
八训练来源×50报告项完整；基线逐像素复现、STE 前向一致、所有绑定源码/
配置哈希、sample/rank 完整性及独立均值重算均通过。每卡峰值 15.309712 GiB。
仅四卡 H100 实测通过；不冒称八卡/H200已实际验收。

原部署基线 39.64431142807007/.07384782587178051。
连续代理的 GT 安全轨迹最终为 40.51412868499756/.06995537877082825：
PSNR +.8698172569274902 dB，LPIPS 降 5.270902772020114%。
双量化 STE 轨迹为 40.512683391571045/.07000666065141559，
PSNR +.8683719635009766 dB，LPIPS 降 5.201460131045987%。
总体目标仍是 +.30 dB / LPIPS 降 9.928107050377874%，对应
39.944311428070066/.0665161346648535；全部48个引导报告项无一同时通过。
不得把这些 GT 结果当作训练出的 energy、VLM贡献、部署或泛化成绩。

两种代理均五张达到单图目标，三张（索引0/2/7）在固定方向/幅度下无安全
候选而停住；它们占原LPIPS总和约75.28%。所有轨迹至多接受四次更新，
step4=step8，不是继续更新没有涨。连续代理初始 MSE/LPIPS 梯度强度比
中位数6.197、范围1.472–42.466；平衡确实改善LPIPS，但还不够。
换STE没有解决缺口，不能因此说量化无影响。样本2没有接受修正，
保留FP32 latent对照，相对历史BF16仍有-.000534dB/+.0000516 LPIPS小差异；
不得声称每张图相对历史基线绝对无损。

本次证明生成中有双指标改善空间，尚未证明完整目标可行，更未证明energy可学习。
旧ACTIVE/readiness段落仅为本次历史计划。后续有限诊断或训练需新的用户授权，
不扩大预算/数据、不更改绑定源码或放宽门槛、不恢复Restormer/MoE/EBT。
报告及全50项均值：reports/foundir_flux_tail_tradeoff_v2_terminal_2026-10-28.{md,json}。
原始result SHA256：78e442d5f3a4cda82c15b2903b02d66a2ab790ac167df8bc12e9a8bcf1858625。

## HISTORICAL PLAN: GT-only LPIPS/PSNR trade-off diagnostic V2 — 2026-10-28

用户在看完V1完成结果后授权下一步。只做training-source-only的有限诊断，
不启动energy训练、不变更已有FLUX权重、不恢复Restormer/MoE/EBT。
V1结果及绑定源码保持不动；新入口/输出根为foundir_flux_tail_tradeoff_v2。
入口：smoke/start_foundir_flux_tail_tradeoff_v2_tmux.sh <fresh_id> <4|8>。

使用完全相同8训练来源/512裁图/seed/Noise step400/50步/alpha=.30/双uint8量化。
仅index48的真实剩余两步rollout，不再同时扫多个窗口，避免把预览误差混入。
先分开测normalized MSE与LPIPS的梯度RMS、比值和夹角，再比较：
MSE-only、LPIPS-only、原联合loss梯度、两种梯度各自归一化后的平衡方向。
原联合loss通过单独的联合backward获得，不用两个BF16 backward相加冒充原实现。
幅度冻结.0005/.001/.002/.005/.01；所有40个single-update arms保留。

两种目标：原continuous blend；保留真实双round前向但使用straight-through近似梯度。
STE不是取整的真实导数，不能宣称其通过有限差分。每个实际输出仍按真实量化评分，
并核验STE前向逐像素等于deployed_proposal，记录连续指标与量化指标差异。
每种目标额外一个最多8次的balanced GT-only轨迹：每次用同5步长试完，
只接受量化PSNR/LPIPS都不损害当前结果且至少一项改善的候选；
无安全候选或达到原双目标则停止，保留1/2/4/8步报告及全部接受/拒绝记录。
这依赖GT，不是可部署TTS/自验证，也不证明energy可学习。严格no-harm有限搜索失败
不能证明不存在需暂时权衡指标的其他路径，8训练图也不证明泛化。

共50报告arms；原BF16及同精度无引导control仍保留。
相对匹配FLUX的+.30dB/LPIPS下降9.928107050377874%目标不变，不以旧Restormer抵扣。
单机4/8同型号H100/H200；4卡每rank顺序2张，8卡每rank1张。
预算cap：4卡1小时，8卡.5小时，均4 GPU-hours，另最多30秒清理；
这是上限，不是已实测ETA。Codex无GPU，本轮只完成实现/CPU及元数据预检，
不声称新GPU任务已启动。完成前不得自动扩大数据或启动energy长训。
报告：reports/foundir_flux_tail_tradeoff_v2_readiness_2026-10-28.md。

## MILESTONE: four-H100 tail diagnostic complete; joint target NOT met — 2026-10-28

mechanism_4gpu_v1已正常结束，4×NVIDIA H100 80GB，2026-10-28T20:13:53Z–
20:22:22Z，launcher status=0，TAIL_ORACLE_RESULT_VERIFIED。总509秒/
.1413888889普通小时/.5655555556使用GPU-hours；纯worker阶段.0938851541小时。
8训练来源×46个arms完整；8/8基线逐像素复现，16/16 FP32预览/两步rollout
有限差分通过，最大相对误差.0097435826。单卡峰值22.5287GiB；源码/配置身份一致。
这实际验证了4×H100路径，不能扩称8卡或H200已完成GPU验收。

原基线39.64431142807007/.07384782587178051。
最佳PSNR固定arm preview_skip_i46_w0_rms0.005：
40.46877956390381/.07429557666182518，PSNR+.8244681358dB，但LPIPS恶化.606315%。
真实两步rollout+联合目标、小幅度.005：
40.45664739608765/.0739425360225141，PSNR+.8123359680dB，LPIPS仍恶化.128250%。
最佳LPIPS固定arm preview_skip_i48_w1_rms0.02：
39.27623128890991/.0733890135306865，LPIPS只降.621294%，PSNR反降.3680801392dB。
42个引导arms无一同时达到+.30dB和LPIPS下降9.928107%；本8图对应LPIPS目标.0665161347。

结论：GT引导在有限窗口有明确PSNR修正空间，不是简单“FLUX末段改不动”；
本次单步/权重/幅度配置尚未证明双指标可行，尤其LPIPS不足。
不是energy学会、不是VLM贡献、不是泛化或论文成功；没有训练energy。
下一步需先针对LPIPS目标/步幅及实际量化终图对齐提出有边界诊断，
不能直接据此自动启动energy长训，也不能由8图有限搜索否定全部路线。
报告：reports/foundir_flux_tail_oracle_v1_terminal_2026-10-28.md。
result SHA256：4985f367cca8bc1fb730dee173d111eb4929fc6ebda4b9d0db6da223d523f7b1。

## ACTIVE: Noise FLUX training-only tail mechanism diagnostic — 2026-10-28

用户明确授权执行最小机制诊断；此节覆盖下面所有旧Rain/Generalist/MoE/residual优先级。
唯一当前工作是冻结Noise FLUX生成过程的GT-only机制诊断，不训练energy或修改LoRA，
不恢复最终pixel descent、Restormer residual、MoE、EBT Generalist，不自动长训。
报告：reports/foundir_flux_tail_oracle_v1_readiness_2026-10-28.md。
入口：smoke/start_foundir_flux_tail_oracle_v1_tmux.sh mechanism_v1。

冻结复用已绑定qualification_v1缓存的前8条（8个training sources），512裁图；
不是最终1859条开发集，也不是泛化证据。GT只用于非部署诊断，全部测试像素仍禁止。
保持Noise step400、50步真实shifted schedule、reference time10、CFG1、alpha=.30、
两次uint8量化。零引导必须逐像素复现历史入口和缓存，否则立即失败。
仅在梯度数值检查阶段把模型转换FP32并关闭TF32，随后恢复原参数/buffer精度及TF32。
引导使用FP32 latent累积/BF16 FLUX；每个窗口都有同精度无引导control。

比较零基索引40/46/48处的一次预览梯度修正（完整及skip-J近似），
48处另做真实两步rollout梯度；不是默认最后几步最佳。
预先固定latent RMS幅度.005/.02/.08，目标为归一化MSE或MSE+LPIPS。
实际修改后重新计算FLUX velocity并完成剩余轨迹，按真实量化终图评分。
FP32完整预览及两步rollout均须通过有限差分，不能把skip-J叫精确梯度。
逐arm报告失败和收益，不用GT选优隐藏失败，不把有限搜索叫全局上界。
GT诊断成功不保证energy可学习；8来源失败也不证伪所有窗口/训练方案。
VLM等级仅记录来源，本诊断不用VLM引导，不能算VLM必要性证据。

单节点同型号4或8张H100/H200，一次一个任务；硬件空闲检查、节点锁、fresh ID、
输出不覆盖、源码/数据哈希绑定和timeout均启用。4卡上限3普通小时，
8卡上限1.5普通小时，均12 GPU-hours，另最多30秒清理；不是实测ETA。
记录实际forward/block recompute/backward/解码计数、显存和运行时间。
Codex当前无GPU设备；CPU/static及预检不等于实际GPU验收，不能说任务已启动。
正式energy仍须新决策/冻结配方，独立目标+.30dB及LPIPS降低9.928107050377874%不变。
继续简明中文汇报；每个实质节点focused commit/push并核验远端。

## REVIEW: FLUX tail-guidance deep paper/code audit — 2026-10-28

用户本轮要求先细读关键论文和代码，再复述计划；不是授权启动新GPU实验。
完成报告：reports/foundir_flux_tail_energy_guidance_deep_audit_2026-10-28.md。
12篇指定版本的正文/公式/实质附录已分段细读；9个官方仓库相关调用链做了静态核查。
不能说已复现全部论文、检查了所有图像或排除了所有bug；ResFlow-Tuner未找到方法代码，
D-Flow/KDS本轮只核原文/算法，明确保留实现核验缺口。

继续以冻结Noise FLUX生成过程中的条件energy引导为唯一当前方向，
不恢复最终pixel descent、Restormer residual、MoE或从零EBT。
零引导需复现50步/alpha=.30/两次uint8量化；保护全部封存/测试像素。
当前入口inference_mode不能直接启用输入梯度；需独立可微入口。
真实shifted t下预览为z_t-t*v，导数为I-t*J_v；完整预览梯度不等于真实终点梯度。
FlowChef跳过Jacobian是近似，并在公开路径复用旧velocity；不要宣称等价。
KDS附录关闭t/T<.3引导，FreeDoM使用中段窗口，故不能默认最后2–4步最佳。
IRED仍有二阶导；更低energy不保证正确；CEP有限候选目标不保证exact。

建议下次实施顺序（本轮未实施）：零引导一致性/梯度检查 →
training-source-only GT目标的真实tail可改进空间诊断 →
训练真实FLUX预览分布上的energy（排序和方向）→ 实际量化终图配对评估。
GT oracle只诊断，不得进入部署，oracle可行不证明可学习；小范围失败也不证伪全部方向。
保留energy独立+.30dB/LPIPS相对降低9.928107050377874%的原目标。
不默认复用25步energy，也不把32张工程资格当科学规模；先实测再冻结正式规模。
固定预算引导先于未校准的自适应停止；同一energy自验证仍须证据。
比较原基线及相同总计算/精度的无energy控制，计入全部反向、VAE和rollout。
本轮仅文档，无新训练代码/GPU任务，GPU-hours=0；窗口、预算、正式配方尚未冻结。

## ACTIVE: investigate FLUX tail guidance; abandon output-pixel descent — 2026-10-27

用户最新明确决定：不再继续当前FLUX最终输出上的直接pixel energy descent，
改为探索引导FLUX最后少量采样步骤。此决定覆盖此前“必须先证明pixel energy有效/
纹理受损才允许转采样引导”的前置条件；但用户本轮要求先广泛查论文和既有方法，
不是授权立即新训或把现有25步energy直接接入FLUX。
旧qualification已完成，无需杀进程；不得自动formal或恢复旧PPT/Restormer/MoE。
旧25步结果保留为工程通过/量化收益0，不宣称科学上已证伪整条energy路线。

调研依据：reports/foundir_flux_tail_energy_guidance_literature_2026-10-27.md。
已核查IRED、Reduce Reuse Recycle、CEP、DPS、Universal Guidance、FreeDoM、
On the Guidance of Flow Matching、D-Flow、FlowChef、Trust Sampling、
KDS和ResFlow-Tuner等原文方法，并只读检查IRED/FlowChef/flow_guidance公开代码。
中间态适配有已有解法：t/noise-conditioned energy；predicted-clean preview评分；
短段真实rollout后评分。不要再把它说成未解决、必须从零发明的障碍。
但解决输入分布错位不等于已有energy已经可靠或能通过PSNR/LPIPS门槛。

建议下一设计以FLUX实际末段的predicted-clean preview + 条件quality energy为起点，
保留冻结FLUX及其生成步骤；energy训练数据必须覆盖实际预览/候选分布。
不是直接复用25步energy，不是把IRED完整conditional score盲目加到已有conditional
FLUX向量场，不是把DDPM公式或DDPM噪声时标照抄到rectified flow。
本仓库t=1为噪声、t=0为clean；z0_hat=z_t-t*v(z_t,t)是单步endpoint估计，
不是精确最终结果；真实shifted schedule而非step_index/50决定t。
完整输入梯度是准确对照；FlowChef-style跳过FLUX Jacobian是可测试加速近似，
不是“完全不需要梯度”。可微preview仍需VAE/energy反传。

保留Noise step400/50步/alpha=.30部署基线、最终uint8量化、匹配开发协议和
energy独立+.30dB / LPIPS相对降低9.928107050377874%的目标。
不允许换成较弱raw FLUX，也不把GT、未知真实退化算子或perfect noise level当推理条件。
后续需零引导一致性、原50步基线、匹配总计算预算的无energy对照；
预算包括前向/反向、解码、评分/搜索，不能仅比步数。末段具体窗口和引导强度尚未冻结。
文献不证明只引导最后几步一定足够，也不保证同升两个指标。

Novelty边界：FlowChef已有FLUX引导；ResFlow-Tuner已有VLM语义条件/FLUX恢复/
中间轨迹搜索；普通energy guidance、TTS或其简单组合不能称首次。
本轮仅文献、代码只读核对及决策文档；无新GPU任务、无实验脚本改动。
单节点8xH100/H200约束保持；未来实验普通小时/GPU-hours待具体设计资格实测。

## MILESTONE: FLUX-only energy qualification completed; zero measured gain — 2026-10-27

qualification_v1于22:42:21Z正常结束，25/25步，launcher status=0且
FLUX_ENERGY_RESULT_VERIFIED。完整耗时992秒（16分32秒，.275555556普通小时，
2.204444444 allocated H100 GPU-hours）；不是先前保守预估的.5–1.5小时。
FLUX候选阶段106.3025267579942秒；energy平均20.694357499321924秒/步，
每rank峰值14.600883483886719 GiB。八rank全部参数hash一致，checkpoint哈希已复核，
运行绑定源码与当前文件一致。没有受保护像素访问，没有额外Restormer。

仅32条训练来源上的量化评估：baseline、energy 1/2/3和selected逐条记录指标完全相同，
均值40.41285765171051/.05783069503377192，即PSNR增益0、LPIPS改善0。
初始选择全为起点；训练后32张全选step3，但不产生量化指标收益。
这证明工程可运行，不证明energy正确评价画质、自验证或科学成功。
末步训练有非零修正（rank0平均.00011517549137352034，[-1,1]空间）；
幅度远小于一个8bit灰度间隔，符合量化掩盖微小修正的解释，但未存逐像素
最终float差值，不能据此断言所有像素都未变化或量化是唯一瓶颈。
25步仍在100步warmup内，不能仅凭本次验收终判方法失败，也不能假定扩训必然成功。

不要自动formal、扩退化、末段FLUX引导或恢复旧Restormer/MoE。
qualification权重不得seed正式训练。后续需另行冻结足够训练预算和独立开发验证，
目标仍是energy相对FLUX独立+.30dB / LPIPS降低9.93%，未改变。
报告：reports/foundir_flux_energy_qualification_v1_terminal_2026-10-27.md。
结果SHA256：1b906a58c423c3fcda6264143c86008b857be6c9727ab5b3ed904ab4e428df75。
checkpoint SHA256：3640540281ea20a9dd50788cd356366f0a4398b7613c73977cccdffc1766caf2。

## ACTIVE: FLUX-only energy qualification ready; sampler guidance conditional — 2026-10-27

用户授权先执行冻结FLUX输出上的energy修图接入与25步工程验收。
只有后续独立开发验证证明energy判断可靠、且像素修正导致纹理损伤限制收益，
才考虑引导FLUX最后少量采样步骤；当前不实现、不启动该备选。
未来必须匹配总计算预算的无energy FLUX对照，计入energy前向/反向与VAE开销，
不能只匹配采样步数。MoE继续暂缓，一次一个GPU任务，不恢复旧Restormer-energy。

新入口：smoke/start_foundir_flux_energy_qualification_v1_tmux.sh。
配置：configs/foundir_flux_energy_qualification_v1.json。
先单GPU离线BF16解量化Qwen生成Noise/Haze文本缓存，退出后八卡生成真实FLUX候选；
再退出全部FLUX/AE进程，仅八卡训练energy。不得使用Rain提示词缓存。
32条来自32个training source的固定512裁图；训练256同步裁图，验收512。
Noise step400、50步、先量化raw再固定alpha=.30再量化，保留原部署基线。
真实wrong-Haze expert、人工不足/过度恢复和低通损坏、训练GT、完整1/2/3步轨迹
均有明确身份。GT只用于训练/评分，不进入trajectory或推理选择。
不加载额外residual Restormer，不训练直接refiner。

只做fresh 25步工程验收；32张训练图上的指标不是泛化结果，不自动formal、
不允许用qualification权重seed正式训练。内部NAF beta/gamma初始化为.1，
输出head仍为0，初始energy轨迹必须严格等于FLUX基线；这是解除初始化退化，
不是已证明的画质提升。后续科学目标仍是energy独立增加>=.30dB并降低LPIPS
>=9.928107050377874%，正式匹配协议仍1859条/34来源/512，不拿旧43.x指标抵扣。
源/配置、权重、提示词、数据、裁图与候选缓存都有身份检查；训练/heldout来源和GT哈希隔离。

CPU单测、双rank Gloo二阶反传和真实权重/元数据预检已通过；GPU验收尚未执行。
预检确认54585训练条目/1755来源全部满足512尺寸，未读取图片。
单机同型号8xH100/H200，预计.5–1.5普通小时/4–12 allocated GPU-hours，
包含串行文本编码、候选生成和energy验收；以实测替换估计，再决定全量缓存预算。
当前Codex无GPU执行通道，需用户在GPU节点运行入口，不得声称已经启动。
详见reports/foundir_flux_energy_qualification_v1_readiness_2026-10-27.md。

旧stage1_retry1最新日志仅到580/1706，随后21:55:21Z出现launcher END status=0，
elapsed_seconds=13182（3.661667小时，29.293333 allocated GPU-hours）。
没有训练完成标志或最终科学验证，不能把退出码0当作实验成功。
未核验远端PID、未替用户杀进程；新入口用同节点锁、精确旧进程检查和空闲显存检查防争卡。
旧未完成nonzero_init Restormer草稿继续禁止启动，所有历史文件保留。

## ACTIVE: FLUX proposal + energy; measure energy-only relative gains — 2026-10-27

用户最新决定覆盖下面所有Restormer/residual-basis PPT active/live目标。
主实验先做Noise：冻结当前可用FLUX输出，VLM条件energy通过同一个energy的梯度修图。
不加额外residual Restormer或可训练直接refiner。MoE仍暂缓。
用户并非否认Restormer可作为其他实验基线；关键是energy收益必须独立计量。

门槛必须相对匹配FLUX基线重新计算：PSNR增加>=0.30 dB；
LPIPS降低比例>=0.005/0.050362067760033824=0.09928107050377874（约9.93%）。
不得沿用43.424160178005695/.045362067760033824，也不得说energy只差.027 dB
或LPIPS已经完成。基线已有提升不能抵扣energy目标。

保留已验证Noise FLUX step400 + 固定alpha=.30/50步的部署处理。
alpha只是算术混合，不是residual Restormer；不得换成较弱的raw FLUX制造增益。
最终比较使用同一1859条/34来源/512 crop/量化/指标/种子协议；旧256条43.x不适用。
仅当匹配重现旧FLUX基线时，换算目标约40.45055349220034/.062041844133373776。
这些只是目标，不是新结果。自选优于固定三步、VLM必要性和跨退化证据另行验证。

当前完成的是协议与配对增益检查，不是GPU launch readiness：
configs/foundir_flux_proposal_energy_noise_v1_plan.json；
reports/foundir_flux_proposal_energy_noise_v1_decision_2026-10-27.md。
先核对Noise prompt、现有FLUX输出身份和训练来源缓存吞吐，再接完整PPT候选/三步轨迹，
冻结预算和资格后才能启动；不直接外推旧Restormer训练速度，不自动全量缓存或长训。
单节点同型号8xH100/H200，保持全部受保护像素禁令。

旧stage1_retry1最后只读见step565，仍是Restormer-basis energy，不能当FLUX结果。
该旧任务不再是必须完成的研究前置条件；不自动恢复、延长或切改其绑定代码。
停止状态需实际GPU节点确认，不在此声称已停止，也不擅自杀进程或与它争卡。
未测试的nonzero_init_qualification Restormer脚本是已被新方向替代的草稿，禁止启动。
保留旧权重/日志/草稿，不销毁历史。所有汇报用简明中文，说明数据和增益归属。

## MILESTONE: PPT step400 checkpoint verified; training near step420 — 2026-10-27

截至20:56Z，stage1_retry1日志到420/1706（24.6189917936694%），持续更新，
无OOM/NaN/NCCL/runtime错误。100/200/300/400步均产生完整pt与JSON receipt。
step400 SHA82a36b802920972ec234bd217a8a4663f55366d64521620de3fe37692c1604a6
与receipt一致，23项源码绑定未变，next_sample_step=401。
这证明实际GPU运行能保存checkpoint；尚未在GPU上演练中断恢复。

320–420步的21个日志采样平均22.325224876484192秒/步，峰值18.720479488372803 GiB。
从420到1706纯训练外推7.9750664419885195小时；含最终验证暂估8–8.5小时/
64–68 allocated GPU-hours，依实际速度更新。不改计划或自动扩训。

科学信号仍弱：同一日志窗口平均gradient_cosine=-.0003576330413222552，
平均update_abs_mean=.00004885955797546904（[-1,1]图像空间）。
这是训练批次诊断，不是开发集PSNR/LPIPS，不把工程健康说成画质已改善。
尚无训练后开发验证，不能宣布PPT成功或失败；继续固定1706步验证。
MoE仍暂缓。本次仅查日志/JSON/哈希，未读取图片或重新运行训练/评价。

## LIVE: PPT stage1_retry1 is training on 8 H100 — 2026-10-27

stage1_retry1于18:15:39Z启动，八卡硬件/空闲显存、真实数据与源码绑定、
NCCL通信检查均通过。节点hostname为eight-h200-copy-copy--b212aa683678-4227pzjtb4，
实际硬件为8×H100 80GB；不按hostname中的h200判断型号。
初始256条开发验证已完成，记录无重复，Fidelity和1/2/3步及自选输出逐图完全一致：
43.39715355634689/.028955043449059303。它是零初始化基线，不是训练后的增益。

已确认真实训练第1/5/10步连续推进，step10于约18:23:36Z写入日志，
损失/参数梯度有限，未见OOM/NaN/NCCL错误。
step10约22.233646298001986秒，峰值18.720479488372803 GiB；
能量修图更新非零，但不代表已达到画质或自验证目标。
仅凭启动和几步训练，不能声称checkpoint保存/恢复已在本轮GPU验证。
第一个完整断点计划step100，第一个训练后开发验证仍为step1706。
暂保留10.5–13普通小时/84–104 allocated GPU-hours规划，后续以实测修正。

日志：logs/foundir_novel_ppt_full_data_stage1_v1_stage1_retry1.log。
输出：checkpoints/foundir_novel_ppt_full_data_stage1_v1/stage1_retry1/。
不要重复启动、改变绑定代码/配方或自动扩训；只监测固定计划。
只做PPT、MoE暂缓的用户决定继续有效。

## ACTIVE: PPT only; MoE deferred until PPT outcome — 2026-10-27

用户报告节点2被不明人员停止，并明确改为一次只推进一个实验。
唯一active research job现在是Novel PPT Approach全数据stage1。
这覆盖下面“两节点并行”、MoE改进优先级及其他旧active表述。
MoE的已完成11000步结果完整保留；不重启旧MoE，不启动新的梯度诊断、
LPIPS .20/.40对照或VLM训练。上一轮提出的MoE两组各2000步仅是建议，未执行。
只有PPT在来源隔离开发集验证有效，或用户明确决定放弃PPT后，才重新讨论MoE；
不得将tmux启动、工程验收或32张训练拟合成功当作PPT已work而自动启动MoE。

PPT stage1_v1日志最后停在加载冻结模型，只有launch.json，没有训练步记录、
validation文件或完整checkpoint/receipt，故没有可恢复的优化状态。
没有异常栈/END标记；停止由用户报告，具体原因和操作者未知，不能归因代码或外部管理员。
保留旧日志与目录不动，不宣称已远程核实旧节点PID。
在用户确认旧任务停止的节点2，用新ID重新从零启动相同冻结配方：
bash smoke/start_foundir_novel_ppt_full_data_stage1_v1_tmux.sh stage1_retry1

单机同型号8×H100/H200，完整54585条/1755来源训练池、固定256开发条目。
仍只到1706步，不加载trainfit/qualification权重，不自动扩到8530。
预计10.5–13普通小时/84–104 allocated GPU-hours，每100步保存完整恢复状态。
这是重新启动命令，不是假称从某训练步续训；Codex没有替用户启动GPU。
源码/config不改，节点锁及空闲显存检查继续拒绝同节点重复占卡。

## TERMINAL: migrated MoE finished; Blur LPIPS still short — 2026-10-27

节点1resume_v1已于17:28:57Z正常结束，status=0，10500→11000共500步。
单机8×H100，launcher1845秒/.5125普通小时/4.1 GPU-hours；
trainer.45572002159471997小时/3.6457601727577598 GPU-hours。
不要重启、延长或把完成的MoE当成仍在跑。节点2PPT全数据阶段授权与配方不变。

固定11000步：Rain100条34.41416323292532/.06439921100391074，仍通过34.2/.08。
Blur2595条26.61952123148184/.18833398326176268，PSNR过25.9，LPIPS未过.18，
还差约.008334。相对原配方26.74785967610281/.2319519135801782，Blur PSNR
下降.1283384446209673、LPIPS改善.04361793031841552。
预先规定的修复信号通过，但联合绝对门槛失败；不是VLM优势或novelty证明。
旧H200→新H100的迁移和中断恢复必须披露，不称严格同节点消融。

41项源码、终局权重/Adam哈希、逐图汇总和完整记录数已核验，未重跑或读取图片。
报告：reports/foundir_moe_blur_lpips_resume_v1_terminal_2026-10-27.md。
result SHA：d11d47c16403e1e70389e33a91ee73de8b3d6d70708660de92c775e986bcdd39。

## ACTIVE: node2 PPT full-data stage1 authorized — 2026-10-27

用户已在节点1启动MoE迁移，并要求节点2的PPT命令。这覆盖下面“第二节点待授权”。
两实验分开节点，各独占单机同型号8×H100/H200；不跨节点DDP，不停止或改写MoE。
本次节点1共享日志已通过通信并完成恢复起点Rain验证，尚不据此宣称续训已完成。

PPT fixed32/300步已结束，不重跑，不加载其权重。新energy从零训练，
使用完整54585条/1755来源Noise池，固定256条source-disjoint开发验证。
模型/loss/三步轨迹/候选/global32/100步warmup/8530步LR跨度全不变。
只运行至原定首个检查点1706步并停止，不自动继续8530；不增加额外方法。
入口：bash smoke/start_foundir_novel_ppt_full_data_stage1_v1_tmux.sh stage1_v1
预算10.5–13普通小时/84–104 allocated GPU-hours，按实际全数据速度更新。

新增独立入口，不改旧绑定源码。每100步保存权重、Adam、8 rank RNG、采样位置、
config/source哈希及原子receipt。续训须确认旧进程已停，在新ID从明确完整receipt恢复，
不得拿日志步数当断点，不重置optimizer/warmup，不从trainfit/qualification恢复。
1706 checkpoint后验证中断也可从其receipt重做固定验证；H100/H200迁移不承诺bit一致。
绝对PSNR/LPIPS门槛和相对Fidelity贡献分开；修图收益与自选收益分开。
不读取official/sealed/internal/parent-test像素。保留历史失败和旧数据协议。

配置SHA：dc7a41787e75731f2034644d58c5c9fdef2744a34f7ccf1e0c1fba723a2fab2b。
19项CPU测试、两个shell语法和真实权重/数据元信息/23项来源绑定预检查通过，零图片读取。
新增恢复功能尚未在GPU实测；八卡模型工程资格沿用原成功配方，不把CPU通过当GPU通过。
报告：reports/foundir_novel_ppt_full_data_stage1_v1_readiness_2026-10-27.md。
本节只授权、记录启动准备，不声称GPU已启动；看共享日志确认。

## ACTIVE: migrate unfinished MoE; PPT trainfit complete — 2026-10-27

用户要求换新节点，并明确单节点8×H100/8×H200都要兼容；不跨节点DDP。
这覆盖此前“整组必须同一实体节点”的限制，但不能抹去迁移对可比性的影响。
只恢复未完成blur_lpips4x：从保存的10500到11000，500步；日志10750不是checkpoint。
恢复权重、Adam、每rank RNG及336000采样位置，loss/LR/data不变。
新增独立resume_v1入口，不改旧绑定代码或旧输出；每100步保存一次以防再中断。
control已经完成，不重跑；新节点记录型号/UUID，最终比较明确跨节点及可能H200→H100，
不得称严格同节点消融或逐bit一致。旧严格比较器保持不变。

确认旧任务停用后，在一台新空闲八卡节点：
OLD_MOE_STOPPED=1 bash smoke/start_foundir_moe_blur_lpips_resume_v1_tmux.sh resume_v1
预计.5–.9小时/4–7.2 allocated GPU-hours。用户认为旧任务已停，但没有远程PID证明；
日志停更不是证明，flag代表用户确认，不能声称Codex杀掉或核实了旧进程。
迁移配置SHA：39afd8616ada4b8a22cac7e41d7453e999ce81f3e5cab161693cf26e8f462f4d。
13项CPU测试、真实权重/Adam/绑定预检查及语法通过；GPU尚未启动。
报告：reports/foundir_moe_blur_lpips_resume_v1_readiness_2026-10-27.md。

PPT fixed32/300步于13:15:42Z正常结束，8×H100，总1.7594444444444444小时/
14.075555555555555 GPU-hours。训练裁块上Fidelity43.393339693546295/.03337287811154965，
energy自选43.93097627162933/.032854294855496846，即+.5376365780830383 dB/
-.000518583256052807 LPIPS。只证明32个反复训练裁块可拟合，不证明泛化。
近质量排序正确率从.5976470588235294到.6023529411764705，未证明自验证有收益。
不要重跑/迁移已完成trainfit，不能拿其权重seed正式，不能自动开1706/8530步。
第二新节点留待PPT全数据阶段获授权；现有PPT入口本已兼容H100/H200。
终局：reports/foundir_novel_ppt_trainfit_v1_terminal_2026-10-27.md。

## MILESTONE: MoE control complete; treatment log stale — 2026-10-27

截至2026-10-27T14:46Z，共享日志证明两组在同一台8×H200上顺序启动。
control已完成11000步、engineering_pass=true；trainer耗.6895869781746943小时/
5.516695825397554 GPU-hours。最终Rain为34.496691444868425/.06531600838992745，
Blur为26.74785967610281/.2319519135801782，Blur LPIPS仍未过.18。

匹配10500步：control Blur26.673934584810702/.22695887966412578，
blur_lpips4x Blur26.58181729682454/.186522385071396。
修改版LPIPS改善.04043649459272977、PSNR下降.09211728798616292，
但PSNR仍过25.9；修改版Rain34.609037982905335/.06519713898887858也通过。
Blur LPIPS距离.18还差.00652238507139602；这是中期收益，不是最终达标。

第二组最后日志10750步、mtime约13:14Z，检查时已超过90分钟未更新。
没有异常栈、OOM/NaN、结束标记、11000检查点或comparison.json。
最后完整可恢复状态为10500。不能说正常运行，也不能断定外部停机或代码故障；
需要原训练节点进程/GPU状态。不得自动杀进程、重启、续训或延长。
本次仅核对日志/JSON/来源哈希，不读取新图片，不改变训练代码。
详见reports/foundir_moe_blur_lpips_pair_v1_status_2026-10-27.md。

## ACTIVE: same-node matched MoE Blur-LPIPS continuation — 2026-10-27

用户批准MoE修复对照，并明确两组必须在同一节点。不要按原配方重跑10k，
不要拆成两个4卡任务或放到两台节点。使用同一台同型号8×H100/H200，
control完成后顺序运行blur_lpips4x，每组独占8卡。
两组均从原image-only formal_v1的step10000恢复权重、Adam、每rank RNG和
样本位置320000，新增1000步到11000；第二组不能继承第一组结果。
唯一科学改动：Blur LPIPS系数.05→.20；Rain仍.05；其他损失、学习率、
数据、采样、架构和image-only router不变。不重置优化器或warmup。

入口：bash smoke/start_foundir_moe_blur_lpips_pair_v1_tmux.sh pair_v1
配置：configs/foundir_moe_blur_lpips_continuation_v1.json
配置SHA：f9840280eabc089567168ab55d7ce8f6431772f6fcb94ddbd208a3ff16449d20。
同节点顺序总预算2–3普通小时/16–24 allocated GPU-hours，依实际速度更新。
固定10500/11000完整评估Rain100和Blur2595，保持原裁块/指标。
只用11000判断，不挑中期最好，不按开发集搜索新权重，不自动扩训。
修改版Blur LPIPS相对control的来源bootstrap上界<0、Blur PSNR>=25.9、
Rain通过34.2/.08才是继续研究信号；Blur LPIPS<=.18仍是独立绝对目标。
不要把改善、工程通过或GPU利用率说成完整达标/VLM贡献/novelty。

启动器全程持有PPT/MoE和Rain/Blur节点锁，核对hostname及实体GPU UUID，
禁止与已运行PPT争卡；不得自动停止、恢复或修改PPT任务。
PPT fixed32/300-step检查的授权与历史不变，旧绑定源文件保持不动。
本轮7项CPU测试及语法通过，真实父权重/Adam/来源/数据元信息预检查通过，
预检查零图片读取；GPU尚未启动，CUDA工程行为仍须实际运行确认。
Codex无GPU执行通道，需要用户在空闲节点输入上述一个命令。
报告：reports/foundir_moe_blur_lpips_continuation_v1_readiness_2026-10-27.md。
继续严格保护数据、保留脏目录、简明中文汇报并focused commit/push。

## ACTIVE: PPT train-only fixed32/300-step learnability check — 2026-10-27

用户明确授权执行“能否学会修图”检查。当前PPT下一步不是8530步正式，
也不是继续25步资格权重：从零energy初始化，固定32训练来源的32个256裁块，
每步各曝光一次，训练300步，第0/100/200/300步检查相同训练裁块。
保留原energy/完整三步/loss/AdamW/100步warmup/8530步LR跨度，
只固定训练数据并缓存冻结输出。不得读取开发或受保护像素；原split元信息排重继续。
此实验只诊断训练拟合能力，不作泛化/论文成绩，权重永不初始化正式。
不自动开启1706或8530步。若学不动先分析目标/优化/容量；若能学再决定全数据阶段。

入口：bash smoke/start_foundir_novel_ppt_trainfit_v1_tmux.sh trainfit_v1
单节点同型号8×H100/H200，预算2.0–2.5普通小时/16–20 allocated GPU-hours，
缓存可能减少耗时，须看实测。旧速度下step100训练部分约36.4分钟，另加启动/检查。
逐项报告Fidelity、1/2/3步、自主选择的MSE/PSNR/LPIPS、梯度方向与改变量、
固定相近候选的真实质量排序准确率（同时报参与对数和step0基线）。
能量下降或都选最后一步不构成画质/自验证证据。10%近质量分组仅用于诊断读数，
不作用于训练/推理，不进行阈值调优；零有效对时结果为null。
5项CPU回归测试、语法与真实来源/权重预检查通过；未执行新GPU任务。
Codex无GPU/远端执行通道，需用户在空闲节点输入命令，不能声称已启动。
配置：configs/foundir_novel_ppt_trainfit_v1.json
SHA-256：dd3a7748f6c821bc870e685c1cd8613ffc113877b64f6a8773f5279a29326cb5。
详情：reports/foundir_novel_ppt_trainfit_v1_readiness_2026-10-27.md。

并行MoE diagnostic_v2已于11:14:22Z正常完成，status=0、64条完整记录，
8×H200耗331秒/.09194444444444444小时/.7355555555555555 GPU-hours。
全部51项来源绑定核验一致；零优化步。不要重跑它或把诊断成功说成MoE性能改善。
解释梯度结果属于下一步分析，尚未确定Blur唯一瓶颈。
终局：reports/foundir_moe_gradient_diagnostic_v2_terminal_2026-10-27.md。
PPT25步资格亦已结束，历史指标/成本不变。新文件隔离新增，旧绑定代码保持不动。
简明中文、保护数据、唯一HDD3运行根和专用Python、focused GitHub tracking规则继续适用。

## ACTIVE: isolated MoE diagnostic V2 retry; PPT qualification complete — 2026-10-27

用户要求监测两任务至终局，并在MoE错误后明确要求修复执行、重新给GPU命令。
MoE diagnostic_v1已于10:54:47Z status=1退出，耗234秒/.065小时/.52 H200 GPU-hours。
首个梯度抽样越界，无完整样本/result.json；零优化步，不能解释为MoE科学失败。
CPU已用真实3072×9216矩阵尺寸复现：FP32 linspace把合法末端28311551
舍入为越界28311552。改为int64整数索引是工程修复，不改变训练方案。

不要重启V1。新增隔离V2诊断和专用启动器，4项回归测试与真实权重预检查通过。
为保护同时运行的PPT源文件绑定，旧V1、共用runtime/config/launchers都保持不变。
在空闲MoE节点运行：
bash smoke/start_foundir_moe_gradient_diagnostic_v2_tmux.sh diagnostic_v2
每任务仍单节点8×H100/H200，预算.25–.75小时/2–6 GPU-hours。
新ID日志与输出不覆盖V1。Codex无GPU执行通道；V2尚未被共享日志确认启动。
详见reports/foundir_moe_gradient_diagnostic_v2_repair_2026-10-27.md。

PPT qualification_v1在8×H100完成25/25步，11:03:13Z status=0，
705秒/.19583333333333333小时/1.5666666666666667 allocated H100 GPU-hours。
32张Fidelity为42.97121322154999/.03313419208279811；最终energy自主选择为
42.97113627195358/.03313359947060235。PSNR略降、LPIPS极微下降，没有双指标收益。
工程通过不等于科学通过；25步/32张不能判定正式8530步/256张方向成败。
配置/19项来源/checkpoint哈希和逐图重汇总均独立核验通过，旧绑定代码未改变。
中位21.83572708698921秒/步，8530步单训练外推51.738542236671655小时/
413.90833789337324 GPU-hours，另需验证等开销。不得自动正式或从资格权重续训。
先明确实测成本；如优化执行效率则重做匹配资格，不能改配方后沿用旧receipt。
终局报告：reports/foundir_novel_ppt_energy_v1_qualification_terminal_2026-10-27.md。
所有历史与保护数据约束继续适用。

## ACTIVE: two retry GPU entry points ready, not launched — 2026-10-27

用户的“execute both”现已落实为两个独立入口。本节覆盖下面“没有新启动器”
以及旧 Rain-only、Generalist-first、energy 永久冻结等过时优先级；保留全部历史失败。
不能把执行准备完成说成训练已启动。当前 Codex 检查为 0 个 CUDA 设备，无远端执行通道。

A. MoE/VLM：运行不更新权重的训练来源诊断，而非再盲训 router。
冻结两个 step10000 模型，各检查16个 Rain 和16个 Blur训练来源。
比较7项损失在代表性共享LoRA/专家输出矩阵上的梯度强弱和方向。
L2范数为所选张量的精确值；余弦为固定坐标抽样，不是完整网络的精确梯度。
零 optimizer steps，不读开发/受保护像素，不据此直接宣布泛化或调好损失。
入口：bash smoke/start_foundir_dual_retry_v1_tmux.sh moe diagnostic_v1
预算 .25–.75 普通小时、2–6 allocated GPU-hours。

B. Novel PPT Approach：冻结强 Noise proposal 和 step8530 residual basis，
训练634032参数的候选相关能量模型；同一能量既排序又通过图像梯度修图。
使用规范化残差坐标、空间能量求和求梯度、detach后的质量标签、完整三步训练，
以及真实错误专家/重复恢复/不足或过度恢复候选，不是七档alpha选择器。
先25步GPU工程验收：32张开发子集，仅查数值、显存、checkpoint和完整推理；
不能用这25步判定方向成败。训练池54585条/1755来源，正式预设8530步、
每1706步验证固定256张。正式预算须由验收速度推算，不能自动启动，
不能从资格权重续训；正式入口要求匹配成功资格receipt，从零energy初始化。
入口：bash smoke/start_foundir_dual_retry_v1_tmux.sh ppt qualification_v1
资格预算 .25–1.0 普通小时、2–8 allocated GPU-hours。

两任务各占一台同型号8×H100/H200；两台空闲节点可并行，一台则顺序。
启动器有节点锁、空闲显存检查、180秒有界NCCL检查和新ID防覆盖。
两入口均已通过真实权重/数据元信息预检查（零图片读取），7项CPU测试及
完整三步能量二阶反传的双CPU-rank同步测试通过。CUDA显存/二阶反传仍待实测。
本轮实际GPU消耗0小时/0 GPU-hours，未产生新的科学结果。

不能把原Fidelity收益算作energy收益。分别报告原proposal、Fidelity、
1/2/3步修图和能量自主选择结果。绝对目标仍为43.424160178005695 PSNR、
LPIPS<=.045362067760033824；还须单独证明相对匹配Fidelity的PSNR提高且LPIPS不差。
最低能量和soft no-harm只是训练/选择机制，不是画质安全保证或novelty证明。
详见 reports/foundir_dual_retry_v1_readiness_2026-10-27.md。
冻结配置 configs/foundir_dual_retry_execution_v1.json：
69db2e2f8a0e2dd3675bb7999968d4109ac4abb708151f909045dfb8578426ef。
不得修改正在运行的绑定代码/配置。保留脏工作树、保护像素禁令、简明中文汇报，
并在启动准备、关键验证和终局结果时focused commit/push。

## ACTIVE: VLM/MoE diagnosis AND Novel PPT Approach retry — 2026-10-27

用户已明确授权同时推进两条路线，先分析实际瓶颈、阅读原 PPT 和旧试验。
本节覆盖下面所有禁止重试、Generalist-first、energy 永久冻结及终局后等待决定的旧优先级。
历史失败和终局数字仍有效；不恢复 EBT Generalist，不覆盖旧实验，不复跑未改的失败配方。

A. 保留导师批准的 VLM→router 主线。两个 MoE formal_v1 已正常结束：
Rain 达标，Blur LPIPS 为普通 .23083898932184047、VLM .2282890860716721，均未到 .18。
固定 Blur 专家也不能修复差距；当前主要证据指向恢复能力/训练目标，而非单纯选错专家。
VLM 是每图五个强度等级，不是 raw tokens；只加在全局 router logits。
下一步先在训练来源上检查 Blur 拟合及各损失的梯度作用，保持 Rain 监测。
不把数据集类别当 VLM 严重度真值，不按开发集选新权重/阈值，不直接合并不兼容旧 Blur LoRA。

B. 重新尝试原 PPT 的完整 candidate-dependent energy：同一 energy 既排序又通过
候选图梯度修图。旧 V2 已实现真实候选和 1/2/3 步梯度训练，效果失败；不能说从未试过。
强 residual basis 的 +.27299337834119797 dB 是 refiner 收益，不是 energy 收益。
最后 pairwise gate 在独立82张上相对 Fidelity 为 -.0007919683689010526 dB；
其 joint_gate_pass=true 只是绝对阈值，不代表选择器贡献。中期小幅正值不可后验选用。

本轮已读完整8页PPT和4张方法图、重汇总既有指标、复核旧 step5118 能量数值。
确认 mean-energy 更新有显式1/(H*W)缩放，且旧局部排序质量标签参与反传。
新隔离组件 restoration_energy_numerics.py 修正这两个数值约定，7项CPU测试通过。
这不证明画质提升或失败唯一原因；旧日志最后20个256-crop记录方向余弦约.1409，
不能仅放大步长。下一次完整 energy 重试须对比强 Fidelity 本身，检查方向/幅度，
真实好坏候选和1/2/3步效果，不再把七选一或普通refiner冒充完整PPT方法。

PPT 已按用户要求命名为 Novel PPT Approach，GitHub 路径：
presentations/novel_ppt_approach/Novel_PPT_Approach_VLM_Restoration_Energy.pptx
从 original_attempt_retry 改名，文件字节与 SHA-256 不变：
cdb8d5e0fe682d6cf30f0ba217d64c426e682fbbc7c70bd51e950ed38edacefb
名称不是经过验证的 novelty 声明；README 保留“曾试过失败、现在重试”的历史。

审计报告：reports/foundir_vlm_and_novel_ppt_retry_audit_2026-10-27.md
审计JSON：reports/foundir_vlm_and_novel_ppt_retry_audit_2026-10-27.json
数值复核：reports/novel_ppt_energy_numerics_audit_2026-10-27.json
本轮新增GPU用量0小时/0 GPU-hours。没有新GPU训练、没有已验收的正式重训命令。
用户授权两条后续实施；新GPU资格/正式训练前仍须冻结设计、来源隔离、预算和验收条件，
使用新ID，不把25步工程测试当科学成败，不承诺必过指标。正式规模依可学习性与实测成本决定。
每任务只用单节点8×H100/H200；若有第二空闲节点可独立并行，不用跨节点DDP。
当前Codex无GPU或远端执行通道；不能把准备完成说成已启动。
所有受保护像素禁令、唯一HDD3工作根/专用Python、简明中文汇报及focused push规则继续适用。


## TERMINAL: BOTH joint MoE formal arms complete; joint quality FAIL — 2026-10-27

This supersedes every earlier running/monitoring/pending statement below.
Both formal_v1 arms finished10000/10000 normally with launcher status0,
engineering_pass=true and no logged nonfinite loss/gradient, OOM or fatal
runtime/NCCL failure. image-only ended03:14:25Z on8xH100; VLM03:41:07Z on8xH200.
The monitoring request is fulfilled. Do not resume, extend, relaunch, retune,
or start another experiment without a new user decision.

Frozen final Rain100: image34.73973430724847/.06654019097797573;
VLM34.742482612955634/.06634078806499019. BOTH pass Rain34.2/.08.
Frozen final FULL Blur2595: image26.655435556096634/.23083898932184047;
VLM26.61442463761589/.2282890860716721. BOTH pass Blur PSNR25.9, but
BOTH FAIL LPIPS<=.18. Do not describe engineering completion as joint success.

The existing final comparator passed matched config/all29 sources/data/input
metrics and recomputed paired rows. Rain VLM benefit is inconclusive (both
source-bootstrap intervals include zero). Blur VLM-minus-image is
-.04101091848074167 dB / -.0025499032501683986 LPIPS: perceptual/fidelity
tradeoff, NOT dominance. This is one seed on different H100/H200 hardware,
not training-seed uncertainty or a hardware-matched causal proof.
At identical VLM weights, switching VLM off changes Rain almost negligibly;
on matched Blur100, VLM on lowers LPIPS but also lowers PSNR. No novelty/SOTA
or VLM-necessity claim is established. This remains Rain/Blur only, not five
types, coupled restoration or raw-token conditioning.

Both final weight/Adam file SHA-256 values independently match their
step10000 resume receipts; next_sample_position320000. Result/metric hashes
and current bound sources/config were checked. No protected pixels or extra
GPU evaluation were accessed in this terminal audit.
Launcher costs: image26910s=7.475h/59.8 H100 GPU-hours;
VLM27969s=7.769166666666667h/62.153333333333336 H200 GPU-hours.
Total121.95333333333333 allocated GPU-hours, excluding prior qualifications.

Report: reports/foundir_joint_moe_v2_terminal_2026-10-27.md
Machine-readable: reports/foundir_joint_moe_v2_final_comparison_2026-10-27.json

## MILESTONE: image-only step6000 — Rain PASS, Blur still short — 2026-10-26

image-only scheduled step6000: Rain10034.3965064204137/.07232339138630778
passes both Rain goals; Blur100 preview25.542105757879234/.29018930681515487
still fails both Blur goals. Both metrics improved since3000, but Blur remains
about.358 dB short and.11019 LPIPS above its ceiling. This is not joint success
or fullBlur2595. VLM step6000 is pending. Full step6000 resume receipt saved,
next_sample_position192000. Continue BOTH frozen10000-step runs to terminal
completion as requested; no retuning, checkpoint selection or extra GPU work.
Report: reports/foundir_joint_moe_v2_step6000_2026-10-26.md.

## MILESTONE: BOTH step3000 Rain PASS, Blur still FAIL — 2026-10-26

Both formal arms completed the scheduled step3000 evaluation.
Rain100 image-only34.32609266908536/.07442104332614691;
VLM34.32999647989338/.0712783659575507: both pass Rain AT THIS CHECKPOINT.
Blur100 preview image-only25.399459516822663/.31455456842668356;
VLM25.38343235445578/.30998902085004376: both still fail Blur.
VLM has lower LPIPS on both tasks, near-equal Rain PSNR and slightly lower
Blur PSNR. This is an encouraging perceptual trend, NOT consistent two-metric
VLM superiority, a final joint PASS, significance, novelty or fullBlur2595.
Both saved full step3000 receipts (next_sample_position96000) and continue.
Preserve fixed10000-step final selection; do not select step3000 or retune.
User requested monitoring BOTH jobs to terminal completion. No new GPU job,
protected-data access or source/config change.
Report: reports/foundir_joint_moe_v2_step3000_2026-10-26.md.

## MILESTONE: both joint MoE formal arms at step1000 — 2026-10-26

The user requested monitoring BOTH formal_v1 jobs until terminal completion.
Continue read-only monitoring; do not stop, restart, retune, change bound code,
auto-launch another experiment or interpret successful execution as quality PASS.

Both first scheduled post-training evaluations are available.
Rain100 image-only: 33.9101104544897/.07389099790481851;
VLM: 33.92674730344711/.07357312987558544.
Blur100 PREVIEW image-only: 24.841265019393163/.35753815472591666;
VLM: 24.776332465206465/.35622713952558116.
VLM is slightly better on Rain, but has lower Blur PSNR and better Blur LPIPS.
No consistent VLM advantage or absolute quality PASS is established.
Both improve Blur versus initialization but worsen Rain. Keep the frozen
10000-step schedule; next quality milestone is3000, then6000 and10000.
Full Blur2595 and final constant-expert/VLM-off diagnostics remain pending.

Both step1000 complete resume receipts are present, next_sample_position32000.
Each binds the same config/source identities. Checkpoint hashes in the report
are trainer-recorded; this milestone audit hashes metric/receipt files only.
Hardware remains8xH100 versus8xH200, with the documented step0 numerical caveat.
Report: reports/foundir_joint_moe_v2_step1000_2026-10-26.md.

## ACTIVE: BOTH formal MoE arms confirmed training — 2026-10-26

Read-only shared-log check at 2026-10-26T20:05:52Z confirms BOTH formal_v1
arms are training, superseding earlier "VLM not started" statements below.
image_only: 480/10000 on 8xH100; vlm_assisted: 60/10000 on 8xH200.
Both advance with finite logged losses/gradients, peak memory about 42.57 GiB,
and no observed traceback/OOM/NCCL fatal or launcher END. These are log-based
observations, not direct remote GPU telemetry. Do not restart or retune.
The VLM launch was at 19:54:58Z; image-only at 19:45:55Z.

Config, all 29 source bindings and data identities match. Inputs/record IDs
and input metrics match exactly. Formal step0 OUTPUTS differ slightly across
H100/H200: VLM-minus-image Rain +.0003037634628881847 dB /
-.000008241008035841935 LPIPS; Blur100 preview +.0009170851552084969 dB /
-.000034258328378222735 LPIPS. Do not claim bitwise-identical formal outputs,
attribute these tiny differences to VLM efficacy, or compare runtime as
hardware-matched. Earlier same-H100 qualification equality remains valid.

image-only step250 weights/Adam/resume receipt are saved; independent file
hashes match the receipt, next_sample_position=8000. No post-training
validation exists yet: first at step1000; full Blur2595 only at step10000.
Keep fixed final selection, frozen recipe and separate single-node jobs.
No protected pixels, new GPU launch or running-code/config edit in this check.
Detailed status/hashes: reports/foundir_joint_moe_v2_parallel_formal_start_2026-10-26.md.


## ACTIVE: independent parallel formal arms on TWO nodes — 2026-10-26

The user reported another free GPU node and requested checking the current
run before starting the VLM contrast there. The image-only job is healthy:
image_only/formal_v1 started 2026-10-26T19:45:55Z on8xH100 at
eight-h200-copy-copy-copy--daf90a2bdaa1-k3xjvzod7q. A read-only audit observed
advancing steps50->70, finite logged losses/gradients, peak42.56962299346924
GiB and no terminal marker. All29 bound source hashes and data identities
match qualification; step0 Rain100/Blur100 metrics exactly match its starting
point. No post-training quality milestone exists yet (first at1000).
Do not stop, restart, retune or edit hash-bound training code/config.

The earlier sequential-only plan was a ONE-NODE resource constraint.
It is now superseded: VLM/formal_v1 may run concurrently on the user's
OTHER free homogeneous8xH100/H200 node. Prefer8xH100 to match the active
hardware; if using8xH200, record the difference and do not call runtime a
hardware-matched comparison. This is TWO independent single-node jobs,
not a16-GPU cross-node DDP job. No master IP, node ranks, nnodes change,
global-batch change or cross-node communication setup is needed.

Keep the same frozen code/config, seed, source/crop order, training data,
losses and10,000-step schedule. Each formal arm independently starts from
original Rain step4000, NEVER from image-only progress or qualification.
Each uses all8 GPUs on its own node. Existing /tmp flock/tmux sockets are
node-local; output/log paths include the arm and therefore do not collide
in the shared workspace. Do not start VLM on the busy image-only node.
No launcher changes are needed or authorized merely for parallel scheduling.

At this check VLM/formal_v1 has no log/output yet. Codex has no GPU/node
execution channel; provide the existing command for the OTHER GPU terminal:
bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh formal vlm_assisted formal_v1
Do not claim VLM started until its own shared log confirms it. No GPU job
was started or current job interrupted by this scheduling/documentation turn.

VLM budget stays5.45-9.06 wall hours /43.63-72.45 allocated GPU-hours.
Parallel completion can be roughly one-arm5.5-9.1-hour scale from the second
launch if throughput is comparable, not the old11-18-hour serial wait.
Total compute for both arms remains87.26-144.89 GPU-hours; parallelism does
not halve compute. Shared-storage contention or hardware differences may
extend wall time. Scientific acceptance/quality claims remain unchanged.

## ACTIVE: paired formal joint MoE phase authorized — 2026-10-26

The user explicitly requested "start now" for the formal phase after BOTH
25-step engineering qualifications passed and the paired budget was reported.
This supersedes all older "formal unauthorized" statements below. The
immediate launch is image_only/formal_v1 for10,000 steps on one homogeneous
8xH100/H200 node; vlm_assisted/formal_v1 is the planned matching second arm,
run sequentially on the same node after the first finishes. Do not start both
tmux jobs together or auto-start a second job from a monitoring turn.

Use the existing frozen runner/starter/config unchanged. Both formal arms
independently initialize from original Rain step4000 with fresh Adam and the
same seed/data/crop order/losses. Do not pass a resume receipt and do not
initialize from either qualification. Formal validates BOTH qualification
receipts and checkpoint hashes before training. The two tasks remain Rain
and Blur, not five-type or coupled restoration. No protected pixels.

Read-only formal launch check completed: the trainer's actual
require_qualifications validator passed both receipts and all four weight/
optimizer file hashes against current config/source/data identities.
Both formal_v1 output/log paths are unused; shell syntax checks passed.
No training/evaluation or GPU launch was performed by this readiness check.

Per arm10,000 steps,320,000 presentations (160,000 per task), with frozen
evaluation at0/1000/3000/6000/10000 and fullBlur2595 only at10000. Keep the
predeclared fixed final checkpoint, no interim selection or retuning.
Checkpoint at250, every1000 and milestones includes weights/Adam/eight-rank
RNG for later same-arm exact resume into a fresh ID if explicitly needed.

Budget from matched8xH100 qualifications: per arm5.45-9.06 wall hours /
43.63-72.45 allocated GPU-hours; both sequentially10.91-18.11 wall hours /
87.26-144.89 GPU-hours. These are estimates, not guaranteed efficacy or speed.
The first post-training milestone at1000 is provisionally35-50 minutes from
launch including startup/evaluation. At this handoff neither formal ID has a
log/output; the Codex shell has no GPU, so user GPU-node execution is required.
Never claim it is running until the new shared launch log confirms it.

First command: bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh formal image_only formal_v1
After the first arm completes, the second command is:
bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh formal vlm_assisted formal_v1
Do not join detached tmux starters with && to imply sequencing; starter exit
only means session creation, not completed training. No new GPU job was
launched by this CPU agent. Preserve all successful qualification receipts.

## TERMINAL: BOTH joint MoE V2 qualifications PASS — 2026-10-26

vlm_assisted/qualification_nvls_v2 completed 25/25 steps on eight H100 80GB
GPUs at 2026-10-26T19:34:33Z, launcher status 0, engineering_pass=true.
Total launcher 278 seconds = .07722222222222222 wall hours /
.6177777777777778 allocated H100 GPU-hours. Peak42.583311557769775 GiB;
median training step1.6632833430194296 seconds. On every rank, both FFN
experts, shared LoRA A/B, image router AND VLM projector updated.
Maximum VLM logit adjustment .05847013369202614; insertion error0.
This proves the VLM path executes/learns, NOT a restoration-quality benefit.

Both arms have identical frozen config, all29 source hashes, data identity,
and byte-identical step0 metric files on Rain100/Blur100. At step25:
Rain VLM33.86719081854514/.07618030223064125 versus image-only
33.86460822453734/.07575711041688919, delta+.0025825940077979226 dB /
+.00042319181375205517 LPIPS. Blur100 PREVIEW VLM
24.269234594175536/.4070244254171848 versus image-only
24.26987270296453/.4072400273568928, delta-.0006381087889941739 dB /
-.0002156019397079878 LPIPS. Tiny mixed differences show no clear VLM
advantage at25 steps. Both lose Rain quality versus initialization; both
improve Blur versus initialization but Blur LPIPS remains worse than input.
No fullBlur2595, absolute quality success, significance or novelty claim.

The user's monitoring request is fulfilled. No job was auto-launched and
no formal training is authorized. Both engineering gates now pass, so the
next USER decision can be paired10,000-step formal arms on the same single
eight-GPU node. Fresh independent initialization from original Rain step4000,
never from either qualification; keep data/seed/recipe/evaluation unchanged.
Using the slower measured1.6934155019698665 s/step and the existing launcher
overhead formula, provisional per-arm wall5.453931949916296-9.055897924874444
hours /43.631455599330366-72.44718339899555 allocated GPU-hours.
Both arms sequentially:10.907863899832591-18.111795849748887 wall hours /
87.26291119866073-144.8943667979911 GPU-hours. This includes estimated
evaluation/save/I-O overhead, is not measured full-run speed or a guarantee.
Both successful qualifications cost1.5911111111111111 allocated H100 GPU-hours.

Result SHA-256:
2f589ddf3fc6a2e666f74107fb85062c802149d57a5d9dc4572a91c6566c9506.
Report: reports/foundir_joint_moe_v2_vlm_qualification_terminal_2026-10-26.md.

## ACTIVE: VLM-assisted joint MoE qualification authorized — 2026-10-26

The user explicitly said execute for the separate VLM-assisted 25-step
engineering qualification. No further approval is required for this exact
qualification; formal training remains unauthorized. Run
vlm_assisted/qualification_nvls_v2 on one homogeneous 8xH100/H200 node
through the existing tmux starter, with its bounded NCCL probe first.
Initialize independently from original Rain step4000 with fresh optimizer;
NEVER initialize from the image-only qualification or any failed MoE.
The same frozen config, all 29 bound source files, data, seed, losses and
evaluation protocol match the completed image-only qualification. No code
or scientific protocol changes were made for this authorization.

At this handoff, no VLM launch log/output exists and the Codex shell has
no CUDA device node. Do not claim it started: the user must execute the GPU
command below, then inspect the shared log. Do not launch on the CPU shell.
Expected wall time is provisionally 10-20 minutes / 1.3333-2.6667 allocated
GPU-hours, based on the matched image-only run's 7.3 minutes; not a measured
VLM duration. Retain the earlier conservative .81-hour / 6.47-GPU-hour
envelope for slow startup/shared I/O. The launcher still prints its original
conservative estimate; do not mutate hash-bound launch code to change it.
Once complete, verify both engineering receipts and compare matched
initial/final metrics; 25-step quality changes do not establish novelty.
Then report BOTH-arm measured formal budget and wait for formal permission.

Command: bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh qualification vlm_assisted qualification_nvls_v2

## TERMINAL: joint MoE V2 image-only qualification PASS — 2026-10-26

image_only/qualification_nvls_v2 finished all 25 steps normally on eight
NVIDIA H100 80GB GPUs at 2026-10-26T19:20:40Z, launcher status 0.
The no-update NCCL/DDP probe passed, followed by engineering_pass=true,
zero insertion error, finite training/checkpoint validation, and observed
updates to both FFN experts, image router and shared LoRA A/B on all eight
ranks. VLM was inactive as intended. Peak memory 42.57999897003174 GiB;
median training step 1.6934155019698665 seconds. Total launcher cost:
438 seconds = .12166666666666667 wall hours / .9733333333333334 allocated
H100 GPU-hours. This validates the repaired path on this H100 node, NOT
the original H200 node or a uniquely proven cause of the earlier CUDA401.

Matched Rain100: step0 34.17715539550872/.07015681236982346 ->
step25 33.86460822453734/.07575711041688919. Rain regresses by
.31254717097137785 dB and .00560029804706573 LPIPS; PSNR fails >=34.2.
Matched Blur100 PREVIEW: 23.8596342447602/.4188850187882781 ->
24.26987270296453/.4072400273568928 (+.41023845820432925 dB,
-.01164499143138531 LPIPS). Blur output LPIPS still exceeds input
.37266220796853305. No full Blur2595 evaluation, no quality gate PASS,
no demonstrated VLM benefit and no novelty conclusion. Do not extend or
retune from this 25-step engineering result.

All 29 bound source hashes, config/receipt/history/metric identities and
weight/optimizer SHA-256 were independently checked. No protected pixels.
Result SHA-256:
249071478a7f68d3d3bc8d812e9736497fee3d19a2ca5d3f60147c0c6ab5d278.
Report: reports/foundir_joint_moe_v2_image_only_qualification_terminal_2026-10-26.md.

The user's monitoring request is fulfilled; this launcher is finished.
Next planned task is a separate fresh vlm_assisted/qualification_nvls_v2
25-step qualification, only on user launch instruction. It does not require
image-only quality to pass first. No VLM/formal job was auto-launched.
Both matching engineering receipts are required before any formal launch
decision and measured budget. Never initialize formal from qualification.
Preserve both the failed qualification_v1 and successful qualification_nvls_v2.

## ACTIVE: authorized joint MoE V2 transport repair / retry — 2026-10-26

The user explicitly approved repairing the failed image-only qualification.
This supersedes the previous wait-for-repair-approval instruction, not the
scientific contract. The actual runner now enforces NCCL_NVLS_ENABLE=0 and
NCCL_P2P_DISABLE=0 inside the tmux job, independent of inherited server values.
Before loading restoration models/data, it runs the existing eight-rank
cure_nccl_rehearsal_v1.py (scalar all-reduce + tiny DDP backward, no optimizer
update or image access) under a 180-second timeout plus 30-second kill grace.
Failure/timeout exits without training; success continues ONLY the requested
25-step qualification. Detailed NCCL logging is scoped to the probe process.
No model, config, dataset, loss, training step count or protected-data scope
changed. Formal and the other arm do not auto-start.

Use a fresh image_only / qualification_nvls_v2 run ID. Never overwrite the
failed qualification_v1 log. Default receipt paths now use qualification_nvls_v2
for both arms; explicit paths remain supported. The VLM arm still requires
a separate launch decision after checking this retry's engineering result.
CPU launcher tests cover forced env values, probe-before-training order,
failure/timeout blocking, training error propagation and existing-log safety.
All five launcher tests plus all 18 original joint-MoE/data CPU tests passed.
Actual H200 transport recovery, memory safety and training remain unverified
until the user's GPU-node retry; the Codex shell has no GPU access.
Budget per qualification remains provisional .20-.75 wall hours / 1.6-6.0
allocated GPU-hours; the new probe adds at most 210 seconds (.058333 hours /
.466667 allocated GPU-hours) if its timeout/kill grace is exhausted.

Command: bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh qualification image_only qualification_nvls_v2
Repair report: reports/foundir_joint_moe_v2_nccl_repair_2026-10-26.md.

## TERMINAL: joint MoE V2 first launch failed at NCCL initialization — 2026-10-26

The image_only / qualification_v1 launcher ran on eight NVIDIA H200 GPUs
from 2026-10-26T18:57:06Z to 18:58:49Z, exit1, 103 seconds
(.02861111111111111 wall hours / .2288888888888889 allocated GPU-hours).
All ranks failed in dist.init_process_group('nccl', device_id=device),
CUDA401 "the operation cannot be performed in the present state".
No optimizer step, model loading, output directory, checkpoint or evaluation
was reached. This is initialization failure, NOT scientific quality failure.
The user requested monitoring through completion; that monitoring is fulfilled.

The current launcher lacks NCCL_NVLS_ENABLE=0 / NCCL_P2P_DISABLE=0,
which the existing 2026-09-26 H200 recovery explicitly needed and verified.
The new failure matches that earlier pattern, but this run has no detailed
NCCL NVLS trace, so do not present the specific transport cause as proven.
Do not infer that the hardware is faulty or recommend changing the model.
The proposed next step is the existing H200 transport compatibility setting,
a no-update communication check, then a fresh-ID retry of the SAME25-step
qualification. No launcher/code change or GPU retry was made during this
monitor-only turn. Await user instruction for the repair/retry; preserve the
failed log and never reuse qualification_v1. Neither formal nor VLM arm starts
automatically. All preceding CPU readiness remains CPU-only.

Report: reports/foundir_joint_moe_v2_image_only_launch_failure_2026-10-26.md.
Failed log SHA-256:
bbee8607da8ed02d3b6befff33c01e0339f772da31b9f6ca703d33521493cfb8.

## ACTIVE: joint image-only versus VLM-guided MoE — 2026-10-26

The user explicitly cancelled the serial prerequisite "new one-step Blur
expert must pass first, then image-only MoE, then VLM". This supersedes ALL
older launch holds/order statements below for this new V2 experiment only.
Do not resume failed Blur-only, frozen-Rain FFN-only MoE, EBT or energy runs.
Original successful Rain/old Blur checkpoints remain untouched.

The task is a matched TWO-TASK Rain/Blur study, not yet five-type all-in-one
or coupled restoration. Both arms jointly train the same full FFN experts,
image routers and ONE shared rank32 LoRA at all 50 original attention/stream
targets. The old restriction "freeze Rain/shared attention, train FFNs only"
is removed. Shared LoRA is not a LoRA expert bank. Both arms initialize from
original Rain step4000, use identical empty text conditioning and one-step
endpoint inference. Do not claim one-step Blur feasibility is already proven.

Only the VLM-assisted arm adds per-image frozen Qwen predicted five-slot
strengths through a 5->256->2 MLP to each image router's logits. This is the
advisor-allowed LABEL/STRENGTH interface, NOT raw VLM tokens. Full prediction
cache is joined strictly by ID/source/LQ+GT hashes; never use
type_oracle_levels_json or GT for inference. Existing VLM prediction
center448 crops differ from random512 restoration crops; disclose this
image-level-condition limitation. Qwen preprocessing cost is additional,
not zero merely because predictions are cached.

Both arms use task-neutral decoded RGB/latent/LPIPS/high-pass supervision.
Remove Rain-specific change-mask weighting, unchanged-region identity, and
Restormer teacher; retain remaining coefficients. Train-only task CE and
250/500 hard-route warmup are identical in both arms, absent at inference.
This bundled repair is not a single-factor causal ablation of old failures.
VLM superiority must be assessed ONLY against its matching repaired baseline.

Contract: configs/foundir_rain_blur_joint_vlm_moe_v2.json
SHA-256: 3bdd59b3fb3c010cc7a93ab299f3f67fe85cc5eb5a6f380f610d4338398baebe.
Data: unchanged 38,334 Rain + 105,057 Blur training records, source-balanced
16+16 per batch32, random512/D4; unchanged Rain100/Blur2595 development
crops with Blur100 previews. No protected/official/sealed/internal pixels.

First run separate fresh25-step engineering qualifications for BOTH arms on
one homogeneous 8xH100/H200 node. Each estimate .20-.75 wall hours /
1.6-6.0 allocated GPU-hours, provisional. No GPU is available in the Codex
shell; no new GPU job has been launched by the agent. No automatic formal
or other-arm launch. Qualification checkpoints must NEVER seed formal.
Image-only scientific-quality failure is no longer a veto on the matched
VLM arm; engineering failure still blocks an unsafe formal launch.

Prepared formal recipe: 10,000 steps per arm, 320,000 presentations total /
160,000 per task (not unique records, not guaranteed complete traversal).
Same seed/crops/order/optimizer/loss/inference across arms. This is half the
old Blur expert's 320,000 presentations per task, not a matched old-model
training-budget claim. Measure BOTH qualifications and report wall hours /
GPU-hours before a user formal launch decision. Require both valid
engineering receipts; never require image-only to pass the SOW first.
Save complete weights/Adam/eight rank RNGs at250, every1000 and milestones;
resume only same-arm incomplete formal into a fresh output ID.
Training-state storage is about117 GB per arm before metrics/images.

Final fixed-step comparison reports per-task PSNR/LPIPS, paired source-level
bootstrap intervals, win fractions, expert use and constant-expert/VLM-off
diagnostics. Preserve Rain >=34.2/<=.08 and Blur >=25.9/<=.18 as separate
absolute goals. A tiny gain over a bad baseline is not paper success.
No broad first-VLM-MoE, novelty-established or SOTA claim is authorized.

CPU readiness: 18 tests passed (including two-rank DDP, full-model meta
parameter audit, checkpointed gradients and exact resume). Real-asset
preflight passed; all146,086 allowed records have hash-matched predictions.
No GPU qualification or new scientific result exists yet.
Readiness/commands:
reports/foundir_rain_blur_joint_vlm_moe_v2_readiness_2026-10-26.md.
Launcher:
bash smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh qualification image_only qualification_nvls_v2
Run the VLM qualification separately after checking the first engineering
result; no standalone Blur quality gate in between.

## TERMINAL: Blur-only formal complete — quality gate FAIL — 2026-10-26

This supersedes every older LIVE/ACTIVE/unlaunched Blur-only status below.
formal_v1 completed all 3,000 steps and full Blur2595 evaluation at
2026-10-26T16:49:10Z, launcher exit 0. Engineering PASS, joint quality
gate FAIL. PSNR 26.096176888792407 passes >=25.9, but LPIPS
0.3260975599297016 fails <=.18 by 0.14609755992970158.
Old matched MoE is 26.10596660902048/0.3271988157297777:
single-task changes only -0.009789720228074117 dB /
-0.0011012558000761397 LPIPS. Removing routing/mixed training did
not materially solve this Blur branch. Do not claim a working MoE or novelty.

Input is 25.849544629199475/0.29110622298011024; LPIPS is worse on
2280/2595 records. Final matched Blur100 preview is
25.067034539869926/0.37915658176876604; never compare the
full2595 final mean directly against earlier100-record preview means.
This control does not prove one-step FLUX is impossible or isolate a specific
loss/prompt/update-scope cause. Original Rain/old Blur remain untouched.

Launcher 3371 seconds = 0.9363888888888889 wall hours /
7.4911111111111115 allocated H100 GPU-hours. Median step
0.8779867839766666 seconds; rank0 peak 29.185157299041748 GiB.
48,000 Blur presentations, 12 complete checkpoints. Source/config/metric
identities and final artifact hashes verified. All 424,673,280 parameters and
10 Adam states independently verified finite FP32, Adam step 3,000, next
sample position 48,000, eight rank RNG states. No protected pixels or runtime error.

The user's monitoring request is fulfilled. No job is active from this
launcher. Do not auto-extend, restart, tune, add VLM, or launch another
experiment. Preserve all results and wait for the user's next decision.
Report: `reports/foundir_blur_single_endpoint_formal_terminal_2026-10-26.md`.
Result SHA-256:
`6224638f9b2a65f15f00839f338eb7cb0782b5c4ba028de16af41bad7bb2d1b9`.

## MILESTONE: Blur-only formal step 2,000 — 2026-10-26

formal_v1 remains LIVE beyond 2,000. Matched Blur100:
24.91918972080556/0.38884398228721695. Relative to step1,000:
+0.06952350409890329 dB / -0.0028100751596502604 LPIPS.
Old MoE at step2,000 is 24.823461629491018/0.38983108070679007;
single-task advantage is only +0.09572809131454108 dB /
-0.000987098419573118 LPIPS. Output LPIPS remains worse than input
0.37266220796853305. No full Blur gate or solved-branch claim.
Continue frozen final 1,000 steps and full2595 evaluation, no extension/retune.

Source/config/record/metric identities and weight/Adam hashes verified.
Checkpoint saved 2026-10-26T16:28:57.101968+00:00; next sample position32,000.
Cost to checkpoint: 0.5994727688888889 wall hours /
4.795782151111111 allocated GPU-hours, not terminal cost.
Report: `reports/foundir_blur_single_endpoint_formal_progress_2026-10-26.md`.

## MILESTONE: Blur-only formal step 1,000 — 2026-10-26

formal_v1 remains LIVE beyond 1,000; continue unchanged to 3,000 and full
Blur2595 evaluation. On the matched Blur100 preview, single expert is
24.849666216706655/0.3916540574468672, versus old MoE at the same step
24.8058798027948/0.3938900137133896.
Only +0.04378641391185667 dB / -0.0022359562665223986 LPIPS.
Output LPIPS remains worse than input 0.37266220796853305.
Do not call this a solved Blur branch, a working MoE, or a full-development
gate pass. No restart, extra tuning, VLM, or new experiment is authorized.

Checkpoint/Adam hashes, source/config identities, record identities and
metric means verified; checkpoint step1,000, next Blur position16,000.
Checkpoint saved at 2026-10-26T16:11:34.685216+00:00;
0.30991256 wall hours / 2.47930048
allocated GPU-hours since launch, not terminal cost.
Report: `reports/foundir_blur_single_endpoint_formal_progress_2026-10-26.md`.

## LIVE: Blur-only formal training started — 2026-10-26

The user launched `formal_v1` at 2026-10-26T15:52:59Z on one 8xH100 node
and asked to monitor through successful completion. Initial Rain100/Blur100
parity passed with exact original metrics; finite optimization is observed
beyond step 30. This supersedes the prior authorized-but-unlaunched status.
Do not duplicate, restart, retune or stop the frozen run without user direction.
Continue through 3,000 steps, the full 2,595-record Blur evaluation and launcher
END status. Normal execution is not proof of the PSNR/LPIPS quality gate.

Session: `blur_single_formal_v1`.
Log: `logs/foundir_blur_single_endpoint_v1_formal_v1.log`.
Output: `checkpoints/foundir_blur_single_endpoint_v1/formal_v1/`.
Same original Rain initialization, no qualification continuation, no router/VLM.
Save every 250 steps; evaluate at 1,000/2,000/3,000 as frozen. Preserve all
artifacts. Codex sees shared files from its CPU-only node, not GPU telemetry.
Estimate remains 1.11-2.14 wall hours / 8.88-17.12 allocated H100 GPU-hours,
to be updated from measured progress. No extension or next experiment is authorized.

## AUTHORIZED: fresh Blur-only formal launch — 2026-10-26

After qualification passed, the user requested the official formal launch
command. The frozen 3,000-step Blur-only formal is now authorized under run ID
`formal_v1`. Start from the original successful Rain initialization, NOT the
25-step qualification weights. No router/VLM, loss change or schedule change.
Qualification/source/config identities match; formal output/log were absent
at prelaunch check. The agent has NOT launched a GPU job; wait for the user's
node launch and verify actual logs before marking it live.
Command: `bash smoke/start_foundir_blur_single_endpoint_v1_tmux.sh formal formal_v1`.
Expected 1.11-2.14 wall hours / 8.88-17.12 allocated GPU-hours on 8xH100;
H200 is supported but its speed is not measured by this qualification.

## MILESTONE: Blur-only qualification PASS; formal not launched — 2026-10-26

qualification_v1 completed 25/25 steps on one 8xH100 at 2026-10-26T15:44:10Z,
launcher exit 0. This supersedes the older GPU-qualification-pending status.
Initial Rain100 and Blur100 per-record PSNR/LPIPS parity errors are exactly 0.
Five FFN blocks updated, finite training, no NaN/OOM/NCCL/runtime failure.
Final Blur100 preview: 24.17730141751229/0.39682197285816073,
versus initialization 23.788320945242468/0.42077609404921534;
deltas +0.3889804722698216 dB / -0.02395412119105461 LPIPS.
PSNR improves on 50/100 and LPIPS on 93/100. Engineering PASS only:
no full Blur2595 evaluation and no formal scientific success yet.

Launcher: 331 seconds = 0.09194444444444444 wall hours /
0.7355555555555555 allocated H100 GPU-hours.
Median step 0.9119640614371747 seconds; rank0 peak
29.184199810028076 GiB. Formal estimate now 1.11-2.14 wall hours /
8.88-17.12 GPU-hours, including provisional full-eval/checkpoint IO.
Source/config/metric identities and saved artifact hashes verified.
All 424,673,280 parameters and 10 Adam states independently verified finite
FP32; Adam step 25, next sample position 400, eight saved rank RNG states.
The user's monitoring request is fulfilled. No automatic formal or VLM.
Formal must start fresh from original successful Rain, never qualification.
Keep the frozen 3,000-step/48,000-Blur exposure protocol and save every 250.
Report: `reports/foundir_blur_single_endpoint_qualification_2026-10-26.md`.
Result SHA-256:
`d1d5e1ca2ae616b3c9ea9473381b125ca79cb37b23b9f95f44e40bbe9f4110f0`.

## ACTIVE: Blur-only one-step diagnostic, no router — 2026-10-26

This supersedes older active/hold instructions for the immediate next task.
After the failed joint Rain+Blur gate and weak fixed-expert Blur result, the
user authorized a standalone Blur branch diagnostic. Implement/qualify ONLY
this control; do not resume the failed MoE, EBT/energy, add VLM or alter the
successful original Rain/old Blur experts.

Use `configs/foundir_blur_single_endpoint_v1.json`, SHA-256
`a273b55ccfd9ca128b976ea5a6fa314ce862d77273a4127f0e5836e58957eefb`.
Same original Rain step4,000 initialization, five full image FFNs with ONE
expert per block and no router (424,673,280 trainable FP32 parameters).
Shared model/Rain LoRA stay frozen. Keep the same fixed Rain prompt, one-step
t=1/zero-noise FP32 endpoint, losses and LR1e-5; teacher weight remains zero.
Train Blur only. Batch16 (8 ranks x 2 accumulation), 3,000 steps =48,000 Blur
presentations. Exact Blur record/crop/augmentation positions match the old
MoE's 16-Blur half of each batch. Scale mean Blur loss by .5 to preserve its
original mixed-objective contribution; do not silently renormalize it.

This is a no-routing/single-task diagnostic, NOT a parameter-matched or
pure-router causal ablation. A failure would not prove one-step FLUX is
impossible; frozen Rain conditioning, update scope and objective remain
possible limitations. No new scientific improvement is available yet.

First run fresh25-step `qualification_v1` on one homogeneous 8xH100/H200.
Require initial per-record Rain100 AND Blur100 parity, finite gradients,
actual FFN updates, FP32 weight/Adam saves and valid matched evaluation.
No automatic formal. Formal must restart original Rain, never qualification.
Formal evaluates at0/1000/2000/3000; fullBlur2595 only at final. Target
>=25.9 PSNR / <=.18 LPIPS, with per-record regressions reported separately.
Save complete state every250 steps. Resume requires a finished hash-bound
`.resume.json`, formal mode, unchanged code/config, Adam and absolute
sampling state, and a fresh output ID. Never overwrite a prior run.

Qualification estimate .15-.50 wall hours /1.2-4.0 allocated GPU-hours.
Formal provisional1.0-2.5 hours /8-20 GPU-hours, to be re-estimated from
qualification throughput and checkpoint/evaluation IO. No protected data.
Agent runtime is CPU-only; GPU qualification remains pending user launch.
Readiness checks PASSED: 28 CPU tests including two-rank DDP, exact Blur
exposure/crop matching and uninterrupted-versus-resumed state equality;
Python/shell/JSON syntax and full real-asset read-only preflight passed.
These are not GPU memory qualification or scientific quality results.
Readiness: `reports/foundir_blur_single_endpoint_readiness_2026-10-26.md`.
Launch: `bash smoke/start_foundir_blur_single_endpoint_v1_tmux.sh qualification qualification_v1`.
Continue Mandarin-only, plain-language reporting and focused GitHub tracking.

## TERMINAL: image-only Rain/Blur MoE formal — joint gate FAIL — 2026-10-26

This supersedes every older LIVE/ACTIVE/unlaunched MoE status below.
formal_restart_v1 completed 3,000/3,000 steps plus full evaluation and
fixed-expert diagnostics at 2026-10-26T13:30:15Z, launcher exit 0.
Engineering pass, but JOINT SCIENTIFIC GATE FAIL. No new run is active
from this launcher; do not automatically extend, restart, retune, run dense,
or add VLM. User requested monitoring through completion, which is fulfilled.

Final Rain100: 34.36398316275458/.07267702806275338, both >=34.2/<=.08 pass.
Final full Blur2595: 26.10596660902048/.3271988157297777, PSNR>=25.9
passes but LPIPS<=.18 FAILS by .14719881572977772. Do not call the two-task
baseline successful, novelty established or SOTA. Blur input itself is
25.849544629199475/.29110622298011024: output improves PSNR only
+.2564219798210061 dB and WORSENS LPIPS by .036092592749667474.
LPIPS worsens vs input on 2,324/2,595 Blur records.

Same final Blur100 preview reaggregated from full output:
24.966814240094454/.3831212885398418. Fixed expert1 on those same records
is 25.041084236119197/.38290703640086576, only slightly better and still
poor. Do not blame the main gap solely on the router. Explicit VLM remains
deferred until a genuinely working image-only baseline. Next proposed work,
if requested, is an evidence-based Blur quality audit, not an automatic
architecture/training change. Preserve original successful Rain/old Blur.

Launcher cost: 6,847 seconds = 1.9019444444444444 wall hours /
15.215555555555556 allocated H100 GPU-hours on 8xH100, excluding prior
interrupted run/qualification. Trainer-only 1.8740228895263862 hours /
14.99218311621109 GPU-hours. Median step 2.001327611040324 seconds;
rank-0 peak allocated memory 37.40372896194458 GiB.
96,000 presentations; unique Rain27,228 / Blur30,808. No protected pixels.

All source/config/checkpoint hashes and metric aggregates verified.
Final 50 Adam states are finite FP32, step=3,000, next_sample_position=96,000.
Initial Rain per-record equality remained exact. No NaN/OOM/NCCL/runtime
error. Do not confuse successful completion with scientific gate success.
Report: `reports/foundir_rain_blur_image_moe_formal_terminal_2026-10-26.md`.
Machine-readable: `reports/foundir_rain_blur_image_moe_formal_terminal_2026-10-26.json`.
Result SHA-256: `e13f70ca2169f0f1d0bb584764ab3176ecad16700c96a40d3eac5c1b08e42254`.

## MILESTONE: image-only MoE formal restart step 2,000 — 2026-10-26

formal_restart_v1 remains LIVE beyond 2,000. Rain100:
34.350766710211964/.07022410760167987, both gates pass, improves on 1,000.
Blur100 preview: 24.823461629491018/.38983108070679007, only
+.017581826696218883 dB / -.004058933006599541 LPIPS versus 1,000.
Blur progress has slowed markedly; no evidence warrants promising a joint
pass after the final 1,000 steps. Preview is still NOT full Blur evaluation.
Weights/Adam hashes, FP32 weight header, record identities and metric means
verified. Save time 12:48:41 UTC, cumulative 1.209202438888889 wall hours /
9.673619511111111 allocated H100 GPU-hours since restart (not terminal cost).
Continue unchanged through 3,000 and full Blur evaluation. No extension,
retuning, dense arm or VLM authorized.
Report: `reports/foundir_rain_blur_image_moe_formal_progress_2026-10-26.md`.

## MILESTONE: image-only MoE formal restart step 1,000 — 2026-10-26

formal_restart_v1 remains LIVE and has passed its first saved milestone.
Rain100: 34.238711513652106/.07147581973578782, both >=34.2/<=.08 gates pass.
Relative to original Rain initialization: +.011278978157413633 dB,
-.0022033531684428342 LPIPS. Blur100 preview:
24.8058798027948/.3938900137133896, improved but far from target, especially
LPIPS; this is NOT the full 2,595-record Blur gate. No two-task success claim.
Task-average router weights now clearly distinguish Rain/expert0 and
Blur/expert1; do not claim causal routing superiority without controls.
Weights/Adam saved and SHA-verified, FP32 weight header and metric identities/
aggregates verified. Save time 12:14:14 UTC, about .635 wall hours /
5.08 allocated H100 GPU-hours since restart launch (not terminal cost).
Continue unchanged to 2,000 and 3,000; no automatic other arm/VLM.
Report: `reports/foundir_rain_blur_image_moe_formal_progress_2026-10-26.md`.

## ACTIVE: Rain-initialized image-only Rain/Blur MoE — 2026-10-26

The user explicitly approved executing the plain image-only MoE plan after
the Rain gate passed. This supersedes the prior Rain-before-MoE hold and the
failed raw-FLUX one-MLP pilot. The only new task is Rain+Blur ordinary DiT-MoE;
no VLM branch, LoRA expert bank, EBT/energy, pixel refiner, extra task or mixed
degradation experiment. Preserve the successful Rain and existing Blur experts.

Contract: `configs/foundir_rain_blur_image_moe_v1.json`, SHA-256
`8988e2374ad736f3e406945f1d9db83ae14587d0aba28a04dbf9cdc900ab5f12`.
Start from the hash-bound successful Rain step-4,000 LoRA on FLUX.2 Klein 4B.
Keep that single Rain LoRA and shared backbone frozen. Replace image MLPs in
all five double-stream blocks with two ordinary full FFN clones per layer;
learn only those FFNs and image routers (853,313,290 FP32 parameters).
There is no second external MoE and no bank of LoRA experts. The dense
control trains one FFN per same block (424,673,280 parameters); disclose the
capacity/compute difference, never call it a parameter-matched causal test.

Both tasks use the SAME fixed cached Rain text context and SAME one-step,
t=1, zero-noise FP32 endpoint. The fixed prompt preserves Rain initialization;
it does not expose each image's task label. No runtime Qwen/VLM.
Do not load the old 50-step Blur LoRA into this one-step path or retrain that
old expert. The NEW MoE must learn Blur from allowed paired GT. Keep the
existing eight Rain loss terms/coefficients, with the auxiliary Restormer
teacher enabled only on Rain; its weight is zero on Blur.

Use 38,334 Rain / 105,057 Blur training records, balanced 16+16 per global
batch32, source-balanced within each task, aligned random 512 crops and D4.
Cross-task development source groups and GT hashes are excluded; the audited
pools have zero such overlaps. Keep the exact 100 Rain development records
and all 2,595 original Blur development crops. A deterministic 100-source
Blur preview is for intermediate diagnostics only, never the full Blur gate.
No official/sealed/internal/parent-test pixels.

Run ONLY a fresh 25-step `moe/qualification_v1` first on one homogeneous
8xH100/H200 node. Before updates, replay all 100 Rain records and require
per-record agreement with the successful checkpoint (PSNR error <=1e-4,
LPIPS error <=1e-5) AND preserve its 34.2/.08 gate. Then exercise hard training
task routes, transition, and image-only routing (qualification hard6/end12).
Task labels never enter validation/inference; fixed-expert diagnostics do
not use labels either. Require finite gradients, actual expert/router
updates, memory safety, F32 model/Adam files and matched evaluation artifacts.

Formal is configured at 3,000 fresh optimizer steps, 96,000 presentations
(48,000 per task, NOT full traversal), with hard warmup250/transition-end500.
Validate at 0/1000/2000/3000, using full Blur only at formal final. Formal
requires a passing SAME-ARM qualification and never seeds from it. The dense
arm needs its own qualification. No automatic formal/other-arm/VLM launch.
Do not claim a working MoE merely from numeric gates: inspect expert usage
and constant-expert ablations, and compare against the matched-data dense arm.
Rain gate is >=34.2/<=.08; full Blur average gate >=25.9/<=.18, with tail
regressions reported separately as in the historical Blur result.

24 CPU tests, a separate two-rank CPU DDP test and real-file preflight passed.
The user then launched `moe/qualification_v1` on 8xH100. It TERMINALLY
COMPLETED at 2026-10-26T10:41:54Z, launcher exit 0, 25/25 steps, engineering
pass. Initial Rain per-record PSNR/LPIPS errors are both exactly zero.
Both expert branches and routers updated; finite training, FP32 checkpoint
and optimizer artifacts, no OOM/runtime failure. Recorded rank-0 peak memory
37.39800262451172 GiB; median step 1.7897918989765458 seconds.
Launcher: 484 seconds, .13444444444444445 wall hours,
1.0755555555555556 allocated H100 GPU-hours. Trainer-only timer:
.10672626883945971 hours / .8538101507156777 GPU-hours.

Do NOT describe this engineering pass as a working two-task MoE or gate pass.
Rain fell from 34.22743253549469/.07367917290423065 to
33.77674230661362/.07843095208052546 (PSNR gate FAIL, LPIPS pass).
Blur preview improved from 23.788320945242468/.42077609404921534 to
24.169359410868655/.4072999455779791; this is only 100 records, NOT the full
Blur gate. Fixed-expert diagnostics do not establish successful task
specialization, and task-average router weights remain close to half/half.
Preserve the successful original Rain checkpoint and the qualification.
Never resume qualification as formal or repeat its used run ID.

The user explicitly AUTHORIZED the fresh 3,000-step MoE formal on 2026-10-26
after reviewing qualification. Use run ID `formal_v1`, from the ORIGINAL
successful Rain, with unchanged 250/500 warmup and frozen protocol. Never
initialize from qualification weights. Qualification receipt/source/config
identities match and formal_v1 output/log were absent at launch handoff.
The user launched formal_v1 at 2026-10-26T10:59:57Z on one 8xH100 node.
GPU/data preflight and initial 100-record Rain parity PASSED (both maximum
per-record errors 0). Formal step 1 completed with finite loss/gradient.
INTERRUPTION UPDATE: the last formal_v1 log is step 810, last modified
2026-10-26T11:30:30Z, with no launcher END or traceback. The user reports
possibly stopping the node/job accidentally and asks to continue. This does
NOT prove an internal training error or that all old GPU processes are dead.
No step checkpoint or Adam file exists: first save was at 1,000. Only receipt,
initial parity and step-0 metrics/images exist. Therefore there is NO saved
state from which to resume step 810, and no trained development result.

Preserve formal_v1. Once the user confirms the old job/node is stopped, use
fresh run ID `formal_restart_v1` with the SAME frozen 3,000-step recipe and
original successful Rain initialization:
`bash smoke/start_foundir_rain_blur_image_moe_v1_tmux.sh formal moe formal_restart_v1`.
This is a FROM-ZERO FORMAL RESTART, not optimizer/sample-position resume.
Do not reuse the qualification weights or overwrite the old run. The launcher
lock prevents same-node overlap only; never infer cross-node exclusion.
The user launched formal_restart_v1 at 2026-10-26T11:36:08Z on 8xH100.
GPU/data preflight and exact initial Rain parity passed; step 1 completed
with finite loss/gradient. This RESTART is now LIVE. User requested monitoring
until terminal completion. Do not duplicate/restart/retune/interrupt it.
New session: `rain_blur_moe_formal_restart_v1`.
New log: `logs/foundir_rain_blur_image_moe_v1_moe_formal_restart_v1.log`.
New output: `checkpoints/foundir_rain_blur_image_moe_v1/moe/formal_restart_v1/`.
The old formal_v1 paths below describe preserved interrupted history only.
Preserve the unchanged 1,000/2,000/3,000 evaluations; normal execution does not
establish a scientific gate pass. Codex observes shared artifacts from its
CPU-only node, not direct GPU process telemetry.
Session: `rain_blur_moe_formal_v1`; log:
`logs/foundir_rain_blur_image_moe_v1_moe_formal_v1.log`.
Output: `checkpoints/foundir_rain_blur_image_moe_v1/moe/formal_v1/`.
No dense/VLM run is authorized by this launch. Qualification's 6/12 warmup
is only an engineering stress test, not a reason to retune formal.
Measured provisional formal estimate: 1.94–3.24 wall hours / 15.53–25.90
allocated H100 GPU-hours; full Blur evaluation and filesystem IO add uncertainty.
Result SHA-256: `5c2d2b8cdcfc8177c58d938bb87e0e1c76eb4cc81a7ec4c6cae66a927215c071`.
Terminal report: `reports/foundir_rain_blur_image_moe_qualification_2026-10-26.md`.
Readiness: `reports/foundir_rain_blur_image_moe_readiness_2026-10-26.md`.

## TERMINAL PASS: Rain FLUX continuation completed — 2026-10-25

This supersedes every earlier active/live/failed/unstarted Rain status below.
`continue_v1` completed the authorized 3,000→4,000 continuation on 8xH100:
2026-10-25T23:20:16Z to 23:51:52Z, launcher exit 0, no runtime/OOM/nonfinite/
NCCL failure. Final fixed 100-record development PSNR/LPIPS is
34.22743253549469/.07367917290423065: BOTH >=34.2 / <=.08 gates PASS.
PSNR margin is .027432535494689603 dB, LPIPS margin .006320827095769352.
Compared with step 3,000, PSNR gains .21904474864805223 dB; LPIPS worsens
slightly by .0002468797005713008. Do not call this improvement in both means.
The train probe is 34.171933008547796/.05955105775501579 and is NOT the gate.
This passes the frozen development protocol, not an independent official test,
a novelty claim or a SOTA claim. Near-clean over-restoration remains: 14/24
records with input PSNR >=34 lose PSNR versus the input.

The successful mergeable Rain expert is:
`checkpoints/foundir_rain_flux2_precision_resume_v1/continue_v1/step_004000.safetensors`
SHA-256 `020113ce209728452917d49859da7be220f638f0d27f6f87f182ffdab0a8fc2e`.
Inference remains FLUX LoRA only; Restormer was a frozen training-only teacher.
Keep the expert and all historical checkpoints; do not continue training or
change the successful recipe automatically. The Rain prerequisite for the next
image-only DiT-MoE stage is now cleared. Only after a working image-only MoE,
add VLM token routing; no MoE job has been launched by this completion.

Final LoRA/Adam tensors, Adam step=4000, next_sample_position=128000,
configuration/source identities, metric means and matched record/input
identities were independently verified. No protected pixels. Additional
exposure is 32,000 presentations / 17,181 unique records / 5,780 sources;
cumulative presentations 128,000. Total launcher cost is 31m36s,
.5266666666666666 wall hours / 4.213333333333333 allocated H100 GPU-hours.
Trainer-only clock is .5022755164518539 hours / 4.018204131614831 GPU-hours.
Terminal evidence: `reports/foundir_rain_precision_resume_progress_2026-10-25.md`
and `reports/foundir_rain_precision_resume_terminal_2026-10-25.json`.
No Rain training remains active; do not mistake the older LIVE sections for
current instructions.

## MILESTONE: Rain continuation step 3,500 — 2026-10-25

Development is 34.0641582441431/.07644298265688121 on the identical 100
records. Relative to step 3,000: PSNR +.05577045729646102 dB, LPIPS worsens
by .0030106894532218598. PSNR still misses 34.2 by .1358417558569016 dB;
LPIPS remains below .08. This is a modest PSNR improvement, NOT a joint gate
pass or evidence of uniformly improving perception. Per-image wins are
57/100 PSNR, 35/100 LPIPS, 28/100 both. Train probe is
33.95975334657288/.06225986051140353 (not generalization evidence).
Saved LoRA is 100 F32 tensors; all 100 Adam states remain F32/finite with
step=3500 and next_sample_position=112000. Source hashes remain unchanged,
and aggregate means/record identities match the frozen evaluation files.
Training resumed normally beyond 3,500. Continue only to the authorized
4,000-step terminal milestone without retuning or early termination.

## LIVE: Rain step-3,000 continuation running — 2026-10-25

The user launched `continue_v1` on 8xH100 at 2026-10-25T23:20:16Z and
requested monitoring through terminal completion. GPU preflight passed.
The log explicitly confirms LoRA step=3000, Adam step=3000, exact optimizer
state restoration, first sample position=96000 and LR=1e-5. Step 3,001 and
subsequent steps have finite loss/gradients and real A/B updates. Do not
restart, duplicate, interrupt, retune or advance to MoE. Continue unchanged
through the frozen 3,500/4,000 validations; scientific success still requires
development PSNR >=34.2 and LPIPS <=.08, not merely normal process exit.
Early steady steps are ~1.5 s with ~24.86 GiB peak allocated memory per rank.
The earlier "not launched" wording below is now historical.
Progress: `reports/foundir_rain_precision_resume_progress_2026-10-25.md`.

## ACTIVE: true Rain continuation 3,000 → 4,000 — 2026-10-25

The user explicitly authorized exactly 1,000 more optimizer steps from the
completed full-data `formal_v1` step 3,000. This supersedes the older
"no continuation authorized" status, but not the frozen Rain-before-MoE gate.
Use `configs/foundir_rain_flux2_precision_resume_v1.json` and the new
`smoke/train_foundir_rain_flux2_precision_resume_v1_ddp.py`; do not edit or
restart the completed parent trainer/config, and never start with fresh Adam.

Restore the hash-bound step-3,000 FP32 LoRA AND all 100 Adam moment states /
step counters. Preserve LR 1e-5, the eight loss terms/coefficients, batch32,
512 crops, source-balanced schedule, augmentation, teacher and evaluation.
Resume absolute sample positions 96,000 through 127,999: exactly 32,000 new
presentations, cumulative 128,000. Validate/save only at global steps 3,500
and 4,000 on the same 100 development records and 32 training-probe records.
No protected data, new architecture, pixel refiner, EBT or energy experiment.
Do not automatically extend beyond 4,000 or advance to MoE.

The resume code has passed real LoRA/Adam/source/data preflight, CPU exact
state restoration and interrupted-versus-uninterrupted update tests. Old
checkpoints do not contain global RNG states: data sampling/crop/D4 are
stateless by absolute position, AE encoding uses its mean, anchor noise and
dropout are zero. No claim of bitwise uninterrupted GPU equivalence.
The original files/artifacts remain unchanged. New output namespace:
`checkpoints/foundir_rain_flux2_precision_resume_v1/continue_v1/`.
Log: `logs/foundir_rain_flux2_precision_resume_v1_continue_v1.log`.

Launch on ONE homogeneous 8xH100/H200 node:
`bash smoke/start_foundir_rain_flux2_precision_resume_v1_tmux.sh continue_v1`.
Codex has no GPU; no GPU continuation has been launched here. Session creation
alone is not success: verify `RAIN_RESUME_STATE_RESTORED` and step 3,001.
Estimate 40–50 normal wall minutes / 5.3–6.7 allocated GPU-hours on 8xH100;
replace with observed throughput after launch. Current baseline remains
34.00838778684664/.07343229320365935, PSNR gap .19161221315336263 dB.
Do not promise passing 34.2/.08. Readiness:
`reports/foundir_rain_precision_resume_readiness_2026-10-25.md`.

## TERMINAL: full-data Rain formal completed — 2026-10-25

This supersedes every earlier LIVE/unstarted status for this run.
`formal_v1` completed 3,000/3,000 steps on 8xH100 at
2026-10-25T22:55:59Z, launcher status 0. Final/best development is
34.00838778684664/.07343229320365935; training probe is
33.984372473834554/.060009161999914795. LPIPS passes .08; PSNR remains
.19161221315336263 dB below 34.2. Thus engineering completion succeeds but
the joint Rain gate is NOT passed. Do not advance image-only MoE/VLM routing.
Relative to matched step0, gains are +1.9236860265013505 dB and
-.015389716671779755 LPIPS. The final 1,000 steps add .384159465303874 dB;
do not promise an unrun extension will pass. No automatic continuation.
Actual exposure: 96,000 presentations / 36,156 unique records / 5,780 sources.
Total launcher cost: 1h36m33s, 1.6091666666666666 wall hours and
12.873333333333333 allocated H100 GPU-hours. All final artifacts, F32 tensor
headers, source/config identities and aggregate metric means were verified.
Final LoRA: `checkpoints/foundir_rain_flux2_precision_full_v1/formal_v1/step_003000.safetensors`,
SHA-256 `336d412299cd7cd9ff919d8983b2b4c435105161496510c0cec8748785c15904`.
No next run is active. Any extension/recipe change requires a fresh user
decision and frozen protocol; preserve all prior artifacts.
Complete report: `reports/foundir_rain_precision_full_formal_progress_2026-10-25.md`.

## LIVE: full-data Rain formal started — 2026-10-25

`formal_v1` started at 2026-10-25T21:19:26Z on 8xH100 and logged finite
training steps after matching the qualification contract. User requests
monitoring until terminal completion. Do not restart, duplicate, retune,
interrupt, or advance to MoE. Frozen schedule: 3,000 steps, evaluations at
0/250/500/1000/2000/3000; keep engineering success separate from 34.2/.08.
Log: `logs/foundir_rain_flux2_precision_full_v1_formal_v1.log`.
Output: `checkpoints/foundir_rain_flux2_precision_full_v1/formal_v1/`.
Launch/config source snapshot is backed up at commit 761a1f5 and the
qualification evidence at 2ad559c. The previous unstarted status is historical.

## Latest: full-data Rain qualification PASSED — 2026-10-25

The 25-step full-data qualification completed on eight H100s at
2026-10-25T21:16:59Z, launcher status 0. Actual exposure: 800 records from
800 unique training sources. Development PSNR/LPIPS improved from
32.08470176034529/.0888220098754391 to
32.64423398615503/.08657967696897685; the 34.2/.08 gate remains false.
Train-probe result: 32.66665987243047/.0708855870179832. This is engineering
qualification, not evidence that the final scientific target will pass.
Finite gradients, actual A/B updates, saved F32 LoRA/optimizer, matched
metrics and unchanged source/config hashes were verified. No protected data.
Total launcher elapsed 173 seconds (.04805555555555555 wall hours,
.3844444444444444 allocated H100 GPU-hours). Median step 1.5106424172408879
seconds, reported peak allocated memory 24.854560375213623 GiB.
Update the formal estimate to 1.5-2.0 wall hours / 12-16 allocated GPU-hours,
subject to sustained full-data IO. Formal has NOT been launched.
The already-authorized independent 3,000-step formal can now be started on
the user's GPU node with
`bash smoke/start_foundir_rain_flux2_precision_full_v1_tmux.sh formal formal_v1`.
Initialize from original step-2,000 LoRA/fresh Adam, never qualification.
Do not change source/config after qualification or auto-advance to MoE.
Full evidence: `reports/foundir_rain_precision_full_qualification_2026-10-25.md`.

## ACTIVE: full-data FP32 Rain FLUX — 2026-10-25

The user explicitly authorized full-data matched-resolution training after
the 300-step fitting diagnostic. This supersedes the earlier instruction to
wait for a new full-data decision. Do not resume the 32-image fit run.
Keep Rain development gate >=34.2 PSNR / <=.08 LPIPS before image-only MoE
and then VLM-token routing. No pixel refiner, EBT or action-energy branch.

Use `configs/foundir_rain_flux2_precision_full_v1.json` and its new trainer.
Independent initialization remains the original hash-bound direct-endpoint
step-2,000 LoRA with fresh Adam; neither diagnostic nor qualification weights
may seed formal. Preserve FP32 LoRA/Adam, BF16 compute, shared FP32 endpoint,
teacher training-only, rank32 and all eight original loss terms/coefficients.
The pool is all 38,334 Rain records / 5,780 sources. Cycle uniformly through
source groups then each group's records, random aligned 512 crops and D4
augmentation. Reflect-pad only the 35 training records with short side <512;
never resize or change development images. No decoded training-image cache.
Two spawned workers per rank prefetch two batches each; verify image SHA-256
before first decode and again if file stat changes. Metadata-path checks run
once on rank0, not redundantly on all ranks.

First run a fresh 25-step full-data engineering qualification on one homogeneous
8xH100/H200 node. Then independently train 3,000 steps, batch32, evaluating
at 0/250/500/1000/2000/3000. Formal requires a passing matching qualification
receipt and cannot auto-start. Keep the same 100-record development gate and
32 training probe; probe scores are not SOW results. No protected data.
Qualification estimate .10-.50 wall hours / .8-4 allocated GPU-hours.
Provisional formal estimate 1.5-3.0 wall hours / 12-24 allocated GPU-hours;
replace with full-data qualification throughput (cached 32-image speed is not
a guarantee). Report actual unique records/sources and sample presentations.

Eight CPU tests and full artifact/data preflight passed. No GPU full-data
qualification has been launched from this GPU-less Codex environment.
Start on the user's node:
`bash smoke/start_foundir_rain_flux2_precision_full_v1_tmux.sh`.
Readiness: `reports/foundir_rain_precision_full_readiness_2026-10-25.md`.

## Latest terminal update: Rain precision fit complete — 2026-10-25

The independent `fit_v1` completed 300/300 steps on 8xH100 at
2026-10-25T20:59:43Z, launcher status 0. Training-fit reaches
35.09303906168583/.04928288975497708 (numeric target passes); development
reaches 32.44553214852097/.08547242213506251 (Rain SOW gate fails).
Step-100 development was 32.54685884616055/.08776275758864358.
Do not describe training-fit success as Rain gate success or evidence that
FP32 alone caused the gain. This proves fitting ability on these 32 fixed
training inputs, not unseen-source generalization. Of 24 development inputs
with input PSNR >=34, 20 still lose PSNR after restoration at step 300.
The run is complete: do not resume/extend it on the same 32 images or
auto-launch full-data training. Next proposed work should address
generalization with a frozen full-data, matched-resolution protocol and
measured budget, requiring a new user decision before GPU launch.
Preserve the Rain gate -> image-only MoE -> VLM router order. No pixel refiner.
Total launcher time: 546 seconds (9m06s), .15166666666666667 wall hours,
1.2133333333333334 allocated H100 GPU-hours. This is faster than the prior
provisional estimate. Full result and limits:
`reports/foundir_rain_precision_fit_terminal_2026-10-25.md`.

## Latest Rain qualification update — 2026-10-25

The 25-step FP32 precision qualification completed successfully at
2026-10-25T20:39:00Z on 8xH100, status 0. Development improved from
32.08470176034529/.0888220098754391 to
32.61941190970921/.08720199926523492, still failing the 34.2/.08 gate.
The user authorized the independent 300-step fit diagnostic, not full-data
training. Initialize from the original step-2,000 LoRA with fresh optimizer,
never the qualification checkpoint. Command on the GPU node:
`bash smoke/start_foundir_rain_flux2_precision_fit_v1_tmux.sh fit fit_v1`.
Estimate 15-20 wall minutes / 2.0-2.67 allocated GPU-hours. Fit has no launch
log at this update; this Codex environment has no GPU. Keep Rain gate first,
image-only MoE second, VLM router third; no pixel refiner.
See `reports/foundir_rain_precision_qualification_2026-10-25.md` for evidence.

## Active workspace and runtime

- The sole active project root is `/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang` (migration decision 2026-10-30).
- Use the validated project Python environment; its last known location and the required new-node verification are stated in the CURRENT AUTHORITY section above. Do not silently substitute system Python.
- Preserve all dirty and untracked project files. Do not reset, clean, or overwrite unrelated work.
- Keep prior workspaces frozen; no execution or automatic synchronization back to them. Historical paths below are provenance, not active workspace choices.
- Never use official, sealed, parent-test, or internal-test pixels for training, hyperparameter selection, checkpoint selection, or early stopping.

## ACTIVE: Rain FLUX precision repair and training-fit diagnosis — 2026-10-25

The user rejected the proposed small pixel refiner and authorized the
precision/fit diagnosis. This supersedes the refiner suggestion below.
Keep the sequence Rain DiT >=34.2 PSNR / <=.08 LPIPS, then working image-only
MoE, then VLM tokens. Do not train the observation-anchored pixel head.

The prior run stored trainable LoRA weights and both Adam moments in BF16
at learning rate 1e-5. At step 3,000, 64.2164% of LoRA A entries were exactly
unchanged from its parent. This is evidence to investigate update rounding,
not proof that it explains the entire failure. Training kept the encoded
observation FP32 for subtraction while inference cast it to BF16 first.

The new isolated implementation is configured by
`configs/foundir_rain_flux2_precision_fit_v1.json`: FP32 LoRA parameters and
Adam states, BF16 CUDA compute, and one shared FP32 endpoint subtraction in
training/evaluation. Start independently from the hash-bound best Rain
direct-endpoint step-2,000 LoRA with a fresh optimizer. Retain the eight
parent loss terms and weights. Use 32 deterministic training source groups
with fixed 512 crops; evaluate the exact same training inputs and the
unchanged 100-source development gate at 512. The initial evaluation also
reports the legacy subtraction path on the same checkpoint.

Run a 25-step qualification first on one homogeneous 8xH100/H200 node. Its
checkpoint cannot seed the independent 300-step fit diagnostic. Fit requires
a passing qualification with matching config, source and sample hashes.
Qualification/fit milestones are [0,25]/[0,100,300]. No automatic full-data
training follows. Passing the 32-image numeric target is evidence of fitting
ability only; development PSNR/LPIPS remain separate. This is not a matched
long-training BF16-versus-FP32 ablation and cannot attribute all gains to
precision alone. Both launchers preserve existing run directories/logs.
Read `reports/foundir_rain_flux2_precision_fit_v1_readiness_2026-10-25.md`.

## ACTIVE: Rain DiT gate, then image-only MoE, then VLM router — 2026-10-25

This user decision supersedes the older MoE-first scheduling text below. The
order is strict: (1) repair the Rain FLUX/DiT expert until it passes the frozen
Rain development gate (`PSNR >=34.2`, `LPIPS <=0.08`); (2) build and validate
an ordinary working image-only restoration DiT-MoE using successful experts;
(3) only then add pooled VLM output tokens through a small projector to
adjust that same router. Blur DiT has already passed its SOW gate; do not
repeat Blur training. The failed 1,000-step image-only MoE is not a usable
baseline, and its VLM-assisted arm must not be launched.

The full-data direct-endpoint Rain FLUX formal run completed on 2026-10-21 at
3,000/3,000 steps (8xH100, `6.470578311735836` wall hours,
`51.76462649388669` allocated GPU-hours). Its frozen 100-record development
metrics were `31.900863136522702/.08921627756208181`, failing both gates.
The best predeclared checkpoint was step 2,000 at
`32.085315339143946/.08882607079111039`; 3,000 regressed. Do not repeat or
extend the same direct-endpoint recipe. On the 3,000-step result, all 24
records with input PSNR >=34 dB lost PSNR after restoration; 14 of the other
76 records also lost PSNR. A no-training, fixed 0.75 blend of input and the
step-2,000 output reaches only about 32.545 dB PSNR on saved development
triptychs. Even a non-deployable GT-only best blend per image reaches only
33.444 dB. These are quantized-export exploratory calculations, not official
model results. Simple strength control cannot repair the gap. Diagnose the
Rain action/representation and verify a distinct fix before another GPU run.
See `reports/foundir_rain_dit_gate_priority_and_failure_audit_2026-10-25.md`.
An observation-anchored correction head now exists and passes three CPU unit
tests, but it has no Rain trainer or GPU result. Do not call the gate passed.

## ACTIVE: advisor-approved minimal VLM-token-to-MoE-router pilot — 2026-10-25

This section supersedes every older Rain, EBT, energy, specialist, and
explicit-VLM scheduling priority below. The user reported that the advisor
approved the minimal architecture and explicitly authorized execution. Do not
restart an older branch unless the user explicitly returns to it.

Implement and test only the minimal advisor proposal:

1. start from one ordinary restoration DiT-MoE with one expert bank;
2. retain its image-feature router and restoration experts;
3. pool frozen VLM output tokens/features, pass them through one small MLP,
   and use the result to adjust the original router's expert logits;
4. select/weight the same MoE experts and combine their outputs normally.

There must not be a second MoE, a bank of LoRA experts, EBT/energy refinement,
atomic-token composition, counterfactual routing, hard diagnosis priors, or
block/timestep scheduling in this first experiment. The matched comparison is
image-only routing versus the identical MoE with the VLM score branch enabled.
Use the same backbone, expert bank, data, optimization, and inference budget.

CPU/static implementation and a training-only data preflight are complete.
The frozen 25-step eight-GPU qualification is defined by
`configs/foundir_vlm_token_moe_minimal_qualification_v1.json`. Run the
`image_only` and `vlm_assisted` arms separately; neither qualification may seed
formal training or auto-launch later work. Keep official, sealed, parent-test,
and internal-test pixels out of development. Report wall-clock hours and
allocated GPU-hours for every GPU run.

The `image_only` qualification completed successfully on 8xH100 at
2026-10-25T15:38:24Z. It ran 25/25 steps in `.0563902` wall-clock hours and
`.451122` allocated GPU-hours, with finite training, no OOM/NCCL/runtime
failure, and `12,653,083,136` bytes peak allocated memory per rank. The VLM
logit adjustment was exactly zero, expert weights summed to one within
`1.1921e-7`, and the checkpoint hash was verified. This is engineering
qualification only, not a restoration-quality result, and its checkpoint may
not seed formal training. The next active action is the fresh matched
`vlm_assisted` 25-step qualification; do not launch formal training yet.

The independent `vlm_assisted` qualification also completed successfully on
8xH100 at 2026-10-25T15:53:30Z. It ran 25/25 steps in `.0460070` wall-clock
hours and `.368056` allocated GPU-hours, with no OOM/NCCL/runtime failure and
`12,669,009,408` bytes peak allocated memory per rank. Its VLM branch was
active: maximum logit adjustment was `.75` and the output-head L1 update was
`.78411865234375`. Expert weights remained normalized within `1.1921e-7`.
Both engineering arms have therefore passed. Their last-five training losses
(`1.1534720` image-only versus `1.1534493` VLM-assisted) are not restoration
metrics and must not be presented as scientific improvement. Neither
qualification checkpoint may seed formal work. The next step is to freeze a
matched scientific train/evaluation protocol before any longer GPU run.

That matched protocol is now frozen and CPU-qualified in
`configs/foundir_vlm_token_moe_matched_formal_v1.json`. Each arm starts
independently from the same pretrained FLUX.2 weights, trains for 1,000
optimizer steps (32,000 fit presentations, about 14.2 fit-set exposures), and
uses the same data/order/objective/compute. Calibration is evaluated at steps
500 and 1,000; the source-disjoint 450-record holdout is read exactly once at
the fixed final step. Metrics are PSNR, AlexNet LPIPS, per-degradation results,
and routing traces. Four protocol unit tests and both arm-specific CPU/data
preflights pass. No protected data was accessed. Run the arms separately with
`smoke/run_foundir_vlm_token_moe_matched_formal_v1_8gpu.sh`; start with
`image_only`. The launcher is resume-safe within the same arm but forbids all
qualification checkpoints and never launches the other arm automatically.
Estimated cost per arm is `1.3-2.5` wall hours and `10.4-20.0` allocated
GPU-hours. Do not claim scientific success until both final holdouts have been
paired by `smoke/compare_foundir_vlm_token_moe_matched_formal_v1.py`.

The `image_only` formal arm completed normally at 2026-10-25T16:53:00Z:
1,000/1,000 steps, `.643642` wall hours, `5.14914` allocated H100 GPU-hours,
finite training, and no OOM/NCCL/runtime failure. It failed the scientific
prerequisite that the ordinary MoE be usable. On the fixed 450-record holdout
it reached only `15.634318159133873` PSNR and `.34884399509264363` LPIPS,
versus the unprocessed input at `23.71294072670594/.3023991875624698`.
Routing collapsed almost completely onto expert 1: mean weights are about
`5.8873e-7/0.99999941`, with entropy `8.6261e-6`. Step 500 showed the same
failure and step 1,000 did not repair it. This is a valid terminal result for
the current image-only construction, not a compute/runtime failure.

Do not call this a working restoration MoE or use it as paper evidence. Do not
automatically launch `vlm_assisted`: an additive VLM score cannot by itself
repair a restoration backbone/expert bank whose image-only output is already
catastrophically below the input. The next design must begin from an actually
working restoration model/MoE and explicitly prevent router collapse; obtain
the user's decision before replacing the frozen matched protocol.

## MANDATORY: regular focused GitHub progress tracking

GitHub is the durable project history. Do not allow material work to remain
only in the dirty local workspace for long periods. The following is a hard
operating rule for all later agents, not an optional cleanup suggestion.

Create a focused commit and push it to `origin/main` after each coherent,
material milestone, including:

- a new or materially changed model, training script, evaluation script,
  configuration, launcher, or reproducibility contract;
- a code failure that required a substantive fix once the corrected code has
  passed the available static/CPU/GPU qualification;
- an advisor/user research-priority decision that changes the active plan;
- a completed qualification, pass-level validation, selected checkpoint, or
  terminal experiment result after its report/handoff has been updated;
- before leaving a long-running experiment unattended, provided its launch
  code/config and current handoff form a coherent reproducible snapshot.

For an active run, do not commit every log line or optimizer step. Commit the
launch-ready code before or immediately after launch, then commit again at
meaningful validated milestones and terminal completion. A running job is not
an excuse to leave its source/config untracked.

Every push must follow this safety procedure:

1. Run `git fetch origin` and inspect local/remote divergence.
2. Preserve the dirty workspace. Never use `git reset --hard`, `git clean`, or
   broad checkout commands to prepare a push.
3. Stage an explicit allowlist of relevant source/config/report files. Never
   use `git add -A`, `git add .`, or a directory-wide add in this repository.
4. Exclude checkpoints, model weights, raw/generated images, datasets, caches,
   telemetry, live logs, temporary files, and large generated manifests. Keep
   data identity through small metadata summaries and SHA-256 bindings.
5. Scan the staged files for credentials, access tokens, private keys, and
   unintended machine-specific secrets. Never commit the previously exposed
   GitHub token or any replacement credential.
6. Run syntax/JSON checks and `git diff --cached --check`; inspect the staged
   file list and size before committing.
7. If the main workspace is heavily dirty or an untracked file conflicts with
   an incoming remote path, create an isolated temporary Git worktree from
   `origin/main`, copy only the explicit allowlist, commit there, and push from
   there. Do not disturb the live training workspace.
8. Push the focused commit to `origin/main`, verify the remote commit hash, and
   report the commit hash and included scope to the user.

If authentication, networking, a remote conflict, or a required secret scan
blocks the push, report that immediately and preserve the ready commit. Do not
silently defer Git tracking across later milestones. Do not claim the work is
backed up until the remote hash has been verified.

## TERMINAL: residual-basis formal run complete — 2026-10-17

The user explicitly authorized addressing the oracle-proven lack of corrective
information in the V2 direct residual. The next experiment is not another
energy/gate run. Freeze the existing Noise proposal and train exactly one
restoration-native residual-basis model on the same 54,585-record training
split. It receives the degraded image, proposal, and their difference and
produces two independent directions: fidelity trained explicitly with MSE and
perception trained with stronger LPIPS while retaining MSE/local no-harm.

This is a causal prerequisite for later energy work. The fresh 25-step
eight-GPU engineering qualification completed successfully on `8 x H100
80GB` with status `0`, finite training, checkpoint and matched 256-record
development validation. It consumed about `.0217` wall-clock hours and `.173`
allocated GPU-hours. At step 25 the fidelity/perceptual heads improved PSNR by
about `+.0116/+.0114 dB` but worsened LPIPS by about `.00020`; this is only an
engineering check, not scientific evidence or a reason to alter the frozen
recipe.

The next action is a fresh formal run ID for the predeclared five-pass/8,530-
step schedule, validating after every effective pass. Never initialize it from
`qualification_v1`. Do not restart V2, train an energy/gate, use protected
data, or claim novelty success until the new residual basis itself
demonstrates adequate headroom on the fixed 256-record development protocol.
Formal runtime is conservatively estimated at `3.5–6.0` wall-clock hours on
one eight-GPU node (`28–48` allocated GPU-hours); update this estimate from
the measured pass-1 runtime.
Implementation contract:
`configs/foundir_proposal_residual_basis_noise_v1.json` and
`reports/foundir_proposal_residual_basis_noise_v1_readiness_2026-10-17.md`.

`formal1` is now live on `8 x H100 80GB` and passed its first effective pass
at step `1,706`. Do not restart, duplicate, retune, or interrupt it. The
fidelity head reached `43.16839203238487/.0400125195938017`, simultaneously
improving over the frozen proposal by `+.044231854379177094 dB` and
`-.010349548166232125` LPIPS. This already exceeds the old direct residual's
GT-only oracle in both metrics, so the restoration-native basis has resolved
the earlier lack-of-information bottleneck at least for LPIPS/headroom. It has
not passed the full `+.30 dB/-.005 LPIPS` gate because PSNR gain is still only
`.0442 dB`. Continue the frozen five-pass run and judge the next validations
without changing the recipe. Full pass-1 record:
`reports/foundir_proposal_residual_basis_noise_v1_pass1_2026-10-17.md`.

Pass 2 at step `3,412` improved further. Fidelity is now
`43.22963780909777/.03339703518213355`, or `+.10547763109207153 dB` and
`-.016965032577900274` LPIPS versus proposal. This lands inside the
predeclared `+.09–.12 dB` pass-2 hopeful trend range. It has not passed the
`+.30 dB` PSNR gate, but the evidence now supports completing the unchanged
five-pass schedule. Do not extrapolate linearly or retune from this interim
result. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass2_2026-10-17.md`.

Pass 3 at step `5,118` strengthened the result materially. Fidelity reached
`43.354813657701015/.030396450317311974`, equal to
`+.23065347969532013 dB` and `-.01996561744272185` LPIPS versus proposal. It
now needs only `.06934652030467986 dB` more PSNR to pass the joint target, and
the pass-2 to pass-3 PSNR increment was `+.1251758486032486 dB`. Continue the
unchanged run through passes 4 and 5; do not stop early or alter the schedule.
Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass3_2026-10-17.md`.

Pass 4 at step `6,824` did not cross the gate. Fidelity is
`43.365845650434494/.02899720575715037`, or `+.24168547242879868 dB` and
`-.021364862002883456` LPIPS. The PSNR gain increased only
`+.011031992733478546 dB` from pass 3 and remains `.05831452757120131 dB`
short, showing clear late-stage saturation after the 256-crop transition and
with the learning rate already near `6e-6`. Complete the frozen fifth pass,
but do not claim likely gate success or modify the live recipe. Full record:
`reports/foundir_proposal_residual_basis_noise_v1_pass4_2026-10-17.md`.

`formal1` completed successfully at step `8,530` with launcher status `0`.
Final fidelity is `43.39715355634689/.028955043449059303`, equal to
`+.27299337834119797 dB` and `-.02140702431097452` LPIPS versus proposal. It
therefore misses the PSNR gate by only `.02700662165880202 dB` while passing
the LPIPS gate widely. The joint gate is terminally **not passed** for this
run. Do not resume or silently extend it. The next authorized reasoning step
is a no-training GT-only amplitude/local feasibility audit of the frozen
step-8,530 fidelity residual; no energy/gate or new training may start before
that evidence is reviewed. Terminal report:
`reports/foundir_proposal_residual_basis_noise_v1_terminal_2026-10-17.md`.

## TERMINAL: final fidelity-residual oracle passes local headroom — 2026-10-17

The user authorized the next feasibility audit. Run exactly one no-training,
GT-only diagnostic on the fixed 256-record source-disjoint development subset
using the hash-bound step-8,530 checkpoint. It evaluates global fidelity-
residual amplitudes from `0` to `2` in `.025` increments plus analytic local
8/16/32/64-patch amplitudes, and reports fixed-candidate and per-image Pareto
upper bounds. It must reproduce both proposal and alpha-1 fidelity metrics
within the frozen tolerance before writing a result. This is not deployable
evidence and may not access protected data. Code and CPU/unit qualification
are complete; the Codex container exposes no GPU, so the next action is the
single-node eight-GPU launcher:
`smoke/run_foundir_proposal_residual_basis_noise_v1_oracle_8gpu.sh`.
The audit completed successfully in `72` seconds (`.020` wall-clock hours,
`.160` allocated GPU-hours), with exact proposal and fidelity metric
reproduction and no protected-data access. No global amplitude crosses the
joint gate; alpha `1.000` remains the maximum-PSNR global setting. In contrast,
the GT-derived local 8x8 amplitude upper bound reaches
`43.65973751991987/.03243445468615391`, or `+.5355773419141769 dB` and
`-.01792761307387991` LPIPS, and all tested local patch families cross the
gate. Therefore the frozen residual direction contains ample local corrective
information; the remaining problem is learning a deployable spatial amplitude
map without GT. Do not report any patch-oracle metric as model performance.
Result SHA-256:
`ce6ab400c5003a630f3e05fed60a5e1fe9e6d6514af86cddc1fed6b9e65e2f66`.
Full report:
`reports/foundir_proposal_residual_basis_noise_v1_oracle_terminal_2026-10-17.md`.

## LIVE: deployable 8x8 spatial amplitude predictor ready — 2026-10-17

The user authorized the minimal causal follow-up to the local oracle. Freeze
both the proposal and step-8,530 residual basis and train only the `223,393`-
parameter spatial amplitude predictor. It receives observation, proposal,
fidelity and fidelity-minus-proposal, outputs one `[0,2]` amplitude per 8x8
patch, and starts at exact alpha `1` so step zero reproduces fidelity. Training
GT provides analytic patch-amplitude supervision; inference and validation
must never use GT to select amplitudes. The loss contains restored-image MSE,
residual-energy-weighted teacher regression, weak LPIPS and local no-harm.

JSON, Python, Bash, focused unit tests, strict frozen-checkpoint loading and
diff checks pass. The next action is a fresh 25-step engineering qualification
on one homogeneous 8xH100/H200 node. It must not seed a formal run. Do not
train energy, use protected data, or report the GT oracle as model quality.
Contract:
`configs/foundir_spatial_residual_amplitude_noise_v1.json` and
`reports/foundir_spatial_residual_amplitude_noise_v1_readiness_2026-10-17.md`.

The fresh `qualification_v1` run completed successfully on `8 x H100 80GB`
with status `0`, finite training, checkpoint/validation output and no protected
data access. At step 25, predicted amplitude is still near its exact-identity
initialization (`mean 1.0000218451`, standard deviation `.0004146491`), so
spatial output reproduces fidelity within `+.0000015423 dB` and
`-.0000000773` LPIPS. This is an engineering pass, not a scientific result.
The next action is a fresh-ID, from-zero five-pass/8,530-step formal run; never
seed it from qualification. Estimated H100 cost is `2.5–3.5` wall-clock hours
and `20–28` allocated GPU-hours, to be updated from pass-1 throughput.

`formal1` is live and completed pass 1 at step `1,706`. Spatial output is
`43.395900413393974/.029359601011492487`, which is worse than frozen fidelity
by `.00125314295291574 dB` PSNR and `.000404557562433184` LPIPS. The amplitude
map has moved materially (`mean 1.0358318537`, standard deviation `.1107148176`),
so this is an early generalization miss rather than a no-learning failure.
Continue the unchanged five-pass schedule; do not retune or stop from pass 1.
Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass1_2026-10-17.md`.

Pass 2 at step `3,412` is mixed but slightly more encouraging. Spatial output
is `43.3995122089982/.033109975053776`, which improves PSNR over frozen
fidelity by `+.00235865265130997 dB` but worsens LPIPS by
`.004154931604716697`. Relative to proposal it remains
`+.27535203099250793 dB/-.017252092706257827 LPIPS`, so LPIPS still passes the
joint target while PSNR is `.02464796900749206 dB` short. Continue unchanged;
do not claim oracle transfer or retune from this interim result. Full record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass2_2026-10-17.md`.

Pass 3 at step `5,118` remains a small positive PSNR result but is flattening.
Spatial is `43.400297187268734/.0317960577121994`, or
`+.0031436309218444425 dB` PSNR and `.0028410142631400966` worse LPIPS versus
frozen fidelity. Relative to proposal it is
`+.27613700926303864 dB/-.018566010047834425 LPIPS`, leaving
`.023862990736961353 dB` to the PSNR gate. Continue passes 4 and 5 unchanged;
current evidence does not support likely gate success or early retuning. Full
record:
`reports/foundir_spatial_residual_amplitude_noise_v1_pass3_2026-10-17.md`.

## STOP: V2 and direct-residual oracle are terminal — 2026-10-17

The user stopped `noise_energy_v2_formal1` externally after its last logged
step `6,655`; do not resume it. The last complete checkpoint with a matched
validation is step `5,118`. Its GT-only direct-residual feasibility diagnostic
finished on the same fixed 256-record source-disjoint development set. Even
the target-selecting upper bound failed the `+.30 dB/-.005 LPIPS` joint gate.
The maximum-PSNR oracle point was only `+.042452 dB/-.003946 LPIPS`; when the
LPIPS requirement was enforced, the best PSNR gain was only `+.037468 dB`.
Therefore the learned direct residual does not contain enough corrective
information, and a learned gate over these candidates cannot meet the target.
Do not resume V2, train a gate over this frozen residual, or claim the oracle
as deployable evidence. No V3 or fallback is authorized until the user makes
the next research decision. Full result:
`reports/foundir_proposal_energy_v2_direct_residual_oracle_terminal_2026-10-17.md`.

## STOP: max-complete PPT energy V2 only — 2026-10-15

This section supersedes the V1/fallback sequence below wherever they differ.
The proposal-relative energy V1 must finish its already-running frozen
five-pass endpoint and then stop; do not extend it to ten passes. Its observed
failure mode is higher PSNR with worse LPIPS, poor gradient-direction
alignment, and multi-step drift. Preserve V1 as causal evidence.

Before switching to the DiT-MoE fallback, implement and test exactly one
max-complete V2 of the advisor-approved PPT energy idea. V2 must contain all
and only these six missing parts:

1. LPIPS/perceptual-aware training in addition to pixel/edge fidelity;
2. actual frozen-model outputs for wrong-expert, under-restored,
   over-restored, and detail-damaged contrastive candidates;
3. directly unrolled and trained 1/2/3-step energy trajectories;
4. per-step no-harm constraints for both pixel fidelity and LPIPS, with the
   learned energy also allowed to select the unchanged proposal;
5. per-image VLM degradation severity from the existing frozen Qwen manifest,
   not a constant Noise one-hot vector;
6. spatially preserved/local energy supervision rather than reducing all
   quality evidence to one early global scalar.

Do not add another backbone, dataset, degradation task, specialist, routing
mechanism, MoE, compositional branch, or unrelated loss/idea to V2. Keep the
frozen Noise proposal, the 54,585-record source-disjoint training split, the
fixed 256-record development validation, 1/2/3-step reporting, protected-data
prohibitions, and the existing minimum `+.30 dB` and `-.005 LPIPS` gate.
Engineering qualification and the adequately scaled scientific run are
allowed only to validate this exact V2; they are not extra research branches.
Do not claim the PPT approach works until both metrics improve on the frozen
development protocol.

Current status: V1 completed `8,530` steps and failed the gate because its best
energy arm improved PSNR only `+.10285484790802033 dB` while worsening LPIPS
by `+.006770727247385367`. V1 is terminal and must not resume. The exact-six-
change V2 code passed its fresh 25-step qualification on `8 x H100 80GB` with
status `0`, finite saved weights, optimizer state for every trainable entry,
successful validation/checkpointing, and no protected-data access. The next
action is a new-run-ID, from-zero `8,530`-step formal V2 run. The 25-step
checkpoint must never seed the formal run.

## DEFERRED: advisor-approved two-stage novelty plan — 2026-10-15

This section supersedes the 2026-10-14 energy-TTS decision below wherever the
two differ. The phrase "PPT direction" means the restoration-energy method
already presented in the advisor slides; it does not mean making more slides.

The advisor's decision is:

1. **Try the PPT restoration-energy idea first.** Test whether a lightweight
   action-contrastive / proposal-relative energy can improve the already
   frozen FLUX.2/Restormer output. The energy is conditioned on the degraded
   input and explicit VLM degradation information. It should learn to
   distinguish a correct restoration from under-restoration, over-restoration,
   a wrong expert action, or newly damaged details, and then use that signal
   for a small number of candidate-image correction steps. This is currently a
   hypothesis, not an established working result.
2. **Use a staged but adequately scaled causal test.** Start from Noise on
   source-disjoint development data and compare against both the frozen
   proposal and a matched feed-forward refiner. Preserve the numerical gates
   and data restrictions in the 2026-10-14 section below. The first 500 steps
   are an engineering qualification only: they test correctness, memory,
   throughput, checkpoint/resume, and whether gradients and validation are
   finite. They are not enough to declare the research idea dead. After that
   qualification, train on the complete 54,585-record Noise training split.
   The first scientific decision point must include at least one complete
   effective pass over those records; if the validation trend is still
   improving, continue through a predeclared multi-pass schedule rather than
   stopping because the model is initially undertrained. Derive the exact
   optimizer-step schedule and GPU-hours from measured throughput before the
   formal launch. Do not claim success merely because learned energy decreases;
   PSNR/LPIPS and candidate ranking/refinement must improve.
3. **Only if the PPT energy idea does not work, switch to the advisor's DiT-MoE
   fallback.** The fallback is a FLUX/DiT restoration model whose internal
   experts are controlled using explicit VLM predictions of degradation type,
   severity, and composition. Do not begin this fallback in parallel with the
   first energy test unless the user explicitly changes the sequence.

Novelty boundaries for the fallback must be stated honestly:

- FoundIR-v2 already uses MoE, although its backbone is SDXL rather than DiT.
- MiM-DiT already uses MoE inside a diffusion transformer for All-in-One image
  restoration. Therefore **"MoE + DiT" is not a new contribution**.
- The remaining hypothesis is narrower: MiM-DiT directly looks at degraded
  image features to choose experts, while the proposed fallback first asks a
  VLM to state the degradation type, severity, and composition and then uses
  that explicit diagnosis to control DiT experts.
- "We added a VLM to the router" is not automatically enough novelty. It only
  becomes defensible if a matched comparison shows that explicit VLM guidance
  improves mixed/OOD degradation restoration or expert composition over an
  otherwise identical image-feature router.

Advisor-chat provenance, in plain language: the advisor said the PPT ideas are
reasonable but may not work and asked whether any already work. The advisor
then suggested MoE, initially noting FoundIR-v2 and its lack of this DiT
combination. After learning that MiM-DiT already covers the broad MoE+DiT idea
but does not explicitly output degradation type/severity through a VLM, the
advisor approved the sequence: **try the project's own PPT energy idea first;
if it fails, the VLM-guided DiT-MoE fallback is still acceptable because it is
not a complete overlap.** No method-success claim follows from that approval.

### Frozen scale and hardware plan for the first energy experiment

Only one homogeneous eight-GPU node may be used for this experiment. It may be
**8 x NVIDIA H200** or **8 x NVIDIA H100 80GB**, but never a mixed node and
never multiple nodes. Do not create or propose a cross-node launch. All
training, validation, checkpointing, and resume logic must work on either
supported single-node profile while preserving global batch 32.

- H200 profile: micro-batch 4/GPU, gradient accumulation 1.
- H100 profile: micro-batch 1/GPU, gradient accumulation 4.

These profiles have the same eight GPUs, global batch, optimizer-step schedule,
and training-record sampler. The smaller H100 micro-batch is only a memory
compatibility change; it does not authorize fewer data passes or different
method settings.

The Noise training split contains 54,585 paired records. Use global batch 32
unless the measured H200 memory/throughput qualification proves a different
batch is materially better. At global batch 32, one complete effective data
pass is `ceil(54,585 / 32) = 1,706` optimizer steps.

Use this frozen staged scale:

1. **Systems qualification:** 500 optimizer steps. This checks correctness,
   finite second-order gradients, memory, throughput, validation, checkpoint,
   and exact resume. It is not a scientific failure decision.
2. **First scientific checkpoint:** 1 complete effective pass, 1,706 total
   optimizer steps. Evaluate the frozen proposal, matched feed-forward refiner,
   and 1/2/3-step energy refinement on the fixed source-disjoint development
   validation set.
3. **Primary adequately scaled run:** 5 complete effective passes, 8,530 total
   optimizer steps. Save and validate at the end of every pass. This is the
   minimum scale for the main method judgment unless an earlier run has a
   numerical/code failure rather than merely weak metrics.
4. **Pre-authorized continuation:** extend the same run to at most 10 complete
   effective passes, 17,060 total optimizer steps, only when pass-4 to pass-5
   source-disjoint validation still shows a meaningful improving trend and no
   overfitting. Do not change the validation set, loss, candidate steps, or
   checkpoint rule based on protected reporting data.

Do not launch 100,000 steps by default: that would be approximately 58.6
passes over this split and is not currently justified. The 10-pass cap may be
changed only by a later explicit user decision backed by the pass-by-pass
development curve.

Exact wall-clock and allocated GPU-hours cannot be claimed before the H200
qualification measures seconds per optimizer step. After the 500-step run,
compute and report:

- `wall hours = remaining_steps * measured_seconds_per_step / 3600`;
- `allocated GPU-hours = wall hours * 8`.

Never transfer a timing estimate between H100 and H200, and never substitute
an old multi-node EBT benchmark. Measure seconds/step on the exact single-node
GPU family used for the run.

## Mandatory reporting style

All future user-facing reports must be easy to understand on the first read:

- Start with the practical answer or current outcome, then explain why.
- Use short, direct Mandarin sentences; retain English technical terms only
  where they prevent ambiguity.
- Translate each necessary term immediately into plain language. For example,
  say "模型直接看图选专家" before or alongside "latent-feature routing".
- Clearly separate **confirmed fact**, **our hypothesis**, and **next test**.
- For every comparison, name both sides, the dataset/protocol, and what each
  number means. Never use an unexplained phrase such as "coverage", "gate", or
  "better".
- Do not present an idea, advisor approval, lower energy, or a successful code
  run as evidence that image quality improved. State "not tested yet" when no
  metric result exists.
- Give time estimates in normal wall-clock hours and also report allocated
  GPU-hours for every proposed or completed GPU experiment.
- Prefer one concrete example over dense jargon. Add equations or internal
  implementation details only after the plain-language explanation.

## STOP: frozen FLUX/Restormer proposal + energy-TTS decision — 2026-10-14

This section supersedes the older Generalist-first and live-run sections below.
The from-scratch EBT Generalist is paused and must not be resumed, duplicated,
reconfigured, evaluated on the protected 660 endpoint, or used to select a
checkpoint. Its existing artifacts and failure logs remain historical evidence.

The only active restoration-model research objective is a lean test of
EBT-style test-time optimization applied to the already successful frozen
FLUX.2/Restormer routed system. FLUX.2/Restormer produces an immutable initial
proposal. A lightweight observation- and VLM-conditioned scalar energy
`E(y; x, c)` may then refine candidate `y` by a small number of gradients with
respect to the candidate image. Do not retrain or alter the frozen proposal
models during the first causal test.

Work source-disjoint development data only. Never use the official, sealed,
parent-test, internal-test, or frozen 660 reporting pixels for training,
hyperparameter selection, checkpoint selection, early stopping, or deciding
the number of test-time steps.

Proceed in this order:

1. Start with the Noise singleton, because the frozen routed result
   `40.133486/.096746` is behind Full FoundIR `42.655305/.067910` in both PSNR
   and LPIPS.
2. Compare the frozen proposal against a matched-parameter feed-forward bounded
   refiner and fixed 1/2/3-step candidate-gradient refinement. Require extra
   steps to improve restoration rather than merely lower an uncalibrated
   energy.
3. A minimum technical pass is at least `+.30 dB` and `-.005` LPIPS relative
   to the frozen proposal on protected source-disjoint development validation.
   Paper-worthy evidence is at least `+.50--.75 dB` and `-.010` LPIPS, a useful
   compute-quality curve, and superiority to the matched feed-forward refiner.
4. Only after the Noise gate passes may the same frozen mechanism be tested on
   the remaining singleton categories. A convincing singleton aggregate target
   is at least `29.2 dB` with LPIPS no worse than `.135`, with gains across
   multiple categories rather than one category dominating the mean.
5. Only after multi-category singleton evidence may the unchanged mechanism be
   evaluated on Double compositions. The frozen Double baseline is
   `24.321537/.192313`; an indicative paper-strength target is at least
   `24.8/.180`, subject to per-category no-harm evidence.
6. Do not reopen Triple, specialists, compositional EBT, or another from-scratch
   Generalist until this lean mechanism succeeds or fails under its frozen
   causal comparison.

The working novelty hypothesis is not simply "EBM", "more inference steps",
or "VLM + EBT". It is candidate-dependent, differentiable test-time refinement
of frozen heterogeneous AiOIR proposals under an observation- and structured
VLM-conditioned scalar energy, with the same energy used for iterative
restoration and stopping/verification. Treat "first" as unverified until the
targeted prior-art audit is complete. Current audit and exact claim boundary:
`reports/foundir_flux_restormer_energy_tts_novelty_audit_2026-10-14.md`.

The first GPU qualification, if the implementation and protocol pass CPU
review, is capped at 500 optimizer steps on one 8-GPU node. Its duration is
measured rather than constrained by the obsolete `.5` wall-hour estimate,
because second-order candidate gradients may take longer on H100 than H200.
**This is a systems check, not a method-failure gate.** After it passes, use
its measured throughput to freeze
an adequately scaled schedule over all 54,585 Noise training records, with at
least one complete effective data pass and further predeclared passes whenever
the source-disjoint validation trend is still improving. Report both
wall-clock hours and allocated GPU-hours. Stop only for a code/numerical
failure, a clearly saturated or worsening development curve after adequate
exposure, or completion of the frozen scale schedule—not merely because a
small run misses the final metric target.

## DEFERRED: Generalist-first research decision — 2026-10-11

This was the active objective on 2026-10-11 but is superseded by the
2026-10-14 decision above. Preserve it as historical protocol context only.

Do not design, implement, train, tune, benchmark, or launch any low-light,
weather, trajectory-energy, residual-energy, or other specialist. Preserve the
specialist prior-art/design notes as deferred research only. Do not use a
specialist result to claim that the Generalist gate passed.

The decisive comparison is Generalist versus Generalist on the existing 660
full-resolution whole-image composition set, using identical records and the
frozen metric protocol (`full_rgb_uint8_image` PSNR and
`area_weighted_nonoverlap_512_rgb_uint8` LPIPS). Neither side may run a
specialist. The released FoundIR Generalist reference is hash-bound in
`checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json`.

For an unambiguous pass, the EBT Generalist must beat the released FoundIR
Generalist in both PSNR (higher) and LPIPS (lower) for every composition tier,
not merely in a weighted aggregate:

| Tier | Released FoundIR Generalist PSNR | Released FoundIR Generalist LPIPS |
|---|---:|---:|
| Singleton, 300 images | 27.2323939564508 | 0.1744119563086327 |
| Double, 300 images | 22.245453630403492 | 0.2066620594635684 |
| Triple, 60 images | 24.9455711880497 | 0.03775338436922618 |
| Overall, 660 images (reporting only; not a substitute for tier gates) | 24.757709920211013 | 0.17664758756638468 |

## HISTORICAL FAILED RUN: real-data Generalist and pass-1 diagnostic — 2026-10-12

`real_generalist_4node_mixed_hopper_v1` ran on 8 x H200 plus 24 x H100. It used
global batch 64, 10,233 steps/pass, three planned passes, EMA `.9999`, and the
frozen 654,881-pair real training split. It failed during pass 2 at pass step
1,300 / global step 11,533 after a 600-second NCCL ALLREDUCE watchdog timeout.
Only pass 1 is complete. Do not resume or evaluate this run while the
2026-10-14 decision is active. Its 16,461-record 256x256-crop validation is
diagnostic only and is not comparable to the decisive 660-image protocol.

Pass-1 raw tier metrics were Singleton `28.829671/.172490`, Double
`23.916114/.242836`, and Triple `33.070977/.025842`. The apparent Double/Triple
inversion is not contradictory:

- Triple contains only `blur+low_light+noise` (579 validation records), whereas
  Double averages five heterogeneous categories.
- Double raw per-category LPIPS is `blur+noise .017810`,
  `haze+low_light .196602`, `low_light+noise .242655`, `haze+rain .277549`,
  and `blur+low_light .312665`. The last three drive the Double weakness.
- The sampler assigns one third of presentations to each composition tier. The
  sole Triple category therefore receives approximately five times the
  category-level exposure of each of the five Double categories.
- LPIPS has weight `.05` and is included only every four optimizer steps, so
  PSNR can improve before hard Double perceptual artifacts are resolved.

The former instruction to complete all three passes and run a matched 660
evaluation is deferred. Do not access the 660 endpoint for this failed run
while the 2026-10-14 decision is active.

## Historical FLUX/Restormer gap record for the monolithic/generalist route

The old routed FLUX.2/Restormer system's Triple gap to FoundIR was not a simple
parameter-capacity problem. Preserve these established causes. They motivate
the bounded energy-refinement test above but do not authorize specialist or
Triple work:

1. Most FLUX work used synthetic/task-specific data derived from roughly 1,600
   clean sources, versus FoundIR metadata covering 1,010,064 pairs and the
   frozen 654,881-pair real training split now used by EBT.
2. FLUX experts received hundreds to roughly 10,000 task-specific steps,
   whereas the released FoundIR Generalist is at step 1,300,000. Raw steps are
   not batch-normalized, but the exposure gap is material.
3. FLUX/Restormer was a heterogeneous routed expert system, not one jointly
   trained Generalist over all singleton/double/triple categories.
4. Sequential experts have measured inter-expert domain shift: on 53 Triple
   records, Noise was nearly inert (`+.069859 dB`), Low-light overshot GT luma
   by `+29.503`, and Blur worsened 49/53 records and lost `.272308 dB`.
5. Synthetic operator order/strength did not fully match FoundIR, including the
   documented Noise/JPEG visibility mismatch.
6. Short FLUX/Restormer selections lacked a mature EMA selection path; current
   EBT uses EMA `.9999` and validates raw and EMA after every pass.
7. FLUX is a large pretrained generative DiT, while FoundIR's approximately
   36.3M-parameter UNetRes is restoration-native. More FLUX parameters did not
   substitute for paired restoration alignment.
8. Full FoundIR also has Low-light and Weather specialist refinement. That is a
   separate full-system advantage and must not be mixed into the frozen
   Generalist-versus-Generalist gate. Specialist work remains paused.

After the active real-data run completes, the remaining path is:

1. Audit all pass/checkpoint/validation artifacts and hashes.
2. Select and freeze raw or EMA weights without protected-test access.
3. Run one matched 660 whole-image Generalist-versus-Generalist evaluation.
4. Only after all tier gates pass may specialist research be reopened.

Report wall-clock hours and allocated GPU-hours for every proposed or completed
GPU experiment.

## DEFERRED: potential SOTA follow-up after the Generalist gate — 2026-10-12

This records a candidate paper-evaluation program. It does not change the
active Generalist-first decision and does not authorize another training or
evaluation while `real_generalist_4node_mixed_hopper_v1` is running. First
finish all three frozen passes, select raw versus EMA using only the frozen
16,461-record validation evidence, and run exactly one matched 660-image gate.
Do not tune from that result. The project is not yet demonstrably SOTA:
pass-1 crop validation is encouraging, but Double LPIPS remains worse than the
released FoundIR Generalist and is not directly comparable to whole-image
metrics.

If and only if EBT beats the released FoundIR Generalist in both PSNR and LPIPS
for Singleton, Double, and Triple on the frozen 660 protocol, the lean
potential-SOTA program is:

1. Train/evaluate one matched CDD-11 model and report all eleven categories
   plus Singleton/Double/Triple aggregates with PSNR/SSIM. Main composite
   targets are AllRestorer `33.72/.9436`, DAR-Net `29.73/.888`, ClearAIR
   `29.34/.886`, MoCE-IR-S `29.05/.881`, and OneRestore `28.47/.878`.
   AllRestorer lacks a verified reproducible public implementation, but its
   published number cannot be omitted from an unrestricted SOTA claim.
2. Train/evaluate one matched standard 5D model on SOTS, Rain100L, BSD68 sigma
   25, GoPro, and LOL. Important average PSNR/SSIM targets include D3Net
   `32.36/.901`, DAR-Net `31.02/.921`, DPC-Net `31.00/.923`, DFPIR
   `30.64/.913`, MoCE-IR `30.58/.919`, and CyclicPrompt `30.52/.935`.
   Disclose metric/protocol tradeoffs and never compare unmatched numbers.
3. Run two matched retraining ablations: no VLM conditioning and no learned
   energy/direct restoration. Use the frozen full checkpoint for evaluation-
   only adaptive-TTS fixed-step sweeps and self-verification on/off tests.
4. Report parameters, FLOPs, peak memory, latency, and actual adaptive descent
   steps. Cite and discuss FoundIR-v2, VLU-Net, DFPIR, InstructIR, Perceive-IR,
   ClearAIR, and EBT even where benchmarks are not directly matched.

The minimum defensible follow-up has eight blocks: frozen 660 evaluation;
matched CDD-11 train/eval; matched 5D train/eval; two retraining ablations;
adaptive-TTS evaluation; self-verification evaluation; and efficiency
evaluation. This means four new training runs and roughly six to eight
evaluation jobs after the current gate. A broad AiOIR claim additionally needs
separately trained 3D, real-world, and robustness/OOD evidence, raising the
program to roughly 12--15 blocks and six to eight training runs.

The current 30,699-step 32-GPU schedule measures about 8.97 wall hours/pass,
or about 27 wall hours and 864 allocated heterogeneous GPU-hours for three
passes. Four mechanical repeats would be an unapproved conservative ceiling of
108 serial wall hours and 3,456 GPU-hours. Before any CDD-11, 5D, or ablation
launch, run a protocol-specific 500-step throughput qualification (target at
most 0.5 wall hour and 4 allocated GPU-hours on eight GPUs), then size training
by matched epochs and unique-sample exposure. Narrow the intended claim to
composite/mixed Generalist SOTA or best matched average; winning the
project-specific 660 comparison alone is not universal AiOIR SOTA. Specialist
work remains paused.
