# 无SSH的共享目录GPU执行：所有接手代理必读

本项目已经具有CPU代理向GPU节点派发任务的通道。用户在GPU节点启动一次常驻worker后，能访问同一共享目录的后续代理可以提交已授权任务、读取运行日志、完成CPU收尾并推送报告。不得在未检查现有worker/队列前，仅因本机没有CUDA或SSH就宣布只能给用户命令。

这是执行通道，不是自动科研决策系统。worker执行明确提交的脚本，不自行选择配方或扩大预算。代理退出对话后，仍在运行的是tmux中的worker和另行启动的CPU监视器；不能声称语言模型本身一直在线思考或监控。

## 原理及本项目的来源

```text
CPU代理：准备/预检代码 → 发布GitHub → 写共享job.json
                                    ↓
GPU节点：已启动worker轮询队列 → 校验源码/领取任务 → 本地启动子进程
                                    ↓
共享目录：heartbeat + started + node.log + ended + 模型/结果
                                    ↓
CPU代理或持久监视器：核验真实进度 → CPU收尾 → 推小报告
```

不需要CPU到GPU的SSH端口；需要双方都能访问同一个可读写共享文件系统，且GPU节点已有能够执行任务的进程。GitHub保存代码和小报告，不承担实时任务派发。GPFS共享目录才是本项目的通信通道。

可核验历史：`f285a8a`（2026-11-06 12:10 UTC）记录用户全流程授权；`6e0f985`（12:24 UTC）新增初版执行器及说明；`e7d0a2b`（12:36 UTC）隔离旧四节点队列并适配当前三台H100。会话中用户明确选择“提供一次性GPU常驻执行器命令，我在各节点启动”，后来确认三台均已READY。这是根据“没有SSH、但有共享存储、用户可在节点执行一次命令”的条件搭建的常规任务队列，并非隐藏的SSH权限或平台自动赋予的远程执行能力。

## 当前实现与适用边界

项目根：`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`

| 项目 | 当前入口 |
|---|---|
| Python | `.conda/gpu-train-stable/bin/python` |
| worker/submit/cancel/inventory | `smoke/foundir_fullstack_executor_3x2_v2.py` |
| 一次性worker启动 | `smoke/start_foundir_fullstack_worker_3x2_v2_tmux.sh` |
| 活跃队列 | `checkpoints/foundir_fullstack_executor_3x2_h100_v2` |
| worker日志 | `logs/foundir_fullstack_worker_3x2_v2_node<N>.log` |
| 节点登记 | 队列下 `workers/node<N>/registration.json` |
| 心跳 | 队列下 `workers/node<N>/heartbeat.json` |
| 任务记录 | 队列下 `jobs/<job_id>/` |
| tmux | `.tools/tmux-env/bin/tmux -L zhenhaowang_jobs`，`TMUX_TMPDIR=/tmp` |

当前映射（节点重建后必须重新实测）：

| node | IP | notebook | GPU |
|---|---|---|---|
| 0 | 10.244.211.17 | 37497631-24c8-4a19-907e-494c701599ab | 2×H100 80GB |
| 1 | 10.244.47.55 | 2551e049-18b3-4c01-8490-735ced1e58cd | 2×H100 80GB |
| 2 | 10.245.39.185 | 55ef95b9-862f-4ffa-a2ed-6eaa7156ccad | 2×H100 80GB |

方法适用于其它节点和其它代理，但当前v2实现不是通用任意节点调度器：ROOT、NODES=3、IP映射、H100型号、每节点两卡、入口前缀、任务范围均有检查。其它代理必须有共享目录和执行工具权限，并先读取最新AGENTS及本手册；只有GitHub网页访问权限并不能派发任务。新节点、H200、不同卡数或其它项目需要先适配并验证新的节点配置、环境、任务脚本及分片。

不要更改运行中的worker或任务绑定源码。换集群应新建版本/队列/节点登记，保留旧队列与结果；不要删除started/ended以强迫重跑。旧四节点v1入口和队列仅供历史查看，不能混用。不能同时让旧、新执行器向同一GPU派发工作。

## 接手代理第一步：只读检查，而非重新启动

在共享项目根执行：

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
.conda/gpu-train-stable/bin/python -S - <<'PY'
import json, time
from pathlib import Path
queue = Path('checkpoints/foundir_fullstack_executor_3x2_h100_v2')
for node in range(3):
    path = queue / f'workers/node{node}/heartbeat.json'
    if not path.exists():
        print(f'node{node}: heartbeat missing')
        continue
    value = json.loads(path.read_text())
    print(f'node{node}: age={time.time()-value["utc_unix"]:.1f}s '
          f'state={value["state"]} job={value.get("job")} host={value["hostname"]}')
for path in sorted((queue/'jobs').glob('*/job.json')):
    job = json.loads(path.read_text())
    for node in job['nodes']:
        start = path.parent / f'node{node}.started.json'
        end = path.parent / f'node{node}.ended.json'
        status = json.loads(end.read_text())['status'] if end.exists() else 'unfinished'
        print(job['id'], 'node', node, 'started', start.exists(), 'status', status,
              'cancelled', (path.parent/'cancel.json').exists())
PY
```

心跳通常运行时约2秒更新，空闲轮询约5秒；共享存储延迟可能使年龄暂时变大。检查时间、state、job、日志和检查点的持续变化，不能仅凭一次心跳或一段时间没有新图片就杀进程。心跳更新证明worker仍在运行，不单独证明CUDA正在计算；`started`证明launcher开始，不证明模型已成功加载。还须看rank就绪、NCCL（若需要）、实际step/图块/保存状态。

`tmux ls`只列当前所在主机的tmux，不会列出另外两台GPU的会话。CPU端用共享心跳和任务文件检查远端。缺心跳/blocked时先核对日志；无法经现有通道核实时，再请用户在相应节点做一次本地检查。

## 节点一次性启动命令

以下命令仅供未启动worker或核实worker确已退出后的恢复。当前三台已经启动，正常接手不需要重跑。分别在对应GPU节点终端执行：

```bash
# 仅node0（10.244.211.17）
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
bash smoke/start_foundir_fullstack_worker_3x2_v2_tmux.sh 0
```

```bash
# 仅node1（10.244.47.55）
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
bash smoke/start_foundir_fullstack_worker_3x2_v2_tmux.sh 1
```

```bash
# 仅node2（10.245.39.185）
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
bash smoke/start_foundir_fullstack_worker_3x2_v2_tmux.sh 2
```

启动器核验IP、真实型号和显存；worker登记UUID并拒绝把同一物理GPU登记为两个节点。必须看到`FULLSTACK_WORKER_READY`及新鲜共享心跳。`FULLSTACK_WORKER_TMUX_CREATED`或`SESSION_EXISTS`不是任务成功证明。使用实际自带tmux，不能用永远返回0的fakebin脚本伪造状态。

## 代理提交任务的准确接口

先读本轮科学范围和预算，完成prepare；检查现有任务及持久监视器，确保不重复提交已安排任务。轻量GPU工作默认选择一个节点，例如`--nodes 0`；是否跨节点由任务成本决定，不因有三台就全用。任务代码本身也必须按所选节点/卡数分片，不能仅改`--nodes`就认为算法已适配。

先将本轮小源码安全推送到GitHub，fetch后确认提交已在origin/main历史中；本地源码必须与指定提交逐字节一致。保留共享checkout的未提交修改，不pull/reset/clean覆盖；使用独立临时worktree发布明确文件清单。运行中绑定文件不得修改。

以下是**模板，不能原样执行**；替换大写占位值后只提交已授权的新任务：

```bash
.conda/gpu-train-stable/bin/python -S smoke/foundir_fullstack_executor_3x2_v2.py submit \
  --job-id UNIQUE_NEW_JOB_ID \
  --entry smoke/run_foundir_fullstack_TASK.sh \
  --commit PUBLISHED_FULL_COMMIT_SHA \
  --purpose '本次已授权任务的范围与终点' \
  --nodes 0 \
  --sources smoke/run_foundir_fullstack_TASK.sh smoke/TASK.py
```

三节点任务使用`--nodes 0 1 2`，不是修改全局NODE_MAP。`--sources`列出本轮相关源码；入口和executor会自动加入。附加参数用`--arguments`列表传递。worker实际执行：

```text
bash <entry> <node编号> <arguments...>
```

因此入口第一个位置参数必须是node。入口负责使用正确Python、GPU空闲检查及节点锁、CUDA_VISIBLE_DEVICES、torchrun进程数和本任务预检。当前各新launcher用`/tmp/zhenhaowang_existing_flux_replay.lock`及GPU compute-process检查防止冲突。

worker每节点串行处理队列中的job，且按目录名排序；**不是FIFO、依赖调度器、全局资源调度器或自动重试器**。多节点训练仍需入口设置MASTER_ADDR/PORT/NODE_RANK、验证互通/NCCL；队列不解决网络问题。独立推理/评价可以各节点standalone运行，无需跨节点NCCL。

多个代理可以接手同一通道，但同一轮应只有一个明确的提交负责人；其它代理先看队列与已运行CPU监视器，不各自提交重复实验。文件锁只防重复worker/并发写入，不能替代实验预算及跨代理协调。

## 进度、收尾和故障处理

在CPU共享目录读取本轮`jobs/<job_id>/node0.log`、node1/2日志，以及对应的`node<N>.started.json`、`node<N>.ended.json`。按job.json中的nodes检查所需节点，不要求单节点任务也有三份结束文件。

只有所需节点全部退出0、任务所有rank/记录齐全、专用finish核验成功，才报告工程完成。科学成功必须另看完整指标和门槛。保留失败、退步、费用及已有曝光限制；GPU结束后完成已授权CPU收尾并推送小JSON/Markdown报告，不上传权重/图片/缓存。

退出记录包含`status`、`seconds`、`gpus=2`及`allocated_GPU_hours`。汇总费用为各节点GPU-hours之和；并行wall time不能把各节点seconds直接相加。开始时间字段是started中的`utc_unix`，结束记录没有独立时间戳，可按开始时间加seconds计算。

失败不会自动重试。有started无ended时先核验原进程与资产，不能删除标记重跑；现有worker会拒绝盲目重新执行。需要取消已明确决定停止的任务时，接口为：

```bash
# 模板：仅在已明确决定停止指定任务时使用
.conda/gpu-train-stable/bin/python -S smoke/foundir_fullstack_executor_3x2_v2.py cancel UNIQUE_JOB_ID \
  --reason '具体停止原因'
```

cancel仅标记该job；worker终止自己创建的子进程组并保存退出记录。不要宽泛pkill，也不要为了解决文档或报告问题停止正常GPU训练。

## 当前已存在的持久CPU监视器：不要重复启动

- `smoke/watch_foundir_fullstack_rain_continue_v1_cpu.py`：观察固定800→1000，推首次五步/900/终态报告，最终调用训练finish。
- `smoke/watch_foundir_fullstack_rain_step1000_v1_cpu.py`：等待上述训练正常结束、CPU核验及报告发布，执行最终权重prepare，再提交固定Rain100评价，完成后核验并推报告。

CPU tmux session分别为`rain_fullstack_monitor_v1`、`rain_step1000_eval_monitor_v1`；日志为`logs/foundir_fullstack_rain_continue_v1_cpu_monitor.log`和`logs/foundir_fullstack_rain_step1000_v1_cpu_monitor.log`。这说明自动后续工作由实际持久程序驱动，而不是在回复里承诺后就凭空运行。

本手册不新增任何实验授权。训练、评价、节点数、数据范围和预算继续服从用户当前指令及AGENTS顶部；保留旧资产，完成本轮后按真实结果决策。

## 2026-11-06新增第四台：node3独立任务扩展

新node3 IP10.244.90.116、notebook e087f2fc-829f-4a0d-8aa5-5df18b6cd56a，用户回传两卡H100及共享Python检查通过。用`bash smoke/start_foundir_fullstack_worker_node3_v1_tmux.sh`在第四台一次性启动；代理先看node3实际登记/心跳再声称可用。当前3台v2源码/进程不改。

新工具`smoke/foundir_fullstack_node3_extension_v1.py submit`固定nodes=[3]，复用共享队列。旧worker会跳过该任务；新node3拒绝混合节点列表。**不能直接向旧v2派发四节点job**，四节点联合训练仍需另行适配。单节点任务无需为第四台到来扩大范围。完整规则见`reports/foundir_fullstack_node3_extension_v1_readiness_2026-11-06.md`。
