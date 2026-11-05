# 官方V4评分身份修复与恢复

首轮四节点在第一块之前因`official metric weights changed`全部退出。CPU完整评分身份仍与receipt匹配；pyiqa的`Parameter(empty).to(device)`在发生设备复制时变成普通Tensor，使包装器state_dict缺失dummy_param，形成设备相关哈希。CPU/meta复现该注册差异；实际旧GPU侧各项哈希没有输出，未宣称排除所有其它问题。

新增执行层保留原科学producer、原unlimited层及全部绑定文件：先调用原metric_models('cpu')，与原receipt完整state比较；随后整个Module.to(device)，保持Parameter注册并迁移所有真实网络权重，再将所有state回CPU逐字节哈希与原receipt复核。没有过滤dummy键或跳过真实权重检查。生成和评分仍调用原render/finish，只替换评分模型构造顺序；权重、精度、默认指标、输入与路由不变。

新metric_identity_fix.json绑定原receipt/plan/unlimited_execution以及新执行代码SHA。任一源码、真实评分权重或迁移后的状态不匹配即拒绝。旧记录不覆盖，原19,460块特征和600路由决定保留；不重新prepare/route/plan。累计失败费用0.918888889 GPU-hours继续保留（已完成route另1.74 GPU-hours，官方当前合计2.658888889 GPU-hours）。

CPU先执行：
```bash
bash smoke/run_foundir_official_vlm_v4_metricfix.sh prepare
```
该prepare只核验原授权/绑定、已结束账目和CPU评分模型，不提取特征或读图。
四台GPU终端先进入共享根目录，各运行对应编号：
```bash
bash smoke/start_foundir_official_vlm_v4_metricfix_tmux.sh 0
bash smoke/start_foundir_official_vlm_v4_metricfix_tmux.sh 1
bash smoke/start_foundir_official_vlm_v4_metricfix_tmux.sh 2
bash smoke/start_foundir_official_vlm_v4_metricfix_tmux.sh 3
```
四行分别在四台执行，不在同一节点全部运行。无跨节点NCCL。每节点内两张同型号H100/H200，沿用原runtime身份。
新日志`logs/foundir_official_vlm_router_v4_4x2_render_node<N>_metricfix.log`。旧tmux session名用于防重；旧失败session若仍存在先检查，不自动杀任务。
首次启动仍会在原render中核验完整plan，上一轮花约7分钟后才抵达评分模型构造；这是启动核验耗时，不是训练。看到`OFFICIAL_VLM_V4_METRIC_DEVICE_VERIFIED device=cuda:<编号> full_state=true`说明评分迁移及身份核验通过；只有TILE落盘才算实际生成。

全部8worker完成、四节点END status0后：
```bash
bash smoke/run_foundir_official_vlm_v4_metricfix.sh finish
```
不限时、不自动重试，保留原锁、每块/整图保存和错误退出；旧失败和新尝试统一进入原attempt账。
只推小进度/终态报告，原终态报告之外也需记录本次修复binding哈希。CPU合成回归通过不代表官方CUDA或完整指标通过。
