# 本轮官方恢复：用户已批准不设最长时限

用户明确要求“不需要最长小时限制，直接启动”。因此不再等待数字预算，不把123.24
GPU-hours外推写成硬要求，不擅自改成160或其他有限cap。
范围仍是已固定的官方600张、V4 step4800、五个FLUX专家及三组对照。
保持四节点×两卡（节点0/1 H200，2/3 H100的既有runtime），不重跑route或训练。

## 实现和已有进度

原始producer、contract、wrapper、receipt及render_plan全部保持不变；它们已经绑定
19,460块特征和600个路由决定，直接改其源码会触发哈希拒绝。
新增执行覆盖层只负责记录不限时授权、运行原hardware/render/finish和保留耗时账。
科学推理代码、21,828个待生成块及GT读取顺序不变；不在这次授权里偷偷换高效推理算子。

新文件：

- smoke/official_vlm_v4_unlimited_execution.py：绑定原receipt/plan及执行层哈希。
- smoke/run_foundir_official_vlm_v4_unlimited.sh：无timeout、无累计时长上限；仍有锁和错误退出。
- smoke/start_foundir_official_vlm_v4_unlimited_tmux.sh：原四节点同名render session，拒绝重复启动。

render_budget.json的approved_render_GPU_hours/cap_seconds_per_node/maximum_wall_hours为null，
time_limit_disabled=true。用户原话和授权文件哈希记入记录。旧producer只读取该记录绑定
的plan哈希，并在最终报告中保留整份授权，不需要修改原producer。
若现场已经存在不一致的render_budget，不覆盖、不删除；报告冲突并保留旧记录。

保持每块原生输出、每图PNG/指标的原子保存和哈希复用；失败/重试成本累计。
取消时间上限不等于忽略OOM、非有限值、哈希错误、缺图或无关GPU任务。
没有自动重启循环；同命令可以恢复已保存项，孤立attempt账须先核对真实耗时。
不更改历史route上限或已结束费用；当前phase只有authorize/render/finish。

## CPU代理立即执行顺序

在唯一新工作区安全同步main，保留所有本地修改和大资产。不要重新prepare/route/plan。

1. `bash smoke/run_foundir_official_vlm_v4_unlimited.sh authorize`
   只核对原producer元数据/已发布plan身份及记录授权，不解码测试像素、不重新运行模型推理。
2. 在原四个GPU节点各启动自己的编号：
   `bash smoke/start_foundir_official_vlm_v4_unlimited_tmux.sh 0`（其他节点分别1/2/3）。
   若无远程启动权限，把四条命令给用户；不得宣称已经启动。
3. 读日志核对8个worker真实开始产生OFFICIAL_VLM_V4_TILE；tmux创建不代表开始生成。
   原producer仍会先检查源/资产/全部plan，耗时应如实计入，不误报为训练卡住。
4. 用正式任务已保存的块进度更新ETA，不另开测速或测试集，不把老估计当保证。
5. 四节点END status0和8个WORKER_COMPLETE后，执行
   `bash smoke/run_foundir_official_vlm_v4_unlimited.sh finish`。
   finish同样不设墙钟上限，但错误仍会退出。完成后推小终态报告。

日志沿用logs/foundir_official_vlm_router_v4_4x2_render_node0.log至node3.log，追加不覆盖。
输出仍在checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2。
最终需OFFICIAL_VLM_V4_RESULT_VERIFIED及finish END status0；指标未过也必须报告。

## 验证边界

主代理没有目的地挂载、远程GPU控制通道或本轮原图指标，不能亲自启动或声称已经启动。
24项CPU测试通过：7项新的执行覆盖层测试、14项官方评价回归、3项H100运行策略回归。
新测试只用合成记录，覆盖null上限、幂等授权、原产物不变、执行层篡改、旧批准冲突、
超长/失败耗时无cap、孤立账目拒绝、错误拓扑/plan拒绝、shell无timeout/有锁/不自动重试。
官方完整画质仍待GPU完成；不限时不是成功保证。
