# 4/8卡兼容与后续实验固定规则

用户要求本次以及未来所有实验支持单机4/8×H100/H200。已将长期规则置于
AGENTS.md及current handoff最高优先级；不恢复任何历史研究方向。

## 本次实现

- canonical launcher：`smoke/run_foundir_flux_tail_oracle_v1_gpu.sh <run_id> <4|8>`。
  `_8gpu.sh`保留为向后兼容wrapper。默认8卡保持旧命令兼容，四卡显式传4。
- tmux传递卡数及CUDA可见列表，不受已有tmux server旧环境误导。
- Python hardware/DDP检查实际4/8数量、同节点、同型号H100/H200和>=75GiB显存。
- 固定8训练来源：四卡分片[0,4]/[1,5]/[2,6]/[3,7]；八卡各一条。
  每rank按顺序处理，函数返回后gc/empty_cache释放模型，不增大batch。
- 输出按sample index而非仅rank命名，避免四卡第二张覆盖第一张；汇总按sample
  排序，核对0–7各一次及8个独立record/source，保留相同46个arms。
- 原权重、seed、裁图、loss目标、步长、量化和科学判据不变；不是缩小实验。
- prepare绑定实际world_size；run/verify不能改成另一卡数复用同一run。
- 四卡3h、八卡1.5h预算cap，均12 GPU-hours，额外最多30秒清理；不是已测速ETA。
  结果与launcher按实际使用卡数计GPU-hours，不能继续乘固定8。
- 若用户实际占用8卡完整节点只暴露4卡，所用4卡GPU-hours之外还要另报8卡占用成本。

## 运行

四卡H100或H200，同一条命令：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/start_foundir_flux_tail_oracle_v1_tmux.sh mechanism_4gpu_v1 4
```

八卡仍可运行，必须独立fresh ID，不要同时重复占用一个节点：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/start_foundir_flux_tail_oracle_v1_tmux.sh mechanism_8gpu_v1 8
```

## 验证与限制

10项CPU测试通过（3.646秒，不含启动），包含原8项和新增4/8分片完整性、
卡数/预算防越界、汇总结果一致性。Python/shell语法检查通过。
四卡及八卡参数的真实权重/缓存/来源身份预检均通过，未解码原始图片。
本轮不运行GPU、不训练energy；GPU真实4/8资格均待用户节点运行。
长期规则要求未来训练通过梯度累积保持global batch和数据曝光量，
卡数改变时严格审查续训状态；不是把所有旧脚本一律宣称已改完。
