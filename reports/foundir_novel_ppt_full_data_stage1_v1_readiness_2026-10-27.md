# Novel PPT Approach：全数据第一阶段启动约定

## 范围

用户要求节点1运行MoE迁移，节点2运行PPT。已结束的fixed32/300步不重跑，
也不把其权重带入本阶段。PPT energy从零初始化，冻结原Noise proposal、
step8530 residual basis及negative experts。先到原定第一个开发检查点1706步，
验证完成后停止；不能自动扩成8530步约52小时训练。

32张训练裁块能拟合不代表能泛化，也没有证明energy自选优于固定三步。
本阶段检查这一缺口，不增加新方法，不重新设计损失或提高更新幅度。

## 不变的实验配方

- 完整Noise训练池54585条、1755来源；按来源均匀抽样，再抽记录，不是固定32张。
- 全局batch32、每rank四个microbatch；训练与开发裁块256。
- 固定256条source-disjoint开发集；第0步和1706步评估。
- 634032参数energy、完整1/2/3步轨迹、真实错误/重复专家候选、逐图VLM强度。
- 所有loss、AdamW、100步warmup及8530步余弦LR跨度保持资格版本不变。
- 不读取受保护像素。内部split仅读清单元信息作排重。

配置：`configs/foundir_novel_ppt_full_data_stage1_v1.json`。
SHA-256：`dc7a41787e75731f2034644d58c5c9fdef2744a34f7ccf1e0c1fba723a2fab2b`。
资格receipt仍为已通过的25步版本，只核验身份，不加载它的energy权重。

## 启动与断点

单机同型号8×H100或8×H200，不跨节点DDP。启动器检查空闲显存、两把旧任务
节点锁、GPU型号/数量和180秒NCCL通信；新ID避免覆盖。旧源码和config不动，
保证正在运行的MoE来源绑定不受影响。

节点2启动：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_novel_ppt_full_data_stage1_v1_tmux.sh stage1_v1
```

日志：`logs/foundir_novel_ppt_full_data_stage1_v1_stage1_v1.log`。
输出：`checkpoints/foundir_novel_ppt_full_data_stage1_v1/stage1_v1/`。
tmux session：`novel_ppt_full_stage1_v1`。

每100步及1706步保存模型、Adam、8个rank的CPU/CUDA/Python/NumPy RNG、
下一采样步、参数顺序和config/source绑定。权重与小JSON receipt均先写临时文件
再原子改名，完整receipt是可恢复依据，日志步数不是。

若中断，先确认旧进程停止，再用新run ID和明确的完整receipt恢复。例如
**仅当step100 receipt确实存在**时：

```bash
PPT_PREVIOUS_RUN_STOPPED=1 bash smoke/start_foundir_novel_ppt_full_data_stage1_v1_tmux.sh resume_v1 checkpoints/foundir_novel_ppt_full_data_stage1_v1/stage1_v1/step_000100.json
```

不会重置Adam、warmup或采样顺序。拒绝qualification/trainfit/不同源码的checkpoint。
即使1706权重写完而最终验证中断，也可从1706 receipt在新ID重做同一固定验证，
不再训练。更换GPU型号记录为迁移，不承诺逐bit一致。

## 成本和判读

旧资格中位21.83572708698921秒/步，1706步纯训练约10.35小时；规划
10.5–13普通小时、84–104 allocated GPU-hours，依据全数据实际速度修正。
8×H100与8×H200均支持，但不假定相同速度。

逐项报告proposal、Fidelity、energy第1/2/3步和energy自选的PSNR/LPIPS。
绝对目标为PSNR>=43.424160178005695、LPIPS<=.045362067760033824；
还必须相对同协议Fidelity提高PSNR且LPIPS不差，才有本开发集energy贡献。
单独比较自选与固定第三步，不把梯度修图收益当作自验证收益。
本阶段不自动声明其他退化泛化、SOTA或论文创新成立。

## 验证状态

19项CPU测试全部通过：7项新增状态/LR/采样/恢复/损坏保护测试，加12项原数值
和trainfit回归。CPU小模型的恢复后下一次更新与未中断版本权重、Adam逐bit一致；
这不是跨GPU型号逐bit一致的承诺。两个shell入口分别通过bash语法检查。
真实qualification、冻结权重、数据元信息和23项源码绑定预检查通过，读取图片0张，
确认54585训练记录、1755来源、256开发记录、1706停止步和8530 LR跨度。
新增GPU恢复分支尚未在CUDA上实测；模型/三步二阶反传复用已通过的8×H100资格。
本轮不在Codex启动GPU训练；用户执行节点2命令后才能确认实际启动。
