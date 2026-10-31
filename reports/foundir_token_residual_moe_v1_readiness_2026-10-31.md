# 右图VLM分数修正MoE：实现与四节点交接

## 当前决定

只做③，不先要求跑完①/②。用户最后资源更正是8卡，按其此前明确的
每节点2卡，即4节点×2张H100/H200。不是单机8卡，也不是3节点6卡。
节点内同型号、节点间可混用，global batch32不变。
GPU尚未启动；本文不是工程验收或画质成功报告。

另一位代理继续处理FLUX-native energy显存，不覆盖其模块、不终止其运行。
本轮新文件均使用token_residual_moe前缀，不修改历史MoE或energy实现。

## 做什么，为什么这样做

1. 从image-only step11000的专家、router与共享LoRA权重出发。
   Rain已达34.41416/.064399，Blur为26.61952/.188334；这些是历史结果，不是VLM贡献。
2. 冻结现有Qwen2.5-VL与V8 adapter，仅提取视觉merger输出token的均值（3584维）。
   这是VLM视觉表征，不是生成的退化标签、语言推理token，也不是Qwen3 FLUX文本编码器。
   输入为与FLUX完全相同的512退化裁块/D4增强，随后按既有Qwen processor处理。
   GT仅用于损失和指标，不输入VLM或router。旧整图token/五等级cache不冒充新裁块cache。
3. 每个原MoE块增加LayerNorm→Linear(3584,256)→SiLU→Linear(256,2)；
   输出加到原图像router的两专家logits，然后softmax。固定alpha=1，不学习额外机制。
   末层全零，接入时必须复现已加载image-only模型的输出，CUDA验收误差<=1e-5。
4. 原五个FLUX双流块的两FFN专家、图像router、共享rank32 LoRA继续训练；
   不另造恢复网络，不调用Restormer，不使用可训练CNN，不做energy二阶反传。
5. 保留t=1、512裁块、RGB/latent/LPIPS等历史损失，Rain LPIPS系数.05、Blur .20；
   学习率仍expert/shared LoRA 1e-5、router/VLM 1e-4。保留历史训练辅助router CE .01，
   类别只监督损失，绝不作为硬路由输入，推理不看任务标签。

这样先回答“在已有恢复能力上接入VLM分数修正能否工作”，不重做失败的原始随机专家路线。
但专家与router共同继续训练意味着：单组变好也不能独自证明VLM贡献。匹配消融留待有信号后。

## 25步资格与训练数据

- 完整旧source-disjoint Rain/Blur训练池不变，按来源均衡取样；没有降成几十张固定训练图。
- 25步×batch32=800次训练曝光，16 Rain/16 Blur每步；沿历史全局样本位置352000开始，
  使用新分支seed。不是恢复旧Adam/RNG，也不冒称和旧硬件逐bit可比。
- 同一批样本先提取冻结VLM token cache，再在独立进程加载FLUX训练，避免双模型同时占显存。
- step0/25评估固定Rain100与Blur preview100。preview不能代替完整Blur2595科学门槛。
- 25步检查有限loss/梯度、VLM分支真实更新、接入不破坏初始输出、checkpoint/指标写出。
  不因为25步未提高就宣判科学失败；不因工程通过就宣称VLM有效。
- 没有自动1000/3000步长训入口。正式规模/耗时在本轮实测后冻结，不能从资格checkpoint起训。

## 工程与中断恢复

默认torchrun nnodes=4、nproc_per_node=2；rank0/1/2/3是节点编号，不是GPU编号。
每卡每次一图，累积四次；DDP八rank平均后是严格batch32，不补样本。
启动检查跨rank代码/config/版本一致、四个独立hostname、节点内GPU一致、
所有节点可读彼此新建共享标记。不得沿用旧IP。

每5步保存完整训练参数、Adam、8rank RNG和相对采样位置；receipt最后写出。
未完成文件不能当断点；重试用新run ID并显式指定最近完整resume receipt。
恢复只支持同代码/配置/数据/拓扑的本轮qualification，拒绝旧MoE/三节点断点。
同型号跨节点也不承诺逐bitCUDA复现。最多损失最后不足5步；不说每步都已保存。
所有历史checkpoint与日志保留。五次完整checkpoint粗估约50–60GB，目的地需留余量。

未实测预算：.5–2小时、4–16 allocated GPU-hours，包括Qwen提取和评估；
硬cap3小时/24 GPU-hours（从每节点启动计，含等待）。网络与共享盘可能主导开销；
不能把旧单机1.79秒/步当跨节点测速。每次启动打印起止时间，报告两个阶段总耗时。

## CPU核验与剩余边界

已完成11项CPU单元检查：零初始化输出保持、加性公式与梯度、错误/缺失/非有限
VLM输入拒绝、禁止oracle路由、关闭VLM恢复图像路由、batch32/任务均衡、采样序列、
cache身份与不可覆盖写入、checkpoint重计算条件保留、分片梯度对串行batch32。
另完成真实8进程Gloo训练：参数/Adam/RNG从checkpoint逐位恢复；继续优化与未中断
对照的最大参数差为7.450580596923828e-09，CPU容差atol1e-7/rtol1e-6通过。
重新创建DDP reducer可能改变FP32归约顺序，因此不称“恢复后继续计算逐位相同”。
串行batch32与DDP两步Adam比较也通过（atol2e-6/rtol2e-5）。
新Python文件编译、shell语法检查通过；没有安装依赖或改动训练环境。
旧资产目录只读metadata核验通过：Rain38334、Blur105057训练记录，计划800训练
曝光+200开发检查；该核验没有读取像素，也没有代替新目录的完整资产哈希预检。

主会话只能读旧目录中的历史资产，
没有新GPU挂载/四节点IP，不能声称目的地资产存在或NCCL/CUDA已通过。
实际目的地为：
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
不得在旧agent-driven-world-model目录启动实验。

## 新CPU代理同步清单

从origin/main读取本节，只同步这些文件，保留其energy显存改动与所有数据/权重：

- `configs/foundir_token_residual_moe_v1.json`
- `lowlight_restoration/token_residual_moe.py`
- `smoke/token_residual_moe_v1.py`
- `smoke/train_foundir_token_residual_moe_v1.py`
- `smoke/run_foundir_token_residual_moe_v1_4x2.sh`
- `smoke/smoke_token_residual_moe_cpu_ddp_v1.py`
- `tests/test_token_residual_moe_v1.py`
- 本报告与AGENTS.md、agents.md、current handoff顶部新节（有本地新内容时合并，不覆盖）。

先fetch，检查dirty差异并备份/合并冲突；不要reset/clean或整仓覆盖。
使用验证过的Python环境；启动器允许PROJECT_PYTHON显式指定，不退回系统Python。
在目的地运行训练脚本的`--phase preflight`，只做CPU身份/资产核验。
通过后核对四节点当前地址、端口及空闲GPU，再给用户每节点的启动命令；
用户此前要求主代理只push并给同步提示，因此本轮主代理不直接发GPU启动命令。

启动器参数依次为node_rank、当前node0地址、四节点共同run_id，
可选第四参数为本轮完整resume receipt。预检通过不授权自动正式长训。
三节点或单节点命令明确不适用于本轮。
