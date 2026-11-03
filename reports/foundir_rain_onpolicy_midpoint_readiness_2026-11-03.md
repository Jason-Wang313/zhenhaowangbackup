# Rain：让训练直接面对模型自己生成的状态

## 当前决定与证据

用户已授权执行这一个独立实验：从原生Rain续训的step400完整状态出发，
增加200次更新，固定第26次模型调用的终点预测。不是继续旧2000步任务，
不是energy、MoE或一步Rain→50步蒸馏，也不新增恢复网络。

前一轮100开发图已完成（GitHub d781a1a）：step400固定prefix26为
33.129105/.07536413；同权重full50为31.680601/.07709204；
原一步Rain为34.226794/.07364027。固定prefix26提高1.448504dB，
但离34.2仍差1.070895dB。100/100两项轨迹/历史uint8一致性通过；
相比full50有9张PSNR退步、26张LPIPS退步，不能只报平均改善。

已证明：固定中途读出比走完原流程好。尚未证明：后段损害的唯一原因，
或者新训练一定能补齐差距。旧训练在GT插值得到的状态上预测，和实际生成
状态不同，因此本轮针对这一具体差异；不把“多加步就能过”当结论。

## 冻结方案

- 恢复 `continue2000_hopper_4x2_v1` **step400** 的LoRA、Adam、各rank RNG和
  采样位置12800。不是恢复其最新step2000，不用fresh Adam，不覆盖父目录。
  全局step401..600＝新200步；新目录也保存独立step400起始快照。
  **状态继承但目标函数改变**，不是原配方严格连续训练的宣称。
- 每张图：当前FLUX在原50步时间表上运行前25次，无梯度；截取第26次调用
  之前的真实状态，再以同一当前模型进行一次可微预测，得到 `x_t-t*v`，
  原生VAE解码后与训练GT比较。每次呈现都重生成前缀，不长期缓存旧模型状态。
- 只对最后一次FLUX预测和原生VAE解码反传；前缀导数被截断。
  更新共享LoRA会影响下一次生成的整个前缀，这是该近似的风险，不能保证单调变好。
  不宣称全26次端到端反传，也不声称MoE共同生成流程已经适配完成。
- 原生CFG=1的BF16 `uncond+(cond-uncond)`算式保持；不是直接丢掉uncond。
  50步shifted时间表、reference time10、512裁图、alpha1、单uint8评价保持。
  原采样函数使用inference_mode；捕获的状态必须在其外clone成普通tensor，
  否则无法正确保存给反传。这一点有专门回归测试。
- 仅两项loss：原权重的GT RGB MSE（[-1,1]域）×10，GT LPIPS×.05。
  去掉旧GT插值flow MSE、GT latent L1和teacher RGB L1；不额外添加机制。
  学习率仍1e-5，global32，micro1，每rank累积4次，FP32 LoRA/Adam、BF16 FLUX。
  FLUX之外只有原生VAE恢复图像；冻结LPIPS只是损失/评价，不是CNN修图器。
- 继续原source-uniform训练池的槽位12800..19199，共6400次呈现。
  完整池38334记录/5780来源不变；不是只拿32图训练，也不称遍历完整数据集。
  所有这些LQ/GT缓存已由旧任务生成，逐张核验后复用；不生成新teacher、GT latent。
  缓存文件可能含旧teacher/latent，但新生成与目标函数只接收LQ/GT，不使用其它字段。
- 不使用开发图构造训练状态、损失、噪声或选输出；官方/封存/最终测试全部禁止。
  第26次预测位置固定，不扫描位置，不逐图选最优，不用GT筛回原图。

## 检查与成功口径

1. 训练前在本轮真实前32次训练呈现上，比较新可微路径的无梯度输出与原验证过的
   prefix26读出，须32/32逐像素相同。不相同则失败退出，不训练、不放松断言。
2. 新0/50/100/200步（全局400/450/500/600）检查这32张固定训练图。
   这是确实被本轮呈现的训练拟合探针，不再把旧32个未曝光裁图称为过拟合图。
   其噪声固定且与训练对应；不能把这些分数包装成泛化证据。
3. 只在预定最终新200步评价原100开发图，重跑同环境step400 prefix26作为匹配对照，
   同时保留原已验证历史指标，报告新环境对照与历史差异，避免换弱分母。
   两套权重分批使用同一个FLUX实例，不同时驻留两个恢复网络；评完step400后
   恢复并核对step600权重，不碰Adam状态，不继续更新。多2600次模型调用只是对照成本。
   不混入32训练图平均，不选择450/500的开发最优值。
   原100开发图已被使用，不是独立盲测。
4. 同一输出平均PSNR≥34.2且LPIPS≤.08才过门槛；同时报告相对33.129105/.07536413
   的历史差异，以及相对同环境起点的增量和逐图退步，即便绝对门槛过了也不掩盖LPIPS比起点变差。
5. 运行终态只做轻量核对：四节点退出、关键step600/最终导出、milestone均值和完整性。
   不全扫201个checkpoint tensor、不重跑LPIPS。`result.json`是发布verified的同一文件，
   不再继承旧light_result.json/result.json名称不一致的收尾错误。

每完成一步原子保存LoRA/Adam/8rank RNG/采样位置/历史及哈希；新目录400..600
共201份。中断未完成的那一步可能重算。每张probe独立落盘，恢复先补完中断评价。
同run_id、新的共同launch_id恢复，禁止改绑定源码或换ID伪装恢复。换world_size
不支持严格恢复；H100/H200各节点软件和指标环境必须一致，单节点两卡同型号。
异常/非有限值停止并保留最后完整步；不自动改loss、增加步数、扩预算或启动MoE。

## 资源与入口

仍是**4节点×每节点2张H100/H200，总8卡**；每卡≥75GiB。跨节点混型号为代码支持，
不是本实验已实测。保持原Hopper显存策略：VAE query256分块保留全部keys/values，
非叶中间激活CPU转存、原生FLUX activation checkpoint、cuDNN工作区策略。
本轮最后一次使用完整CFG双分支反传，H100真实显存/吞吐尚待CUDA确认。

粗估45–120普通分钟、6–16使用GPU-hours，**不是实测保证**。
每节点硬cap3小时，即最多24使用GPU-hours加清理；恢复尝试成本累计，不清零。
用最初5步的训练时间加保存时间更新ETA，前缀耗时单独记录。
不生成6400份新图像缓存；201份状态约73GiB，加输出留余量，首次要求100GiB空闲。
空间不足报告，不删除旧模型。主代理看不到目的地GPU挂载，不声称已启动。

目的地只在新根运行：
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
先安全同步GitHub，保留本地未提交文件和大资产；运行：

```bash
.conda/gpu-train-stable/bin/python smoke/train_foundir_rain_onpolicy_midpoint_v1.py preflight
```

preflight只读元数据，核验父权重receipt、baseline结果和所需缓存receipt存在及绑定；
父step400大tensor在真实加载时校验。预检不通过不要自动重生成旧缓存。

目的地确认当前node0 IP后，给用户四条命令（node_rank分别0/1/2/3）：

```text
bash smoke/start_foundir_rain_onpolicy_midpoint_v1_tmux.sh <node_rank> <当前node0地址> step400_onpolicy200_4x2_v1 launch_onpolicy_midpoint_v1
```

端口29696；四节点相同run/launch，恢复要用fresh共同launch。不要照抄占位IP。
节点共享旧训练锁，不杀其它任务。tmux创建不是训练成功。
结果目录 `checkpoints/foundir_rain_onpolicy_midpoint_v1/step400_onpolicy200_4x2_v1/`；
日志 `logs/foundir_rain_onpolicy_midpoint_v1_step400_onpolicy200_4x2_v1_node<N>.log`。
终态须`RAIN_ONPOLICY_RESULT_VERIFIED`和四节点`RAIN_ONPOLICY_NODE_END status=0`；
训练完成与质量通过分开报告。

目的地请在首5步、50/100步及终态回传小报告至GitHub：最后完整全局步/新步数、
loss、梯度范数、显存、每步/保存/前缀秒数、训练探针两项指标；最终加开发均值、
逐图退步计数、每节点状态/耗时及累计GPU-hours。禁止上传tensor、权重、图片或实时日志。
不等全量CPU审计，不自动新实验；分析和代码修复仍由主代理负责。

## 本地验证状态

11项CPU回归全部通过，包括：真实native sampler的25+1调用与原读出一致、
inference tensor转为可反传输入、只有LoRA得到梯度、当前权重更新会改变前缀、
GT隔离、槽位/评价分片、step400完整Adam/RNG恢复、坏checkpoint拒绝、门槛与启动契约。
测试最初一次恢复样例误用了不同随机底座，已修正为相同冻结底座并全套重跑通过；
这不是放宽恢复精度或修改生产恢复逻辑。

两进程Gloo小模型测试通过：真实detached-prefix梯度、梯度累积、跨rank参数一致、
Adam/RNG恢复后的下一步参数和Adam状态逐元素相同。单元测试真实执行400次小模型更新；
DDP测试用一次真实Adam初始化后将计数置400来测试边界，不冒称跑过400步多节点FLUX。
Python AST、两入口bash语法、配置/发布报告身份核对及focused密钥扫描通过。
没有读取目的地大tensor、没有CUDA运行、没有验证真实4节点NCCL或H100峰值显存。
本地测试通过只表示实现可交付目的地预检，不是训练已启动或质量已改善。
