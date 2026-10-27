# 两条路线的执行准备 — 2026-10-27

## 结论与状态

用户已授权“execute both”。两条独立GPU入口已经实现，CPU测试及真实权重/元信息
预检查通过。当前环境没有CUDA设备，也没有远端GPU执行通道，因此**未启动GPU任务**，
本轮消耗0 GPU-hours。以下是可以交给GPU节点的下一步，不是新的效果报告。

旧审计与PPT：

- 审计：reports/foundir_vlm_and_novel_ppt_retry_audit_2026-10-27.md。
- 原PPT已改名：presentations/novel_ppt_approach/Novel_PPT_Approach_VLM_Restoration_Energy.pptx。
- 名称不代表方法已被证明新颖或有效。旧失败记录保留，不恢复EBT Generalist。

## A：先找到MoE的Blur瓶颈，不再只改router

已完成的image-only与VLM辅助MoE都通过Rain目标，但Blur LPIPS分别为
.23083898932184047与.2282890860716721，均未达到.18。
固定Blur专家也未修复差距。因此目前证据更支持“专家恢复能力或训练目标有问题”，
尚不能断言是哪个损失造成，也不能保证改router会解决。

本次冻结两个step10000模型，不更新任何参数。按来源独立选取16个Rain、
16个Blur训练样本；同一512裁块分别进入两个模型，共64个模型—样本组合。
不按成绩选图，不使用开发图或受保护图。

检查flow MSE、latent L1、RGB MSE、Charbonnier、高频损失、LPIPS及router CE
在代表性参数上的加权梯度。训练类别只用于CE损失，不作为推理router的额外提示。
探针为首个共享LoRA B和第0/4个FFN块的两个专家输出矩阵。
张量L2范数精确计算；方向余弦使用每张量最多16384个固定坐标，
**不是整个网络的精确梯度余弦**。无优化器、零优化步；参数抽样检查未改变。

这些结果用于判断损失是否压过/对抗感知目标、两专家的训练信号是否不同。
它只是小规模训练来源诊断，不能证明某个损失权重最优，也不能证明开发集会改善。
看完实测结果后再决定改变哪一项训练机制，不能直接重复旧10000步配方。

预计单节点8×H100/H200：.25–.75小时，2–6 allocated GPU-hours。
启动：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_dual_retry_v1_tmux.sh moe diagnostic_v1
```

日志：logs/foundir_dual_retry_v1_moe_diagnostic_v1.log。
结果：checkpoints/foundir_dual_retry_v1/moe/diagnostic_v1/result.json。
进度标记：MOE_TRAIN_GRADIENT_RECORD；结束标记：MOE_TRAIN_GRADIENT_COMPLETE。
同时确认启动器DUAL_RETRY_END的status=0及完整result.json，不只看tmux存在。

## B：完整PPT能量重试，不把普通refiner收益算作创新成功

旧V2确实做过候选排序与1/2/3步梯度修图，不能说它从未被实现。
旧数值审计发现空间平均造成更新幅度随像素数缩小、局部质量标签没有detach；
但旧方向本身也差，所以只放大步长不够。

本次保留强Fidelity作为起点，冻结Noise proposal、step8530 residual basis
以及外部候选生成模型。只训练634032参数的候选相关能量网络，不重训完整EBT。

具体改动：

1. 在规范化残差坐标中学习修正方向，而不是直接对非常小的图像梯度统一放大。
   图像在[-1,1]；u=(candidate-Fidelity)/.02，能量含二次锚定项和学习的空间势能。
2. 对空间能量求和后求梯度，消除显式batch/像素平均稀释；
   固定步长.5、u限制[-4,4]、图像限制[-1,1]，完整训练三步轨迹。
3. 同一个能量网络看原图、proposal、Fidelity、当前候选、残差坐标和真实每图五级VLM条件。
   它同时接受全局/局部质量排序监督、修正梯度监督和三步MSE/LPIPS监督。
4. 训练候选含GT、观测、proposal、Fidelity、perceptual、半幅/过幅恢复、
   三个冻结外部模型输出、重复恢复以及三步能量输出。
   候选好坏由训练GT实测，不按“错误专家”等名字强制赋予优劣。
   所有排序质量标签detach，不允许通过改变标签来降低排序损失。
5. 推理不用GT；同一个能量在未修改Fidelity与1/2/3步结果中选择。
   初始学习势能为零，必须严格返回Fidelity；相同分数优先保留Fidelity。

soft no-harm只是惩罚变差，不保证每图安全；最低能量也不天然等于最好画质。
修正方向、数值幅度和实际PSNR/LPIPS都要分别核验。即使运行正常仍可能科学失败。

### 数据、规模与验收

- 训练：54585条Noise记录、1755个来源组，来源均衡采样，固定256 crop。
- 单节点8卡：micro-batch=1、accumulation=4、global batch=32。
- 25步资格：使用完整训练池抽样；开发检查只取原固定256中的前32张。
  用于检查二阶反传、显存、梯度、checkpoint及1/2/3步评估，**不作为完整科学门槛**。
- 正式预设8530步，在1706/3412/5118/6824/8530步评估固定256张。
  这是约五次等效数据曝光，不保证均衡采样中每条记录恰好出现五次。
- 资格权重不得初始化正式；正式从零energy开始，并要求代码/配置/权重绑定
  匹配的成功资格receipt。当前启动器只跑资格，不自动启动正式。
- 保存energy、Adam和各rank RNG/采样位置。当前没有resume CLI；
  中断后必须做真正恢复实现，不能改总步数冒充续训。

资格预计.25–1.0小时，2–8 allocated GPU-hours。
正式8530步耗时必须根据本次实测速度与完整开发评估成本推算；尚不报价或启动。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && bash smoke/start_foundir_dual_retry_v1_tmux.sh ppt qualification_v1
```

日志：logs/foundir_dual_retry_v1_ppt_qualification_v1.log。
结果：checkpoints/foundir_dual_retry_v1/ppt/qualification_v1/result.json。
进度标记：NOVEL_PPT_TRAIN、NOVEL_PPT_VALIDATION；结束标记：NOVEL_PPT_COMPLETE。
同时检查DUAL_RETRY_END status=0及checkpoint绑定。

### 科学判断必须分开

所有评估分别列出proposal、Fidelity、energy一步/两步/三步、自主选择结果。

- 绝对目标：PSNR>=43.424160178005695，LPIPS<=.045362067760033824。
- 能量贡献：对同一批图，相对匹配Fidelity，PSNR严格提高且LPIPS不差。
- 原Fidelity的43.39715355634689/.028955043449059303是旧普通refiner结果，
  不是这次energy的成绩；新运行须重新核对匹配基线，不能直接贴旧数代替测量。
- 完整256张上的正向结果仍不等于论文创新已成立；跨退化和消融尚未完成。
- 资格32张的正负变化都不能后验替代正式256张结论。

## 已完成验证与边界

- 7项CPU单元测试全部通过：初始完全恒等、三步二阶反传且能够更新、
  ranking标签detach、batch复制不稀释更新、仅train来源选择、
  单节点/无自动正式约束、梯度探针识别相反方向。
- 真实双CPU-rank Gloo验证通过：完整三步、2步优化、累积2次；
  两rank的非零更新权重逐位一致。仅合成输入，未用数据图片。
- MoE真实预检查通过：绑定最终checkpoint、FLUX base/AE/prompt缓存/Rain初始化、
  训练来源与旧配置，零图片打开。
- PPT真实预检查通过，专用环境普通Python启动也通过：
  54585条/1755来源/256开发，权重与manifest绑定正确，零图片打开。
- train/validation来源隔离，并检查train GT hash不与validation/internal元信息重叠。
  只扫描internal split元信息，不访问其像素。
- Bash语法与非法参数拒绝检查通过。首次CUDA显存、真实LPIPS二阶反传、
  8卡通信和吞吐仍待GPU资格，不以CPU结果代替。

每任务使用一台同型号8×H100/H200。若只有一台空闲节点，先后执行；
两台空闲节点可独立并行，禁止两任务在同节点争抢八卡，不做跨节点DDP。
启动器有节点锁、GPU型号/数量/显存空闲检查、180秒有界NCCL测试、
新ID防覆盖；tmux创建成功不等于训练成功。输出和日志保留，不覆盖旧失败运行。

## 冻结身份与提交范围

配置SHA-256：
69db2e2f8a0e2dd3675bb7999968d4109ac4abb708151f909045dfb8578426ef。

代码身份在每次launch.json/result.json记录；正式须匹配成功资格身份。
新文件只包含energy模块、两GPU任务、共用运行约束、两个启动器、配置、
单元/双CPU-rank测试及本报告。旧绑定训练脚本不改。
AGENTS与handoff更新当前优先级。通过隔离Git worktree进行focused提交；
不提交权重、图片、数据、日志或大型manifest，不清理主目录脏工作树。

第三方依赖不随本仓库提交：third_party/flux2被原.gitignore排除。
它是官方FLUX.2源码，固定revision为50fe5162777813d869182b139e83b10743caef15，
现有离线修改记录于已跟踪的patches/flux2-offline-smoke.patch。
本次用到的autoencoder.py/model.py/sampling.py/util.py未相对该revision修改；
其SHA-256依次为：

- bb578f78d097a88f6e7e8aa01032c10f55659729b85203ae2d5852a7889ac96a
- e062691d789ec80a04a5fbc421b09d3b7e264305be1dfcb6da65d68c259e18c1
- f1a1e89651d450c05ab87840ed65011a2d25b83d811c61853ec6ed38e938d640
- ab39e974b1f696f9d51cb64eef9d82086c683771ac63e7ef14b30cced4b089e0

隔离Git worktree不含该外部checkout，因此只用于focused提交，不能直接当GPU工作根。
新节点若共享当前HDD3目录无需重新下载；全新clone仍须恢复这些外部依赖和授权数据/权重。
