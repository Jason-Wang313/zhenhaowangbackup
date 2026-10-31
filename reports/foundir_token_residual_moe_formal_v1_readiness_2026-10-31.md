# 右图③ VLM residual-router MoE：1000步正式训练交接

## 已批准什么

用户在25步资格完成后批准正式训练。此文覆盖旧资格readiness中的“正式规模未定”。
本次主代理交付代码、测试和GitHub同步说明；没有在本会话启动GPU。
目的地是共享目录
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
四节点各两张H100/H200，共八卡；节点内同型号，跨节点允许混用。
不要改成单节点八卡、三节点六卡，或者沿用历史master IP。

25步资格在8×H200上status0、峰值42.644GiB、训练中位5.946秒/步，
但Rain与Blur preview均没有联合指标改善。资格证明实现可运行，不证明VLM有益。
报告及完整证据见同日qualification_4x2_terminal的md/json，源码提交528383c。

## 不改变科学配方

从原image-only MoE step11000初始化，VLM投影末层归零，fresh Adam。
不能从25步资格权重继续，也不恢复旧image-only optimizer。
保留原五块FLUX FFN专家、图像router、共享LoRA；冻结Qwen视觉token经小MLP，
加到image-router logits后softmax，同一组专家接受加权混合。
不是VLM生成语言标签；这是视觉merger token均值。模型详细定义见旧资格readiness。
专家、image router、共享LoRA、VLM投影共同训练；不新增CNN、Restormer或energy。
仅做③，不同时训练①image-only或②拼接对照，不触碰另一代理的energy显存工作。

正式配置以hash绑定的资格配置为底，只允许调度/运行相关字段覆盖。
Rain/Blur训练池仍为38334/105057条，来源隔离、512裁块/D4增强不变。
global batch32、每步Rain16/Blur16，学习率与损失保持不变：
expert/shared LoRA 1e-5，router/VLM 1e-4，Rain LPIPS .05、Blur .20。
训练1000步共32000次曝光，不是32000张不同图片，不是完整训练池的一轮。
采样位置从352000开始，末次383999，与25步资格相同起点但独立权重/Adam。
GT只用于训练损失和开发指标，不输入VLM/router，不使用受保护测试集。

## 评估和成功标准

| 步数 | 固定开发评估 | 用途 |
|---|---|---|
| 0 | Rain100＋Blur2595 | 同环境完整起点，不用历史预览冒充完整分母 |
| 250、500 | Rain100＋Blur preview100 | 中途趋势，不宣布完整Blur过线 |
| 1000 | Rain100＋Blur2595 | 正式终态，与step0及历史基线分别对照 |

完整评估还汇总其中的固定Blur preview100，不重复推理。
缓存只存一份canonical开发记录；预览直接引用相同特征绑定。
绝对门槛：Rain PSNR≥34.2/LPIPS≤.08；Blur PSNR≥25.9/LPIPS≤.18。
历史image-only完整指标Rain34.414163/.064399、Blur26.619521/.188334单列保留。
本轮即使过绝对门槛也不能独自证明VLM贡献：原专家等参数也在继续训练。
程序明确输出VLM_advantage_proven=false；不能用单组改善代替后续匹配消融。
不自动增加步数、不从预览调学习率或损失、不自动开展新分支。

## 保存、恢复和预算

每100步保存完整训练参数、Adam和8rank RNG，250步额外保存；一共11组。
resume收据最后写出，含文件SHA、采样进度、配置/源码身份；partial不能当断点。
必须同模式、同配置/源码/数据/拓扑，qualification收据会被formal拒绝。
中断后使用新run ID，第四参数指向最近完整formal resume.json。
基线引用始终保留第一次正式step0的路径及SHA，不把恢复点当成更弱的新分母。
若1000步已保存但最终评估中断，可以从1000步收据恢复，只补最终评估。
权重/Adam/RNG状态能精确恢复；重新创建DDP后继续计算不保证逐位相同。
保留旧运行目录、日志、缓存和全部权重；不覆盖、不自动清理。

特征阶段先独立提取34695份缓存：32000次训练裁图＋2695开发裁图。
Qwen进程结束后才加载FLUX，避免两模型同时占显存。资格缓存不重命名冒充正式缓存。
相同正式配置、绑定和裁图的已完成缓存可在重试时复用并核验。
新训练预检至少160GiB可用空间；不足则停止，不删除旧资产腾空间。

粗估2–3小时、16–24 allocated GPU-hours，并非实测保证。
其中1000×5.946秒≈1.65小时只是训练，不含特征提取、完整评估、存盘与启动。
H100/混用节点受最慢rank及网络影响，需按feature与train日志更新预估。
每次四节点启动各自4小时硬cap（含等待和两个阶段），最多约32 allocated GPU-hours，
达到上限会停止，不自动延长；按最后完整收据人工决定恢复。
终态报告同时记录各节点完整wall time与allocated GPU-hours，不能只报训练进程耗时。

## 本次核验范围

17项CPU测试通过（旧11＋新6）：加性路由、零初始化、条件梯度、任务均衡、
formal配置禁止暗改科学参数、资格证据绑定、canonical缓存、模式隔离和最终步恢复。
真实8进程Gloo formal测试通过：权重/Adam/RNG逐位恢复；继续计算最大参数差
7.450580596923828e-09，通过atol1e-7/rtol1e-6；串行batch32对照也通过。
Python编译和shell语法检查通过。
旧目录实际manifest/step11000收据的只读元数据核验通过，未读取图片像素：
训练池Rain38334/5780来源、Blur105057/7339来源，与资格data binding一致；
正式32000次曝光覆盖上述全部5780/7339来源，34695份spec无重复cache key，
预览200、完整2695条。资格报告绑定的科学源码一致（本轮helper/trainer编排除外）。
该检查使用隔离工作树中的临时只读资产链接，不提交链接、manifest或第三方代码，
不代表目的地模型大文件或空间已核验。
这些是CPU工程检查，不是本次1000步CUDA验收或科学结果。
主代理没有目的地挂载，不能声称目的地磁盘/资产/GPU已检查。

## 目的地CPU代理操作

先fetch origin，安全同步以下白名单；本地dirty文件逐项合并，禁止reset/clean/整仓覆盖。
已存在的qualification配置/模型/数据/FLUX/Qwen科学代码必须仍与资格报告hash匹配，
本轮只允许helper/trainer改变编排；不为绕过不匹配而更新报告或关闭检查。

- `configs/foundir_token_residual_moe_formal_v1.json`
- `smoke/token_residual_moe_v1.py`
- `smoke/train_foundir_token_residual_moe_v1.py`
- `smoke/run_foundir_token_residual_moe_formal_v1_4x2.sh`
- `smoke/smoke_token_residual_moe_cpu_ddp_v1.py`
- `tests/test_token_residual_moe_formal_v1.py`
- 本报告、AGENTS.md、agents.md、`reports/foundir_current_handoff_2026-09-23.md`顶部新节。

先确认25步qualification结果json/step25收据与GitHub已审计报告一致，
原step11000权重/收据存在，原FLUX/Qwen/manifest资产可访问，空间足够。
用已验证的项目Python运行CPU预检（不是GPU启动命令）：

```text
smoke/train_foundir_token_residual_moe_v1.py --config configs/foundir_token_residual_moe_formal_v1.json --phase preflight --run-id formal_4x2_v1
```

上述参数需跟在已验证Python解释器之后，不直接当shell命令；不得用系统Python替代。
通过后获取当前四节点地址/空闲GPU/网络，给用户四条tmux启动命令，不擅自启动。
正式launcher参数是node_rank(0/1/2/3)、当前node0地址、共同run ID、可选resume收据。
默认端口29762，和资格29761区分；所有节点PROJECT_PYTHON版本必须一致。
正式授权已经给出，不必再要求重复25步资格或重复申请1000步规模。
实际运行后回传小JSON/Markdown日志摘要和必要终态证据到GitHub，
不上传权重、缓存或实时日志大文件；由主代理判断科学结果和下一步。
