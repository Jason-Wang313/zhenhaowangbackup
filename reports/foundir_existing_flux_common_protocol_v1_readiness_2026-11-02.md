# 下一步：共同提示词和输出处理，四节点H200

## 这次具体做什么

用户确认四节点均为H200，并在静态继承128/128通过后授权继续。现在执行原方案B，
不重训已有四类LoRA、不再重复单选等价、不启动旧retention/energy，也不直接长训Rain。
共同条件尚未确定时就训练Rain，会让新Rain适应一套可能随后被改掉的输入/输出规则。

冻结一个不含真实退化标签的共同提示词：

> Restore the reference image to a clean, sharp, clear, normally exposed image. Remove image degradations while preserving the exact content, geometry, identity, color, and texture.

输出候选先用最简单的alpha=1、一次uint8，也就是直接使用FLUX结果，不再按任务与LQ混合。
这不是宣告原alpha无用，更不是擅自把所有类型改成alpha=1上线；需要测出它对各类的代价。
不做多prompt/多alpha搜索，不训练辅助恢复器，不把四专家各生成一张再平均。

固定四组对照，所有组用相同图片、裁图、种子、权重和50步：

| 组别 | 提示词 | 输出处理 | 用来回答 |
|---|---|---|---|
| 原方案 | 各专家原提示词 | 原alpha/原量化 | 保留原分母 |
| 只换提示词 | 一个共同提示词 | 原alpha/原量化 | 提示词是否损害已有能力 |
| 只换输出处理 | 各专家原提示词 | alpha=1，一次uint8 | 原来的输入混合有多重要 |
| 两者都换 | 同一个共同提示词 | alpha=1，一次uint8 | 最简单共同接口能否直接使用 |

Haze本来就是alpha=1一次量化，所以对应两种输出处理应一致；它是自然的内部检查。
原专家仍由任务身份固定选择，这只是接口诊断，不能说已经实现不靠标签的部署router。
共同prompt本身不按GT标签或GT质量变化。没有逐图选择最好的组，没有GT控制生成/接受/停止。

## 复用什么，额外算什么

父任务固定为`replay_4x2_v1`，报告为`foundir_existing_flux_replay_4node_terminal_2026-11-02.json`，
报告SHA `be18efd0610848e23b4d589c667072429f0a81116073364f9a3429e943c4d22f`。
直接读取父任务保存的128份原始decoded/uint8和原单选收据，不重跑原prompt采样。
新生成128份共同提示词结果，四个分支权重全部只读；每张仍只静态选择一个专家。
三种变更组通过这两种decoded结果和两套量化/混合规则计算，不需要额外三次FLUX采样。

Noise/Blur/Low-light保留“先量化FLUX，再与LQ按原alpha混合，再量化”的原对照顺序；
原alpha分别.30/.75/.925。Haze原路径不做第二次混合。先重算原处理并检查与保存uint8相同，
不悄悄换量化或降低原分母。原结果、Tensor、收据及源码保持不动，新输出目录完全隔离。

## 数据及指标权限

同一批四类各32条训练记录，共128条；继承原source、crop_xy、inference_seed、512裁图、
50步、CFG1和reference time10，样本选择摘要不变。
本轮为了判断共同条件的质量影响，新增读取**这128条同一训练记录的配对GT**，只用于评分。
manifest逐字段匹配父记录、核验GT哈希和Train/category/GT路径，不能拿其他GT或开发图替代。
正式/封存/internal/parent-test及最终1859 Noise开发像素全部不读。
这些来源可能已被原FLUX训练见过，所有数值标为training_fit_only，不能当来源分离泛化或SoW验收。

指标固定为uint8/255的RGB float32 PSNR（MSE≤1e-12时100dB）和torchmetrics AlexNet v0.1
LPIPS（normalize=True、float32、无autocast）。每张图每组各算一次，按任务32张算平均值，
同时报告相对原方案的PSNR/LPIPS差值和受损图片数量，不混合不同组的最佳数值。
不更改后续正式Rain的≥34.2/≤.08要求；本训练子集不能代替Rain100图验收。

AlexNet离线权重SHA及LPIPS head SHA绑定；四节点metric包版本、LPIPS实现源码SHA必须相同。
LPIPS只是评分器，不是CNN恢复网络或训练教师。生成结束后先释放FLUX/VAE，再加载LPIPS，
不让评分网络与137.7GiB的采样峰值同时占GPU。新生成阶段没有GT参与。

## 结果如何解释

完成状态表示128条×4组的指标与文件身份齐全，不代表共同方案已经通过。
每类每组报告描述性的均值不退步（数值容差PSNR1e-6dB、LPIPS1e-8），并报告单图受损数。
这不是统计显著性、泛化no-harm或新的科学gate，不自动挑配置、调阈值或启动Rain。

- 如果只换prompt明显伤害某类，先处理该类的条件兼容；不能假设已有LoRA可无损换prompt。
- 如果prompt基本保留但alpha=1明显伤害某类，保留这一事实，下一步处理输出协议，
  不能用训练掩盖、不能擅自把按router平均alpha当作已验证方案。
- 若同一个共同方案保留四类能力，再明确冻结它并交付Rain25步多步适配工程资格。
  Rain仍从成功一步权重的独立副本初始化，fresh Adam；不覆盖原一步Rain。

## 四节点入口、恢复与预算

唯一GPU根为`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
必须四节点×每节点2张H200，总8卡；每卡可见内存≥139GiB。此新入口不接受H100，
用户已确认资源，不为H100另行启动优化实验。共享父输出必须在四节点可见。

新入口`smoke/start_foundir_existing_flux_common_protocol_v1_tmux.sh`，仍为四个必填参数：
`<node_rank 0..3> <当前node0地址> <run_id> <本次共用fresh_launch_id>`；默认端口29682。
建议新run ID `common_protocol_4x2_v1`；目的地代理确认当前地址后给四条命令，主代理不猜地址。
默认Python为新根`.conda/gpu-train-stable/bin/python`，须由目的地逐节点检查。

各node runtime须匹配父任务对应node的GPU系列/软件版本/驱动；否则不能直接复用原分母，
应报告差异而不是删断言。跨节点GPU型号已固定H200。GPU卡数分片、随机种子、保存不变。
沿用已验收的fresh launch标记、四节点等待、失败联动、本机锁和每node共享锁；
与旧replay共用本机锁避免重叠运行，新任务输出目录和日志名独立。

逐图不可变保存共同采样及评分，same run ID + 新launch ID验证后跳过完成项。
生成和评分分开：中断后已生成的图不必再次跑FLUX。GT不保存到新缓存，不上传图像。
四个节点done后node0校验原/新tensor SHA、指标身份、完整128记录并重算任务均值。
这是文件/身份与汇总复算，不冒称独立再跑过全部LPIPS。目的地可另行只读复算作审计。
终态需`COMMON_PROTOCOL_RESULT_VERIFIED records=128`和四份`COMMON_PROTOCOL_NODE_END status=0`。

父任务实测9分52秒，当前只有一轮新采样但新增评分；粗估15–30分钟、2–4 allocated GPU-hours，
不是实测ETA。每节点硬cap1小时，总8GPU-hours；另30秒子进程清理及最多20秒失败标记写入。
没有自动恢复、自动增预算或自动正式训练。四份每node2×耗时相加报告成本，失败尝试不清零。

## CPU交付与目的地同步

45项CPU回归全部通过，测试体92.814秒（不含Python/torch导入）；shell语法和Python编译通过。
源资产只读预检通过：同128条训练LQ/GT元数据，选图摘要未变，39项源码绑定复核一致。
详细小报告见同名前缀`_cpu_preflight_2026-11-02.json`；不把CPU通过说成新GPU任务已通过。

本地只在隔离worktree实现/测试、只读旧资产；新根运行缓存不在本会话，不能假称GPU已启动。
本地preflight检查原权重、已核验审计、同128条训练LQ/GT元数据、离线AlexNet，不解码训练图。
它必须明确标记`parent_runtime_artifacts_verified=false`；不能替代新根实际缓存检查。
目的地安全同步后，用新脚本的`preflight --require-parent`检查父任务全部共享缓存收据、
source/asset绑定，再给用户四节点命令。保留本地脏文件、数据、权重和历史日志。
CPU测试模块为旧三个`test_existing_flux_*`及新`test_existing_flux_common_protocol_v1`；
还需新shell语法和Python编译检查。正式跨节点GPU资格尚未发生。
回传小JSON报告、每类四组均值、逐图差值/身份、四节点状态和成本；不上传权重/缓存/实时日志。
