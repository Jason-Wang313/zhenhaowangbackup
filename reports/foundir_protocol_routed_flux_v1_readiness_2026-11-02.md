# 保留原专家能力：路由与原输出处理绑定

## 这次要解决什么

前两轮已经说明两件不同的事：

- 四份已有LoRA确实装对了。原50步入口和静态单选bank的128/128输出完全一致。
- 取消各专家原来的输出处理会明显变差。仅统一alpha=1，Noise约掉4.98dB，
  Low-light约掉.91dB，Blur约掉.83dB。这不是权重没继承，也没有发生训练遗忘。

因此本轮不重新训练四个专家，而是保留原来的使用方法，把它接到内部路由接口。
共同prompt也没有证明四类两指标全部不退步，所以此轮一起保留原prompt。
用户授权execute；本轮是新接口工程资格，不是正式训练。

## 接口怎么工作

输入是一张LQ图和一组明确的专家比例，例如Noise .9、另三份各.1/3。

1. 比例最高的专家作为主专家，选择它原来的prompt和最终输出处理。
2. FLUX内部所有原50个LoRA位置都按整组比例混合增量：
   `base(x) + sum_i weight_i * B_i(A_i(x))`。不平均A/B因子。
3. 只跑一条50步生成轨迹，输出一张图，再执行主专家原输出处理。

| 主专家 | 原alpha | 原量化顺序 |
|---|---:|---|
| Blur | .75 | FLUX先uint8→与LQ混合→再uint8 |
| Noise | .30 | 同上 |
| Haze | 1 | 一次uint8，不混LQ |
| Low-light | .925 | 双uint8 |

这是内部混合，不是先生成四张恢复图再平均。比例会改变FLUX内部计算，但不会把alpha
按比例平均。平局固定按blur/noise/haze/lowlight顺序选择。

代码用显式输入绑定每次生成的比例；调用结束恢复未绑定标记，缺失比例立即报错。
CPU测试包括CFG批次顺序、相邻调用无串扰、checkpoint重算时路由梯度一致。
`probabilities(image_logits, vlm_delta)`只实现分数相加再softmax的接口；
**尚无新image-router或VLM投影器训练。本轮手工比例不能冒充完整右图已学会。**

这也不是最终已经验证的混合退化方案。主专家切换处不连续，原alpha可能压低其他专家
的作用，soft混合可能破坏画质；本轮先确认单选保留和真实混合执行，全部退步如实记录。
不会用GT挑主专家、比例、alpha或prompt，不会以“固定.9没有提升”直接否定所有路由。

## 本次GPU范围

从父任务每类固定前两条取8张训练图，不根据父指标选图，每rank一张，保持原rank归属。
只做两个固定组：

- one_hot：新显式路由选中原专家。decoded最大差≤1e-6且uint8逐像素完全相同才通过。
- near_one_hot：主专家.9，其他三个各.1/3。检查真正内部混合可执行、数值有限、协议一致。

两组均固定原50步、512裁图、seed、CFG1、reference time10、四份原LoRA与原base/VAE。
原输出直接复用`replay_4x2_v1`的对应tensor；原四条prompt缓存也复用，不重新加载Qwen。
先保存两组输出，再读对应训练GT评分；只读8条已授权training GT，所有保护像素仍禁止。
报告原图、单选、soft各自完整两指标，每类只有2张，仅描述工程影响，不是SoW/泛化门槛。
恢复模型仍只有FLUX及原生VAE；LPIPS的AlexNet只在FLUX释放后加载评分。

父证据：

- `reports/foundir_existing_flux_replay_4node_terminal_2026-11-02.json`
  SHA256 `be18efd0610848e23b4d589c667072429f0a81116073364f9a3429e943c4d22f`。
- `reports/foundir_existing_flux_common_protocol_v1_terminal_2026-11-02.json`
  SHA256 `cf398ba37f6d73a0696f77e816ca622d5a494cd12fe089d2681ea2406a3251ae`。

## 资源、时间、恢复

4节点×每节点2张H200，总8卡，每卡至少139GiB。上轮静态峰值约137.74GiB；
**新soft路径的真实峰值尚未测量，不能借用旧峰值声称新任务已通过。**
本入口不支持H100，不以旧兼容承诺代替这轮真实显存要求。软件、GPU系列及driver
必须匹配父任务的对应节点，原分母不能悄悄换环境。4节点需共享迁移后的新根。

预计5–15分钟，合计.67–2使用GPU-hours，仅是估计；每节点硬cap1800秒，
四节点合计最多4使用GPU-hours，另计清理。不能把每个节点算成8卡。
超过预算不自动重启、加预算或降裁图。建议至少4GiB新空间，旧文件不删除。

每个arm输出独立保存tensor、SHA、原图身份、比例、协议和时间，完成后才发布不可变JSON。
中断孤立文件保留。恢复沿用run_id、换fresh共用launch_id；检查已提交项后跳过，
不覆盖失败单选记录、不把坏记录当完成。四节点失败联动，只停自己的launcher子进程。
逐图工作无需DDP梯度同步，torchrun只分发8个固定rank；四节点完成由共享目录barrier保证。

终态必须同时有8张单选复算通过、16份新输出完整、8份三组指标、四节点done、
`PROTOCOL_ROUTED_RESULT_VERIFIED`及四份`PROTOCOL_ROUTED_NODE_END status=0`。
汇总会重读tensor核验单选，而非只信pass字段；不会独立重跑LPIPS。
近单选质量变差不伪装成质量成功；工程PASS只表示等价性和混合执行通过。

## CPU／目的地交付

本地61项CPU回归全部通过，覆盖显式路由、精度、重算梯度、四节点barrier、
错误收据拒绝以及从保存tensor重新核对单选。5个Python文件语法和2个shell检查通过。
源资产预检通过，47份源码绑定在最终生产代码上重新核对一致；8条来源摘要为
`978a9c299b47acdb4ee1e3881b518b03d8563f57d38e445c99d1eee9cab9fdbb`。
没有解码任何真实图，没有新GPU指标。小JSON记录准确边界，
`parent_runtime_artifacts_verified=false`明确说明主代理看不到目的地父缓存。

代码和测试在隔离worktree实现，不修改父实验绑定文件，不动另一代理energy。
唯一GPU根：
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
旧agent-driven-world-model目录只读核查资产，不能启动GPU。

目的地代理从`origin/main`安全同步本次明确文件，保留本地修改、权重、数据和缓存。
先读AGENTS.md顶部；本地如有冲突，逐项比对，不reset/clean，不全目录覆盖。
使用已验证的`.conda/gpu-train-stable/bin/python`执行CPU预检：

```bash
.conda/gpu-train-stable/bin/python smoke/qualify_foundir_protocol_routed_flux_v1.py preflight --require-parent --report reports/foundir_protocol_routed_destination_preflight_v1.json
```

此步只校验代码/权重/数据元信息/父缓存，不解码图、不用GPU。`--require-parent`不可省略，
本地主代理只读源资产预检不能代替目的地检查。依赖和当前4节点环境随后由GPU入口检查。
通过后由目的地代理取得**当前**node0地址，给四条分节点命令：
入口`smoke/start_foundir_protocol_routed_flux_v1_tmux.sh`，四参数为node_rank、当前node0地址、
run_id、fresh共享launch_id，默认端口29683。不得复用历史IP，主代理不编造启动状态。
GPU尚未启动。完整结果后只回传小报告和失败摘要，禁止上传模型、图片、tensor或live日志。

通过后再结合soft结果交付独立Rain25步适配资格；此时Rain仍是原一步endpoint，
尚未放入50步flow bank，也尚未实现本方案下的Rain训练器。没有自动正式MoE、长训或energy任务。
