# 五类单退化 VLM-token MoE：25步工程验收准备 — 2026-11-01

## 结论与下一步

用户已批准扩展当前右图③模型。路线是：**五类接入验收 → 五类适应训练 →
五类single与现有五种double、一种triple联合训练**。不是逐类另训五个独立专家，
也不先把所有组合同时塞进来。先接入单类是为了分清失败来自新增任务本身，还是混合任务。
无需等五类全部过SOW才能进入混合阶段，但应先确认新增任务能学习、旧任务没有严重退化。

本次只交付第一步：25个新优化步，五类都参与；CPU检查通过，**GPU尚未启动**。
正式训练步数、每类曝光量、评估点与预算将在本次实测后单独冻结，不自动长训。
25步不能证明五类科学达标，更不能证明VLM独立贡献。用户暂不做image-only消融；
保留旧权重、数据身份及配方，后续仍可做匹配对照，不以历史不匹配对照替代它。

## 起点与不变的模型

成功起点是 `continue_4x2_v1` 的全局step2000，不是原step11000或25步资格权重。
其完整开发结果：Rain100为34.610111/.064487，Blur2595为26.755594/.176108，
已过Rain≥34.2/≤.08与Blur≥25.9/≤.18。原独立50步Blur仍为26.559141/.151993；
当前一步MoE的PSNR更高、LPIPS仍更差，不能说当前全面超过独立Blur。
已完成报告：`reports/foundir_token_residual_moe_continue_v1_terminal_2026-11-01.json`。

模型保持五个FLUX double block的FFN、每处两个专家。**五种退化不等于五个专家**；
专家是共同学习的内部网络，不是各自对应一个类别的五份LoRA权重。
原图像router保留；冻结Qwen2.5-VL及V8 adapter，对同一恢复裁图的视觉merger tokens
取3584维均值，经每块LN→Linear(3584,256)→SiLU→Linear(256,2)，
加到图像router的两个分数，再softmax混合专家。

FLUX专家、image router、共享rank32 LoRA、VLM投影都继续更新；Qwen冻结。
仍为t=1、无anchor noise、一步端点恢复，沿用空文本context。
不加载Noise/Haze/Low-light的旧独立LoRA来拼成专家库，不加CNN、Restormer或energy。
FLUX原生VAE、固定LPIPS评价/损失网络照旧。另一个代理的energy显存工作不在本次改动范围。

## 这是新任务阶段，不冒称原两类配方原样续训

恢复step2000的全部训练权重、Adam状态和8rank RNG，不重置VLM、不用fresh Adam。
全局Adam步2000→2025，只有25个新步。两项有意变化：

1. 从Rain/Blur采样切换到五类轮转、类内来源均衡的新采样计划，seed=2026110105。
   global batch仍32（8rank×4累积），每个global microbatch为同一任务，五类轮转。
   每5步各类32次曝光；25步各160次，总800次。曝光不保证800张互不重复的图。
2. 去掉历史Rain/Blur二分类router CE（原权重.01）。五类标签不能监督两个专家的二分类输出。
   新路由只由恢复损失训练；GT类别只用于采样和选择任务损失权重，绝不作为router输入。

Rain LPIPS权重.05、Blur .20保留；新增三类使用原通用.05。
其余恢复损失和学习率不变：expert/shared LoRA 1e-5，image router/VLM投影1e-4。
训练仍512裁图和D4增强；小于512的训练图沿用原反射填充，不把其放大伪称原生高分辨率。
开发仍512固定裁图，不resize；指标沿用uint8量化后RGB PSNR与Alex LPIPS。

## 数据身份与预览范围

只读取批准的MillionIRData/Train集合中的train与validation像素。
扫描所有11类别的split/source/hash元数据，排除所有非train来源与相同GT哈希；
internal_test只读取隔离所需元数据，不打开其像素。
若父Rain/Blur已训练来源与新增开发来源重叠，直接拒绝启动，不能靠删新训练记录掩盖污染。
本地元数据审计通过，重叠剔除记录数均为0，父两类数据身份不变。

| 类别 | 可用训练记录 | 训练来源 | 完整开发记录/来源 | 本次固定预览记录/来源 |
|---|---:|---:|---:|---:|
| Rain | 38,334 | 5,780 | 100/100 | 100/100 |
| Blur | 105,057 | 7,339 | 2,595/165 | 100/100 |
| Noise | 54,585 | 1,755 | 1,859/34 | 100/34 |
| Haze | 76,650 | 7,642 | 1,580/158 | 100/100 |
| Low-light | 37,295 | 723 | 1,400/21 | 100/21 |

训练池合计311,921条，但本次仅抽取800次训练曝光，不能说跑完31万条。
Noise/Low-light的100张预览分别只有34/21个来源，不是100个独立来源。
Rain/Blur预览沿用父任务；新增三类按ID哈希排序、来源轮转选100，不根据画质挑图。
起点step2000和终点2025各评估同一500张，不能用此预览宣称完整五类门槛通过。
这不是另跑image-only对照，而是记录同一模型在接入五类前后的结果。

新增三类保持历史完整开发裁图公式与seed，并将其协议SHA写入新配置：

- Noise：`configs/foundir_noise_flux2_scalar_residual_full_development_v1.json`，seed2026091602，参考门槛≥38.9/≤.10。
- Haze：`configs/foundir_haze_flux2_step3000_full_development_v1.json`，seed2026091702，参考门槛≥22.1/≤.19。
- Low-light：`configs/foundir_lowlight_flux2_scalar_residual_full_development_v1.json`，seed2026091602，参考门槛≥18.9/≤.28。

旧配置里的独立专家、固定输出混合等推理方法不随评估协议继承；这里只继承开发记录与裁图身份。
未来混合阶段仅有Blur+Noise、Low-light+Haze、Rain+Haze、Low-light+Blur、Low-light+Noise、
Low-light+Blur+Noise六类，即总共11个已有类别，不是所有数学组合。
本次没有打开混合数据像素，没有训练混合数据。原manifest的685,495条含train/validation/internal，
不能称为685k训练记录。

## 权重、缓存和中断恢复

配置：`configs/foundir_token_moe_five_single_qualification_v1.json`。
入口：`smoke/run_foundir_token_moe_five_single_v1_4x2.sh`。
唯一父收据：
`checkpoints/foundir_token_residual_moe_continue_v1/continue_4x2_v1/step_002000.resume.json`，
SHA `d5eb06cd745fd62741c0f7cc90ebd3c1538d53a57770acef0c6e7b1aada55174`。
父终态报告SHA `6f6ce25fdd485e15a9f231170b09e9dac8e9029733bd9315a0a01d39ff8c9032`。

验证父report/config/launch/result/receipt，CPU预检还hash父大权重与Adam。
父科学源码必须匹配，只有公用trainer编排文件允许本次明确修改。
先构建原模型与零接入检查，再强制恢复已训练step2000状态，然后评估/更新；
零接入检查不是将最终VLM权重清零。
此处从父formal状态显式进入新qualification阶段，后续恢复要求新代码/配置/数据绑定严格一致。

新缓存绑定本轮配置/数据/源码：800份训练裁图＋500份开发裁图，共1,300份Qwen特征。
不改名复用旧特征；Qwen提取与FLUX训练分两个进程阶段，先释放Qwen再加载FLUX。
本轮已完成且身份匹配的特征可在中断重启后复用。

每5步保存权重、完整Adam和8rank RNG，完成后有2005/2010/2015/2020/2025五份收据。
`stage_baseline.json`保留恢复step2000后的同环境预览；历史完整Rain/Blur结果仍独立保留。
首次启动不传第四参数，入口自动加载父step2000；中断后新run ID＋第四参数传本轮完整收据，
继续同一采样位置，不使用partial文件。若2025已保存但评估中断，只补评估、不增加训练步。
终态`steps=2025`为全局步数，`optimizer_updates_this_launch`才是该次新增步数；
中断续跑需合计各段曝光量，不把最后一段的曝光数误称整个25步。
本次资格权重不得初始化后续formal；formal应重新从成功step2000开始。

## 资源与已完成检查

默认4个节点×每节点2张H100/H200，共8卡；节点内同型号，跨节点可不同。
共享运行根为
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
不得复用旧node0 IP，不在旧agent-driven-world-model目录启动。
原验证Python环境若未挂载，必须先确认等效环境；不能静默换系统Python。

预算0.3–0.8小时（18–48分钟）、2.4–6.4 allocated GPU-hours；含提取、两次预览、存盘。
这是估计，非保证。父8×H200训练中位6.818秒/步、峰值42.660125GiB；
本次五类/H100或混用的速度与显存仍待测。每次启动总硬cap1小时，至少80GiB空闲，
不自动删旧权重/缓存腾空间。只锁本节点token MoE作业，不停止其他人的任务。

31项CPU unittest通过（原23＋新8），涵盖五类均衡/恢复采样、固定裁图、来源隔离、
父状态身份、禁止二类CE、恢复loss连接VLM梯度、formal→qualification保留Adam。
8进程Gloo batch32恢复回归通过，最大参数差7.450580596923828e-09，
满足原atol1e-7/rtol1e-6。Python编译、JSON解析、shell语法、diff检查通过。
这些不是完整4B CUDA五类验收，也不保证跨硬件逐位一致。

GPU验收须看四节点status0、25个新步无NaN/OOM、VLM分支有梯度且实际更新、
五类各160次总曝光、五份完整收据、两次500张预览齐全。
逐类报告PSNR/LPIPS及变化，不用平均数遮住Rain/Blur退步。
成功标记是`TOKEN_MOE_FIVE_SINGLE_QUALIFICATION_PASS`，只表示工程验收通过。

## 目的地CPU代理交接（同步后再给用户GPU命令）

安全同步origin/main的本次提交，保留本地dirty、energy改动与全部大资产。
不要reset/clean/整仓覆盖；冲突逐项合并。同步白名单：

- `configs/foundir_token_moe_five_single_qualification_v1.json`
- `smoke/token_moe_five_single_v1.py`
- `smoke/train_foundir_token_residual_moe_v1.py`
- `smoke/run_foundir_token_moe_five_single_v1_4x2.sh`
- `tests/test_token_moe_five_single_v1.py`
- 本报告、`AGENTS.md`、`agents.md`、`reports/foundir_current_handoff_2026-09-23.md`的顶部新节。

父cd4d322报告与631845f等依赖须已同步；GitHub不包含数据/权重/缓存，不用git同步替代资产核验。
在已验证的项目Python后接以下参数做CPU预检：

```text
smoke/train_foundir_token_residual_moe_v1.py --config configs/foundir_token_moe_five_single_qualification_v1.json --phase preflight --run-id qualification_4x2_v1
```

通过后确认四节点当前地址、空闲卡与连通性，再给用户tmux命令，不自行启动GPU。
入口参数为rank0/1/2/3、当前node0 IP、共同新run ID、可选本轮恢复收据；默认端口29764。
勿使用旧两类入口或手改其步数。不要自动formal、mixed、消融或energy。
完成后将小JSON/Markdown报告推送GitHub：四节点起止/状态、配置源码SHA、
两次五类逐图/汇总指标、实际曝光/更新数、checkpoint收据身份、峰值显存和完整墙钟/GPU-hours。
若有中断，保留各段日志及恢复链；大文件留共享目录。
