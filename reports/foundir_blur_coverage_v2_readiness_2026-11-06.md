# Blur：固定训练曝光量的1080p来源覆盖对照

最新delta200在完整官方150上21.153159/.225559，旧50步为21.546762/.202622；均未达25.9/.18。固定8训练图缓存/重算/训练/推理模式全部逐值相等，排除所测路径差异。训练侧仅相对LQ+.278510dB、LPIPS仍不如旧模型，尚不能定性纯过拟合或证明单一数据根因。

第一性原理假设：在128个固定裁图、混合空间尺度上反复学习，不足以让32,440,320个原生LoRA参数形成适用于官方1080p输入的修正方向。本轮只改变训练数据人口，不追加旧失败模型的训练：1024个不同1080p训练来源、每来源一个预先固定512裁图，原Blur step10000/fresh Adam重新开始200步。输入原尺寸与来源多样性共同变化，所以不能分离这两个因素的因果贡献；未证明会改善。

清单来自已有合法`manifests/foundir_blur_heavy_training_v1.csv`（SHA072dcca8f80d09713fbc43e68ed1b9cb19cd20636ce4b3c70bdcc809e51b6212），仅train/01Blur/1920×1080。复用已有官方来源排除组和官方600的文件哈希做防重叠，不用官方指标挑来源，不解码官方GT。过滤后2768来源可选；按固定SHA顺序选择1024组及各组一记录，固定随机裁图；不按模糊强度/GT误差筛图。旧权重历史曝光不被抹去，源排除不保证语义完全分离。

保持v1数学、原生50处LoRA、FP32可训练LoRA/Adam/latent/VAE、BF16 FLUX、native CFG1/reference10、两次FLUX预测差推理、损失10*MSE_RGB01+.1*pyiqa LPIPS、lr1e-5、global8、clip1及固定200步。不增加CNN、RGB refiner，不启动Restormer或energy。共1600训练曝光，与前轮相同；每来源平均约1.56次而非12.5次，欠拟合是明确可能性，不以未通过推断所有数据扩展无效。

不重跑旧资格/旧训练/路由。复用已实测的相同数值实现CUDA资格；新数据在训练前仍检查固定前8图的初始化float/uint8及训练/推理一致性，每步检查有限非零梯度、参数及rank一致。每完整步原子保存模型/Adam/两rank RNG/采样位置/SHA；损坏拒绝、同world恢复，旧资产全保留。代码测试核对数值函数及官方推理与原版AST完全相同，另验来源排除/顺序稳定性/1024采样全覆盖。

只用node3双H100。预检2048文件SHA及裁图、原模型/提示词和完整pyiqa身份通过，35.119302秒、退出0/GPU0；5项CPU测试通过。新1024缓存及训练尚待GPU，以共享队列实际开始为准。其它三台不提交任务。

固定链：新缓存1024 → fresh200 → CPU最终权重/Adam/RNG/哈希/201份receipt预检 → 完整官方150原尺寸生成 → CPU完整性/均值核验。仅最后step200，无其它开发/测试集gate、中间checkpoint选择或GT回退。复用冻结无GT路由，全部150由自身logits选Blur；同512/128重叠/sqrt-Hann拼接/完整RGB pyiqa。完整150同时PSNR≥25.9且LPIPS≤.18才通过；工程退出0不能代替质量达标。官方反馈反复用于开发，明确非独立盲测。

源码发布核验后派发：`smoke/watch_foundir_blur_coverage_v2_cpu.py monitor --commit <SHA>`；独立upload进程推首次训练、50/100/200步和终态小报告，不让网络延迟阻塞GPU。job为round10_blur_coverage_{cache,train,official}_v2；入口`smoke/run_foundir_fullstack_blur_coverage_v2_gpu.sh 3 <cache|train|official>`，仅供共享worker调用，勿人工并发启动。

全部费用持续记录，无累计GPU-hours上限；本轮固定200步+官方150，失败停链、无自动重试或扩大。沿用前轮速度，训练约19分钟、官方约16分钟；新缓存粗估4–8分钟，整体约40–50分钟，非保证。checkpoint约78GiB加新缓存约8GiB，均留共享盘。只推小JSON/Markdown及源码。Blur通过前不推进其它类别。
