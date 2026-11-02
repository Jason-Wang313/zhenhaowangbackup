# Rain正式训练首段：完整训练池＋嵌入50/100/200步评价

## 当前决策，不再做什么

用户取消独立32图过拟合阶段，要求直接进入下一阶段训练并保留早期评价。
本轮是正式数据训练的前200步，不是把32张图重复训练200遍，也不是200步就应达标的承诺。
200步结束后保留完整可续训状态；是否继续及总预算由结果决定，不自动训练到几千步。
旧25步资格权重不复用；从原成功Rain step4000复制FP32 LoRA并fresh Adam。
这不是必须从零初始化的科学定律，而是保留原资格实验边界；正式状态将来可以真正接着训。

父资格四节点status0，25步完成；8训练图student为16.417211/.394619，teacher为33.696391/.034100。
工程通过但质量差距很大；不能用loss下降替代真实50步评价。旧驱动失败加资格累计
0.721666667使用GPU-hours保留。原权重、缓存、收据、代码与另一代理energy均不改。

## 数据与监督

- 原训练池38334条、5780来源；均匀选来源、再选该来源记录。
- 按绝对sample position确定512随机裁图、D4旋转/翻转与noise/t。35条小图reflect padding，
  不缩放图像。来源隔离沿用主manifest全部非训练source/GT哈希审计。
- 首200步×global batch32=6400次曝光，固定种子下为5012条不同记录、3867个来源；
  不声称覆盖全部38334条。训练池、采样表和裁图全部绑定。
- native noise-to-clean flow监督、原五项loss系数、lr1e-5保持资格方案不变。
  GT主要监督；原一步Rain FLUX只作弱RGB示范。本轮不加载Restormer，
  不添加CNN恢复器、refiner、energy或router。只更新独立新Rain的50处LoRA。
- 原教师先生成当前精确裁图的缓存，然后释放，再加载学生；不常驻两个FLUX。
  缓存6532份：6400训练曝光＋32固定训练诊断＋100固定开发评价。
  只有train_update记录编码GT latent；评价缓存的该字段置零，训练入口禁止读取评价索引。
  训练每次只从磁盘加载一个缓存，避免8rank各自将全池缓存载入内存。

采样表SHA（JSON规范化摘要）：
`8e68cdeb04d25b9dc68e9f2fa82a1da9672fd43de21ea3cede2344b174e020fe`。

## 早期检查就是训练的一部分

step0/50/100/200固定评估32张训练诊断图与原100来源Rain开发图，共528份50步输出；
训练和开发分别汇总，不混均值。每图都报告同图原一步Rain教师。
开发crop_x/y沿用原manifest，不改512裁图、不做增强；单uint8，PSNR与LPIPS的浮点
算术对齐原Rain开发评分函数。所有评价使用固定逐图seed，不搜索alpha/prompt。
开发图从不进入反传；这是原开发集上的调参与评估，不是官方/封存测试，也不是系统全新盲测。

目标仍是开发PSNR>=34.2、LPIPS<=.08，两项必须由同一checkpoint取得。
每次保存milestones/step_XXXXXX.json，包含两组均值、开发门槛、与前一milestone的差值。
若开发两指标都未改善，标记review_required。该标记要求人工审查，不自动换损失/扩大预算，
也不是预先承诺50步达标。目的地代理应在50/100步提交小摘要到GitHub，终态再回传完整小报告。
主代理按实际回传检查，不声称能直接实时读取另一节点日志。

## 恢复、成本与资源

4节点×2H200，8rank，micro1×累积4；每卡>=139GiB。
资格峰值138.045GiB/卡，当前裁图与微批不变，但新完整路径仍待GPU验收；不宣称H100可跑。
新输出根checkpoints/foundir_rain_flow_formal_stage1_v1，不接旧25步输出目录。
每完整步0–200保存FP32 LoRA、Adam、8rank RNG、采样进度与哈希，201份约73GiB。
训练/评价缓存、输出与快照合计约160GiB量级，启动要求至少220GiB可用，不删除旧资产。
评价不改变训练RNG；中断后最高完整checkpoint恢复，先完成当前milestone，再训练下一步。
同run_id＋fresh共享launch_id恢复本段，损坏/源码/world/runtime变化拒绝，未落盘一步可重算。
四节点失败联动，所有已提交缓存和评价可核验后跳过；超时不自动重启。

原25步训练本体平均1.509秒/步，但不包括保存和评价。本轮新增大量教师缓存、100张开发集
和磁盘I/O，不能拿200×1.509秒当全程ETA。粗估25–50分钟/3.33–6.67使用GPU-hours；
硬cap每节点60分钟，合计8GPU-hours加清理，尚未有本轮实测，不自动扩大预算。

## 目的地同步与启动交接

唯一GPU根：/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。
主代理实现/CPU检查/push；目的地保留本地修改、权重和数据，安全同步最新main，先执行：

```bash
.conda/gpu-train-stable/bin/python smoke/train_foundir_rain_flow_formal_stage1_v1.py preflight --report reports/foundir_rain_flow_formal_stage1_destination_preflight_v1.json
```

通过后给用户四节点命令；入口smoke/start_foundir_rain_flow_formal_stage1_v1_tmux.sh，
四参数node_rank、当前node0地址、run_id、fresh共用launch_id；端口29686。
run_id=formal_stage1_4x2_v1，首次launch_id=launch_formal_stage1_v1。
不沿用旧IP，不重跑旧资格。正式运行结束必须看到RAIN_FLOW_FORMAL_STAGE1_RESULT_VERIFIED
及四份RAIN_FLOW_FORMAL_STAGE1_NODE_END status=0；完成不等于开发门槛通过。
小报告包含132图分组指标、201快照/6532缓存/528输出核验、实际耗时和GPU-hours、
运行资产哈希与核验边界。禁止上传tensor、权重、数据、实时日志。

## CPU验证

新采样表及来源隔离已用真实manifest完成标准库检查，不解码图片。
26项回归全部通过（14项原资格＋12项新正式训练测试，15.966秒不含依赖加载）。
覆盖完整池确定性采样/隔离、开发像素禁止训练、真实裁图/反射padding/D4、132图分片、
原开发评分算术一致、分组均值、防漏样本、评价中断RNG保持、201份保存与下一步精确恢复。
原小尺寸真实FLUX的重算输出/梯度一致性测试也在本次回归中通过。

另有两进程CPU Gloo DDP检查通过：micro4×累积4×world2=global32，评价前后保持RNG，
从保存状态恢复后下一步LoRA/Adam与不中断路径逐元素一致，两rank参数一致。
不把CPU toy DDP称为四节点4B CUDA验收；CUDA/NCCL、新吞吐和新指标仍待目的地运行。

真实源资产预检通过、52份最终源码绑定复核一致；仅元数据/权重检查，没有读取真实图片。
源预检SHA：c9d473217bfe536199609bc737ecdde382bea4b28dfad0b13ce87035d836025c。
配置SHA：817227e084d138ed0e2254e29287ae4fa2da5cf32af854734e1b75b48deffbce。
主代理没有新GPU根挂载，不能用源预检代替目的地预检。本轮GPU尚未启动。
