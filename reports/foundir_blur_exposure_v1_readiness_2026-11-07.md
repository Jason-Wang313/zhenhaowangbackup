# Blur1024来源：原step200迁移六卡，固定总1280步后官方150

当前唯一目标类别Blur，完整官方150的PSNR≥25.9且LPIPS≤.18是唯一质量门槛。旧结果21.819776/.241321失败。刚完成固定8图100步拟合26.408422/.136296，8/8双项优于LQ，只证明当前映射有小样本学习能力；不能证明官方只缺步数，也不采用该8图权重。

本轮可证伪假设：原1024来源仅1600曝光（每来源1–2次）不足以训练当前映射。保持原1024来源、512裁图、缓存、原生LoRA结构、FP32 LoRA/Adam/VAE/latent、BF16 FLUX、loss=10*MSE_RGB01+.1*LPIPS、lr1e-5、wd0、clip1和全局batch8；从原1024来源完整step200模型与Adam继续到固定1280（新增1080步），总10240曝光、每来源正好10遍。仅最终1280输出完整官方150，官方中途检查点不评价、不选择；不增加其它测试门槛。

用户偏好非轻量训练跨节点。本轮node0/1/2三节点×两H100，master10.244.211.17:29873，TCP NCCL；node3保留空闲。原两rank改六rank，模型/Adam矩/step/全局采样位置1600精确保留，六rank RNG按2026110714+rank显式重建，**不是逐位等价两卡续训**。原两rank RNG及checkpoint原封保留。已有模型Dropout为0，运行时检查；浮点归约顺序仍改变，不隐瞒这项混杂。迁移后的同六卡断点恢复保存完整六rank RNG，旧200步不重算。

每步同一有序global8分给六rank为2/2/1/1/1/1真实样本。每图loss除8，本地累积后复用已实测flat SUM梯度汇总，再clip/Adam；不补假样本、不额外除world、不改lr。六rank实际NCCL SUM先验收21；每个完整step200..1280原子保存模型/Adam/六rank RNG/采样位置及SHA，201/202/512/1024/1280核对跨rank参数hash，202后实际加载保存的Adam/RNG再继续203。CPU测试中的六进程Gloo不是物理跨节点CUDA验收。

7项针对性CPU测试通过：完整十遍采样、模型/Adam迁移不偷偷恢复旧rank RNG、六进程global8梯度/Adam与单进程对照及同world逐值恢复、官方150六路完整分片、双门槛同一完整输出、GPU节点失败及控制器失败只取消本轮对应job。Python/shell语法通过。目的地预检14.163213秒/退出0/GPU0：原step200实际SHA/FP32参数/Adam/两个原RNG/采样完整，1024缓存元数据/文件存在，参考缓存实际tensor SHA在每次读取时检查。无重复全历史大checkpoint扫描。

官方metadata plan已通过：六rank每卡25张/375块，共150图/2250块/4500模型调用；复用原无GT conditional_v4路由，每张图新候选固定落盘后才读取GT评分。保持原512/128拼接、量化、指标和完整分辨率。只评价Blur；其它类别不动。官方已经反复用于开发，不是独立盲测；官方GT不拟合参数或阈值、不逐图挑输出。

源码先发布并验证，再启动持久CPU链：smoke/watch_foundir_blur_exposure_v1_cpu.py monitor --commit <SHA>；独立upload进程负责小报告非force push。顺序为固定训练→CPU train finish→最终1280导出/预检→三节点官方评价→CPU完整性核验→终态简报。GitHub超时仅延迟上传，不阻塞已发布GPU工作。201/512/1024/1280的实际checkpoint/Adam/RNG独立核验并推报告。任何节点失败或控制器核验失败取消本轮仍在跑的对应job，保留所有资产，不自动重试/追加步数。

训练job round14_blur_exposure_train_v1，入口smoke/run_foundir_fullstack_blur_exposure_v1.sh，输出checkpoints/foundir_blur_exposure_v1。官方job round14_blur_exposure_official_v1，入口smoke/run_foundir_fullstack_blur_exposure_official_v1.sh，输出foundir_blur_exposure_official_v1。两入口均由共享worker传入node0/1/2，不需要SSH或用户再操作终端。

未实测粗估六卡训练45–90分钟、官方6–12分钟，共约5.1–10.2 GPU-hours；必须按首段真实吞吐修正，不能当承诺。每步checkpoint预计约392GiB新增，预检要求500GiB可用；不上传大文件、不清理旧资产。无累计GPU-hours上限，但本轮固定终点1280和官方150，不自动扩大训练或切其它类别。进程退出不等于质量通过，须完整官方双门槛核验。
