# Blur：四个部署网格裁图 / 同四卡完整状态续训

官方尚无双指标改善：原50步FLUX21.546762/.202622 → 200步21.819776/.241321 → 1280步20.936988/.196516。25.9/.18均未过。训练64图改善和8图上下文诊断仅定位问题；同一512像素换全图分块后PSNR下降1.228024dB，中心384仍下降1.076556dB。融合降低MSE，误差在分块预测中已有。8图不能解释全部官方失败。

固定方案：原1024训练来源各4个不同部署网格512裁图，以SHA256(seed, record_id, x, y)排序取4，无GT或指标输入。每来源每epoch一个不同视图，4epoch/4096视图/global8/512新增步；来源顺序保持旧shuffle。原1280完整模型、Adam及四rank RNG恢复到1792，不从小样本拟合权重初始化，不重跑旧步。原模型/loss/lr/clip/global8保持；输入归一化使用官方CPU FP32运算再上GPU，记录与原GPU归一化存在微差。该受控变化包含裁图和部署数值路径对齐，不声称纯粹单一变量因果。

新缓存键含record_id/view/x/y，独立根checkpoints/foundir_blur_multiview_v1，绝不覆盖旧train_record_id缓存。冻结原参考LoRA生成LQ/latent/reference velocity/reference decode，缓存不含GT。每个完整optimizer步保存模型/Adam/四rank RNG/采样/哈希；保存1282后真实CUDA重载检查。固定最终1792官方全150/2250块，无中间官方选择、GT回退或其它测试gate。官方反复参与开发，不称独立盲测。

10项CPU测试通过，包含真实四进程global8与Adam汇总、精确同world恢复、全4096视图覆盖/旧缓存键不冲突、只提交本轮新入口、失败只取消同组job。目的地预检21.459116秒，父checkpoint实际SHA/100 Adam/4 RNG/next_global_slot10240和2048图文件SHA通过。GPU本轮尚未运行，不能用CPU通过代替CUDA结果。

源码安全发布后启动持久CPU控制器的monitor与独立upload。物理node1/node3各两H100；cache和official独立分片，train同四rank NCCL，master10.244.47.55:29883。队列分别是round17_blur_multiview_{cache,train,official}_v1_node{1,3}。先读取daemons.json/共享heartbeat/job/进度以防重复派发。轻任务不扩大节点；本轮4096视图+训练+官方属于较长任务。

控制链只含缓存→CPU检查→固定训练→CPU检查→官方150→CPU核验/小报告推送。1281/1536/1792及终态回传。故障停止本轮同组任务、不重试、不扩步、保留全部资产和费用；无累计GPU-hours上限不是任意扩大本轮预算。官方未过不切其它类别。估算合计45–65分钟、3–4.33使用GPU-hours，待缓存/训练实测修订。最终报告记录退出码、各阶段普通耗时及实际GPU-hours，全部旧费用另计保留。
