# 已授权：Rain前缀与末次预测四组对照

用户明确execute。本轮固定读取历史训练探针索引12800–12807，原32图中的前8张，未按分数选图；不训练、不读开发或官方像素、不启动其它任务。仅node0两卡H100，独立每卡4图，不需要跨节点NCCL。未来H200需按worker登记核验，不能直接沿用旧IP映射。

固定两个完整训练状态step800/1000，FP32 LoRA逐参数核对再复制，模型base/VAE/prompt不变。每张分别生成800与1000权重的前25调用状态，在各状态上分别用800/1000权重做第26次endpoint预测。保持原50步时间表、相同seed、CFG1、reference time10、512裁图、当前VAE运行路径和Rain单次uint8量化。每图54次模型调用、8图共432次，32个输出全部保存，不逐图选择或部署混搭结果。

同权重的800/800和1000/1000必须与原探针uint8逐像素相同且两项指标差<=1e-5，才解释交叉差异。保留原torchmetrics训练探针指标身份，不把本轮数值与pyiqa官方值混作直接对照。GT只在四个输出已固定保存后算训练侧指标，不进入状态生成或读出选择。

报告两种固定前缀下的末次权重变化、两种固定末次下的前缀变化及2×2交互，另报告生成token状态RMS变化。预定描述性特征为：固定800前缀换1000末次时PSNR改善且LPIPS不恶化，随后前缀换1000时PSNR下降。即使出现该特征，也只支持在这8张研究前缀耦合，不证明普适因果、显著性、泛化或官方成功。不出现则保留负结果，不盲目冻结前缀继续训练。

5项CPU测试覆盖交叉权重顺序、先完整验证再原子复制适配器、固定分片、对照复现保护和缺失记录拒绝。真实CPU预检核验两份完整模型状态SHA/FP32/schema、8份训练缓存和16份原probe tensor、base/VAE/prompt/native初始化SHA；不解码官方图。CPU预检12.989秒，0GPU-hours。CUDA/耗时待真实运行。

预估含加载约3–8分钟，按实测更新；硬cap900秒/单节点两卡=.5 GPU-hours，另最多30秒清理。不自动重试/扩图/训练。任务ID`round7_rain_prefix_cross_v1`，输出`checkpoints/foundir_fullstack_rain_prefix_cross_v1`。日志位于该job的node0.log，必须看到实际记录/worker退出以及CPU finish，tmux/submit不算完成。

入口`smoke/run_foundir_fullstack_rain_prefix_cross_v1.sh 0`由现有共享执行器派发；先发布源码、核验SHA、检查node0空闲。CPU持久观察器`smoke/watch_foundir_fullstack_rain_prefix_cross_v1_cpu.py`负责finish与小JSON/Markdown报告推送，随后停止。所有旧缓存、模型、取消记录和运行日志保留。
