# V4按用户现有资源适配：两节点共四张H100

用户明确只能提供两节点×两张H100，覆盖5c6ae26原四节点H200限制。已安全同步5c6ae26并保留全部旧资产和无关修改；原V4尚无receipt/缓存/训练，适配在新run首次prepare前完成。

四个独立分片覆盖全部11457新特征及固定100确认图四组去重恢复，不做跨节点NCCL，不需要master地址。全局rank=2*node+local，node仅0/1；CPU汇总、worker收据、rank检查、耗时账同步改为4worker/2节点，硬编码旧8分片提取器没有被直接复用。

显存执行策略：VAE encode/decode使用FP32原生PyTorch卷积，显式在该范围禁用cuDNN，保留原attention query chunk256、全部键值及上下文。模型权重、BF16 FLUX/Qwen及原LoRA精度、LPIPS、512裁图、采样种子、50步/Rain26调用、alpha与双uint8规则均不改。不使用量化、切图或新offload。

旧137–138GiB峰值原因尚未通过CUDA内存分析隔离，不能认定已查明根因；本改动避免VAE调用cuDNN卷积工作区。H100实际峰值/吞吐尚未实测，CPU测试不等于CUDA资格。原生卷积与旧cuDNN数值可能有差异，不宣称跨硬件/内核bitwise相同；保留2960份历史缓存并绑定原producer，新receipt记录数值策略及源码差异。新的三组预测及类别参考恢复都用同一新策略。

总预算不增加：features每节点3小时×4卡=12GPU-hours，render每节点1.5小时×4卡=6GPU-hours；CPU6400步最多1小时，prepare/finish各10分钟。各失败尝试累计，不保证在预算内完成、不自动超时续跑；若OOM/超时保留所有已完成项并报告，不静默改配方。

阶段：CPU prepare → 两个GPU节点features 0/1 → 全四worker/两节点status0 → CPU train → 两节点render 0/1 → CPU finish。没有旧每类分类门槛，不自动启动GPU，不增加资格实验。代码支持不代表H100跑通，须看实际hardware/worker状态。

检查项：四分片覆盖不漏样本/不重复，4卡GPU-hours cap与原8卡相同，VAE作用域恢复/异常恢复/权重dtype不变，V4既有CPU训练/确认锁定/恢复计划/汇总测试。真实prepare结果和测试数量在执行后补记；无新科学结果。

实际目的地验证：16项focused CPU测试通过（13项原V4更新后回归＋3项执行策略测试），Python编译/shell语法/diff空白检查通过。prepare wrapper实测22秒/status0，清单哈希与原568ffd35…一致：fit13917、新特征11457、原缓存复用2960、校准400/确认100。prepare未加载GPU/像素。尚无新H100 CUDA结果，不把合成CPU测试输出当实验指标。
