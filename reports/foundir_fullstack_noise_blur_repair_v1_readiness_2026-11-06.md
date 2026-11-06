# Noise/Blur下一轮：只剩实际GPU执行，不能称修复已成功

本轮按用户“继续到只剩GPU运行”准备代码、真实CPU预检、执行器入口及自动收尾。当前不提交GPU任务，Rain评价保持运行。Noise先、Blur后；保留所有模型、旧失败、energy工作和运行绑定源码。Haze/Lowlight不动；五类候选均通过之后才统一600图端到端重生成。

## 已知问题与选择依据

Noise有两个独立问题：10张同场景误路由到Lowlight；即使使用正确专家，官方均值37.693592/.110755仍未过38.9/.10。之前统一倍率、平均概率汇总、21参数物理margin修正均无有效修复。新705参数独立二分类在固定400步后也未通过：校准Lowlight75/77、确认16/17，分别低于原77/77和17/17；CPU12.833秒，0GPU-hours。失败完整保留，不部署、不跑官方挑阈值。

新的路由候选是42,153参数的LQ图像**分类器**，不是RGB恢复网络。它看原分辨率128像素四角/中心及全图缩略图，加固定局部高频残差，补充全局20个统计量丢失的纹理分布。没有保证这些特征足以解决跨场景误判；GPU后用来源隔离检查否定或支持它。

Noise全图均值偏差可解释部分误差；Blur150张路由正确、91张PSNR差于输入，常量颜色误差很小。Blur历史真实50步末四步梯度及512来源空间监督已失败，不能重命名为新方案。此次保持FLUX和LoRA全部冻结，测试**六频段残差校准**：不同频段采用不同的全局系数，而非旧统一scalar倍率。不改变结构位置、不生成新的细节，可能仍无法解决Blur几何错误；这是有限修复假设，不是已证实的原因或效果。

## 固定配方

### Noise纹理路由

- 复用已筛除官方匹配来源的LQ；fit1514来源（Noise1001/Lowlight513）、calibration157（80/77）、confirmation37（20/17）。不读训练GT像素，不把official图放入训练。旧冻结路由的来源曝光不能被新筛选抹去。
- 每图固定6视图：全图缩略图及四角/中心原生128裁图。三层卷积/GroupNorm/SiLU，平均与标准差池化，六视图logit均值；42,153参数、单一标量输出。
- fresh AdamW，400步，global batch32，每rank16（Noise8/Lowlight8），lr=.001、wd=.01、clip1；固定seed2026110613，FP32。没有参数/阈值/步数扫描，没有FLUX梯度。
- 单节点两卡，逐完整步0..400保存模型/Adam/历史及SHA；采样由step/rank固定seed决定，无dropout或随机增强。相同两rank可读取最后完整状态，损坏/缺口拒绝，失败不自动重试或扩预算。不承诺CUDA断电恢复已实测。
- 原路由不是Noise/Lowlight时完全保持；二分类阈值固定0。校准与确认每类正确数均不得低于原路由，才允许以后评价官方；不自动部署。

### Noise/Blur专家输出校准

- 各128训练来源和32开发来源，固定512裁图/seed；沿用 `foundir_fullstack_noise_blur_sources_v1/plan.json`，未按官方指标重新挑图。
- 原50调用、CFG1、reference time10、BF16模型及现用H100 VAE数值路径不变，GT不进入恢复。保存第一遍量化raw及原双uint8基线（Noise alpha .30、Blur .75）。原实现基线为主对照，不假设频域数值路径逐像素相同。
- 对 `raw_uint8 - LQ` 做正交二维DCT。六段：DC、归一化max轴频率<1/16、1/16–1/8、1/8–1/4、1/4–1/2、>=1/2。DCT采用反射边界；按该512块尺寸定义频率，未来整图部署应沿相同512块后拼接，不能默默改成全图DCT。
- 每类6个系数，RGB共享、所有图共享，范围[0,1]；只以128训练GT计算充分统计量，闭式求解带固定1e-3相对能量ridge的MSE目标，正则中心为原alpha。零能量回到原alpha。没有开发或官方反馈，没有逐图最佳系数，也没有LPIPS/alpha网格搜索。
- 推理仅 `LQ + IDCT(a_band * DCT(raw_uint8-LQ))`，clip及一次显式uint8取整。无训练GT、评价GT、类别真值或路径决定输出；此轮训练/开发采用已知专家标签以单独测专家，不包装为已完成端到端路由评价。
- 固定评分：LQ、真实原基线、六段全设原alpha的数值对照、新候选。pyiqa PSNR/LPIPS/SSIM身份与官方当前版本一致；原基线和频域常量对照在取整临界点可能相差一个灰度级，必须保留差异。
- 采用条件：train/development各自平均PSNR严格优于原基线、LPIPS不恶化，开发最大单图PSNR退步<=1dB。只决定能否进入下一轮官方评价，**不代替**官方Noise38.9/.10、Blur25.9/.18绝对门槛。开发曾参与历史研究，不称盲测。

## 阶段与预算

执行器是现有共享目录worker，不需要SSH或重新在三台终端启动worker。GPU入口先核验节点、型号、空闲与本机锁，再由torchrun启动；tmux创建不是任务成功。

| 顺序 | 工作 | 使用资源 | 硬cap |
|---|---|---|---|
| 0 | 等待Rain三节点退出0、CPU终态及简报发布 | CPU监视器 | 不占新GPU |
| 1 | Noise纹理路由400步与固定holdout | node0两卡 | 30分钟 / 1 GPU-hour |
| 2 | Noise160图冻结专家缓存 | node0/1/2六卡 | 每节点45分钟 / 共4.5 GPU-hours |
| 3 | Noise六系数解析拟合 | CPU | 30分钟 |
| 4 | Noise160图四项评分 | node0两卡 | 15分钟 / .5 GPU-hours |
| 5 | Blur160图缓存→CPU拟合→160图评分 | 缓存六卡，评分node0两卡 | 同Noise / 共5 GPU-hours |
| 6 | CPU核验、推送小报告、停止 | CPU | 不部署/不追加任务 |

本轮GPU硬cap合计11 GPU-hours，旧Rain和全部历史成本另计；不是可自动反复重启的预算。使用卡数与完成节点实耗由队列ended.json汇总；若失败/未知结束时间，不能把未完成节点费用当零总成本。缓存160张按已有Rain推理速度粗估每类10–20分钟，整轮约30–60分钟，仅估计；首批实测后更新，不承诺H100/H200精确比例。

缓存支持明确的单节点或三节点计划，默认三节点以缩短160次原生50步恢复；不能对已经绑定的同目录更改world。轻量路由/评分均单节点。当前worker只登记三台H100；代码硬件检查兼容75GiB以上H100/H200，换H200仍需按执行器手册重新登记，不冒称本轮混卡实测通过。

## 当前预检与数值环境

- 21项CPU测试：系数端点/零能量、不同频段解析恢复、取整临界点、无漏样/重复分片、来源隔离、双卡等权梯度、确定性均衡采样、路由其它三类保持、退步拒绝、队列失败/部分完成/重复提交保护。
- Noise/Blur实际预检分别核验320份LQ/GT文件SHA、尺寸与固定512裁图解码，模型/专家/提示词共12项权重SHA；CPU构造原pyiqa并核对完整state identity。
- 纹理路由1708来源的LQ SHA、六视图输入已准备，约66.828秒，0GPU-hours。大缓存只保留共享盘，不推GitHub。
- 首次Noise预检用 `python -S` 时缺失环境通过sitecustomize提供的urllib3，已保留失败日志。改用原GPU环境的正常Python入口后通过，未安装包、未修改环境。涉及pyiqa的预检必须用下方正常Python命令；纯标准库/解析拟合/监视器可用已配PYTHONPATH的 `-S`。
- 这些是CPU/文件/数值契约通过；CUDA吞吐、显存、NCCL和真实科学效果都待本轮GPU运行。未改Rain当前绑定文件；第三方FLUX源仍使用共享盘原资产并逐文件SHA绑定，不复制覆盖或假称所有第三方文件已纳入Git跟踪。

预检结果：
`reports/foundir_fullstack_noise_texture_router_v3_preflight_2026-11-06.json`
`reports/foundir_fullstack_noise_frequency_repair_v1_preflight_2026-11-06.json`
`reports/foundir_fullstack_blur_frequency_repair_v1_preflight_2026-11-06.json`

如需重验已有准备状态（不重复prepare、不启动GPU）：

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$PWD/.conda/gpu-train-stable/lib/python3.12/site-packages:$PWD"
.conda/gpu-train-stable/bin/python -S smoke/train_foundir_fullstack_noise_texture_router_v3.py verify
.conda/gpu-train-stable/bin/python -S smoke/run_foundir_fullstack_frequency_repair_v1.py verify --task noise
.conda/gpu-train-stable/bin/python -S smoke/run_foundir_fullstack_frequency_repair_v1.py verify --task blur
```

首次prepare的准确入口（已执行，不重复覆盖不可变receipt）：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TORCH_HOME="$PWD/models/torch" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD:$PWD/third_party/flux2/src" \
.conda/gpu-train-stable/bin/python smoke/run_foundir_fullstack_frequency_repair_v1.py prepare --task noise
# Blur同命令末尾 --task blur；正常Python，不加-S。
```

## 真正启动入口（本轮尚未执行）

在共享CPU目录用已发布的40位提交SHA启动一次持久监视器：

```bash
bash smoke/start_foundir_fullstack_noise_blur_v1_tmux.sh PUBLISHED_FULL_COMMIT
```

`PUBLISHED_FULL_COMMIT`必须替换为本轮实际核验后的提交，不能原样执行。入口只在该提交包含本轮源码且Rain收尾后派发固定队列；无需用户分别进入三台GPU终端。

日志：`logs/foundir_fullstack_noise_blur_v1_cpu_monitor.log`。
任务ID：`round6_noise_texture_v3_v1`、`round6_noise_frequency_cache_v1`、`round6_noise_frequency_score_v1`、`round6_blur_frequency_cache_v1`、`round6_blur_frequency_score_v1`。

监视器逐阶段检查所需节点真实ended状态，调用CPU fit/finish，再非force推送小报告。任何非零退出、绑定变化或Git推送失败停止后续阶段，资产保留，不自动重试。科学门槛未过如实报告，可继续独立的另一个类别有限试验，不自动扩充候选。

最终输出按候选报告，不覆盖已有官方600、已接受Haze或Rain结果；失败候选不进入部署。下一轮官方类别评价需根据本轮结果确定固定组合；不是本轮自动任务，更不是统一600端到端复跑。
