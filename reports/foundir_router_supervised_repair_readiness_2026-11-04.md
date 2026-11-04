# 路由修复：先真正训练VLM选路，单退化不默认混入其它专家

## 已知事实，以及尚未证明的解释

四图补测已结束（a68d01a），不是待续跑任务。2×H200，77秒，0.042778使用GPU-hours。
四张错误图的模型自选单专家相对正确专家平均低11.193566dB、LPIPS高.137361；
原soft平均反而补回.024818dB、LPIPS下降.020667。仅把这些错例改hard不能解决选错。
正确36图仍有混合损害：总体平均PSNR低.057323dB、LPIPS高.004949，21/36 LPIPS退步。
以上不能解释成专家参数被训练改坏：它们一直冻结。

实际代码中，旧400步分类初始化关闭VLM，之后VLM仅获得8质量步×global32=256条样本曝光。
旧质量loss=10*MSE + task_weight*LPIPS + .1*CE，主专家argmax决定prompt/alpha/读出并detach。
修图loss通过soft权重确实有梯度，但不对离散切换后的另一套prompt/读出协议求导；
不能把它等同于充分训练过的五类VLM分类器。旧head只训练8步是明确训练安排，
是否就是全部错例的根因仍未知；VLM特征是否可泛化识别它们必须实测。

## 本轮唯一配方

使用原缓存和原来源划分：640训练记录/634来源、40开发记录/37来源；五类各128/8。
训练来源及GT哈希与开发分离，但冻结FLUX/Qwen可能见过这些来源，不宣称整系统盲测。
不增加数据、不读任何GT/LQ像素、不生成恢复图、不加载FLUX/VAE/Qwen/LPIPS模型。
仅从已核验safetensors取6144维图像特征和3584维VLM特征；不加载reference latent。

两组从同一随机初始化分别fresh模型/Adam开始，不读取旧qualification模型或优化器：

| 组别 | 训练时的分数 | 损失 | 步数 / batch / lr |
|---|---|---|---|
| image_only | 图像MLP分数 | 训练类别CE | 400 / 32 / .001 |
| image_plus_vlm | 图像分数＋VLM MLP修正 | 同一训练类别CE | 400 / 32 / .001 |

沿用原LayerNorm→Linear256→SiLU→Linear5和VLM最后一层零初始化。
图像MLP初始化、样本顺序、优化器配置、步数完全匹配。每组每条训练记录曝光20次。
VLM组从第一步就得到分类监督，不是关闭400步后只训8步。
不叠加quality loss、学习稀疏阈值、蒸馏、额外refiner或新架构。
CPU FP32、8线程、确定性算法；完整batch32求CE，不冒称与旧CUDA微批累积bitwise相同。
image_only中的VLM梯度为0、权重保持初始化；VLM组必须记录实际非零梯度。

每个完整步（含0）保存模型、Adam、CPU/Python RNG、下一个slot及SHA，两个组共802份。
约24GiB，只保留共享目录，不推GitHub；首启要求30GiB空闲。
同入口恢复最后完整步，不能悄悄跳过损坏状态或换环境冒称精确恢复。孤立文件保留。
两组都完成400步后才加载开发特征；仅评价固定最终400步，没有验证集挑步/早停。
训练标签只用于CE，推理函数没有标签参数。分类损失不是像素监督，也不是GT选路。

## 如何处理混合损害

当前全部是singleton。新部署候选采用预测top-1：图像分数＋VLM修正→argmax→one-hot。
one-hot进入原内部FLUX LoRA bank，保留该专家原prompt/alpha/26或50次读出/量化。
这是保留已有单专家能力的基线，不是RGB图像ensemble，也不是标签指定专家。
选错仍会失败；hard本身不修分类。这轮把完整类型监督与singleton单选一起作为修复候选，
不能声称只改hard就解决了上一轮问题。
混合退化需要单独学习/验证多专家组合，本轮不承诺top-1能覆盖double/triple。

## 本轮报告与下一步

输出每组640训练及40开发的正确数、CE、五类混淆表；开发逐图保留预测分数、VLM修正、
soft诊断概率和实际top-1权重，并标出旧4错例是否改正、旧正确图是否新增错选。
该40图小样本和单seed匹配比较不能证明统计稳健性；旧400→408也不是VLM消融。
没有新PSNR/LPIPS，不因分类提高就宣布恢复门槛通过。

完成后主代理读报告：如果分类改善，再冻结相应router做匹配生成验收；
如果没有改善，先分析训练/开发差距和VLM信号是否学到，不能自动追加昂贵FLUX反传。
生成验收和新正式训练均不在本入口自动启动。当前不是EBT/energy/Restormer任务。
之前连纯推理峰值也达137.933GiB/卡，不能声称原完整FLUX路径能跑H10080GB。
本轮本来就是CPU训练，不受H100/H200型号影响；这不等于解决了后续GPU显存问题。

## 分工与入口

主代理实现、测试、push并分析回传；目的地CPU代理同步和运行现成代码，不自行改配方。
唯一工作根仍为/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。
新CPU入口：bash smoke/start_foundir_router_supervised_repair_v1_tmux.sh。
纯元数据预检：.conda/gpu-train-stable/bin/python -S smoke/train_foundir_router_supervised_repair_v1.py prepare。
完整运行用目的地已验收Python正常启动，不依赖系统Python或另装环境。
tmux显式隐藏全部CUDA设备，CPU8线程，累计硬cap30分钟（不是实测ETA），0 GPU-hours。
运行范围小，但逐步写约24GiB可能受共享盘影响；不保证瞬间完成，超时保留步骤，不自动扩预算。
若硬断导致attempt账缺失，先核对日志记账，不清零失败成本。

日志：logs/foundir_router_supervised_repair_v1.log。
输出：checkpoints/foundir_router_supervised_repair_v1/fresh_ce400_matched_v1。
成功标志：SUPERVISED_ROUTER_FINAL_VERIFIED及SUPERVISED_ROUTER_CPU_END status=0。
CPU代理回传reports/foundir_router_supervised_repair_terminal_2026-11-04.json和简短退出/耗时记录；
不推特征、图片、checkpoint、Adam状态或live logs。没有本轮GPU启动命令。

## 验证边界

本机没有目的地特征缓存。实际640/40训练和分类结果待目的地执行，不把测试夹具当实验成果。
33项CPU回归通过（11项新增＋22项旧诊断回归），覆盖完整双组400步流程（缩小维度的合成夹具）、真实Adam下一步精确恢复、
同初始化/同样本曝光、VLM第一步梯度、image-only不更新VLM、argmax无标签、
开发特征访问时机、缓存只取image/vlm以及最终报告独立计数核验。
旧绑定源码、权重与既有实验结果均不改。
本机CPU测试使用原训练环境的Python/torch，以-S和显式site-packages避开旧共享盘上
缓慢的sitecustomize自动导入；这只适用于本地CPU单测，不修改目的地正常启动方式，
不把本机导入成功当作新节点或CUDA运行验收。
