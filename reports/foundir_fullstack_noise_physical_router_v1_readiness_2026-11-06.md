# Noise/Lowlight物理统计路由：固定CPU小试验

用户已要求继续下一步。当前三节点六卡全部继续用于Rain训练及固定Rain100评价；本试验仅CPU，不新增GPU任务、不修改FLUX专家、不自动部署。

先前冻结路由CPU重放完成：4901条Noise/Lowlight训练、校准、确认缓存及官方150图。训练来源平均后Noise1024/1024、Lowlight543/543正确；官方10个Noise错路由每图23–27/40块也判Lowlight。唯一预定替代“平均概率”未改变任何150图路由，不能靠此简单汇总修复。CPU222.413秒，GPU0。

来源屏蔽使用原manifest固定pHash汉明距离≤4规则及官方GT文件哈希；官方GT仅用于排除来源，不进入分类输入/损失。222个唯一GT文件索引和元数据扫描33.194秒，Noise/Blur候选池排除56来源组，留下Noise train1714/validation34、Blur train7325/validation165。预留每类128训练/32开发来源仅为元数据计划，不授权GPU规模或证明语义完全分离。

发现旧路由fit池159个裁图（30个跨类别来源组）与官方共享GT文件，全部都有精确GT SHA命中；这确认池的来源重叠，不等于GT像素进入路由训练。路由只读LQ特征和类别标签。旧权重的曝光不因新过滤消失，官方仍为自适应基准。

新CPU试验只在原路由为Noise或Lowlight时计算二分类修正，其余Blur/Haze/Rain决策完全不变。输入是原冻结整图平均logits加20个LQ统计：RGB均值/标准差、亮度分位数/黑白比例，以及固定五个原分辨率512裁图的Haar高频MAD/幅度/梯度。高频统计混有纹理，不宣称是真实噪声强度；亮度也不能单独区分暗物体与欠曝光。

修正器为零初始化线性层，21个可训练参数，无RGB输出。训练集均值/标准差归一化（下限1e-6、标准化值固定clip±8）。新的Noise–Lowlight margin = 原margin + 线性修正。固定400步，batch64两类各32，AdamW lr.001、weight_decay.01、clip1，BCE+.01×修正平方均值；seed2026110609。无温度、阈值、checkpoint或超参数扫描；终点唯一step400。

已筛除165个旧缓存裁图（fit159，calibration/confirmation各3）。新数据按来源聚合：fit Noise1001/Lowlight513，calibration80/77，confirmation20/17。只fit更新参数；校准、确认在固定400保存后报告，不用于选步。各role来源集合分离，保留旧数据。过滤是近似来源屏蔽，不能保证彻底语义去重，也不能消除冻结旧路由已有曝光。

四项CPU测试通过：常量图高频为零、加噪改变高频但不明显改变亮度、零初始化路由保持、非Noise/Lowlight输出不受任意修正影响。prepare已完成；尚无新训练或官方结果。

CPU顺序：`train_foundir_fullstack_noise_physical_router_v1_cpu.py run`，然后`evaluate_foundir_fullstack_noise_physical_router_v1_cpu.py`。共享环境Python，CUDA_VISIBLE_DEVICES为空，最多两线程/两图解码并行，CPU训练阶段最多1800秒，无自动扩展或重试。

官方阶段固定400权重，只读150个已被旧路由判为Noise/Lowlight的LQ；先落盘无标签决策，再查旧GPU输出及指标。缺少候选专家输出就明确报告待GPU，不能用其它图补齐或假称完整指标。只报告Noise/Lowlight，不覆盖已接受Haze或正在生成的新Rain，不改当前部署/队列。全部成果为诊断，失败也推送。即使路由全对，Noise专家自身37.693592/.110755仍未过38.9/.10，后续仍需专家能力改善。
