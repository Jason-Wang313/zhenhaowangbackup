# Blur单来源16上下文拟合：固定200新增步能力诊断

官方150当前21.948140/.202188，目标25.9/.18均未过。上一轮上下文差距缩小主要来自独立裁图退步，不能当修复通过。本轮只检验同一训练图不同上下文能否同时学习正确修复，不通过扩数据或无限加步回避问题。

原1024来源列表第一条foundir_train_0034314（训练文件0034315.png），15个512部署网格裁块加原[985,489]裁图，共16；不是按GT或分数挑图。完整1792模型/Adam复制到新目录，四rank转两rank重新显式seed；不称逐位等价父训练。保持原架构、loss、lr1e-5、global8，1792→1992固定新增200步；16视图无放回按epoch打乱，每个恰100次曝光。仅物理node3两H100。

初始16视图和拼接必须逐像素复现已有冻结1792输出；新增0/50/100/200固定记录全部视图和全图。每完整步保存完整模型、Adam、两rank RNG、采样位置和SHA，父历史1792项保留。异常停止，无自动重试/续步/官方/后续类别。诊断权重绝不部署或seed正式训练。

6项CPU测试通过，含真实两进程global8 Adam汇总对比串行、同world下一步精确恢复、父模型/Adam迁移不暗改RNG、200步采样覆盖、禁用非训练路径/重复视图、快照漏样本/混权重拒绝。语法检查通过。目的地CPU预检12.472429秒，父完整checkpoint和16旧PNG、训练源图和指标身份核验；GPU0。代码/预检不等于CUDA验收。

入口smoke/train_foundir_blur_singleframe_fit_v1.py prepare|run|finish；专用GPU入口smoke/run_foundir_fullstack_blur_singleframe_fit_v1.sh 3，由已授权共享worker执行。新输出checkpoints/foundir_blur_singleframe_fit_v1，不修改旧绑定源码/缓存/权重。18–25分钟/.60–.83 GPU-hours仅粗估，以实际日志和ended记录更新。

持久CPU观察器smoke/watch_foundir_blur_singleframe_fit_v1_cpu.py等待唯一round19_blur_singleframe_fit_v1结束，执行CPU finish，核验连续receipt、最终实际state、64视图输出和4张拼接、重算均值，然后只推小报告/耗时/退出码；不重新运行GPU。全程training_fit_only，无新质量门槛。是否同时学好多个上下文由完整快照如实判断，不用任一最好指标挑部署输出。唯一质量门槛仍完整官方Blur150，未过不推进其它类别。
