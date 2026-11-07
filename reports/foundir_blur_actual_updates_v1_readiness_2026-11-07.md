# Blur历史真实更新重放：目的地预检完成

方案已固定于 reports/foundir_blur_actual_update_audit_v1_plan_2026-11-07.md；该文件的“尚无实现”是当时状态，本文件更新为实现和CPU预检完成。仍只攻Blur，唯一质量门槛是完整官方150张的PSNR≥25.9且LPIPS≤.18。

当前1024训练来源试验的step0、1、199、200四份实际checkpoint SHA、step/world/采样位置、FP32参数hash、Adam状态、两rank RNG已核验，耗时10.120323秒，退出0。原始权重/RNG只读导出到新目录，旧资产不动。4项CPU测试通过，Python/shell语法通过。测试覆盖checkpoint完整性/原1024采样/有限步与一阶判断差异/固定重放容差。没有新训练，也没有官方像素访问。

两个转移各自恢复原global8训练图和RNG，重新计算10*MSE/.1*LPIPS分项参数梯度及一次真实combined backward，NCCL SUM后对照原日志。loss/MSE/LPIPS误差各≤1e-7，gradient norm按max(1,原norm)归一误差≤3e-4；不匹配则失败并保留记录，不自动放宽。随后计算各梯度与已保存真实参数delta的内积，加载after权重测同批有限步指标变化。显式权重切换之外参数hash不变，无optimizer.step或新参数拟合。分项梯度相加与一次混合梯度的舍入误差单独报告。

入口 smoke/diagnose_foundir_blur_actual_updates_v1.py prepare|run|finish；launcher smoke/run_foundir_fullstack_blur_actual_updates_v1.sh 3。输出 checkpoints/foundir_blur_actual_updates_v1；结果 reports/foundir_blur_actual_updates_v1_result_2026-11-07.json。源码发布核验后，通过已运行共享worker提交 round12_blur_actual_updates_v1，单节点node3两H100。预计2–4分钟；记录实耗/GPU-hours，不因简单任务占用其它三台。

两rank、launcher均退出0且CPU finish核验后才能报告诊断完成。短任务无额外持久观察器，由当前代理跟踪真实终态并推小报告。失败不盲重试，先保存错误证据；不能把已启动或训练侧诊断称为官方修复成功。结果仅training_fit_only；官方仍21.819776/.241321失败，无其它测试集gate，保留全部历史失败与费用。
