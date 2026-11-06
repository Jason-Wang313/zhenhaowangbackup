# Blur当前训练侧64图只读诊断

1024来源固定200步官方仍21.819776/.241321，未过25.9/.18。修正平均幅度减小、平均方向改善，但90/150仍非正相关。先检查当前训练图本身是否学会正确修正，避免把所有问题先归因泛化或盲加训练。

固定当前1024训练来源的前64条及已有512裁图，均实际参加当前200步训练；不按GT/结果挑图，不读取其它开发/测试集。使用同一最终200权重与已核验参考缓存；生成PNG先固定，再读训练GT计算默认pyiqa PSNR/LPIPS/SSIM及RGB修正与GT-LQ的相关、均方幅度。零目标/零修正明确记录，不能假定必有非零修正。全部结果标training_fit_only，不作为泛化或官方通过。

只用node3双H100、每rank32张，无optimizer/更新、无旧50步生成，无新官方评价；核对实际加载参数SHA及前后不变，64图完整性及PNG SHA。所有旧绑定文件/资产保留。准备阶段核对64缓存、最终权重与失败官方父结果、源码；上线前零残差边界处理修改的旧receipt保存在同目录receipt_prelaunch_before_zero_case_fix_*，无GPU重跑。

入口`smoke/diagnose_foundir_blur_coverage_fit_v2.py prepare|run|finish`，launcher `smoke/run_foundir_fullstack_blur_coverage_fit_v2.sh 3`。源码发布后通过常驻worker提交round10_blur_coverage_fit_v2；输出checkpoints/foundir_blur_coverage_fit_v2。本短任务由当前代理监测并手动CPU finish/推小报告，无单独自动后续训练。GPU失败不自动重试。

当前prepare与Python/shell语法检查已通过，GPU未提交。没有新增质量门槛；唯一验收仍完整官方150同时25.9/.18。该诊断不会证明更多训练必然有效，后续方案须依据结果另行冻结；完整自主研究授权持续有效，不要求重复许可。
