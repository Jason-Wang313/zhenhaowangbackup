# Blur当前一步预测差：固定8训练图、fresh100拟合诊断

当前官方150图21.819776/.241321仍失败，唯一质量门槛仍25.9/.18，未过不推进其它类别。上一轮真实step0→1及199→200重放表明原loss下降并不保证逐图两项改善。当前只核对当前原生delta映射是否能够在充分重复曝光的同一小批训练图上学好；不是新的验证集gate。

历史核对：2026-09-11的16图400步使用pure-noise rectified flow与50步推理，曾训练侧28.4124/.08520；2026-09-16的16来源末段监督训练侧改善但来源分离评价失败，512来源空间锚定也没有联合改善。这些是明确保留的历史，均不等价于当前LQ+VAE解码预测差映射，本轮不重跑它们，也不把小样本可拟合当可泛化。

固定当前1024来源列表前8条（就是已有梯度诊断前8图），不按指标挑图。复用原512裁图/原LQ参考缓存、原step10000 LoRA，fresh Adam；不继承两个失败step200或资格权重。每个完整step用8图各一次，100步共800曝光、每来源100次。模型、32,440,320原生LoRA参数、FP32 LoRA/Adam/VAE/latent与BF16 FLUX、loss=10*MSE_RGB01+.1*pyiqa LPIPS、lr1e-5、clip1、wd0和原seed不变；不加RGB网络，不修改energy。

固定0/1/25/50/100快照，全8图以同一固定权重记录PSNR/LPIPS/SSIM和逐图退步，保存每图PNG与SHA；不选最好的检查点。step0必须浮点精确复现LQ。每个optimizer step都原子保存两rank RNG、参数、Adam、采样位置和SHA；同world恢复，可中断保留，失败不自动重试。评价前后参数hash不变且恢复RNG，避免诊断改变训练随机过程。终态独立核验完整101份step receipt和最后实际checkpoint、五组40图输出，以及与原1024试验step200同8图的描述性比较；不声称控制了所有数据/曝光混杂，也不把固定8图直接当可部署方案。

解释：若固定8图拟合明显改善，支持当前映射至少有训练拟合能力，再研究覆盖与跨图干扰；若仍学不好，只说明此固定100步/原优化配置不足，需研究优化或表示，不能宣判所有FLUX不可能。只标training_fit_only，不用官方GT训练或调阈值。本任务不会自动跑官方或接续训练，不增加其它测试集gate。

11项CPU测试通过（4项新合同/完整性测试、7项既有数学/全局batch/真实Adam恢复测试），Python/shell语法通过。目的地预检12.978861秒，退出0：16训练源图SHA/裁图、8参考缓存、模型/提示词/指标身份核验，0GPU-hours。

源码入口smoke/train_foundir_blur_fixed8_fit_v1.py prepare|run|finish，launcher smoke/run_foundir_fullstack_blur_fixed8_fit_v1.sh 3。输出checkpoints/foundir_blur_fixed8_fit_v1；队列round13_blur_fixed8_fit_v1，发布核验后提交node3双H100，其它节点空闲。预估8–12分钟/.27–.40 GPU-hours，约40GiB新增逐步checkpoint；预检要求至少55GiB可用。不设累计GPU-hours上限，固定100步，无自动追加。

当前代理需监控实际日志及0/1/25/50/100快照、两个rank/launcher终态，再CPU finish和小报告推送。无独立持久观察器；接手查同job状态，不重复启动。全部旧资产、失败与历史费用保留。
