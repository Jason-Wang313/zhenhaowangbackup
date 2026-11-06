# Blur固定残差幅度对照：单节点两卡 — 2026-11-06

先行规则提交1618439：轻量任务默认单节点。当前使用新node0（10.244.211.17）两张H100，nodes=[0]，node1/2待命。

已知当前部署Blur150图PSNR21.546762、LPIPS.202622；输入21.863903/.353603。恢复改善LPIPS却损失PSNR。
历史2026-09小训练集alpha扫描和16来源末四步训练不能当作当前官方证据；后者明显过拟合，不直接复跑。
本轮先用现成官方输出检查恢复幅度的权衡，避免再次耗时运行FLUX。

固定四组：saved uint8输出相对LQ残差的0.5、0.75、1.25、1.5倍，整型分子运算、round-to-even、clip0..255再uint8。
仅按既有conditional_v4预测anchor=blur启用（150图），不用GT类别选路；其余450图完全复用原部署图像及评分身份。
先生成所有候选，再读GT计算原pyiqa PSNR/LPIPS/SSIM；禁止逐图选最佳、GT进loss/梯度，禁止自动选部署候选。
这不是原始连续FLUX输出的alpha重算：已保存PNG先前已量化/拼接，本轮明确为post-uint8残差幅度对照。
不训练、不重跑路由/FLUX、不读其它测试集。每一固定组均报告完整官方600图及原十项门槛；反复使用官方集，非独立盲测。

源码入口smoke/audit_foundir_fullstack_blur_strength_v1.py，GPU shell smoke/run_foundir_fullstack_blur_strength_v1.sh。
CPU prepare通过：600行与原结果/路由身份一致；5项数值/完整性工程测试与shell语法通过。真实GPU执行待发布后提交。
输出checkpoints/foundir_fullstack_blur_strength_v1；新候选PNG及逐图结果本地保存，不推GitHub；原图/模型/缓存不修改。
预期仅数分钟，以实际耗时为准；不是完整FLUX重新生成。两worker完成、node0退出0后CPU finish汇总并推小报告及耗时。
若四组均不达标，只能说明这四个幅度没有解决本轮缺口，不能外推所有连续系数或所有FLUX路线不可能。
