# Blur固定1024来源十遍曝光：当前node1/node3，两节点四H100

本文件替代本轮未提交的六卡运行安排。原3x2代码/CPU预检及提交10be490全部保留，六卡GPU job从未提交。启动前node0/node2心跳曾中断约9分钟，未有明确退出记录，不能宣称实例已停止；随后node0恢复、node2仍陈旧。node1和node3连续新鲜idle，故本轮固定采用两台，不等待不稳定节点，也不在运行中更改world。

科学范围保持：原1024训练来源/512裁图/参考缓存、原完整step200模型与Adam继续到唯一最终1280；新增1080 optimizer steps，总10240曝光、每来源正好10遍。global8、loss=10*MSE_RGB01+.1*pyiqa LPIPS、lr1e-5、wd0、clip1、原生32,440,320 LoRA参数、FP32 LoRA/Adam/VAE/latent及BF16 FLUX均不变。不从固定8图拟合权重启动，不修改energy或其它类别。仅最终1280做完整官方Blur150，25.9/.18为唯一质量门槛，不增加其它测试gate，不选中途最好检查点。

物理node1 IP10.244.47.55对应torchrun node_rank0；物理node3 IP10.244.90.116对应node_rank1，master10.244.47.55:29879。每节点两H100，每rank每step两个真实样本，loss/8后flat SUM，不额外除world。原world2→4精确保留模型、Adam矩/step与采样位置1600，四rank RNG按2026110714+rank重建，**不是逐位等价父两卡续训**；旧两rank RNG/资产保持。迁移后同四卡checkpoint完整恢复，202后真实GPU加载保存的Adam/RNG再继续203。

为适配现有执行器而不改其绑定源码，新成对控制器分别提交nodes=[1]和nodes=[3]两个子job，后者使用node3 extension入口；从不向旧worker发送nodes=[1,3]或四节点job。两个子job启动同一receipt/commit的两节点torchrun，由新控制器统一等待、核验和取消失败的同组peer。这是新增的两节点协同适配，CPU已验收，真实NCCL仍须看四rank日志。故障只停止本轮匹配的子job，旧资产保留，不自动重试。

CPU预检11.341932秒/退出0/GPU0：原step200实际SHA/模型/Adam/RNG/采样核验，1024缓存元数据和文件存在，实际tensor SHA逐次读取时验证。8项针对性CPU测试通过：完整十遍采样、参数/Adam迁移不误恢复旧rank RNG、真实四进程Gloo全局batch8/Adam对照与同world精确恢复、官方150四路完整分片与同一输出双门槛、成对提交/peer失败/控制器失败。语法通过。官方metadata plan每rank38/38/37/37张、570/570/555/555块，共150/2250，4500模型调用。CPU/Gloo不冒充物理NCCL验收。

训练脚本smoke/train_foundir_blur_exposure_2x2_v2.py，launcher run_foundir_fullstack_blur_exposure_2x2_v2.sh；输出checkpoints/foundir_blur_exposure_2x2_v2。官方脚本evaluate_foundir_blur_exposure_official_2x2_v2.py，对应launcher，输出foundir_blur_exposure_official_2x2_v2。子job为round14b_blur_exposure_{train,official}_v2_node{1,3}。

源码发布并核验后，启动smoke/watch_foundir_blur_exposure_2x2_v2_cpu.py monitor --commit <SHA>及独立upload进程。持久链负责训练→CPU finish→最终权重导出/预检→官方150→CPU核验→推报告。201/512/1024/1280实际checkpoint/Adam/四rank RNG检查并推小报告；GitHub超时只延迟上传，不阻塞GPU。进程PID及commit记录在输出根daemons.json，日志logs/foundir_blur_exposure_2x2_v2_{monitor,upload}.log，接手先核对这些及实际GPU queue/step，勿重复启动。

预计四卡训练45–90分钟、官方8–15分钟，约3.5–7 GPU-hours，均未实测，应以真实首段吞吐更新。新增逐步checkpoint约392GiB，预检至少500GiB可用；不上传权重或清理旧资产。无累计GPU上限，但本轮固定1280+完整150，不自动追加旧步数或切类别。官方已多轮用于开发，非独立盲测，GT仅输出固定后评分，不参与参数/阈值拟合或逐图选择。
