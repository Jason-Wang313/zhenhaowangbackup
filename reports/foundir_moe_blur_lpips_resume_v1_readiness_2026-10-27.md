# 新节点恢复MoE修改版：H100/H200兼容

用户希望迁移到两个新节点，并明确要求两者均可使用8×H100或8×H200。
核查后，PPT fixed32/300步已经正常结束；原配方MoE也完成。
只有MoE的blur_lpips4x未完成，因此现在只需一个新节点恢复它，第二节点暂留给PPT下一阶段。
不为了占满节点而重跑已完成任务，不自动开始新的PPT正式训练。

## 精确恢复范围

- 最后日志：10750；最后完整权重/Adam/RNG：10500。10750不是可恢复检查点。
- 从10500恢复到冻结终点11000，共500步；未保存的10501–10750需要重算。
- 使用原blur_lpips4x模型，不从control输出或原step10000重新训练。
- 严格恢复Adam动量、step、参数顺序、学习率、每rank RNG和样本位置336000。
- 不改数据、裁块、D4、loss、学习率、架构、router或最终评估协议。
- 更频繁保存：10600/10700/10800/10900/11000；只是中断保护，不增加验证或挑最优。
- 原结果目录保持不动，新目录为`checkpoints/foundir_moe_blur_lpips_resume_v1/<run_id>`。
- 同型号单节点8卡，支持H100>=75GiB和H200；两种型号不能混在一次DDP内，不跨节点。

父receipt SHA：`615027865b805994cc0af1fe4216839c1c32f6dc6e4765832134af318c92716f`
10500权重SHA：`d6a9fd0e95b74da5098d039469bbb74a7147b32248809b7426e52e2433efc20d`
10500 Adam SHA：`2bce7801d95b20056d111beba3fcff79591df4a41871833ca1369328107fcb64`
迁移配置SHA：`39afd8616ada4b8a22cac7e41d7453e999ce81f3e5cab161693cf26e8f462f4d`

## 旧节点状态与新节点安全

截至16:41Z，旧日志仍停在约13:14Z的10750，无结束标记。
日志停更不等于远程进程已退出。Codex没有旧GPU节点的执行通道，未杀任何进程。
入口要求用户确认旧任务已停止或旧节点已关闭，再设置`OLD_MOE_STOPPED=1`。
这是一条用户确认，不会伪装成自动验证了远端PID。
如果原输出随后出现完成result，新入口会拒绝重复恢复。
新节点用本地双任务锁、空闲显存检查、有界NCCL测试与新ID防覆盖。

## 可比性与数据顺序

原control全程在旧H200；修改版前500步旧H200、后500步可能是新H100或H200。
因此最终比较明确标记迁移与GPU型号变化，不能叫原先严格同节点消融，
也不能承诺不同GPU上逐bit复现。绝对PSNR/LPIPS仍按原开发协议报告。
保留原严格比较器不变，另设迁移比较器，不通过伪造hardware字段绕过原约束。

新节点先在10500上重算完整Rain100/Blur2595，记录恢复后的数值；
11000再次完整验证，不用恢复时的差异调超参。
前500步采样摘要按冻结metadata调度重建，后500步以实际DataLoader记录追加，
最后与已完成control的全部1000步逐rank摘要比较。前半摘要是重建，不谎称是旧程序记录。
配置、旧源码、数据、父权重/Adam身份与优化器设置仍要求严格匹配。
比较继续区分LPIPS改善、绝对双任务达标与VLM/论文创新。

## 命令与预算

确认旧任务已停止后，在任意一台空闲8×H100/H200节点执行一次：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && OLD_MOE_STOPPED=1 bash smoke/start_foundir_moe_blur_lpips_resume_v1_tmux.sh resume_v1
```

日志：`logs/foundir_moe_blur_lpips_resume_v1_resume_v1.log`。
预计.5–.9普通小时、4.0–7.2 allocated GPU-hours，包含恢复/五次保存/两次完整评估的暂估。
Codex没有GPU，未声称已经启动。新文件独立新增，不修改旧MoE/PPT绑定代码。

13项CPU测试通过（6项迁移测试及7项原续训回归），包括采样摘要前缀恢复、
跨H100/H200比较明确标注、禁止改变科学配方、旧源码/凭证绑定、八卡与停止确认入口。
真实10500权重/Adam的完整SHA、旧control/launch、冻结模型与数据元信息预检查通过，
零图片读取。Python AST/JSON与shell语法通过。CUDA恢复仍需实际运行确认。
