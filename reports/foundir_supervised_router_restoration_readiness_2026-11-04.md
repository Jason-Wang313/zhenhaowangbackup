# 匹配路由画质对照：只补四个缺失输出

## 目的与已知事实

CPU路由训练已完成，报告提交23ac684。两组同初始化/顺序/400步/global32/lr.001，
仅区别是否从第一步启用VLM修正。图像组33/40正确，VLM组38/40正确。
VLM仍错0007534（Blur→Haze）和0479048（Low-light→Noise），两者预测概率都接近1；
不能把高置信度当正确保证，也不能用分类改善替代恢复指标。

本轮只回答：两组采用同样预测top-1规则后，PSNR/LPIPS实际相差多少？
不重新训练、不更新FLUX底座/五个LoRA/VAE/Qwen、不增加CNN/Restormer/refiner。
不是混合退化MoE方案已经成功，也不要求此轮同时解决混合退化。

## 为什么只生成四个输出

两组各40图，共80条预测。相同图、相同预测专家对应同一个输出，合计47条不同路由。
已有40个正确类别单选参考，以及此前4个错误主专家单选输出；其中43条用于当前预测，
可覆盖76/80条预测。参考与旧输出的存在不能让GT参与新路由：先固定预测，再按图+专家查缓存。
绝不比较指标后选更好的缓存。原40个参考全部仍报告，所以缓存目录44项、新输出4项，
报告artifact目录共48项，但两组实际使用47种路由。这些数不是漏图或多生成。

| 组别 | 缺失输入（完整前缀validation_foundir_train_） | 预测专家 |
| --- | --- | --- |
| image_only | 0009805、0391222、0377282 | Haze |
| image_plus_vlm | 0007534 | Haze |

四个新输出沿用Haze原prompt、50步、alpha1、原seed/512裁图/量化；不改Rain26调用或其它协议。
模型不读取GT选择专家；GT只在输出完成后算原torchmetrics Alex LPIPS与PSNR。
缓存核验JSON、输出SHA、原receipt、输入身份及native route；新记录绑定代码与数值运行栈。
报告两组各40图、各类8图、每图和配对差异；同时保留正确类别参考及旧step408 soft。
比较旧soft同时改变训练和选法，不能将全部收益归于VLM。40图/单seed开发结果不是官方门槛。

## 运行与恢复

- run-ID：matched_top1_four_missing_v1。
- 唯一根：/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。
- Python：该根下.conda/gpu-train-stable/bin/python，不使用旧HDD3或系统Python执行GPU。
- 单节点两张H200，每卡至少139GiB；旧完整纯推理峰值137.93256GiB，H10080GB明确拒绝。
  这是本轻量任务资源例外，不是擅自把后续正式训练缩成两卡。不得声称H100已通过。
- 同软件/指标/内存策略，可记录不同驱动；缓存复用不承诺跨驱动逐像素一致。
  老hardware辅助函数返回world=8用于继承数值签名，实际任务明确记录1节点/2卡，
  torchrun仅两个独立推理worker，无DDP/NCCL训练、无跨节点连接。
- 每卡两个新输出；每图独立原子保存uint8和JSON，恢复跳过已验证记录，孤立tensor保留。
- 全部尝试累计上限900秒（15分钟/.5使用GPU-hour），另最多30秒清理，不是ETA。
  旧四图77秒仅参考；若独占更多卡的整机，还需另计整机资源成本。
- 本地GPU锁+共享run锁避免重复；不杀无关任务。超时不自动重启/扩预算。
  attempts开始/结束记录累计失败成本；硬断导致缺结束账时必须核实补账后再恢复，不清零。

CPU代理同步GitHub并保留无关修改/大资产后先运行（无模型加载/像素解码）：

```bash
.conda/gpu-train-stable/bin/python -S smoke/evaluate_foundir_supervised_router_restoration_v1.py prepare
```

通过后给用户GPU节点tmux入口，不在CPU节点冒称已启动GPU：

```bash
bash smoke/start_foundir_supervised_router_restoration_v1_tmux.sh
```

入口执行prepare→hardware→四图推理→CPU汇总；同一命令用于中断后恢复，不新建run-ID。
输出：checkpoints/foundir_supervised_router_restoration_v1/matched_top1_four_missing_v1。
日志：logs/foundir_supervised_router_restoration_v1.log。
成功必须同时有MATCHED_RESTORATION_RESULT_VERIFIED和MATCHED_RESTORATION_END status=0。
tmux创建不是GPU启动成功。只回传小JSON及耗时/退出报告，禁止上传tensor、模型或live logs。
报告：reports/foundir_supervised_router_restoration_terminal_2026-11-04.json。
全部完成后由主代理分析，入口没有后续训练或自动扩展。

## 本地验证与限制

33项CPU测试通过（11项新测试+22项既有回归），包含真实已发布报告的缺失集合、
80条汇总、错路由/重复/缺失/非有限值拒绝、44项缓存身份与SHA、恢复与损坏拒绝、
完整合成汇总及重复执行一致性；Python编译、shell语法、git diff检查通过。
合成测试打印的指标不是真实实验结果。未挂载目的地数据，未运行本轮GPU；
真实缓存preflight和四个新指标必须在目的地完成，不能把本地测试算成GPU验收。
