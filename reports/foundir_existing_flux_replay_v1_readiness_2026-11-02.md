# 四类已有FLUX：完整50步单选重放验收

> 2026-11-02 资源纠正：下文单节点4/8卡入口已废止，不要执行旧命令。
> 用户当前要求四节点×两卡，总八卡；科学协议保持不变。最新执行说明见
> `foundir_existing_flux_replay_4node_readiness_2026-11-02.md`。旧CPU证据保留为历史，
> 不把旧单节点测试报告当四节点CUDA资格。本文件下文仅供方案沿革参考。

## 本次要验证什么

用户授权继续执行。目的地CPU资产审计已在39e4575通过，主代理也已核对报告SHA、
13项源码绑定及全部资产摘要。retention历史日志差异已由目的地核实为CR换行归一化，
原始文件SHA均一致；旧报告不改写，不再把这个已解释问题当新任务阻碍。

当前交付新的GPU工程入口，尚未运行。比较同环境、同图、同prompt/seed/crop的：

1. 原FLUX单LoRA＋原50步采样；
2. 同一共享底座安装四个已有LoRA，仅固定选择同一个专家＋同样50步采样。

这次不训练、不读GT、不算PSNR/LPIPS、不启动Rain适配，不触碰另一代理energy。
检查的是“装入bank后有没有破坏原能力”，不是新质量或VLM贡献。
原历史SOW结果仍保留；本重放没有宣称复现历史环境的每个像素。

## 固定协议

- 四类Blur/Noise/Haze/Low-light，各32张、32个训练来源，共128张。
- 只从原manifest的train记录中按固定哈希选取，每个来源最多一张；排除全manifest
  非训练来源及同GT哈希，最短边≥512。不按图像效果挑样本，不读取最终1859开发像素。
- 保留每任务历史crop seed与`full_development_crop`算法、inference seed namespace、
  50步official shifted schedule、CFG1、reference time10、任务prompt、BF16权重。
- 输入只读LQ并核验其文件SHA。运行收据只携带GT哈希身份，不携带GT路径，禁止解码GT。
- Blur/.75、Noise/.30、Low-light/.925按原顺序先quantize模型输出，再与LQ混合并quantize；
  Haze只做一次原quantize。输出alpha不等于LoRA alpha32，不改两者含义。
- 全部4个flow LoRA实际装入50个原位置；选择固定专家索引，仅执行该专家的delta。
  原成功Rain不进入bank。固定选择不是已实现/验收动态VLM router或soft混合。
- 共同GT、LPIPS网络和Restormer都不加载。FLUX原生VAE保持原调用。

每张要求浮点decoded RGB最大绝对差≤1e-6，并且最终uint8逐像素完全相同。
任一图不通过，总结果失败，保留两份输出和差异；不自动放宽阈值或用平均值掩盖。
总汇聚必须128/128，无漏样本/重复，并从保存的tensor重新计算差异，不只信日志。

## 提示词与显存

先用GPU0短时生成四个固定prompt缓存：使用已验证的离线Qwen FP8权重→BF16
dequantize路径，`local_files_only=True`，不依赖在线deep-gemm核。绑定Qwen全部九项
模型/配置/tokenizer文件SHA；每个prompt单独保存。Qwen进程结束后才启动FLUX重放。
Qwen此处只是FLUX的原文本编码器，不是正在学习的新VLM router。
原入口和bank共享本次同一prompt缓存；没有历史四类缓存的逐字节复现证据，不能冒称有。

每rank只常驻一个FLUX和一个VAE。先保存该rank全部原入口结果，再拆掉原LoRA包装，
在相同底座tensor上装入四分支bank；不同时加载两个4B底座。
4卡：每卡负责一类32图；8卡：每类两卡、每卡16图。总记录集合不变。
torchrun仅用于并行推理进程，没有训练/DDP梯度通信，不使用跨节点 rendezvous。

## 恢复、绑定与时间上限

每张原输出和bank输出保存decoded FP32、最终uint8、SHA与不可变收据；文件fsync，
收据用临时文件＋原子链接发布并fsync目录。半写/无收据tensor仅作为孤立文件保留，
不会当作完成，也不覆盖。重试使用新唯一tensor名字。
同run-ID/同卡数/同运行环境可核验并跳过完成记录；未完成的一张需重算。
不同world size或GPU型号/torch/CUDA/cuDNN/driver/transformers变化拒绝混合恢复。
生产源码、所有FLUX第三方.py、配置、权重、选择记录和prompt均绑定；每次prepare重验
实际模型SHA，运行阶段检查size/mtime是否改变。旧结果和原成功权重永不覆盖。

单节点同型号4/8×H100/H200，≥75GiB/卡；默认8卡，不使用旧4节点×2卡入口。
启动器检测实际GPU、其他compute进程和显存占用；加节点锁/共享run锁，不自动杀进程。
至少5GiB空闲存储。每次尝试在`attempts/`写状态和allocated GPU-seconds，重试成本累加；
prompt短阶段只有GPU0计算，但按整次分配的4/8卡计费，不能隐藏空闲卡成本。

未实测粗估：8卡0.5–1.5小时／4–12 allocated GPU-hours；4卡1–3小时／同GPU-hours，
不承诺精确2倍。硬cap：8卡2小时、4卡4小时，均16 GPU-hours，另最多30秒清理。
超时保存已完成图片，不自动重启、扩预算或进入Rain训练。真实耗时以END与attempts为准。

## 交付文件与目的地操作

### 本地已验证，GPU仍未验证

23项CPU测试通过（71.136秒，不含Python/torch导入），包括原算子12项、4/8分片同集合、
来源/同GT隔离、量化顺序、不可变发布、损坏收据拒绝、同world恢复限制和汇总独立重算。
Python AST/JSON、两份shell的`bash -n`通过。测试中的4图模拟结果不是GPU恢复结果。
真实旧资产只读4/8两路预检通过：16项运行资产哈希、全部128个训练LQ路径存在，未解码图片。
4/8选择摘要均为`2682b41f653d2e3bc69d1a5964895da8e1cf20f0d7cec0d08762b5dc52c03b36`。
完整证据：`reports/foundir_existing_flux_replay_v1_cpu_preflight_2026-11-02.json`。
所有预检源码绑定已再次与最终文件核对；旧资产源通过不是本次新GPU节点通过。
CUDA显存、完整50步差异、真实中断恢复和吞吐均待用户GPU实际运行，不外推CPU结果。

- 配置：`configs/foundir_existing_flux_replay_v1.json`
- CPU契约：`smoke/existing_flux_replay_contract_v1.py`
- 执行/核验：`smoke/replay_foundir_existing_flux_experts_v1.py`
- GPU：`smoke/run_foundir_existing_flux_replay_v1_gpu.sh`
- tmux：`smoke/start_foundir_existing_flux_replay_v1_tmux.sh`
- 测试：`tests/test_existing_flux_replay_v1.py`

目的地CPU代理：安全同步明确文件，不覆盖本地PPT/agents修改或另一代理energy。
使用39e4575已核验的`<新根>/.conda/gpu-train-stable/bin/python`，或明确验证的PROJECT_PYTHON。
先运行CPU测试与`preflight --gpus 4/8`；同一selection_digest必须相同，源码绑定检查通过。
新GPU根仍是`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
GPU脚本拒绝旧根执行；CPU可显式对旧根做只读资产预检，不解码任何图片。

确认目的地CPU通过后，由目的地代理给用户GPU命令，不由主代理假称已启动：

```bash
bash smoke/start_foundir_existing_flux_replay_v1_tmux.sh replay8_v1 8
```

四卡使用独立ID `replay4_v1 4`。同ID重试仅恢复本任务，不恢复任何旧训练。
日志：`logs/foundir_existing_flux_replay_v1_<run-id>.log`。
输出：`checkpoints/foundir_existing_flux_replay_v1/<run-id>/`。
必须同时看到`EXPERT_REPLAY_RESULT_VERIFIED pass=True records=128`和`EXPERT_REPLAY_END status=0`。
tmux会话创建不等于CUDA已启动。完成后只回传小汇总/收据身份/成本，不上传tensor或实时日志。

## 后续

通过只证明静态单选继承，不证明专家互补或混合退化恢复有效。
共同提示词/输出处理与Rain多步适配仍需冻结并实现。Rain教师保留原一步模型，
未来配对GT仍为主监督，不能把endpoint速度直接当所有时间点的flow标签。
原retention1000步不启动，不自动训练或开展匹配消融。
