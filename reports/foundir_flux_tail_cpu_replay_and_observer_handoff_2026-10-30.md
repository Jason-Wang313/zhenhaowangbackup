# 只读 CPU 核对：不用再跑 GPU

## 更新：真实重放已完成，以下运行命令仅作归档，不要重跑

新节点已在提交 `0b93b95` 回传真实CPU结果。主代理核对JSON SHA、执行脚本与
scalar源码SHA、差异计数关系及CPU/训练状态字段；并未在本地重新读取远端tensor。
完整JSON：`foundir_flux_tail_quantization_cpu_replay_2026-10-30.json`，SHA
`02b4990d0be484ce8dd72e7641153375ea5d5059208db26d7714f506539d40a8`。

结论：**所检查的最终量化计算没有复现分歧；替换取整算法不是本次修复。**

| 实际控制 | 与新GPU输出不同的通道元素 | 与历史图不同的通道元素 |
| --- | ---: | ---: |
| CPU完整两次取整重放 | 0 | 3630 |
| 第二次取整改用CPU归一化输入 | 0 | 3630 |
| 第二次取整改为FP64 | 26579 | 29407 |
| 第二次取整改为精确有理数 | 46143 | 48541 |

第一次取整单独重放同样差异0。旧新3630处差异中1214处位于精确半整数点，
2416处不位于半整数点；不能将全部差异解释为最后一步临界点舍入。
输入归一化CPU与保存GPU值在786429/786432处相差至多1.1920928955078125e-7。
只在第二次取整替换这个输入不会改变结果。该小差异是否通过VAE/FLUX生成放大，
本次没有检验，不能排除也不能当作根因已确定。

因此已排除“当前CPU/GPU两次取整自身不一致”这一具体假设；
没有排除上游数值/运行环境差异，也没有找到旧生成路径最早的分歧点。
旧raw中间态缺失，继续对同一份最终图换舍入方式无法补齐这段证据。
V3在训练前检查处退出，0训练步，不是energy学习失败。

### 主代理的下一步建议（未实现、未启动）

停止重复同一诊断。若用户同意，应单独冻结**新环境匹配对照与缓存迁移契约**：
保留历史图/结果不动；权重、种子、裁图、50步、alpha=.30、双uint8和指标不变；
明确记录新环境未经energy修改的输出，并核查与历史基线的质量差异，不能只因
最大1灰度级就宣称PSNR/LPIPS无影响。候选、proposal条件、标签及导数缓存哪些
能复用、哪些需迁移，必须逐项审计，不重命名旧cache冒充新cache。
energy增益应与同环境零引导对照比较，且保留原历史基线对照，防止换弱分母。
同环境原始入口与tail零引导一致性仍应保留，不一并取消。

这不是批准删除当前断言或开启新GPU实验；当前没有新的运行命令。
也不能仅凭这一来源的两卡重复，声称其它23来源或H100/H200混用已经通过。

## 结论与分工

主代理继续负责分析、改代码、判定下一步和推送。新目录代理只安全同步指定文件、
执行这条固定 CPU 检查，并把小报告推回 GitHub。不是让新代理重新设计实验。
用户后续直接指令优先；未获明确授权不得更换配方、基线或重试 GPU。

主代理可见旧 HDD3 根和本地 CPU 环境，但不可见新根
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
新诊断四份 `.safetensors` 只在新根，不在 GitHub；旧根没有这些文件。
所以可以在本地检查代码和旧资产，但不能声称已经在本地核对新张量。

## 主代理已实际完成

- 本地 CPU 穷举全部65536个 `(LQ灰度, raw_uint8灰度)` 组合：现有归一化FP32
  `alpha=.3` 混合与精确有理数 `round_even((7*LQ+3*raw)/10)` 有1422组相差1级。
  全部在精确半整数取整边界，边界外差异为0。
- 这是使用 `torch 2.7.0+cu128` 的CPU合成算术检查，没有初始化CUDA。
  结果见 `foundir_flux_tail_quantization_cpu_synthetic_2026-10-30.json`。
- 只读核对旧训练index0缓存内容SHA：
  `148c100714395865b7c2ad89ee252f95fa1ccd92751004256290a43351b818b4`，与元数据一致。
  父preflight SHA：`1cb45e9dbb28e4f9fb12d14d7693f5cba7d1e09f393152013d27c5e82d0258af`。
  查看了缓存header；没有解码target/GT或其它来源。
- 旧缓存生产链/现有记录没有提供能与新160阶段trace直接比对的旧raw中间张量。
  不能用“今天本地Python版本”代替“历史GPU运行时完整证明”。
- 新脚本10项CPU测试全部通过（测试体2.130秒，不含环境首次导入）；覆盖精确取整、差值不溢出、逐阶段重放、不修改输入、
  来源/路径/SHA/shape/dtype限制、只解码允许的六个张量、输入缺失报错及不覆盖输出。

**1422组不是新节点实际偏差像素数，也不是画质指标。**它只证明最终浮点取整
确实存在敏感边界。生成过程差异是否也存在，必须读新节点已保存的输入再判断。

## 新节点代理同步什么

先fetch并核对本地/远端差异，保留所有dirty文件和运行资产；不要盲目pull/reset。
本次新增运行文件只有 `smoke/replay_flux_tail_quantization_cpu_v1.py`；
新增测试 `tests/test_flux_tail_quantization_cpu_v1.py`；文档为本报告、synthetic JSON、
AGENTS.md、agents.md、current handoff及设计演变记录。
只同步本次明确文件，不覆盖已绑定的V1/V2/V3训练代码、配置、cache或preflight。
运行依赖 `lowlight_restoration/scalar_residual.py` 的原SHA版本，脚本会检查；
若不符，报告错误，不顺手改文件或hash让检查通过。

## 唯一下一条命令：CPU，不需要GPU节点

在能读到新根现成诊断文件的CPU节点执行这一整行：

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .conda/gpu-train-stable/bin/python -u smoke/replay_flux_tail_quantization_cpu_v1.py --output reports/foundir_flux_tail_quantization_cpu_replay_2026-10-30.json
```

不是GPU命令，不需要torchrun/tmux/节点IP。没有新增GPU预算，使用GPU-hours=0。
脚本最多使用2个CPU线程；首次导入torch/读取共享盘可能较慢，未给定实测ETA。
只创建指定的小JSON，不改任何输入；若输出已存在，保留并检查，不能覆盖旧报告。
如果输入缺失/SHA不符/读取失败，报告具体错误，不自行重跑生成诊断。

固定输入为 `baseline_parity_2gpu_v1` 的hash-bound result/preflight、
`gpu0_repeat0.safetensors`及父训练`cache_000`。四份trace已由上轮核实字节相同，
这里只重放一份，不重复读取全部四份。只解码4个trace阶段和cache的observation/proposal。
完整文件SHA检查会读取文件字节，但不解码或使用target/GT张量。

检查包括：

1. CPU与已保存输入归一化是否一致。
2. 第一次取整重算是否等于已保存raw_uint8。
3. 使用已保存raw_uint8重算第二次取整，及完整两次取整重算。
4. 仅作诊断的FP64/精确有理数控制；统计哪些旧新差异被解释，哪些位置新增差异。
5. 输入SHA、来源、shape/dtype、CPU-only及不覆盖检查。

这些控制不是候选部署算法，没有修改 `alpha=.30`、双uint8或旧baseline。
即使某个控制完全匹配旧图，也不单独证明旧生成过程相同，不能自动换算法。

## 新代理如何回报

运行结束后核对新JSON的 `kind=saved_tensor_cpu_replay`、证据SHA与
`cuda_initialized=false`，并在简短Markdown中写：执行是否完成、第一/第二次
取整差异数、旧新差异有多少落在半整数点、哪个控制匹配/不匹配、是否有错误。
不额外读GT计算指标，不推断PSNR/LPIPS影响，也不自行放宽断言或恢复训练。

遵守AGENTS的focused push：fetch检查并发提交、显式stage新JSON/简短报告，
secret scan与diff检查后commit/push，核验远端hash。不要上传原始tensor、权重、
图片、缓存或整份活日志。主代理收到远端报告后负责判断和下一步代码。

之后GPU训练的报告至少含UTC采集时间、run ID/源代码身份、最后日志更新时间、
optimizer step、最后完整checkpoint、实际metrics/错误与终态标记。只在有可验证
证据时说“运行正常/完成”；GitHub没有更新不能当作远端进程停止的证据。
