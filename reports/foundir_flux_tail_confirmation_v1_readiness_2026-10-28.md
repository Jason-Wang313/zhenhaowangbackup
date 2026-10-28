# Noise FLUX：固定方案在另外24个来源上确认

## 执行范围

用户已授权执行这项确认。上一轮八个训练来源的 GT 诊断中，固定
`mixed_psnr_budget_step16` 达到 PSNR +.7009305954 dB、LPIPS 降低
27.273115%。现在检查它是否只对那八张有效，不再换方向、步幅、排序或预算。

这仍然使用真实答案 GT 来求梯度、比较候选、接受或停止；没有训练 energy，
不是可部署结果、VLM贡献或最终泛化评价。成功也不自动授权正式训练。
不恢复 Restormer residual、最终像素下降、MoE 或 EBT Generalist。

## 数据与原实现

- 原缓存 `checkpoints/foundir_flux_energy_qualification_v1/qualification_v1`，
  固定索引8–31，24张、24个训练来源。排除旧诊断索引0–7。
- 预检核验32条原记录全部为 training Noise，并检查 record ID、source group、
  GT SHA256、LQ SHA256各自唯一；确认集与旧八图来源及图像内容身份不重叠。
- 旧25步工程 energy 资格曾用过完整32张。因此它们不是“从未使用过的测试集”，
  只能称与八图诊断来源分开的训练集机制确认。
- 预检不解码图片；运行只读这24张的既有训练缓存。全部测试/封存像素及最终
  1,859条开发集不访问。旧八图只读身份元数据，不混入均值。
- 直接调用已哈希绑定 V3 的 `diagnose_record` 与 `validate_trajectory`，不复制
  或重写梯度、FLUX rollout、候选接受及量化逻辑；V1/V2/V3源码、配置、结果不修改。

V3结果SHA256：
`06eed91ab7c2c0942b221a14d4c295d1e269bf57b17a8d79354ec9be074d24e4`。
父缓存preflight SHA256：
`b09c57e2af4b7732504944e39222fce73aa4011ecab2dfbff47c5cdd57065cff`。
每次启动继续检查实际权重、缓存、提示上下文及源码配置身份。

## 唯一冻结方案

Noise step400 FLUX、512裁图、原seed、50步、CFG1、reference time10、alpha=.30、
两次uint8量化全部不变。基线必须逐像素复现历史入口和缓存，否则停止。
仅修改索引48的FP32 latent，用冻结BF16 FLUX真实执行余下两步，不改模型权重。

连续代理求MSE和LPIPS梯度，各自RMS归一化后，按LPIPS权重
0/.25/.5/.75/1混合并再次归一化。步幅固定.0005/.001/.002/.005/.01，
每轮最多25个候选，最多16轮。实际评分和接受均看双量化终图。

尚未达到原基线+.30dB时保留PSNR进度；达到后只允许用超出+.30dB的余量换
LPIPS改善。LPIPS不得恶化且至少一项严格改善；无合格候选才停止，达标不早停。
候选先缩小PSNR目标缺口，再比较LPIPS、PSNR、步幅、固定方向顺序。
未接受更新时保留FP32无引导结果，不用GT事后换回历史图来美化均值。

## 预先固定的报告方式

主终点唯一为 `mixed_psnr_budget_step16`。所有24图一起算PSNR平均值及LPIPS平均值，
相对这24图自己的历史部署FLUX基线判断：PSNR至少+.30dB，LPIPS至少降低
9.928107050377874%。不沿用旧八图绝对分数，不逐图挑最好checkpoint，不重新挑方案。

1/2/4/8/16轮均保留描述性报告，加历史BF16与FP32无引导两组对照，共7项。
提前停止的图保持停止时结果直至16轮报告，不丢图。另报告逐图双目标通过率和退步项。
全部候选、拒绝、梯度、FLUX前向/重算/反向/VAE成本都保留并重新核验。
“16次更新”不等于“16次前向”，不宣称等计算量或廉价部署优势。
本轮不另行生成定性图像，不凭LPIPS改善保证视觉自然。

## 硬件、预算及启动

单节点同型号4/8×H100/H200，每卡至少75GiB可见显存，不混卡、不跨节点。
四卡每rank顺序六张，八卡每rank三张，均完整24张、不改变单卡batch。
含空闲显存检查、节点锁、fresh ID、防覆盖、超时和终态独立核验。

参考前轮四卡H100，暂估15–30分钟（.25–.50普通小时，1–2使用GPU-hours），
这是粗估，不是新24图实测。若独占八卡整机仅用四卡，另计2–4整机GPU-hours。
硬预算：四卡1小时或八卡.5小时，均4使用GPU-hours，外加最多30秒清理。
独占八卡仅用四卡时，预算上限对应8整机GPU-hours；是否独占需实际确认。
八卡/H200尚无本任务实测ETA，不能从兼容代码声称已经GPU验收。

四卡入口：

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/start_foundir_flux_tail_confirmation_v1_tmux.sh confirm24_4gpu_v1 4
```

八卡用独立新ID `confirm24_8gpu_v1 8`。二选一，不同时重复跑两套。
日志：`logs/foundir_flux_tail_confirmation_v1_<run_id>.log`。
输出：`checkpoints/foundir_flux_tail_confirmation_v1/<run_id>/`。
终态必须同时有 `CONFIRMATION_RESULT_VERIFIED` 和 `CONFIRMATION_LAUNCH_END status=0`；
tmux创建本身不等于GPU计算已开始。

## 当前验收状态

新Python与shell静态检查通过；34项CPU回归全部通过（含6项新确认测试）。
真实权重、提示上下文及全部24缓存的四卡、八卡预检均通过，没有解码原图。
配置SHA256：`c3ac68f96f2231a434d90b5fa733948d3c9dbe3bb08bfc8f86f839e4b97d3dcf`。
Codex节点无GPU，本轮GPU-hours=0，新GPU任务未启动；不自动启动后续energy训练。
