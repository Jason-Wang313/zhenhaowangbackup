# V2 image-only 第一次验收启动失败：NCCL 初始化，尚未训练

## 已确认的结果

- 实验：`foundir_rain_blur_joint_vlm_moe_v2/image_only/qualification_v1`。
- 节点：`eight-h200-copy--5653e415ed62-rbdkwcl3k7`。
- 启动器已识别8张 NVIDIA H200；不是未分配到八卡。
- 开始：2026-10-26T18:57:06Z；结束：2026-10-26T18:58:49Z。
- 退出状态1；103秒，即0.02861111111111111普通小时、0.2288888888888889 allocated GPU-hours。
- 所有 rank 在 `dist.init_process_group('nccl', device_id=device)` 失败。
- NCCL 2.26.2；CUDA401：`the operation cannot be performed in the present state`。
- 没有达到模型加载、训练第1步、checkpoint或评估。输出目录尚未创建。
- 训练器在通信初始化之后才开始内部计时，因此没有有效训练器用时或 result.json。
- `pkg_resources` 弃用警告不是此次致命错误。
- 没有新 PSNR／LPIPS，也没有科学质量结论；不能说新方案已失败。

## 原因边界与历史证据

当前日志能定位到 CUDA／NCCL 初始化，但未启用详细 NCCL transport 日志，
不能仅凭401错误码确认具体 CUDA 调用或直接归咎于坏卡。

项目在2026-09-26的H200恢复中曾通过详细日志定位到 `transport/nvls.cc`
的相同401错误。仅禁用P2P没有解决；设置 `NCCL_NVLS_ENABLE=0`、
保留 `NCCL_P2P_DISABLE=0` 后，八卡无更新通信／反向传播检查通过，
随后原训练也实际推进。证据见当前handoff中的
`2026-09-26 Low-light+Haze NCCL recovery correction`。

新启动器 `smoke/run_foundir_rain_blur_joint_vlm_moe_v2_8gpu.sh`
没有显式设置这两个变量。这是新启动器未继承已有兼容处理的遗漏；
当前节点是否也确实是NVLS问题，需要在节点上验证，不能当作已经证实。

## 下一步（尚未执行）

1. 用户确认后，把已有H200兼容设置补入当前实际执行的启动器。
2. 先做不读图片、不更新模型的八卡通信检查；仓库已有
   `smoke/cure_nccl_rehearsal_v1.py` 实现 scalar all-reduce 和小模型DDP反传。
3. 检查通过后，以新ID重试相同25步验收，不变更模型／数据／损失／步数。
4. 保留失败日志，禁止重复使用 `qualification_v1`。

本轮用户请求为监控；只做日志／历史代码核查和结果记录，没有改训练代码、
启动器、环境或自动重新启动GPU任务。没有启动VLM臂或正式训练。
当前Codex shell不具备GPU节点的实时驱动／NCCL环境，不能声称已在该节点修复。

## 身份记录

- 配置：`configs/foundir_rain_blur_joint_vlm_moe_v2.json`。
- 配置SHA-256：`3bdd59b3fb3c010cc7a93ab299f3f67fe85cc5eb5a6f380f610d4338398baebe`。
- 训练器SHA-256：`bc55e9d120ee26830b9aa1a919ee776fbbef582b839668253029b58b8a01e5ee`。
- 日志：`logs/foundir_rain_blur_joint_vlm_moe_v2_image_only_qualification_v1.log`。
- 日志SHA-256：`bbee8607da8ed02d3b6befff33c01e0339f772da31b9f6ca703d33521493cfb8`。
- 启动代码Git提交：`d174766ca96a866c0911902dae55723817bd75a1`。
- 输出目录／权重SHA：不存在；不得填写虚构值。
- 本轮未读取任何official／sealed／internal／parent-test像素。
