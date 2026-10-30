# V3 直接推进版：本次只同步到共享工作区，不启动 GPU

## 用户最新决定

停止重复历史基线诊断/额外迁移验收，直接推进V3。主代理完成代码并推GitHub；
新代理只同步代码到CPU/GPU共用的新工作区、核对文件，不自行训练，也不提供GPU启动命令。
旧报告的“尚未批准取消历史相等检查”已被这次用户决定覆盖。

唯一执行根仍是：
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
主代理的隔离Git工作树只是编辑/提交位置，不是接收方同步目的地。

## 改了什么

1. V3的新适配层调用原FrozenFlow时不再传入旧环境expected_proposal。
   历史图的差异与SHA直接落日志和结果，不作为启动关卡。旧FrozenFlow/双uint8算法未改。
2. 当前run原始FLUX输出既作为energy的proposal条件，又作为成对训练的参考候选。
   该参考的分数和真实图像导数一同保存/替换，避免条件与标签不对应。
3. 其余历史候选仍可使用：它们的真实质量/导数由candidate和训练GT决定，不依赖
   proposal。旧生成候选不用重跑；新派生缓存中重新计算标签，旧缓存完全保留。
4. 当前参考和初始反例缓存位于每个run的mining/bootstrap，不跨硬件/任务共享。
   相同run刷新/恢复仍核对当前参考；当前native入口与BF16 tail一致性仍由旧runtime检查。
5. probe保存历史、当前original、FP32 control、raw、bounded五组结果；对三种
   对照分别算增益。`psnr_gain`的主分母是当前original；历史和FP32差异不能被隐藏。
6. 支持1x2、1x4、1x8、3x2；H100/H200均可。默认短任务1x2，全局32对不变，
   两rank各16对、每对乘world/32后DDP平均。单节点不会继承旧NNODES/MASTER_ADDR。

模型、数据来源24张、probe训练来源8张、crop512、种子、50步、alpha=.30、
双uint8、损失/学习率、8步资格(step4刷新)、独立200步拟合及最终两指标目标不变。
没有加入新restoration网络，没有访问保护像素，没有宣称训练/泛化已经成功。
每完整optimizer step仍保存模型、Adam、RNG和采样进度，拓扑/设备映射改变不能冒充精确恢复。

新输出根：`checkpoints/foundir_flux_tail_relative_energy_v3_direct`。
新默认ID：`qualification_1x2_direct_v1`。不覆盖/续训旧cluster失败目录。
新run必须fresh模型/Adam；先前失败0训练步，没有optimizer进度可继承。
8步成功也不自动启动200步。已有12 used GPU-hours资格硬上限按卡数换算，
2卡6小时、4卡3小时、6卡2小时、8卡1.5小时；这是保护上限，不是实测ETA。
本次没有新GPU任务，CUDA链路/真实吞吐仍由后续用户启动的任务验证。

## 已完成的本地 CPU 检查

49项相关单元测试通过（测试体4.234秒，不含环境导入），包含：历史偏差不阻断、
当前run参考及native/tail错误仍会报错、新proposal条件与参考候选/标签/导数一致、
不会把环境收益算成energy收益、2/4/6/8分配与全局均值、旧跨节点变量不会污染两卡入口、
数据角色限制和保存恢复。两进程真实Gloo二阶反传与每rank RNG/Adam恢复测试也通过。
shell语法检查通过。这些都是CPU工程检查，不是新CUDA运行成功或科学指标。

## CPU代理应怎样同步

仓库：`https://github.com/Jason-Wang313/zhenhaowangbackup`，`origin/main`。
以主代理交付的**固定commit**为准；先fetch，确认该commit存在，再从其读取
`reports/foundir_flux_tail_v3_direct_sync_manifest_2026-10-30.json`。
清单只包含明确允许同步的文件及SHA，不含checkpoint、图像、数据、缓存或活日志。

- 检查共享目标根的dirty状态及运行情况。只同步清单文件，不pull/reset/clean全仓库。
- 若本地与目标不同，先保留该文件的备份/补丁；能确认是旧版本的按清单更新。
  遇到不明新修改或冲突，保留并报告具体路径，不覆盖用户工作、不自行改实验方案。
- `AGENTS.md`/`agents.md`中 unrelated本地内容同样保留。若需三方合并，报告合并后的
  本地SHA和不等于目标SHA的原因；绑定代码/配置不能自行合并出另一个版本。
- 真正更新的是上述共享项目根，不是/tmp worktree或CPU私有目录。清单中部分
  未改动的启动器/保存代码也要核对，以确认GPU将读取同一套文件。
- 核验每个绑定代码/配置的SHA与清单一致；可以做AST/JSON/bash语法检查和CPU单元测试。
  不运行prepare/pipeline、基线replay/生成、任何CUDA检查或训练；不改旧cache contract。
  新run的preflight/cache metadata由未来正常训练入口创建，不提前覆盖旧记录。
- 只用新根已验收的 `.conda/gpu-train-stable/bin/python` 做Python检查，不改环境依赖。
- 写一个小型同步报告：目标commit、实际共享根、逐文件SHA/一致性、备份/冲突、CPU
  检查结果、是否启动GPU（必须false）。报告不包含图片/权重/活日志。
- 在新代理可访问范围内确认共享挂载。若无法直接访问GPU节点，只说“CPU侧共享根
  已同步，GPU端尚未直接核验”；不能把/tmp内测试成功说成GPU节点已看到文件。
- 按focused push规则仅推同步报告，fetch防并发、显式stage、扫描secret、核验远端commit。
  不重新提交主代理已推的代码，不另开研究分支，不返回GPU启动命令。

同步完成后，等待用户下一条启动指令。下一步是已准备好的训练任务，不是重复诊断。
