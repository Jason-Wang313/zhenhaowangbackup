# 原生50步Rain：换匹配起点，保留训练配方

## 为什么做这一轮

用户授权直接改善旧原生50步Rain，而不是继续把成功的一步endpoint强行转换成flow。
上一轮200步开发28.946285/.099400；一步教师34.226794/.073640。它确实在学习，
但转换后仍远弱于原教师。这个结果不证明50步Rain不可能，也不支持直接追加几千步。
本轮回答更具体的问题：已有多步恢复能力作为起点，在同样数据与目标下是否更好？

原生起点固定source_diverse_pilot step2000，不搜索若干checkpoint择优。
权重SHA：2c4537258783f51ad15163b1a3377ce811d1418d51b4da68b5d75ef502bd45af。
原结果文件SHA：194c57cfaa5860699277979ffa5123c645e63fa4fbf2483d57ed530d7ca67fb9。
旧100图为30.859952170505053/.08347748989239336，未过34.2/.08。
原100图与当前100图的record/source/GT/LQ哈希及裁图一致；原1580条训练来源与它们分离。
本轮仍重算step0：当前固定seed与FP32 LoRA计算不同，不能复用历史数值冒充当前基线。
旧48图identity实验不是本轮100图分母，也不是本轮初始化。

## 实验保持哪些东西不变

- 复制旧50步LoRA；BF16存储值准确提升为FP32，fresh Adam。不复用旧optimizer，
  因为训练数据、损失和学习率已不同；称继续微调权重，不称精确恢复历史优化轨迹。
- FLUX base、VAE冻结，仅新Rain的50处LoRA训练；其余四个专家只读不更新。
- 上一轮完整38334记录/5780来源训练池，均匀选来源再选记录，随机512裁图/D4，
  35条小图reflect padding；相同seed、lr1e-5、global32、micro1×累积4×8rank。
- 原native noise-to-clean目标及flow MSE1、GT latent L1 .25、GT RGB MSE10、
  GT LPIPS .05、冻结一步Rain RGB L1 .1全部不变。一步Rain只提供既有缓存示范，
  不是新模型初始化，不加载Restormer或CNN恢复网络。
- 200步/6400曝光：5012不同记录、3867来源；不是200步应当过线的承诺。
  step0/50/100/200分别评价32训练诊断与100开发，共528份完整50步输出。
  逐图seed、prompt、CFG1/reference10、alpha1、单uint8与上一轮一致。
- 开发集只评分，不反传；两个均值分开，GT latent只对train_update编码。
  official/sealed/internal/parent-test等保护像素不读取。
- 目标PSNR>=34.2且LPIPS<=.08，必须同一个checkpoint。仍报告同图一步教师。
  review_required用于提示开发两指标都未改善，不自动改配方/扩预算/训练MoE。

## 缓存复用与保存

父目录checkpoints/foundir_rain_flow_formal_stage1_v1/formal_stage1_4x2_v1。
父receipt SHA：8b2209f880307af41eec03c571a8c98cf0661f7714ac8d4f63d63cd96dfb75ae。
6532份缓存包括6400训练曝光、32训练诊断和100开发；latent/教师与新学生初始化无关。
不复制、不新命名冒充新生成；新receipt显式记录producer与consumer。
核验父源码/资产/采样身份，逐文件SHA、角色、教师、形状/有限值及评价GT latent为零。
父完整核验超时不能说已通过，但不妨碍独立核验后复用其有效缓存；缺损立即报告，
不暗中换数据/重新生成整池。本地主代理看不到目的地缓存，目的地预检必需。

新输出checkpoints/foundir_rain_native50_warmstart_v1/native50_stage1_4x2_v1。
每步0–200原子保存LoRA/Adam/8rank RNG/采样进度；评价保留RNG，断点先补完评价。
同run_id、fresh共享launch_id恢复；world/runtime/绑定源码变化拒绝，坏快照不静默回退。
正式step200状态可以后续授权后继续，不自动接续。旧全部实验产物保留。
新磁盘约75–85GiB，至少100GiB空闲；另依赖父缓存原位置保持不动。

## GPU结束和完整核验分开

4节点×2H200，总8卡，每卡>=139GiB；H10080GB没有该训练路径资格。
GPU阶段暂估25–55分钟/3.33–7.33GPU-hours，cap每节点60分钟、合计8GPU-hours加清理。
这是未实测估算；虽省缓存生成，528次50步生成及201次保存仍有成本。
四节点GPU_COMPLETE和status0仅表示训练/评价结束，明确CPU_VERIFICATION_PENDING。
随后由CPU代理独立执行完整缓存/201快照/528输出核验，CUDA_VISIBLE_DEVICES为空，
90分钟CPU cap；不再次运行FLUX/LPIPS，不占CUDA显存，不声称重算了LPIPS。
如果平台仍保留GPU节点，资源预约仍可能计费；应释放/复用GPU，不把空闲自动当免费。
CPU核验超时/失败如实报告，不能因此重训或自动扩GPU预算。
旧轮总8.722777778GPU-hours保留在历史账；约1普通小时×8卡不是8普通小时。

## 新工作区代理执行说明

安全同步origin/main，保留本地修改和所有数据/权重/缓存。唯一运行根：
/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang。

先在该目录执行CPU预检：

```bash
.conda/gpu-train-stable/bin/python smoke/train_foundir_rain_native50_warmstart_v1.py preflight --require-cache --report reports/foundir_rain_native50_destination_preflight_v1.json
```

通过后给用户四节点命令，入口smoke/start_foundir_rain_native50_warmstart_v1_tmux.sh，
参数依次node_rank 0/1/2/3、当前node0地址、native50_stage1_4x2_v1、launch_native50_stage1_v1。
端口29688；四节点同run/launch；不复用旧IP。首次任务不是旧step200续训，不能指向旧目录。
若新入口此前已运行/中断，保留run_id，用fresh共用launch_id恢复；不复用旧失败launch。
不重跑25步资格，也不启动energy/MoE/其它分支。预检通过不等于GPU已开始。
在step0/50/100回传简报，终态成功/失败/超时都推小报告到GitHub。

GPU节点全部结束后，在共享该目录的CPU节点执行：

```bash
bash smoke/verify_foundir_rain_native50_warmstart_v1_cpu.sh native50_stage1_4x2_v1 launch_native50_stage1_v1
```

用实际完成的launch_id；必须四节点done才允许核验。最终需要
RAIN_NATIVE50_WARMSTART_RESULT_VERIFIED和CPU_VERIFY_END status0。
报告四个里程碑两组指标、是否过门槛、快照/缓存/输出核验、GPU耗时及CPU核验耗时，
GPU-hours按节点实际分别累加。禁止上传权重、tensor、图片、原始/live日志或凭据。

## 本地验证

19项CPU回归通过，覆盖配置冻结、训练/开发隔离、缓存身份/损坏拒绝、旧100图裁图匹配、
八rank分片、指标算术、201份快照恢复、评价RNG保持和CPU核验拆分。
最终版重复运行19项仍全部通过，测试本体3.428秒，不含共享磁盘依赖启动时间。
两进程CPU Gloo DDP也通过：累积4次，保存后恢复的下一步LoRA/Adam与不中断路径逐元素相同。
这是小模型CPU恢复测试，不是4节点4B FLUX CUDA训练验收。
真实源资产只读预检通过，60份最终运行源码绑定一致；没有读取真实图片像素。
训练选择摘要8e68cdeb04d25b9dc68e9f2fa82a1da9672fd43de21ea3cede2344b174e020fe，
配置SHA 01efcdbd81143d4728fbcc9e369214f9a52640e12c7d1a71ca54e02f30a539e9。
真实旧LoRA加载检查通过：50处、100张量、32,440,320参数；BF16值提升到FP32后逐元素一致，
Adam为空的新状态，原权重SHA未变。使用真实LoRA及生产初始化函数，但base线性层为meta
形状占位，未执行真实4B前向，不把它说成CUDA/恢复质量通过。详见同名JSON报告。
主代理没有新GPU根挂载/控制通道；源资产核对不能替代目的地缓存预检或CUDA验收。
