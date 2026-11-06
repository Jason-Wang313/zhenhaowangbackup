# 第四节点独立worker入口

用户已在第四台实测并回传：notebook `e087f2fc-829f-4a0d-8aa5-5df18b6cd56a`；hostname `qwen-copy-copy-5df18b6cd56a-6gppgovd6s`；IP `10.244.90.116`；GPU0/1均H100 80GB HBM3，各81559MiB、已用1MiB、利用率0%、无compute-apps。共享目录和原Python可用，torch2.7.0+cu128、CUDA可用、两卡。以上为用户节点实测回传；CPU代理尚不能代替真实worker登记UUID或声称它已READY。

第四台登记为node3。保留现有node0–2的v2 worker进程、源码、IP和队列，不要求重新启动它们。新入口`smoke/foundir_fullstack_node3_extension_v1.py`只在本进程内扩展节点表；所有磁盘上的旧执行器源码保持不变。

使用同一共享队列`checkpoints/foundir_fullstack_executor_3x2_h100_v2`，新增`workers/node3/`登记和心跳。新submit仅能生成nodes=[3]的独立任务；旧worker在验证前跳过不含自己的任务，因此不会因node3任务阻塞。禁止向旧worker提交nodes=[0,1,2,3]，它们仍只有三节点校验。以后四节点联合训练需专门适配入口与派发协议，当前不冒称四节点NCCL已支持/实测通过。

这使第四台可供后续单节点双卡工作使用，并不启动当前Rain、Noise/Blur或任何训练。启动时实测IP/型号/显存/UUID，跨全部登记检查UUID不重复；node3本机锁防重复worker。真实GPU任务入口仍须检查GPU空闲和作业锁。

只在第四台终端执行一次：

```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
bash smoke/start_foundir_fullstack_worker_node3_v1_tmux.sh
```

预期`FULLSTACK_NODE_VERIFIED`、`FULLSTACK_WORKER_TMUX_CREATED`；后者不是READY或实验开始。共享日志`logs/foundir_fullstack_worker_node3_v1.log`应出现`FULLSTACK_WORKER_READY node=3`，共享`workers/node3/heartbeat.json`持续更新并记录extension源码SHA。代理需检查这些实际证据。

3项CPU回归和shell语法通过：旧worker对node3任务无副作用地跳过、拒绝四节点混合job、导入扩展不改变原节点表。实际node3登记仍待用户在该节点启动，无GPU实验已提交。

代理以后提交node3独立任务用新工具的submit子命令（没有--nodes参数，固定[3]），其余entry/commit/sources/arguments与v2一致。entry必须支持node3且保持明确独立分片，不能直接复用内部强制node0的脚本。所有相关源码须先推送且本地与指定提交一致；新submit自动绑定扩展与原执行器的SHA。
