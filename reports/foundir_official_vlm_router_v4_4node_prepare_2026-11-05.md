# 官方V4四节点适配与依赖修复 — 2026-11-05

已按用户授权安装pyiqa==0.1.16；wheel SHA256为`42bbcf1c274a8b53f040a3ec5ec304a15b7725afb5a85b81050e9cc7e6d09ed4`，与PyPI官方记录一致。采用`--no-deps`，现有375个包版本均未变化；没有覆盖clip模块或升级torch/CUDA/NumPy。仅所需默认PSNR/LPIPS/SSIM模型实际导入及身份核验通过，不声称安装全部无关可选指标依赖。原有官方LPIPS权重复用。

默认4节点×2张H100/H200，8独立worker，无跨节点NCCL；节点内同型号，跨节点软件/数值策略一致。`FOUNDIR_OFFICIAL_NNODES=2`仍支持旧两节点入口。四节点使用独立run `official_vlm_v4_frozen4800_v1_4x2`，保留旧失败记录、资产和本地修改。旧prepare失败43秒继续记账；本次prepare35秒/status0，0 GPU-hours。历史V4成本2.473888889 GPU-hours另列，不清零。

本次prepare已通过：600官方记录，8worker块数2423/2423/2370/2370/2492/2492/2445/2445，合计19460；三组去重后恢复预计19460–58380块，精确值待route后plan。冻结step4800、三组、原图拼块与官方pyiqa默认口径不变。未解码测试像素、未启动GPU，未运行其他测试集。

两节点及四节点各14项合成CPU工程测试通过，shell/Python语法通过。覆盖600图完整分片、相同seed、独立run、固定总预算、8worker完整性/篡改拒绝/去重与失败成本。官方CUDA和完整原图metric显存尚待实测。

四节点route总cap仍12使用GPU-hours，每节点累计5400秒；不是实测ETA。完整恢复未获具体预算批准，不启动render，不自动扩大预算。

四台GPU终端都先执行：
```bash
cd /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
export FOUNDIR_OFFICIAL_NNODES=4
```
各节点只执行自己对应的一行：
```bash
# node0
bash smoke/start_foundir_official_vlm_router_v4_tmux.sh route 0
# node1
bash smoke/start_foundir_official_vlm_router_v4_tmux.sh route 1
# node2
bash smoke/start_foundir_official_vlm_router_v4_tmux.sh route 2
# node3
bash smoke/start_foundir_official_vlm_router_v4_tmux.sh route 3
```
日志：`logs/foundir_official_vlm_router_v4_4x2_route_node0.log`至`node3.log`。
八个ROUTE_COMPLETE且四节点END status0后，CPU代理执行：
```bash
FOUNDIR_OFFICIAL_NNODES=4 bash smoke/run_foundir_official_vlm_router_v4.sh plan
```
仅推小route_plan及耗时简报，给出精确去重恢复量后等待用户明确批准恢复GPU-hours；没有官方GT选路、重训或扩大范围。
