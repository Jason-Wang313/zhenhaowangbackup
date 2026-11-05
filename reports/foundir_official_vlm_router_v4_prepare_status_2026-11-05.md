# 官方V4目的地prepare状态 — 2026-11-05

已安全同步main的b6e0a2d。11份变更文件按三方合并同步，原本地文件备份于`reports/local_sync_backups/official_v4_20261105T182411Z`；未reset/clean，未触碰大资产。

Python及shell语法通过。执行指定prepare后，43秒退出码1：指定`.conda/gpu-train-stable/bin/python`环境缺少`pyiqa`，报`ModuleNotFoundError: No module named 'pyiqa'`。官方metric模型身份尚未核验，prepare未通过，未发布成功receipt。

readiness要求pyiqa==0.1.16且不自动安装替换；本次未修改环境、降级评分或绕过检查。未启动GPU、route、plan或完整恢复，未运行其他测试集。0 GPU-hours。失败attempt记录保留，后续重试计入累计CPU上限。

下一步需补齐指定环境的pyiqa 0.1.16及既定官方metric权重，再重试prepare；通过后提供两个节点route命令。四worker及两节点成功后才运行plan并发布工作量报告。完整恢复仍等待具体预算批准。
