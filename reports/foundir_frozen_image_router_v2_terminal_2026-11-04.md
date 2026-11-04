# 冻结图像路由V2修复与独立验证终态 — 2026-11-04

expanded_frozen_image_v2全部完成：2720份新缓存，原640份复用；8个worker和四节点缓存均status0。CPU训练1600步、校准选择、一次独立holdout评价及verify完成，FROZEN_IMAGE_V2_RESULT_VERIFIED、train status0。

按预定400/800/1200/1600校准CE最低选择step800，选择先不可变落盘，再读holdout。未按holdout挑模型。冻结原image_only step400，仅训练fresh VLM修正头；1601份逐步状态保留共享目录。未改变专家或配方，未读取旧40开发特征/像素。

**分类筛选未通过**：总体正确数与CE改善，但Blur和Haze正确数下降，不满足预设每类不下降条件。工程成功不等于筛选通过，不自动重训、改阈值、扩大预算或GPU恢复评价。

| holdout | 冻结image_only | 修正版 |
|---|---:|---:|
| 总体 | 350/400 (87.5%) | 374/400 (93.5%) |
| CE | .315386 | .210631 |
| Blur | 77/80 | 76/80 |
| Noise | 61/80 | 76/80 |
| Haze | 76/80 | 69/80 |
| Lowlight | 76/80 | 78/80 |
| Rain | 60/80 | 75/80 |

校准组：冻结349/400、CE .459019；修正365/400、CE .272625。修正头训练数据2560而冻结图像头原训练640，不是数据量匹配的VLM消融；holdout仅保证路由训练来源分离，不保证FLUX/Qwen未见过。不产生新恢复图或PSNR/LPIPS，不宣称官方/混合退化通过。

缓存各节点耗时：node0 608秒、node1 424秒、node2 433秒、node3 470秒；各2卡，本轮合计3870 GPU-seconds = 1.075使用GPU-hours。8worker峰值allocated均137.473185GiB，不能宣称H10080GB已通过。CPU8线程训练+验证350秒（5分50秒），0GPU-hours。五个attempt均status0，无失败成本；此前历史实验成本独立保留，不清零。独占整机额外资源成本范围未确认。

仅推逐图小JSON及本耗时/退出/显存简报；特征、checkpoint、权重和实时日志不上传。当前授权一轮工作完成，不自动启动后续。

各阶段退出账：
```json
[
  {
    "file": "cpu_train_20261104T223805_785907.ended.json",
    "status": 0,
    "seconds": 350,
    "gpus": 0,
    "used_gpu_seconds": 0
  },
  {
    "file": "features_node0_20261104T221530_2873074.ended.json",
    "status": 0,
    "seconds": 608,
    "gpus": 2,
    "used_gpu_seconds": 1216
  },
  {
    "file": "features_node1_20261104T221541_3988104.ended.json",
    "status": 0,
    "seconds": 424,
    "gpus": 2,
    "used_gpu_seconds": 848
  },
  {
    "file": "features_node2_20261104T221716_2958865.ended.json",
    "status": 0,
    "seconds": 433,
    "gpus": 2,
    "used_gpu_seconds": 866
  },
  {
    "file": "features_node3_20261104T221744_2326341.ended.json",
    "status": 0,
    "seconds": 470,
    "gpus": 2,
    "used_gpu_seconds": 940
  }
]
```

日志路径及SHA256：
logs/foundir_frozen_image_router_v2_features_node0.log b3e75e2c1151db949c319c33b2a89753c5993438579ccdaa1dd7bad582328469
logs/foundir_frozen_image_router_v2_features_node1.log 3cf74d51568d3edb56b18fefc4e794311c60a8e51c0b211752c038dce1461071
logs/foundir_frozen_image_router_v2_features_node2.log 9d7b37653cf762841a23193c627935670d76a91bf817b209820b2cb75896012b
logs/foundir_frozen_image_router_v2_features_node3.log 06ff6dc283227306a00a780713e08e6b7c6af5193c7ac05f1b0c166921ee500d
logs/foundir_frozen_image_router_v2_cpu_train.log 3df5ffe99c576e8bcb4722baea9a10e5e1c04a82e0bd206dfa6cb04f21564ce0
