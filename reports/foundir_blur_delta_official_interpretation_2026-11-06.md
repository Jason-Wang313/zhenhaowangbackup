# Blur200官方终态与下一步

只以官方FoundIR作各类验收门槛，仍Blur优先。最终200步官方150/150、2250/2250块、两rank及launcher退出0，CPU完整性核验通过，67128e9已推送。结果旧FLUX21.546762/.202622，新21.153159/.225559：PSNR下降.393603dB、LPIPS变差11.319935%，未过25.9/.18；111/150比LQ的PSNR差，只有38/150比旧FLUX两项都好。不采用、不宣布修复、不转其它类别。
两步资格、缓存、200步训练和官方生成共1.271449232任务GPU-hours。训练1122.147524秒，官方960.149235秒，官方CPU finish35.423898秒；全部资产保留，旧开发评价未提交。四台worker空闲，无待续训练。
额外只读CPU误差恒等式/DCT诊断135.560801秒、0GPU-hours：新修正与GT−LQ方向平均cosine .019366，90/150非正相关；旧为.236089、24/150。新改动幅度更小但像素误差反而更大，不是仅“大步幅过冲”；按图平均全图常量色偏只占1.1686%误差。只做统计，无GT拟合系数、候选生成或GT回退；复算全150的PSNR匹配pyiqa。
32,440,320参数中96.4773%在BF16下仍改变；训练online loss下降，不能代替固定权重训练图评价或官方收益。新旧路径差异尚需排除，暂不盲加训练步数或把失败直接归因数据。
下一项已授权实现诊断：固定前8张原训练图，step200不更新参数，比较缓存参考/重新计算参考/官方requires_grad=false推理/训练checkpoint模式四路径，核对tensor、float/uint8和实际加载权重SHA。只用node3双卡，不另设测试集或质量放行门槛，不读其它类别。入口smoke/diagnose_foundir_blur_delta_path_v1.py；当前prepare后待提交，以共享队列实际状态为准。
报告reports/foundir_blur_delta_official_v1_result_2026-11-06.json、reports/foundir_blur_delta_official_error_identity_v1_CPU_2026-11-06.json。训练来源56/128为1920×1080，其余多种分辨率；这只是分布描述，不是已证实根因。先核验路径，再确定下一轮受控修复。


固定定性样例为官方0001/0051/0101的全图缩略和原尺寸中心512裁图，只用于描述，PNG保留在checkpoints/foundir_blur_delta_official_v1/qualitative，不上传GitHub。三例仍有模糊和结构偏差，不能替代全150结果。
