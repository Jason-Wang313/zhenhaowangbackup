# 官方V4评分身份修复：CPU核验通过，待CUDA恢复 — 2026-11-05

本次新增执行层修复pyiqa包装器的设备相关state_dict问题：原默认评分模型先在CPU构造，核对原receipt中的完整状态哈希，再将整个Module迁移到目标设备并复核完整状态。没有过滤任何state键、跳过真实权重检查或改变评分算法、默认设置、精度。原InferenceModel初始化时`Parameter(empty).to(device)`会使dummy_param在跨设备复制时失去注册，整个Module.to可保持注册。

6项新合成CPU回归及14项原官方入口回归共20项通过；包括真实权重篡改和设备迁移时状态篡改拒绝、dummy注册、前向数值保留、修复层源码绑定、shell无时限/有锁/无重试循环。CPU/meta验证设备注册机制，不冒充CUDA验收。

现场metricfix prepare：退出码0，7.518秒，0GPU；真实默认PSNR/LPIPS/SSIM的CPU完整状态与原receipt匹配，新修复binding已落盘。原producer/contract/unlimited层、receipt、render_plan、render_budget、unlimited_execution的SHA逐一确认未变。没有重跑prepare/route/plan，没有读取其他测试集像素或启动GPU。

四节点首轮失败0.918888889 GPU-hours继续保留；加route后官方累计2.658888889 GPU-hours，历史V4另计。21,828块原恢复计划、不限时授权和逐块恢复机制保留。当前仍0恢复块，CUDA修复待用户在原四节点执行新metricfix入口实测。

各节点命令见metricfix readiness。新日志为`logs/foundir_official_vlm_router_v4_4x2_render_node<N>_metricfix.log`。生成前会先检查原plan；实际评分迁移校验通过标志为METRIC_DEVICE_VERIFIED，真实生成标志为TILE及落盘，不以tmux创建当作生成成功。
