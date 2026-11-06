# Rain续训观察：first5

当前最后完整step：805；新增5/200步。完整节点状态、日志SHA、loss与训练探针见JSON。

固定step800→1000、global32、三节点六卡；非bitwise八卡恢复。新训练排除已知官方重叠来源，但旧权重的3来源/30张官方Rain历史GT来源曝光仍保留，不能称干净独立泛化。当前探针均training_fit_only，不表示官方Rain已修复。

step800训练32图：PSNR 34.718863 / LPIPS 0.059369。
