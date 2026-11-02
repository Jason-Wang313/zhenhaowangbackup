# existing_flux_experts：新共享目录CPU审计 — 2026-11-02

已安全同步a9208f3，目的地只读审计通过：`asset_and_structure_audit_pass_only`，
`destination_audit=true`。实际资产根为
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`。
保留本地PPT handoff/agents.md改动，26份energy绑定源码哈希未变；全部大资产保留。

- 五份LoRA、共享FLUX底座/VAE、Rain prompt缓存SHA与原审计一致。
- 每份LoRA为50个安装位置、100个有限张量、32,440,320参数；合计500个张量。
- Blur/Noise/Haze/Low-light保持BF16，Rain保持FP32；Rain endpoint仍禁止进入flow bank。
- 13份审计源码/配置绑定逐一匹配固定提交a9208f3。
- Rain训练38,334条/5,780来源，开发100条/100来源，跨任务非训练来源/GT哈希交叠0。
- 历史声明训练池：Blur105057/7339、Noise16/16、Haze1580/1580、Low-light16/16；交叠均0。
  此为本地声明微调清单审计，不涉及FLUX底座预训练数据。
- 12项CPU算子测试通过，Python/JSON语法通过；GPU使用0、图片像素读取0。

原样审计输出见同名JSON；目的地执行及日志SHA补充证据见
`foundir_existing_flux_experts_destination_execution_2026-11-02.json`。
本报告不上传权重、数据或实时日志。

## retention历史日志SHA差异已核对

四份共享盘原始日志均精确匹配旧报告记录的SHA。旧JSON内嵌字符串与原日志按
Python通用换行读取后的文本相同；原文件分别含4/4/5/4个CR，读取时CRLF/CR被归一为LF，
因此内嵌文本编码回UTF8后的字节SHA不能与原始文件SHA直接比较。
补充JSON分别记录原始/内嵌/归一化SHA及字节数；旧报告、旧日志均未改写。
这解释了内嵌文本和原始SHA不一致，不用它替代逐项资产审计。

## 边界与下一步

CPU资产和小算子通过不等于完整CUDA恢复等价、原协议单专家50步重放通过，
也不等于训练就绪或新PSNR/LPIPS成功。四类共同prompt、输出alpha协议和Rain多步适配仍待交付。
按新交接等待原协议GPU重放及Rain资格入口；本轮未启动GPU、retention或任何旧任务。
未来资源按当前交接的单节点同型号8卡、保留4卡兼容，不复用旧四节点IP。
