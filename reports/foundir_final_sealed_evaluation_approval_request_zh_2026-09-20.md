# FoundIR 一次性 sealed evaluation 授权申请

日期：2026-09-20 UTC

## 请求 advisor 作出的决定

请求授权对已经完全冻结的 FoundIR 十一类恢复系统进行一次终局、只读的
official/sealed test 评估。

获批后的运行将严格遵守以下规则：

1. 只使用下文列出的冻结 router、五个单退化专家和六条混合退化恢复路径；
2. 在 advisor 确认这与 official SOW 规则一致后，按“每张测试图片等权”计算
   PSNR 和 LPIPS；
3. router 安全拒绝时保持输入不变，除非冻结配置明确规定使用既有的 Low-light
   保守 fallback；
4. 不根据 sealed 结果修改 checkpoint、alpha、置信度阈值、专家顺序、prompt、
   seed 规则或后处理；
5. 只运行一次，如实报告通过或失败，随后终止，不在 sealed test 上调参或重跑。

准备本申请期间没有重新访问 FoundIR official test 的图片、标签或指标。必须披露
的历史事实是：2026-08-16 曾下载并解压官方归档，对全部 1,500 对图片进行结构、
解码、尺寸和哈希验证；当时没有运行恢复模型、没有计算模型成绩，也没有据此调参
或选择模型。因此 official test 在模型性能评估意义上仍未使用，但不能称为“文件
从未被打开”。

## 五个冻结单退化专家

以下均为完整、source-disjoint native development split 上按每张图片等权计算的
均值。五个最终专家均为项目自行训练的模型，不是 released FoundIR restoration
specialist。

| 类型 | 图片数 / source groups | 冻结方案 | PSNR / LPIPS | 相对 SOW 余量 |
|---|---:|---|---:|---:|
| Blur | 2,595 / 165 | FLUX.2 step-10k，`LQ + .75*(FLUX-LQ)` | 26.559141 / .151993 | PSNR +2.5449%；LPIPS安全余量15.5594% |
| Noise | 1,859 / 34 | FLUX.2 step-400，`LQ + .30*(FLUX-LQ)` | 40.150553 / .068880 | PSNR +3.2148%；LPIPS安全余量31.1196% |
| Haze | 1,580 / 158 | FLUX.2 step-3k | 23.311888 / .101155 | PSNR +5.4837%；LPIPS安全余量46.7605% |
| Rain | 835 / 117 | 项目自行训练的 Rain Restormer step-5k | 42.284347 / .047452 | PSNR +23.6384%；LPIPS安全余量40.6848% |
| Low-light | 1,400 / 21 | FLUX.2 step-400，`LQ + .925*(FLUX-LQ)` | 19.779135 / .154058 | PSNR +4.6515%；LPIPS安全余量44.9792% |

证据 registry：
`reports/foundir_five_type_source_disjoint_development_registry_v1.json`

SHA-256：
`225004da30a29854423ef1e921787830344d3af824d1b213900c0481c5328e47`

最终十一类 router 比五分类 singleton router 更保守。在完整 2,595 张 Blur 上，
它产生 2,223 个正确路由和 372 个安全拒绝，错误路由为零。把冻结专家输出和
拒绝时的原图按真实部署规则合并后，得到：

- PSNR：26.614726，相对 SOW 25.9 为 `+.714726 dB / +2.7596%`；
- LPIPS：.156974，相对上限 .18 的安全余量为 `.023026 / 12.7920%`；
- 两项均通过。

机器可读证据：
`checkpoints/foundir_blur_record_weighted_final_eleven_router_v2/result.json`

SHA-256：
`c30b43fe85edb2a8698d2178079ffc2adcacd9932e3c6b285ee805ff4ffb0423`

## 六条冻结混合退化恢复路径

- Blur+Noise：专用 paired Blur+Noise Restormer；
- Haze+Low-light：先 Haze，后 Low-light；
- Rain+Haze：先 Rain，后 Haze；
- Blur+Low-light：先 Low-light，后 Blur；
- Noise+Low-light：先 Noise，后 Low-light；
- Blur+Noise+Low-light：先 Noise，再 Low-light，最后 Blur。

每个阶段之间都固定执行 uint8 PNG-equivalent 量化；测试时不搜索顺序或强度。

在十一类各 16 张、总计 176 张的 routed-versus-direct integration audit 中：

- 272 对中间 stage PNG 全部通过一致性检查；
- 176 对最终 metric sidecar 全部通过一致性检查；
- FLUX 路径 byte-exact；
- Rain Restormer 的重复推理仅存在最多 1/255 的 GPU 数值舍入差异；
- checkpoint、alpha、crop、文件处理、量化边界及专家顺序的真实错误为零。

证据：
`reports/foundir_final_eleven_route_integration_terminal_2026-09-20.md`

## Router 结果及其正确定位

最终 fail-closed 十一类 router 在 16,461 张 native development 图片上的结果为：

- 正确接受：13,586；
- 安全拒绝：2,858；
- 错误接受：17；
- 覆盖率：82.6377%；
- 错误路由率：.1033%。

17 个错误只出现在混合退化边界：5 个 Blur+Low-light 和 12 个
Noise+Low-light。

项目此前额外设定了“development 必须绝对零错误”的保守安全门，因此该内部
零错误门没有通过。必须同时明确：

- “绝对零错误”不是 SOW 的 PSNR/LPIPS 验收条件；
- .1033% 是已测得的低频 development routing 风险，不等于 official 指标失败；
- 不能删除或隐藏该证据，也不应把它夸大成一个新的 SOW 硬门；
- 本次申请请求 advisor 在知情的情况下，决定是否允许冻结系统进行一次终局
  sealed evaluation，而不是请求继续调 router。

冻结 decisions：
`checkpoints/foundir_final_eleven_route_cpu_gate_v1/development_decisions.csv`

SHA-256：
`0342bf00bef98eccb9c3d3ad8e3c42c2368db2761f3c2ddd38f9b1342ad31225`

## 十一类 all-in-one development rehearsal

冻结系统已经完成一轮 native all-in-one rehearsal：十一类各 128 张，共 1,408
张。实际执行包括：

- 1,247 个正确路由；
- 1 个错误路由；
- 73 个保持输入的安全拒绝；
- 87 个冻结的 Low-light partial-restoration fallback。

这轮 rehearsal 是 source-group-balanced，刻意提高稀有困难来源的权重，因此适合
暴露 routing、fallback 和尾部风险；它不是按图片自然频率计算的 SOW 替代品，也
不是一套此前完全未使用的独立 lockbox。

Blur 对聚合分布尤其敏感。在同一套历史 development 证据中：

- 每张图片等权：26.559141 / .151993，通过 SOW；
- 每个 source group 等权：23.104172 / .226661，不通过 SOW。

因此报告同时保留两项事实：record-weighted 完整 Blur development 通过；如果
sealed test 显著提高稀有困难 source 的比例，仍存在 distribution-shift 风险。该
风险不能靠访问 sealed test 后调参来消除。

all-in-one 结果：
`checkpoints/foundir_final_eleven_all_in_one_development_rehearsal_v3/terminal_result.json`

SHA-256：
`5e56a69a19ff1ed63ab78176799d93f6df62258799a46e58e1a111acec14c024`

## 为什么不声称拥有独立十一类 lockbox

provenance audit 已确认：在当前获准使用的数据中，无法诚实构造一套十一类 native
数据，使其同时满足“未参与训练、checkpoint/alpha/router 选择及历史评估”。现有
validation 已被历史组件评估使用。official test 曾被执行数据完整性验证，但从未
用于恢复推理、模型选择、参数选择或性能指标计算。

因此当前证据用于提出 sealed authorization 申请，而不被包装成 official 成绩或
通过保证。

证据：
`reports/foundir_presealed_lockbox_provenance_audit_2026-09-20.md`

## 请求的可审计回复

请 advisor 明确回复以下二者之一：

- **APPROVED：**允许冻结的十一类系统按 official 规则执行一次 sealed test；结果
  产生后不得调参或重跑；或
- **DENIED：**继续禁止访问 sealed partition，并明确仍需满足的前置条件。

在收到明确的 `APPROVED` 之前，不会运行任何 official/sealed test 命令。
