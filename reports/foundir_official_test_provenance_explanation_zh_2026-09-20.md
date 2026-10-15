# FoundIR official test 的准确含义与历史访问说明

日期：2026-09-20 UTC

## 一句话结论

本项目所说的 `FoundIR official test` 是 `House-Leo/FoundIR` 官方发布的配对
测试集，不是项目从训练数据中自行划分的 `internal_test`。它共有 1,500 对
LQ/GT、20 个类别。官方 GT 可公开下载，因此它不是由外部服务器保密标签的
竞赛 blind test；“sealed”是本项目为了防止测试集调参而实施的内部治理规则。

## 官方身份

- 官方仓库：`House-Leo/FoundIR`；
- 冻结 revision：`6d19958499622d90ca71fe971d83b1e428c17cd0`；
- 官方配对索引：`Testset_meta_info.txt`；
- 官方指标入口：`cal_metrics.py`；
- 官方数据归档：`LQ.zip` 与 `GT.zip`；
- 总规模：1,500 对、20 个类别。

官方身份已由仓库 revision、metadata SHA、metric-script SHA、类别/ID覆盖以及归档
哈希共同固定，不是根据本项目自定义 manifest 命名推断出来的。

## 历史上已经做过什么

2026-08-16，项目已经完成一次官方数据完整性验证：

1. 下载并解压官方 LQ/GT 归档；
2. 对全部 1,500 对图片做结构、类别、ID及一一配对检查；
3. 打开并解码全部图片，核对尺寸和颜色模式；
4. 计算文件哈希，并检查部分训练来源的精确哈希重叠；
5. 为 isolated Low-light 的 50 对图片生成冻结 manifest。

当时没有运行任何恢复模型，没有计算任何模型的 PSNR/LPIPS/SSIM，没有选择
checkpoint、alpha、router threshold 或专家顺序，也没有根据官方图片调参。

所以准确表述是：

> official test 已被做过数据完整性访问，不是字节级“从未打开”；但它从未被用于
> 模型推理、模型性能评估或参数选择，因此在模型性能意义上仍是未使用的最终测试。

## 与项目自建 sealed_internal 的区别

仓库中的 `sealed_internal_test` 是项目从 FoundIR training archive 自行划出的
source-group-disjoint 内部分区：11 类、660 张。它不是官方 test，而且已在
2026-09-11 被旧 FLUX mainline 运行过一次，因此既不能称为 official，也不能再
称为当前系统的未见最终测试。

两者不得混用：

| 名称 | 来源 | 是否官方发布 | 是否已跑过模型 |
|---|---|---:|---:|
| `sealed_internal_test` | 项目从 training archive 自行划分 | 否 | 是，旧模型已运行 |
| `FoundIR official test` | 官方 LQ.zip/GT.zip及官方metadata | 是 | 否，尚未进行正式恢复推理 |

## 当前系统支持的官方范围

当前 all-in-one pipeline 支持官方20类中的11类，共1,000对：

| 类别 | 官方图片数 |
|---|---:|
| `01Blur` | 150 |
| `02Blur_Noise` | 50 |
| `05Noise` | 100 |
| `08Haze` | 200 |
| `09Lowlight_Haze` | 100 |
| `10Rain` | 100 |
| `13Rain_Haze` | 50 |
| `14Lowlight` | 50 |
| `15Lowlight_Blur` | 100 |
| `16Lowlight_Noise` | 50 |
| `18Lowlight_Blur_Noise` | 50 |

五个具有明确 Stage-2 SOW PSNR/LPIPS 阈值的 isolated 类别共600对：Blur 150、
Noise 100、Haze 200、Rain 100、Low-light 50。

当前系统不支持其余9类，包括 JPEG、Blur+JPEG、Noise+JPEG、Raindrop、
NightRain 以及若干 Low-light+JPEG 组合。因此不能声称要评估“完整20类最终
系统”；授权范围必须明确为五个SOW类别600对，或当前支持的十一类1,000对。

## 官方评分与当前 development 的区别

官方协议要求：

- 使用完整原始分辨率输出；
- 与对应 GT 做 RGB 比较；
- 指标阶段不 resize、不crop、不裁边、不转Y通道；
- 每张图片等权取算术平均；
- 使用官方发布的 `pyiqa` 调用计算 PSNR、SSIM、LPIPS；
- 当前SOW主要验收PSNR和LPIPS，SSIM作为辅助报告。

当前大部分 development 指标来自512×512对齐crop，并使用项目冻结的指标实现。
因此 development 结果是重要预测，但不能冒充官方 full-resolution 成绩。正式运行
前还必须冻结兼容的 `pyiqa` 版本、LPIPS backbone/weights、全分辨率分块与拼接
规则，并证明输出尺寸与官方输入完全一致。

## 当前运行准备状态

仓库已有：

- 官方归档获取与安全解压脚本；
- 官方revision、metadata和metric脚本身份验证；
- 1,500对完整性验证器；
- historical Low-light official manifest；
- 最终模型、router及十一条恢复路径的development证据。

仓库目前没有一条已经冻结并验证的“最终十一类 pipeline -> 官方全分辨率输出 ->
官方指标”的完整runner。现有 `run_foundir_official_validation.sh` 只负责数据获取和
完整性验证，不运行最终模型。

因此获得advisor授权后的正确流程不是立即随意运行现有development evaluator，而是：

1. 先以不读取官方图片的方式审查并冻结full-resolution official runner；
2. 明确授权范围是五类600对还是十一类1,000对；
3. 记录advisor书面授权及全部模型/config/runner hash；
4. 只运行一次；
5. 使用官方协议计算并如实报告结果；
6. 结果产生后不调参、不选择模型、不重复测试。

## 建议向advisor使用的准确说明

> 老师，我们说的“官方测试”不是自己从训练集划出的internal test，而是
> House-Leo/FoundIR官方发布的LQ/GT配对测试集，共1,500对、20类。该归档在
> 2026-08-16做过解压、解码、尺寸和哈希完整性检查，但从未运行恢复模型、从未
> 计算模型成绩，也没有用于调参。当前系统支持其中11类1,000对；具有明确SOW
> 阈值的五个单退化类别共600对。我们申请的是冻结模型后按官方原始分辨率和官方
> 指标协议运行一次，结果出来后不再调参或重跑。请确认授权范围是五类600对，还是
> 当前支持的十一类1,000对。

在advisor明确授权前，不重新检查官方运行目录、不读取图片、不运行模型或指标。
