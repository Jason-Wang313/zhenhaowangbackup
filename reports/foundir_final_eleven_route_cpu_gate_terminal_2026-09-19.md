# Final eleven-route CPU gate：GPU integration 前终结结果

日期：2026-09-19

## 结论

最终 truth-free router candidate 未满足用户要求的零错误路由条件，因此在 GPU integration 前 fail fast。没有启动 restoration 推理，也没有访问 sealed/internal/official test。

## 统一 development 结果

- Records：16,461
- Correct accepted：13,586
- Wrong accepted：17
- Safe abstention：2,858
- Coverage：82.6377%
- Wrong rate：0.1033%
- 最低类别 coverage：68.2731%

覆盖率条件全部通过；唯一失败条件是 `wrong routes == 0`。

错误只来自：

- Blur+Low-light：5/2,131，被所有现有 heads 一致判断成 Blur+Noise+Low-light；
- Noise+Low-light：12/1,743，其中 11 被判断成 Low-light，1 被判断成 Noise。

其余九类错误路由均为 0。所有类别的 wrong rate 均低于 1%，但这不能替代本轮明确要求的绝对零错误。

## 为什么没有启动 GPU

完整 restoration integration 预计涉及约两万次冻结专家调用。路由在 CPU 阶段已经不满足验收条件，继续 GPU 只会验证错误路线能够被忠实执行，不能修复错误路由。

以下做法均被明确拒绝：

- 根据 development truth 选择不同 router；
- 按 record ID 屏蔽已知错误；
- 查看错误后事后选择 confidence threshold 并把同一 development 称为独立验证；
- 访问 sealed/internal/official test 寻找更好结果。

## 第一性原理限制

对 8 个 `Noise+Low-light → Low-light` 边界样本进一步计算了部署可用的 exposure 和 dark-region noise。它们处于纯 Low-light 分布内部；极暗量化会同时压低真实噪声与测量残差。因此简单物理阈值不能可靠分开，不能诚实地产生零错误。

## 状态

- 五个 singleton experts：保持冻结；
- 六条 mixed restoration paths：保持冻结；
- final mixed router：未通过零错误 gate，不得冻结为最终部署版本；
- GPU integration：未启动；
- sealed-test advisor approval：目前尚不具备申请条件。

机器可读结果：`checkpoints/foundir_final_eleven_route_cpu_gate_v1/result.json`
