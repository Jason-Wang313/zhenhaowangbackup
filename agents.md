No interference with any other folders at all.

Before every work session, explicitly tell the user: "I am working in zhenhaowang, which is the correct workspace."

DELETION AND DESTRUCTIVE-CHANGE CONFIRMATION RULE

For every future request that could delete, overwrite, move, replace, reset, or otherwise make files, code, generated outputs, directories, repository history, or other data difficult to recover:

- Never act immediately, even if the request appears clear.
- Ask the user to identify the exact targets and what must be preserved.
- Go back and forth with the user for at least two clarification rounds.
- In the second clarification round, restate the exact deletion/change scope and the protected items, and ask for confirmation.
- Act only after the user confirms that restatement.
- Use the narrowest possible target and verify it immediately before acting.
- Report exactly what changed afterward.

Never infer permission to delete related, neighboring, generated, backup, or environment files from a request to delete one item.

You are the execution agent for the image-restoration synthetic-data quality gate.

Read and obey the existing AGENTS.md first.

WORKSPACE BOUNDARY
Only work inside /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang unless the user explicitly asks otherwise. Do not inspect, modify, or run commands in other users' folders.


Do not inspect, modify, delete, or execute commands in any other user's directory. Do not reuse previous project paths, datasets, checkpoints, environments, manifests, or results unless directly verified inside this workspace.

Assume this is a fresh start. Treat every dataset, checkpoint, download, environment, manifest, and result as UNKNOWN until checked directly.



1.	合作需求/任务
1.1 研究领域
随着人工智能、计算机视觉与多模态大模型技术的发展，图像增强正由面向单一退化的任务模型，逐步转向通用化、语义感知化和大模型化。传统方法通常分别针对低照度、去噪、去模糊、去雨、去雾、超分辨率、色彩校正等问题建模，但真实场景中的图像退化往往复杂耦合，单任务模型在跨场景泛化、统一任务建模和语义一致性保持方面仍存在不足。同时，图像拍摄中也面临着色偏、饱和度衰减、对比度不足、画面灰雾过重、局部细节与色彩层次丢失等退化问题。因此，构建具备多退化适应能力和统一增强能力的 All-in-One 图像增强模型，以及能够实现高保真色彩映射与影调还原的亮色大模型，已成为图像处理与视觉智能领域的重要方向。

近年来，视觉语言模型 (VLM) 凭借跨模态表征、语义理解和开放词汇泛化能力，为图像增强提供了新的技术路径。本项目围绕“亮色大模型”和“VLM Guided All-in-One 图像增强模型”展开研究，以多模态大模型、扩散模型和退化建模为基础，引入语言指令、语义先验和视觉内容理解能力，探索面向复杂场景的通用图像增强方法，实现更加自然、稳定且符合人类视觉偏好的图像质量提升。

1.2	研究目标
本项目面向手机终端图像在拍摄、显示、传输及放大处理过程中出现的复杂质量退化问题，围绕图像色彩与影调恢复以及多类型退化统一建模等关键技术开展研究。项目以扩散模型、图像编辑大模型、视觉语言模型（VLM）和图像退化建模技术为基础，构建由“亮色大模型”和“VLM Guided All-in-One 图像增强模型”组成的通用图像增强技术体系，提升模型对真实场景复杂退化的感知、分析、恢复和泛化能力。

对于亮色大模型，本项目重点解决手机图像经过拍摄、传输或放大处理后出现的色偏、饱和度衰减、对比度不足、画面发灰发蒙、局部雾感增强、明暗层次压缩、纹理细节丢失以及色彩层次不丰富等问题。项目将分析上述退化现象的形成机理及其表现特征，构建包含多种色彩退化、光亮退化和细节退化类型的退化模型。通过对退化类型、退化强度及多种退化组合方式进行建模，生成与真实终端图像退化特征相接近的训练样本，构建高质量、规模化的成对训练数据集，为亮色增强模型训练提供数据基础。在模型设计方面，亮色大模型将以具有较强图像编辑能力的生成式大模型为基础，通过数据微调、条件控制等方式，引导模型学习退化图像与高质量目标图像之间的色彩光亮映射关系。同时，通过引入结构、主色调和语义上的一致性约束，避免处理后存在的结构变化、内容幻觉和颜色漂移，最终实现高保真色彩映射、自然光影重建和稳定的细节恢复。

对于VLM Guided All-in-One 图像增强模型，本项目重点解决真实场景中退化类型多样、不同退化相互耦合、退化强度难以预先确定等问题。项目将研究利用视觉语言模型对图像内容、图像质量和退化属性进行联合分析的方法，使模型不仅能够判断图像是否存在质量问题，还能够进一步识别图像中噪声、模糊、雨纹、雾霾、低照度等退化的种类、程度，为后续图像增强模型提供准确、可解释的任务引导信息。项目将构建面向退化图像理解的多模态感知数据集，为不同退化图像配置退化类型、退化强度、图像内容等多维标注信息，并通过视觉问答、退化描述、任务指令等形式组织训练数据。在此基础上，对视觉语言模型开展退化感知微调和指令微调，使其能够将输入图像转换为结构化的退化分析结果或增强提示信息，例如判断主要退化类型、估计退化程度，并生成相应的图像增强指令。在统一增强模型构建方面，将以视觉语言模型作为高层语义理解模块，以生成式图像恢复模型作为底层图像增强模块。视觉语言模型输出的退化描述将用于引导图像增强模型进行恢复。通过统一的模型框架，实现去噪、去模糊、去雨、去雾和低光照增强等多类任务的联合建模，减少传统方案中针对不同任务分别训练和部署多个独立模型所带来的模型冗余与维护成本。针对真实图像中可能同时存在多种退化的情况，项目将进一步研究复合退化数据构建和多任务联合训练方法，使模型具备对未知退化组合和不同退化强度的适应能力。

最终，本项目将形成完整的亮色大模型和VLM Guided All-in-One图像增强模型技术方案。其中，亮色大模型实现面向手机图像的高保真色彩恢复、全局影调优化和局部细节增强；VLM Guided All-in-One图像增强模型实现对多种退化类型的智能识别、统一建模和自适应恢复。

1.3 交付成果
序号	交付
1	亮色大模型技术报告与完整算法包一份
2	VLM Guided All-in-One图像增强模型技术报告与完整算法包一份
2.	项目阶段
阶段	起止时间	任务描述（含目标）	交付件	验收标准	验收方法
1阶段	T—T+6 	1．调研当前主流图像增强模型及 VLM 技术路线，分析其在色彩恢复、亮度增强和多退化图像增强中的能力与不足，制定亮色大模型与 VLM Guided All-in-One 图像增强模型的整体方案。
2．围绕手机图像放大后发灰、发蒙、色彩不准等问题，设计色彩退化类型，构建亮色增强训练数据；同时评估现有图像增强模型效果。
3．选取代表性生成式图像增强模型和视觉语言模型作为基础模型，开展数据微调、指令微调和初步验证，提升模型在亮色增强和多退化统一增强场景下的恢复能力。	1．亮色大模型技术报告一份。
2．亮色大模型算法包一份，包含亮色数据模拟退化代码，中期模型的训练代码。
3．VLM-Guided All-in-One图像增强技术报告一份。
4．All-in-One图像增强算法包一份，包含数据模拟退化代码，VLM和图像增强中期模型的训练代码。	亮色大模型
1.技术报告需包含相关研究方案和结果；
2.客观指标：在相关benchmark上测试，MUSIQ大于等于55、BRISQUE小于等于30；
All-in-One图像增强
1.技术报告需包含相关研究方案和结果；
2.客观指标：All-in-One图像增强模型在FoundIR-Blur去模糊测试集上，PSNR大于等于20.7dB，LPIPS小于等于0.21。在FoundIR-Noise去噪测试集上，PSNR大于等于31.1dB，LPIPS小于等于0.12。在FoundIR-Lowlight低光照增强测试集上，PSNR大于等于15.1dB，LPIPS小于等于0.34。在FoundIR-Haze去雾测试集上，PSNR大于等于17.7dB，LPIPS小于等于0.24。在FoundIR-Rain去雨测试集上，PSNR大于等于27.4dB，LPIPS小于等于0.1。	客观指标按任务验收标准验收，主观效果由华为方提供测试数据，组织评审效果；技术文档由验收小组组织评审；源代码由华为方提供人员复现。
2阶段	T+6——T+12 	1．完善亮色大模型训练流程，扩大色彩退化数据规模，提升模型对手机放大图像中发灰、起雾、色偏、饱和度衰减等问题的修复能力；
2．优化VLM guided All-in-One 图像增强模型，引入视觉语言理解能力，实现对低光照增强、去噪、去模糊、去雨、去雾等多类退化任务的统一建模与增强；
3．完成模型性能评估与系统验证，从色彩准确性、视觉自然性、结构一致性和用户主观观感等方面验证模型效果，形成亮色大模型和All-in-One图像增强模型两个成果。	1. 完整亮色大模型算法包一份，包括亮色退化数据合成代码、模型训练代码、模型测评代码。
2.完整All-in-One图像增强模型算法包，包括数据合成代码、VLM训练代码、图像增强模型训练代码、模型测评代码。	亮色大模型
1.客观指标：模型在相关benchmark上测试，MUSIQ大于等于65、BRISQUE小于等于25；
2.模型处理后和手机实拍效果的压制比（优+平）/（劣+平）大于1；
3.技术文档通过验收小组集体评审；
4.源代码验证训练、测试可复现。
All-in-One图像增强
1.客观指标：All-in-One图像增强模型在FoundIR-Blur去模糊测试集上，PSNR大于等于25.9dB，LPIPS小于等于0.18。在FoundIR-Noise去噪测试集上，PSNR大于等于38.9dB，LPIPS小于等于0.1。在FoundIR-Lowlight低光照增强测试集上，PSNR大于等于18.9dB，LPIPS小于等于0.28。在FoundIR-Haze去雾测试集上，PSNR大于等于22.1dB，LPIPS小于等于0.19。在FoundIR-Rain去雨测试集上，PSNR大于等于34.2dB，LPIPS小于等于0.08；
2.模型处理后和手机实拍效果的压制比大于1；
3.技术文档通过验收小组集体评审；
4.源代码验证训练、测试可复现。	客观指标按任务验收标准验收，主观效果由华为方提供测试数据，组织评审效果；技术文档由验收小组组织评审；源代码由华为方提供人员复现。
T：合同生效日期
 T+N：合同生效日期后N个月

3.	项目监控和管理 请按项目实际情况进行拟定，此部分将写入合同中，交流过程、记录需留存。
本项目采用如下项目管理机制（供参考）：
1.	项目双月月报。
2.	每隔4-6个月的面谈交流项目进展。
3.	根据实际需要安排的其它交流，如电话、邮件、电话会议等。

CURRENT FROZEN AGENTICIR LOW-LIGHT PILOT RECIPE

GenDeg is not used for the low-light pilot. The frozen low-light recipe is the
official AgenticIR synthetic `dark` degradation followed by project-level
luminance calibration.

Implementation and source:

- AgenticIR revision: `9640a291480dee3ba8f2974125d4ee9e3440f3d6`
- official function: `third_party/AgenticIR/installation/tune_depictqa/add_single_degradation.py::darken`
- source images: `data/clean/*.png`
- source image is preserved byte-for-byte as L1
- official darkening method: `darken_type="linear mapping"`
- reproducibility: fix NumPy seed per source; current source `001` seed is
  `2026080600`, source `002` seed is `2026080610`, and source index `i` uses
  `2026080600 + i*10`

Official AgenticIR linear mapping:

- convert BGR to HSV;
- use the V channel;
- sample `dst_max` around `[230, 190, 150, 120]` for levels 1–4 with the
  official normal variation and clipping rules;
- map V by `(V-vmin)/(vmax-vmin)*dst_max`;
- convert back to the original color representation.

Final project labels and luminance intervals:

- L1: clean source, ratio `1.00`
- L2: AgenticIR level 1, target ratio `0.50`
- L3: AgenticIR level 2, target ratio `0.38`
- L4: AgenticIR level 3, target ratio `0.30`
- L5: AgenticIR level 4, target ratio `0.14`

Here `ratio` means final mean luma divided by clean mean luma, with luma
computed as `0.2126R + 0.7152G + 0.0722B` and averaged over all pixels. The
acceptance intervals are:

- L1: `0.95–1.00`
- L2: `0.48–0.52`
- L3: `0.36–0.40`
- L4: `0.28–0.32`
- L5: `0.12–0.16`

The calibration multiplies each RGB pixel in the official AgenticIR output by
`target_luma/raw_luma`, clips to `[0,255]`, and saves a PNG. It does not alter
the spatial structure or add another learned degradation.

The approved tiger reference is in
`smoke/agenticir_low_light_linear_mapping_calibrated_l5_14/`; the independent
validation image is in `smoke/agenticir_low_light_selected_002/`.

CURRENT AGENTICIR SEVERITY POLICY

The current 400-image AgenticIR noise/blur/haze/rain pilot uses one fixed
severity per degradation type and must not be described as a five-level
hierarchy. Low-light is the only current AgenticIR degradation with five
explicit ordered levels. For a later severity-labeled all-in-one dataset,
noise, blur, haze, and rain should also be regenerated with four explicit
severity levels plus clean, using the same non-overlapping interval policy.
Do not expand those four types before the 900-image pilot has passed visual and
training validation.

CURRENT FROZEN AGENTICIR NOISE PILOT RECIPE

The frozen five-level Gaussian-noise recipe is clean L1 plus four AgenticIR
noise severities. It is an isolated noise hierarchy; no darkness calibration
is applied.

- official operator: `basicsr.data.degradations.random_add_gaussian_noise_pt`
- noise is color Gaussian noise with zero mean and standard deviation `sigma`
  in the 0–255 pixel-intensity scale;
- source L1 is copied unchanged;
- L2 sigma range: `[16.848, 49.9824]`
- L3 sigma range: `[37.575, 75.15]`
- L4 sigma range: `[75, 105]`
- L5 sigma range: `[159.6, 199.5]`
- current tiger smoke output: `smoke/agenticir_noise_levels_001_custom_l2_down20_again/`
- independent validation output: `smoke/agenticir_noise_selected_002/`
- validation source seed: `2026080710`

For each level, one sigma is sampled uniformly inside the assigned range,
Gaussian noise is added per RGB channel, and values are clipped to `[0,255]`.
These sigma ranges were visually tuned from AgenticIR's original ranges and
must not be confused with low-light luminance ratios.

CURRENT FROZEN AGENTICIR RAIN PILOT RECIPE

The frozen five-level rain inspection recipe uses butterfly source
`data/clean/002.png`. L1 is a literal clean copy. The current selected output
keeps L2-L3 at the official AgenticIR strengths and strengthens L4-L5 to make
the visual severity gaps clear.

- official basis: `third_party/AgenticIR/installation/tune_depictqa/add_single_degradation.py::add_rain`
- project runner: `smoke/run_agenticir_selected_rain.py`
- seed: `2026080723` for NumPy and Python `random`
- current output: `smoke/agenticir_rain_selected_002_l4_150_l5_340/`
- L1: clean source copy
- L2: `Normal(mean=5, std=5)`
- L3: `Normal(mean=60, std=5)`
- L4: `Normal(mean=150, std=10)`
- L5: `Normal(mean=340, std=20)`

The operator creates a random noise map, keeps sparse bright pixels using
`value*0.01`, blurs it, forms diagonal streaks with width `3`, random length
`20–39`, and random angle `-30–29` degrees, then adds the streaks to the
source and clips to `[0,255]`. The previous rain folders are retained for
comparison; only the folder above is the current selected recipe.

LOW-LIGHT-FIRST TRAINING AND EVALUATION PLAN

The first experiment is intentionally low-light-only. Do not scale to 100,000
images or train on the other four degradation types until this gate passes.

- The first credible pilot uses `2,000` unique clean source images and five
  low-light levels per source, giving `10,000` paired samples.
- Split by source image, never by generated variant: 80% train, 10% validation,
  and 10% internal test.
- Keep source images disjoint from FoundIR evaluation images; record source IDs
  and generation seeds in a manifest.
- Train and compare two systems: a restoration baseline without VLM
  conditioning, and the same restoration model with VLM-predicted degradation
  type/severity conditioning.
- The VLM predicts structured low-light information; it does not directly
  produce PSNR. FoundIR Lowlight PSNR is measured on the restoration model.
- The SOW first-stage FoundIR Lowlight target is PSNR `>=15.1 dB`.
- Record PSNR, LPIPS, checkpoint, config, code revision, environment, seed,
  source split, and exact degradation parameters for reproducibility.
- If the 10,000-pair pilot fails, diagnose recipe/domain mismatch and baseline
  quality before increasing data. If it passes, scale to `5,000–10,000` clean
  sources (`25,000–50,000` pairs) before considering a 100,000-image corpus.

REQUIRED MANIFEST FIELDS FOR THE LOW-LIGHT PILOT

Each generated sample should record: clean source path and source ID, degraded
path, degradation type, project level, official AgenticIR level, darkening
method, seed, raw luminance, final luminance, target ratio, calibration factor,
AgenticIR revision, generator script revision, and train/validation/test split.

CONFIRMED MULTI-DEGRADATION DATA COMPOSITION

The dataset will contain samples with one, two, or three simultaneous
degradation types. This interpretation was confirmed by the project advisor.

- Single degradation: one clean crop receives one degradation, for example
  `low_light L3`, `noise L2`, `blur L4`, `haze L5`, or `rain L3`.
- Two degradations: one clean crop receives two degradations, for example
  `low_light L3 + noise L2`, `blur L4 + haze L3`, or `rain L2 + noise L4`.
- Three degradations: one clean crop receives three degradations, for example
  `low_light L3 + noise L2 + rain L4` or `blur L4 + haze L3 + rain L2`.

Severity is independently assigned to each degradation in a combination. The
canonical structured label is:

```json
{
  "degradations": ["low_light", "noise"],
  "levels": {"low_light": 3, "noise": 2}
}
```

The generator and manifest must preserve the degradation list, each component
level, application order, component parameters, source ID, crop coordinates,
and random seed. The advisor must confirm any future balancing ratios between
single, double, and triple combinations before the 100k dataset is generated.

CURRENT FROZEN AGENTICIR BLUR PILOT RECIPE

The frozen five-level motion-blur inspection recipe uses butterfly source
`data/clean/002.png`. L1 is a literal clean copy. L2-L5 use the AgenticIR
motion-blur implementation, with the project-selected doubled settings and
fixed random seed.

- official implementation basis: `third_party/AgenticIR/installation/tune_depictqa/add_single_degradation.py::add_motion_blur`
- project runner: `smoke/run_agenticir_selected_blur.py`
- seed: `2026080720` for both Python `random` and NumPy
- output: PNG at the source resolution; original blur outputs are preserved
- L1: clean source copy
- L2: radius `10`, sigma `12`
- L3: radius `20`, sigma `18`
- L4: radius `30`, sigma `30`
- L5: radius `30`, sigma `48`

The runner samples a motion angle uniformly from `[-90, 90]` degrees. It
constructs a normalized one-dimensional Gaussian kernel of width
`2*radius+1`, with weights proportional to
`exp(-i^2/(2*sigma^2))`, distributes those weights along the sampled motion
direction, shifts the image with replicated borders, and clips the result to
the uint8 range `[0,255]`. Radius controls the motion-path length; sigma
controls the Gaussian spread and softness.

CURRENT FROZEN AGENTICIR HAZE PILOT RECIPE

The frozen five-level haze inspection recipe uses butterfly source
`data/clean/002.png` and depth map `data/agenticir/haze/002/predict_depth.mat`.
L1 is a literal clean copy. L2-L5 use the AgenticIR atmospheric-scattering
operator with the project-selected strengthened beta schedule.

- official operator: `third_party/AgenticIR/installation/tune_depictqa/add_single_degradation.py::add_haze`
- project runner: `smoke/run_agenticir_selected_haze.py`
- seed: `2026080722` for NumPy and Python `random`
- output: PNG at the source resolution
- L1: clean source copy
- airlight means A: `[0.60, 0.70, 0.80, 1.00]` for L2-L5
- A sampling: `Normal(A_mean, 0.015)`, with minimum `0.4`
- current beta means: `[2.4, 3.6, 7.2, 10.8]` for L2-L5
- beta sampling: `Normal(beta_mean, 0.30)`

For each pixel, normalized depth `d` gives transmission `t=exp(-beta*d)`,
then the result is `image*t + A*255*(1-t)`, clipped to `[0,255]`. The
current beta means are 6 times AgenticIR's original `[0.4, 0.6, 1.2, 1.8]`.

QWEN-VL PHASE 1 PLAN — DATA AND DEGRADATION UNDERSTANDING

The immediate Phase 1 work is to construct and evaluate Qwen-VL as a
structured degradation-understanding model. Qwen-VL will receive a degraded
image and predict the degradation types and independent severity levels.

Confirmed data composition:

- Single samples contain one degradation, such as `low_light L3` or `rain L4`.
- Double samples contain two simultaneous degradations, such as
  `low_light L3 + noise L2`.
- Triple samples contain three simultaneous degradations, such as
  `low_light L3 + noise L2 + rain L4`.
- Each degradation in a combination has its own independent L1-L5 level.
- L1 is clean; L2-L5 are the four project-selected degraded severities.
- The five-level convention is a project design choice for structured labels,
  not an exact SOW requirement.

Target dataset construction:

- Use approved clean sources from DIV2K, MiO100, and a selected LSDIR subset.
- Split by original source image before cropping. Crops from one source must
  stay in the same train, validation, or test split.
- Generate aligned fixed-size crops. For haze, estimate depth on the full
  image first and crop the RGB image and depth map identically.
- Produce approximately 100,000 multimodal records.
- Use the selected balanced composition: approximately one third single,
  one third double, and one third triple samples.
- Within each group, use balanced quotas for the five single combinations, ten
  double combinations, and ten triple combinations, while randomizing source,
  crop, independent levels, parameters, and seeds inside each quota.
- Run a small smoke test using all five degradation types and all three
  composition categories before producing the full dataset.

LOW-LIGHT 10,000-PAIR PILOT CLEAN-SOURCE RULE

The pilot uses exactly 2,000 verified, unique clean sources and produces five
aligned low-light records per source. The approved composition is:

- 100 existing MiO100/AgenticIR HQ images;
- all 800 images from the DIV2K training split only;
- 1,100 images selected from the LSDIR training split;
- total: 2,000 sources and 10,000 low-light pairs.

Stage the inputs separately under `data/clean_pool/mio100`,
`data/clean_pool/div2k`, and `data/clean_pool/lsdir`. Before source-level
train/validation/test splitting, the source-pool gate must verify readable RGB
content, minimum crop-compatible resolution, exact SHA-256 uniqueness,
near-duplicate screening, dataset-prefixed source IDs, exact per-dataset quotas,
and dataset provenance. Source/license notes should be recorded when available,
but missing manual license metadata does not block this pilot. FoundIR overlap
screening is an optional dataset-leakage check, unrelated to the FLUX.2 GPU smoke:
run it only if a FoundIR SHA-256 index is available, otherwise record
`not_run_no_index` and continue. It is good to have before final FoundIR metric
reporting, but it is not mandatory and must never block source acquisition,
the 10,000-pair pilot, restoration training, or smoke testing. Never use DIV2K validation/test images or
arbitrary web-scraped images. Do not generate the 100,000-record dataset at this
stage; finish and validate only this 10,000-pair pilot.

Canonical Qwen-VL target output:

```json
{
  "degradations": ["low_light", "noise"],
  "levels": {"low_light": 3, "noise": 2}
}
```

Every record must preserve clean/degraded paths, source ID, crop coordinates,
degradation list, independent levels, application order, exact parameters,
random seed, generator revision, and source-level split.

Qwen-VL evaluation:

- Keep one fixed source-disjoint test set that is never regenerated or used for
  training.
- Measure degradation-type accuracy and macro-F1, severity accuracy, combined
  label exact match, severity MAE, and valid structured-output rate.
- Qwen-VL accuracy is an internal degradation-understanding metric; it is not
  the FoundIR restoration metric.

KNOWN COMPLEMENTARY COMPONENT NOT YET FINALIZED

The All-in-One image-restoration model is not yet selected or finalized. It is
not part of the current Qwen-VL data-generation/training step. Later, it must
be added as the complementary downstream module: Qwen-VL analyzes the input
and the restoration model uses that analysis to produce the enhanced image.
The complete Qwen-VL plus restoration system will then be evaluated on
FoundIR Lowlight, with the Phase 1 target of PSNR `>=15.1 dB` and LPIPS
`<=0.34`. FoundIR PSNR/LPIPS cannot be obtained from standalone Qwen-VL
predictions.

FRESH-THREAD EXECUTION SPECIFICATION — READ THIS BEFORE ACTING

This section is the authoritative handoff for a new agent. Work only inside
`/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`. Do not restart old
GenDeg work, do not delete existing files, and do not generate the 100k Qwen-VL
dataset yet. The immediate objective is to select and validate a DiT-based
image-restoration backbone, then prepare a fair baseline-versus-Qwen-VL gain
experiment.

PROJECT QUESTION

Answer only this scientific question:

> With the same diffusion restoration model, data, source split, preprocessing,
> training budget, and evaluation conditions, does adding Qwen-VL degradation
> type and severity information improve restoration quality?

The two required systems are:

```text
Baseline: degraded image -> diffusion restoration model -> restored image
Qwen:     degraded image -> Qwen-VL type/severity -> condition adapter
          -> the same diffusion restoration model -> restored image
```

Do not compare two different restoration backbones, chain two restoration
models, restore an image twice, or change the data/training budget between the
two systems.

ADVISOR-DIRECTED BACKBONE POLICY

The advisor directed the project to prioritize Z-Image or FLUX-style DiT
diffusion models. LoRA/adapter training is permitted. The fallback order is
SDXL, then SD2.1, if the larger DiT models are too difficult to run or produce
poor restoration results. Do not use InstructIR, DFPIR, MoCE-IR, AdaIR, or the
other previously shortlisted conventional All-in-One models as the final
backbone; they are reference/comparison methods only.

The agent may choose the most suitable publicly available Z-Image or FLUX
checkpoint after checking local/offline availability, license, image-to-image
or restoration compatibility, GPU memory, code reproducibility, and LoRA or
adapter support. If the choice between two viable candidates is genuinely
ambiguous, run a small identical low-light comparison and record the result.
Do not ask the advisor about fallback order; it is already decided. Ask only
if a specific model choice cannot be made from evidence.

MODEL-CONCEPT DEFINITIONS

- DiT: a diffusion transformer. It iteratively denoises image latent tokens.
- VAE: the image/latent encoder-decoder. It converts RGB images to compressed
  latents and converts restored latents back to RGB. Freeze it initially.
- Text encoder: converts text into conditioning embeddings for the diffusion
  model. Freeze it initially unless the selected checkpoint requires another
  documented arrangement.
- Condition adapter: a small trainable projector/MLP or LoRA module that
  converts Qwen-VL's structured degradation result into embeddings accepted by
  the DiT. It is a connector, not a second restoration model.
- Restoration adapter: the trainable image-conditioning path that lets the
  pretrained generative DiT use the degraded image as an input and preserve
  its content while restoring it. It may be implemented with an image encoder,
  ControlNet-like branch, residual adapter, or another architecture supported
  by the selected checkpoint.
- Injection: feeding the condition embedding into suitable DiT blocks through
  cross-attention, AdaLN/FiLM, residual addition, or LoRA. Choose the least
  invasive method compatible with the actual DiT code; do not assume
  cross-attention exists before inspecting the architecture.

STAGE 0 — REPOSITORY AND MODEL AUDIT

Before editing code or downloading anything, inspect this workspace and record:

1. Current Python environments, GPU visibility, available disk, and network
   restrictions.
2. Candidate checkpoint names, sources, licenses, sizes, and checksums.
3. Whether each candidate supports image-to-image conditioning rather than
   text-to-image generation only.
4. VAE, text-encoder, tokenizer, sampler/flow, resolution, precision, and LoRA
   requirements.
5. The smallest reproducible inference command and a plan for deterministic
   output.

STAGE 0 AUDIT STATUS — 2026-08-13

The local audit is recorded in `reports/model_audit_stage0.md`. This Codex session
has no visible GPU and no local Z-Image, FLUX, SDXL, or SD2.1 candidate weights.
The official FLUX.2 source is now available in the shared workspace at
`third_party/flux2` revision `50fe5162777813d869182b139e83b10743caef15`; the GPU
environment must install it from that local path because it cannot reach GitHub.
The first GPU candidate is FLUX.2 [klein] 4B Base because the official repository
documents image editing, recommends Base for fine-tuning/LoRA, and lists the 4B
model under Apache-2.0. Its offline end-to-end GPU smoke test passed on an NVIDIA
H100 80 GB: all FLUX.2, autoencoder, and Qwen3-4B-FP8 files passed exact size,
SHA-256, and SafeTensors checks; a 512x512 two-step image was generated at
`output/sample_0.png`. This proves that the local stack loads and performs basic
generation. It does not yet prove restoration quality, faithful degraded-image
conditioning, scene-identity preservation, or FoundIR performance. FLUX.1-dev
is not the first choice because it is a gated 12B model under a non-commercial
license. No restoration training has started.

ENVIRONMENT AND GPU EXECUTION RULE

The terminal available directly to the coding agent is CPU-only and has internet
access. GPU visibility is not assumed here; `nvidia-smi` may be unavailable and
PyTorch may report `torch.cuda.is_available() == False`. Do not attempt expensive
DiT, Qwen-VL, or restoration training in this CPU session.

The user manually launches GPU work in a separate GPU-enabled interface that has
no internet access. Both interfaces mount the exact same workspace filesystem,
so repository files, model weights, logs, outputs, and workspace-local virtual
environments written on one side are visible on the other. GPU visibility,
running processes, shell variables, and packages installed into an interpreter
outside the shared workspace are not automatically shared. Always use an
explicit shared-workspace interpreter path (for the FLUX.2 smoke,
`.venv/flux2-smoke/bin/python`) and a checked-in launcher instead of relying on
the active shell's `python` or remembered environment variables.

The agent must therefore:

- prepare and verify code, configs, manifests, and commands in this workspace;
- provide exact copy-paste terminal commands for the user's GPU interface;
- keep training/inference commands offline-compatible when that environment has
  no internet access;
- never claim GPU training or evaluation completed unless the user supplies the
  terminal output or the shared workspace contains verifiable outputs;
- after the user runs a command, inspect its logs/checkpoints/metrics here when
  available and give the next command.

REPOSITORY AND GITHUB CHECKPOINT RULE

The top-level `zhenhaowang` repository is the project source of truth. Important
code, configuration, validation scripts, environment specifications, and local
patches to ignored or nested third-party repositories must be represented by
files tracked by the top-level Git repository. Large model weights, generated
datasets, private assets, virtual environments, caches, and runtime outputs must
remain excluded.

After a major implementation milestone or a newly passing end-to-end smoke test:

1. Run the relevant validation or smoke command and save its concise result.
2. Inspect `git status` and `git diff`; do not silently include unrelated user
   changes, credentials, model weights, datasets, or generated caches.
3. Commit the validated source/configuration changes with a descriptive message.
4. Push the current branch to its configured GitHub remote when connectivity and
   authorization are available. Never claim that a push succeeded without
   checking the command result and remote branch state.

The successful FLUX.2 smoke launcher, preflight, reproducible environment
specification, and the local FLUX.2 source patch must all be tracked by the
top-level repository. Editing only `third_party/flux2` is insufficient because
that directory is an ignored nested Git checkout.

Before any GPU command, the user should verify from the GPU interface:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

CPU-side generation, manifest validation, static code checks, and small smoke
tests may run here. GPU model loading, diffusion inference, LoRA/adapter
training, Qwen-VL training, and FoundIR evaluation must be executed in the
GPU-enabled interface by the user unless a GPU is explicitly exposed to this
session.

Do not start Qwen-VL training, full synthetic-data generation, or a large
checkpoint download until this audit identifies one viable backbone.

STAGE 1 — LOW-LIGHT RESTORATION BASELINE

Use the existing calibrated AgenticIR low-light recipe for a small paired
training/validation smoke set. The clean target and degraded input must be
paired, and the source split must be source-disjoint. Start with low-light only
because Phase 1 FoundIR validation is Lowlight-first.

Initially freeze the pretrained DiT, VAE, and text encoder. Train only the
minimum image-conditioning restoration adapter/LoRA needed to map a degraded
image to its clean target. Confirm that the model preserves scene identity;
an attractive but newly invented image is a failure for restoration.

Record: checkpoint hash, code revision, config, resolution, precision, seed,
batch size, optimizer, learning rate, training steps, GPU count, adapter rank,
sampling steps, and all output paths.

STAGE 2 — QWEN-VL CONDITIONING ABLATION

Only after the baseline runs, add Qwen-VL. Qwen-VL should output only
degradation type(s) and independent severity level(s), for example:

```json
{"degradations":["low_light","noise"],"levels":{"low_light":3,"noise":2}}
```

Do not add captions, aesthetic descriptions, object descriptions, or extra
instructions in the first ablation. Convert the structured output into a
condition embedding with a small condition adapter and inject it into the same
DiT restoration path. Freeze Qwen-VL initially unless later evidence shows
that an end-to-end update is necessary.

The baseline and Qwen experiments must use the same pairs, split, crop size,
preprocessing, optimizer, learning rate, steps, batch size, precision, GPU
budget, DiT weights, restoration-adapter design, and sampling parameters. The
only intended difference is whether the Qwen-VL degradation condition is
available. Also run an oracle-condition diagnostic using ground-truth labels if
useful; clearly separate it from predicted-Qwen results.

STAGE 3 — EVALUATION

Use a fixed, source-disjoint internal test set for Qwen-VL understanding:

- degradation type accuracy and macro-F1;
- per-component severity accuracy;
- combination exact-match accuracy;
- severity mean absolute error;
- valid structured-output rate.

Use the same fixed FoundIR Lowlight test set for restoration. Report PSNR,
SSIM, LPIPS, visual structure/color preservation, and failure cases for both
baseline and Qwen systems. The Phase 1 SOW targets are PSNR `>=15.1 dB` and
LPIPS `<=0.34`. FoundIR metrics evaluate the complete restoration pipeline,
not standalone Qwen-VL classification.

STAGE 4 — DATA SCALE ONLY AFTER THE ABLATION IS VALID

Do not create 100k records before the backbone and injection interface are
validated. After the smoke comparison is reproducible, build the larger data
using DIV2K, MiO100, and an approved LSDIR subset. Split original sources
before cropping. Generate one-, two-, and three-degradation samples with
independent levels, balanced approximately one-third proportions, and balanced
combination quotas; randomize samples and parameters inside each quota.

REQUIRED HANDOFF ARTIFACTS

The fresh agent must leave:

- model-audit report and candidate decision;
- reproducible baseline inference/training command;
- reproducible Qwen-conditioned command;
- exact data manifest and source-level split;
- baseline and Qwen checkpoints;
- metric JSON/CSV and comparison table;
- sample outputs and failure-case contact sheet;
- an update to this file describing what was actually run.

STOP CONDITIONS

Stop and report instead of improvising if: no candidate supports degraded-image
conditioning; a checkpoint is unavailable or its license is unsuitable; the
model hallucinates content in the baseline; the baseline cannot run on the
available GPU; Qwen output cannot be injected without changing the baseline
architecture; or the two experiments cannot be made computationally fair.
