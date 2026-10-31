# Legacy instructions — current authority moved to AGENTS.md

最新实施状态：FLUX-only 25步代码已实现、9项CPU测试通过，尚未GPU启动/验收。
读 `reports/foundir_flux_native_energy_v1_readiness_2026-10-31.md`，不得运行旧CNN/V3。
先同步，再在GPU节点运行新native-energy入口；不自动200步。

## 2026-10-31 最新覆盖：FLUX-only，禁止外挂 CNN / Restormer

只按 AGENTS.md 顶部和 `reports/foundir_flux_only_energy_redesign_2026-10-31.md`
执行。旧CNN的200步已完成；不得续训/重启SpatialConditionalEnergy或TailQualityEnergy，
不得使用Restormer/refiner/其蒸馏。当前为FLUX自身Transformer表示上的energy重做，
训练器已实现、CPU通过、未启动GPU。下方旧“下一步200步/继续V3”全部失效。
原生冻结FLUX VAE及固定LPIPS保留；它们不是新增可训练CNN。历史代码/结果不删除。

## READ FIRST: workspace handoff, 2026-10-30

The sole active root is
`/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`.
Fetch `origin/main` and read its `AGENTS.md` and
`reports/foundir_workspace_migration_handoff_2026-10-30.md` before acting.
The old copied local checkout may be stale; do not blindly pull/reset over it.
All operational priorities, old workspace paths and mandatory old-handoff
reading orders BELOW are historical wherever superseded by those documents.
Do not restart an old Rain/MoE/Restormer/EBT/25-step task from this legacy file.
Next: read the latest AGENTS.md completed-qualification/200-step section and
reports/foundir_flux_tail_v3_direct_trainfit_readiness_2026-10-30.md.
The eight-step 3x2 qualification completed; the user approved an independent
fresh 200-step trainfit, not a repeat qualification or restored qualification weights.
Historical bitwise parity is recorded, not a launch gate. Do not repeat diagnostics.
The source-session agent leads decisions/code; the destination agent only
synchronizes the specified files to the shared root and verifies hashes.
The destination CPU observer does not launch GPU jobs; the user launches the
approved trainfit on the GPU nodes. No source/config/recipe changes are needed.
No new-root GPU verification has been done by the source-workspace agent.
Preserve safety/protected-data restrictions and unrelated files; speak Mandarin.

---

No interference with any other folders at all.

Before every work session, explicitly tell the user: "I am working in zhenhaowang, which is the correct workspace."

USER COMMUNICATION LANGUAGE

Communicate with the user in Mandarin Chinese in commentary and final messages.
Keep necessary technical terminology in English when that is clearer or more
precise. Repository code and existing technical documents may continue using
their established language and style.

FRESH-SESSION CONTINUATION — 2026-09-12 (LATEST OPERATIONAL AUTHORITY)

- Read this entire `agents.md` first, then read
  `SESSION_HANDOFF_2026-09-12.md` completely before inspecting, editing,
  launching, stopping, or restarting project work.
- `SESSION_HANDOFF_2026-09-12.md` is the latest authority for live runtime
  state and the exact next task. It supersedes the stale operational status
  and next-task statements in the 2026-09-11 sections below and all older
  handoffs. It does not override the mainline goal lock, advisor governance,
  workspace boundary, destructive-action rule, H100/H200 portability rule, or
  protected-test restrictions in this file.
- The current authorized work is the already-running FLUX.2 isolated-Blur
  full development evaluation requested by the advisor. Monitor that same job
  to completion. Do not restart it merely because its remote PID is invisible
  from the Codex shell; use its advancing telemetry, log, and immutable
  per-record sidecars as authoritative liveness evidence.
- At each completed checkpoint, and at any terminal success or failure, update
  `SESSION_HANDOFF_2026-09-12.md` with the verified timestamp, counts, metrics,
  and exact next action before ending the work session.

GOAL LOCK AND MANDATORY ADVISOR REPORT GATES — 2026-09-11

- The active main goal is the advisor-approved VLM-guided FLUX.2 All-in-One
  restoration system for Blur, Noise, Haze, Rain, and Low-light. Released
  FoundIR generalists, specialists, Restormer, routing experiments, and other
  reference systems are diagnostic/comparison evidence only. They must never
  replace the FLUX.2 mainline or be reported as its result.
- Before changing the model family, restoration backbone, architecture,
  evaluation target, data split, checkpoint-selection rule, or acceptance
  protocol, stop and obtain an explicit advisor/user decision. A promising
  diagnostic result is not authorization to change the goal.
- Generate an advisor-ready report artifact immediately after each of these
  gates: Stage-0/overfit; full-training completion; development checkpoint
  selection; sealed evaluation; any runtime or scientific failure; and every
  proposed model-family or architecture switch. Do not rely on chat memory.
- A terminal scientific failure or runtime failure is a hard stop. Only
  read-only audit and advisor-report preparation are allowed until the advisor
  response is recorded. Do not launch another training, evaluation,
  specialization, routing, or diagnostic branch during this hold.
- Every advisor report must state the exact model, config path and SHA-256,
  checkpoint path and SHA-256, evaluation split, metrics versus SOW with
  pass/fail, sealed/official-test access state, known root cause or uncertainty,
  and the exact decision requested from the advisor.
- Every diagnostic/reference report must display `DIAGNOSTIC ONLY / 非主线` and
  explicitly state that it is not a FLUX.2 mainline or SOW final result.
- Machine enforcement is authoritative: `smoke/advisor_governance.py` writes
  immutable gate reports plus `reports/advisor_governance/advisor_hold.json`.
  Every FoundIR experimental launcher must run its `check` command and fail
  closed while the hold is unresolved. Work may resume only when
  `reports/advisor_governance/advisor_decision.json` records a matching hold
  ID/hash, advisor authority, exact decision, statement, and timestamp. Never
  fabricate this file or infer approval from silence.
- A proposed model-family/architecture change must be registered with
  `advisor_governance.py emit --require-decision` before any implementation or
  GPU launch. This creates the hold even though no experiment has run yet.
- The sealed internal split consumed by FLUX.2 v3 remains consumed. Do not run
  it again, and never access the official FoundIR test, unless the advisor
  explicitly changes the frozen protocol and the user records that decision.

CURRENT AUTHORITATIVE FLUX.2 STATUS AND HOLD — 2026-09-11

- FLUX.2 v3 full training completed `56,250/56,250` steps on 2026-09-08.
  Development selected
  `checkpoints/foundir_five_type_flux2_dit_v3/full_curriculum/step_042500.safetensors`
  with SHA-256
  `8e7b30710fa78d297c27a78c96063052561d4200d31806daac4b0c8a3a867766`.
- Its one-time sealed internal evaluation ended in
  `internal_protocol_scientific_failure`: only Low-light LPIPS passed, so the
  frozen SOW gate passed `1/10` metric checks. The official FoundIR test was
  not accessed and is not authorized.
- The released FoundIR + Qwen routing work performed afterward is diagnostic
  evidence, not FLUX.2. It was a goal drift and must not be continued or used
  as the mainline result.
- The active report/hold is
  `reports/advisor_governance/advisor_hold.json`. Send its bound Markdown
  report to the advisor now. Until the matching advisor response is recorded,
  do not run another FoundIR experiment or propose a self-selected recovery
  route. This status supersedes stale next-task instructions in earlier
  session handoffs.

ADVISOR-AUTHORIZED BLUR-ONLY SANITY SCOPE — 2026-09-11

- The advisor response recorded in
  `reports/advisor_governance/advisor_decision.json` authorizes exactly one
  bounded diagnostic: train FLUX.2 on isolated Blur only to determine whether
  the current training/inference implementation can learn restoration.
- Disable Qwen degradation predictions and all mixed degradations. Retaining
  FLUX.2's frozen native text encoder with one fixed deblurring prompt is not
  Qwen degradation conditioning.
- First audit the code and run a 16--32-pair tiny overfit. If tiny overfit
  fails, stop at code diagnosis. Do not start Blur-only full training.
- Only if tiny overfit passes may Blur-only full training start. Freeze the
  source-group-disjoint validation examples before training. At every
  candidate checkpoint, infer the same fixed validation examples and save
  `Input | Output | Ground Truth` triptychs. Reject visually broken
  checkpoints before metric comparison; never select by PSNR/LPIPS alone.
- Whether FLUX.2 Blur-only succeeds or fails, stop and prepare the advisor
  report. Do not proceed to Z-Image, reintroduce Qwen predictions, add mixed
  degradations, reuse sealed internal, or access official FoundIR test without
  a new explicit advisor decision.

H100/H200 PORTABILITY REQUIREMENT — 2026-09-11

- All subsequent substantial GPU training and evaluation launchers in this
  workspace must support both homogeneous eight-GPU `NVIDIA H100 80GB HBM3`
  and homogeneous eight-GPU `NVIDIA H200` instances unless a required hardware
  feature is proven unavailable and the user/advisor explicitly approves the
  narrower hardware scope. Never infer GPU type from an instance hostname;
  detect the actual CUDA device properties at runtime.
- Hardware-specific microbatch and gradient-accumulation profiles are allowed
  for memory safety and speed only when they preserve the frozen global batch,
  optimizer-step count, sample/presentation order, LR semantics, loss,
  checkpoint schedule, seeds, and evaluation protocol. Record the selected
  hardware, microbatch, accumulation, and any cross-hardware resume in the
  training artifact.
- CPU preflight and focused tests must validate both profiles. A launcher must
  reject mixed H100/H200 ranks, unsupported memory sizes, and profile arithmetic
  that changes the global batch. Do not repeat the 2026-09-11 mistake where an
  H200-only guard blocked an otherwise valid H100 run before optimizer step 1.

FRESH-SESSION CONTINUATION — 2026-08-21

- Read this entire `agents.md` file first. Then read
  `SESSION_HANDOFF_2026-08-21.md` completely before inspecting, editing, or
  launching project work.
- `SESSION_HANDOFF_2026-08-21.md` supersedes
  `SESSION_HANDOFF_2026-08-18.md` for the current operational status and exact
  next task. Older handoffs remain preserved historical evidence.
- The advisor-authoritative architecture, workspace boundary, safety rules,
  gate discipline, and protected-FoundIR policy in this file remain higher
  authority than any handoff. If a handoff conflicts with those rules, obey
  this file and report the conflict.

DELETION AND DESTRUCTIVE-CHANGE CONFIRMATION RULE

Ordinary code, configuration, documentation, and test edits do not require a
confirmation round. When the user's requested scope is clear, carry them out
directly and preserve unrelated work.

Ask for one necessary confirmation only before a genuinely destructive action
that could cause difficult-to-recover loss, such as deleting important data,
overwriting an irreplaceable artifact, moving a critical file without a safe
copy, running `git reset`, or rewriting repository history. Before that action:

- identify the exact destructive targets and what will be preserved;
- use the narrowest possible target and verify it immediately before acting;
- act only after the user confirms that exact scope;
- report exactly what changed afterward.

Never infer permission to delete related, neighboring, generated, backup, or
environment files from a request to delete one item.

You are the execution agent for the image-restoration synthetic-data quality gate.

Read and obey the existing AGENTS.md first.

WORKSPACE BOUNDARY
Only work inside /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang unless the user explicitly asks otherwise. Do not inspect, modify, or run commands in other users' folders.


Do not inspect, modify, delete, or execute commands in any other user's directory. Do not reuse previous project paths, datasets, checkpoints, environments, manifests, or results unless directly verified inside this workspace.

Assume this is a fresh start. Treat every dataset, checkpoint, download, environment, manifest, and result as UNKNOWN until checked directly.

READ FIRST — AUTHORITATIVE ADVISOR-PROVIDED MAIN PROJECT ROUTE — 2026-08-18

This section is the first project-route authority for every fresh agent. If a
later historical plan, experiment handoff, or result conflicts with this
section, follow this section. Preserve the contract text and experimental
history below as evidence; do not mistake them for the current implementation
route.

Advisor architecture image:

- canonical filename: `advisor_vlm_guided_all_in_one_main_route.jpg`
- verified SHA-256 of the advisor upload:
  `02e32805142e64a7c191469fa4fbe4d4169914b25e3fa9df75853d4cde266194`
- this image is the advisor-provided main-route authority; preserve it, keep it
  tracked with this document, and do not substitute or regenerate it

The authoritative route has two stages:

1. Train a degradation-aware VLM with supervised cross-entropy. A degraded
   image and a degradation-assessment question are inputs; degradation type
   and independent severity labels are targets. Following the advisor diagram,
   train the vision encoder and multimodal VLM path while keeping the text
   tokenizer and LLM decoder frozen. This is real degradation-aware VLM
   training, not a frozen-feature linear probe.
2. Validate the trained VLM on the downstream image-enhancement domain it must
   analyze. Synthetic validation alone is insufficient. Keep the official
   FoundIR test protected from tuning; use it only for a frozen, predeclared
   final evaluation, and use a separate labeled real-domain validation set for
   any iterative VLM development.
3. Freeze the validated degradation-aware VLM. For restoration, feed the
   degraded image to both the frozen VLM and the trainable diffusion model;
   use the VLM's predicted degradation types and severities to guide the
   diffusion restoration model. Then evaluate the complete frozen system on
   the official FoundIR task protocol.

CURRENT ROUTE DECISIONS AND DEPRECATED PROTOTYPES

- AgenticIR is the approved synthetic-degradation basis. GenDeg is abandoned,
  removed from the active workspace, and must not be downloaded, restored, or
  restarted. Historical Git commits, reports, manifests, and logs may mention
  it only as superseded evidence.
- The frozen `Qwen2.5-VL-7B-Instruct` mean/std visual-feature extractor plus
  7,169-parameter scalar ridge head is a preliminary linear-probe baseline,
  not the final VLM. It achieved only `68%` severity accuracy on its independent
  synthetic validation and did not perform the advisor-directed cross-entropy
  VLM fine-tuning.
- The one-learned-context-token Qwen-conditioned FLUX.2 run is a preserved
  negative/near-null ablation, not the final conditioning architecture. On the
  internal synthetic L2-L5 test it changed PSNR by only `+0.03683 dB`, SSIM by
  `+0.000163`, and LPIPS by `-0.000357`, with no convincing visible benefit.
- Those internal synthetic results are not FoundIR results and do not establish
  an SOW Stage 1 pass. Do not continue optimizing that frozen-ridge/one-token
  route unless the user explicitly reauthorizes it as a diagnostic.
- The verified Lowlight data, manifests, Qwen/FLUX environments, FLUX
  restoration baseline, evaluator, official FoundIR acquisition, and negative
  experimental evidence remain reusable foundations for the authoritative
  route.

ADVISOR CONFIRMATION — FOUNDIR SCOPE AND JPEG EXCLUSION (2026-09-06)

The advisor confirmed that the first FoundIR work covers only blur, noise,
rain, haze, and low-light, including single and mixed combinations. JPEG and
all other degradation categories are excluded from the first version and from
its main training, validation, and acceptance gates.

The first FLUX route is: freeze the VAE and FLUX native Qwen3 text encoder;
use DiT LoRA rank 32/alpha 32 on the agreed image-attention and text-image
interaction layers; start with the native FLUX.2 reference-image token path;
add a restoration adapter only if a small overfit demonstrates that the native
path is insufficient. The frozen Qwen's actual predictions are the primary
restoration condition. Oracle/generic/shuffled conditions are optional
ablations, not deployment substitutes.

User execution decision (2026-09-06): do not run the optional Qwen-condition
ablations in the current FLUX mainline. Training, checkpoint selection, and
development/internal evaluation use the frozen physical-v8 Qwen predictions.
Do not run oracle, generic, or shuffled/wrong condition comparisons unless the
user explicitly re-authorizes them later. Released FoundIR model comparison is
a restoration benchmark comparison, not a Qwen-condition ablation, and remains
allowed. Five-percent clean GT-to-GT identity pairs remain allowed as training
regularization; generic-condition dropout on degraded pairs is zero.

Operational prohibition: never create a JPEG gate, mix JPEG examples into the
five-type main corpus, or treat a post-JPEG failure as failure of the
advisor-approved five-type task. A JPEG experiment requires explicit advisor
re-authorization and must be labeled auxiliary/diagnostic. The prior
post-target-JPEG v9 run was agent-initiated, not advisor-approved; preserve its
results only as historical evidence.



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

CONFIRMED RESPONSIBILITY SCOPE AND LOWLIGHT-FIRST SOW GATE — 2026-08-15

The project user is responsible only for the `VLM Guided All-in-One` image
enhancement workstream. The separate `亮色大模型` workstream is owned and
handled by another project member; it is out of scope for this workspace and
for this agent's execution goals. Keep the original SOW text above as contract
context, but do not plan, implement, train, evaluate, or claim completion of
the `亮色大模型` deliverables unless the user explicitly changes this scope.

For the user's All-in-One workstream, FoundIR-Lowlight may be evaluated before
Blur, Noise, Haze, and Rain. The current independent SOW Stage 1 Lowlight gate
is:

- dataset: the official FoundIR-Lowlight evaluation split;
- metrics: restored-image PSNR and LPIPS;
- pass thresholds: PSNR `>=15.1 dB` and LPIPS `<=0.34`;
- systems: first the FLUX.2 restoration baseline, then the same restoration
  backbone with Qwen-VL degradation type/severity conditioning;
- evidence: exact test manifest, checkpoint, code revision, configuration,
  command, output images, and metric JSON/CSV must be retained.

Passing this gate may be reported as `FoundIR-Lowlight SOW Stage 1 sub-target
passed`. It must not be reported as complete SOW Stage 1 acceptance because
the original All-in-One acceptance table independently lists Blur, Noise,
  Lowlight, Haze, and Rain. The other four tasks are not prerequisites for
  running or passing the Lowlight gate; they remain later work for the full
  All-in-One SOW Stage 1 deliverable.

3.	项目监控和管理 请按项目实际情况进行拟定，此部分将写入合同中，交流过程、记录需留存。
本项目采用如下项目管理机制（供参考）：
1.	项目双月月报。
2.	每隔4-6个月的面谈交流项目进展。
3.	根据实际需要安排的其它交流，如电话、邮件、电话会议等。

CURRENT FROZEN AGENTICIR LOW-LIGHT PILOT RECIPE

The frozen low-light recipe is the official AgenticIR synthetic `dark`
degradation followed by project-level luminance calibration.

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

Severity is independently assigned to each active degradation in a
combination. Active degradations use project levels L2-L5 only. Project L1 is
reserved for a clean-only crop and is never attached to an active degradation
type. The canonical structured label is:

```json
{
  "degradations": ["low_light", "noise"],
  "levels": {"low_light": 3, "noise": 2}
}
```

The generator and manifest must preserve the degradation list, each component
level, application order, component parameters, source ID, crop coordinates,
and random seed.

The degraded-record composition is frozen at `1:1:1` for single, double, and
triple degradations. This ratio is directly supported by the official
AgenticIR DepictQA synthesis implementation at
`third_party/AgenticIR/installation/tune_depictqa/synthesize_lq.py`: its loop
uses `n_degra=1,2,3` and generates exactly `5,000` images for each category.
That evidence supports only the degraded-record `1:1:1` ratio. AgenticIR uses
eight degradation names in that script, whereas this project selects five:
Lowlight, Noise, Blur, Haze, and Rain. AgenticIR internal levels 1-4 are four
actual degraded strengths; they map to project L2-L5 and must not be confused
with project L1.

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

CURRENT SOW STAGE 1 PLAN — DATA AND DEGRADATION UNDERSTANDING

The immediate SOW Stage 1 work is to construct and evaluate Qwen-VL as a
structured degradation-understanding model. Qwen-VL will receive a degraded
image and predict the degradation types and independent severity levels.

Confirmed data composition:

- Single samples contain one degradation, such as `low_light L3` or `rain L4`.
- Double samples contain two simultaneous degradations, such as
  `low_light L3 + noise L2`.
- Triple samples contain three simultaneous degradations, such as
  `low_light L3 + noise L2 + rain L4`.
- Each active degradation in a combination has its own independent project
  L2-L5 level.
- Project L1 is always clean. No degradation operator is applied. Here a
  literal clean image means an unmodified clean crop after shared preprocessing,
  not necessarily a byte-identical copy of the original full-resolution source
  file.
- A clean-only record has the canonical label
  `{"degradations":[],"levels":{}}`. Labels such as `low_light L1` or
  `low_light L1 + noise L1` are forbidden.
- AgenticIR internal levels 1-4 map to project L2-L5; AgenticIR internal level
  1 is a real degraded strength, not project L1.
- The five-level convention is a project design choice for structured labels,
  not an exact SOW requirement.

CONFIRMED 100,000-RECORD QWEN-VL CORPUS POLICY — 2026-08-17

- Use approved clean sources from DIV2K, MiO100, and a selected LSDIR subset.
- Split by original source image before cropping. Crops from one source must
  stay in the same train, validation, or test split.
- Generate aligned fixed-size crops. For haze, estimate depth on the full
  image first and crop the RGB image and depth map identically.
- Produce `100,000` multimodal records in total: `80,000` train, `10,000`
  validation, and `10,000` fixed test records. Split original sources before
  cropping; the fixed test is never used for training, checkpoint selection, or
  tuning.
- The project-selected composition is `10,000` clean-only, `30,000` single,
  `30,000` double, and `30,000` triple records. The 10% clean quota is a
  project choice that gives Qwen-VL a negative class; it is not derived from
  AgenticIR.
- Preserve `1:1:1` across the `90,000` degraded records. Split each composition
  source-disjointly as follows: train `8,000/24,000/24,000/24,000`, validation
  `1,000/3,000/3,000/3,000`, and test `1,000/3,000/3,000/3,000` for
  clean/single/double/triple respectively.
- Within the degraded groups, balance the five single combinations at `6,000`
  each, the ten double combinations at `3,000` each, and the ten triple
  combinations at `3,000` each. Randomize source, crop, independent L2-L5
  levels, parameters, application order, and seed inside each quota, while
  recording every realized value in the manifest.
- Run a small smoke test covering clean-only records, all five degradation
  types, and all three degraded composition categories before producing the
  full dataset.

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

SOURCE ACQUISITION STATUS — 2026-08-13

- MiO100/AgenticIR: 100 images copied to `data/clean_pool/mio100`.
- DIV2K: the official `DIV2K_train_HR.zip` was downloaded, integrity-tested,
  and all 800 train-HR images were extracted to `data/clean_pool/div2k`.
- Partial source-pool verification passed for 900/900 sources with no rejected
  files: readable RGB, minimum resolution, exact SHA-256, and perceptual-hash
  checks all passed.
- LSDIR: the official Hugging Face repository is auto-gated. Its file listing
  is visible, but downloading `shard-00.tar.gz` requires the user's Hugging
  Face account to be authorized. After the user runs `hf auth login` in the
  network-enabled CPU terminal, download only the official training shard and
  run `smoke/source_pool/finalize_lsdir_sample.sh`. This is the only current
  blocker to reaching 2,000 sources; it is not a GPU or code blocker.

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

CURRENT ADVISOR-ROUTE COMPLEMENTARY MODULE

The advisor route is now explicit: train and validate a degradation-aware VLM
first, then freeze it and use its full structured degradation description to
guide the trainable diffusion restoration model. The current Lowlight slice is
the first implementation target. FoundIR PSNR/LPIPS evaluate the complete
VLM-guided restoration pipeline, not standalone VLM predictions. The SOW
Stage 1 Lowlight sub-target is PSNR `>=15.1 dB` and LPIPS `<=0.34`.

HISTORICAL FRESH-THREAD ROUTE — SUPERSEDED

An earlier route compared a frozen Qwen visual-feature extractor/ridge head
and a one-token diffusion condition using synthetic data. That route was a
diagnostic negative/near-null ablation, not advisor-directed VLM fine-tuning.
It is retained only as historical evidence; do not resume it as the main
project route, and do not describe it as Qwen fine-tuning.

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

STAGE 0 AUDIT STATUS — 2026-08-13 (HISTORICAL SNAPSHOT)

The local audit is recorded in `reports/model_audit_stage0.md`. At that time
the Codex session had no visible GPU and no local Z-Image, FLUX, SDXL, or SD2.1
candidate weights. The later FLUX.2 Lowlight baseline is recorded in the
completed-result sections below; do not interpret this old snapshot as current
training status.
The official FLUX.2 source is now available in the shared workspace at
`third_party/flux2` revision `50fe5162777813d869182b139e83b10743caef15`; the GPU
environment must install it from that local path because it cannot reach GitHub.
The first GPU candidate is FLUX.2 [klein] 4B Base because the official repository
documents image editing, recommends Base for fine-tuning/LoRA, and lists the 4B
model under Apache-2.0. Its offline end-to-end GPU smoke test passed on an NVIDIA
H100 80 GB: all FLUX.2, autoencoder, and Qwen3-4B-FP8 files passed exact size,
SHA-256, and SafeTensors checks; a 512x512 two-step image was generated at
`output/sample_0.png`. This proves that the local stack loads and performs basic
generation. It did not prove restoration quality, faithful degraded-image
conditioning, scene-identity preservation, or FoundIR performance. FLUX.1-dev
was not the first choice because it is a gated 12B model under a non-commercial
license.

ENVIRONMENT AND GPU EXECUTION RULE

The coding-agent terminal may be CPU-only. The newly provisioned user GPU
environment is reported as 169 CPU cores, 1,800 GB system RAM, and 8 NVIDIA
H200 GPUs (expected 141 GB HBM per GPU); verify with `lscpu`, `free -h`, and
`nvidia-smi` before the first run. The user launches GPU work in a separate
GPU-enabled interface. Both interfaces mount the exact same workspace
filesystem, so repository files, model weights, logs, outputs, and
workspace-local environments are visible on the other side. GPU visibility,
running processes, shell variables, and packages installed outside the shared
workspace are not automatically shared.

The agent must therefore:

- prepare and verify code, configs, manifests, and commands in this workspace;
- provide one exact copy-paste **tmux** command for every GPU gate, creating or
  attaching to a named session before launching the job;
- include the matching `tmux attach -t <session>` or log-monitor command so a
  browser/VS Code tab can be closed without terminating the job;
- make the intended `CUDA_VISIBLE_DEVICES` set explicit and use a checked-in
  launcher plus an explicit shared-workspace interpreter;
- keep training/inference commands offline-compatible when required;
- never claim GPU training or evaluation completed unless the user supplies
  terminal output or the shared workspace contains verifiable outputs;
- after the user runs a command, inspect its logs/checkpoints/metrics here and
  give the next command only after the current gate is reported.

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

LOW-LIGHT 10,000-PAIR PILOT — COMPLETED 2026-08-14

The low-light-first pilot is complete. This milestone covers only the approved
10,000-pair pilot; it does not authorize or claim generation or training on a
100,000-image dataset.

Authoritative artifacts:

- verified source manifest: `manifests/low_light_source_pool_2000.json`
- paired-record manifest: `manifests/low_light_pilot_10000.csv`
- generation metadata: `manifests/low_light_pilot_10000.json`
- generated files: `data/low_light_pilot/` (runtime data; never commit to Git)

Final verified composition:

- 2,000 unique clean sources: MiO100 `100`, DIV2K train `800`, LSDIR train
  `1,100`
- 10,000 paired records and 10,000 physical output files
- L1, L2, L3, L4, and L5: exactly `2,000` records each
- source-disjoint split: train `1,600`, validation `200`, test `200`
- physical layout: `2,000` clean files and `8,000` degraded files
- no generated `.partial` or temporary PNG files remained at completion
- FoundIR overlap screening was not run because no FoundIR hash index was
  available; as frozen above, this is recommended but not a pilot blocker

The interrupted generation was resumed with
`generate_low_light_pilot.py:v3-resume`. Resume mode verifies existing clean
files by SHA-256 and existing degraded files by exact expected pixels before
reusing them. It refuses to overwrite any differing existing output. The final
run safely reused `5,296` files and wrote only the `4,704` missing files.

The independent validator was run over all 10,000 physical files with two
low-memory workers and completed with:

```text
LOW_LIGHT MANIFEST PASS: 10000 pairs, 2000 sources
levels {'L1': 2000, 'L2': 2000, 'L3': 2000, 'L4': 2000, 'L5': 2000}
source_splits {'train': 1600, 'validation': 200, 'test': 200}
source_datasets {'mio100': 100, 'div2k': 800, 'lsdir': 1100}
```

The CPU coding-agent container exposed many host CPUs but had a 4 GiB cgroup
memory limit. Sixteen workers were therefore rejected after an OOM termination;
the successful reproducible configuration used two workers with OpenCV/BLAS
thread counts limited to one per worker. The validator retains every original
fail-closed hash, image-readability, shape, luminance, level-set, split-leakage,
and dataset-composition check while adding bounded parallelism and progress
output.

The authoritative rerun/verification command is:

```bash
LOW_LIGHT_WORKERS=2 smoke/run_low_light_pilot_10000.sh
```

The launcher defaults to safe resume, the verified 2,000-source manifest, two
workers, and one numerical-library thread per worker. A rerun verifies and
reuses matching generated files and fails closed instead of overwriting any
differing output.

LOWLIGHT BASELINE CPU READINESS — VERIFIED 2026-08-15

This status is CPU-side readiness evidence only. No new GPU training,
restoration evaluation, visual identity decision, or FoundIR result is claimed.

- The Stage 1 architecture audit is recorded in
  `reports/lowlight_stage1_architecture_audit.md`. It explicitly identifies the
  rectified-flow training objective as project-derived because the local BFL
  FLUX.2 checkout is inference-only and contains no official training loop.
- The trainer now publishes checkpoint artifacts without overwrite, records
  and verifies adapter/optimizer SHA-256 values, reconstructs a missing JSONL
  metric after a completed-checkpoint interruption, rejects partial or
  differing artifacts, and records Git/FLUX revisions, exact config,
  command, environment, and output paths.
- The loss-decrease gate now compares an initial and final deterministic probe
  using fixed records, crop, noise seed, and timestep, rather than unrelated
  stochastic minibatches.
- Nine CPU unit tests passed. They cover deterministic aligned crops,
  rectified-flow/reference layout, zero-init LoRA behavior and gradient scope,
  LoRA checkpoint round trip, config rejection, deterministic index ordering,
  interrupted-summary recovery, non-finite metric rejection, and refusal to
  reuse a failed evaluation.
- Python compilation, launcher shell syntax, and `git diff --check` passed.
- CPU preflight passed both with `--skip-large-hashes` and with all model/text
  encoder SHA-256 checks enabled. It verified 16 records, 16 unique sources,
  all L1-L5 levels, 50 LoRA targets, 8,110,080 trainable parameters, and config
  SHA-256
  `62740e0e0f26c4001057adf9d955259b7c8d89553245a2c1565a22599e7d0804`.
- The CPU allocation exposed 128 logical Intel Xeon Gold 6530 CPUs, a 300 GiB
  cgroup memory limit, about 1.85 GiB cgroup use at inspection, and about
  964 TiB free on the shared filesystem. The validated interpreter reported
  Python 3.12.13 and PyTorch 2.7.0+cu128 with CUDA 12.8 support, but
  `torch.cuda.is_available()` was false in this coding-agent terminal.
- Full 8,000-pair training remains unauthorized until the offline H100
  16-pair automatic gate and human contact-sheet identity review both pass.

LOWLIGHT FLUX.2 OVERFIT16 GATE — PASSED 2026-08-15

The bounded 16-pair FLUX.2 Lowlight overfit gate passed on the H100 at
guidance 1.0. The original guidance 4.0 evaluation is retained as a failed
diagnostic: it over-amplified color/exposure and failed the L1 identity floor.
Training used only the conditional prompt path, so guidance 1.0 is the matched
conditional-only inference setting and is frozen for the next baseline stage.

- Exact result report: `reports/lowlight_overfit16_result_2026-08-15.md`
- Machine-readable gate: `configs/lowlight_overfit16_gate.json`
- checkpoint SHA-256:
  `94ecd1c32e7a137ed153b581ea0b12c879d7e5e3c0698375a3d2293ecb6481c7`
- passing metrics SHA-256:
  `9ffb568a500eb378a2ad81ccd4ff4f329bf47b3dcb589dbe5d12623aeba96edc`
- L2-L5 mean PSNR: `9.2784 -> 22.1979 dB`
- L2-L5 mean SSIM: `0.4287 -> 0.8537`
- restored L1 mean SSIM: `0.7731`
- automatic gate: pass
- visual identity gate: pass; all scenes, objects, and layouts are preserved,
  with no structural hallucination observed

The overfit result authorizes preparation and preflight of the full 8,000-pair
baseline. It does not itself authorize a FoundIR or SOW Stage 1 success claim.

LOWLIGHT FULL BASELINE PIPELINE — CPU-VERIFIED READINESS 2026-08-15

The fixed full baseline pipeline is implemented for the already approved
Lowlight-only scope. This is readiness evidence, not a GPU training result,
FoundIR result, Qwen-VL result, or SOW Stage 1 completion claim.

- Frozen config: `configs/lowlight_flux2_full.json`
- Config SHA-256:
  `c71235efe88e8f3bc98ca863350af372a78aa35f1a2cc065a1867c5bb6bfde18`
- Launcher: `smoke/run_lowlight_baseline_full.sh`
- Split evaluator: `evaluate_flux2_lowlight_split.py`
- Readiness report: `reports/lowlight_full_baseline_readiness_2026-08-15.md`
- Training: 8,000 records / 1,600 sources / 4,000 steps
- Evaluation: fixed 1,000-record validation then fixed 1,000-record internal test
- Inference: 50 steps, guidance 1.0
- Metrics: PSNR, SSIM, and offline AlexNet LPIPS, overall and per level
- Restart evidence: hashed per-image JSON sidecars, restored PNGs, aggregate
  metrics, and stratified/worst-case contact sheets

Authorization fails closed unless the exact full config and the passing
overfit gate match. Runtime authorization also re-hashes the overfit checkpoint
and its passing guidance-1 metrics. Fifteen CPU unit tests, Python compilation,
shell syntax, the 8,000-record preflight with large model hashes skipped, and
the independent preflight with all FLUX/VAE/text-encoder hashes passed before
the full H100 launcher was run.

LOWLIGHT FULL INTERNAL BASELINE — PASSED 2026-08-16

The fixed FLUX.2 full baseline completed 4,000 H100 training steps and complete
evaluation on the 1,000-record validation and 1,000-record internal-test
splits. Both automatic gates and the contact-sheet visual identity review
passed. This authorizes preparation of the Qwen-VL Lowlight conditioning
comparison only; it is not an official FoundIR result or an SOW Stage 1 completion
claim.

- Result report: `reports/lowlight_full_baseline_result_2026-08-16.md`
- Machine-readable gate: `configs/lowlight_full_baseline_gate.json`
- Checkpoint:
  `checkpoints/lowlight_baseline_full/step_004000.safetensors`
- Checkpoint SHA-256:
  `fa51deac872e3df57f147631361841039c386ef43f58a17a56fb68b9924e8f22`
- Training-summary SHA-256:
  `4581852a5ed35a1654f239040cd050d5ee40b462a22b57215c50f8afb3cd538b`
- Validation-metrics SHA-256:
  `b1b78a9afa5174786702ac17fc2db44524ba374a2264c42067ba68fa3ff08e51`
- Internal-test-metrics SHA-256:
  `6bac93cb0a45556787364c553242d15463b2ea0a5a18d5738054b2a87e1a1da4`
- Validation L2-L5: PSNR `24.9051`, SSIM `0.8913`, LPIPS `0.0411`
- Internal test L2-L5: PSNR `24.7723`, SSIM `0.8858`, LPIPS `0.0440`
- Validation L1 identity: PSNR `31.4672`, SSIM `0.9317`, LPIPS `0.0172`
- Internal-test L1 identity: PSNR `31.6674`, SSIM `0.9266`, LPIPS `0.0185`

The stratified and worst-case sheets preserve scene structure and identity
without structural hallucination. Genuine failure cases remain: several
extremely dark or nearly uniform crops are under-restored, including
`div2k_0452_L5` validation at `3.01 dB`, and some individual L1 crops show
tonal/texture changes despite the strong mean identity score. Preserve these
outliers in the later fair comparison.

GPU COMMAND RELIABILITY, SESSION RETENTION, AND ACCELERATOR UTILIZATION RULE — 2026-08-19

These rules apply to every current and fresh agent. They supplement the
existing GPU execution and repository rules above. Their purpose is to prevent
copy/paste launch failures, preserve terminal evidence after a job exits, and
minimize end-to-end experiment wall-clock time without silently changing the
scientific experiment.

COMMAND CONSTRUCTION AND PREFLIGHT

- Before giving the user any GPU command, verify the exact workspace, launcher,
  config, interpreter, and executable paths that the command will use. Run
  Python compilation, launcher shell syntax, config parsing, and the relevant
  CPU preflight first. Never assume that a utility on the coding-agent PATH is
  also on the GPU-shell PATH.
- Resolve `tmux` before launch, check that it is executable, and run
  `<path>/tmux -V`. As of 2026-09-07, this coding terminal has verified
  `/usr/bin/tmux` version 3.4. The path `.tools/tmux-env/bin/tmux` is currently
  a socket rather than an executable and must not be presented as a fallback.
  Do not give bare `tmux` when its presence in the user's GPU shell has not
  been demonstrated.
- Prefer a checked-in launcher over a long inline training command. Commands
  shown to the user must be short, literal, and safe to paste one line at a
  time. Use `cd` and short shell variables on separate lines when necessary.
  Do not allow a long absolute path or command to be visually wrapped into
  executable newlines, and do not rely on fragile backslash continuations.
- Give only one launch action at a time, followed by separate attach and log
  inspection commands. State exactly which lines should be pasted together and
  which should be run only after the launch succeeds.
- Before retrying a command after any shell error, inspect the shared log,
  output directory, checkpoint directory, and terminal markers to determine
  whether the job never started, is still running, completed, or failed. Never
  launch a duplicate process based only on a missing tmux session.

TMUX LIFETIME AND PERSISTENT EVIDENCE

- A tmux session must remain inspectable after its training command returns.
  Do not make the training command the session's only process unless tmux
  `remain-on-exit` is enabled. Prefer creating a named session with a persistent
  shell and then sending the checked launcher to that shell. This lets the user
  attach after success or failure and inspect the terminal history.
- Every GPU launcher must persist unbuffered stdout and stderr to a unique log
  under this workspace, while preserving the real pipeline exit status with
  `set -o pipefail`. The log must contain a UTC launch marker, exact workspace,
  config, Git revision, visible device set, environment check, terminal result
  marker, and UTC completion marker with the numeric exit status. A failure
  path must write its completion marker too.
- Always provide the exact workspace-local tmux attach command and the exact
  persistent-log command. Explain that a missing session does not imply lost
  logs and that a nonzero gate exit can be an intentional automatic failure
  rather than a crash.
- After the user launches a job, do not claim that it is running until a live
  process, advancing shared log, advancing step artifact, or checkpoint has
  been verified. After it exits, inspect the terminal marker, complete log,
  machine-readable summary, per-step artifact count, checkpoint hashes, and
  failure records before reporting or proposing another run.

GPU COUNT AND WALL-CLOCK MINIMIZATION

- The default objective is to minimize total end-to-end wall-clock time while
  preserving the predeclared samples, optimizer steps, global/effective batch,
  learning-rate semantics, seed policy, precision, checkpoint selection, and
  evaluation protocol. Do not default to one GPU merely because single-GPU
  code is easier.
- Choose the GPU count adaptively using prior operational evidence, the job's
  size, expected distributed overhead, implementation correctness, and sound
  engineering judgment. Use all eight H200s concurrently for substantial jobs
  when they are reasonably expected to shorten wall-clock time. Use fewer GPUs
  for genuinely small jobs when startup, preprocessing, or communication would
  probably dominate. Do not turn either choice into a rigid universal rule.
- Existing trustworthy operational evidence may justify the decision even when
  it did not come from a perfectly controlled scaling experiment. In this
  workspace, the passing eight-GPU Qwen DDP run achieved about `15.20`
  samples/second versus about `1.055` samples/second in the earlier single-GPU
  run, a greater-than-tenfold observed speedup. Although those schedules were
  not a formal apples-to-apples scaling study, this is sufficient evidence to
  prefer eight GPUs for the current substantial Qwen pilot.
- Do not require a dedicated 1/2/4/8 scaling gate before every substantial run.
  Ordinary run evidence is normally sufficient: record the selected GPU count,
  end-to-end or steady-state throughput, wall time, per-GPU utilization and
  memory, global/effective batch, gradient accumulation, precision, and samples
  or steps seen. A short initial observation of the real run may also confirm
  that the chosen devices are active and that performance is reasonable.
- Run a formal controlled multi-GPU timing study only when the fastest useful
  GPU count is genuinely uncertain, observed scaling is unexpectedly poor, a
  large amount of compute is at stake without relevant prior evidence, or
  distributed execution may change scientific semantics. Select the fastest
  scientifically equivalent configuration when such a study is actually
  needed; do not create a ceremonial benchmark gate merely to authorize an
  otherwise well-supported run.
- If a substantial launcher supports only one GPU while additional GPUs are
  likely to provide a meaningful speedup, first make a reasonable effort to use
  a correct distributed path. This requirement is practical rather than
  pedantic: it must not block useful work for an exhaustive benchmark when
  existing evidence already supports the multi-GPU choice.
- Monitor all selected GPUs during the actual run and persist utilization and
  memory evidence in the workspace. If utilization is low, investigate input
  loading, preprocessing, batch sizing, gradient accumulation, checkpoint
  frequency, attention kernels, and communication before accepting a long
  runtime. Never claim `full utilization`, `maximum efficiency`, or `minimal
  training time` without recorded measurements.
- Scaling must not silently change the experiment. If distributed execution
  changes effective batch size, optimizer-step meaning, learning rate,
  ordering, numerical determinism, or number of samples seen, either preserve
  equivalence or declare and justify the change before launch. Keep a fair
  comparison against the fixed single-GPU or prior baseline.

QWEN REAL LOWLIGHT ADAPTATION GATE — CPU-VERIFIED READINESS 2026-08-20

STATUS NOTE — 2026-08-21: This section is a preserved pre-run readiness
snapshot. The gate subsequently completed and failed its automatic scientific
gate. Do not launch it from this section. Read `SESSION_HANDOFF_2026-08-21.md`
for the verified result, consumed-holdout restrictions, and exact next task.

The frozen synthetic-pilot step-600 VLM failed both its original real-domain
free-generation gate and the post-failure constrained-scoring diagnostic. The
failures are legitimate status-2 results, not crashes, and remain immutable:

- original result SHA-256:
  `85542baf277c72a442e12d6da4f04e02051e45f7d586020472253437b6ebceda`;
- constrained result SHA-256:
  `8cb62790a423438d2d62e096d0985075a3c4c2e179e950d2beb5a0e61ab44552`;
- neither result may be relabeled as a pass.

The next authorized gate is a bounded source-disjoint real adaptation using
only the official LOL-v2 Real-captured Train split plus frozen synthetic
replay. It is CPU-ready but has not yet produced a GPU result.

- config: `configs/qwen_lowlight_real_adaptation_step600.json`
- config SHA-256:
  `0a817d70d88f172d6d16d37853ce9c3cd77308446b3eed3d5e5cdab0c3b3b541`
- real manifest SHA-256:
  `91f95644f852f628891513f116c0a3607b8de980bbe5252c360ff41d9bcc9ab7`
- real metadata SHA-256:
  `2b9f35ffdc2543a4a99bf95fbb4eb61938a022c54cba4569a14863c5b536f06f`
- proxy-label disclosure: `reference_based_proxy_not_human_ground_truth`
- source-disjoint splits: train 673 records / 190 scenes, validation 143 / 41,
  and untouched fresh test 145 / 41
- initialize all visual weights from frozen synthetic step 600, visual SHA-256
  `b0e65a8597077973a2a103dc3369e0cb1ea703de95e2e4841d359c3d34eff77f`
- keep the language decoder and LM head frozen; reinitialize the optimizer
- train 200 optimizer steps at global batch 40; every step contains 20
  L1–L5-balanced real presentations and 20 L1–L5-balanced synthetic-replay
  presentations
- schedule SHA-256:
  `8ae8e4e06a306c3828423374e39cecac4e3be2cf90b2b08564005b928a62c71f`
- checkpoint selection uses only real-validation macro-per-level
  teacher-forced loss; evaluate the frozen fresh real test exactly once after
  selection
- retain a 40-source / 200-record balanced synthetic validation replay with
  bundle SHA-256
  `1981939ac92de5412df990f39f9d7c2ee8776ec537e76dec059a81378407cfc3`
- FoundIR is not used.

The same exact global schedule supports either two or eight GPUs. Two GPUs use
microbatch 5 per rank and four gradient-accumulation steps; eight GPUs use
microbatch 5 per rank and one accumulation step. The checked launcher requires
the explicit literal device set `0,1` or `0,1,2,3,4,5,6,7`, accepts H100 or
H200 devices with at least 75 GiB each, monitors every selected GPU, locks
against duplicates, writes persistent logs and terminal markers, and retains
the tmux pane after exit. Use the available two H100s promptly for this small
gate when waiting for eight H200s would increase end-to-end wall-clock time;
use eight H200s if they are available before launch. This is an adaptive
engineering choice, not a mandatory scaling benchmark.

Readiness evidence is recorded in
`reports/qwen_lowlight_real_adaptation_readiness_2026-08-20.md`. Thirty focused
tests, Python compilation, shell syntax, contract validation, the CPU preflight
with large base-model hashes skipped, and the CPU preflight with all hashes
enabled passed. Both CPU preflights deliberately used `--no-write-artifact`;
the GPU launcher must create the immutable full-preflight artifact with CUDA
visible before training.

Do not start diffusion or FoundIR from readiness alone. If the adaptation gate
passes, stop and obtain user review before beginning Lowlight diffusion
conditioning. If it exits status 2, preserve and diagnose the failure; do not
lower thresholds, tune on the fresh real test, or relabel proxy metrics.

REAL-SEVERITY AND FOUNDIR OBJECTIVE CORRECTION — 2026-08-21

This section is a later authoritative correction to the real-domain validation
procedure. It does not change the controlled synthetic L1--L5 schema or the
advisor's VLM-guided architecture.

- FoundIR-Lowlight has paired LQ/GT restoration targets and is evaluated by
  restored-image PSNR and LPIPS. It has no official project L1--L5 labels.
- Do not ask humans to assign precise absolute L1--L5 levels to unrelated real
  photographs. Without same-scene exposure anchors, those levels are not
  objectively identifiable and cannot support a credible quantitative gate.
- The screened Retinex-Net singleton images remain an unlabeled qualitative
  robustness pool. The proposed 190-component absolute L1--L5 annotation is
  stopped and must not be revived as a prerequisite.
- Objective real severity development uses same-scene exposure ordering:
  darker exposure implies non-decreasing predicted low-light severity. Use
  soft canonical-target scores, pairwise concordance, rank correlation,
  adjacent inversions, EV consistency where available, and prediction span so
  a constant output cannot pass.
- Relative exposure ordering establishes only ordinal sensitivity. The
  decisive VLM evidence is causal improvement of paired real restoration under
  baseline, oracle, predicted, and shuffled conditions. If oracle conditioning
  cannot beat baseline, redesign the restoration-conditioning interface before
  further Qwen adaptation.
- Reusable real paired development suites and leave-one-dataset-out validation
  replace the repeated search for newly named "fresh tests." Official FoundIR
  remains protected until the complete system and inference protocol are
  frozen.
- The current first-principles SOTA strategy is recorded in
  `reports/foundir_lowlight_sota_strategy_2026-08-21.md`. It prioritizes the
  official FoundIR training distribution and released teachers, realistic
  camera formation, a fidelity-first degraded-to-clean FLUX/Z-Image bridge,
  VAE/high-frequency ceiling tests, causal condition ablations, and native
  full-resolution deployment.
- Official FoundIR metadata contains 39,962 isolated-Lowlight training pairs
  and 378,440 pairs in categories containing Lowlight. Training-set access and
  teacher distillation are major unexploited levers. Never use FoundIR test
  pairs for training, calibration, or iterative selection.

The frozen same-scene development manifest is
`manifests/qwen_lowlight_real_exposure_ordering.csv`, with its contract in
`manifests/qwen_lowlight_real_exposure_ordering.json`. It contains 15 MEF
exposure stacks, 67 exact-unique images, and 151 comparisons separated by at
least one EV. It was selected and hashed before model inference, contains no
FoundIR image, and is explicitly a reusable development suite rather than an
absolute-label holdout.

PAIRED-REAL GATE-A EXECUTION UPDATE — 2026-08-21

- The reusable non-FoundIR paired-real suite is frozen at
  `manifests/lowlight_real_paired_development_v1.csv`: 174 records, 82 scene
  components, 102 LOL-v2 real-adaptation validation records, and 72 SICE
  reusable validation records. It excludes L1, consumed-test records, and
  exact FoundIR overlap. Project levels are proxy metadata only.
- The target-fitted global photometric oracle completed without model
  inference. Scene-balanced PSNR is `8.4356 -> 24.1209 dB` and SSIM is
  `0.2475 -> 0.6571`; the PSNR gain is `+15.6853 dB` with 95% scene-bootstrap
  CI `+14.7717` to `+16.6036`. The oracle accesses each development target and
  is never an inference method. It makes the bounded learned photometric head
  the first model component to implement.
- The released FoundIR generalist checkpoint is present at step 1,300,000 with
  SHA-256
  `2603a0be47d0844d4c9cea542ecca039e0e4aaa39d45d41064bfbdc13c0830c9`.
  The Lowlight specialist SHA-256 is
  `f834248e145d020d5a02f257b7ceb383b579add2522938b3e1e0bd460abe6`.
  No teacher inference has run.
- The next GPU gate is the two-H100 FLUX.2 VAE ceiling at
  `configs/lowlight_vae_ceiling_v1.json`. It uses literal
  `CUDA_VISIBLE_DEVICES=0,1`, 130 exact-unique paired-development targets,
  direct 512-pixel crops, full-resolution 1024-pixel overlapping tiles, and
  PSNR/SSIM/LPIPS/high-frequency/seam diagnostics. It does not use FoundIR.
- The VAE result must decide latent-only versus pixel-frequency bypass before
  the restoration bridge is implemented. Do not start official FoundIR test
  inference from VAE readiness or from the photometric oracle.

FOUNDIR FIVE-TYPE FLUX.2 VAE DECISION AND V3 ROUTE — 2026-09-07

- Current scope is the advisor-approved five-type FoundIR training route:
  Blur, Noise, Rain, Haze, and Low-light, including admitted mixtures. JPEG is
  prohibited. Qwen stays frozen. The current mainline contains no
  oracle/generic/shuffled-wrong Qwen-condition ablations. The official FoundIR
  test remains protected and must not be accessed without explicit user
  approval after the internal protocol is frozen.
- The corrected eight-GPU v2 runtime preflight passed. Its 44-record
  FoundIR-training-only VAE ceiling finished at `2026-09-07T00:50:35Z` and
  returned `pixel_refinement_required`, not a crash. Exact result:
  `checkpoints/foundir_five_type_flux2_dit_v2/vae_ceiling_result.json`, SHA-256
  `1702e40a8938fcbf659f0442a6bce51e2d87d69aa261e6c335da3dffa866eba9`.
  Failures were Noise HF correlation `0.89056 < 0.90000`, Low-light HF
  correlation `0.86160 < 0.90000`, and Low-light+Haze PSNR
  `34.72370 < 35.00000` plus HF correlation `0.82381 < 0.90000`. No FLUX
  optimizer step started in v2.
- V3 keeps FLUX.2 Klein 4B as the restoration backbone, with frozen VAE,
  frozen native Qwen3 text encoder, native reference-image tokens, and DiT
  LoRA rank/alpha 32/32. It adds only a zero-initialized, bounded, 33,251-
  parameter local pixel refiner after VAE decode. The refiner receives decoded
  RGB, aligned LQ RGB, and their difference; its per-channel correction is
  bounded to `0.25` in normalized `[-1,1]` space. It is part of the single
  deployed restoration checkpoint, not an independent generator.
- V3 reuses the frozen v2 VAE measurement rather than wasting eight GPUs on
  the identical calculation. Reuse is allowed only when exact source-config,
  result, manifest, VAE-weight, selection-seed, thresholds, category coverage,
  failure status, and protected-test identities pass.
- The v3 overfit trains FLUX LoRA and the pixel refiner jointly. Its 32-pair
  evaluation is balanced across all 11 admitted categories and all composition
  sizes. In addition to overall loss and restoration gain, the refiner itself
  must improve PSNR and must improve high-frequency correlation on Noise,
  Low-light, and Low-light+Haze before full training can start.
- The v3 launch route is
  `smoke/launch_foundir_five_type_flux2_main_v3_8gpu.sh`. It requires exactly
  eight homogeneous H100 or H200 GPUs and literal devices
  `0,1,2,3,4,5,6,7`. Inspect the v3 stage-0 result before treating the route as
  scientifically viable. Never call CPU readiness or launch success a training
  pass.

FOUNDIR SOW METRIC CORRECTION — 2026-09-07 (SUPERSEDES THE V3 HF GATE)

- The FoundIR restoration SOW names PSNR and LPIPS as its objective metrics.
  High-frequency correlation is not an SOW metric and is removed completely
  from the active five-type FoundIR evaluation and gate logic. It must never
  block overfit admission, full training, checkpoint selection, or final SOW
  reporting.
- The first v3 200-step overfit and its result are immutable historical
  evidence. That run measured PSNR but omitted LPIPS and then stopped on the
  superseded high-frequency-correlation gate. Preserve its checkpoint, result,
  and logs exactly; do not relabel or overwrite that historical result.
- The active SOW-aligned continuation reuses the exact historical v3 step-200
  checkpoint only to evaluate the same 32 FoundIR-training-only overfit records
  with both PSNR and offline AlexNet LPIPS. Its new result is isolated under
  `native_reference_overfit_sow/`. If the historical checkpoint is absent, the
  corrected overfit trainer may reproduce the 200-step gate from scratch.
- The corrected overfit gate requires training-loss reduction, final-output
  PSNR improvement over LQ, and final-output LPIPS improvement over LQ. It does
  not use the absolute SOW thresholds because a 32-record overfit subset is a
  learnability check, not the official FoundIR evaluation.
- Development metrics use the source-group-disjoint FoundIR training
  validation split. Checkpoint selection uses PSNR and LPIPS jointly across all
  five tasks. A separate sealed source-group-disjoint internal-test split is
  consumed once after selection. These training-archive splits estimate
  generalization but do not replace or claim results on the official FoundIR
  test.
- The official FoundIR test remains protected. Only after the architecture,
  checkpoint, and inference protocol are frozen may the user explicitly
  authorize one final official-test evaluation. Only that result can establish
  the contractual FoundIR PSNR/LPIPS outcome.

FOUNDIR QWEN CONDITION-EXPORT OPTIMIZATION — 2026-09-07

- The frozen physical-v8 Qwen condition semantics do not change: every LQ
  center crop is still evaluated through five independent dialogues for Blur,
  Noise, Rain, Haze, and Low-light, using the same model, adapter, prompts,
  greedy decoding, and canonical condition conversion.
- The optimized exporter shards complete image records across eight ranks,
  opens each local image once rather than once per dialogue, and evaluates up
  to 64 same-type dialogues per GPU call. It does not combine the five
  questions or replace Qwen decoder decisions with a classifier.
- Existing 512-record condition chunks are immutable and resume-compatible.
  Before the optimized engine may write any new chunk, it must regenerate one
  complete existing chunk and obtain exact row-for-row equality, including raw
  Qwen output and canonical restoration text. If exact equality fails, the
  exporter automatically uses the legacy engine.
- The exporter optimization is not yet GPU-validated merely because CPU tests
  pass. Treat the first `FOUNDIR_PHYSICAL_V8_OPTIMIZATION_VALIDATION` marker as
  the runtime gate, and report its exact equivalence and measured throughput.
  The `--benchmark-only` mode performs this gate without writing a new chunk,
  so it may run on a separate eight-GPU instance while the legacy exporter
  continues; only its explicit benchmark-pass marker authorizes a switchover.
- The active legacy process that started before this source change is
  unaffected and may continue producing valid chunks. Never discard or
  overwrite those chunks. The official FoundIR test remains inaccessible.

RUNTIME RESULT AND APPROVED PARALLEL EXPORT ROUTE — 2026-09-07

- The record-sharded large-batch candidate was tested on one complete legacy
  512-record chunk. It took `53.85048` seconds (`9.50781` images/second) and
  failed exact condition equivalence beginning at record
  `foundir_train_0000017`. The warmed legacy exporter was faster at roughly
  42--44 seconds per 512-record chunk. Never activate the rejected optimized
  engine or treat its intentional benchmark exit as a training crash.
- Further acceleration must preserve the legacy `generate_outputs` call
  exactly. The approved route is chunk-level multi-instance partitioning:
  eight mutually exclusive partitions, with every partition using its own
  eight-rank legacy Qwen process. No image/question batching, prompt, decoder,
  crop, weight, or condition conversion changes.
- On each of two homogeneous eight-H200 instances, four partition workers may
  run concurrently because the measured legacy footprint is about 19.8 GiB
  per GPU and four copies remain well within 143.8 GiB. The H200 group runner
  rejects other hardware or insufficient memory. H100 remains supported by
  the original one-worker main route; do not use the four-worker H200 launcher
  on H100.
- Partition workers never publish the final combined CSV and never enter FLUX
  training. After all eight partition-pass markers exist, restart exactly one
  normal v3 main pipeline: it verifies all immutable chunks, publishes the
  combined conditions, builds the prompt cache, and continues FLUX training.
  The old unpartitioned exporter must be stopped before partition workers
  start so it cannot race or redundantly recompute their chunks.
