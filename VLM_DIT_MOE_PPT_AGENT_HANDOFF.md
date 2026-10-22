# VLM-guided DiT-MoE PPT agent handoff

## Read this first

This file is the single entry point for an external agent that will explain,
draft, and repeatedly revise a short research-proposal PowerPoint with the
user. The deck is made outside this repository. Do not run experiments or
invent results while working on the deck.

The advisor's question is:

> Can the VLM's estimated latent, or explicit degradation-type and severity
> labels, be introduced into MoE to directly guide expert merging instead of
> leaving the router to learn everything by itself? What mechanisms can be
> borrowed from LLM MoE?

The proposal is **not experimentally validated yet**. Use `[待实验验证]` on
unverified method advantages. The complete repository evidence audit is
[`reports/foundir_explicit_vlm_dit_moe_novelty_audit_2026-10-19.md`](reports/foundir_explicit_vlm_dit_moe_novelty_audit_2026-10-19.md).

## How the external agent must collaborate with the user

1. Explain each design choice in simple Chinese before revising it in the
   slides. Start with the intuitive answer, then give the technical form only
   if it helps.
2. When the user requests a revision, state briefly what changed and why.
   Preserve previously confirmed decisions unless the user changes them.
3. Keep the deck focused on **how VLM directly guides expert merging**. Do not
   add the project's old failed gates, EBT history, or unrelated experiments.
4. Never manufacture metrics, ablations, figures, implementation status, or
   novelty claims. Clearly separate `已有工作`, `我们的设计`, and `待验证假设`.
5. Do not equate a visually attractive architecture diagram with scientific
   novelty. Explain which matched experiment would prove each claim.
6. Use simple language. For example, explain the proposed constraint as:
   “VLM先判断图像有什么问题、问题多严重；MoE可以修正这个判断，但不能完全无视它。”

## The method in one sentence

The VLM explicitly predicts the type, confidence, and severity of each atomic
degradation; those predictions become an interpretable prior over compatible
DiT LoRA experts; image/VLM latent features may make only bounded corrections,
and the final mixture may vary by DiT block and restoration timestep.

In Chinese:

> VLM先明确判断有哪些退化、各有多严重，再把这个判断直接变成DiT中各个LoRA expert的基础权重；图像特征只能有限修正这些权重，而不能重新自由路由。

## Proposed mechanism

### 1. Atomic diagnosis

For each degradation atom `i` (Blur, Noise, Rain, Haze, Low-light), the VLM
emits:

- presence/type probability `p_i`;
- severity `s_i`;
- confidence or uncertainty.

The initial expert prior can be written as:

\[
\pi_i = \operatorname{Normalize}(p_i s_i).
\]

This means that a confident, severe degradation receives more restoration
capacity. An atom declared absent should normally have zero support.

### 2. Shared base plus atomic LoRA experts

- Use one shared DiT/FLUX backbone rather than copying a full model per task.
- Attach compatible LoRA/adapter experts for Blur, Noise, Rain, Haze, and
  Low-light.
- The frozen/shared base path already provides shared image processing. A
  separate `Preservation Expert` is an optional hypothesis, not a frozen
  requirement; do not present it as already implemented.
- Compound degradation activates several atomic experts instead of requiring
  one new expert for every possible combination.

### 3. Constrained correction rather than a free router

A low-capacity router may correct the VLM prior, but only within a bounded
range:

\[
w_i^{(l,t)} = \operatorname{Softmax}\left(
\log(\pi_i+\epsilon)+c\tanh(\delta_i^{(l,t)})
\right).
\]

Explain the symbols simply:

- `π_i`: the VLM's initial opinion;
- `δ_i`: a learned correction from image/VLM features;
- `c`: a limit on how far the router may deviate;
- `l`: DiT block depth;
- `t`: restoration/diffusion timestep.

The exact formula remains a proposal. The important operational distinction is
that the learned router cannot silently replace the explicit diagnosis with an
unrelated expert plan.

### 4. Block- and time-dependent merging

The expert mixture may change across DiT blocks and restoration timesteps.
This is meant to test whether different restoration operations should receive
different emphasis at different processing stages. Do not claim this helps
until matched experiments demonstrate it.

## LLM/MoE papers that inspired the design

These papers provide **mechanism inspiration**, not the restoration novelty.
Use their exact titles and explain the transfer honestly.

| Paper | Mechanism borrowed | How it informs this proposal |
|---|---|---|
| [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) | Sparse routing and load balancing | Activate only a small number of relevant restoration experts and avoid route collapse. |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | Top-k experts with weighted output combination | A compound degradation may use several experts instead of being forced into one class. |
| [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) | Shared experts plus finer-grained routed experts | Motivates separating shared processing from degradation-specific expert capacity. |
| [Soft Merging of Experts with Adaptive Routing (SMEAR)](https://arxiv.org/abs/2306.03745) | Differentiable soft parameter merging | Motivates continuous merging instead of a brittle hard choice between restoration experts. |
| [X-LoRA: Mixture of Low-Rank Adapter Experts](https://arxiv.org/abs/2402.07148) | Dynamic mixtures of LoRA experts | Supports using multiple compatible low-rank experts on one shared backbone. |
| [LoRA-Flow: Dynamic LoRA Fusion for Large Language Models in Generative Tasks](https://arxiv.org/abs/2402.11455) | Layer/token-dependent LoRA fusion | Inspires allowing expert weights to vary across DiT blocks/stages. |

Do not say that these papers use explicit VLM degradation diagnosis for image
restoration. They do not establish our proposed result.

## Closest restoration work and the novelty boundary

The broad idea `VLM + MoE + restoration` is already occupied. The slides must
acknowledge the following closest work rather than hide it:

- **LoRA-IR**: CLIP degradation probabilities perform top-k selection and
  weighted merging of restoration LoRA experts, including unseen mixed tests.
- **M2IR**: DA-CLIP image/text degradation priors guide spatial routing over
  specialized and shared restoration experts.
- **VLMoE-Restore**: its accessible abstract says a VLM parses degradation
  type, degree, and cause to guide MoE gating for composite infrared
  restoration. Full-paper details were not accessible in the repository audit.
- **FoundIR-v2**: uses timestep-dependent soft MoE driven by LQ/noisy latent
  features; its VLM captions condition the model separately rather than acting
  as an explicit type/severity constraint on the gate.
- **MiM-DiT**: already combines hierarchical MoE and a pretrained DiT for
  all-in-one restoration.
- **SLER-IR**: already studies layer-wise compositional expert paths.
- **UARE**: already performs explicit analyze-then-restore vision-language
  restoration, although not diagnosis-constrained atomic LoRA routing.

Therefore, never claim:

- first MoE restoration;
- first MoE + DiT restoration;
- first VLM-guided restoration;
- first VLM-guided MoE restoration;
- first degradation classifier that merges LoRA experts;
- first expert composition for unseen mixed degradation;
- first layer-wise degradation-aware routing.

The narrow candidate contribution is:

> Compile explicit atomic multi-label degradation diagnosis and per-atom
> severity into a constrained, interpretable, block- and timestep-dependent
> schedule over DiT LoRA experts, while permitting only bounded learned
> corrections inside the VLM-declared support.

This is still a **candidate**, not guaranteed paper-level novelty. Reviewers
may describe it as a combination of LoRA-IR, SLER-IR, FoundIR-v2, and dynamic
LoRA fusion. Its value must be established by matched compositional evidence.

## Minimal evidence needed later

All comparisons must use the same backbone, experts, data, training budget,
and inference budget:

1. image-feature-only learned router;
2. generic frozen VLM embedding concatenated to the router;
3. explicit type-only static top-k prior;
4. explicit type + severity static merge;
5. proposed constrained type/severity block-time schedule;
6. oracle diagnosis as an upper-bound diagnostic only.

Report singleton, double, and triple restoration separately. The decisive
question is not only whether the VLM classifies degradation correctly, but
whether its explicit diagnosis improves restoration and expert composition
over the matched image-only and generic-feature routers.

Go/no-go rule:

> If the constrained VLM merge does not consistently outperform image-only
> MoE and simple VLM-feature conditioning under matched conditions, the core
> hypothesis is unsupported and should not be scaled up.

## Compact deck specification

Keep the PPT to **six content slides plus an optional title slide**:

1. **Advisor's question and direct answer**  
   Why a fully learned MoE router is insufficient; one-sentence proposal.
2. **Overall pipeline**  
   Image → VLM type/confidence/severity → expert prior → bounded correction →
   block/time-dependent LoRA merge → restored image.
3. **How merging works**  
   Shared DiT base, atomic LoRA experts, compound activation, simple equations.
4. **What is borrowed from LLM MoE**  
   Switch/Mixtral, DeepSeekMoE, SMEAR, X-LoRA/LoRA-Flow and the exact mapping
   above.
5. **What is different from existing restoration MoE**  
   Contrast image-only routing, generic VLM feature concatenation, static VLM
   merge, and the proposed constrained block/time schedule. State overlap and
   forbidden broad claims honestly.
6. **How to test it and what the advisor must decide**  
   Matched controls, go/no-go rule, and the four questions below.

Questions for the advisor:

1. Is explicit VLM-constrained expert merging a worthwhile hypothesis?
2. Should the prior use labels only, latent only, or the proposed hybrid?
3. Is shared base + atomic LoRA experts the right lean implementation?
4. Is the narrow block/time-dependent constrained schedule sufficiently
   different to justify a matched pilot?

Do not include unrelated historical experiments or unverified performance
predictions in this deck.

## Copy-paste prompt for a PPT-generation agent

```text
请制作一份适合向导师确认研究方案的中文研究型PowerPoint，题目为：
《显式VLM引导的DiT-MoE图像复原》
英文副标题：VLM-Guided Constrained Expert Merge for DiT Restoration。

你的任务不只是排版，还要作为研究讲解者与我持续迭代。每次修改前，先用简单中文解释相关设计；修改后说明改了什么、为什么。已经确认的决定不要自行推翻。不得虚构实验结果，所有尚未验证的优势标记“[待实验验证]”。

导师的问题是：能否把VLM估计的latent，或者明确的退化类型和强度标签，引入MoE，直接指导多个experts的merge，而不是让router完全自己学习？LLM MoE中有什么机制可以借鉴？

核心回答是：VLM明确预测Blur、Noise、Rain、Haze和Low-light的存在概率、置信度与强度，并将其转换为可解释的expert prior。图像/VLM latent只允许在这个prior附近进行有界修正。最终权重可以随DiT block和恢复timestep改变，用来融合共享DiT backbone上的原子LoRA experts。

用直白的话解释为：“VLM先判断图像有什么问题、问题多严重；MoE可以修正这个判断，但不能完全无视它。”

请制作六页正文，另加一页可选标题页：

1. 导师问题与一句话回答：说明普通image-only MoE完全依赖隐式router，而我们让VLM诊断直接决定expert基础权重。
2. 整体流程：退化图像→VLM类型/置信度/强度→expert prior→有界latent修正→按DiT层和恢复阶段动态融合LoRA experts→恢复图像。
3. Merge机制：一个共享DiT backbone；Blur、Noise、Rain、Haze、Low-light原子LoRA experts；复合退化同时激活多个expert，不为每种组合复制完整模型。Preservation Expert只能作为可选假设，不要写成已经实现。
4. LLM MoE启发：准确列出Switch Transformers、Mixtral of Experts、DeepSeekMoE、SMEAR、X-LoRA和LoRA-Flow。分别对应稀疏Top-k、共享与路由expert、软参数融合和分层动态LoRA融合。明确说明这些论文提供结构启发，但没有解决VLM-guided all-in-one image restoration。
5. 与现有工作的边界：必须提到LoRA-IR、M2IR、VLMoE-Restore、FoundIR-v2、MiM-DiT、SLER-IR和UARE。宽泛的“VLM+MoE+DiT”不是首创。候选贡献只能表述为：把显式原子多标签诊断和逐原子强度编译成受约束、可解释、随DiT层和timestep变化的LoRA expert schedule，并只允许router在VLM声明的支持范围内做有界修正。[待文献审查与实验验证]
6. 最小验证和导师确认：同backbone、experts、数据、训练量和推理量，对比image-only router、普通VLM embedding、type-only static routing、type+severity static merge和proposed constrained block/time schedule。若proposed不能稳定优于image-only MoE与简单VLM conditioning，就停止扩大。最后询问导师是否认可该窄假设、hybrid输入、原子LoRA设计及matched pilot。

可以用两个简化公式，但不要复杂推导：
π_i = Normalize(p_i × s_i)
w_i^(l,t) = Softmax(log(π_i+ε) + c·tanh(δ_i^(l,t)))

解释：π是VLM基础判断，δ是学习到的有限修正，c限制修正幅度，l表示DiT层，t表示恢复阶段。

禁止宣称：首个MoE图像复原、首个MoE+DiT、首个VLM-guided restoration、首个VLM-guided MoE restoration、首个LoRA expert merge、首个未见复合退化expert组合或首个layer-wise routing。

风格要求：简洁、现代、研究型；每页一个核心观点；正文尽量不超过五个要点；图示优先；不用旧实验失败历史，不写未经验证的性能数字。输出可编辑PPTX。
```

## Short teaching glossary

- **MoE**：把模型的一部分拆成多个“专长模块”，每张图只调用其中一部分。
- **Expert merge**：不是简单按顺序跑多个模型，而是在同一个共享网络中按权重组合多个expert的增量。
- **LoRA expert**：挂在共享大模型上的小型低秩参数模块，复制成本远低于复制完整模型。
- **Image-only router**：只看图像内部特征并自行学习选expert，选择过程通常不容易解释。
- **VLM prior**：VLM先给出的退化判断与基础expert比例。
- **Bounded correction**：允许模型纠正VLM，但限制纠正幅度，避免完全绕开显式诊断。
- **Atomic degradation**：Blur、Noise等最基本退化；复合退化由多个原子共同描述。
- **Block/time schedule**：不同DiT层和不同恢复阶段可以采用不同expert比例。
- **Matched comparison**：除被比较的路由机制外，其余backbone、数据、训练量和推理量全部相同。

## Source status

- The closest-restoration overlap claims and repository implementation facts
  come from the linked 2026-10-19 audit.
- The six LLM/MoE titles and arXiv identifiers above were checked against the
  arXiv records on 2026-10-22.
- No proposed VLM-constrained DiT-MoE experiment is reported in this handoff.

