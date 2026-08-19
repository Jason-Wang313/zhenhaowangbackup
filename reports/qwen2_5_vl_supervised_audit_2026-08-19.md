# Qwen2.5-VL supervised-path audit — 2026-08-19

## Status

**CPU/offline architecture and artifact audit: PASS.** The local model,
processor, manifest, installed Transformers implementation, and intended
gradient scope are sufficient to run the advisor-directed tiny supervised-VLM
gate. The H200 load/gradient/memorization gate has **not** run in the coding
terminal because CUDA is not exposed here.

This is an implementation audit, not a scientific VLM result, a real-domain
validation result, a diffusion result, or a FoundIR result.

## Authority and repository state

- Workspace: `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`
- Branch audited: `main`
- Local and `origin/main` base at audit: `8d81b30a13741abc493857b15d6ed13c7c2811e1`
- Advisor diagram SHA-256:
  `02e32805142e64a7c191469fa4fbe4d4169914b25e3fa9df75853d4cde266194`
- Model repository: `Qwen/Qwen2.5-VL-7B-Instruct`
- Frozen model revision:
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Local model path: `models/qwen2.5-vl-7b-instruct`
- Model license recorded by the frozen inventory: Apache-2.0
- Model inventory config SHA-256:
  `20d1dfdc1bb25611508a06d8c241146ebad246b2404c44c68831dcf19d49b6b1`

The full preflight re-hashed all 16 model/tokenizer/processor files and checked
all 729 SafeTensors keys against the index. No second model copy or download
was needed.

## Installed implementation

- Workspace interpreter: `.venv/flux2-smoke/bin/python`
- Python: `3.12.13`
- PyTorch: `2.7.0+cu128`
- Transformers: `4.56.1`
- Transformers model source:
  `.venv/flux2-smoke/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Conditional model class: `Qwen2_5_VLForConditionalGeneration`
- Processor: `Qwen2_5_VLProcessor`
- Frozen slow image processor: `Qwen2VLImageProcessor`
- Tokenizer: `Qwen2Tokenizer`
- Checkpoint/model dtype: BF16
- Attention implementation: SDPA; the deterministic smoke selects the math
  SDPA backend

The coding terminal has 128 logical Intel Xeon Gold 6530 CPUs and about 1.5 TiB
system RAM. It has no `nvidia-smi`, and PyTorch reports CUDA unavailable and
zero visible devices. Therefore, live H200 device placement, BF16 execution,
allocated memory, and gradients remain GPU-gate evidence; the trainer records
and checks them at runtime.

## Chat and image format

The local processor uses Qwen's chat template with system, user, and assistant
turn delimiters. The supervised input is:

1. a fixed system instruction requiring one compact JSON object;
2. one image placeholder plus a fixed Lowlight assessment question; and
3. one deterministic assistant target.

The slow image processor converts to RGB, rescales and normalizes with the
checkpoint's CLIP-like mean/std, uses 14-pixel spatial patches, temporal patch
size 2, and spatial merge size 2. The smoke freezes both `min_pixels` and
`max_pixels` at 100,352. The five audited representative images each realized
117 merged visual tokens.

Only assistant answer tokens receive cross-entropy loss. Preflight verified an
exact input-prefix boundary after multimodal image-token expansion. It realized
12 supervised tokens for clean L1 and 17 for each active L2-L5 target.

## Authoritative label serialization

Project L1 is a clean negative, not an active `low_light L1` operator:

```json
{"degradations":[],"levels":{}}
```

L2-L5 contain the active type and independent level, for example:

```json
{"degradations":["low_light"],"levels":{"low_light":3}}
```

The parser rejects `low_light L1`, extra keys, wrong degradation types,
non-integer severities, markdown, and trailing non-JSON text.

## Audited module and parameter scope

Transformers 4.56.1 does not expose a separate `mm_projector` for this model.
The multimodal connector is the patch merger inside the visual path:
`model.visual.merger`. A meta-device construction of the installed class and a
separate scan of every checkpoint tensor agreed exactly:

| Scope | Runtime module | Parameters | Training state |
|---|---|---:|---|
| Patch embedding | `model.visual.patch_embed` | 1,505,280 | trainable |
| 32 vision blocks | `model.visual.blocks` | 630,470,400 | trainable |
| Multimodal merger/connector | `model.visual.merger` | 44,574,464 | trainable |
| Language decoder | `model.language_model` | 7,070,619,136 | frozen |
| Vocabulary output projection | `lm_head` | 544,997,376 | frozen |
| **Total** |  | **8,292,166,656** |  |

The complete trainable visual path has 676,550,144 parameters (8.159% of the
model). The tokenizer has no trainable parameters. Cross-entropy gradients flow
through the frozen language decoder into visual embeddings, while decoder and
`lm_head` parameters remain `requires_grad=False` and are excluded from the
optimizer.

## Implemented runtime proofs

The H200 trainer fails closed unless all of the following are true:

- exactly one CUDA device is visible and it has at least 75 GiB memory;
- all model parameters load on that device in BF16;
- the optimizer parameter identity set exactly equals `model.visual`;
- every trainable visual parameter receives a finite gradient on the first
  backward pass;
- patch embedding, vision blocks, and merger each have nonzero gradient norm;
- no frozen decoder/output parameter has a gradient;
- frozen parameter version counters and exact sentinel values are unchanged
  both before the intentional restart and after the resumed run;
- visual weights and optimizer/RNG state are hashed and published together in
  new checkpoint directories;
- a fresh process reloads the step-2 checkpoint before continuing;
- deterministic probe loss decreases; and
- greedy generation memorizes all 40 strict structured labels, including every
  L1-L5 subgroup.

The trainer saves only the trained visual path plus optimizer/RNG state. It
records the frozen base-model revision and inventory rather than duplicating
the 7.6-billion frozen decoder/output parameters.

## Verification completed

- Full model-file SHA-256 validation: PASS
- SafeTensors index/key/shape validation: PASS (`729` tensors)
- Loaded-class meta parameter audit: PASS (`8,292,166,656` parameters; zero
  unexpected parameters)
- Selected-image SHA-256 validation: PASS (`40/40`)
- Slow processor and teacher-label boundary preflight: PASS
- Python compilation: PASS
- Shell syntax: PASS
- Black formatting check: PASS
- Repository whitespace check: PASS
- Complete unit suite: PASS (`62` tests)

The known `pkg_resources is deprecated` warning appeared without a traceback
and is not treated as a failure, consistent with the repository instructions.
