# Qwen multi-degradation validation correction 4

## Structural diagnosis

Three validation-driven corrections show that capacity and JSON fluency are
not the limiting factors. The variable schema requires Qwen to choose a list
length, emit every active type, repeat those types as level-object keys, and
avoid illegal cardinalities or levels. Most triple failures close the list
after two types. Visual reweighting and decoder LoRA cannot reliably repair
that sequence design.

The advisor-approved scientific objective is five independent type/strength
decisions, not a variable-length list. A fixed slot per type is a more direct
representation of that objective.

## Final structural correction

Correction 4 uses this compact Qwen target:

`{"blur":1,"noise":1,"rain":1,"haze":1,"low_light":1}`

Each value is independently L1 absent, L2 mild, L3 moderate, L4 strong, or L5
extreme. Active degradations are deterministically the slots above L1. The
resulting internal label is still translated through the existing canonical
restoration text before FLUX conditioning.

The target always has 23 tokenizer tokens; only five level digits vary. This
eliminates early list termination, duplicated type names, missing level keys,
four-type free-form outputs, and out-of-range L6 generation. Strict key order,
integer range, canonical serialization, and the maximum-three synthetic
contract remain validated.

Training starts from correction 2's best hash-pinned visual adapter and a
fresh optimizer/new rank-8 decoder attention LoRA. It uses two independently
shuffled passes of all 9,600 training images (600 optimizer steps, global batch
32). The five L1--L5 digit tokens receive assistant-CE weight 8; syntax remains
supervised at weight 1. All Qwen base weights and tokenizer remain frozen.

The same validation and fixed-test thresholds apply. The runner stops before
test access on any validation failure. If every validation gate passes, it
automatically performs the exactly-once fixed test, real-sanity inference, and
contact-sheet rendering.

CPU evidence before launch: full hash/meta-architecture preflight passed, all
12,000 generated images were verified, 51 relevant tests passed, and every
prior fixed-test artifact remained absent. Frozen config SHA-256:
`b954bd3822b1250ee0571bec91d1a4c7d1e4503bb72017e952db28ebfc82768a`.
