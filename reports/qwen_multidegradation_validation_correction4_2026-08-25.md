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

The target always has 25 supervised tokenizer tokens; only five level digits vary. This
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

## GPU execution result

Correction 4 ran on two NVIDIA H100 80 GB GPUs from
`2026-08-25T04:14:58Z` through `2026-08-25T04:49:52Z`. All 600 optimizer
steps completed. The training stage took `1910.61` seconds, the final reported
loss was `0.0582643`, and the distributed gradient-scope audit passed on both
ranks: every trainable tensor received a gradient and no frozen tensor did.
The complete runner then exited with the intentional scientific-failure status
`2` after validation.

The frozen 1,200-image validation result is `status="failed"` and
`automatic_gate_pass=false`:

| Frozen validation gate | Required | Observed | Result |
|---|---:|---:|---|
| Valid structured output | >= 0.99 | 0.9967 | pass |
| Type macro-F1 | >= 0.90 | 0.9287 | pass |
| Type micro-F1 | >= 0.92 | 0.9291 | pass |
| Degradation-set exact | >= 0.80 | 0.7875 | fail |
| Type-and-severity combined exact | >= 0.65 | 0.6075 | fail |
| Typed severity accuracy | >= 0.75 | 0.6801 | fail |
| Typed severity MAE with missing penalty | <= 0.50 | 0.5694 | fail |
| Triple-degradation set exact | >= 0.65 | 0.5750 | fail |
| Gradient scope | required | pass | pass |

By composition size, degradation-set exact was `0.9750` for clean,
`0.8917` for single, `0.8333` for double, and `0.5750` for triple examples.
Micro precision was `0.9623`, while recall was only `0.8981`: there were 220
active-type false negatives and 76 false positives across the five slots.
The fixed schema therefore remained conservatively biased toward L1/absent,
especially on compound examples.

## Interpretation and stop decision

The structural hypothesis did not survive the frozen validation. Relative to
correction 2, fixed slots slightly improved structured-output validity
(`0.9942 -> 0.9967`) but regressed degradation-set exact
(`0.7958 -> 0.7875`), combined exact (`0.6333 -> 0.6075`), typed severity
accuracy (`0.7130 -> 0.6801`), typed severity MAE (`0.5296 -> 0.5694`), and
did not improve triple set exact (`0.5750 -> 0.5750`). It is not a near-pass
that can be made credible by relaxing thresholds.

The runner stopped before selection, fixed-test evaluation, real-sanity
inference, or rendering. No Correction-4 fixed-test result or prediction
artifact exists, so the exactly-once synthetic fixed test remains unconsumed.
FoundIR training and official test data were not used.

No fifth same-objective correction is authorized from this evidence. Four
validation-driven variants have now plateaued around the same type-recognition
level, and changing schema, loss weighting, or decoder LoRA did not resolve
compound omissions. Further work requires a genuinely new scientific decision
rather than another validation-tuned continuation: either revise the data/task
supervision to make compound evidence more identifiable, adopt a direct
five-head discriminative readout while preserving the advisor-facing Qwen JSON
interface, or obtain advisor approval for a different VLM optimization design.
The current adapter is preserved as negative evidence, not selected for FLUX.

Artifact identities:

- adapter SHA-256:
  `f5b2fd7d303142e8655c11457f943efb4c3bc01e2205eda8279b61f00b84afcf`;
- validation result SHA-256:
  `8972c5d9d4dd201b123286bd7eff4a9ca16e6ad5d3eb5b7aab635dd826e5029d`;
- validation predictions SHA-256:
  `2b09e832d8fdf0c2f5e13a94f2968600c5c8c89fa4b6c3080329bf88a3dccd80`.
