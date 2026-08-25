# Qwen multi-degradation validation correction 3

## Architectural diagnosis

Across the initial trainable-merger run and two full-mixture corrections,
valid JSON and type F1 passed while triple recall and independent severity
plateaued. The prior adapter places all LoRA modules in the visual encoder and
keeps the language decoder completely adapter-free. This forces visual LoRA
to warp mixed-degradation evidence into exactly the representation expected
by a frozen generic decoder, even though the difficult part is converting
interleaved image tokens into a project-specific multi-label JSON decision.

The advisor required LoRA and requested empirical comparison of projection
trainability. An attention-only decoder LoRA is inside that approved LoRA
design: every 7B base weight and the tokenizer remain frozen. It is not full
decoder fine-tuning.

## Bounded correction

Correction 3 initializes the hash-pinned correction-2 visual LoRA and merger,
then adds rank-8, alpha-16 LoRA to q/k/v/o projections in all 28 decoder
self-attention layers. This is 112 modules and 5,046,272 new parameters. The
complete trainable adapter has 272 LoRA modules and 60,772,096 parameters,
including the unchanged 44,574,464-parameter merger.

Only one shuffled pass over the same 9,600-image training split is allowed:
300 optimizer steps, global batch 32. Existing visual LoRA uses LR 5e-6, the
merger uses 5e-7, and new decoder LoRA uses 2e-5. The semantic assistant-token
CE from correction 2 remains unchanged at weight 4 for type/cardinality/level
decision tokens. Prompt, schema, data, labels, generation, and all gates stay
unchanged.

Correction-2 validation and adapter artifacts are hash-pinned. The original,
correction-1, correction-2, and correction-3 fixed-test artifacts must remain
absent until every validation gate passes. If validation fails, the runner
stops; if it passes, the existing exactly-once test and real-sanity pipeline
continues automatically.

CPU evidence before launch: full hash/meta-architecture preflight passed, 46
relevant tests passed, and all four possible fixed-test artifacts were absent.
Frozen config SHA-256:
`cbb483bbbc5c8d6d5ae57088ca61ee298dd836ca0a0b9c2504e8892bc4a2b066`.

## Correction-3 outcome

Correction 3 completed all 300 steps and stopped before fixed-test access.
Overlap-only severity accuracy improved from 0.7926 to 0.7970, but set exact
fell from 0.7958 to 0.7900 and macro-F1 fell from 0.9314 to 0.9282. Triple
set exact rose only from 0.5750 to 0.5806. Attention-only decoder LoRA does
not remove the variable-length output bottleneck and will not be continued on
that schema.
