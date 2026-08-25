# Qwen multi-degradation validation correction 2

## Correction-1 outcome

Correction 1 completed 600 optimizer steps over two shuffled passes of the
complete 9,600-image training mixture.  It stopped before fixed-test access
because the unchanged 1,200-image validation gate did not pass.

| Metric | Initial trainable merger | Correction 1 | Frozen gate |
|---|---:|---:|---:|
| Valid structured output | 0.9883 | 0.9950 | >= 0.9900 |
| Type macro-F1 | 0.9159 | 0.9285 | >= 0.9000 |
| Type micro-F1 | not reported here | 0.9289 | >= 0.9200 |
| Set exact | 0.7617 | 0.7842 | >= 0.8000 |
| Combined exact | 0.4983 | 0.6258 | >= 0.6500 |
| Typed severity accuracy | 0.6005 | 0.7074 | >= 0.7500 |
| Typed severity MAE | 0.7167 | 0.5398 | <= 0.5000 |
| Triple set exact | 0.4833 | 0.5500 | >= 0.6500 |

Correction 1 materially improved every failing headline metric but did not
cross five frozen gates.  The fixed synthetic test remains untouched.

## Failure audit

Precision is already 0.9651 while recall is 0.8954.  Of 360 validation
triples, 138 were decoded as two degradations and only 208 were decoded as
three.  Triple false negatives were blur 45, haze 39, low-light 42, noise 11,
and rain 47.  Only 12 triple false positives occurred.  The dominant failure
is therefore conservative early termination of the degradation list, not
uncontrolled overprediction or insufficient trainable capacity.

The average assistant-token CE is small because compact JSON contains many
predictable syntax tokens.  The remaining type-presence and L2--L5 decisions
occupy a minority of target positions.  Continuing the same uniformly
weighted objective would spend most supervision on already-solved syntax.

## Bounded semantic-loss correction

Correction 2 preserves the advisor-approved task, prompt, JSON schema,
synthetic corpus, source-disjoint splits, Qwen revision, visual LoRA scope,
trainable merger, frozen language decoder/tokenizer, global batch 32, greedy
deployment decoder, and every validation/fixed-test gate.

It starts from correction 1's hash-pinned adapter with a fresh optimizer and
runs one shuffled pass over all 9,600 training records (300 optimizer steps).
Assistant-token CE remains the only objective.  JSON syntax retains weight
1, while frozen tokenizer IDs corresponding to clean/active array decisions,
degradation names, list/object continuation or termination, and severity
digits L2--L5 receive weight 4.  Loss is computed only at unmasked assistant
positions, which is mathematically equivalent to ignoring prompt, image, and
padding labels and avoids unnecessary vocabulary CE work.

The LoRA learning rate is 2e-5, merger learning rate is 2e-6, warmup is 10
steps, and cosine decay ends at 25% of each base rate.  No validation examples
are replayed in training and no data are added or relabeled.

The pipeline remains fail closed:

1. full hash and runtime preflight;
2. correction-2 training;
3. unchanged validation gates;
4. only if every validation gate passes, checkpoint selection and exactly-once
   fixed synthetic test evaluation;
5. only after fixed-test success, frozen real-image sanity inference and
   contact-sheet rendering.

CPU evidence before launch: actual correction-2 preflight passed, 42 relevant
tests passed, and the original, correction-1, and correction-2 fixed-test
artifacts were all absent.  Frozen config SHA-256:
`50f0d4720664df496eb9fb2a9eabcfd922699b8dc63ada1d678ed4ab3602843f`.
