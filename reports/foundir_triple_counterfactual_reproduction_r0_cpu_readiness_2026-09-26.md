# FoundIR triple counterfactual reproduction R0 CPU readiness

Date: 2026-09-26 UTC
Status: CPU protocol ready; neural implementation and GPU launch not authorized
Scientific role: DIAGNOSTIC ONLY / 非主线

## Outcome

The data/protocol layer for a bounded OneRestore-style reproduction and future counterfactual-subset ablation is complete and independently re-checkable. No restoration model was implemented or trained, no GPU ran, and neither the official FoundIR test nor the 60-image FoundIR triple endpoint was accessed.

The package freezes 130 source-disjoint synthetic parents from the existing hash-bound AgenticIR/Qwen physical-v8b manifest:

| Role | Singleton | Double | Triple | Use |
|---|---:|---:|---:|---|
| Train | 20 | 40 | 0 | Gradients allowed |
| Development | 10 | 20 | 0 | Checkpoint selection allowed |
| Held-out triple | 0 | 0 | 40 | Gradients and selection forbidden |

The 40 held-out records cover all ten three-of-five combinations. The builder expands them and the singleton/double parents into 620 ordered-subset nodes and 1,710 counterfactual input/remove/target relationships.

## Frozen artifacts

- Config: `configs/foundir_triple_counterfactual_reproduction_r0_v1.json`
  SHA-256: `f099113d41867bf42f4143c70ebc7cbbc31d393368d81c7cc0fea3f9ca1b0cd8`
- Builder/auditor: `smoke/build_foundir_triple_counterfactual_reproduction_r0_v1.py`
  SHA-256: `1990abd2d7209e47dfdccea4e5677b6ca154852563722a09ac45f26700cda5c1`
- Focused tests: `tests/test_foundir_triple_counterfactual_reproduction_r0.py`
  SHA-256: `acab8897cd8fbbd54a2b6fa1c0ffcd61ba2db4e297b45569741ce759b7c59abb`
- Manifest: `manifests/foundir_triple_counterfactual_reproduction_r0_v1.jsonl`
  SHA-256: `29fafafec8929d57136068433f081c8998053df0ec638b8f654a5f69e0c4ac85`
- Metadata/audit: `manifests/foundir_triple_counterfactual_reproduction_r0_v1.json`
  SHA-256: `82302da80acde9de29a8964f4125c69b8f893686db81bc31bd5d140afa97f948`
- Parent manifest: `manifests/qwen_multidegradation_100k_physical_v8b.jsonl`
  SHA-256: `777d0130bb0ff9590ad967e8bc848b13399de71df58bf72739f95be62720fb02`

## Public reproduction binding

The config pins OneRestore remote HEAD `23a2e6af066864dca013aeed7f6c6f5958aa7259` and the exact SHA-256 of its descriptor, restorer, loss, dataset, and multi-GPU training files. This is a source binding only; no third-party code was copied into the workspace and no claim of exact CDD-11 reproduction is made yet.

## Audit invariants

The builder and independent `--check` enforce:

- source identity is disjoint across train, development, and held-out roles;
- every source file matches its frozen SHA-256;
- train and development parents have composition size at most two;
- every triple is fail-closed for gradients and checkpoint selection;
- each subset preserves parent operator order;
- every subset reuses the exact parent severity, operator parameters, seeds, crop, and nuisance;
- all expected subset nodes exist;
- every counterfactual pair obeys `target = input minus requested components` in parent order;
- no target is inferred by pixel subtraction and no operator is reordered;
- all ten triple combinations are represented;
- `official_foundir_test_accessed=false`;
- `foundir_60_image_triple_accessed=false`.

## Verification

- Python compilation: pass.
- Six focused tests: pass.
- Full build against all 120,000 parent-manifest rows: pass.
- Independent second `--check`: pass with identical audit and manifest hash.
- Triple gradient leakage: zero.
- Triple checkpoint-selection leakage: zero.

## Implementation defects caught and fixed

1. Lattice nodes initially shared in-memory operator dictionaries with their parent. A node mutation could therefore mutate the comparison reference and evade validation before serialization. All nested operator/crop/nuisance objects are now deep-copied.
2. Combination names initially used alphabetical sorting, inconsistent with the project's canonical order `blur, noise, rain, haze, low_light`. A fixed canonical rank now prevents one composition from acquiring multiple string identities.

Neither fix changes the scientific protocol.

## GPU readiness and timing

This is **not GPU-ready**. The diagnostic neural architecture, loss weights, optimizer budget, checkpoint-selection rule, and launcher do not yet exist. Consequently there is no safe GPU command and no defensible end-to-end runtime estimate. The first estimate must be produced after the architecture is authorized, implemented, and a no-update throughput benchmark is available.

Current work consumed normal CPU wall time only and exactly `0 GPU-hours`.

Before any future GPU command, the project must state separately:

- total normal wall time including rendering, loading, training, all evaluations, aggregation, and reporting;
- assumed GPU/node concurrency;
- aggregate GPU-hours;
- the dominant uncertainty.

## Governance gate and exact next task

An actual OneRestore-style backbone or new conditioning/loss path is a model-family/architecture experiment. Per project governance it requires a design hold and an explicit user/advisor decision before neural implementation or GPU execution.

After authorization, the next bounded implementation is R0 only:

1. implement the fixed categorical descriptor and CDRL reproduction arm;
2. render the frozen train/development nodes without touching held-out triples for selection;
3. add no-update forward/backward and DDP tests;
4. benchmark throughput and publish total wall-time/GPU-hour estimates;
5. run the smallest R0 gate;
6. stop at result; do not automatically proceed to continuous prompt, spatial factorization, or CSR.
