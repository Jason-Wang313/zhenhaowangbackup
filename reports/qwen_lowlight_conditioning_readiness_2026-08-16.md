# Qwen-VL Lowlight conditioning readiness

Date: 2026-08-16 UTC

## Scope

The real vision-capable Qwen-VL checkpoint and the bounded 25-image internal
Lowlight understanding smoke are frozen and CPU-verified. This milestone
authorizes only the offline H100 smoke command recorded below. It does not
claim a Qwen restoration improvement, an official FoundIR result, or Lowlight
Phase 1 completion. No official FoundIR image has been used for Qwen prompting,
tuning, prediction, or restoration.

## Model decision

The selected model is `Qwen/Qwen2.5-VL-7B-Instruct` at immutable revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`.

- Architecture: `Qwen2_5_VLForConditionalGeneration`
- Precision: BF16
- Attention implementation: PyTorch SDPA
- Frozen model parameters: `8,292,166,656`
- SafeTensors: `729` tensors in five shards
- Repository payload: `16,595,981,281` bytes across 16 files
- Weight payload: `16,584,414,560` bytes
- Local path: `models/qwen2.5-vl-7b-instruct`
- License: Apache-2.0, declared in the frozen model-card metadata

The repository has no separate `LICENSE` file. The license claim is therefore
limited to the `license: apache-2.0` field in the model card at the frozen
revision. Exact expected size and SHA-256 evidence for all 16 files is embedded
in `configs/qwen_lowlight_smoke25.json` and was verified by the preflight.

`Qwen2.5-VL-3B-Instruct` was rejected because its Qwen Research License is not
suitable for this commercial project. `Qwen3-VL-4B-Instruct` is Apache-2.0 but
is not supported by the frozen Transformers 4.56.1 environment. The selected
7B checkpoint is Apache-2.0, supported by that environment, and is expected to
fit the available H100 80 GB.

## Frozen understanding interface

Qwen-VL remains frozen and receives only the degraded image plus the frozen
Lowlight severity rubric. Decoding is deterministic with `do_sample=false` and
at most 64 new tokens. Captions, object/content descriptions, aesthetic
judgments, color-style descriptions, and restoration instructions are
explicitly prohibited.

The only accepted output shape is:

```json
{"degradations":["low_light"],"levels":{"low_light":1}}
```

The level must be an integer from 1 through 5. Markdown, extra text, extra
fields, other degradation types, strings, booleans, and out-of-range values
are rejected. Invalid outputs are not repaired; the frozen downstream fallback
is level 3.

Prompt SHA-256 values:

- system prompt:
  `1126ca589db1ca67a3b9188e3c36aa17e3a89a58f0e152774606f42e54393242`
- user prompt:
  `66af2fe88dd2c2133a01d27268e012ea7859ffcda5c90c824f2896e17797be04`

The processor is frozen to its slow implementation with visual budgets of
256–512 tokens. Preflight realized exactly 486 visual tokens for each of five
balanced inspection images.

## Frozen smoke set and evidence

The smoke deterministically selects 25 records from the source-disjoint
internal-test split of `manifests/low_light_pilot_10000.csv`: exactly five
records per L1-L5 level. It records raw model output, strict parse result,
fallback use, true and predicted levels, degraded-image hash, config hash, and
model revision in fail-closed per-image JSON files.

Reported understanding metrics are valid structured-output rate, degradation
type accuracy and macro-F1, severity accuracy, exact match, severity MAE,
per-level metrics, and the severity confusion matrix. These are internal Qwen
understanding metrics; they are not FoundIR restoration metrics.

## Condition adapter

The first frozen injection method is one learned FLUX text-context token chosen
by the predicted severity. The adapter has five 7,680-dimensional embeddings,
for exactly `38,400` trainable parameters. Qwen-VL and the FLUX base remain
frozen. Passing `condition=None` returns the original context and position-ID
tensors unchanged by object identity, preserving the exact baseline path.

The adapter is defined and interface-tested here, but it is not trained by the
25-image understanding smoke. A fair baseline/Qwen/oracle restoration
comparison remains subsequent work after the Qwen prediction smoke is valid.

## Frozen configuration and verification

- Config: `configs/qwen_lowlight_smoke25.json`
- Config SHA-256:
  `20d1dfdc1bb25611508a06d8c241146ebad246b2404c44c68831dcf19d49b6b1`
- Output: `checkpoints/qwen_lowlight_smoke25/`
- Log: `logs/qwen_lowlight_smoke25.log`

CPU-side verification passed:

- all 16 model files: exact byte size and SHA-256
- SafeTensors index: exact shard, key, and tensor coverage
- model shape: 729 tensors and 8,292,166,656 parameters
- Transformers architecture: exact match
- slow processor and vision inputs: pass
- balanced five-image processor inspection: levels L1-L5, 486 tokens each
- full repository unit suite: 25 tests pass
- focused Qwen suite: 6 tests pass
- Python compilation: pass
- launcher shell syntax: pass
- `git diff --check`: pass

The validated CPU environment used Python 3.12.13, PyTorch 2.7.0+cu128,
Transformers 4.56.1, SafeTensors 0.4.5, and Pillow 11.1.0. CUDA is not exposed
to the coding-agent terminal, so no Qwen inference result is claimed here.

The original concurrent download ended after four verified shards because of
a transient Hugging Face Xet token-refresh network error. The remaining shard
was resumed at the same revision with Xet high-performance mode, then the
complete five-shard payload passed the independent hash and structure
preflight.

## Authorized H100 command

Run exactly:

```bash
CUDA_VISIBLE_DEVICES=0 smoke/run_qwen_lowlight_smoke25.sh
```

Afterward, inspect `checkpoints/qwen_lowlight_smoke25/metrics.json` and the
log. Do not prepare the full 1,000-image Qwen prediction or any official
FoundIR restoration until the smoke outputs are structurally valid and the
resulting system definition is frozen.
