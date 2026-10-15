# FoundIR lower-order EBT scaled-data v1 readiness — 2026-10-08

## Status

**CPU DATA CONTRACT READY; GPU TRAINING NOT LAUNCHED.**

The scaled lower-order corpus is frozen and independently verified. It is a
matched data contract for future Plain and Compositional EBT arms, not a model
result and not evidence that either arm generalizes to unseen triples.

## Frozen identities

- Config: `configs/foundir_compositional_ebt_scaled_data_v1.json`
  - SHA-256: `b40aefd81a5c294ebd448fd70418e288baeaac6a39300787fb31c9db736a078a`
- Builder/checker: `smoke/build_foundir_compositional_ebt_scaled_data_v1.py`
  - SHA-256: `2939f17bea88cb789f8b38fa0fa71fb299d76d2b624a8c56731c83f6f5588419`
- Parent manifest: `manifests/qwen_multidegradation_100k_physical_v8b.jsonl`
  - SHA-256: `777d0130bb0ff9590ad967e8bc848b13399de71df58bf72739f95be62720fb02`
- Frozen output manifest:
  `manifests/foundir_compositional_ebt_scaled_data_v1.jsonl`
  - SHA-256: `d7add2be0ec4edff95034caee1287b4746acffcec76d43ab3f288503bbf3b449`
- Machine-readable metadata:
  `manifests/foundir_compositional_ebt_scaled_data_v1.json`
  - SHA-256: `44f4546a941fa7e150f51eb6e148b7bfb64d4346cbb8a342d73a82c970e51c7a`

## Verified composition contract

| Split | Singleton records | Double records | Total | Unique sources |
|---|---:|---:|---:|---:|
| Train | 30,000 | 30,000 | 60,000 | 1,600 |
| Validation | 3,000 | 3,000 | 6,000 | 200 |

- Five atomic degradations are present: Blur, Noise, Rain, Haze, and
  Low-light.
- Every singleton has 6,000 train and 600 validation records.
- All `C(5,2) = 10` unordered degradation pairs are present.
- Every pair has 3,000 train and 300 validation records.
- Train/validation source overlap is exactly zero.
- Triple records are exactly zero.
- Parent-manifest test records are exactly zero.
- Every selected degraded PNG passed SHA-256 verification.
- Every unique source image was SHA-256 verified once.
- Targets are deterministically regenerated as the same clean frozen crop and
  non-target nuisance with zero labeled degradation operators applied.

The ten pair labels are:

1. Blur + Noise
2. Blur + Rain
3. Blur + Haze
4. Blur + Low-light
5. Noise + Rain
6. Noise + Haze
7. Noise + Low-light
8. Rain + Haze
9. Rain + Low-light
10. Haze + Low-light

This is broader than the historical FLUX manifest, which contained only five
pair labels. The additional five are Blur + Rain, Blur + Haze, Noise + Rain,
Noise + Haze, and Rain + Low-light. They are necessary for a future claim
about arbitrary three-of-five lower-order coverage. They are not required
merely to cover the three constituent pairs of the specific
Low-light + Blur + Noise target triple, because those three historical pairs
were already present.

## Verification record

The full build performed content-hash verification before writing the frozen
artifacts and returned
`foundir_compositional_ebt_scaled_data_v1_ready`. A separate `--check` pass
then reloaded the frozen manifest, checked config and manifest identities, and
revalidated its structural contract. A separate aggregation confirmed all
per-composition counts and zero train/validation source overlap.

No held-out triple images, FoundIR 60-image triple endpoint, sealed internal
data, or official FoundIR test data were accessed. No GPU training was
launched; cost is `0.0 GPU-hours`.

## Interpretation and next gate

The previous 10k toy experiment used only 60 train parents and a 146,432-
parameter model. Its Plain arm led the Compositional arm by `0.941337 dB` on
the frozen lower-order development set. This scaled data contract fixes the
clearly insufficient data regime but does not erase that result or guarantee
a Compositional advantage.

Do not launch a 100k full run with the 146k toy model merely because the data
contract is ready. First benchmark a credible larger EBT configuration on the
available 8xH200 node with no-update and a short bounded update run. Freeze
model size, global batch, optimizer-step semantics, candidate-gradient rule,
and measured throughput before issuing a wall-time/GPU-hour estimate. A new
model/data-scale training protocol must have a recorded governance decision
before full training. Do not access unseen triple data during tuning.
