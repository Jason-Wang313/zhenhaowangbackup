# FoundIR scaled EBT cached-target learning v2 readiness — 2026-10-08

## Status

**CPU PREFLIGHT PASS; TWO-GPU MATCHED RERUN NOT YET STARTED.**

v2 repeats real-data smoke v1 with the same records, tensors, model,
initialization, optimizer, schedule, and validation. The only intervention is
reading the frozen target PNG instead of regenerating and source-hashing the
target in each DataLoader worker.

Frozen identities:

- Config: `configs/foundir_compositional_ebt_scaled_learning_cached_v2.json`
  - SHA-256: `cf9a07cb37f583b1765b39b1387dd1424f18044ca2b97e533935f69352a40126`
- Trainer: `smoke/train_foundir_compositional_ebt_scaled_learning_cached_v2.py`
  - SHA-256: `e92cea6c6cd5f1d075da66d0eec2aa8a626ed8cc7addac842505f43389df6636`
- Runner: `smoke/run_foundir_compositional_ebt_scaled_learning_cached_v2_2gpu.sh`
  - SHA-256: `aba29f46b342324b622ee8f4d3321598d934ebbdd95fdc1fb2f87044c9bffeba`

Equivalence preflight passes:

- 51,200 train record IDs and ordering identical to v1.
- 512 validation record IDs and ordering identical to v1.
- Cached and on-the-fly tensors are bit-identical for `none`, `jpeg`, and
  `resample` nuisance families under the same crop/augmentation seed.
- Fixed validation tensors are bit-identical.
- No triple, official FoundIR, or sealed data is accessible.

The first launch under run ID `cached400_v2` stopped before data/model access:
direct script execution could not import the sibling `smoke` module because
the workspace root was inserted into `sys.path` too late. Both arms exited at
import with status 1 and no result. The import ordering is corrected and the
exact direct-script invocation plus tensor-identity regression now pass. Keep
the failed run as evidence and use the new retry ID below.

Launch:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_SCALED_CACHED_RUN_ID=cached400_v2_retry1 CUDA_VISIBLE_DEVICES=0,1 bash smoke/run_foundir_compositional_ebt_scaled_learning_cached_v2_2gpu.sh
```

- Estimated wall time: `0.03–0.25 hours`.
- Estimated allocated compute: `0.06–0.50 GPU-hours`.
- Result:
  `checkpoints/foundir_compositional_ebt_scaled_learning_cached_v2/cached400_v2_retry1/result.json`.

Do not automatically launch 100k after v2. Compare v2 metrics exactly against
v1 and use measured data-wait/compute time for the full-training design.
