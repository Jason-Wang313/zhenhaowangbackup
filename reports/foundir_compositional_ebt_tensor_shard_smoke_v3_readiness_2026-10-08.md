# FoundIR EBT tensor-shard smoke v3 readiness — 2026-10-08

## Status

**CPU PREFLIGHT PASS; TWO-GPU QUALIFICATION NOT YET STARTED.**

v3 uses the frozen 480k tensor bank for training and the unchanged fixed
512-image cached validation protocol. The 51,200-presentation smoke schedule
selects 50 shuffled shards, reads each selected shard sequentially, contains no
duplicate samples, and is exactly balanced: 25,600 singleton and 25,600 double.

CPU loader measurement over 20 batches is approximately `6,002 img/s`, versus
`160.71 img/s` in cached-PNG v2: about 37x higher data supply. The measured EBT
GPU kernel requires approximately 2,560 img/s at batch 128, leaving about 2.34x
feed headroom before GPU compute saturation.

Frozen identities:

- Config SHA-256: `08dfb1b143dd2a961e40aeef407e15bb36e01c544aab9e59f4cd4dd8c56d68a2`
- Trainer SHA-256: `72173a8b6837aa8255716a37ed5a894111e1997ca42a68cc287d291bf84e08ff`
- Runner SHA-256: `c22e79b8d3f199e9a2d679baadb658864bfd8c30a5ecebe81e13f6834d0e5d51`

Launch:

```bash
cd /inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang && EBT_TENSOR_SHARD_RUN_ID=shard400_v3 CUDA_VISIBLE_DEVICES=0,1 bash smoke/run_foundir_compositional_ebt_tensor_shard_smoke_v3_2gpu.sh
```

- Estimated wall time: `0.02–0.15 hours`.
- Estimated allocated compute: `0.04–0.30 GPU-hours`.
- Result: `checkpoints/foundir_compositional_ebt_tensor_shard_smoke_v3/shard400_v3/result.json`.

Training crops differ from v1/v2 because v3 freezes an eight-view bank. The
validation set and metrics remain identical. Therefore compare throughput
directly, but treat post-training metric differences as a data-protocol change,
not solely an I/O intervention.
