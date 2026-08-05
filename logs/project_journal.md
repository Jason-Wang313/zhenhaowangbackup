# Synthetic-data quality-gate journal

## 2026-08-04 — initial audit

- Workspace used: `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`
- Required path in `agents.md`: `/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang`
- Path discrepancy: recorded; no access to or work in the other path.
- `AGENTS.md`: not present; `agents.md` was read completely.
- Scope: 100 clean sources; 100 each AgenticIR noise, blur, haze, rain; 100 each GenDeg low-light L1-L5.
- VLM training/evaluation: not started.
- Forbidden large datasets: GenIRData, LSDIR, and FoundIR not downloaded.
- Current workspace assets: no clean dataset, AgenticIR implementation, GenDeg checkpoint, manifests, logs, or prior outputs found before this audit.
- Environment: workspace-local Miniconda and GPU training environments exist; canonical stable environment is `.conda/gpu-train-stable`.
- Initial state: all dataset, implementation, checkpoint, and output items are `MISSING` until independently verified.

## Execution-loop record

| timestamp (UTC) | process | source/destination | status | evidence | next action |
|---|---:|---|---|---|---|
| 2026-08-04 | none | workspace audit | VERIFIED | tree, disk, and process checks performed | locate official AgenticIR and GenDeg assets |
| 2026-08-04 | complete | AgenticIR repositories | VERIFIED | AgenticIR `9640a291480dee3ba8f2974125d4ee9e3440f3d6`; GenDeg `24656bfd4458e87f0efefae4692e7ad48dd4ca44` | obtain required data/checkpoints |
| 2026-08-04 | complete | official HQ/depth archives | VERIFIED | HQ.zip 128583194 bytes, 100 PNGs; depth.zip 15.1M, 100 MAT pairs; SHA256 in `manifests/download_sha256.txt` | generate AgenticIR batch |
| 2026-08-04 | complete | AgenticIR smoke | VERIFIED | official functions; four readable 512x1024 outputs; repeated seeded hashes identical | run 100x4 batch |
| 2026-08-04 | complete | AgenticIR 100x4 batch | VERIFIED | exactly 400 readable outputs and 400 manifest rows; hashes/dimensions verified; contact sheets reviewed | acquire valid GenDeg checkpoint |
| 2026-08-04 | failed | GenDeg official checkpoints | FAILED | README URLs retried verbatim: generator returned HTTP 401/empty response; structure link returned HTML share page, not a checkpoint; all responses quarantined as `.part` | obtain refreshed official checkpoint or authenticated access |

## Current blockers

- GenDeg low-light outputs remain ungenerated because the official generator link requires authentication (HTTP 401) and the structure link returns an HTML share page rather than checkpoint bytes. No invalid response was promoted to a checkpoint.
- The current shell exposes no CUDA GPU (`torch.cuda.is_available() == False`, device count 0), so GenDeg inference cannot be validated here even after checkpoint recovery.
- Final retry of the exact README URLs: `gendeg.ckpt` returned HTTP 401 with an empty body; `nafnet_sd.ckpt` returned HTTP 200 but 266,443 bytes of HTML. Both remain `.last.part` quarantine artifacts and were not accepted.
- Uploaded chunks `checkpoints/gendeg_part_00` through `checkpoints/gendeg_part_07` were concatenated in numeric order into `checkpoints/gendeg.ckpt`. Final size is 7,706,335,711 bytes; PyTorch archive reader found 1,839 records and `unzip -t` reported no errors. Full CPU `torch.load` was killed by RAM pressure, so GPU-side loading remains the authoritative next test.
- Offline GenDeg runtime staging completed in the shared environment: NumPy, OmegaConf 2.1.1, official k-diffusion source, scikit-image, jsonmerge, torchdiffeq, torchsde, dctorch, clip-anytorch, clean-fid, trampoline, antlr4 runtime, and ftfy were installed without GPU-shell internet. `import numpy, torch, k_diffusion, omegaconf` passes.
- One-time authorized read-only reference check of `guangzhaoli` project area found no GenDeg environment. Its base Python has no Torch; its `z-image` environment has Torch 2.9.1+cu128 but not the GenDeg dependency chain. No files were copied or modified outside this workspace.
