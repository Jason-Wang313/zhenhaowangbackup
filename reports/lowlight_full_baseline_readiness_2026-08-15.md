# Lowlight full baseline readiness

Date: 2026-08-15 UTC

## Scope

The passing 16-pair FLUX.2 overfit gate authorizes the fixed full Lowlight
baseline pipeline. This milestone prepares the 8,000-pair training run and the
fixed 1,000-record validation and 1,000-record internal-test evaluations. It
does not claim FoundIR performance, Qwen-VL improvement, or Phase 1 completion.

## Frozen run

- Config: `configs/lowlight_flux2_full.json`
- Config SHA-256:
  `c71235efe88e8f3bc98ca863350af372a78aa35f1a2cc065a1867c5bb6bfde18`
- Training records/sources: `8,000` / `1,600`
- Validation records/sources: `1,000` / `200`
- Internal-test records/sources: `1,000` / `200`
- Per-split level balance: exactly equal across L1-L5
- Training steps: `4,000`, with effective batch size `4`
- Trainable parameters: `8,110,080` in 50 rank-8 LoRA modules
- Inference: 50 steps, guidance `1.0`, seed `2026081503`
- Output: `checkpoints/lowlight_baseline_full/`

Authorization is fail closed. The trainer and preflight require the exact full
config hash and the passing overfit gate. At runtime they also re-hash the
overfit checkpoint and passing guidance-1 metrics named by that gate.

## Restart and evidence behavior

`smoke/run_lowlight_baseline_full.sh` holds a non-stale `flock` lock and appends
all preflight, training, validation, and internal-test output to one log:
`logs/lowlight_baseline_full.log`.

Training resumes only from a matching frozen config and verified checkpoint,
optimizer, metadata, and deterministic probe evidence. It refuses to overwrite
differing checkpoint artifacts.

The split evaluator is restart safe at per-image granularity. Each restored PNG
has a hashed JSON sidecar binding its split, index, pair ID, level, config,
checkpoint, guidance, seed, crop, and PSNR/SSIM/LPIPS values. Aggregate reuse
revalidates every sidecar, every restored-image hash, and every contact-sheet
hash. It produces overall, per-level, L2-L5, and L1-identity aggregates plus
stratified and worst-case contact sheets.

## CPU verification

- Python compilation: pass
- Shell syntax: pass
- Unit tests: 15 pass
- Full-data preflight with large model hashes skipped: pass
- Full-data preflight with all FLUX, VAE, and text-encoder hashes: pass
- Offline AlexNet LPIPS asset: size and SHA-256 verified; metric instantiated
- Manifest: 8,000 train records, 1,600 train sources, 1,600 records per level
- LoRA scope: 50 modules and 8,110,080 parameters

GPU training and evaluation remain pending until this tracked milestone is
committed and pushed.
