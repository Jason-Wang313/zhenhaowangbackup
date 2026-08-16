# Lowlight FLUX.2 full internal-baseline result

Date: 2026-08-16 UTC

## Result

The fixed FLUX.2 Lowlight baseline completed 4,000 H100 training steps and the
complete 1,000-record validation and 1,000-record internal-test evaluations.
Both splits passed the automatic sanity gate, and the contact-sheet identity
review passed with documented failure cases. This authorizes preparation of the
Qwen-VL Lowlight conditioning comparison. It does not establish performance on
the official FoundIR-Lowlight split or complete the Lowlight Phase 1 target.

## Bound evidence

- Training Git revision:
  `4e878b38caa4eef987a9a3900a8b6ae160f60210`
- FLUX revision: `50fe5162777813d869182b139e83b10743caef15`
- Config SHA-256:
  `c71235efe88e8f3bc98ca863350af372a78aa35f1a2cc065a1867c5bb6bfde18`
- Final checkpoint:
  `checkpoints/lowlight_baseline_full/step_004000.safetensors`
- Checkpoint SHA-256:
  `fa51deac872e3df57f147631361841039c386ef43f58a17a56fb68b9924e8f22`
- Optimizer SHA-256:
  `a202903493ef27ae6f632ebaaa7de1a771d39b6ca3a5c8d7205e4f56417fe847`
- Training-summary SHA-256:
  `4581852a5ed35a1654f239040cd050d5ee40b462a22b57215c50f8afb3cd538b`
- Validation-metrics SHA-256:
  `b1b78a9afa5174786702ac17fc2db44524ba374a2264c42067ba68fa3ff08e51`
- Internal-test-metrics SHA-256:
  `6bac93cb0a45556787364c553242d15463b2ea0a5a18d5738054b2a87e1a1da4`
- Machine-readable gate: `configs/lowlight_full_baseline_gate.json`

The H100 environment was Python 3.12.13, PyTorch 2.7.0+cu128, CUDA 12.8,
cuDNN 90701, and one NVIDIA H100 80GB HBM3. The deterministic probe decreased
from `0.2323607057` to `0.1966449618`.

## Internal paired metrics

The L2-L5 rows exclude clean L1 identity samples.

| Split | Scope | Input PSNR | Restored PSNR | Input LPIPS | Restored LPIPS | Restored SSIM |
|---|---|---:|---:|---:|---:|---:|
| Validation | L2-L5 | 10.4028 | 24.9051 | 0.2824 | 0.0411 | 0.8913 |
| Internal test | L2-L5 | 10.1692 | 24.7723 | 0.2891 | 0.0440 | 0.8858 |
| Validation | L1 identity | 100.0000 | 31.4672 | 0.0000 | 0.0172 | 0.9317 |
| Internal test | L1 identity | 100.0000 | 31.6674 | 0.0000 | 0.0185 | 0.9266 |

All automatic checks passed on both splits: deterministic loss decrease, L2-L5
PSNR improvement, SSIM improvement, LPIPS improvement, and mean L1 SSIM at or
above `0.70`.

Internal-test restored results by level:

| Level | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| L1 | 31.6674 | 0.9266 | 0.0185 |
| L2 | 26.2704 | 0.9073 | 0.0307 |
| L3 | 25.9072 | 0.8996 | 0.0350 |
| L4 | 25.0044 | 0.8909 | 0.0405 |
| L5 | 21.9071 | 0.8456 | 0.0701 |

## Visual identity review and limitations

The validation and internal-test stratified sheets preserve scene geometry,
objects, layout, and semantic identity from L1 through L5. The worst-L1 sheets
also preserve structure; their errors are primarily tone, local brightness,
and texture differences rather than content invention. No object insertion,
removal, or scene replacement was observed.

The worst-PSNR sheets expose genuine limitations:

- `div2k_0452_L5` validation is a nearly uniform dark crop and remains largely
  under-restored (`3.01 dB`, LPIPS `0.419`).
- `div2k_0130_L5` validation remains too dark (`11.16 dB`).
- The weakest internal-test dark cases are `lsdir_0001487_L5` (`9.72 dB`) and
  `lsdir_0000904_L5` (`9.85 dB`).
- Individual clean crops can change tone or texture despite the strong mean L1
  identity result. The weakest internal-test L1 SSIM is `div2k_0055_L1` at
  `0.5153`; structure and crop identity are nevertheless preserved visually.

These outliers must remain visible in the later baseline-versus-Qwen comparison.
They do not invalidate the aggregate internal baseline, but they prevent any
claim that restoration is uniformly successful on every synthetic crop.

## Authorization boundary

This result authorizes only the next Lowlight step: selecting and validating a
real vision-capable Qwen-VL checkpoint, freezing it initially, and preparing a
fair condition-versus-no-condition comparison on the same restoration
backbone. The current `models/qwen3-4b-fp8` remains only FLUX's text encoder and
must not be represented as Qwen-VL. Official FoundIR-Lowlight acquisition and
evaluation remain mandatory before any Phase 1 sub-target claim.
