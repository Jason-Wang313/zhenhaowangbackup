# Stage 0 restoration-backbone audit

Date: 2026-08-12
Workspace: `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang`

## Local verification

- GPU: not visible in this Codex session (`nvidia-smi` unavailable).
- Local candidate weights: no Z-Image, FLUX, SDXL, or SD2.1 restoration checkpoint was found.
- Disk: sufficient for a separately authorized model checkout, but no model was downloaded.
- Existing verified assets remain the 100-image AgenticIR pilot and the frozen low-light generator.
- Official FLUX.2 source checkout is now present at `third_party/flux2`, revision
  `50fe5162777813d869182b139e83b10743caef15`.
- The 4B Base checkpoint download is active at
  `models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors`; it is resumable
  and currently transferring slowly. The required `ae.safetensors` is not in
  the 4B Base repository; the FLUX.2-dev repository returns HTTP 401 without
  Hugging Face authorization, so the VAE is not yet downloaded.

## Candidate decision

The first GPU candidate is **FLUX.2 [klein] 4B Base**. The official FLUX.2 repository
documents single-reference and multi-reference image editing, recommends the Base
variant for fine-tuning/LoRA, and lists the 4B model as Apache-2.0 and suitable for
consumer GPUs. This is a better initial project fit than FLUX.1-dev: FLUX.1-dev is
12B, requires gated access, and uses a non-commercial license.

This is a candidate decision, not a completed restoration validation. Reference-image
editing is not automatically equivalent to a faithful paired restoration path. The
GPU smoke test must verify identity preservation before any training decision.

Fallback order, as directed in `agents.md`: FLUX/Z-Image alternative, then SDXL,
then SD2.1. Conventional All-in-One models are comparison references only.

## Required GPU smoke test

1. Verify GPU, CUDA, Python, and available VRAM.
2. Make the official FLUX.2 code and the locally approved 4B Base checkpoint
   available in the GPU environment.
3. Run one deterministic clean/degraded pair through image editing at the smallest
   supported resolution.
4. Check that the output preserves scene identity and structure; an attractive
   hallucinated image fails.
5. Record checkpoint revision/hash, license, resolution, dtype, seed, steps,
   sampler/flow settings, VRAM, and output paths.

Do not start Qwen-VL, 100k synthetic-data generation, or adapter training until this
smoke test passes and the image-conditioning interface is inspected.

## GPU handoff commands

Run these in the GPU-enabled interface, from this workspace:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python -m pip install -e third_party/flux2 --no-deps
python -c "import torch; print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
```

The `--no-deps` flag is intentional: the GPU environment already has a custom
PyTorch/CUDA stack and cannot reach the internet. The checkpoint download and
license acceptance must be performed by the user in that GPU environment or by
placing an authorized checkpoint in the shared workspace. Do not paste
credentials into this workspace. After the checkpoint is available, the next
implementation task is a small paired low-light inference harness and
identity-preservation report.
