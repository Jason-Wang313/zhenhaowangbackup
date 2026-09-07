# FoundIR five-type FLUX.2 mainline v2 readiness — 2026-09-06

## Scope

This route follows the advisor's confirmed five-type scope: Blur, Noise, Rain,
Haze, and Low-light, including admitted single and mixed FoundIR training
categories. JPEG is excluded from training, development validation, internal
evaluation, and every gate. The protected FoundIR official test is not read.

The user disabled Qwen-condition effectiveness ablations for this run.
Oracle, generic, and shuffled/wrong condition evaluations are not executed.
Released FoundIR checkpoint evaluation remains a restoration-benchmark
comparison. It is not a Qwen-condition ablation.

## Frozen Qwen condition

The FLUX condition uses the physical-v8 per-type Qwen checkpoint. Qwen asks
five separate questions on one deterministic 448-pixel center crop and renders
the five predicted levels into canonical restoration text. Synthetic
validation passed with type micro-F1 96.47%, degradation-set exact 87.83%,
triple set exact 82.10%, active severity accuracy 87.42%, and 100% valid
structured output.

The preserved 128-image FoundIR-training-only diagnostic is not hidden: type
micro-F1 was 82.84%, macro-F1 84.05%, minimum per-type recall 54.35%, exact
type-set accuracy 52.34%, and valid output 100%. FoundIR provided type labels
but no project L1--L5 truth, so severity was not scored. The advisor prohibited
training Qwen on FoundIR; Qwen remains frozen. The user explicitly authorized
proceeding to restoration after reviewing this evidence.

## FLUX route

- FLUX.2 [klein] 4B Base with frozen VAE and frozen native Qwen3 text encoder.
- Native reference-image token path; no restoration adapter before the
  128-pair overfit gate demonstrates it is needed.
- DiT LoRA rank 32 / alpha 32 on the declared image-attention and
  image/text-interaction layers.
- Five-percent GT-to-GT identity pairs; zero generic-condition dropout on
  degraded pairs.
- Development checkpoint selection uses only Qwen-conditioned validation
  PSNR/LPIPS. Internal test remains sealed until selection is frozen.
- Official FoundIR test remains untouched until the user explicitly approves
  the final frozen evaluation.

## Execution order

1. Verify frozen artifacts, full five-type training manifest, offline metric
   weights, GPU runtime, and the no-JPEG/no-ablation contract.
2. Run the VAE ceiling and 128-pair native-reference overfit gate. Stop on a
   scientific failure before the expensive full condition export.
3. Export resumable physical-v8 Qwen conditions for all 685,495 admitted
   training/development rows, five questions per image.
4. Cache only the Qwen and clean-identity FLUX Qwen3 prompt contexts.
5. Freeze the source-group-disjoint development/internal evaluation contract.
6. Run the 1.8-million-presentation FLUX curriculum and evaluate the six
   predeclared checkpoints on development data.
7. Select the checkpoint by worst-task normalized PSNR/LPIPS margin, then run
   native-resolution and one-time sealed internal evaluation.
8. Stop before the protected official FoundIR test.

The full-data existence scan is not repeated at launch. The already frozen
685,495-row manifest/hash and completed installation marker are verified
instead; this avoids millions of redundant shared-filesystem metadata calls.

## Runtime-preflight correction — 2026-09-07

The first v2 launch exited before any VAE, Qwen export, or FLUX training work.
Its runtime preflight imported the FLUX/Qwen stack first and therefore cached
`transformers 4.56.1` in the parent Python process. It then tried to verify the
released FoundIR specialist's intentionally separate `transformers 4.47.1`
stack in that same process and falsely reported dependency drift. The installed
workspace-local dependency bundles had not changed.

The specialist identity check now runs in a fresh subprocess whose import path
is explicitly pinned to `transformers 4.47.1`, `tokenizers 0.21.0`, and
`huggingface_hub 0.27.1`; the FLUX parent remains on 4.56.1. The subprocess must
also initialize CUDA and import the actual WaveMamba architecture before it can
emit its pass marker. Missing bundles, wrong versions, an import failure, a
wrong architecture, or a missing marker still fail closed. The failed launch
log is preserved, and restart remains safe because no stage output was created.
