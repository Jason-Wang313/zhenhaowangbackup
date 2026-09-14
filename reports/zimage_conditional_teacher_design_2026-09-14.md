# Z-Image conditional teacher: first-principles redesign

DIAGNOSTIC ONLY / 非主线. No official FoundIR or SOW acceptance claim.

## Evidence and limits

The latest latent-first run learned the 16 training crops (29.2780 dB /
.1247 LPIPS) and damaged the 16 development crops (21.4293 / .4635).
This demonstrates failure of that recipe at that data scale, not impossibility
of Z-Image restoration. Requiring a 79M-parameter adapter trained on 16 fixed
crops to generalize before allowing a diverse pilot confounds learnability
with generalization. Earlier teacher adapters also used BF16 optimizer state.
The current reproducibility repairs resolve the measured repeatability issue;
they do not establish a unique cause of all historical quality failures.

The paired-flow code supplies x_s=(1-s)GT+s*LQ, asks the pretrained model for
velocity at native noise level s, and supplies no persistent LQ condition.
Thus s=1 contains an image although pretraining s=1 expects Gaussian noise.
At s<1, training states contain GT information that deployment states need
not contain. A deterministic paired transport without a separate condition
can be mathematically valid, but these choices discard useful pretrained
semantics and make conditional restoration harder to learn with tiny data.
The FLUX full-development reference instead used noise-to-GT training with
persistent native reference tokens and achieved 25.7342/.1454 on 2595 records.
It is evidence for a design direction, not an apples-to-apples backbone test.

GT VAE roundtrip is 40.7184/.0076 on the fixed development subset. Direct
training-latent oracles also work. Decoder tuning previously damaged GT
roundtrips. Keep the original VAE frozen for this upstream correction.

## Model and mathematical contract

Use the pinned undistilled Tongyi-MAI/Z-Image Base 6B, frozen native text
encoder and VAE, fresh rank-32 FP32 LoRA, fresh FP32 AdamW. Do not inherit any
failed teacher/endpoint/decoder checkpoint. Native DiT architecture, RoPE,
timestep transform, output sign, and pretrained prediction head stay intact.

For clean latent z, observed latent y, and independent epsilon~N(0,I):

    x_s = (1-s) z + s epsilon
    target = epsilon - z
    L = mean((v_theta(x_s, s, y) - target)^2), s~Uniform(0,1)

Each model call embeds aligned 2x2 patches as W_x patch(x_s) + b_x +
W_y patch(y), with W_y initialized to zero and trained in FP32. This is
equivalent to channel concatenation with a pretrained state half and a zero
condition half, adds 245,760 parameters, and does not increase token count.
The explicit y is present at every integration step. It is image conditioning,
not Qwen degradation conditioning. At initialization the new function must
match the repaired native Base function exactly. After an update, output must
respond to changing y while x_s, s and prompt are held fixed.

Deploy from Gaussian noise, integrate v in descending s with the Base 50-step
static-shift-6 grid, CFG=1. Solver state and velocity target arithmetic are
FP32; pretrained operations stay BF16. Uniform training s covers the complete
inference domain without a finite solver-grid curriculum. No GT, target-derived
filter, or target-based correction is available during inference.

## One recipe, two distinct questions

Tiny stage: same immutable 16 train and 16 development crops, 1200 steps,
global batch 8, LR 5e-5 (condition projection 1e-4), 100-step warmup. Evaluate
0/400/1200 with fixed Gaussian seeds and save Input|Output|GT and PSNR/LPIPS.
Train admission requires final PSNR gain >1 dB, LPIPS improvement >.03,
fixed-noise/timestep probe loss below .8 times initial, and visual review.
Development is reported in full but does not masquerade as a statistically
adequate generalization gate at this training scale. These are new prospective
criteria; old failures remain failures under their original criteria.

Pilot stage: only after that tiny metric AND visual pass, warm-start the
passing teacher and reset optimizer. Use 2048 unique training source groups
(one deterministic aligned crop per group), 128 development source groups,
2000 steps, global batch 32, same loss/LRs. Include the original 16 fixed
development examples and 112 additional groups; never move records between
the frozen parent train/validation splits. Freeze the selection and hashes
before GPU inference. Deterministic spatial flips/rotations of training crops
add variation without changing degradation category. Final development must
improve both PSNR and LPIPS over input and pass visual review. Report record
and source-group metrics; the selection has one record per source group.

Do not automatically chain stages before visual review. This is one bounded
formulation, not an LR sweep. A failed tiny stage is evidence against this
implementation/conditioning learnability; a failed diverse pilot is stronger
evidence against this formulation. Preserve failure evidence and interpret
the condition-sensitivity checks, train/probe curves and visual failures.

If the pilot succeeds, full training and all 2595 development records become
the next quality stage; only a quality-valid teacher is distilled to few/single
steps. Decoder work remains conditional on measured need. No full training,
successful distillation, or SOW success exists at design time.

## Verification and authority

Before GPU launch: analytic flow/solver tests, aligned patch tests, zero-init
native forward/backward equivalence, real 6B CPU updates, condition sensitivity,
checkpoint reload equivalence, strict deterministic repeated updates and
eight-rank synchronized-state checks. Verify H100/H200 profiles preserve global
batch and sample order. Reuse existing trusted padding/Flash controls. CPU
checks do not establish CUDA quality or full-resolution memory.

User instruction: "execute, first principles, jump outside of the box, first
principles, u can do it, u can push everything well above the original goal".
Record this as existing advisor route plus new explicit USER authorization,
never as a fabricated advisor response. The model-switch pause is satisfied.
Qwen degradation conditioning, mixed degradations, sealed internal, official
FoundIR test and development-GT optimization remain prohibited.
