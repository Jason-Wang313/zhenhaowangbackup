# Advisor-approved minimal VLM-token MoE router plan — 2026-10-25

## Decision

The advisor approved the lean proposal: begin with one working restoration
MoE, pass frozen VLM output tokens through a small model, and connect that
signal to the existing router. The user then explicitly authorized execution.

The earlier presentation mixed LoRA-adapter merging, MoE experts, constrained
priors, and block/time schedules. Those additions are removed from the first
experiment. There is exactly one MoE expert bank.

## Minimal mechanism

For restoration image tokens `h` and frozen VLM tokens `v`:

\[
r_{img}=R_{img}(\operatorname{MeanPool}(h)),
\]

\[
r_{vlm}=R_{vlm}(\operatorname{MeanPool}(v)),
\]

\[
g=\operatorname{Softmax}(r_{img}+\alpha r_{vlm}),
\qquad
y=\sum_i g_i E_i(h).
\]

`R_img` is the ordinary image router, `R_vlm` is the small VLM projector, and
`E_i` are the unchanged experts in the sole MoE bank. VLM tokens affect only
expert scores. They do not enter the experts and do not create another MoE.

## Frozen first comparison

1. Image-only router: set `r_vlm=0`.
2. VLM-assisted router: enable the small VLM score branch.

The backbone, expert definitions, training data, optimization, and inference
budget must otherwise match. This first comparison does not test atomic-token
composition, counterfactual training, hard priors, or dynamic schedules.

## Implementation sequence

1. CPU/static module qualification:
   - dense and optional top-k weights sum to one;
   - top-k keeps exactly the requested number of experts;
   - zero-initialized VLM score head exactly reproduces image-only routing;
   - nonzero VLM tokens change only the routing-score path;
   - finite gradients reach the image router, VLM projector, and experts;
   - invalid shapes fail loudly.
2. Select exactly one restoration-DiT insertion point and define a usable
   image-only MoE baseline. The repository currently has no released
   FoundIR-v2 or MiM-DiT implementation to reuse directly, so this choice must
   be explicit before GPU use.
3. Freeze a development-only data split with aligned paired images and frozen
   VLM features/tokens. Official, sealed, parent-test, and internal-test pixels
   remain forbidden.
4. Run a fresh engineering qualification. It may not seed formal training.
5. Only if the image-only MoE itself is usable, run the matched VLM-assisted
   arm and compare restoration quality plus router traces.

## Current reusable VLM material

The repository already stores frozen Qwen visual features of dimension 7,168
(mean and standard-deviation pooling of 3,584-dimensional visual tokens) for
multiple development sets. The 685,495-record five-type manifest also stores
structured Qwen outputs, but not raw Qwen token sequences for every record.
The first GPU protocol must explicitly choose between an existing frozen
feature subset and a new training-only token cache; it must not silently call
structured labels “tokens.”

## Current boundary

The reusable module and concrete FLUX insertion are now implemented in
`lowlight_restoration/vlm_guided_moe.py`. The frozen qualification replaces
`double_blocks.4.img_mlp` with one two-expert dense soft-MoE bank. Both experts
are exact clones of the pretrained image MLP, so insertion initially preserves
the pretrained function. The image router and optional VLM score branch only
control weights over this one bank.

The training-only joined pilot manifest contains 3,150 source-audited records:
2,250 fit, 450 calibration, and 450 holdout, with no source-group overlap.
Both arms use exactly the same generic restoration prompt; therefore the
image-only arm does not receive the Qwen diagnosis through FLUX text context.

CPU/static qualification completed on 2026-10-25:

- `10/10` unit tests passed;
- insertion preserves the original MLP output within the frozen numerical
  tolerance;
- dense/top-k normalization, image-only fallback, scoped conditioning,
  finite gradients, and frozen-backbone trainability checks passed;
- both `image_only` and `vlm_assisted` data preflights passed;
- prompt context shapes are `[1,512,7680]` and IDs `[1,512,4]`;
- no protected data was accessed.

The 25-step eight-GPU engineering launcher is now frozen by
`configs/foundir_vlm_token_moe_minimal_qualification_v1.json` and
`smoke/run_foundir_vlm_token_moe_minimal_qualification_v1_8gpu.sh`. Run the
two arms separately. A qualification checkpoint may not seed formal training,
and neither arm auto-launches a later run. Estimated per-arm budget is
`.25–1.00` wall-clock hours and `2.00–8.00` allocated GPU-hours; replace this
estimate with measured runtime.

No GPU experiment has been launched for this new direction yet: `0`
wall-clock GPU hours and `0` allocated GPU-hours.
