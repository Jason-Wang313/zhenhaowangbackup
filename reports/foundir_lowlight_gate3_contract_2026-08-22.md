# FoundIR-Lowlight Gate 3 frozen contract — 2026-08-22

## Decision this contract protects

Gate 3 answers two independent questions on the reusable non-FoundIR paired
development suite:

1. Can a compact deterministic raw-to-restored pixel model beat the cheap
   photometric control or provide complementary information to the released
   FoundIR cascade?
2. Does the advisor-directed Qwen model carry objective real exposure-order
   information, and—only after that passes—does Qwen add paired-real
   restoration value beyond deterministic measured image statistics and a
   fixed shuffled-Qwen control?

No FoundIR-Lowlight ID 1101--1150 may be opened or inferred. Gate 3 uses the
same 429 frozen training records and 245 reusable outer-development records as
Gates 1--2. A failure is diagnosed on those same families; it does not trigger
another test-set acquisition.

## Compact control

The compact model has 1,684,172 parameters. It is a two-level deterministic
residual U-Net with:

- a bounded whole-image RGB gain/gamma/offset estimate from one 128-pixel
  global view;
- one identical global tone estimate reused for every native-resolution tile;
- a bounded local RGB residual initialized to exact identity;
- 1,200 optimizer steps, global batch 32, and exactly 38,400 family/group-
  balanced crop presentations on GPUs 2--3;
- MSE-led RGB supervision, a bounded Charbonnier term, low-frequency MSE, and
  a 10% identity replay stream;
- a predeclared final-step checkpoint, with no outer-development checkpoint or
  hyperparameter selection.

The deployable compact output is compared to fixed train-mean exposure and the
FoundIR cascade under the frozen two-path PSNR/LPIPS evaluator. A target-only
compact/cascade convex blend is retained only as a complementarity ceiling.

## Qwen gates

The existing frozen same-scene exposure diagnostic runs unchanged on GPUs
0--1. Its thresholds are 0.75 concordance, 0.80 hard non-inversion, 0.60 median
stack Spearman, 0.80 positive-stack fraction, and 1.0 expected-level span.

If and only if the ordinal gate passes, the downstream causal comparison uses
identical restoration weights for measured-only, oracle-plus-measured,
measured-plus-Qwen, and measured-plus-shuffled-Qwen. Qwen must improve PSNR by
at least 0.20 dB over both controls on every family, regress LPIPS by no more
than 0.005, and recover at least half of the incremental oracle gain. The total
Qwen/basic-interface correction budget remains two. Missing the final causal
gate blocks Gate 4 and the protected event for user/advisor review.

## Runtime and execution

The four-H100 bundle runs the independent compact and Qwen ordinal jobs
concurrently. The target is 20 minutes, but this is an engineering objective,
not a scientific rule that permits truncating training or evaluation. All
checkpoints are cache-preserving and immutable, so an infrastructure failure
resumes from the latest predeclared checkpoint rather than restarting.

Machine-readable contract:
`configs/foundir_lowlight_gate3_v1.json`.

Protected-test status: `NOT_ACCESSED_DURING_GATE3_CONTRACT_FREEZE`.
