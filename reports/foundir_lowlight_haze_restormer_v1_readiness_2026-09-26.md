# FoundIR Low-light+Haze direct Restormer v1 — GPU readiness

Status: CPU-ready, governance-resolved, not yet launched because the Codex shell exposes no GPU.

The bounded run targets only `09Lowlight_Haze`. It trains a direct project Restormer from the public task-native deraining initialization; it does not use released FoundIR weights. All 221 source groups represented by the 660-image comparison are excluded from training and development checkpoint selection.

- frozen training manifest: 75,370 train records / 7,514 groups and 128 full-resolution development records / 128 groups;
- training: 6,000 optimizer steps, global batch 32 on one homogeneous 8-GPU node;
- milestones: steps 1,000, 3,000, and 6,000, each evaluated full-resolution with overlapping 512 tiles;
- selection: worst normalized PSNR/LPIPS gate margin, then earlier step;
- final: one evaluation on the frozen 60-image `09Lowlight_Haze` comparison subset, which is not used for selection;
- Full FoundIR target: `20.939309 / .159126`;
- no official test access and no automatic follow-on.

Artifacts:

- config SHA-256: `d60d67ba2a285fa0cb0691ac04f9630e2dc3129f1bf95d7563934ce777235cd3`;
- manifest SHA-256: `afa0e22b27c9c9773be67454a36c80bfcc7fd3e26f97b91be7577d17bd8b8a0e`;
- trainer SHA-256: `fa8c8bd9afa6cd867882c0ea5eb3c00f6242022f1763a444ee0508d5e394ec55`;
- evaluator SHA-256: `5993cc7ebef88e781100753c38baf67aab5681c46c0e55c306fef0f657488957`;
- launcher SHA-256: `4af1404fa28694ee6c0ecf53421506124b79bc875b6d249fe5b6f3d8f01ef10d`.

Checks passed: Python compilation, shell syntax, immutable hashes, manifest cardinality and source isolation, config contract, H100/H200 runtime guards, governance resolution, and launcher `--check`. No existing terminal result is present.

Runtime budget for the entire launcher, including startup, training, three development evaluations, selection, final 60-image evaluation, metrics, and artifact writes:

- 8xH200: 100–160 minutes, center 125 minutes; 13.3–21.3 GPU-hours, center 16.7;
- 8xH100: 130–200 minutes, center 160 minutes; 17.3–26.7 GPU-hours, center 21.3.

Concurrency is one homogeneous eight-GPU node. The main uncertainty is full-resolution tiled LPIPS plus shared-storage hashing throughput.
