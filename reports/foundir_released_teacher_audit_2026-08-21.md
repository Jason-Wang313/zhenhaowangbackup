# FoundIR released-teacher and training-metadata audit

Date: 2026-08-21 UTC

No model inference was performed and no official FoundIR test image was
opened. This audit covers checkpoint and metadata identity only.

## Released checkpoints

| Teacher | Bytes | SHA-256 | Stored training state | Tensor audit |
|---|---:|---|---|---|
| Generalist | 580,829,530 | `2603a0be47d0844d4c9cea542ecca039e0e4aaa39d45d41064bfbdc13c0830c9` | step 1,300,000 | online model: 290 tensors / 36,275,187 parameters; EMA wrapper present |
| Lowlight specialist | 6,294,432 | `f834248e145d020d5a02f257b7ceb383b579add2522938b3e1e0bd460abe6` | iteration 60,000, epoch 413 | 591 tensors / 1,512,718 parameters |

The checked FoundIR source revision is
`6d19958499622d90ca71fe971d83b1e428c17cd0`.

## Official training-distribution metadata

- Single-degradation metadata: 472,046 pairs, including 39,962 isolated
  Lowlight pairs; SHA-256
  `5a204d5d2c4a641e91737ccebf1731dea7c73f14d650f1c11eba28fd6335b99d`.
- Complete metadata: 1,010,064 pairs, including 378,440 records whose category
  contains Lowlight; SHA-256
  `fcc5668470fda3f49750019d9a0ac4d009bbecaa45d928f24d1c615ff8f54ad5`.

The metadata is present, but the approximately 4.85 TB image archive is not.
Access still requires the official author request form.

## Use in the SOTA route

Both released models will first be evaluated on non-FoundIR paired development
images. They may then provide confidence-weighted training targets: agreement
between generalist, specialist, and paired target increases distillation
weight; disagreement marks a hard/uncertain example. They are not permitted to
select settings from FoundIR test results. The deployed project output remains
FLUX/Z-Image-based, as required by the advisor route.

The machine-readable companion is
`manifests/foundir_released_teacher_audit.json`.
