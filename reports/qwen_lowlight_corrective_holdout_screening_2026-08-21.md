# Qwen Lowlight corrective-holdout screening — 2026-08-21

## Outcome

The authors' official Retinex-Net `Testing Images` archive was extracted and
screened successfully. No model inference was performed. All 335 image files
are exact- and perceptually disjoint from the 13,793 protected files. The
closest candidate/protected 64-bit DCT-pHash distance is 10, above the frozen
rejection boundary of 4.

This is acquisition and annotation readiness only. The holdout is not labeled
or frozen for inference, and it does not authorize corrective training,
diffusion, consumed-test reuse, or FoundIR inference.

## Source identity and extraction

- Official page: `https://daooshee.github.io/BMVC2018website/`, link label
  `Testing Images`
- Archive: `data/real_lowlight_corrective_holdout/sources/Test.zip`
- Bytes: `94,051,972`
- SHA-256:
  `0033eb74d4c964755f02d4b46e6db3eb457637c116415571fa2d0212577b3f29`
- ZIP integrity: passed
- Extracted image-only root:
  `data/real_lowlight_corrective_holdout/official_test_0033eb74d4c/`
- Images: 335; non-image `.DS_Store` and `__MACOSX` metadata were excluded

Subset inventory:

| Subset | Images |
|---|---:|
| DICM | 44 |
| Fusion | 18 |
| LIME | 10 |
| MEF | 79 |
| NPE | 75 |
| VV | 24 |
| low | 85 |

## Protected union and overlap method

The tool re-read every protected image, recomputed its SHA-256, and refused to
continue if any recorded identity changed. The exact protected union is:

| Membership | Files |
|---|---:|
| Real adaptation source inventory | 1,378 |
| Records actually used in prior real evaluation | 315 |
| Official FoundIR Lowlight LQ and GT | 100 |
| Synthetic Lowlight pilot records | 10,000 |
| Synthetic pilot clean sources | 2,000 |
| Total unique paths | 13,793 |

The prior-real acquisition inventory also contains 108 unselected SICE
exposures. They are not part of the frozen `records actually evaluated`
boundary and therefore are not counted as prior evaluation. The 315 protected
paths are the exact union of clean, degraded, and reference paths in the
frozen prior-evaluation CSV.

- Exact method: SHA-256
- Perceptual method: project-standard 64-bit 32x32 DCT pHash
- Rejection rule: Hamming distance at most 4
- Protected inventory bundle SHA-256:
  `32dba2fa2267ef9e4a10bd5371be422718983b9c626afdd059782a5f405004a3`
- Candidate exact overlaps: 0
- Candidate perceptual overlaps: 0
- Minimum observed candidate/protected distance: 10

## Internal scene and exposure grouping

The 335 paths contain 260 exact-unique images. All 75 files under `NPE` are
byte-identical copies of corresponding paths under `low`. Exact duplicates,
pHash neighbors at distance at most 4, and conservative filename-identified
exposure/encoding series were unioned into 190 scene components.

The component size distribution is: 105 singletons, 67 pairs, two triples,
ten groups of four, one group of five, one group of six, two groups of seven,
one group of eleven, and one group of fourteen. Cross-subset grouping also
detects one DICM image corresponding to a MEF exposure stack. At most one image
may be selected from any component.

## Human annotation package

The annotation template has 190 unlabeled component rows. It deliberately has
blank selection, eligibility, severity, rater, confidence, and notes fields.
Two raters must work independently without consulting model predictions; a
third human adjudicates disagreements. Fused/enhanced outputs, masks/maps,
non-natural images, and unsuitable inputs must be rejected during review.

Artifacts:

- Candidate JSON SHA-256:
  `c95b5069ad7756265ed0a3c71411a6cabbc0e877e8a2460122f8e93b293a0537`
- Candidate CSV SHA-256:
  `81b3b6080f92124813f207a3d950b9dc922a006a7582c3276142f7c24611b265`
- Blank annotation CSV SHA-256:
  `b01d90e495eb1c514c0842bf710ac093b412289511a1d75c15f2bad043a3382a`
- Frozen rubric SHA-256:
  `3d9f57bc1690e874f57240b340a759032ed17f078e25307da3aaffa25adc70e1`
- Local visual review index:
  `data/real_lowlight_corrective_holdout/human_annotation_package_20260821/review_index.html`
- Review-index SHA-256:
  `5f91a4397aca6b4638cd3ea799ba34d0b9ddde32ba742113060727d3150f9b83`
- Review-index validation: 335 image references, 335 unique, zero missing

Seven focused unit tests passed. Python compilation and `git diff --check`
also passed.

## Required next gate

1. Make two copies of the blank annotation CSV and obtain independent human
   labels under the frozen rubric.
2. Adjudicate disagreements, retain no more than one image per component, and
   verify that all L1-L5 levels have defensible independent-scene support.
3. Run a fail-closed finalizer that records agreement, hashes, labels,
   thresholds, and the exact final manifest; repeat the protected-overlap
   screen before publishing it.
4. Only after that immutable freeze, predeclare and run the corrective VLM
   adaptation. The GPU path must preserve the declared schedule and support
   the user's available two H100s through `CUDA_VISIBLE_DEVICES=0,1`.
5. Infer the new holdout exactly once, report pass/fail, and stop for review.
