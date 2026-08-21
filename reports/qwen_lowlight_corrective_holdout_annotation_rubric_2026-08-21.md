# Qwen Lowlight corrective-holdout human annotation rubric

Status: frozen annotation instructions; images are screened candidates only.
No Qwen checkpoint, consumed real test, or FoundIR image may be inferred while
this package remains unlabeled.

## Annotation unit and independence

The annotation unit is one scene component in
`manifests/qwen_lowlight_corrective_holdout_annotation.csv`. A component may
contain byte-identical copies, perceptual near-duplicates, alternate encodings,
or exposure variants. Inspect every path listed in `candidate_paths_json`, then
select at most one path. Never select two images from the same component.

Two human raters should independently complete separate copies of the CSV.
They must not inspect model predictions. A third human resolves disagreements.
Record each rater ID, confidence (`high`, `medium`, or `low`), and a concise
note for exclusions or ambiguous severities.

## Eligibility

Set `valid_natural_lowlight_input=yes` only when the selected image is a
readable natural photograph whose principal relevant degradation is exposure
or illumination. Reject (`no`) synthetic graphics, already enhanced or fused
outputs, grayscale masks/maps, severe unrelated corruption, and images whose
content or exposure cannot be judged. Leave `selected_path` empty for a
rejected component and explain why in `notes`.

The selected path must be copied literally from `candidate_paths_json`. Do not
crop, resize, recompress, rename, or otherwise transform it.

## Severity labels

Assign exactly one project level to each eligible selected image:

- `L1`: normally exposed/clean; no meaningful low-light enhancement needed.
- `L2`: mildly underexposed; content is clear and only modest brightening is
  warranted.
- `L3`: moderately underexposed; important shadow detail is reduced, but most
  scene content remains readily interpretable.
- `L4`: strongly underexposed; substantial content is difficult to see and
  enhancement is clearly necessary.
- `L5`: extremely underexposed; large regions are near black or essential
  content is barely visible.

Judge the need for enhancement, not whether the scene depicts nighttime.
Legitimately dark night skies do not by themselves imply `L5`. Ignore the
filename, folder name, and mean-luma metadata when deciding severity; use the
visible image. Do not infer a reference exposure that is not present.

## Freeze requirements before inference

After adjudication, a separate finalization step must:

1. retain at most one image per screened scene component;
2. contain every required level L1-L5 with a defensible number of independent
   scenes, or stop and acquire more data;
3. report raw agreement and quadratic-weighted kappa between the two raters;
4. disclose labels literally as `human_severity_labels`;
5. freeze the selected paths, image SHA-256 values, component IDs, labels,
   rubric hash, manifest hash, evaluation thresholds, and selection logic;
6. prove exact and pHash-distance-at-most-4 disjointness from all protected
   sources again; and
7. perform model inference exactly once only after that freeze passes.

The future holdout evaluation must use the same real-test metric definitions
and thresholds predeclared for the completed adaptation unless a stricter
contract is frozen before inference. Thresholds may not be weakened after any
holdout prediction is observed.
