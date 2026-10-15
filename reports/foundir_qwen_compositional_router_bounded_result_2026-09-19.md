# FoundIR Qwen compositional router bounded result

Status: improved substantially, but failed the complete deployment gate.

No GPU was used. Qwen and every restoration expert remained frozen. No FoundIR internal or official test data was accessed.

## V1: five independent component heads

The eleven mutually exclusive classes were replaced by five independent linear present/absent predictions for Blur, Noise, Rain, Haze, and Low-light.

- Complete development correct accepted: 13,928 / 16,461.
- Wrong accepted: 125 (0.7594%).
- Refused: 2,408.
- The previous eleven-class consensus had 201 wrong accepts, so compositional prediction reduced errors by 37.81%.
- Blur+Low-light improved from 47 wrong to 16 wrong.
- Noise+Low-light improved from 138 wrong to 107 wrong, but remained the blocker.

## V2: low-light-conditional Noise head

One binary linear head was fitted only on low-light-bearing classes to decide whether Noise was also present.

- Complete development wrong accepted: 58 (0.3523%).
- Overall wrong-rate gate passed.
- Noise+Low-light: 44 wrong / 1,743 (2.5244%), so the per-class gate still failed.

## V3: asymmetric absence safety

The positive Noise gate was frozen. Only the evidence required to declare Noise absent was strengthened. Calibration selected a `0.99` negative confidence margin.

- Complete development correct accepted: 13,243 / 16,461 (80.4508%).
- Wrong accepted: 24 (0.1458%).
- Refused: 3,194.
- Noise+Low-light: 10 wrong / 1,743 (0.5737%), coverage 51.4630%; both gates passed.
- All mixed classes passed the wrong-rate and coverage gates.
- Pure Low-light coverage fell to 47.2857%, below the required 50%; therefore the integrated gate failed.

This is the stopping boundary for threshold-only work: loosening the negative Noise gate reintroduces wrong Noise+Low-light routes, while tightening it further reduces pure Low-light coverage. Do not continue threshold sweeping on these linear heads.

## Historical native-Qwen audit

The saved physical-v8 native Qwen outputs were also checked before proposing more training. On the historical 128-record FoundIR training-only diagnostic they achieved only 52.3438% exact degradation-set accuracy, with Noise recall 54.3478%. They are not an already-solved mixed router and cannot replace the failed gate.

## Next bounded direction

If routing work continues, use one small nonlinear binary classifier specifically for the remaining `Low-light` versus `Noise+Low-light` ambiguity, trained on frozen Qwen features. Do not change Qwen, restoration experts, or the other ten routes. Because complete development has now been inspected repeatedly, any claim of success must use a newly frozen training-only source-disjoint holdout and explicitly report its limited remaining source diversity.

## Artifacts

- V1 result: `checkpoints/foundir_qwen_compositional_router_v1/result.json`
- V1 head SHA-256: `7d8a8d2584fc5193f091523a7b2c76ff91898bd7612d9dd167abb9b0cd0af6bb`
- V2 result: `checkpoints/foundir_qwen_compositional_router_v2/result.json`
- V2 head SHA-256: `05c45e9ce63900c69aaf088281d02f8b0be0625c09c47354e4a8d93e95fa7bf9`
- V3 result: `checkpoints/foundir_qwen_compositional_router_v3/result.json`
- V3 predictions SHA-256: `62adefca53a7230e236be4a6ee95af79f366173ea7b0e0ed9b17f61d091a37db`
