# FoundIR Qwen five-class full development terminal result

Status: passed.

## Frozen selection

The `0.925` confidence threshold was selected over `0.95` before this full development run. On the same 750 source-disjoint training holdouts, both thresholds produced three wrong routes, while `0.925` correctly accepted three additional records. The frozen router therefore used:

- Qwen physical-v8 step-2500 adapter;
- frozen five-way linear head;
- temperature `0.75`;
- confidence threshold `0.925`;
- abstention below threshold;
- no model or head retraining.

## Complete development result

- Records: 8,269.
- Correct accepted routes: 7,931 (95.9124%).
- Wrong accepted routes: 0 (0%).
- Safe refusals: 338 (4.0876%).
- Accepted accuracy: 100%.

| Class | Records | Correct | Wrong | Refused | Coverage |
|---|---:|---:|---:|---:|---:|
| Blur | 2,595 | 2,305 | 0 | 290 | 88.8247% |
| Noise | 1,859 | 1,856 | 0 | 3 | 99.8386% |
| Haze | 1,580 | 1,559 | 0 | 21 | 98.6709% |
| Rain | 835 | 820 | 0 | 15 | 98.2036% |
| Low-light | 1,400 | 1,391 | 0 | 9 | 99.3571% |

## Frozen gate

- Overall wrong-route rate <=0.5%: pass (`0%`).
- Every-class wrong-route rate <=1%: pass (all `0%`).
- Overall coverage >=65%: pass (`95.9124%`).
- Every-class coverage >=50%: pass (minimum Blur `88.8247%`).
- Final gate: pass.

## Improvement over the old singleton audit

The previous physical-v8 singleton audit on these same 8,269 development records had 68.5089% correct singleton routes, 4.3778% wrong singleton routes, and 27.1133% refusals.

- Correct routing: `+27.4035` percentage points, approximately `+40.00%` relative.
- Wrong routing: `-4.3778` percentage points, a `100%` reduction.
- Refusal: `-23.0257` percentage points, approximately an `84.92%` reduction.

## Runtime and artifacts

- Launcher-to-result wall time: approximately 217.70 seconds (3 minutes 37.7 seconds).
- Allocation: approximately 0.484 GPU-hours on 8 GPUs.
- Result: `checkpoints/foundir_qwen_five_class_development_v1/result.json`.
- Result SHA-256: `36e625892118bae9bdedbfdb64ea941dda11ccb2c5288e82593a4815fb78ff84`.
- Predictions: `checkpoints/foundir_qwen_five_class_development_v1/predictions.csv`.
- Predictions SHA-256: `3445123a82782d40ad979dcaf0e6951ea74e0bfa404388944ae583fc44547b36`.
- Internal test and official FoundIR test were not accessed.

## Interpretation

The singleton Qwen routing stage is complete. Every accepted development record selected its correct frozen expert; ambiguous records failed closed. The five experts' complete direct development quality results remain the frozen restoration evidence and are not rerun or reselected here.
