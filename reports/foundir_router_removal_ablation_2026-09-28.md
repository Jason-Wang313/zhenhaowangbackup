# FoundIR router removal ablation

CPU-only reaggregation of 16,461 frozen development predictions; no training, inference, GPU, or official/sealed-image access.

| Variant | Coverage | Correct | Wrong | Abstain | Accepted error |
|---|---:|---:|---:|---:|---:|
| full_current_router | 82.6377% | 13,586 | 17 | 2,858 | 0.1250% |
| minus_eleven_class_head | 81.4835% | 13,388 | 25 | 3,048 | 0.1864% |
| minus_physical_guard | 82.6377% | 13,586 | 17 | 2,858 | 0.1250% |
| minus_eleven_and_physical | 83.7738% | 13,764 | 26 | 2,671 | 0.1885% |
| candidate_five_atomic_nonlinear_only | 83.1966% | 13,669 | 26 | 2,766 | 0.1899% |

## Definitions

- `full_current_router`: Frozen deployed final router.
- `minus_eleven_class_head`: Drop eleven-class agreement; retain Atomic/nonlinear route and physical agreement for Noise family.
- `minus_physical_guard`: Drop physical agreement for Noise family; retain deployed decisions elsewhere.
- `minus_eleven_and_physical`: Accept the frozen Atomic + nonlinear route directly.
- `candidate_five_atomic_nonlinear_only`: For singleton routes require five-class confirmation of the Atomic/nonlinear route; accept supported mixed Atomic/nonlinear routes directly.

## Limitations

- This reuses the inspected 16,461-record development set and is not an independent lockbox result.
- The requested simplified candidate changes the consensus wiring; it is not a pure one-component deletion.
- No downstream restoration inference was rerun, so this ablation measures routing behavior only.
