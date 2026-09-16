# Four-task FLUX.2 capacity sanity terminal audit

## Outcome

The four independent arms completed all 400 optimizer steps and all frozen train/validation evaluations on one homogeneous 8×H100 host. Start was `2026-09-16T13:12:08Z`; terminal marker was `2026-09-16T13:30:07Z`; complete end-to-end wall time was **17 minutes 59 seconds**. There was no OOM, NCCL, NaN, model-loading, or artifact failure. Torchrun status 1 for each arm only propagates the intentional scientific-failure exit after `result.json` was written.

Every task has four LoRA checkpoints, four optimizer states, 192 triptych PNGs, 12 metric JSON files, and a terminal result. The aggregate result SHA-256 is `94e7740268d8aff9328b177042e68329c82c48bdfcde3ad20cf40ee7db580fbe`.

## Step-400 result

| Task | Train input → output PSNR | Train input → output LPIPS | Descriptive validation input → output | Formal capacity gate |
|---|---:|---:|---:|:---:|
| Noise | 40.458 → 37.655 | .0954 → .0251 | 40.346/.1038 → 34.358/.0512 | fail |
| Haze | 15.020 → 29.104 | .2985 → .0497 | 13.881/.3249 → 19.956/.1381 | fail |
| Rain | 29.531 → 33.483 | .2574 → .0438 | 29.151/.2514 → 30.716/.0815 | fail |
| Low-light | 8.190 → 26.352 | .5383 → .0548 | 9.106/.5159 → 17.483/.1925 | fail |

All four fail the predeclared aggregate gate and must remain labeled failures. The failure modes are not equivalent:

- **Haze and Low-light:** four of five capacity checks pass. Both train SOW proxies pass PSNR and LPIPS. Their only automatic failure is the probe-loss ratio (`.853` and `.791` versus required `.35`). Validation improves both metrics strongly, but remains below the PSNR SOW thresholds. These are strong positive capacity signals, not permission to relabel the gate.
- **Rain:** average train restoration is strong and LPIPS passes the train proxy, but train PSNR is `33.483`, `0.717 dB` below 34.2; only 56.25% of train records improve PSNR, and probe ratio is `.715`. Validation is close in LPIPS (`.0815` versus `.08`) but PSNR remains `3.484 dB` below 34.2. Capacity is promising but nonuniform.
- **Noise:** substantive fidelity failure. Input already exceeds the PSNR target, while the trained model loses `2.803 dB` on train and `5.988 dB` on validation. It improves LPIPS strongly but overprocesses mild/near-clean records. This task needs identity-preserving strength control/correction before joint training can be considered robust.

## Visual audit

All 128 final step-400 train/validation `Input | Output | GT` triptychs were reviewed. The overall visual gate fails. Haze and Low-light show coherent, large restoration improvements on severe examples but have exposure/color outliers. Rain removes strong rain well but can damage mild cases. Noise frequently smooths or shifts already-good inputs. Hash-bound findings are in `reports/foundir_four_task_flux2_capacity_sanity_visual_review_v1.json`.

## Scope and next decision

No Qwen degradation conditioning, mixed degradation, pixel refiner, development-GT optimization, sealed internal, or official FoundIR test was used. Balanced five-task joint training has not started. The active terminal hold is `flux2_four_task_capacity_sanity_result-323541cca1c3b528`; further experiments remain stopped until a matching recorded decision.
