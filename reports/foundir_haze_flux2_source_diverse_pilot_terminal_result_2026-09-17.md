# FoundIR Haze FLUX.2 source-diverse pilot terminal result — 2026-09-17

The isolated-Haze FLUX.2 Klein Base 4B fresh rank-32 LoRA pilot completed
2,000 optimizer steps on 1,580 independent training source groups and evaluated
the prospectively frozen 128 source-group-disjoint validation records. Sealed
internal and official FoundIR test data were not accessed.

The frozen validation result is **21.969477 PSNR / 0.111869 LPIPS** against
Stage-2 SOW **PSNR >=22.1 / LPIPS <=0.19**. PSNR misses by **0.130523 dB**, or
**0.5906%** relative to the floor. LPIPS has **0.078131** safety margin, or
**41.1218%** relative to the cap. PSNR improves over input on 92.1875% of
records and LPIPS improves on 96.8750%.

The fixed sixteen-record curve remains positive through the last milestone:
step 0 is `14.669386/.333913`, step 1000 is `20.992393/.131172`, and step 2000
is `21.541213/.123312`. From step 1000 to 2000, PSNR improves 0.548820 dB and
LPIPS improves 0.007861.

The dominant identity failure is `foundir_train_0206837`: input
`34.821428/.018172` becomes `15.029782/.150632`. Replacing only this output by
identity would raise the 128-record mean PSNR to 22.124099, above the SOW
floor. This is descriptive counterfactual evidence, not a tuned gate.

All computation completed and the step-2000 checkpoint, optimizer state,
result, and 128 outputs are intact. Start/end were `2026-09-17T07:08:26Z` and
`2026-09-17T07:52:03Z`: 43m37s end-to-end on 8xH100, approximately 5.82
GPU-hours. The old runner incorrectly classified torchrun's normalized
scientific-gate exit as runtime failure after result creation. This was a
post-result status-normalization defect, not a training/runtime crash.

User-authorized next action: resume the exact step-2000 checkpoint and optimizer
on the identical 1,580 sources for 1,000 additional steps to absolute step
3000, then run only the frozen final validation. No new data, module, loss,
Qwen degradation condition, mixed degradation, sealed evaluation, official
test, or automatic Rain launch is authorized.
