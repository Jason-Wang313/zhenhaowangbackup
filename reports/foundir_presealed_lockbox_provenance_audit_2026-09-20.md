# FoundIR pre-sealed lockbox provenance audit

Date: 2026-09-20 UTC

## Finding

A genuine native FoundIR 11-class lockbox cannot be constructed from the currently authorized corpus without misrepresenting prior data use.

- The deployed Blur expert trained on all 7,339 Blur training source groups.
- The deployed Rain expert trained on all 5,780 Rain training source groups.
- The complete 11-class FoundIR validation partition has already been read by historical component evaluation and router development.
- The internal/sealed/official partitions remain prohibited and were not accessed.

Therefore, no currently authorized native subset can simultaneously cover all 11 classes and be unseen by training, checkpoint/alpha/router selection, and historical evaluation. Existing external synthetic sources were also used in earlier Qwen development and do not reproduce the native FoundIR distribution closely enough to support a sealed-test pass claim.

## Authorized substitute

The project freezes a final native development rehearsal, explicitly not an independent lockbox:

- 11 classes × 128 records = 1,408 records;
- source-group round-robin selection independent of router status;
- 440 source groups;
- 1,247 correct routes, 1 wrong route, and 160 abstentions;
- correct and wrong accepted routes execute the router-selected expert path;
- abstentions return the input unchanged and remain in PSNR/LPIPS;
- all models, checkpoints, alpha values, expert order, and routing decisions stay frozen;
- no tuning is authorized after this result;
- no internal, sealed, or official test access.

This rehearsal measures real end-to-end behavior and the metric cost of abstention. It is descriptive evidence for an advisor request, not an unbiased estimate or guarantee of official-test performance.
