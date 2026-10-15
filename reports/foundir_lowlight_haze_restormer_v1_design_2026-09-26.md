# Low-light+Haze direct Restormer v1 frozen design

- Scope: improve only `09Lowlight_Haze`; preserve the other four Double categories unchanged.
- Architecture: project-owned direct Restormer mapping mixed LQ to clean GT.
- Initialization: public task-native Restormer `deraining.pth`; no released FoundIR weights.
- Isolation: every source group represented by the frozen 660-image comparison is excluded from train and development validation.
- Training: deterministic source-group-balanced crops with geometric augmentation.
- Development selection: 128 source-disjoint, full-resolution images, evaluated by tiled inference at steps 1000, 3000, and 6000.
- Final comparison: the frozen 60-image `09Lowlight_Haze` subset is evaluated once after checkpoint selection; it is neither gradient data nor checkpoint-selection data.
- Metrics: full-image RGB uint8 PSNR and area-weighted non-overlapping 512-tile LPIPS, matching the existing comparison protocol.
- Target: exceed Full FoundIR `20.939309 / .159126` on the frozen 60-image category.
- Stop rule: one bounded run; no automatic hyperparameter search and no automatic Low-light+Blur follow-on.
- User authorization: `yes, restormer, execute` (2026-09-26 conversation).

This is a model-family/architecture change and therefore requires a matching governance decision before any GPU work begins.
