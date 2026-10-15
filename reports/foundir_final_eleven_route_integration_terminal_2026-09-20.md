# FoundIR 11-class end-to-end integration terminal result

Date: 2026-09-20 UTC

## Scope

- Development split only: 176 records, 16 per class across all 11 classes.
- Frozen router decisions and frozen singleton/mixed restoration paths.
- Routed execution compared with direct expert execution using identical checkpoints, seeds, scalar residual strengths, crop coordinates, inter-stage uint8 PNG quantization, and expert order.
- No model-weight update.
- No FoundIR official, sealed, or internal test access.

## Result

- Integration fidelity: **PASS**.
- Stage PNG pairs compared: 272.
- Byte-exact stage pairs: 256.
- Restormer numeric-tolerance stage pairs: 16, all Rain singleton.
- Metric sidecar pairs compared: 176.
- Exactly equal metric pairs: 160.
- Restormer numeric-tolerance metric pairs: 16, all Rain singleton.
- True mismatches: 0.

The 16 Rain reruns differed only because of GPU Restormer numerical rounding:

- maximum pixel error: 1/255;
- maximum changed-pixel fraction: 0.6269%;
- no pixel error exceeded 1;
- maximum per-record PSNR absolute difference: 0.002018 dB;
- maximum per-record LPIPS absolute difference: 0.00005761.

The frozen Restormer-only tolerance is deliberately bounded at max pixel error 1/255, changed-pixel fraction 1%, PSNR difference 0.005 dB, and LPIPS difference 0.0001. FLUX paths remain byte-exact. These bounds were not applied to semantic routing errors or model-quality failures.

## Interpretation

The routed pipeline reproduces direct expert invocation within strict deterministic/numerical limits. No wrong checkpoint, alpha, crop, file path, quantization boundary, or mixed-expert order was detected.

The legacy evaluator printed `pass:false` because it still contains the withdrawn requirement that every 16-image category sample improve mean PSNR over its input. That is not the integration criterion and was not used to pass this gate. The terminal criterion is routed-versus-direct fidelity recorded in `checkpoints/foundir_final_eleven_route_integration_v1/result.json`.

The successful rerun began at 00:16:21 UTC and the original GPU runner reached its strict byte comparator at 00:31:35 UTC (15 minutes 14 seconds). The tolerance-aware CPU comparator then established the terminal pass without additional GPU work.

## Governance

This result does not authorize access to the FoundIR official/sealed test. Such access remains prohibited until an advisor gives explicit, auditable approval.
