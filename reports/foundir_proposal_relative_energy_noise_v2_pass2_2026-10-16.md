# FoundIR proposal-relative energy Noise V2 — pass-2 interim

Date: 2026-10-16 UTC

Run `noise_energy_v2_formal1` reached the frozen second-pass checkpoint at
step `3,412` and continued normally beyond step `4,100`. The resumed node is
homogeneous `8 x H200`, global batch `32`. The fixed 256-record development
validation and checkpoint were written. No protected-data access was reported.

Higher PSNR and lower LPIPS are better. All rows use the same fixed development
records and the same frozen proposal.

| Arm | PSNR | LPIPS | PSNR versus proposal | LPIPS versus proposal |
|---|---:|---:|---:|---:|
| Frozen proposal | `43.123698368668556` | `.05036422960256459` | — | — |
| Direct refiner | `42.94062531739473` | `.040550720553255815` | `-.18307305127382278` | `-.009813509049308777` |
| Energy, 1 step | `43.12399557232857` | `.050377762617245025` | `+.0002972036600112915` | `+.000013533014680433553` |
| Energy, 2 steps | `43.12429237365723` | `.050391448035952635` | `+.0005940049886703491` | `+.000027218433388043195` |
| Energy, 3 steps | `43.12458612769842` | `.05040514755910408` | `+.0008877590298652649` | `+.00004091795653948793` |
| Adaptive proposal/1/2/3 selector | `43.12455575913191` | `.05040373227166128` | `+.0008573904633522034` | `+.000039502669096691534` |

The energy arms remain essentially identical to the proposal and slightly
worsen LPIPS. The direct refiner improves LPIPS but lowers PSNR. No arm meets
the predeclared `+.30 dB / -.005 LPIPS` gate at pass 2. This is an interim
result; the frozen five-pass run continues.

Artifact SHA-256: checkpoint
`b8db24cabdc8928b229a51f480aff1bf6addfab98a276196466381a59783b3ac`;
validation JSON
`832202d5a97a8a21dc9106033973c54d607c78e2f8115a8efcf529f9c8105cef`.
