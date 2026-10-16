# FoundIR proposal-relative energy Noise V2 — pass-3 interim

Date: 2026-10-16 UTC

Run `noise_energy_v2_formal1` completed its third effective pass at step
`5,118` and continued to step `6,635/8,530` as of `19:09 UTC`. It runs on a
single homogeneous `8 x H200` node with global batch `32`. This is the same
fixed, source-disjoint 256-record development validation used at prior passes;
it is not the official, sealed, internal, or frozen 660 set. All protected-data
access flags in the validation artifact are `false`.

Higher PSNR and lower LPIPS are better. Every row uses the same records.

| Arm | PSNR | LPIPS |
|---|---:|---:|
| Frozen proposal | `43.123698368668556` | `0.05036422960256459` |
| Adaptive proposal/1/2/3 selector | `43.12445606291294` | `0.05040028668190644` |
| Energy, 1 step | `43.123958714306355` | `0.05037631940376741` |
| Energy, 2 steps | `43.124218851327896` | `0.05038842481098982` |
| Energy, 3 steps | `43.12447692453861` | `0.050400731362969964` |
| Direct refiner | `42.9707455933094` | `0.04185937133115658` |

The energy arms are essentially unchanged from the proposal and slightly
worsen LPIPS. The direct refiner lowers LPIPS but also lowers PSNR. No arm
simultaneously meets the predeclared `+.30 dB` PSNR and `-.005 LPIPS` gate.
This is an interim result. Continue the frozen five-pass run without retuning
or early stopping; assess the final checkpoint only after its validation.

SHA-256 of `step_005118.pt`:
`b9444ff2c3d4dedfbad2022c4d3dbb74b193fd6e64d2dcf52cb33275fa438bd1`.
SHA-256 of `validation_step_005118.json`:
`1a19ba953edc76d7a9d58b871082ed29327c363276b0f41f34c0826292e762e0`.
