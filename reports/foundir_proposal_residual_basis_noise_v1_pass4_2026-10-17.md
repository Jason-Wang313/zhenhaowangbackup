# FoundIR proposal residual-basis Noise V1 pass 4 — 2026-10-17

## Outcome

The live `formal1` run completed pass 4 at step `6,824` on `8 x H100 80GB`.
Fidelity continues to improve both metrics but has entered a PSNR plateau and
has not crossed the joint gate. Complete the frozen fifth pass unchanged.

## Development validation

| Arm | PSNR | PSNR gain | LPIPS | LPIPS reduction |
|---|---:|---:|---:|---:|
| Proposal | `43.124160178005695` | `0` | `.050362067760033824` | `0` |
| Fidelity | `43.365845650434494` | `+.24168547242879868` | `.02899720575715037` | `+.021364862002883456` |
| Midpoint | `43.28140088915825` | `+.15724071115255356` | `.026463300659997913` | `+.02389876710003591` |
| Perceptual | `43.170626521110535` | `+.046466343104839325` | `.02646822422502737` | `+.023893843535006454` |

Positive `LPIPS reduction` means lower and better LPIPS.

## Interpretation

- Fidelity still satisfies the LPIPS target by a wide margin.
- Fidelity needs another `.05831452757120131 dB` PSNR to clear the joint gate.
- Its pass-3 to pass-4 PSNR increment was only
  `+.011031992733478546 dB`, compared with `+.1251758486032486 dB` from pass
  2 to pass 3. The late-stage curve has therefore flattened materially.
- Pass 4 is the first full pass at 256 crop and runs at an already-low learning
  rate near `6e-6`, which will decay toward `1e-6` in the final pass. A large
  last-pass jump should not be assumed.

The step-6,824 checkpoint and validation were written on 2026-10-17. All
protected-data access flags remain false. Continue to the frozen endpoint at
step `8,530` without changes.
