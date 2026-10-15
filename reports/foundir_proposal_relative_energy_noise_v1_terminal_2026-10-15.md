# FoundIR proposal-relative energy Noise V1 — terminal result

Date: 2026-10-15 UTC

## Outcome

Run `noise_energy_v1_pass1` completed the frozen five-pass endpoint at `8,530`
optimizer steps on one homogeneous `8 x NVIDIA H100 80GB` node. It completed
without OOM, NCCL, non-finite, or protected-data-access failure.

V1 does **not** pass the method gate. Its best final energy arm improves PSNR
by only `+.10285484790802033 dB` and worsens LPIPS by
`+.006770727247385367`. The predeclared gate requires at least `+.30 dB` PSNR
and `-.005 LPIPS` relative to the unchanged proposal.

## Final fixed 256-record source-disjoint validation

Higher PSNR is better; lower LPIPS is better.

| Arm | PSNR | Change from proposal | LPIPS | Change from proposal |
|---|---:|---:|---:|---:|
| Frozen proposal | `43.124160178005695` | — | `.050362067760033824` | — |
| Direct refiner | `43.36069244146347` | `+.23653226345777312` | `.05899635784771817` | `+.008634290087684346` |
| Energy, 1 step | `43.227015025913715` | `+.10285484790802033` | `.05713279500741919` | `+.006770727247385367` |
| Energy, 2 steps | `43.18365592509508` | `+.05949574708938332` | `.05904203034606326` | `+.008679962586029436` |
| Energy, 3 steps | `42.947890132665634` | `-.17627004534006074` | `.05569961660330591` | `+.005337548843272088` |

The result confirms the V1 failure pattern: pixel fidelity can move slightly
while perceptual quality gets worse, and additional untrained inference steps
drift rather than reliably improve the proposal. V1 stops here and must not be
extended.

## Runtime and artifact identity

- wall-clock time: `3.1357351445996513 hours`;
- allocated GPU-hours: `25.08588115926418`;
- final checkpoint SHA-256:
  `57cbb42cc0dcbffd105727c6d727e84b0c61ce61d195c23600d2b545ef0910bf`;
- final validation SHA-256:
  `2027e46a94f85e7dd8070f5a12c0fd0e43712919282920cc07e7389a61dcd16d`;
- terminal result SHA-256:
  `587899d50b92ea131d1c146583258088db5b9b00e6b178f9adaf508a71e0546b`.

All protected-data flags are false.
