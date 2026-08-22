# FoundIR-Lowlight Gate 1 result

Date: 2026-08-22 UTC

## Outcome

`FOUNDIR_LOWLIGHT_GATE1_PASS`

The metric implementation, three-family reusable development suite, leakage
policy, released-teacher status, experiment ledger, and Gate 2 proposal lattice
are frozen. No protected FoundIR-Lowlight image was opened and no protected
inference was performed.

## Development suite

| Family / role | Records | Groups |
|---|---:|---:|
| LOL-v2 / outer development | 102 | 41 |
| SICE / training | 54 | 31 |
| SICE / outer development | 18 | 10 |
| SDSD / training | 375 | 125 |
| SDSD / outer development | 125 | 25 |
| **Total** | **674** | **232** |

All splits are scene- or sequence-disjoint. No input or target content hash
crosses training and outer roles. Outer strata are explicitly reusable
development sets, not new hidden tests. The metric contract evaluates saved,
reloaded uint8 RGB and conservatively takes the lower PSNR and higher LPIPS
across its two frozen paths.

## Teacher decision

The released generalist and documented generalist-to-Lowlight-specialist
cascade are admitted for internal non-FoundIR proposal measurement. Their
development-family training overlap is unverifiable, checkpoint-specific
redistribution terms are not separately stated, and runtime/memory have not yet
been measured. Therefore teacher-at-inference is not a deployable dependency at
Gate 1; Gate 2 must measure value and cost, and packaging still requires
clearance.

## Identities

- Combined manifest SHA-256: `01ea1bcc2c4a9166188c5852b20a45b7ce828b47c043b4ca1abbdc3aa994b200`
- Development contract SHA-256: `33963c5467285151e0cb2a5b9bc977762062173f36a31e92ccc8329882bb6e69`
- Teacher contract SHA-256: `d559fe04d000b316d413a991bb39e07596e217c8b7f655e7b84b76110c837418`
- Proposal contract SHA-256: `6d1fca9f956dfedc1ee8546e33c97e0a25775a486b4afb8b5235e396e60e2095`
- Experiment ledger SHA-256: `4848e7bca5054a57be3f58ef66ae21bee0983b944516cec5920978b711cede25`

The machine-readable result is `reports/foundir_lowlight_gate1_result_v1.json`.
