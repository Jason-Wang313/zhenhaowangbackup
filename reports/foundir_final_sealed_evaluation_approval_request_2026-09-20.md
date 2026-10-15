# Request for one-time FoundIR sealed-evaluation authorization

Date: 2026-09-20 UTC

## Decision requested

Please authorize one terminal, read-only evaluation of the exact frozen FoundIR
11-class restoration pipeline on the official/sealed test partition.

The requested run will:

1. use the frozen router, five singleton experts, and six mixed restoration
   paths identified below;
2. compute PSNR and LPIPS with each test record receiving equal weight, subject
   to confirmation that this is the official SOW aggregation rule;
3. preserve the input unchanged whenever the router abstains;
4. make no checkpoint, alpha, threshold, expert-order, prompt, seed-policy, or
   post-processing selection from sealed results;
5. report the result once, including failures, and terminate without tuning on
   the sealed partition.

No FoundIR official-test pixels, labels, or metrics were re-accessed to prepare
this request. A historical data-integrity operation on 2026-08-16 did download,
extract, decode, dimension-check, and hash all 1,500 official pairs. It ran no
restoration model, computed no model performance, and selected no settings.
The official set therefore remains unused for model-performance evaluation,
but it is not literally byte-level untouched.

## Frozen singleton restoration evidence

All values below are record-weighted means on complete source-disjoint native
development partitions. All five experts are project-trained artifacts, not
released FoundIR restoration specialists.

| Type | Records / source groups | Frozen backend | PSNR / LPIPS | SOW margin |
|---|---:|---|---:|---:|
| Blur | 2,595 / 165 | FLUX.2 step-10k, `LQ + .75*(FLUX-LQ)` | 26.559141 / .151993 | PSNR +2.5449%; LPIPS headroom 15.5594% |
| Noise | 1,859 / 34 | FLUX.2 step-400, `LQ + .30*(FLUX-LQ)` | 40.150553 / .068880 | PSNR +3.2148%; LPIPS headroom 31.1196% |
| Haze | 1,580 / 158 | FLUX.2 step-3k | 23.311888 / .101155 | PSNR +5.4837%; LPIPS headroom 46.7605% |
| Rain | 835 / 117 | project-trained Rain Restormer step-5k | 42.284347 / .047452 | PSNR +23.6384%; LPIPS headroom 40.6848% |
| Low-light | 1,400 / 21 | FLUX.2 step-400, `LQ + .925*(FLUX-LQ)` | 19.779135 / .154058 | PSNR +4.6515%; LPIPS headroom 44.9792% |

Evidence registry:
`reports/foundir_five_type_source_disjoint_development_registry_v1.json`
(SHA-256 `225004da30a29854423ef1e921787830344d3af824d1b213900c0481c5328e47`).

The final 11-class router is more conservative than the singleton-only router.
Recombining its 2,223 correct Blur decisions and 372 abstentions with all 2,595
frozen Blur outputs gives **26.614726 PSNR / .156974 LPIPS**. This passes the
Blur SOW 25.9/.18 with PSNR margin **+.714726 dB / +2.7596%** and LPIPS
headroom **.023026 / 12.7920%**.

Machine-readable evidence:
`checkpoints/foundir_blur_record_weighted_final_eleven_router_v2/result.json`
(SHA-256 `c30b43fe85edb2a8698d2178079ffc2adcacd9932e3c6b285ee805ff4ffb0423`).

## Frozen mixed restoration paths

The six mixed categories use these fixed paths:

- Blur+Noise -> paired Blur+Noise Restormer;
- Haze+Low-light -> Haze, then Low-light;
- Rain+Haze -> Rain, then Haze;
- Blur+Low-light -> Low-light, then Blur;
- Noise+Low-light -> Noise, then Low-light;
- Blur+Noise+Low-light -> Noise, then Low-light, then Blur.

Inter-stage images are quantized to uint8 PNG-equivalent pixels. No order or
strength is selected at test time. A routed-versus-direct integration audit on
176 development records (16 per class) found zero true execution mismatches:
all FLUX paths were byte-exact, and Rain Restormer reruns stayed within the
predeclared one-code-value numerical tolerance.

Evidence: `reports/foundir_final_eleven_route_integration_terminal_2026-09-20.md`.

## Router evidence and unresolved risk

The final fail-closed 11-class router was evaluated on 16,461 native
development records:

- correct accepted: 13,586;
- safe abstentions: 2,858;
- wrong accepted: 17;
- coverage: 82.6377%;
- wrong-route rate: .1033%.

The 17 errors are confined to mixed boundaries: five Blur+Low-light records
and twelve Noise+Low-light records. Therefore the predeclared strict
**zero-wrong-route gate did not pass**. This request does not conceal or relabel
that failure. We ask the advisor to decide explicitly whether the observed
.1033% development wrong-route risk is acceptable for a terminal sealed
evaluation. If it is not acceptable, please deny authorization and specify the
required router criterion; the sealed partition will remain untouched.

Frozen decisions:
`checkpoints/foundir_final_eleven_route_cpu_gate_v1/development_decisions.csv`
(SHA-256 `0342bf00bef98eccb9c3d3ad8e3c42c2368db2761f3c2ddd38f9b1342ad31225`).

## Final all-in-one development rehearsal

A 1,408-record native rehearsal (11 classes x 128 records) executed the frozen
end-to-end system with 1,247 correct routes, one wrong route, 73 identity
abstentions, and 87 low-light partial-restoration fallbacks. It is
source-group-balanced and intentionally stresses rare hard sources. It is not
used as a substitute for record-weighted SOW estimates and is not represented
as an independent lockbox.

In particular, source-group-balanced Blur fails while the complete
record-weighted 2,595-record Blur evaluation passes. Across the same historical
Blur evidence, changing only the aggregation from record weighting to equal
source-group weighting changes the result from 26.559141/.151993 to
23.104172/.226661. This distribution sensitivity is disclosed as a sealed-test
risk, not tuned away.

Rehearsal result:
`checkpoints/foundir_final_eleven_all_in_one_development_rehearsal_v3/terminal_result.json`
(SHA-256 `5e56a69a19ff1ed63ab78176799d93f6df62258799a46e58e1a111acec14c024`).

## Why no independent 11-class lockbox is claimed

A provenance audit established that a genuinely untouched native 11-class
lockbox cannot be constructed from the currently authorized data: historical
component and router work has already used the available validation sources,
while all Blur and Rain training source groups were consumed by their selected
experts. The official test remains the only native final evaluation unused for
model-performance measurement. It underwent the disclosed 2026-08-16 integrity
validation but has never been used for restoration inference, model or
parameter selection, or model-performance metrics.

Evidence: `reports/foundir_presealed_lockbox_provenance_audit_2026-09-20.md`.

## Requested authorization wording

Please respond with one of the following auditable decisions:

- **APPROVED:** authorize one execution of the frozen 11-class pipeline on the
  FoundIR official/sealed test, using official record-level PSNR/LPIPS
  aggregation, with no post-result tuning or rerun; or
- **DENIED:** do not access the sealed partition, and state the remaining
  prerequisite, especially whether the .1033% development wrong-route rate or
  the aggregation protocol is unacceptable.

Until an explicit approval is recorded, no official/sealed test command will
be run.
