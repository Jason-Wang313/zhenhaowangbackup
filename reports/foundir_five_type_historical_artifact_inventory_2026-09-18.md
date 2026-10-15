# FoundIR five-type historical artifact inventory and step-2 selection

Date: 2026-09-18

## Scope and stop condition

This is the mandatory pre-GPU historical inventory and best-existing-eligible-
artifact selection for isolated Blur, Noise, Haze, Rain, and Low-light. It completes
steps 1 and 2 of the repository policy and deliberately stops there. It does
not add or test new routing, access sealed internal or official FoundIR test
data, launch evaluation, or launch training.

Eligibility is intentionally narrower than "any checkpoint in the repository":
released FoundIR generalist/specialist weights are baselines and cannot be
selected as our restoration result. Blur, Noise, Haze, and Low-light therefore
use their FLUX results. Rain retains the project-trained Rain-only Restormer
that the user already explicitly selected; it is not a released FoundIR
generalist/specialist checkpoint.

All selected metrics below come from the frozen FoundIR `validation` split.
The split was constructed with GT/component source grouping, so the reported
development records are source-disjoint from training. Percentage PSNR margin
is `(result - SOW minimum) / SOW minimum`; percentage LPIPS safety is
`(SOW maximum - result) / SOW maximum`.

## Unambiguous classification

| Type | Selected existing result | Stage-2 SOW | Absolute margins | Relative margins | Classification |
|---|---:|---:|---:|---:|---|
| Blur | 26.559141 / .151993 | 25.9 / .18 | PSNR +.659141 dB; LPIPS safety .028007 | +2.5449%; 15.5594% | **Pass, but PSNR margin is thin; bounded-improvement candidate** |
| Noise | 40.150553 / .068880 | 38.9 / .10 | PSNR +1.250553 dB; LPIPS safety .031120 | +3.2148%; 31.1196% | **Safe** |
| Haze | 23.311888 / .101155 | 22.1 / .19 | PSNR +1.211888 dB; LPIPS safety .088845 | +5.4837%; 46.7605% | **Robust aggregate pass; visual/tail safety still fails** |
| Rain | 42.284347 / .047452 | 34.2 / .08 | PSNR +8.084347 dB; LPIPS safety .032548 | +23.6384%; 40.6848% | **Safe** |
| Low-light | 19.779135 / .154058 | 18.9 / .28 | PSNR +.879135 dB; LPIPS safety .125942 | +4.6515%; 44.9792% | **Safe on the complete development set** |

Therefore all five eligible results already pass both contractual mean SOW
metrics. Blur has thin PSNR evidence. Haze now has a robust full-development
metric pass but remains a possible bounded-improvement type because its fixed
visual/near-clean tail gate fails. Noise, Rain, and
Low-light have sufficient two-metric margins and must not receive new training
merely to make already-passing averages larger.

## Selected artifacts and evidence

### Blur — isolated FLUX step 10k plus frozen scalar residual

- checkpoint:
  `checkpoints/foundir_blur_only_flux2_full_v1/step_010000.safetensors`
- checkpoint SHA-256:
  `e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`
- inference rule: `LQ + .75 * (FLUX - LQ)`; no weight update
- evidence: 2,595 records / 165 source groups, 512 aligned crops, official
  50-step schedule, fixed sixteen-image visual review passed
- result SHA-256:
  `5b9dac81e11d338d17a36c24bba90f8e5c08299727d15883ebee0d5129e25daf`
- important caveat: the SOW mean passes, but the predeclared conservative tail
  gate fails. PSNR wins versus the unblended base on 89.56%; 32 records regress
  by more than 1 dB, three regress by more than 2 dB, and the worst regression
  is 12.569169 dB. This makes Blur one of the two thin/bounded-improvement
  types, together with Haze.

The older routed full-development system reached 28.504131/.171123, but it is
not selected here: its LPIPS safety is only 4.9314%, its source-group bootstrap
fails the SOW, and its category result mixes several router decisions. The
isolated scalar artifact has the more balanced standalone SOW profile.

### Noise — isolated FLUX step 400 plus frozen scalar residual

- checkpoint:
  `checkpoints/foundir_four_task_flux2_capacity_sanity_v1/noise/step_000400.safetensors`
- checkpoint SHA-256:
  `1b4ed05ad5ad403a3d7fcc633eca97cdb69d00454e357da0c3d3131ac2c3ff1a`
- inference rule: `LQ + .30 * (FLUX - LQ)`; no weight update
- evidence: 1,859 records / 34 source groups on the complete isolated-Noise
  development set
- result SHA-256:
  `527494e719564c159b3e9932f4d36ad8facfbc5d308801081c7c36c6b9f87e25`
- caveat: the mean SOW metrics are safe, but near-clean input tails regress:
  218 records lose more than 2 dB versus input and the worst loses 10.984566
  dB. This is a tail/routing concern, not a thin two-metric SOW margin.

The stronger historical Noise Restormer is recorded as prior project evidence
but is not used to replace the already-completed FLUX Noise selection here.

### Haze — isolated FLUX LoRA step 3000

- checkpoint:
  `checkpoints/foundir_haze_flux2_source_diverse_continuation_v1/step_003000.safetensors`
- checkpoint SHA-256:
  `4d94ecf6dd902704345e28ae0bf3447d7e172235158c2b2177933794fd973e00`
- evidence: complete 1,580-record / 158-source-group validation split,
  disjoint from 76,650 training records / 7,642 source groups; official
  50-step inference
- result SHA-256:
  `66e8a552558f69c7e24830928bed2e9c13cae094e4ec3ecfa8f319d765f733fe`
- 95% equal-source-group bootstrap remains within SOW: PSNR lower 22.882761
  and LPIPS upper .108192
- caveat: both SOW means and bootstrap pass, but 70/1,580 records regress more
  than 2 dB versus input and one of sixteen fixed visuals fails decisively.
  Haze remains a bounded-improvement candidate for near-clean identity safety,
  not because its aggregate SOW evidence is thin.

The released FoundIR generalist plus weather specialist result
23.386211/.147577 is explicitly **ineligible** for selection. It remains only
a baseline/reference and is not our FLUX result.

### Rain — task-native Restormer step 5000

- checkpoint:
  `checkpoints/foundir_five_type_flux2_dit_v3/rain_restormer_adaptation_v2_8gpu/step_005000.pt`
- checkpoint SHA-256:
  `ff829573175df0f25add8a4d8f82db648fd6dad88a261297cfe88d2a8944c444`
- evidence: 835 records / 117 source groups; selected on development
- selection SHA-256:
  `924f64288198d6ebc1aa8b3ef8b744865f260d679b989633b3544a43dce70bc4`
- 95% source-group bootstrap: PSNR lower 38.440023 and LPIPS upper .070381;
  both safely pass the actual SOW 34.2/.08 bounds
- caveat: the old stronger gate used an LPIPS bootstrap limit of .07000, so it
  failed by .000381. That does not constitute an SOW failure.
- status: safely solved; no more Rain training or evaluation is justified.

### Low-light — isolated FLUX step 400 plus frozen scalar residual

- checkpoint:
  `checkpoints/foundir_four_task_flux2_capacity_sanity_v1/lowlight/step_000400.safetensors`
- checkpoint SHA-256:
  `592e49ca2606565c5aeec8aadeb5466cf0c34b30024d5c02f0f2292ad28c90d8`
- inference rule: `LQ + .925 * (FLUX - LQ)`; no weight update
- evidence: complete 1,400-record / 21-source-group isolated Low-light
  development set; all sixteen prospectively fixed visuals passed
- result SHA-256:
  `2673ace6eac831b06684826cd732fbd14c57974eef81eb86e0260cf931bd93ca`
- caveat: 16 records regress by more than 2 dB versus input and only 21 source
  groups exist. Nevertheless the full-development PSNR margin is .879135 dB,
  LPIPS safety is large, and both metrics improve materially over paired FLUX.
  It is selected as safe rather than reopened for marginal mean optimization.

The released FoundIR generalist plus low-light specialist result
26.090304/.116811 is explicitly **ineligible** for selection. It remains only
a baseline/reference and is not our FLUX result.

## Data-access and selection boundaries

- No official FoundIR test data were accessed.
- No sealed internal result was opened or used for this selection.
- All selection evidence is development/validation only.
- Released FoundIR generalist/specialist results were inspected only to detect
  and exclude them; they are not selected project results.
- No GPU work occurred for this inventory.

## Exact stop state

Step 1, historical inventory: complete. Step 2, best existing artifact
selection: complete. Step 3, routing/integration: intentionally not started in
response to the user's instruction. The only possible future quality work is
one bounded Blur improvement and one bounded Haze improvement; either requires
a separate decision after this stop.
