# Mandatory FoundIR pre-GPU artifact inventory policy

Effective immediately, no new FoundIR GPU training or evaluation may be
designed or launched until the repository's existing artifacts have been
audited for the degradation type in scope.

This policy exists because the repository already contained a completed
Rain-only Restormer step-5000 checkpoint with 835-record validation metrics of
42.2843 PSNR / .04745 LPIPS. Those metrics exceeded the Rain SOW of 34.2 / .08,
but later FLUX Rain experiments were run before this artifact was rediscovered.

## Required audit before any GPU proposal

For each of Blur, Noise, Haze, Rain, and Low-light, the audit must search and
hash-bind:

1. configs, checkpoints, optimizer states, result JSON files, reports, logs,
   launchers, and handoff records;
2. the exact data split, record count, source-group count, crop/inference
   protocol, and whether the result was used for checkpoint selection;
3. PSNR and LPIPS, including absolute SOW margins and percentage margins;
4. tail/visual/bootstrap failures separately from the contractual mean SOW;
5. whether the checkpoint is deployable, capacity-only, or merely missing a
   routing/integration step;
6. whether sealed internal or official FoundIR test data were accessed;
7. whether an existing model can be promoted through routing/tests without
   additional training or evaluation.

A GPU proposal is allowed only when the audit identifies a concrete missing
capability that cannot be satisfied by an existing immutable artifact. The
proposal must cite the audit and explain why reuse, promotion, routing, or a
CPU-only diagnostic is insufficient.

## Required order of work

1. Complete the five-type historical artifact inventory.
2. Select the best already-validated checkpoint per isolated degradation.
3. Add fail-closed routing and tests so an expert cannot affect another type.
4. Reuse existing validation evidence when its protocol matches the target.
5. Run new GPU work only for a type whose evidence still has a real gap.

## Joint-training decision

Five-type single-degradation joint training is **not** the next default step.
It would spend GPU time, re-open already solved tasks, and introduce
cross-task interference. It is considered only if the product explicitly
requires one shared checkpoint and the inventory proves that a routed set of
existing experts is unacceptable. Otherwise, independent best checkpoints
plus strict routing are the deployment design.

## Current inventory status

| Type | Historical audit | Immediate action |
|---|---|---|
| Rain | Selected step-5000 Restormer: 42.2843 / .04745 on 835 validation records | Safe; freeze; no more Rain quality work |
| Blur | Selected isolated FLUX step-10k plus frozen alpha .75: 26.5591 / .15199 on 2,595 validation records | SOW pass but PSNR margin thin; bounded-improvement candidate |
| Noise | Selected isolated FLUX step-400 plus frozen alpha .30: 40.1506 / .06888 on 1,859 validation records | Safe; freeze |
| Haze | Selected isolated FLUX LoRA step-3000: 23.3119 / .10116 on all 1,580 validation records | Mean and bootstrap SOW pass; fixed visual/near-clean tail fails; bounded-improvement candidate only for identity safety |
| Low-light | Selected isolated FLUX step-400 plus frozen alpha .925: 19.7791 / .15406 on 1,400 validation records | Safe on complete development; freeze |

The consolidated evidence, identities, percentage margins, caveats, and exact
step-2 stop state are recorded in
`reports/foundir_five_type_historical_artifact_inventory_2026-09-18.md`.
Steps 1 and 2 are complete. Per the user's instruction, step 3 routing is not
started by this audit.

Released FoundIR generalist/specialist checkpoints are baseline-only and are
ineligible as project selections. In particular, the historical Haze
23.3862/.14758 and Low-light 26.0903/.11681 routed results must not be reported
as FLUX achievements.

This document is a workflow gate. “Try a small run first” does not bypass it.
