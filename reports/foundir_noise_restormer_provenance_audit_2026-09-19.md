# Noise Restormer provenance audit

Date: 2026-09-19

The historical Noise Restormer selected for the bounded Noise-family compatibility check is **not** a released FoundIR baseline or specialist.

## Selected project checkpoint

- Path: `checkpoints/foundir_five_type_flux2_dit_v3/noise_restormer_adaptation_v1/step_000250.pt`
- SHA-256: `b3fca23f8271477834c6b3eba2a6dfdf6fcbd480e53c43570e42960a13044429`
- Embedded status: `foundir_task_native_adaptation_checkpoint`
- Embedded task: `noise`
- Embedded step: `250`
- Embedded config SHA-256: `2ff7937ecf64367db21aa014acb4d95a8db9249486619fc3eedbb37114090ce7`
- Embedded protected/internal-test access flags: both `false`

This checkpoint was produced by this project by adapting the upstream task-native Restormer on FoundIR **training-only** Noise pairs.

## Parent checkpoint

- Upstream architecture: `swz30/Restormer`
- Immutable architecture revision: `68dc6ac472db26f16361150cb7a96a1bc87da93f`
- Parent task checkpoint: `models/restormer-task-native/real_denoising.pth`
- Parent SHA-256: `4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387`
- Parent size: `104,611,957` bytes

## Released FoundIR checkpoints explicitly excluded

| Artifact | SHA-256 | Size |
|---|---|---:|
| FoundIR generalist | `2603a0be47d0844d4c9cea542ecca039e0e4aaa39d45d41064bfbdc13c0830c9` | 580,829,530 |
| FoundIR low-light specialist | `f834248e145d020d5a02f257b7ceb383b579add2522932938b3e1e0bd460abe6` | 6,294,432 |
| FoundIR weather specialist | `25fa90003de8fa6965d861fc66159fb0c188654723acc15e8c49ec8359ca8b93` | 104,712,848 |

The parent and project-adapted checkpoint hashes differ from all three released FoundIR artifacts. The compatibility evaluator now contains an explicit forbidden-hash guard and aborts if any of these released artifacts is supplied.
