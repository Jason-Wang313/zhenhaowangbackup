# FoundIR proposal-relative energy Noise V2 — launch readiness

Date: 2026-10-15 UTC

## Scope

This is the single max-complete test of the advisor-approved PPT energy idea.
It contains exactly the six user-authorized corrections and no added research
branch:

1. differentiable LPIPS/perceptual training alongside pixel and edge fidelity;
2. frozen real-model outputs for wrong expert, under-restoration,
   over-restoration, and detail damage;
3. direct training through the complete one-, two-, and three-step energy
   trajectory;
4. per-step MSE and LPIPS no-harm penalties, with the unchanged proposal also
   available to the final lowest-energy selector;
5. each record's existing frozen-Qwen five-degradation severity vector from
   `qwen_levels_json`, rather than a constant Noise descriptor;
6. full-resolution spatial energy maps and local pairwise quality ranking,
   rather than only an early global score.

No new backbone, dataset, degradation, specialist, MoE, router, or
compositional branch is included.

## Frozen scientific protocol

- category: Noise only;
- training: `54,585` records in `1,755` source groups;
- validation: fixed source-disjoint `256`-record development subset;
- proposal: frozen adapted Noise Restormer;
- global batch: `32` on one homogeneous eight-GPU node;
- primary scale: five effective passes, `8,530` optimizer steps;
- success gate: at least `+.30 dB` PSNR and `-.005 LPIPS` versus the frozen
  proposal on the fixed development validation;
- protected official/sealed/internal/parent-test and frozen-660 access: none.

The per-image frozen-Qwen field is not synthetically constant. Across Noise
train plus validation, it contains `32` distinct five-dimensional descriptors.
Noise severity contains `51,029` level-1 and `5,415` level-2 records. These
values are used as existing VLM evidence without generating or changing
annotations.

## Engineering checks completed

- Python compilation: pass;
- JSON parsing: pass;
- Bash syntax: pass;
- whitespace/diff check: pass;
- CPU tests: `5/5` pass, including an explicit test that the adaptive selector
  can retain the unchanged proposal and a per-row VLM-severity parsing test;
- three-step second-order gradients: present and finite for every energy-model
  parameter in the CPU graph test;
- LPIPS network API: verified to accept normalized `[-1, 1]` tensors with
  `normalize=False`;
- validation avoids running the four training-only contrastive experts;
- H100 profile: micro-batch `1/GPU`, accumulation `4`;
- H200 profile: micro-batch `4/GPU`, accumulation `1`.

The inherited PyTorch name collision `self.half` versus `nn.Module.half()` was
found and corrected to `self.half_stack` before GPU launch.

## Next exact action

Run a fresh 25-step eight-GPU engineering qualification. This is only a check
of CUDA memory, DDP, LPIPS backpropagation, second-order gradients,
checkpointing, validation, and measured throughput. It cannot establish that
the research idea works. The formal `8,530`-step run must use a new run ID and
start from step zero after qualification passes.

Exact wall-clock time and allocated GPU-hours are unknown until this heavier
V2 graph is measured on the actual GPU family. The qualification result must
report both.
