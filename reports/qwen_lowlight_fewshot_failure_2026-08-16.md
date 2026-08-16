# Qwen Lowlight visual few-shot validation failure

Date: 2026-08-16 UTC
Scope: frozen-Qwen Lowlight severity validation only

The calibrated visual few-shot experiment completed deterministically on one
NVIDIA H100 80 GB and failed its automatic validation gate. The result is
retained as diagnostic evidence and is not authorized for restoration
conditioning.

## Evidence

- code revision: `8f33757acafba6c48226caf8d5dd8603d128e5eb`
- config SHA-256:
  `25a1ad4085e13b4fffcef9e5f5c13238fa5188af4665d5eb13f7f68ab0d0d3d1`
- calibration bundle SHA-256:
  `f9b05225cb2f68a7e4227eafa05479b993aae2e02a49c41431db66d852fd4e24`
- validation bundle SHA-256:
  `c46a9e473137f9ee0dfd12288b1c6536c2761d4ff64cf191a0f93aac7d87855a`
- metrics artifact SHA-256:
  `9ed698200d5247eb58106758c91bef86f82d4d82ffb26f385ac63fef14058b16`
- log SHA-256:
  `19b54a250c0366f4ec3818294eaa5b9290fc34c49a1e3e184af16484dbbea0e1`
- structured output rate: `1.00`
- exact severity accuracy: `0.44`
- severity MAE: `1.04`
- comparison baseline accuracy: `0.64`
- comparison baseline MAE: `0.40`
- maximum allocated GPU memory: `17.2493 GiB`
- test split used: `false`
- official FoundIR used: `false`

## Failure pattern

The visual demonstrations corrected the original L5 underprediction but
overcorrected the scale:

- predicted counts L1-L5: `6, 1, 0, 1, 17`
- recall L1-L5: `1.0, 0.2, 0.0, 0.0, 1.0`
- every L4 input was predicted L5;
- four of five L3 inputs were predicted L5, and none were predicted L3;
- three of five L2 inputs were predicted L5.

This rejects prompt-only calibration as the Lowlight severity path. The next
bounded fallback is a small ordinal head trained on frozen Qwen visual
embeddings with a train-source selection gate before fixed validation.
