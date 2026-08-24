# Qwen pilot generation incident — uniform-value Lowlight limit

The first 32-worker generation attempt stopped after reporting 6,600 completed
futures because AgenticIR's linear Lowlight mapping encountered a crop whose
HSV value channel had `vmax == vmin`. The official expression
`(V-vmin)/(vmax-vmin)` is mathematically undefined for that input.

No differing file was overwritten. Every completed record has immutable PNG
and per-record hash evidence, so the rerun resumes those outputs.

For this exact limit case only, generation now applies the already-approved
project luminance calibration directly to the input crop. This is the unique
structure-preserving multiplicative solution and preserves the requested
L2--L5 luminance ratio. Non-uniform crops still execute the original AgenticIR
linear mapping followed by the same calibration. A focused constant-value test
guards finiteness, ordering, and the realized luma ratio.
