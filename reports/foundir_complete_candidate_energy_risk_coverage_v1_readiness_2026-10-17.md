# Complete candidate energy risk/coverage audit readiness — 2026-10-17

The user stopped `formal1` after its failed pass-1 validation. Shared-log
telemetry stopped after step `2,285` at UTC `2026-10-17T20:23:20Z`; direct
tmux termination leaves no normal launcher END marker. Never resume the run or
use its post-pass-1 partial state.

The next test is no-training and uses only the immutable step-1,706
checkpoint. It splits the fixed 256-record development subset by source group
into disjoint calibration and evaluation partitions. Candidate acceptance
margins are chosen only on calibration. The selected margin is then applied
once to evaluation, where PSNR, LPIPS and coverage are measured against
fidelity. Evaluation GT never changes the threshold.

This directly tests the remaining first-principles question: does the scorer
contain any deployable high-confidence ordering signal, or did it only learn
uncalibrated scores? If calibration-selected acceptance does not improve
evaluation PSNR without LPIPS damage, reject this scorer and replace its
heteroscedastic self-uncertainty objective with explicit pairwise
safe-versus-candidate learning.

Static readiness:

- checkpoint step and SHA-256 are frozen;
- Python compilation, JSON parsing and Bash syntax pass;
- focused unit tests pass `6/6`;
- source groups are disjoint by construction;
- threshold selection cannot access the evaluation partition;
- all protected-data flags are false.

Expected runtime is `.08–.20` wall-clock hours on eight H100/H200 GPUs,
equal to `.64–1.60` allocated GPU-hours. Replace this estimate with measured
runtime.
