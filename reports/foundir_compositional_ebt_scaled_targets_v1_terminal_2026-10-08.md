# FoundIR scaled EBT target cache v1 terminal result — 2026-10-08

## Terminal status

**READY.** All 66,000 lower-order targets were rendered and full-hash verified.

- Train targets: 60,000
- Validation targets: 6,000
- Triple targets: 0
- Cache size: approximately 24 GB
- Workers: 32 CPU processes
- Wall time: `1183.632873 seconds` (`0.328787 hours`)
- GPU usage: none (`0 GPU-hours`)
- Official FoundIR/sealed access: none

Frozen identities:

- Config SHA-256:
  `843e425a59bc93a0cf521c7457941e46dc2ea3bb7739918d867b38ad9632c806`
- Builder SHA-256:
  `4aff4733931530496b28fc4a9f2db7389841107269df44875a24ec58a2ecc35f`
- Target manifest:
  `manifests/foundir_compositional_ebt_scaled_targets_v1.jsonl`
- Target manifest SHA-256:
  `64bcea7cdb95ae1f399d469dfbb76b4078dbc98e501081084e54a9d086b65178`
- Metadata SHA-256:
  `a44abca0a007fc616fbccc22a9887f357fe65ee8794e2bcaccf21ff32d5b5907`

The build verified every PNG immediately after writing and then performed a
second complete 66,000-file SHA-256 pass before freezing the manifest. A
separate structural `--check` also passed.
