# Historical workspace Git snapshot — 2026-10-15

## Purpose

This snapshot preserves the reproducible text history that had accumulated in
the dirty HDD3 workspace while keeping the current research priority clear.
It is an archival/backfill milestone, not a new method result and not an
instruction to resume every historical branch.

Fresh agents must read `AGENTS.md` and
`reports/foundir_current_handoff_2026-09-23.md` first. Their topmost STOP
sections define the active task. Older configs, scripts, tests, and reports in
this snapshot are historical evidence unless those current instructions
explicitly reactivate them.

## What was included

The live workspace was compared against fetched `origin/main` at
`e0c1451691f9047b8af7836ac634e19fd388ac1b`. Git identified 729 real text-file
differences after already-pushed files were removed from consideration:

- 397 files under `smoke/`;
- 163 files under `reports/`;
- 106 JSON experiment contracts under `configs/`;
- 46 tests under `tests/`;
- 16 implementation files under `lowlight_restoration/`;
- one `.gitignore` update needed to track advisor-governance records.

They cover the historical FLUX.2, Restormer, routing, composition, CURE,
OneRestore, compositional-EBT, and real-data Generalist experiments represented
in the current handoff. Files are split into a source/config/test commit and a
report/context commit so later readers can distinguish executable material
from recorded evidence.

## What was deliberately excluded

- all `manifests/`, including generated CSV/JSONL files up to hundreds of MB;
- checkpoints, model weights, datasets, exported images, caches, telemetry,
  and live logs;
- the stale lowercase `agents.md` and old top-level session handoffs, because
  they can conflict with the authoritative uppercase `AGENTS.md` and current
  FoundIR handoff;
- credentials and tokens. Candidate files were scanned for GitHub tokens,
  bearer credentials, private keys, and common embedded-secret forms before
  staging.

The excluded files remain untouched in the dirty local workspace. Their
absence from Git is intentional and must not be interpreted as deletion.

## Validation scope

Before push, all included JSON files are parsed, all included shell scripts
receive `bash -n`, and Python sources receive a compile-only syntax check.
These checks establish archival integrity only. They do not make every old
experiment active, scientifically valid, or runnable under the current
environment.
