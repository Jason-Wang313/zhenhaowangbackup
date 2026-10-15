#!/usr/bin/env python3
"""Write an immutable per-track CURE stage terminal marker."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("published_style", "strict_holdout"), required=True)
    parser.add_argument("--status", type=int, required=True)
    parser.add_argument("--start", required=True)
    args = parser.parse_args()
    output = ROOT / "checkpoints/cure_two_track_v1" / args.track
    stages = {
        "embedder": output / "embedder/embedder_200.tar",
        "baseline": output / "baseline/model_300.tar",
        "cure": output / "cure/model_300.tar",
    }
    payload = {
        "schema_version": 1,
        "track": args.track,
        "exit_status": args.status,
        "started_at": args.start,
        "ended_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_sha256": sha256(ROOT / "configs/cure_two_track_reproduction_v1.json"),
        "official_commit": "a294116ad3d7dce58723a21bdd64f74b085e7b13",
        "checkpoints": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in stages.items() if path.is_file()
        },
        "official_foundir_test_accessed": False,
    }
    marker = output / "terminal_end.json"
    with marker.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"CURE_TERMINAL_MARKER {marker} sha256={sha256(marker)}", flush=True)


if __name__ == "__main__":
    main()
