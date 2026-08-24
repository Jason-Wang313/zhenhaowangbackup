#!/usr/bin/env python3
"""Generate a bounded representative image smoke from the frozen manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    read_jsonl,
    sha256,
    write_generated_png,
    write_new_or_verify,
)


def select_smoke(records: list[dict[str, object]]) -> list[dict[str, object]]:
    train = [record for record in records if record["split"] == "train"]
    selected: list[dict[str, object]] = []
    selected.append(next(record for record in train if record["composition_size"] == 0))
    for name in ("blur", "noise", "rain", "haze", "low_light"):
        selected.append(next(record for record in train if record["degradations"] == [name]))
    selected.append(
        next(
            record
            for record in train
            if record["composition_size"] == 2 and "haze" in record["degradations"]
        )
    )
    selected.append(
        next(
            record
            for record in train
            if record["composition_size"] == 3
            and {"noise", "rain", "low_light"} == set(record["degradations"])
        )
    )
    return selected


def main() -> None:
    manifest = ROOT / "manifests/qwen_multidegradation_pilot_v1.jsonl"
    records = read_jsonl(manifest)
    selected = select_smoke(records)
    generated = [write_generated_png(record) for record in selected]
    payload = {
        "schema_version": 1,
        "status": "pass",
        "manifest_sha256": sha256(manifest),
        "records": generated,
    }
    result = ROOT / "reports/qwen_multidegradation_generation_smoke_v1.json"
    result_sha = write_new_or_verify(
        result, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    print(
        f"QWEN_MULTIDEGRADATION_GENERATION_SMOKE_PASS records={len(generated)} "
        f"result_sha256={result_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
