#!/usr/bin/env python3
"""Correct split-volume ordering for the selective SDSD builder."""

from __future__ import annotations

import re
from typing import Mapping

import smoke.build_sdsd_selective_development_manifest as implementation


VOLUME = re.compile(r"\.z(\d+)$")


def volumes_from_config(config: Mapping[str, object], subset: str) -> list[dict[str, object]]:
    files = config["files"]
    assert isinstance(files, list)
    selected = [
        dict(item) for item in files if str(item["name"]).startswith(f"{subset}_np")
    ]

    def disk_number(item: Mapping[str, object]) -> int:
        match = VOLUME.search(str(item["name"]))
        return int(match.group(1)) - 1 if match else len(selected) - 1

    selected.sort(key=disk_number)
    return selected


implementation.volumes_from_config = volumes_from_config


if __name__ == "__main__":
    raise SystemExit(implementation.main())
