#!/usr/bin/env python3
"""Run the scaling validator with JSON-object key canonicalization."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke import validate_qwen_lowlight_supervised_scaling as validator


_original_canonical_base_preflight = validator._canonical_base_preflight


def canonical_base_preflight_json_compatible(
    result: dict[str, object],
) -> dict[str, object]:
    """Make in-memory evidence compare exactly with its immutable JSON reload."""

    canonical = _original_canonical_base_preflight(result)
    selected_per_level = canonical.get("selected_per_level")
    if isinstance(selected_per_level, dict):
        canonical["selected_per_level"] = {
            str(key): value for key, value in selected_per_level.items()
        }
    return canonical


def main() -> int:
    validator._canonical_base_preflight = canonical_base_preflight_json_compatible
    return validator.main()


if __name__ == "__main__":
    raise SystemExit(main())
