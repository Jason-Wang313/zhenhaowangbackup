#!/usr/bin/env python3
"""Paired source-bootstrap comparison of the frozen CURE reproduction tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/cure_two_track_v1_eval.csv"
OUTPUT = ROOT / "checkpoints/cure_two_track_v1/comparison/result.json"
TRACKS = ("published_style", "strict_holdout")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_bootstrap(values: list[tuple[str, float]], *, seed: int = 20260926) -> dict:
    by_source: dict[str, list[float]] = defaultdict(list)
    for source, delta in values:
        by_source[source].append(delta)
    source_means = np.asarray([np.mean(items) for _, items in sorted(by_source.items())], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(source_means), size=(10000, len(source_means)))
    bootstrap = source_means[draws].mean(axis=1)
    return {
        "paired_record_mean": float(np.mean([delta for _, delta in values])),
        "source_balanced_mean": float(source_means.mean()),
        "source_bootstrap_95pct_ci": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "source_count": len(source_means),
        "record_count": len(values),
    }


def compare(manifest: Path) -> dict:
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("empty evaluation manifest")
    results = {}
    for track in TRACKS:
        output = ROOT / "checkpoints/cure_two_track_v1" / track / "evaluation"
        aggregate = output / "result.json"
        if not aggregate.is_file():
            raise FileNotFoundError(aggregate)
        payload = json.loads(aggregate.read_text())
        if payload.get("manifest_sha256") != sha256(manifest) or payload.get("rows") != len(rows):
            raise RuntimeError(f"track evaluation manifest/count mismatch: {track}")
        results[track] = output
    groups: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        record = {}
        for track in TRACKS:
            sidecar = results[track] / f"{row['id']}.json"
            if not sidecar.is_file():
                raise FileNotFoundError(sidecar)
            record[track] = json.loads(sidecar.read_text())
        stem = Path(row["input_path"]).stem
        for metric in ("psnr", "ssim"):
            delta = record["published_style"]["metrics"][metric] - record["strict_holdout"]["metrics"][metric]
            for group in (row["kind"], f"{row['kind']}:{row['source_prompt']}"):
                groups[group][metric].append((stem, delta))
    paired = {
        group: {metric: paired_bootstrap(values) for metric, values in metrics.items()}
        for group, metrics in groups.items()
    }
    full = paired["full"]["psnr"]
    selective = paired["selective"]["psnr"]
    return {
        "schema_version": 1,
        "comparison": "published_style minus strict_holdout; positive PSNR/SSIM favors published_style",
        "manifest_sha256": sha256(manifest),
        "paired_differences": paired,
        "predeclared_material_gap_rule": "at least +0.5 dB source-balanced PSNR with 95% source-bootstrap lower bound >0 in either full or selective group; inspect SSIM and failure cases before any novelty claim",
        "material_gap_rule_met": any(
            value["source_balanced_mean"] >= 0.5 and value["source_bootstrap_95pct_ci"][0] > 0
            for value in (full, selective)
        ),
        "not_a_method_novelty_claim": True,
        "official_foundir_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = compare(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
