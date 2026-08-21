#!/usr/bin/env python3
"""Render the screened corrective-holdout components for human review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests/qwen_lowlight_corrective_holdout_candidates.json"
DEFAULT_ANNOTATION = ROOT / "manifests/qwen_lowlight_corrective_holdout_annotation.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "data/real_lowlight_corrective_holdout/human_annotation_package_20260821"
    / "review_index.html"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_or_verify(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing review index: {path}")
        return
    path.write_bytes(content)


def image_url(output: Path, workspace_relative: str) -> str:
    image = ROOT / workspace_relative
    if not image.is_file():
        raise RuntimeError(f"missing candidate image: {image}")
    relative = os.path.relpath(image, output.parent).replace(os.sep, "/")
    return quote(relative, safe="/()")


def render(manifest_path: Path, annotation_path: Path, output: Path) -> bytes:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "screened_unlabeled_no_model_inference":
        raise RuntimeError("candidate manifest is not a screened unlabeled package")
    components = {
        str(item["component_id"]): item
        for item in manifest["components"]
        if item["screening_status"] == "pending_human_annotation"
    }
    with annotation_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row["component_id"] for row in rows} != set(components):
        raise RuntimeError("annotation template/component set mismatch")
    if any(row["project_level"] or row["selected_path"] for row in rows):
        raise RuntimeError("refusing to render a template that already has labels")

    sections: list[str] = []
    for row in rows:
        component = components[row["component_id"]]
        cards = []
        for path in component["members"]:
            cards.append(
                '<figure><img loading="lazy" src="{}"><figcaption>{}</figcaption></figure>'.format(
                    image_url(output, str(path)), html.escape(str(path))
                )
            )
        sections.append(
            '<section id="{id}"><h2>{id}</h2><p>Subsets: {subsets}. '
            'Candidates: {count}. Template display path: <code>{display}</code>.</p>'
            '<div class="grid">{cards}</div></section>'.format(
                id=html.escape(str(component["component_id"])),
                subsets=html.escape(", ".join(component["subsets"])),
                count=len(component["members"]),
                display=html.escape(str(component["display_path"])),
                cards="".join(cards),
            )
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen Lowlight corrective holdout — human review</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;background:#171717;color:#eee}}
h1{{font-size:24px}}h2{{font-size:17px;margin-top:0}}section{{border-top:1px solid #555;padding:20px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
figure{{margin:0;background:#252525;padding:8px}}img{{width:100%;height:240px;object-fit:contain;background:#000}}
figcaption{{font:11px ui-monospace,monospace;overflow-wrap:anywhere;margin-top:6px}}code{{overflow-wrap:anywhere}}
</style></head><body>
<h1>Qwen Lowlight corrective-holdout human review</h1>
<p>This page contains screened, unlabeled candidates only. Review all images in each scene component,
select at most one path in the annotation CSV, and follow the frozen rubric. Do not consult model predictions.</p>
<p>Candidate manifest SHA-256: <code>{manifest_sha}</code><br>
Annotation template SHA-256: <code>{annotation_sha}</code><br>
Components: {count}</p>
{sections}
</body></html>
""".format(
        manifest_sha=sha256(manifest_path),
        annotation_sha=sha256(annotation_path),
        count=len(rows),
        sections="\n".join(sections),
    )
    return document.encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    content = render(args.manifest, args.annotation, args.output)
    write_or_verify(args.output, content)
    print(f"ANNOTATION_REVIEW_INDEX_PASS components=190 output={args.output}")


if __name__ == "__main__":
    main()
