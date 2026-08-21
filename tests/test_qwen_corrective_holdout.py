from __future__ import annotations

import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from smoke.build_qwen_lowlight_corrective_holdout_annotation_package import (
    build_annotation_rows,
    build_components,
    exposure_group_key,
    validate_archive_member,
)
from smoke.render_qwen_lowlight_corrective_holdout_annotation_html import image_url


def candidate(
    path: str,
    digest: str,
    phash: str,
    *,
    subset: str = "low",
    filename: str = "image.jpg",
    overlaps: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "path": path,
        "sha256": digest,
        "perceptual_hash": phash,
        "subset": subset,
        "filename": filename,
        "mean_luma": 10.0,
        "protected_perceptual_matches": overlaps or [],
    }


class QwenCorrectiveHoldoutTests(unittest.TestCase):
    def test_archive_member_rejects_traversal_and_symlink(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_archive_member(zipfile.ZipInfo("../outside.jpg"))
        symlink = zipfile.ZipInfo("Test/low/link.jpg")
        symlink.external_attr = 0o120777 << 16
        with self.assertRaises(RuntimeError):
            validate_archive_member(symlink)

    def test_archive_member_accepts_image_layout(self) -> None:
        path = validate_archive_member(zipfile.ZipInfo("Test/low/night (1).jpg"))
        self.assertEqual(path.as_posix(), "Test/low/night (1).jpg")

    def test_exact_and_phash_neighbors_share_a_component(self) -> None:
        items = [
            candidate("a", "same", "0000000000000000"),
            candidate("b", "same", "ffffffffffffffff"),
            candidate("c", "other", "0000000000000001"),
        ]
        components = build_components(items)
        self.assertEqual(len(components), 1)
        self.assertEqual(set(components[0]["members"]), {"a", "b", "c"})

    def test_protected_overlap_rejects_the_whole_internal_component(self) -> None:
        match = {"path": "protected.png", "hamming_distance": 2, "sources": []}
        items = [
            candidate("a", "one", "0000000000000000", overlaps=[match]),
            candidate("b", "two", "0000000000000001"),
        ]
        components = build_components(items)
        rows, rejected = build_annotation_rows(items, components)
        self.assertEqual(rows, [])
        self.assertEqual(rejected, {components[0]["component_id"]})
        self.assertEqual(
            components[0]["screening_status"], "rejected_protected_scene_overlap"
        )

    def test_low_copy_is_preferred_over_npe_copy_for_display(self) -> None:
        items = [
            candidate("NPE/x.jpg", "same", "0", subset="NPE", filename="x.jpg"),
            candidate("low/x.jpg", "same", "0", subset="low", filename="x.jpg"),
        ]
        component = build_components(items)[0]
        self.assertEqual(component["display_path"], "low/x.jpg")

    def test_conservative_exposure_keys(self) -> None:
        self.assertEqual(
            exposure_group_key(
                candidate(
                    "x", "a", "0", subset="MEF", filename="lampicka08.png"
                )
            ),
            "mef-exposure:lampicka",
        )
        self.assertEqual(
            exposure_group_key(
                candidate(
                    "x", "a", "0", subset="low", filename="cloudy (1)_jpg.jpg"
                )
            ),
            "cloudy-encoding:cloudy (1)",
        )
        self.assertIsNone(
            exposure_group_key(
                candidate("x", "a", "0", subset="low", filename="night (1).jpg")
            )
        )

    def test_review_image_url_is_relative_and_escaped(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            output = root / "package/review.html"
            image = root / "images/night (1).jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            workspace_relative = image.resolve().relative_to(Path.cwd()).as_posix()
            self.assertEqual(
                image_url(output, workspace_relative), "../images/night%20(1).jpg"
            )


if __name__ == "__main__":
    unittest.main()
