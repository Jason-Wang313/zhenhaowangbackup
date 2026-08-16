from __future__ import annotations

import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from smoke.extract_foundir_archive import validate_member
from smoke.validate_foundir_official import (
    EXPECTED_CATEGORY_COUNTS,
    LOWLIGHT_CATEGORY,
    LOWLIGHT_IDS,
    ROOT,
    parse_meta,
    write_or_verify_bytes,
)


class FoundIRAcquisitionTests(unittest.TestCase):
    def test_official_metadata_matches_isolated_lowlight_contract(self) -> None:
        records = parse_meta(ROOT / "third_party/foundir/Testset_meta_info.txt")
        self.assertEqual(len(records), sum(EXPECTED_CATEGORY_COUNTS.values()))
        lowlight = [record for record in records if record["category"] == LOWLIGHT_CATEGORY]
        self.assertEqual(len(lowlight), 50)
        self.assertEqual({record["pair_id"] for record in lowlight}, LOWLIGHT_IDS)
        self.assertNotIn("1301", {record["pair_id"] for record in lowlight})

    def test_archive_member_validation_rejects_traversal_and_symlink(self) -> None:
        with self.assertRaises(SystemExit):
            validate_member(zipfile.ZipInfo("../outside.png"))
        symlink = zipfile.ZipInfo("14Lowlight/1101.JPG")
        symlink.external_attr = 0o120777 << 16
        with self.assertRaises(SystemExit):
            validate_member(symlink)

    def test_archive_member_validation_accepts_normal_image_path(self) -> None:
        result = validate_member(zipfile.ZipInfo("14Lowlight/GT/1101.JPG"))
        self.assertEqual(result.as_posix(), "14Lowlight/GT/1101.JPG")

    def test_output_reuse_is_exact_and_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_or_verify_bytes(path, b"first\n")
            write_or_verify_bytes(path, b"first\n")
            with self.assertRaises(SystemExit):
                write_or_verify_bytes(path, b"different\n")


if __name__ == "__main__":
    unittest.main()
