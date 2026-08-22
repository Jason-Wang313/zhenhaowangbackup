from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

from smoke.build_sdsd_selective_development_manifest import (
    central_directory,
    parse_content_range,
    resolve_zip64_values,
    select_pairs,
)
from smoke.run_sdsd_selective_hf_v1 import _resolve_url, volumes_from_config


class SDSDSelectiveDevelopmentTests(unittest.TestCase):
    def test_range_parser_remains_fail_closed(self) -> None:
        self.assertEqual(parse_content_range("bytes 4-19/20"), (4, 19, 20))
        with self.assertRaises(RuntimeError):
            parse_content_range("4-19/20")

    def test_huggingface_volume_order_and_locator_are_frozen(self) -> None:
        config = {
            "transport_mirror": {"repository": "owner/repo", "revision": "abc123"},
            "files": [
                {
                    "name": "indoor_np.zip",
                    "huggingface_path": "indoor/final.zip",
                    "huggingface_lfs_sha256": "f" * 64,
                },
                {
                    "name": "indoor_np.z02",
                    "huggingface_path": "indoor/part.z02",
                    "huggingface_lfs_sha256": "2" * 64,
                },
                {
                    "name": "indoor_np.z01",
                    "huggingface_path": "indoor/part.z01",
                    "huggingface_lfs_sha256": "1" * 64,
                },
            ],
        }
        volumes = volumes_from_config(config, "indoor")
        self.assertEqual([item["name"] for item in volumes], ["indoor_np.z01", "indoor_np.z02", "indoor_np.zip"])
        url, revision, digest = _resolve_url(str(volumes[0]["google_drive_id"]))
        self.assertEqual(revision, "abc123")
        self.assertEqual(digest, "1" * 64)
        self.assertEqual(
            url,
            "https://huggingface.co/datasets/owner/repo/resolve/abc123/indoor/part.z01",
        )

    def test_sdsd_pairs_streams_by_ordinal_position_not_filename(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs/sdsd_development_acquisition_v1d.json").read_text()
        )
        policy = config["bounded_pair_policy"]
        entries: dict[str, list[dict[str, str]]] = {"indoor": [], "outdoor": []}
        for subset, count in (("indoor", 70), ("outdoor", 80)):
            outer = set(policy[f"official_{subset}_outer_sequences"])
            for sequence_number in range(1, count + 1):
                sequence = f"pair{sequence_number}"
                frames = 30 if sequence in outer else 3
                for position in range(frames):
                    entries[subset].append(
                        {
                            "name": (
                                f"{subset}_np/GT/{sequence}/{1000 + position:04d}.npy"
                            )
                        }
                    )
                    entries[subset].append(
                        {
                            "name": (
                                f"{subset}_np/input/{sequence}/{2000 + position:04d}.npy"
                            )
                        }
                    )
        selected = select_pairs(entries, policy)
        self.assertEqual(len(selected), 500)
        first = selected[0]
        self.assertEqual(first["frame_position"], 0)
        self.assertEqual(first["gt_frame_filename"], "1000.npy")
        self.assertEqual(first["lq_frame_filename"], "2000.npy")

    def test_zip64_offset_sentinel_is_resolved_from_extra_field(self) -> None:
        offset = 4_700_000_123
        payload = struct.pack("<Q", offset)
        extra = struct.pack("<HH", 0x0001, len(payload)) + payload
        values = resolve_zip64_values(1_474_688, 600_214, 0xFFFFFFFF, 0, extra)
        self.assertEqual(values, (1_474_688, 600_214, offset, 0))
        with self.assertRaises(RuntimeError):
            resolve_zip64_values(1_474_688, 600_214, 0xFFFFFFFF, 0, b"")


if __name__ == "__main__":
    unittest.main()
