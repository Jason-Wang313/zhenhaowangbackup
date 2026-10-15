from __future__ import annotations

import unittest

from smoke.acquire_sdsd_development_archives_segmented import SEGMENT_BYTES, segment_jobs


class SDSDSegmentedDownloadTests(unittest.TestCase):
    def test_segment_jobs_cover_each_byte_once(self) -> None:
        size = SEGMENT_BYTES * 2 + 7
        item = {"name": "x", "expected_size": size}
        jobs = segment_jobs([item])
        self.assertEqual([(job[2], job[3]) for job in jobs], [
            (0, SEGMENT_BYTES - 1),
            (SEGMENT_BYTES, SEGMENT_BYTES * 2 - 1),
            (SEGMENT_BYTES * 2, size - 1),
        ])


if __name__ == "__main__":
    unittest.main()
