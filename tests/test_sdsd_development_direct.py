from __future__ import annotations

import unittest

from smoke.acquire_sdsd_development_archives_direct import parse_content_range


class SDSDDevelopmentDirectTests(unittest.TestCase):
    def test_content_range_parser(self) -> None:
        self.assertEqual(parse_content_range("bytes 0-1023/4698669056"), (0, 1023, 4698669056))
        with self.assertRaisesRegex(RuntimeError, "invalid Content-Range"):
            parse_content_range("bytes */4698669056")


if __name__ == "__main__":
    unittest.main()
