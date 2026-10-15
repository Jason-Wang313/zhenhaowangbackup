from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smoke.acquire_sdsd_development_archives import write_once_or_verify


class SDSDDevelopmentAcquisitionTests(unittest.TestCase):
    def test_write_once_is_idempotent_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_once_or_verify(path, {"b": 2, "a": 1})
            original = path.read_bytes()
            self.assertEqual(json.loads(original), {"a": 1, "b": 2})
            write_once_or_verify(path, {"a": 1, "b": 2})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(RuntimeError, "immutable result differs"):
                write_once_or_verify(path, {"a": 2, "b": 1})


if __name__ == "__main__":
    unittest.main()
