from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "smoke/tmux_launcher_lib.sh"
LAUNCHERS = sorted((ROOT / "smoke").glob("launch_*tmux.sh"))


class TmuxLauncherTests(unittest.TestCase):
    def test_launchers_never_assign_reserved_tmux_variable(self) -> None:
        self.assertGreater(len(LAUNCHERS), 0)
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher.name):
                text = launcher.read_text()
                self.assertIsNone(re.search(r"(?m)^\s*(?:export\s+)?TMUX=", text))
                self.assertIn("tmux_launcher_lib.sh", text)
                self.assertIn("tmux_job_server", text)

    def test_helpers_preserve_outer_tmux_connection_variable(self) -> None:
        sentinel = "/tmp/existing-server,123,0"
        script = f"""
set -euo pipefail
source {HELPERS}
TMUX_BIN_PATH="$(select_tmux_binary)"
TMUX_SERVER_LABEL=zhenhaowang_test_only
tmux_job_server -V
test "$TMUX" = "$EXPECTED_TMUX"
test -f "$TMUX_BIN_PATH"
test -x "$TMUX_BIN_PATH"
"""
        environment = os.environ.copy()
        environment["TMUX"] = sentinel
        environment["EXPECTED_TMUX"] = sentinel
        completed = subprocess.run(
            ["bash", "-c", script],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertRegex(completed.stdout, r"tmux \d+\.\d+")

    def test_vae_launcher_has_current_pane_fallback_without_nested_tmux(self) -> None:
        launcher = ROOT / "smoke/launch_lowlight_vae_ceiling_tmux.sh"
        text = launcher.read_text()
        self.assertIn('DEVICE_SET="${CUDA_VISIBLE_DEVICES:-0,1}"', text)
        self.assertIn('export CUDA_VISIBLE_DEVICES="$DEVICE_SET"', text)
        self.assertIn('elif [[ -n "${TMUX:-}" ]]', text)
        self.assertIn('launch_mode="current_tmux_pane"', text)
        self.assertIn("LOWLIGHT_VAE_CEILING_CURRENT_TMUX_FALLBACK", text)
        self.assertIn(
            'exec env CUDA_VISIBLE_DEVICES="$DEVICE_SET" bash "$RUNNER"', text
        )
        self.assertIn('launch_mode="detached_nohup"', text)
        self.assertIn("LOWLIGHT_VAE_CEILING_DETACHED_NOHUP_FALLBACK", text)
        self.assertIn(
            'nohup env CUDA_VISIBLE_DEVICES="$DEVICE_SET" bash "$RUNNER"', text
        )
        self.assertIn('kill -0 "$job_pid"', text)


if __name__ == "__main__":
    unittest.main()
