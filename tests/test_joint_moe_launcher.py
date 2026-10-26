"""CPU shell control-flow tests; these do NOT validate actual CUDA/NCCL."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = 'smoke/run_foundir_rain_blur_joint_vlm_moe_v2_8gpu.sh'
TRAINER = 'smoke/train_foundir_rain_blur_joint_vlm_moe_v2_ddp.py'
PROBE = 'smoke/cure_nccl_rehearsal_v1.py'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='joint-moe-launcher-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = self.root / 'calls'
        for name in ('smoke', 'configs', 'logs', 'bin'):
            (self.root / name).mkdir()
        fake_python = self.root / 'bin' / 'fake-python'
        fake_python.write_text('''#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *cure_nccl_rehearsal_v1.py*)
    echo "probe:$NCCL_NVLS_ENABLE:$NCCL_P2P_DISABLE:$NCCL_DEBUG:$NCCL_DEBUG_SUBSYS:$*" >> "$TEST_CALLS"
    exit "${TEST_PROBE_EXIT:-0}" ;;
  *train_foundir_rain_blur_joint_vlm_moe_v2_ddp.py*)
    echo "train:$NCCL_NVLS_ENABLE:$NCCL_P2P_DISABLE:$NCCL_DEBUG:$NCCL_DEBUG_SUBSYS:$*" >> "$TEST_CALLS"
    exit "${TEST_TRAIN_EXIT:-0}" ;;
  *) echo 'GPU_PASS mocked_only count=8' ;;
esac
''')
        fake_python.chmod(0o700)
        source = (ROOT / RUNNER).read_text()
        root_line = next(line for line in source.splitlines() if line.startswith('ROOT='))
        source = source.replace(root_line, f'ROOT={shlex.quote(str(self.root))}', 1)
        python_line = next(line for line in source.splitlines() if line.startswith('PYTHON='))
        source = source.replace(python_line, f'PYTHON={shlex.quote(str(fake_python))}', 1)
        source = source.replace('/tmp/zhenhaowang_rain_blur_image_moe_v1.lock',
                                str(self.root / 'node.lock'))
        (self.root / RUNNER).write_text(source)
        for name in (TRAINER, PROBE, 'configs/foundir_rain_blur_joint_vlm_moe_v2.json'):
            (self.root / name).write_text('test placeholder\n')
        self.env = dict(os.environ, TEST_CALLS=str(self.calls),
                        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                        NCCL_NVLS_ENABLE='1', NCCL_P2P_DISABLE='1',
                        NCCL_DEBUG='WARN', NCCL_DEBUG_SUBSYS='ALL')

    def run_job(self, **env):
        return subprocess.run(['bash', str(self.root / RUNNER), 'qualification',
                               'image_only', 'retry_test'],
                              env=dict(self.env, **env), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=10)

    def test_pass_enforces_compatibility_and_probe_precedes_training(self):
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].startswith('probe:0:0:INFO:INIT,ENV,GRAPH:'))
        self.assertTrue(calls[1].startswith('train:0:0:WARN:ALL:'))
        self.assertIn('JOINT_MOE_COMM_CHECK_PASS', result.stdout)
        self.assertIn('JOINT_MOE_LAUNCH_END status=0', result.stdout)

    def test_probe_failure_never_starts_training(self):
        result = self.run_job(TEST_PROBE_EXIT='7')
        self.assertEqual(result.returncode, 7, result.stdout)
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)
        self.assertIn('JOINT_MOE_COMM_CHECK_FAIL status=7 training_not_started=1', result.stdout)
        self.assertNotIn('JOINT_MOE_COMM_CHECK_PASS', result.stdout)

    def test_timeout_never_starts_training(self):
        # Simulate GNU timeout's exit code, not an actual 180-second hang.
        fake_timeout = self.root / 'bin' / 'timeout'
        fake_timeout.write_text('''#!/usr/bin/env bash
test "$1" = --signal=TERM && test "$2" = --kill-after=30s && test "$3" = 180s || exit 99
exit 124
''')
        fake_timeout.chmod(0o700)
        result = self.run_job(PATH=str(self.root / 'bin') + os.pathsep + os.environ['PATH'])
        self.assertEqual(result.returncode, 124, result.stdout)
        self.assertFalse(self.calls.exists())
        self.assertIn('JOINT_MOE_COMM_CHECK_FAIL status=124 training_not_started=1', result.stdout)

    def test_training_failure_is_not_hidden_by_probe_success(self):
        result = self.run_job(TEST_TRAIN_EXIT='9')
        self.assertEqual(result.returncode, 9, result.stdout)
        self.assertIn('JOINT_MOE_COMM_CHECK_PASS', result.stdout)
        self.assertIn('JOINT_MOE_LAUNCH_END status=9', result.stdout)

    def test_existing_log_is_preserved(self):
        log = self.root / 'logs/foundir_rain_blur_joint_vlm_moe_v2_image_only_retry_test.log'
        log.write_text('previous failure\n')
        result = self.run_job()
        self.assertEqual(result.returncode, 3, result.stdout)
        self.assertEqual(log.read_text(), 'previous failure\n')
        self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
