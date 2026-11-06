import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from smoke import watch_foundir_fullstack_blur_first_v1_cpu as driver


class BlurScopeTests(unittest.TestCase):
    def test_only_blur_cache_and_score_are_submitted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver.queue, 'QUEUE', Path(tmp)), \
                patch.object(driver.queue, 'ready_idle'), patch.object(driver.queue, 'command') as command:
            driver.submit('cache', [0, 1, 2], 'a' * 40)
            driver.submit('score', [0], 'a' * 40)
            self.assertEqual([call.args[0][-2:] for call in command.call_args_list],
                             [['blur', 'cache'], ['blur', 'score']])

    def test_other_phases_or_topology_cannot_dispatch(self):
        with patch.object(driver.queue, 'command') as command:
            for phase, nodes in [('noise', [0]), ('train', [0]), ('cache', [0, 1, 2, 3]), ('score', [3])]:
                with self.assertRaises(ValueError):
                    driver.submit(phase, nodes, 'a' * 40)
            command.assert_not_called()

    def test_cpu_cannot_start_gpu_or_other_phases(self):
        for phase in ('cache', 'score', 'noise', 'train'):
            with self.assertRaises(ValueError):
                driver.cpu(phase)


if __name__ == '__main__':
    unittest.main()
