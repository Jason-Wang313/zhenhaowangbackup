import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from smoke import watch_foundir_blur_fullgrid_1x2_v1_cpu as watch


class WatchTests(unittest.TestCase):
    def test_submit_only_node3_extension_with_published_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'workers/node3/heartbeat.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'state':'idle','utc_unix':time.time()}))
            (root/'jobs').mkdir()
            with patch.object(watch,'QUEUE',root),patch.object(watch,'command',return_value=SimpleNamespace(stdout='ok')) as command:
                watch.submit('train','published')
            args=command.call_args.args[0]
            self.assertIn('smoke/foundir_fullstack_node3_extension_v1.py',args)
            self.assertIn('round20_blur_fullgrid_resume_1x2_train_v1_node3',args)
            self.assertNotIn('--nodes',args)
            self.assertIn('smoke/train_foundir_blur_fullgrid_1x2_v1.py',args)

    def test_final_chain_has_no_cache_regeneration_and_does_compare_before_done(self):
        events=[]
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            def enqueue(event,detail):
                self.assertFalse((out/'monitor_done.json').exists());events.append(event)
            with patch.object(watch,'OUT',out),patch.object(watch,'cpu') as cpu,patch.object(watch,'submit') as submit,patch.object(watch,'wait'),patch.object(watch,'enqueue',side_effect=enqueue):
                watch.monitor('published')
            self.assertEqual([c.args[0] for c in submit.call_args_list],['train','official'])
            self.assertEqual([c.args[0] for c in cpu.call_args_list],['official_plan','train_finish','official_prepare','official_finish','official_comparison'])
            self.assertEqual(events[-2:],['terminal','official_comparison'])
            self.assertEqual(json.loads((out/'monitor_done.json').read_text())['status'],0)

    def test_failure_does_not_submit_official_or_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(watch,'OUT',Path(folder)),patch.object(watch,'cpu'),patch.object(watch,'submit') as submit,patch.object(watch,'wait',side_effect=RuntimeError('failure')),patch.object(watch,'enqueue') as enqueue,patch.object(watch,'cancel') as cancel:
                with self.assertRaises(RuntimeError):watch.monitor('published')
            self.assertEqual([c.args[0] for c in submit.call_args_list],['train'])
            self.assertEqual([c.args[0] for c in cancel.call_args_list],['train','official'])
            self.assertEqual(enqueue.call_args.args[0],'failed')


if __name__=='__main__':unittest.main()
