from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from smoke import watch_foundir_blur_fullgrid_v1_cpu as w
from smoke.existing_flux_replay_contract_v1 import write_new,read


class PairedWatchTests(unittest.TestCase):
    def test_only_new_round_entries_are_dispatched(self):
        self.assertEqual(set(w.ENTRIES), {'cache','train','official'})
        for entry in w.ENTRIES.values():
            self.assertIn('blur_fullgrid', entry)
            self.assertNotIn('exposure', entry)
        self.assertTrue(all(j.startswith('round20_blur_fullgrid_') for j in w.JOBS.values()))

    def jobs(self,queue):
        for n in (1,3):
            p=queue/'jobs'/(w.JOBS['train']+f'_node{n}')
            write_new(p/'job.json',{'nodes':[n],'entry':w.ENTRIES['train'],'source_commit':'own','args':[]})

    def test_submit_preserves_legacy_worker_and_node3_single_node_envelope(self):
        with tempfile.TemporaryDirectory() as folder:
            queue=Path(folder);calls=[]
            for n in (1,3):write_new(queue/'workers'/f'node{n}/heartbeat.json',{'utc_unix':time.time(),'state':'idle'})
            def command(args):calls.append(args);return SimpleNamespace(stdout='test_submitted')
            with patch.object(w,'QUEUE',queue),patch.object(w,'command',command):w.submit('train','own')
            self.assertEqual(len(calls),2)
            self.assertIn('smoke/foundir_fullstack_executor_3x2_v2.py',calls[0]);self.assertIn('--nodes',calls[0])
            self.assertIn('smoke/foundir_fullstack_node3_extension_v1.py',calls[1]);self.assertNotIn('--nodes',calls[1])

    def test_failed_peer_cancels_only_unfinished_paired_job(self):
        with tempfile.TemporaryDirectory() as folder:
            queue=Path(folder);self.jobs(queue)
            p=queue/'jobs'/(w.JOBS['train']+'_node1');write_new(p/'node1.ended.json',{'status':1})
            with patch.object(w,'QUEUE',queue):
                with self.assertRaises(RuntimeError):w.wait('train')
            self.assertFalse((p/'cancel.json').exists())
            self.assertEqual(read(queue/'jobs'/(w.JOBS['train']+'_node3')/'cancel.json')['reason'],'paired_peer_failed')

    def test_controller_failure_preserves_assets_and_cancels_both_owned_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'out';out.mkdir();queue=root/'queue';self.jobs(queue)
            asset=out/'asset';asset.write_text('keep')
            with (patch.object(w,'OUT',out),patch.object(w,'QUEUE',queue),patch.object(w,'cpu',side_effect=RuntimeError('test')),
                  patch.object(w,'enqueue')):
                with self.assertRaises(RuntimeError):w.monitor('own')
            for n in (1,3):self.assertTrue((queue/'jobs'/(w.JOBS['train']+f'_node{n}')/'cancel.json').exists())
            self.assertEqual(asset.read_text(),'keep')


if __name__=='__main__':unittest.main()
