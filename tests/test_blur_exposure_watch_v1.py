import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from smoke import watch_foundir_blur_exposure_v1_cpu as w
from smoke.existing_flux_replay_contract_v1 import write_new,read


class WatchTests(unittest.TestCase):
    def test_failed_peer_cancels_only_own_job(self):
        with tempfile.TemporaryDirectory() as folder:
            queue=Path(folder);job=queue/'jobs'/w.JOBS['train']
            write_new(job/'node1.ended.json',{'status':1})
            with patch.object(w,'QUEUE',queue):
                with self.assertRaises(RuntimeError):w.wait('train')
            self.assertEqual(read(job/'cancel.json')['reason'],'peer_failed_in_train')
            self.assertFalse((queue/'jobs'/w.JOBS['official']/'cancel.json').exists())

    def test_controller_exception_stops_its_live_job_preserving_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'out';out.mkdir();queue=root/'queue';job=queue/'jobs'/w.JOBS['train']
            write_new(job/'job.json',{'source_commit':'own','nodes':[0,1,2],'entry':w.ENTRIES['train']})
            asset=job/'asset';asset.write_text('keep')
            with (patch.object(w,'OUT',out),patch.object(w,'QUEUE',queue),
                  patch.object(w,'cpu',side_effect=RuntimeError('test')),patch.object(w,'enqueue')):
                with self.assertRaises(RuntimeError):w.monitor('own')
            self.assertTrue((job/'cancel.json').exists());self.assertEqual(asset.read_text(),'keep')


if __name__=='__main__':unittest.main()
