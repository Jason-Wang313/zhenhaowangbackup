import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from smoke import watch_foundir_fullstack_noise_blur_v1_cpu as driver

class QueueTests(unittest.TestCase):
    def test_failure_is_not_completion(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(driver,'QUEUE',Path(tmp)):
            root=Path(tmp)/'jobs/example';root.mkdir(parents=True)
            (root/'node0.ended.json').write_text(json.dumps({'status':1}))
            with self.assertRaises(RuntimeError):driver.status('example',[0,1,2])
    def test_partial_success_does_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(driver,'QUEUE',Path(tmp)):
            root=Path(tmp)/'jobs/example';root.mkdir(parents=True)
            (root/'node0.ended.json').write_text(json.dumps({'status':0}))
            self.assertFalse(all(driver.status('example',[0,1,2])))
    def test_existing_job_identity_never_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(driver,'QUEUE',Path(tmp)):
            root=Path(tmp)/'jobs/example';root.mkdir(parents=True)
            (root/'job.json').write_text(json.dumps({'entry':'x','args':[],'nodes':[0],'source_commit':'old'}))
            with patch.object(driver,'command') as command:
                with self.assertRaises(ValueError):driver.submit('example','x',[],[0],'new',[])
                command.assert_not_called()
    def test_all_complete_exact_nodes(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(driver,'QUEUE',Path(tmp)):
            root=Path(tmp)/'jobs/example';root.mkdir(parents=True)
            (root/'node2.ended.json').write_text(json.dumps({'status':0}))
            self.assertTrue(all(driver.status('example',[2])))

if __name__=='__main__':unittest.main()
