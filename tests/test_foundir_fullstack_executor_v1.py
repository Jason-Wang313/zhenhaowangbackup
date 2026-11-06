from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from smoke import foundir_fullstack_executor_v1 as e


class ExecutorTests(unittest.TestCase):
    def fixture(self, root, exitcode=0):
        entry = 'smoke/run_foundir_fullstack_synthetic.sh'
        file = root/entry; file.parent.mkdir(parents=True)
        file.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@"\nexit {exitcode}\n')
        job = {'id':'synthetic','entry':entry,'nodes':[0],'args':['$(touch SHOULD_NOT_EXIST)'],
               'sources':{entry:e.digest(file)},'source_commit':'synthetic fixture'}
        folder = root/'queue/jobs/synthetic'; e.write(folder/'job.json',job)
        return job, folder

    def test_arguments_are_literal_and_completed_job_is_not_reexecuted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(e,'ROOT',Path(tmp)), patch.object(e,'QUEUE',Path(tmp)/'queue'):
            job, folder = self.fixture(Path(tmp)); e.run_job(folder,0)
            end=(folder/'node0.ended.json').read_bytes()
            self.assertEqual(e.read(folder/'node0.ended.json')['status'],0)
            self.assertIn('$(touch SHOULD_NOT_EXIST)',(folder/'node0.log').read_text())
            self.assertFalse((Path(tmp)/'SHOULD_NOT_EXIST').exists())
            (Path(tmp)/job['entry']).write_text('later version')
            e.run_job(folder,0)  # Finished old job does not block later source versions.
            self.assertEqual(end,(folder/'node0.ended.json').read_bytes())

    def test_failed_job_preserved_and_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(e,'ROOT',Path(tmp)), patch.object(e,'QUEUE',Path(tmp)/'queue'):
            _,folder=self.fixture(Path(tmp),exitcode=7);e.run_job(folder,0)
            end=e.read(folder/'node0.ended.json');self.assertEqual(end['status'],7)
            self.assertGreaterEqual(end['allocated_GPU_hours'],0)
            e.run_job(folder,0);self.assertEqual(end,e.read(folder/'node0.ended.json'))

    def test_unfinished_attempt_blocks_instead_of_duplicate_launch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(e,'ROOT',Path(tmp)):
            _,folder=self.fixture(Path(tmp));e.write(folder/'node0.started.json',{'old_attempt':True})
            with self.assertRaisesRegex(ValueError,'reconciliation'):e.run_job(folder,0)
            self.assertFalse((folder/'node0.log').exists())

    def test_unpublished_changed_source_and_legacy_launcher_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(e,'ROOT',Path(tmp)):
            job,_=self.fixture(Path(tmp));e.validate(job)
            (Path(tmp)/job['entry']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'source changed'):e.validate(job)
            job['entry']='smoke/run_old_experiment.sh'
            with self.assertRaisesRegex(ValueError,'dedicated fullstack'):e.validate(job)

    def test_path_escape_and_duplicate_node_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(e,'ROOT',Path(tmp)):
            job,_=self.fixture(Path(tmp))
            with self.assertRaises(ValueError):e.local_file('../secret')
            job['nodes']=[0,0]
            with self.assertRaises(ValueError):e.validate(job)

    def test_precancelled_job_does_not_execute(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(e,'ROOT',Path(tmp)), patch.object(e,'QUEUE',Path(tmp)/'queue'):
            _,folder=self.fixture(Path(tmp));e.write(folder/'cancel.json',{'reason':'synthetic cancel'})
            e.run_job(folder,0);self.assertFalse((folder/'node0.log').exists())
            self.assertEqual(e.read(folder/'node0.ended.json')['status'],143)


if __name__ == '__main__':unittest.main()
