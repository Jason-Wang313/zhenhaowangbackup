from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from smoke import foundir_fullstack_executor_3x2_v2 as e


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


class NewTopologyTests(unittest.TestCase):
    def gpu_csv(self, model='NVIDIA H100 80GB HBM3'):
        return f'0, {model}, 81559, GPU-a\n1, {model}, 81559, GPU-b\n'

    def test_new_node_mapping_and_h100_accepted(self):
        for node in range(3):
            with patch.object(e.subprocess, 'check_output', side_effect=[self.gpu_csv(), e.NODE_MAP[node]['ip']+'\n']):
                actual=e.inventory(node)
                self.assertEqual(actual['confirmed_mapping'],e.NODE_MAP[node])
                self.assertEqual(len(actual['cards']),2)

    def test_wrong_node_mapping_or_h200_rejected(self):
        with patch.object(e.subprocess, 'check_output', side_effect=[self.gpu_csv(), e.NODE_MAP[1]['ip']]):
            with self.assertRaisesRegex(ValueError,'wrong node mapping'):e.inventory(0)
        with patch.object(e.subprocess, 'check_output', return_value=self.gpu_csv('NVIDIA H200')):
            with self.assertRaisesRegex(ValueError,'H100'):e.inventory(0)

    def test_old_node3_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(e,'ROOT',Path(tmp)):
            job,_=ExecutorTests().fixture(Path(tmp));job['nodes']=[3]
            with self.assertRaisesRegex(ValueError,'invalid node'):e.validate(job)

    def test_duplicate_physical_gpu_registration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(e,'QUEUE',Path(tmp)):
            e.write(Path(tmp)/'workers/node0/registration.json',{'node':0,'cards':[{'uuid':'GPU-a'}]})
            with patch.object(e,'inventory',return_value={'node':1,'cards':[{'uuid':'GPU-a'}]}):
                with self.assertRaisesRegex(ValueError,'same physical GPUs'):e.register(1)


class BaselinePartitionTests(unittest.TestCase):
    def test_six_workers_cover_all_600_once(self):
        from smoke import audit_foundir_fullstack_official_baseline_3x2_v2 as b
        rows=list(range(600)); groups=[b.assigned(rows,r) for r in range(6)]
        self.assertEqual([len(g) for g in groups],[100]*6)
        self.assertEqual(sorted(v for g in groups for v in g),rows)
        with self.assertRaises(Exception):b.assigned(rows,6)

    def test_finish_uses_six_markers_and_rejects_missing_record(self):
        from smoke import audit_foundir_fullstack_official_baseline_3x2_v2 as b
        import contextlib,io
        rows=[{'record_id':f'r{i}', 'category':f'c{i%5}'} for i in range(600)]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); report=out/'summary.json'
            e.write(out/'receipt.json',{'world':6}); sha=e.digest(out/'receipt.json')
            for rank in range(6):
                group=b.assigned(rows,rank)
                e.write(out/f'worker{rank}.json',{'receipt_sha256':sha,'rank':rank,'records':[r['record_id'] for r in group]})
                for row in group:
                    e.write(out/'records'/(row['record_id']+'.json'),{'receipt_sha256':sha,'rank':rank,'row':row,'metrics':{'psnr':30.,'lpips':.1,'ssim':.9}})
            with patch.object(b,'OUTPUT',out),patch.object(b,'REPORT',report),patch.object(b,'bound',return_value={'rows':rows}),contextlib.redirect_stdout(io.StringIO()):
                b.finish()
                result=e.read(report);self.assertEqual(result['records'],600)
                self.assertTrue(all(v['count']==120 and v['psnr']==30. for v in result['categories'].values()))
                (out/'records/r599.json').unlink()
                with self.assertRaises(FileNotFoundError):b.finish()


if __name__ == '__main__':unittest.main()
