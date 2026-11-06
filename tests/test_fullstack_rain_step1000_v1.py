"""Protect exact Rain-only scope, accepted Haze reuse, and completion accounting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smoke import evaluate_foundir_fullstack_rain_step1000_v1 as task
from smoke import watch_foundir_fullstack_rain_step1000_v1_cpu as observer


class RainEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value=task.plan()  # Real600 metadata only; never load official pixels.

    def test_complete_partition_no_duplicates(self):
        rows=self.value['rows'];parts=[rows[r::6] for r in range(6)]
        ids=[i['row']['record_id'] for part in parts for i in part]
        self.assertEqual(len(ids),600);self.assertEqual(len(set(ids)),600)
        self.assertEqual(sum(i['deployed']['anchor']=='rain' for i in rows),100)
        self.assertEqual(sum(len(task.x.tiles(i['row'])) for i in rows if i['deployed']['anchor']=='rain'),4680)

    def test_replace_only_rain_weights_without_mutating_parent(self):
        parent=task.c.read(task.ROOT/task.x.OUTPUT/'receipt.json')['original']
        before=copy.deepcopy(parent);spec={'path':'new-rain.safetensors','sha256':'fixed'}
        bank=task.evaluation_bank(parent,spec)
        self.assertEqual(parent,before)
        self.assertEqual(bank['parent']['experts']['rain']['checkpoint'],spec)
        bank['parent']['experts']['rain']['checkpoint']=before['parent']['experts']['rain']['checkpoint']
        self.assertEqual(bank,before)

    def test_haze_reuse_must_keep_new_accepted_output(self):
        item=next(i for i in self.value['rows'] if i['deployed']['anchor']=='haze')
        old=item['accepted']
        record={'item':item,'rank':0,'receipt_sha256':'test','metrics':old['metrics'],
                'output':{'path':old['path'],'sha256':old['sha256']},'reused_parent':True}
        with patch.object(task.c,'sha',return_value=old['sha256']):
            task.record_check(record,item,'test',0)
            record['metrics']={k:item['deployed'][k] for k in old['metrics']}
            with self.assertRaises((ValueError,RuntimeError)):task.record_check(record,item,'test',0)

    def test_rain_cannot_reuse_old_output(self):
        item=next(i for i in self.value['rows'] if i['deployed']['anchor']=='rain')
        record={'item':item,'rank':0,'receipt_sha256':'test','metrics':item['accepted']['metrics'],'reused_parent':True}
        with self.assertRaises((ValueError,RuntimeError)):task.record_check(record,item,'test',0)

    def test_duplicate_accepted_ids_rejected(self):
        parent=task.c.read(task.ROOT/task.x.OUTPUT/'receipt.json');report=task.c.read(task.ROOT/task.x.REPORT)
        accepted=task.c.read(task.ACCEPTED);accepted['records'][-1]=accepted['records'][0]
        with self.assertRaises((ValueError,RuntimeError)):task.accepted_rows(parent,report,accepted)

    def test_changed_route_rejected(self):
        parent=task.c.read(task.ROOT/task.x.OUTPUT/'receipt.json');report=task.c.read(task.ROOT/task.x.REPORT)
        accepted=task.c.read(task.ACCEPTED);accepted['records'][0]['anchor']='rain'
        with self.assertRaises((ValueError,RuntimeError)):task.accepted_rows(parent,report,accepted)

    def test_observer_terminal_accounting_uses_real_executor_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=root/'job';job.mkdir()
            for n in range(3):
                (job/f'node{n}.started.json').write_text(json.dumps({'utc_unix':100+n}))
                (job/f'node{n}.ended.json').write_text(json.dumps({'status':0,'seconds':20+n,'allocated_GPU_hours':(20+n)*2/3600}))
            with patch.multiple(observer,ROOT=root,JOB=job,OUT=root/'out',MONITOR=root/'monitor'):
                value=observer.snapshot('terminal')
            self.assertEqual(value['wall_seconds'],24)
            self.assertAlmostEqual(value['completed_node_GPU_hours'],126/3600)

    def test_failed_cpu_phase_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'prepare.json').write_text(json.dumps({'status':1,'stderr':'expected'}))
            with patch.object(observer,'MONITOR',p),patch.object(observer.subprocess,'run') as run:
                with self.assertRaises(RuntimeError):observer.run_once('prepare',['unused'])
                run.assert_not_called()


if __name__=='__main__':unittest.main()
