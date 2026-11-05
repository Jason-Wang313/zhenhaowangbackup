import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from smoke import official_vlm_v4_unlimited_execution as u


class UnlimitedExecutionTests(unittest.TestCase):
    def fixture(self,root):
        out=root/u.x.OUTPUT
        for name in u.SOURCES:
            file=root/name;file.parent.mkdir(parents=True,exist_ok=True)
            file.write_text('synthetic execution source fixture\n')
        u.c.write_new(out/'receipt.json',{'synthetic':True})
        u.c.write_new(out/'render_plan.json',{'synthetic_plan':True})
        return out

    def test_unlimited_is_null_not_arbitrary_huge_limit_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(u,'ROOT',Path(tmp)):
            out=self.fixture(Path(tmp))
            before={f:u.c.sha(out/f) for f in ('receipt.json','render_plan.json')}
            with patch.object(u,'scope',return_value=out):
                u.authorize();first=(out/'render_budget.json').read_bytes();u.authorize()
                self.assertEqual(first,(out/'render_budget.json').read_bytes());u.verify()
                b=u.c.read(out/'render_budget.json')
                self.assertIsNone(b['approved_render_GPU_hours']);self.assertIsNone(b['cap_seconds_per_node'])
                self.assertTrue(b['time_limit_disabled']);self.assertEqual(b['world'],8)
            self.assertEqual(before,{f:u.c.sha(out/f) for f in before})

    def test_differing_previous_grant_preserved_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(u,'ROOT',Path(tmp)):
            out=self.fixture(Path(tmp));u.c.write_new(out/'render_budget.json',{'prior_grant':True})
            original=(out/'render_budget.json').read_bytes()
            with patch.object(u,'scope',return_value=out),self.assertRaises(ValueError):u.authorize()
            self.assertEqual((out/'render_budget.json').read_bytes(),original)

    def test_execution_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(u,'ROOT',Path(tmp)):
            out=self.fixture(Path(tmp))
            with patch.object(u,'scope',return_value=out):
                u.authorize();(Path(tmp)/u.SOURCES[-1]).write_text('changed synthetic wrapper')
                with self.assertRaisesRegex(ValueError,'execution overlay changed'):u.verify()

    def test_finite_or_tampered_grant_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(u,'ROOT',Path(tmp)):
            out=self.fixture(Path(tmp))
            with patch.object(u,'scope',return_value=out):
                u.authorize()
                b=u.c.read(out/'render_budget.json');b['cap_seconds_per_node']=99999999
                (out/'render_budget.json').write_text(__import__('json').dumps(b))
                with self.assertRaisesRegex(ValueError,'unlimited grant changed'):u.verify()

    def test_failed_and_very_long_attempts_remain_accounted_without_cap(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(u,'ROOT',Path(tmp)):
            out=self.fixture(Path(tmp))
            with patch.object(u,'scope',return_value=out):
                u.authorize()
                for n,status in enumerate((1,0)):
                    stem=out/'attempts'/f'render_node3_{n}'
                    u.c.write_new(Path(str(stem)+'.started.json'),{'gpus':2})
                    u.c.write_new(Path(str(stem)+'.ended.json'),{'seconds':1000000,'status':status,'gpus':2,'used_gpu_seconds':2000000})
                u.account_check('render_node3')
                with self.assertRaises(ValueError):u.account_check('route_node3')
                u.c.write_new(out/'attempts'/'render_node3_orphan.started.json',{})
                with self.assertRaisesRegex(ValueError,'interrupted accounting'):u.account_check('render_node3')

    def test_scope_rejects_other_topology_and_changed_plan(self):
        with patch.object(u.x,'NODES',2),self.assertRaisesRegex(ValueError,'only completed four-node'):u.scope()
        with tempfile.TemporaryDirectory() as tmp,patch.object(u,'ROOT',Path(tmp)),patch.object(u.producer,'bound',return_value={}):
            self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'approved plan/producer changed'):u.scope()

    def test_shell_has_no_timeout_and_preserves_locks_saving_and_failure_exit(self):
        root=Path(__file__).resolve().parents[1]
        run=root/'smoke/run_foundir_official_vlm_v4_unlimited.sh'
        start=root/'smoke/start_foundir_official_vlm_v4_unlimited_tmux.sh'
        subprocess.run(['bash','-n',str(run),str(start)],check=True)
        active='\n'.join(line for line in run.read_text().splitlines() if not line.lstrip().startswith('#'))
        self.assertNotIn('timeout',active);self.assertNotIn('CAP=',active);self.assertNotIn('REMAINING=',active)
        self.assertIn('flock -n 9',active);self.assertIn('flock -n 8',active)
        self.assertIn('trap finish_attempt EXIT',active);self.assertIn('used_gpu_seconds',active)
        self.assertIn('--max_restarts=0',active);self.assertNotIn(' route ',active)
        result=subprocess.run(['bash',str(run),'render','0'],capture_output=True)
        self.assertEqual(result.returncode,2)  # Non-destination workspace cannot launch GPU.


if __name__=='__main__':unittest.main()
