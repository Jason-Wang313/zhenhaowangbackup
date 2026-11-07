import tempfile
from pathlib import Path
import unittest
import torch
from lowlight_restoration.lora import inject_lora
from smoke.blur_actual_update_tools_v1 import load_exact_step,original_batch,tensor_dot,replay_check
from smoke import blur_delta_state_v1 as state


class ActualUpdateTests(unittest.TestCase):
    def test_exact_step_checksum_and_scope(self):
        m=torch.nn.Sequential(torch.nn.Linear(2,1))
        inject_lora(m,['0'],2,2.)
        o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5)
        with tempfile.TemporaryDirectory() as folder:
            r=state.save(folder,m,o,step=0,world=2,rank_rng=[state.rng_state('cpu')]*2,receipt_sha='bound',history=[])
            _,v=load_exact_step(folder,0,'bound');self.assertEqual(v['next_global_slot'],0)
            with self.assertRaises(ValueError):load_exact_step(folder,0,'wrong')
            p=Path(folder)/r['file'];p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaises(ValueError):load_exact_step(folder,0,'bound')

    def test_restore_original_sampling_order_for_both_transitions(self):
        rows=[{'role':'train','id':i} for i in range(1024)]
        for step in (0,199):
            b=original_batch(rows,step)
            self.assertEqual(len({x['id'] for s in b['items_by_rank'].values() for x in s}),8)
            self.assertEqual(b,original_batch(rows,step))
        self.assertNotEqual(original_batch(rows,0),original_batch(rows,199))

    def test_first_order_prediction_does_not_guarantee_finite_improvement(self):
        gradient=[torch.tensor([2.])];actual_delta=[torch.tensor([-3.])]
        self.assertEqual(tensor_dot(gradient,actual_delta),-6.)
        # f(x)=x²; actual move1->-2 raises f despite negative local derivative.
        self.assertGreater((-2.)**2-1.**2,0.)

    def test_fixed_replay_tolerances_fail_on_loss_and_norm_mismatch(self):
        expected=dict(loss=.3,mse=.02,lpips=.4,gradient_norm=.5)
        value={k:expected[k] for k in ('loss','mse','lpips')}
        self.assertTrue(replay_check(value,.5,expected)['passed'])
        self.assertFalse(replay_check({**value,'lpips':.401},.5,expected)['passed'])
        self.assertFalse(replay_check(value,.51,expected)['passed'])

if __name__=='__main__':unittest.main()
