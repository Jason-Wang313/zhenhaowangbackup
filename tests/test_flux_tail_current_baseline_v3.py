import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from smoke import flux_tail_current_baseline_v3 as policy
from smoke import run_foundir_flux_tail_relative_energy_v3 as runner


class CurrentReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def flow(self,value=129,error=None):
        class Flow:
            def session(inner,observation,seed,expected_proposal=None):
                self.assertIsNone(expected_proposal)
                if error:raise RuntimeError(error)
                return SimpleNamespace(proposal=torch.full((1,3,8,8),value).float()/127.5-1)
        return Flow()

    def test_historical_mismatch_records_without_stopping_or_mutating(self):
        history=torch.full((3,8,8),128,dtype=torch.uint8);before=history.clone()
        session,current,info=policy.current_session(self.flow(),history,123,history)
        self.assertFalse(info['historical_equal'])
        self.assertEqual(info['different_elements'],history.numel())
        self.assertEqual(info['max_abs_uint8'],1.)
        self.assertTrue(torch.equal(current,torch.full_like(history,129)))
        self.assertTrue(torch.equal(history,before))
        self.assertTrue(info['same_environment_native_tail_check_retained'])
        self.assertNotIn('target',inspect.signature(policy.current_session).parameters)

    def test_current_run_changes_and_native_tail_errors_still_fail(self):
        history=torch.full((3,8,8),128,dtype=torch.uint8)
        current=torch.full_like(history,129)
        policy.current_session(self.flow(),history,123,history,current)
        with self.assertRaisesRegex(RuntimeError,'since bootstrap'):
            policy.current_session(self.flow(130),history,123,history,current)
        with self.assertRaisesRegex(RuntimeError,'native tail failure'):
            policy.current_session(self.flow(error='native tail failure'),history,123,history)

    def test_bad_reference_is_not_silently_accepted(self):
        with self.assertRaises(RuntimeError):policy.uint8_image(torch.full((1,3,8,8),float('nan')))
        with self.assertRaisesRegex(RuntimeError,'geometry or dtype'):
            policy.current_session(self.flow(),torch.zeros(3,8,8),1,torch.zeros(3,8,8))

    def test_baseline_pools_are_per_run_not_shared_across_hardware(self):
        r={'cache_root':'irrelevant'}
        self.assertNotEqual(runner.mining_directory(Path('/tmp/a'),r,0),
                            runner.mining_directory(Path('/tmp/b'),r,0))
        self.assertEqual(runner.mining_directory(Path('/tmp/a'),r,4),Path('/tmp/a/mining/step_000004'))

    def test_new_root_leaves_old_run_untouched_and_limits_unchanged(self):
        spec=runner.spec()
        self.assertTrue(spec['output_root'].endswith('_v3_direct'))
        self.assertEqual(spec['historical_parity_policy'],'record_not_gate')
        self.assertEqual(spec['global_batch_pairs'],32)
        self.assertEqual(spec['modes']['qualification']['steps'],8)
        self.assertEqual(spec['modes']['trainfit']['steps'],200)
        self.assertEqual(spec['train_indices'],list(range(24)))
        self.assertEqual(spec['checkpoint_every_optimizer_step'],1)
        self.assertFalse(spec['auto_trainfit'])
        self.assertEqual(spec['allowed_gpu_counts'],[2,4,6,8])

    def test_environment_advantage_cannot_hide_energy_zero_gain(self):
        metrics={'historical':{'psnr':30.,'lpips':.1},
                 'original':{'psnr':31.,'lpips':.08},
                 'fp32_control':{'psnr':31.1,'lpips':.079},
                 'raw_step1':{'psnr':31.,'lpips':.08},
                 'bounded_step1':{'psnr':31.,'lpips':.08}}
        before=copy.deepcopy(metrics)
        result=policy.aggregate_metrics([{'metrics':metrics}],runner.spec())
        chosen=result['bounded_step1']
        self.assertEqual(chosen['psnr_gain'],0.)
        self.assertEqual(chosen['lpips_relative_reduction'],0.)
        self.assertFalse(chosen['joint_target'])
        self.assertEqual(chosen['compared_with']['historical']['psnr_gain'],1.)
        self.assertLess(chosen['compared_with']['fp32_control']['psnr_gain'],0.)
        self.assertEqual(metrics,before)

    def test_missing_controls_are_not_reported_as_verified(self):
        with self.assertRaises(RuntimeError):policy.aggregate_metrics([{'metrics':{}}],runner.spec())


if __name__=='__main__':unittest.main()
