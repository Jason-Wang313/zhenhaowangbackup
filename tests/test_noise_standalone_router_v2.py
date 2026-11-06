import unittest
import torch
from lowlight_restoration.noise_lowlight_standalone_router_v2 import StandaloneRouter
from smoke.train_foundir_fullstack_noise_standalone_router_v2_cpu import admission

class RouterTests(unittest.TestCase):
    def test_margin_independent_of_inherited_confidence(self):
        m=StandaloneRouter(torch.zeros(20),torch.ones(20))
        x=torch.zeros(1,20)
        z=torch.tensor([[0.,-100.,0.,100.,0.]])
        self.assertTrue(torch.equal(m(z,x)[0],m(z*100,x)[0]))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(m(z,x)[0],torch.ones(1));loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.parameters()),0.)

    def test_other_three_routes_preserved(self):
        m=StandaloneRouter(torch.zeros(20),torch.ones(20))
        z=torch.eye(5)[[0,2,4]]*100
        self.assertEqual(m(z,torch.zeros(3,20))[1].tolist(),[0,2,4])

    def test_holdout_regression_blocks_admission(self):
        x={r:{'summary':{t:{'candidate_correct':10,'baseline_correct':10} for t in ('noise','lowlight')}} for r in ('calibration','confirmation')}
        self.assertTrue(admission(x));x['confirmation']['summary']['lowlight']['candidate_correct']=9
        self.assertFalse(admission(x))

if __name__=='__main__': unittest.main()
