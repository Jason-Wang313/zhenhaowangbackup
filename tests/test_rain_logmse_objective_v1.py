import math
import unittest
import torch
from lowlight_restoration.rain_logmse_objective_v1 import per_image_terms,EPSILON_RGB_MINUS1_1 as EPS

class LogMSETests(unittest.TestCase):
    def test_matches_old_local_gradient_at_reference(self):
        x=torch.full((1,3,8,8),.1,requires_grad=True);gt=torch.zeros_like(x);lp=torch.zeros(1)
        value,_=per_image_terms(x,gt,lp,.01);value.sum().backward()
        torch.testing.assert_close(x.grad,20*x.detach()/x.numel(),rtol=1e-5,atol=1e-7)

    def test_per_image_log_not_log_of_batch_average(self):
        x=torch.stack([torch.full((3,8,8),.01),torch.full((3,8,8),.3)]).requires_grad_()
        result,_=per_image_terms(x,torch.zeros_like(x),torch.zeros(2),.01)
        expected=10*(.01+EPS)*torch.log((torch.tensor([.0001,.09])+EPS)/(.01+EPS))
        torch.testing.assert_close(result,expected)
        wrong=10*(.01+EPS)*math.log((.04505+EPS)/(.01+EPS))
        self.assertGreater(abs(float(result.mean())-wrong),.05)

    def test_zero_error_is_finite_and_has_zero_image_gradient(self):
        x=torch.zeros(1,3,8,8,requires_grad=True)
        value,_=per_image_terms(x,torch.zeros_like(x),torch.zeros(1),.01);value.sum().backward()
        self.assertTrue(torch.isfinite(value).all());self.assertEqual(float(x.grad.abs().max()),0.)

    def test_lpips_weight_and_clean_error_weight_preserved(self):
        x=torch.stack([torch.full((3,8,8),.01),torch.full((3,8,8),.3)])
        lp=torch.ones(2,requires_grad=True);values,terms=per_image_terms(x,torch.zeros_like(x),lp,.01);values.sum().backward()
        torch.testing.assert_close(lp.grad,torch.full((2,),.05))
        self.assertGreater(float(terms['effective_mse_weight'][0]),float(terms['effective_mse_weight'][1]))

    def test_rank_local_accumulation_matches_global_mean(self):
        x=torch.rand(4,3,4,4,requires_grad=True);gt=torch.zeros_like(x)
        loss,_=per_image_terms(x,gt,torch.zeros(4),.01);loss.mean().backward();expected=x.grad.clone();x.grad=None
        for i in range(2):
            loss,_=per_image_terms(x[i*2:i*2+2],gt[i*2:i*2+2],torch.zeros(2),.01);(loss.mean()/2).backward()
        torch.testing.assert_close(x.grad,expected)

    def test_invalid_reference_and_target_grad_rejected(self):
        x=torch.zeros(1,3,4,4)
        for r in (0,-1,float('nan')):
            with self.assertRaises(ValueError):per_image_terms(x,x,torch.zeros(1),r)
        with self.assertRaises(ValueError):per_image_terms(x,x.clone().requires_grad_(),torch.zeros(1),.01)

if __name__=='__main__':unittest.main()
