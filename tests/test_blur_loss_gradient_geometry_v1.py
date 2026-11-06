import unittest
import torch
from smoke.blur_loss_gradient_geometry_v1 import geometry,summarize


class GradientGeometryTests(unittest.TestCase):
    def test_opposed_perceptual_gradient_can_reverse_MSE_descent(self):
        a=torch.tensor([1.,0.]);b=torch.tensor([-2.,0.])
        r=geometry([a],[b],['double_blocks.0.lora_A.weight'])['all']
        self.assertEqual(r['cosine'],-1.);self.assertEqual(r['LPIPS_to_MSE_norm_ratio'],2.)
        self.assertTrue(r['raw_SGD_mixed_descent_first_order_increases_MSE'])
        x=torch.zeros(2);loss=lambda z:((z+1)*a).sum();step=1e-4
        self.assertGreater(float(loss(x-step*(a+b))-loss(x)),0.)

    def test_aligned_and_zero_gradients_report_without_fabricated_cosine(self):
        self.assertEqual(summarize(1.,4.,2.)['cosine'],1.)
        self.assertFalse(summarize(1.,4.,2.)['raw_SGD_mixed_descent_first_order_increases_MSE'])
        r=summarize(0.,0.,0.);self.assertIsNone(r['cosine']);self.assertIsNone(r['LPIPS_to_MSE_norm_ratio'])

    def test_global_batch_gradient_differs_from_mean_per_image_norm(self):
        x=torch.tensor([1.,-1.]);p=torch.tensor(.3,requires_grad=True)
        a=torch.autograd.grad((p*x).sum()/2,p)[0]
        self.assertEqual(float(a),0.)
        per=[torch.tensor(1.),torch.tensor(-1.)]
        self.assertEqual(float(torch.stack(per).mean()),0.)
        self.assertEqual(float(torch.stack([g.abs() for g in per]).mean()),1.)

    def test_nonfinite_and_mismatched_shapes_rejected(self):
        with self.assertRaises(ValueError):geometry([torch.tensor(float('nan'))],[torch.tensor(1.)],['x'])
        with self.assertRaises(ValueError):geometry([torch.zeros(2)],[torch.zeros(3)],['x'])

if __name__=='__main__':unittest.main()
