import unittest
import numpy as np
from PIL import Image
import torch
from lowlight_restoration.noise_texture_router_v3 import TextureRouter,views,route,sample_indices

class TextureTests(unittest.TestCase):
    def test_views_content_and_shape(self):
        a=np.arange(160*180*3,dtype=np.uint8).reshape(160,180,3)
        v=views(Image.fromarray(a))
        self.assertEqual(v.shape,(6,128,128,3))
        np.testing.assert_array_equal(v[1],a[:128,:128])
        np.testing.assert_array_equal(v[4],a[-128:,-128:])

    def test_reproducible_balanced_sampler(self):
        labels=torch.tensor([0]*13+[1]*20)
        for rank in (0,1):
            a=sample_indices(labels,37,rank);b=sample_indices(labels,37,rank)
            self.assertTrue(torch.equal(a,b));self.assertEqual(int(labels[a].sum()),8)
        self.assertFalse(torch.equal(sample_indices(labels,37,0),sample_indices(labels,37,1)))

    def test_global_gradient_matches_equal_rank_average(self):
        torch.set_num_threads(2);torch.manual_seed(3)
        m=TextureRouter();x=torch.rand(4,6,3,128,128);y=torch.tensor([0.,1.,1.,0.])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x),y);loss.backward()
        expected=[p.grad.clone() for p in m.parameters()];m.zero_grad()
        for i in range(2):
            (torch.nn.functional.binary_cross_entropy_with_logits(m(x[i*2:(i+1)*2]),y[i*2:(i+1)*2])/2).backward()
        for p,g in zip(m.parameters(),expected):torch.testing.assert_close(p.grad,g,rtol=2e-4,atol=2e-6)

    def test_no_rgb_output_other_routes_preserved(self):
        m=TextureRouter();self.assertEqual(m(torch.zeros(2,6,3,128,128)).shape,(2,))
        z=torch.eye(5)*10
        self.assertEqual(route(z,torch.ones(5))[torch.tensor([0,2,4])].tolist(),[0,2,4])

if __name__=='__main__':unittest.main()
