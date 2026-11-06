import unittest
import numpy as np
import torch
from PIL import Image
from lowlight_restoration.noise_lowlight_physical_router_v1 import features,FEATURE_NAMES,BinaryCorrection

class PhysicalRouterTests(unittest.TestCase):
    def test_constant_has_zero_native_high_frequency(self):
        f=features(Image.new('RGB',(600,550),(100,100,100)))
        self.assertEqual(len(f),len(FEATURE_NAMES));self.assertTrue(np.allclose(f[-5:],0))
        self.assertTrue(np.allclose(f[:3],100/255))
    def test_noise_changes_high_frequency_without_brightness_shift(self):
        rng=np.random.default_rng(14);a=np.clip(128+rng.normal(0,12,(512,512,3)),0,255).astype(np.uint8)
        f=features(Image.fromarray(a));self.assertGreater(float(f[-2]),.02)
        self.assertLess(abs(float(f[0])-128/255),.01)
    def test_zero_initialization_preserves_routes(self):
        m=BinaryCorrection(torch.zeros(len(FEATURE_NAMES)),torch.ones(len(FEATURE_NAMES)))
        z=torch.eye(5)*4;_,route,_=m(z,torch.zeros(5,len(FEATURE_NAMES)));self.assertTrue(torch.equal(route,z.argmax(-1)))
    def test_nonbinary_routes_cannot_be_changed_by_correction(self):
        m=BinaryCorrection(torch.zeros(len(FEATURE_NAMES)),torch.ones(len(FEATURE_NAMES)))
        with torch.no_grad():m.linear.bias.fill_(1000)
        z=torch.eye(5)*4;_,route,_=m(z,torch.zeros(5,len(FEATURE_NAMES)))
        self.assertEqual(route[[0,2,4]].tolist(),[0,2,4]);self.assertEqual(route[3].item(),1)

if __name__=='__main__':unittest.main()
