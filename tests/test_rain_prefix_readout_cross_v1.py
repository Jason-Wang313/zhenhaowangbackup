import unittest
import torch
from lowlight_restoration import rain_prefix_readout_cross_v1 as cross

class CrossTests(unittest.TestCase):
    def test_grid_intervenes_on_prefix_and_endpoint_independently(self):
        active=[None];events=[]
        def activate(step):active[0]=step;events.append(step)
        states,out=cross.run_grid(activate,lambda:active[0],lambda state:(state,active[0]))
        self.assertEqual(states,{800:800,1000:1000})
        self.assertEqual(out,{f'p{p}_r{r}':(p,r) for p in (800,1000) for r in (800,1000)})
        self.assertEqual(events,[800,800,1000,1000,800,1000])

    def test_adapter_copy_validates_every_tensor_before_mutation(self):
        ps={'a':torch.nn.Parameter(torch.ones(2)),'b':torch.nn.Parameter(torch.ones(2))}
        with self.assertRaises(ValueError):cross.copy_adapter(ps,{'a':torch.zeros(2),'b':torch.full((2,),float('nan'))})
        self.assertTrue(torch.equal(ps['a'],torch.ones(2)))
        cross.copy_adapter(ps,{'a':torch.zeros(2),'b':torch.full((2,),2.)})
        self.assertTrue(torch.equal(ps['a'],torch.zeros(2)));self.assertTrue(torch.equal(ps['b'],torch.full((2,),2.)))

    def test_partition_covers_only_fixed_eight(self):
        self.assertEqual(sorted(cross.assigned(0)+cross.assigned(1)),list(range(12800,12808)))
        self.assertFalse(set(cross.assigned(0))&set(cross.assigned(1)))
        with self.assertRaises(ValueError):cross.assigned(2)

    def rows(self):
        return [{'index':i,'training_fit_only':True,'metrics':{
          'p800_r800':{'psnr':32.,'lpips':.06},'p800_r1000':{'psnr':33.,'lpips':.05},
          'p1000_r800':{'psnr':31.5,'lpips':.06},'p1000_r1000':{'psnr':31.7,'lpips':.054}},
          'parity':{str(s):{'max_uint8_difference':0,'max_metric_difference':0.} for s in (800,1000)}} for i in cross.INDICES]

    def test_conditional_effects_and_parity_guard(self):
        rows=self.rows();r=cross.summarize(rows)
        self.assertTrue(r['prefix_cancellation_signature_in_this_eight_crop_probe'])
        self.assertAlmostEqual(r['conditional_deltas']['prefix_change_fixed1000_readout']['psnr'],-1.3)
        rows[0]['parity']['800']['max_uint8_difference']=1
        self.assertFalse(cross.summarize(rows)['prefix_cancellation_signature_in_this_eight_crop_probe'])

    def test_incomplete_or_wrong_arms_rejected(self):
        with self.assertRaises(ValueError):cross.summarize(self.rows()[:-1])
        rows=self.rows();rows[0]['metrics'].pop('p800_r1000')
        with self.assertRaises(ValueError):cross.summarize(rows)

if __name__=='__main__':unittest.main()
