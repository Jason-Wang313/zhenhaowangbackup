import ast
import copy
import inspect
from pathlib import Path
import unittest
import torch
from smoke import rain_midpoint_readout_v1 as r
from smoke.rain_prediction_gap_math_v1 import RolloutObserver
from smoke import evaluate_foundir_rain_midpoint_development_v1 as impl

ROOT=Path(__file__).resolve().parents[1]


class Toy:
    def __init__(self):self.calls=0
    def __call__(self,**kw):
        self.calls+=1
        return kw['x']*.125+kw['ctx'][:,0,0].reshape(-1,1,1)


class MidpointTests(unittest.TestCase):
    def test_fixed_prefix_equals_full_trace_and_stops_at26(self):
        from flux2.sampling import denoise_cfg,batched_prc_img
        torch.manual_seed(9)
        x,ids=batched_prc_img(torch.randn(1,3,2,2,dtype=torch.bfloat16))
        ref=torch.randn_like(x);model=Toy()
        def sampler(m):
            return denoise_cfg(m,x,ids,torch.tensor([[[.13]],[[.31]]],dtype=torch.bfloat16),
                torch.zeros(2,1,4,dtype=torch.long),timesteps=[1-i/50 for i in range(51)],guidance=1.,
                img_cond_seq=ref,img_cond_seq_ids=ids)
        snap,calls=r.run_prefix(model,sampler)
        self.assertEqual(calls,26);self.assertEqual(model.calls,26)
        full_model=Toy();full=RolloutObserver(full_model,[25]);sampler(full)
        self.assertEqual(full_model.calls,50)
        for k in snap:self.assertTrue(torch.equal(snap[k],full.snapshots[25][k]),k)
        from lowlight_restoration.rain_flow_adaptation import endpoint_estimate
        expected=endpoint_estimate(snap['x'][:,:4],snap['guided'],snap['t'].float(),(1,3,2,2))
        self.assertTrue(torch.equal(r.endpoint_latent(snap,'cpu'),expected))

    def test_prefix_does_not_swallow_real_errors_or_incomplete_sampler(self):
        def fail(m):raise RuntimeError('real failure')
        with self.assertRaisesRegex(RuntimeError,'real failure'):r.run_prefix(Toy(),fail)
        with self.assertRaises(ValueError):r.run_prefix(Toy(),lambda m:None)

    def test_generation_has_no_GT_or_scorer_input(self):
        self.assertEqual(list(inspect.signature(impl.generate_images).parameters),['model','ae','lq','prompt','seed','device'])
        code=inspect.getsource(impl.generate_images)
        self.assertNotIn("['gt']",code);self.assertNotIn('metric(',code)
        self.assertIn('steps=50',code)

    def test_100_images_exactly_once(self):
        groups=[r.assigned_indices(rank) for rank in range(8)]
        self.assertEqual([len(v) for v in groups],[13,13,13,13,12,12,12,12])
        self.assertEqual(sorted(i for group in groups for i in group),list(range(64032,64132)))
        with self.assertRaises(ValueError):r.assigned_indices(0,4)

    def test_development_scope_isolation(self):
        train=dict(record_id='train',source_group='train',gt_sha256='train')
        dev=[dict(usage='development',split='validation',category='10Rain',record_id=str(i),source_group=str(i),gt_sha256=str(i)) for i in range(100)]
        rows=[train]*64032+dev
        self.assertEqual(r.validate_rows(rows),dev)
        with self.assertRaises(ValueError):r.validate_rows(rows[:-1])
        with self.assertRaises(ValueError):r.validate_rows(rows[:-1]+[dict(dev[-1],source_group='train')])
        with self.assertRaises(ValueError):r.validate_rows(rows[:-1]+[dict(dev[-1],split='test')])

    def test_paired_metrics_gate_and_harm_counts(self):
        rows=[{'index':i,'metrics':{'fixed_prefix26':{'psnr':34.3,'lpips':.079},
            'matched_full50':{'psnr':31.,'lpips':.08},'original_one_step_teacher':{'psnr':34.2,'lpips':.073}}} for i in range(64032,64132)]
        out=r.paired_summary(rows)
        self.assertTrue(out['quality_gate_pass']);self.assertEqual(out['per_image_both_nonworse'],100)
        self.assertAlmostEqual(out['delta_vs_full50']['psnr'],3.3)
        changed=copy.deepcopy(rows)
        for row in changed:row['metrics']['fixed_prefix26']['lpips']=.081
        self.assertFalse(r.paired_summary(changed)['quality_gate_pass'])
        self.assertEqual(r.paired_summary(changed)['per_image_lpips_worse'],100)
        with self.assertRaises(ValueError):r.paired_summary(rows[:-1]+[rows[0]])
        changed[0]['metrics']['fixed_prefix26']['psnr']=float('nan')
        with self.assertRaises(ValueError):r.paired_summary(changed)

    def test_no_training_and_no_old_code_changes_required(self):
        tree=ast.parse((ROOT/'smoke/evaluate_foundir_rain_midpoint_development_v1.py').read_text())
        attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
        self.assertNotIn('backward',attrs);self.assertNotIn('AdamW',attrs);self.assertNotIn('step',attrs)
        cfg=impl.config();self.assertEqual(cfg['checkpoint_step'],400);self.assertEqual(cfg['call_index'],25)
        self.assertEqual(cfg['training_updates'],0);self.assertFalse(cfg['official_or_sealed_pixels_allowed'])

    def test_launch_and_completion_contract(self):
        src=(ROOT/'smoke/run_foundir_rain_midpoint_development_v1_gpu.sh').read_text()
        for s in ['WALL_CAP=1800','--nnodes=4 --nproc_per_node=2','CUDA_VISIBLE_DEVICES=0,1',
                  'zhenhaowang_existing_flux_replay.lock','evaluate_foundir_rain_midpoint_development_v1.py',
                  'publish --event verified']:self.assertIn(s,src)
        self.assertIn("c.write_new(output/'result.json',result)",inspect.getsource(impl.summarize))


if __name__=='__main__':unittest.main()
