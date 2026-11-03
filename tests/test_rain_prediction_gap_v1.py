import ast
import json
from pathlib import Path
import unittest
from unittest import mock
import torch
from smoke import rain_prediction_gap_math_v1 as m
from smoke import diagnose_foundir_rain_prediction_gap_v1 as impl

ROOT=Path(__file__).resolve().parents[1]


class Toy:
    def __call__(self,**kw):
        # Two conditions differ; observer must reproduce CFG arithmetic, not rewrite outputs.
        return kw['x']*.125+kw['ctx'][:,0,0].reshape(-1,1,1)


def pack(x):
    b,c,h,w=x.shape
    return x.flatten(2).transpose(1,2),torch.zeros(b,h*w,4,dtype=torch.long)


class PredictionGapTests(unittest.TestCase):
    def test_observer_preserves_official_sampler_exactly(self):
        from flux2.sampling import denoise_cfg
        torch.manual_seed(7);x=torch.randn(1,4,3,dtype=torch.bfloat16)
        cond=torch.randn_like(x);ids=torch.zeros(1,4,4,dtype=torch.long)
        txt=torch.tensor([[[.13]],[[.31]]],dtype=torch.bfloat16)
        args=dict(img=x,img_ids=ids,txt=txt,txt_ids=torch.zeros(2,1,4,dtype=torch.long),
                  timesteps=[1,.8,.5,.1,0],guidance=1.,img_cond_seq=cond,img_cond_seq_ids=ids)
        plain=denoise_cfg(Toy(),**args);observer=m.RolloutObserver(Toy(),[0,1,2,3])
        watched=denoise_cfg(observer,**args)
        self.assertTrue(torch.equal(plain,watched));self.assertEqual(observer.calls,4)
        self.assertEqual(set(observer.snapshots),{0,1,2,3})
        snap=observer.snapshots[0]
        self.assertTrue(torch.equal(snap['x'][:,:4],x))
        pred=Toy()(x=torch.cat([torch.cat([x,cond],1)]*2),ctx=txt)
        a,b=pred[:,:4].chunk(2)
        self.assertTrue(torch.equal(snap['guided'],a+1.*(b-a)))

    def test_observer_returns_same_tensor_object(self):
        value=torch.ones(2,8,3)
        model=mock.Mock(return_value=value);observer=m.RolloutObserver(model,[0])
        output=observer(x=torch.ones_like(value),x_ids=torch.zeros(2,8,4),timesteps=torch.ones(2))
        self.assertIs(output,value)

    def test_GT_path_perfect_velocity_recovers_clean(self):
        clean=torch.randn(1,3,2,2);epsilon=torch.randn_like(clean);ref=torch.randn_like(clean)
        velocity=pack(epsilon-clean)[0]
        class Perfect:
            def __call__(self,**kw):return torch.cat([velocity,velocity],1)
        for t in (1.,.75,.25,.01):
            estimate,terms=m.gt_path_prediction(Perfect(),clean,ref,epsilon,torch.tensor([t]),
                                               None,None,pack)
            torch.testing.assert_close(estimate,clean,atol=1e-6,rtol=1e-6)
            self.assertEqual(terms['flow_mse'],0.)

    def test_t1_does_not_contain_GT(self):
        from lowlight_restoration.flow import build_reference_flow_batch
        clean=torch.randn(1,3,2,2);noise=torch.randn_like(clean)
        a=build_reference_flow_batch(clean,clean,noise,torch.ones(1),pack)
        b=build_reference_flow_batch(clean+5,clean,noise,torch.ones(1),pack)
        self.assertTrue(torch.equal(a.model_tokens,b.model_tokens))
        self.assertFalse(torch.equal(a.target_velocity,b.target_velocity))

    def test_all32_once_across8_ranks(self):
        self.assertEqual(sorted(i for r in range(8) for i in m.assigned_indices(r)),list(range(64000,64032)))
        for rank,world in [(8,8),(0,4)]:
            with self.assertRaises(ValueError):m.assigned_indices(rank,world)

    def test_protected_rows_rejected(self):
        selected=[dict(usage='train_probe',split='train',record_id=str(i),source_group=str(i),gt_sha256=str(i)) for i in range(32)]
        rows=[dict(usage='train_update')]*64000+selected
        self.assertEqual(m.validate_indices(rows),selected)
        with self.assertRaises(ValueError):m.validate_indices(rows+[dict(selected[0],usage='development')])
        with self.assertRaises(ValueError):m.validate_indices(rows[:-1]+[dict(selected[-1],split='validation')])

    def test_no_training_or_modified_bound_source(self):
        src=(ROOT/'smoke/diagnose_foundir_rain_prediction_gap_v1.py').read_text()
        tree=ast.parse(src)
        attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
        self.assertNotIn('backward',attrs);self.assertNotIn('AdamW',attrs);self.assertNotIn('step',attrs)
        self.assertIn('requires_grad_(False)',src)
        cfg=impl.config();self.assertEqual(cfg['training_updates'],0)
        self.assertFalse(cfg['development_pixels_allowed'])
        self.assertEqual(cfg['checkpoints'],[400,1500])

    def test_launcher_is_separate_readonly_and_bounded(self):
        src=(ROOT/'smoke/run_foundir_rain_prediction_gap_v1_gpu.sh').read_text()
        for text in ['WALL_CAP=3600','--nnodes=4 --nproc_per_node=2','CUDA_VISIBLE_DEVICES=0,1',
                     'zhenhaowang_existing_flux_replay.lock','diagnose_foundir_rain_prediction_gap_v1.py',
                     'publish --event done','wait --event verified']:
            self.assertIn(text,src)
        self.assertNotIn('train_foundir_rain_native50_continue_v1.py',src)
        self.assertNotIn('pkill',src)

    def test_result_name_matches_cluster_verified_marker(self):
        src=(ROOT/'smoke/diagnose_foundir_rain_prediction_gap_v1.py').read_text()
        cluster=(ROOT/'smoke/existing_flux_replay_cluster_v1.py').read_text()
        self.assertIn("c.write_new(output/'result.json',result)",src)
        self.assertIn('c.sha(output / "result.json")',cluster)


if __name__=='__main__':unittest.main()
