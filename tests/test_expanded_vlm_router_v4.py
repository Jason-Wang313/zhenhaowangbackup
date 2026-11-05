import copy
from collections import Counter
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from lowlight_restoration.conditional_vlm_router_v4 import ConditionalVLMRouter
from smoke import expanded_vlm_router_contract_v4 as x
from smoke import expanded_vlm_router_state_v4 as state
from smoke import run_foundir_expanded_vlm_router_v4 as run
from tests.test_frozen_image_router_v2 import candidates, small_model


def conditional_small():
    reference,_=small_model()
    torch.manual_seed(x.SEED)
    m=ConditionalVLMRouter(6,4,8)
    m.image.load_state_dict(reference.image.state_dict()); m.eval()
    return m,torch.optim.AdamW(m.vlm.parameters(),lr=.001,weight_decay=.01,foreach=False,fused=False)


def metadata():
    rows,original=candidates()
    prev=x.old.parent.select(rows,original,{'fit':4,'calibration':3,'holdout':3})
    confirm=x.old.reserve_confirmation(rows,prev,original,3)['items']
    return rows,prev,confirm


class ExpandedV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_zero_initial_output_equals_frozen_image(self):
        m,_=conditional_small(); im=torch.randn(10,6); vlm=torch.randn(10,4)
        z,_,b,d=m(im,vlm)
        self.assertTrue(torch.equal(z,b)); self.assertEqual(int(torch.count_nonzero(d)),0)

    def test_correction_can_observe_image_scores_but_not_backprop_into_them(self):
        m,_=conditional_small()
        torch.nn.init.normal_(m.vlm.net[-1].weight)
        v=torch.randn(4,4); b=torch.randn(4,5,requires_grad=True)
        other=b.detach().clone(); other[:,0]+=5
        first=m.vlm(v,b)
        self.assertFalse(torch.equal(first,m.vlm(v,other)))
        first.sum().backward(); self.assertIsNone(b.grad)
        self.assertTrue(torch.allclose(first,m.vlm(v,b.detach()+3),atol=1e-6))

    def test_update_only_vlm_and_exact_checkpoint_resume(self):
        m,opt=conditional_small(); image_before=copy.deepcopy(m.image.state_dict())
        im=torch.randn(32,6); vlm=torch.randn(32,4); y=torch.arange(32)%5
        with tempfile.TemporaryDirectory() as temp:
            state.save(temp,m,opt,0,[],'id')
            history=[x.old.update(m,opt,im,vlm,y)]
            state.save(temp,m,opt,1,history,'id'); saved=state.load(temp,'id')
            random_expected=torch.rand(4); x.old.update(m,opt,im,vlm,y)
            other,oo=conditional_small(); state.restore(other,oo,saved)
            self.assertTrue(torch.equal(random_expected,torch.rand(4)))
            x.old.update(other,oo,im,vlm,y)
            self.assertTrue(all(torch.equal(t,other.state_dict()[k]) for k,t in m.state_dict().items()))
            for k,e in opt.state_dict()['state'].items():
                self.assertTrue(all(torch.equal(v,oo.state_dict()['state'][k][n]) for n,v in e.items()))
            with self.assertRaises(ValueError): state.load_step(temp,6401,'id')
        self.assertTrue(all(torch.equal(t,image_before[k]) for k,t in m.image.state_dict().items()))

    def test_new_sources_multicrop_no_cross_role_or_old_holdout_overlap(self):
        rows,prev,confirm=metadata(); items=x.expand(rows,prev,confirm,cap=7,views=3)
        self.assertEqual(items,x.expand(list(reversed(rows)),prev,confirm,cap=7,views=3))
        fit,groups=x.schedule(items)
        self.assertEqual(len(fit),5*7*3)
        self.assertTrue(all(len(v)==7 for v in groups.values()))
        self.assertEqual(sum(bool(i['reuse_key']) for i in items),5*(4+3))
        for t in x.ORDER:
            for source in groups[t]:
                self.assertEqual(len({tuple(fit[j]['row']['crop_xy']) for j in source}),3)
        x.validate(items,prev,confirm)
        bad=copy.deepcopy(items); bad[0]['row']['gt_sha256']=confirm[0]['row']['gt_sha256']
        with self.assertRaises(ValueError): x.validate(bad,prev,confirm)

    def test_small_images_do_not_fake_three_distinct_crops(self):
        rows,prev,confirm=metadata()
        # Limit cap to inherited sources so changing fixture metadata cannot add sources.
        for i in prev:
            if i['role']=='fit': i['row'].update(width='512',height='512',crop_xy=[0,0])
        fit=[i for i in x.expand(rows,prev,confirm,cap=4,views=3) if i['role']=='fit']
        self.assertEqual(len(fit),20)

    def test_balanced_class_source_sampling_and_resumable_slots(self):
        rows,prev,confirm=metadata(); items=x.expand(rows,prev,confirm,cap=7,views=3)
        fit,groups=x.schedule(items)
        counts=Counter(j for step in range(x.STEPS) for j in x.batch_indices(groups,step))
        self.assertEqual(sum(counts.values()),6400*32)
        self.assertEqual(Counter({t:sum(n for j,n in counts.items() if fit[j]['task']==t) for t in x.ORDER}),Counter({t:40960 for t in x.ORDER}))
        self.assertLessEqual(max(counts.values())-min(counts.values()),1)
        with self.assertRaises(ValueError): x.batch_indices(groups,6400)

    def test_confirmation_locked_until_selection(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'ROOT',Path(tmp)):
            with self.assertRaisesRegex(ValueError,'locked'):
                run.load_role({'items':[]},{},'confirmation')

    @staticmethod
    def prediction_fixture():
        items=[{'key':f'confirm_{t}_{j}','role':'confirmation','task':t,'seed':j,'row':{'crop_xy':[0,0]}}
               for t in x.ORDER for j in range(20)]
        rows=[{'key':i['key'],'seed':i['seed'],'crop_xy':[0,0],'predicted_anchor':i['task'],
               'logits':[float(t==i['task']) for t in x.ORDER]} for i in items]
        return items,{arm:{'rows':copy.deepcopy(rows)} for arm in x.ARMS[:-1]}

    def test_render_plan_deduplicates_exact_routes_never_metric_selects(self):
        items,pred=self.prediction_fixture(); plan=x.render_plan(items,pred)
        self.assertEqual(plan['unique_renders'],100)
        r=pred['conditional_v4']['rows'][0]; r['logits']=[0.,2.,0.,0.,0.]; r['predicted_anchor']='noise'
        changed=x.render_plan(items,pred)
        self.assertEqual(changed['unique_renders'],101)
        self.assertFalse(changed['selection_uses_restoration_metrics'])
        self.assertEqual(changed['arms']['known_class_reference'],plan['arms']['known_class_reference'])

    def test_render_plan_rejects_argmax_input_or_population_tampering(self):
        items,pred=self.prediction_fixture()
        pred['conditional_v4']['rows'][0]['predicted_anchor']='rain'
        with self.assertRaises(ValueError): x.render_plan(items,pred)
        items,pred=self.prediction_fixture(); pred['previous_v3']['rows'].pop()
        with self.assertRaises(ValueError): x.render_plan(items,pred)

    def test_quality_results_report_targets_without_claiming_official_pass(self):
        items,pred=self.prediction_fixture(); plan=x.render_plan(items,pred)
        metrics={rid:{'psnr':40.,'lpips':.01} for rid in plan['routes']}
        result=x.aggregate(plan,items,metrics)
        self.assertTrue(result['all_numeric_targets_on_new100_pass'])
        self.assertIsNone(result['official_SoW_pass'])
        self.assertFalse(result['joint_mean_improvement_over_image_and_previous_vlm'])
        self.assertEqual(set(result['arms']),set(x.ARMS))
        metrics.pop(next(iter(metrics)))
        with self.assertRaises(ValueError): x.aggregate(plan,items,metrics)

    def test_four_shards_cover_all_features_and_render_ids(self):
        for count in (11457,100,137,400):
            rows=list(range(count)); from smoke.expanded_vlm_h100_runtime_v4 import shard
            shards=[shard(rows,k) for k in range(x.WORLD_SIZE)]
            self.assertEqual(sorted(j for shard in shards for j in shard),rows)
            self.assertLessEqual(max(map(len,shards))-min(map(len,shards)),1)

    def test_costs_require_completed_attempts_and_include_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for key,gpus in [('prepare',0),('train',0)]+[(f'{phase}_node{n}',2) for phase in ('features','render') for n in range(x.NODES)]:
                x.c.write_new(out/f'attempts/{key}_test.started.json',{'gpus':gpus})
                x.c.write_new(out/f'attempts/{key}_test.ended.json',{'status':0,'seconds':10,'gpus':gpus,'used_gpu_seconds':gpus*10})
            costs=run.compute_costs(out)
            self.assertAlmostEqual(costs['used_gpu_hours'],80/3600)
            x.c.write_new(out/'attempts/features_node0_failed.started.json',{'gpus':2})
            with self.assertRaisesRegex(ValueError,'interrupted'): run.compute_costs(out)
            x.c.write_new(out/'attempts/features_node0_failed.ended.json',{'status':1,'seconds':5,'gpus':2,'used_gpu_seconds':10})
            self.assertAlmostEqual(run.compute_costs(out)['used_gpu_hours'],90/3600)

    def test_conditional_pipeline_selection_before_confirmation_no_classification_gate(self):
        # End-to-end tiny-dimensional CPU fixture; not a real experiment or GPU qualification.
        datasets={}; all_items=[]; gen=torch.Generator().manual_seed(79)
        for role,n in [('fit',5),('calibration',3),('confirmation',20)]:
            items=[{'key':f'{role}_{t}_{j}','role':role,'task':t,'seed':j,'view_index':0,
                    'row':{'crop_xy':[0,0],'source_group':f'{role}_{t}_{j}'}} for t in x.ORDER for j in range(n)]
            datasets[role]=(items,torch.randn(len(items),6,generator=gen),torch.randn(len(items),4,generator=gen),
                            torch.tensor([x.ORDER.index(i['task']) for i in items]))
            all_items.extend(items)
        legacy,_=small_model(); frozen=state.io.parameter_hash(dict(legacy.image.named_parameters()))
        old={'runtime':{'fixture':True},'frozen_image_hash':frozen,'selection':{'step':800}}
        b={'items':all_items,'counts':{'fixture':True}}
        loaded=[]
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'ROOT',Path(tmp)),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),\
             patch.object(x,'STEPS',8),patch.object(x,'CHECKS',(2,4,6,8)):
            out=run.ROOT/x.OUTPUT
            x.c.write_new(out/'receipt.json',{'fixture':True})
            x.c.write_new(run.ROOT/x.old.OUTPUT/'receipt.json',{'old_fixture':True})
            x.c.write_new(run.ROOT/x.old.REPORT,old)
            def load(_b,_index,role):
                if role=='confirmation': self.assertTrue((out/'selection.json').exists())
                loaded.append(role); return datasets[role]
            with patch.object(run,'bound',return_value=b),patch.object(run,'cpu_setup',return_value={'fixture':True}),\
                 patch.object(run,'feature_index',return_value={}),patch.object(run,'load_role',side_effect=load),\
                 patch.object(run,'model',side_effect=lambda b:conditional_small()),\
                 patch.object(x.old.parent,'model',side_effect=lambda b:small_model()),\
                 patch.object(x.old.state,'load_step',return_value={'vlm':legacy.vlm.state_dict()}),\
                 patch('shutil.disk_usage',return_value=type('Disk',(),{'free':200*1024**3})()):
                run.train(); run.verify_training()
                report=x.c.read(run.ROOT/x.TRAIN_REPORT)
                self.assertEqual(loaded,['fit','calibration','confirmation'])
                self.assertFalse(report['classification_blocks_restoration'])
                self.assertTrue(report['ready_for_prespecified_restoration'])
                self.assertEqual(len(list((out/'steps').glob('step_*.json'))),9)
                run.train(); self.assertEqual(loaded,['fit','calibration','confirmation'])


if __name__=='__main__': unittest.main()
