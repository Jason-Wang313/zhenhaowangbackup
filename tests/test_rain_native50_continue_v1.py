import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from smoke import existing_flux_replay_contract_v1 as c
from smoke import rain_native50_continue_contract_v1 as p
from smoke import rain_native50_continue_state_v1 as state
from smoke import rain_flow_formal_stage1_state_v1 as old_state
from smoke import train_foundir_rain_native50_continue_v1 as impl
from lowlight_restoration.lora import inject_lora


def row(i,dev=False):
    return dict(record_id=f"r{i}",source_group=f"s{i}",gt_sha256=f"gt{i}",lq_sha256=f"lq{i}",
        split="validation" if dev else "train",role="direct_endpoint_validation" if dev else "direct_endpoint_train",
        category="10Rain",composition_size="1",width="600",height="700",crop_x="23",crop_y="47",crop_xy=[44,94],
        lq_path=f"data/foundir/MillionIRData/Train/10Rain/LQ/{i}.png",gt_path=f"data/foundir/MillionIRData/Train/10Rain/GT/{i}.png")


def toy():
    torch.manual_seed(17);m=torch.nn.Sequential();m.add_module("fc",torch.nn.Linear(3,2))
    inject_lora(m,["fc"],2,2.,0.)
    return m,torch.optim.AdamW([v for v in m.parameters() if v.requires_grad],lr=1e-5,weight_decay=0.,fused=False)


def update(m,opt):
    opt.zero_grad();m(torch.randn(4,3)).square().mean().backward();opt.step();opt.zero_grad()


class ContinuationTests(unittest.TestCase):
    def test_config_only_authorized_changes(self):
        cfg=c.read(c.ROOT/p.CONFIG);p.validate_config(cfg)
        for key,val in (("steps",3000),("start_step",0),("learning_rate",1e-4),("probe_steps",[2000]),
                        ("initialization","fresh"),("loss_weights",{}),("global_batch",16)):
            with self.subTest(key=key),self.assertRaises(ValueError):p.validate_config({**cfg,key:val})

    def test_full_sampler_prefix_probe_identity_and_position(self):
        cfg=c.read(c.ROOT/p.CONFIG);old=c.read(c.ROOT/p.native.CONFIG)
        training=[row(i) for i in range(40)];dev=[row(i+1000,True) for i in range(100)]
        before=p.build_plan(training,dev,training[:32],old)
        after=p.build_plan(training,dev,training[:32],cfg)
        self.assertEqual(after[:6400],before[:6400]);self.assertEqual(after[64000:],before[6400:])
        self.assertEqual(after[6400]["sample_position"],6400);self.assertEqual(after[63999]["sample_position"],63999)
        p.validate_plan({"config":cfg,"rows":after})
        self.assertEqual(after,p.build_plan(list(reversed(training)),dev,training[:32],cfg))
        after[6400]["source_group"]=dev[0]["source_group"]
        with self.assertRaisesRegex(ValueError,"leaks"):p.validate_plan({"config":cfg,"rows":after})

    def test_state200_migration_and_two_resumes_preserve_next_update(self):
        model,opt=toy()
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            for step in range(201):
                if step:update(model,opt)
                old_state.save_step(a,model,opt,step=step,world=1,rank_rng=[state.rng_state("cpu")],receipt_hash="old",history=[{}]*step)
            parent,_=old_state.load_latest(a,"old",1)
            update(model,opt);expected=state.trainable_state(model);expected_opt=state.cpu_tree(opt.state_dict())
            model2,opt2=toy();state.restore(model2,opt2,parent,0,"cpu")
            state.save_step(b,model2,opt2,step=200,world=1,rank_rng=[state.rng_state("cpu")],receipt_hash="new",history=parent["history"])
            saved,_=state.load_latest(b,"new",1)
            self.assertEqual(saved["next_global_slot"],6400);self.assertEqual(saved["optimizer"]["param_groups"],parent["optimizer"]["param_groups"])
            model3,opt3=toy();state.restore(model3,opt3,saved,0,"cpu");update(model3,opt3)
            self.assertTrue(all(torch.equal(v,expected[n]) for n,v in state.trainable_state(model3).items()))
            for i,entry in expected_opt["state"].items():
                for k,v in entry.items():self.assertTrue(torch.equal(v,opt3.state_dict()["state"][i][k]))
            state.save_step(b,model3,opt3,step=201,world=1,rank_rng=[state.rng_state("cpu")],receipt_hash="new",history=[{}]*201)
            resumed,_=state.load_latest(b,"new",1);self.assertEqual(resumed["next_global_slot"],6432)
            with self.assertRaises(ValueError):state.load_latest(b,"old",1)
            with self.assertRaises(ValueError):state.load_latest(b,"new",2)
            (Path(b)/"steps"/"step_000200.json").unlink()
            with self.assertRaisesRegex(ValueError,"gap"):state.load_latest(b,"new",1)

    def test_no_fresh_state_or_overshoot_allowed(self):
        model,opt=toy()
        with tempfile.TemporaryDirectory() as tmp:
            for step in (0,199,2001):
                with self.assertRaises(ValueError):state.save_step(tmp,model,opt,step=step,world=1,rank_rng=[{}],receipt_hash="r",history=[{}]*step)

    def test_parent_state_requires_exact_progress_and_optimizer(self):
        payload={"step":200,"world":8,"next_global_slot":6400,"history":[{}]*200,"rank_rng":[{}]*8,
            "receipt_hash":p.PARENT["receipt_sha256"],"optimizer":{"param_groups":[{"lr":1e-5,"weight_decay":0}]}}
        record={"sha256":p.PARENT["checkpoint_sha256"]};p.validate_parent_state(payload,record)
        for key,val in (("step",25),("world",4),("next_global_slot",0),("history",[]),("receipt_hash","wrong")):
            with self.assertRaises(ValueError):p.validate_parent_state({**payload,key:val},record)
        with self.assertRaises(ValueError):p.validate_parent_state(payload,{"sha256":"wrong"})

    def test_sampler_and_probe_boundary(self):
        with mock.patch.object(impl,"cached") as get:
            for pos in (0,6399,64000,64131):
                with self.assertRaises(ValueError):impl.training_batch(None,{"rows":[]},pos,"cpu")
            get.assert_not_called()
        indices=[i for rank in range(8) for i in impl.probe_indices(rank)]
        self.assertEqual(sorted(indices),list(range(64000,64132)))
        self.assertEqual(impl.cache_marker(Path('/tmp/test'),64000),c.path(impl.ROOT,p.native.PARENT['path'])/'cache'/'image_6400.json')

    def test_evaluation_reuses_original_cache_index(self):
        r={"usage":"development"};parent={"rows":[None]*6400+[r]}
        receipt={"rows":[None]*64000+[r]}
        with mock.patch.object(p,"parent_receipt",return_value=parent),mock.patch.object(impl.native_impl,"cached",return_value="ok") as get:
            self.assertEqual(impl.cached(None,receipt,64000),"ok")
            self.assertEqual(get.call_args.args[2],6400)
        for i in (0,6399):
            with self.assertRaises(ValueError):impl.cached(None,receipt,i)

    def test_physics_training_loop_and_eval_seed_unchanged(self):
        before=ast.parse((c.ROOT/'smoke/train_foundir_rain_native50_warmstart_v1.py').read_text())
        after=ast.parse((c.ROOT/'smoke/train_foundir_rain_native50_continue_v1.py').read_text())
        for tree in (before,after):
            for n in ast.walk(tree):
                if isinstance(n,ast.Constant) and isinstance(n.value,str):n.value=n.value.replace('RAIN_NATIVE50_WARMSTART','RAIN_NATIVE50_CONTINUE')
        def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('new_model','load_pair','load_prompt','scores','quantize','summarize_entries'):
            self.assertEqual(ast.dump(fn(before,name)),ast.dump(fn(after,name)),name)
        a=fn(before,'probe_one');b=fn(after,'probe_one')
        for n in ast.walk(a):
            if isinstance(n,ast.Constant) and n.value==6400:n.value=64000
        self.assertEqual(ast.dump(a),ast.dump(b))
        loops=[]
        for tree in (before,after):
            loop=next(n for n in fn(tree,'train').body if isinstance(n,ast.For))
            loops.append([ast.dump(n) for n in loop.body])
        self.assertEqual(*loops)

    def test_segment_planner_and_interrupted_evaluation(self):
        cfg=c.read(c.ROOT/p.CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);self.assertEqual(impl.next_segment(out,cfg),(200,400))
            c.write_new(out/'steps'/'step_000400.json',{'step':400})
            self.assertEqual(impl.next_segment(out,cfg),(400,400))
            c.write_new(out/'milestones'/'step_000400.json',{'pause_for_review':False})
            self.assertEqual(impl.next_segment(out,cfg),(400,700))
            c.write_new(out/'steps'/'step_000700.json',{'step':700})
            c.write_new(out/'milestones'/'step_000700.json',{'pause_for_review':True})
            self.assertIsNone(impl.next_segment(out,cfg))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);c.write_new(out/'steps'/'step_002000.json',{'step':2000})
            self.assertEqual(impl.next_segment(out,cfg),(2000,2000))

    def test_final_check_is_light_and_no_formal_autostart(self):
        text=(c.ROOT/'smoke/run_foundir_rain_native50_continue_v1_gpu.sh').read_text()
        for fragment in ('--nnodes=4 --nproc_per_node=2','WALL_CAP=28800','29690','summarize','full_CPU_audit=false'):
            self.assertIn(fragment,text)
        self.assertNotIn('verify_foundir',text)
        self.assertNotIn('for index in range',__import__('inspect').getsource(impl.verify_light))

    def test_pause_requires_two_consecutive_both_metric_nonimprovements(self):
        before={"psnr":31.,"lpips":.08};bad={"psnr":30.,"lpips":.09}
        first=impl.quality_change(before,bad,0);self.assertFalse(first['pause_for_review'])
        self.assertTrue(impl.quality_change(bad,bad,first['no_improvement_streak'])['pause_for_review'])
        for improved in ({"psnr":32.,"lpips":.09},{"psnr":30.,"lpips":.07}):
            decision=impl.quality_change(before,improved,1)
            self.assertEqual(decision['no_improvement_streak'],0);self.assertFalse(decision['pause_for_review'])


if __name__=='__main__':unittest.main()
