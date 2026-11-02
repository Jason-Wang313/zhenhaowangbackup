import ast
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from smoke import existing_flux_replay_contract_v1 as c
from smoke import rain_native50_warmstart_contract_v1 as contract
from smoke import rain_flow_formal_stage1_state_v1 as state
from smoke import train_foundir_rain_native50_warmstart_v1 as impl
from lowlight_restoration.lora import inject_lora


def row(index,development=False):
    return dict(record_id=f"r{index}",source_group=f"s{index}",gt_sha256=f"gt{index}",lq_sha256=f"lq{index}",
                split="validation" if development else "train",category="10Rain",composition_size="1",
                role="direct_endpoint_validation" if development else "direct_endpoint_train",
                width="600",height="700",crop_x="23",crop_y="47",crop_xy=[44,94],
                lq_path=f"data/foundir/MillionIRData/Train/10Rain/LQ/{index}.png",
                gt_path=f"data/foundir/MillionIRData/Train/10Rain/GT/{index}.png")


def toy():
    torch.manual_seed(17)
    model=torch.nn.Sequential();model.add_module("fc",torch.nn.Linear(3,2))
    inject_lora(model,["fc"],2,2.,0.)
    return model


class NativeWarmstartTests(unittest.TestCase):
    def test_frozen_full_pool_scope_and_only_allowed_config_changes(self):
        cfg=c.read(c.ROOT/contract.CONFIG);contract.validate_config(cfg)
        for key,value in (("steps",201),("training_sources",32),("development_sources",8),("learning_rate",1e-4),
                          ("wall_cap_seconds",7200),("initialization","qualification_step25")):
            with self.subTest(key=key),self.assertRaises(ValueError):contract.validate_config({**cfg,key:value})

    def test_sampler_determinism_separation_and_complete_batch_positions(self):
        cfg=c.read(c.ROOT/contract.CONFIG)
        training=[row(i) for i in range(40)];development=[row(1000+i,True) for i in range(100)]
        a=contract.build_plan(training,development,training[:32],cfg)
        random.seed(999)
        b=contract.build_plan(list(reversed(training)),development,training[:32],cfg)
        self.assertEqual(a,b);self.assertEqual(len(a),6532)
        contract.validate_plan({"config":cfg,"rows":a})
        self.assertGreater(len({r["record_id"] for r in a[:6400]}),32)
        self.assertEqual({r["augmentation_code"] for r in a[:6400]},set(range(8)))
        for step in (0,49,50,99,199):
            positions=[step*32+micro*8+rank for micro in range(4) for rank in range(8)]
            self.assertEqual(positions,list(range(step*32,(step+1)*32)))
        self.assertTrue(all(r["crop_xy"]==[23,47] and r["augmentation_code"]==0 for r in a[6432:]))
        a[0]={**a[0],"source_group":a[-1]["source_group"]}
        with self.assertRaisesRegex(ValueError,"leaks"):contract.validate_plan({"config":cfg,"rows":a})

    def test_row_scope_padding_and_frozen_development_crop(self):
        r={**row(1),"usage":"train_update","sample_position":0,"augmentation_code":0,"height":"300","crop_xy":[0,0]}
        contract.validate_row(r)
        for change in ({"split":"validation"},{"lq_path":"data/foundir/Test/a.png"},{"crop_xy":[0,1]},
                       {"usage":"train_probe"},{"augmentation_code":8}):
            with self.assertRaises(ValueError):contract.validate_row({**r,**change})
        d={**row(2,True),"usage":"development","sample_position":None,"augmentation_code":0,"crop_xy":[23,47]}
        contract.validate_row(d)
        with self.assertRaises(ValueError):contract.validate_row({**d,"crop_xy":[44,94]})

    def test_training_cannot_read_eval_cache(self):
        with mock.patch.object(impl,"cached") as cached:
            for pos in (-1,6400,6432,6531):
                with self.assertRaises(ValueError):impl.training_batch(None,{"rows":[]},pos,"cpu")
            with self.assertRaises(ValueError):impl.training_batch(None,{"rows":[{"usage":"development"}]},0,"cpu")
            cached.assert_not_called()

    def test_materialization_matches_frozen_crop_and_paired_padding(self):
        import numpy as np
        from PIL import Image
        for development,height,code in ((True,700,0),(False,300,5)):
            with self.subTest(development=development),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                image=(np.arange(height*600*3,dtype=np.uint32).reshape(height,600,3)%256).astype(np.uint8)
                r={**row(1,development),"usage":"development" if development else "train_update",
                   "sample_position":None if development else 0,"augmentation_code":code,"height":str(height),
                   "crop_xy":[23,47] if development else [0,0]}
                for side in ("lq","gt"):
                    path=root/r[side+"_path"];path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(image).save(path);r[side+"_sha256"]=c.sha(path)
                with mock.patch.object(impl,"ROOT",root):lq,gt=impl.load_pair(r,"cpu")
                self.assertTrue(torch.equal(lq,gt));self.assertEqual(tuple(lq.shape),(1,3,512,512))
                if development:
                    expected=torch.from_numpy(image[47:559,23:535].copy()).permute(2,0,1).float()/127.5-1
                    self.assertTrue(torch.equal(lq[0],expected))
                else:
                    padded=np.pad(image,((0,212),(0,0),(0,0)),mode="reflect")[:512,:512]
                    expected=torch.from_numpy(padded.copy()).permute(2,0,1).float()/127.5-1
                    expected=torch.flip(torch.rot90(expected,1,(-2,-1)),(-1,))
                    self.assertTrue(torch.equal(lq[0],expected))

    def test_probe_sharding_complete_no_duplicates(self):
        all_indices=[i for rank in range(8) for i in impl.probe_indices(rank)]
        self.assertEqual(sorted(all_indices),list(range(6400,6532)))
        self.assertEqual([len(impl.probe_indices(r)) for r in range(8)],[17]*4+[16]*4)
        with self.assertRaises(ValueError):impl.probe_indices(0,world=4)

    def test_probe_preserves_rng_after_partial_failure(self):
        def fake(*args):
            random.random();torch.rand(10)
            if args[-3]==6408:raise RuntimeError("interrupted probe")
        before=state.rng_state("cpu")
        with mock.patch.object(impl,"probe_one",side_effect=fake),self.assertRaises(RuntimeError):
            impl.probe(None,None,None,None,None,None,0,"cpu",50)
        after=state.rng_state("cpu")
        self.assertEqual(before["python"],after["python"])
        self.assertTrue(torch.equal(before["torch_cpu"],after["torch_cpu"]))

    def test_scores_match_legacy_development_arithmetic(self):
        gt=torch.rand(1,3,8,8)*2-1;u8=torch.randint(0,256,gt.shape,dtype=torch.uint8)
        class Metric:
            def __call__(self,a,b):return (a-b).abs().mean()
            def reset(self):pass
        from smoke.train_foundir_blur_only_flux2_overfit_ddp import psnr,lpips_value
        q=u8.float()/127.5-1
        got=impl.scores(Metric(),u8,gt)
        self.assertEqual(got["psnr"],psnr(gt,q))
        self.assertEqual(got["lpips"],lpips_value(Metric(),gt,q))

    def test_summary_does_not_mix_train_with_development(self):
        entries=[]
        for i in range(132):
            usage="train_probe" if i<32 else "development"
            metrics={"psnr":50. if i<32 else 34.19,"lpips":.01 if i<32 else .07}
            entries.append({"identity":{"row":{"record_id":str(i),"usage":usage}},"training_fit_only":i<32,
                            "metrics":{"student_50_step":metrics,"teacher_one_step":metrics.copy()}})
        summary=impl.summarize_entries(entries)
        self.assertFalse(summary["development_gate_pass"])
        self.assertAlmostEqual(summary["splits"]["development"]["means"]["student_50_step"]["psnr"],34.19)
        with self.assertRaises(ValueError):impl.summarize_entries(entries[:-1])
        with self.assertRaises(ValueError):impl.summarize_entries(entries[:-1]+[entries[-2]])

    def test_200_step_state_exact_next_update_and_bounds(self):
        model=toy();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.)
        def update(m,o):
            o.zero_grad();m(torch.randn(2,3)).square().mean().backward();o.step()
        with tempfile.TemporaryDirectory() as tmp:
            for step in range(201):
                if step:update(model,opt)
                state.save_step(tmp,model,opt,step=step,world=1,rank_rng=[state.rng_state("cpu")],receipt_hash="r",history=[{}]*step)
            saved,_=state.load_latest(tmp,"r",1)
            self.assertEqual(saved["next_global_slot"],6400);self.assertTrue(saved["may_seed_formal"])
            update(model,opt);expected=state.trainable_state(model)
            other=toy();o=torch.optim.AdamW([p for p in other.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.)
            state.restore(other,o,saved,0,"cpu");update(other,o)
            self.assertTrue(all(torch.equal(expected[n],t) for n,t in state.trainable_state(other).items()))
            with self.assertRaises(ValueError):state.save_step(tmp,model,opt,step=201,world=1,rank_rng=[state.rng_state("cpu")],receipt_hash="r",history=[{}]*201)

    def test_only_initialization_changes_not_forward_loss_or_sampling(self):
        original=ast.parse((c.ROOT/"smoke/train_foundir_rain_flow_formal_stage1_v1.py").read_text())
        updated=ast.parse((c.ROOT/"smoke/train_foundir_rain_native50_warmstart_v1.py").read_text())
        for name in ("load_pair","load_prompt","runtime_compatibility","check_runtimes","scores",
                     "quantize","probe_indices","training_batch","probe","summarize_entries"):
            a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in updated.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)
        source=(c.ROOT/"smoke/train_foundir_rain_native50_warmstart_v1.py").read_text()
        self.assertIn('receipt["native_checkpoint"]["path"]',source)
        self.assertNotIn('receipt["rain"]["checkpoint"]["path"]',source)
        self.assertNotIn('restore_endpoint(',source)


    def test_launcher_limits_and_no_automatic_followon(self):
        text=(c.ROOT/"smoke/run_foundir_rain_native50_warmstart_v1_gpu.sh").read_text()
        for fragment in ("--nnodes=4 --nproc_per_node=2","CUDA_VISIBLE_DEVICES=0,1","WALL_CAP=3600","29688"):
            self.assertIn(fragment,text)
        self.assertNotIn("rain_flow_trainfit",text)

    def test_configuration_does_not_silently_choose_other_initialization(self):
        cfg=c.read(c.ROOT/contract.CONFIG)
        for key in ("native_checkpoint","cache_parent","native_manifest"):
            value=copy.deepcopy(cfg);value[key]["sha256" if key!="cache_parent" else "receipt_sha256"]="wrong"
            with self.assertRaises(ValueError):contract.validate_config(value)
        old=c.read(c.ROOT/contract.formal.CONFIG)
        for key in ("seed","probe_seed","selection_seed","learning_rate","loss_weights","probe_steps","steps"):
            self.assertEqual(cfg[key],old[key])

    def test_native_manifest_excludes_current_development_and_matches_crops(self):
        train=[row(i) for i in range(1580)]
        dev=[row(2000+i,True) for i in range(100)]
        contract.audit_native_rows(train+dev,dev)
        bad=copy.deepcopy(train+dev);bad[0]["source_group"]=dev[0]["source_group"]
        with self.assertRaisesRegex(ValueError,"trained on"):contract.audit_native_rows(bad,dev)
        bad=copy.deepcopy(train+dev);bad[-1]["crop_x"]="24"
        with self.assertRaisesRegex(ValueError,"crops/files"):contract.audit_native_rows(bad,dev)
        with self.assertRaises(ValueError):contract.audit_native_rows((train+dev)[:-1],dev)

    def test_cache_binding_changes_rejected(self):
        cfg=c.read(c.ROOT/contract.formal.CONFIG)
        train=[row(i) for i in range(40)];dev=[row(1000+i,True) for i in range(100)]
        rows=contract.build_plan(train,dev,train[:32],cfg)
        parent={"config":cfg,"rows":rows,"selection_digest":c.digest(rows),
                "rain":{"teacher":"original"},"metric":{},"model":{},"lora":{}}
        receipt=copy.deepcopy(parent);receipt["config"]=c.read(c.ROOT/contract.CONFIG)
        contract.validate_parent(parent,receipt)
        for key in ("rain","metric","model","lora"):
            bad=copy.deepcopy(receipt);bad[key]={"changed":True}
            with self.assertRaises(ValueError):contract.validate_parent(parent,bad)
        bad=copy.deepcopy(receipt);bad["rows"][0]["augmentation_code"]=(bad["rows"][0]["augmentation_code"]+1)%8
        with self.assertRaises(ValueError):contract.validate_parent(parent,bad)

    def test_cache_reuse_uses_producer_identity_not_new_identity(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);donor=c.path(root,contract.PARENT["path"])
            r={**row(1),"usage":"train_update"};parent={"rows":[r]}
            marker=donor/"cache"/"image_000.json"
            donor.mkdir(parents=True)
            data={k:torch.zeros(1,3,512,512) for k in ("lq","gt","teacher_rgb")}
            data.update(clean_latent=torch.zeros(1,4,2,2),reference_latent=torch.ones(1,4,2,2))
            file=donor/"tensor.safetensors";save_file(data,str(file))
            c.write_new(marker,{"teacher":"frozen_original_Rain_FLUX_step4000","tensor":file.name,
                "files":{file.name:c.sha(file)},"identity":{"receipt_sha256":contract.PARENT["receipt_sha256"],
                "index":0,"row":r,"usage":r["usage"]}})
            with mock.patch.object(impl,"ROOT",root),mock.patch.object(contract,"cache_receipt",return_value=parent):
                result=impl.cached(root/"new",{"rows":[r]},0)
                self.assertEqual(result[1],marker)
                self.assertTrue(all(torch.equal(result[0][k],v) for k,v in data.items()))
                with self.assertRaisesRegex(ValueError,"row mismatch"):
                    impl.cached(root/"new",{"rows":[{**r,"record_id":"changed"}]},0)
                with file.open("ab") as stream:stream.write(b"corruption")
                with self.assertRaisesRegex(ValueError,"saved tensor changed"):
                    impl.cached(root/"new",{"rows":[r]},0)

    def test_cache_shards_complete_without_generation(self):
        receipt={"rows":[{}]*6532}
        for rank in range(8):
            with mock.patch.object(contract,"cache_receipt",return_value={}),mock.patch.object(contract,"validate_parent"),mock.patch.object(impl,"cached") as read:
                impl.prepare_cache(None,receipt,rank,"cpu")
                self.assertEqual([call.args[2] for call in read.call_args_list],list(range(rank,6532,8)))

    def test_frozen_teacher_metrics_cannot_change_with_student_initialization(self):
        original=ast.parse((c.ROOT/"smoke/train_foundir_rain_flow_formal_stage1_v1.py").read_text())
        updated=ast.parse((c.ROOT/"smoke/train_foundir_rain_native50_warmstart_v1.py").read_text())
        # Probe implementation must differ only in log prefix: same flow sampler/seed/metric.
        for tree in (original,updated):
            for node in ast.walk(tree):
                if isinstance(node,ast.Constant) and isinstance(node.value,str):
                    node.value=node.value.replace("RAIN_FLOW_FORMAL_STAGE1","RAIN_NATIVE50_WARMSTART")
        a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=="probe_one")
        b=next(n for n in updated.body if isinstance(n,ast.FunctionDef) and n.name=="probe_one")
        self.assertEqual(ast.dump(a),ast.dump(b))

    def test_gpu_launcher_exits_before_full_cpu_verification(self):
        source=(c.ROOT/"smoke/run_foundir_rain_native50_warmstart_v1_gpu.sh").read_text()
        self.assertIn("CPU_VERIFICATION_PENDING=true",source)
        self.assertNotIn('step "$PYTHON" -u "$SCRIPT" verify',source)
        self.assertIn('wait done',source)


if __name__=="__main__":unittest.main()
