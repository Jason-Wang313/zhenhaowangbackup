import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch
from torch import nn
from lowlight_restoration.flow import build_reference_flow_batch
from lowlight_restoration.rain_flow_adaptation import endpoint_estimate,enable_native_checkpointing,sample_index,training_objective
from lowlight_restoration.lora import inject_lora
from smoke import existing_flux_replay_contract_v1 as c
from smoke import rain_flow_adaptation_contract_v1 as contract
from smoke import rain_flow_adaptation_state_v1 as saved


def pack(x):
    return x.flatten(2).transpose(1,2),torch.zeros(x.shape[0],x.shape[2]*x.shape[3],4,dtype=torch.int64)


def toy():
    torch.manual_seed(17)
    model=nn.Sequential();model.add_module("fc",nn.Linear(3,2))
    inject_lora(model,["fc"],2,2.,0.)
    return model


class RainFlowTests(unittest.TestCase):
    def test_flow_direction_and_endpoint_at_multiple_times(self):
        clean=torch.randn(4,3,2,2);reference=torch.randn_like(clean);noise=torch.randn_like(clean)
        t=torch.tensor([.0001,.1,.5,.9999])
        flow=build_reference_flow_batch(clean,reference,noise,t,pack,10)
        estimate=endpoint_estimate(flow.model_tokens[:,:4],flow.target_velocity,t,clean.shape)
        torch.testing.assert_close(estimate,clean,atol=3e-7,rtol=1e-6)
        self.assertTrue(torch.equal(flow.model_ids[:,4:,0],torch.full((4,4),10)))
        self.assertFalse(torch.allclose(flow.target_velocity,pack(reference-clean)[0]))

    def test_teacher_does_not_supply_velocity_label(self):
        class Model(nn.Module):
            def __init__(self):super().__init__();self.bias=nn.Parameter(torch.zeros(1,1,3))
            def forward(self,x,**kwargs):return x.float()*0+self.bias
        class AE:
            def decode(self,x):return x
        model=Model();clean=torch.zeros(1,3,2,2);noise=torch.zeros_like(clean);t=torch.tensor([.5])
        batch={"clean_latent":clean,"reference_latent":clean+.2,"gt":clean,"teacher_rgb":clean+.1}
        weights=c.read(c.ROOT/contract.CONFIG)["loss_weights"]
        metric=lambda a,b:(a-b).square().mean()
        a,terms=training_objective(model,AE(),batch,None,None,pack,metric,weights,noise=noise,timesteps=t)
        self.assertEqual(float(terms["flow_mse"]),0.)
        a.backward();self.assertGreater(float(model.bias.grad.abs().sum()),0.)
        b,new=training_objective(model,AE(),{**batch,"teacher_rgb":clean+.4},None,None,pack,metric,weights,noise=noise,timesteps=t)
        self.assertEqual(float(new["flow_mse"]),0.)
        self.assertGreater(float(b),float(a))

    def test_native_flux_checkpoint_matches_real_tiny_model_gradients(self):
        from flux2.model import Flux2,Flux2Params
        model=Flux2(Flux2Params(in_channels=4,context_in_dim=8,hidden_size=16,num_heads=2,
                    depth=1,depth_single_blocks=1,axes_dim=[2,2,2,2],use_guidance_embed=False))
        inject_lora(model,["double_blocks.*.img_attn.qkv","single_blocks.*.linear1"],2,2.,0.)
        other=copy.deepcopy(model);enable_native_checkpointing(other)
        model.train();other.train()
        args={"x":torch.randn(1,8,4),"x_ids":torch.zeros(1,8,4,dtype=torch.int64),
              "timesteps":torch.tensor([.4]),"ctx":torch.randn(1,2,8),"ctx_ids":torch.zeros(1,2,4,dtype=torch.int64),"guidance":None}
        expected=model(**args);actual=other(**args)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        expected.square().mean().backward();actual.square().mean().backward()
        for (n,a),(m,b) in zip(model.named_parameters(),other.named_parameters()):
            self.assertEqual(n,m)
            if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)
        with self.assertRaises(ValueError):enable_native_checkpointing(other)

    def test_global_batch_exactly_covers_32_and_resume_schedule(self):
        for step in (0,1,24):
            indices=[sample_index(step,m,r,42) for m in range(4) for r in range(8)]
            self.assertEqual(sorted(indices),list(range(32)))
            self.assertEqual(indices,[sample_index(step,m,r,42) for m in range(4) for r in range(8)])
        self.assertNotEqual([sample_index(0,m,0,42) for m in range(4)],[sample_index(1,m,0,42) for m in range(4)])

    def test_state_resume_reproduces_next_update(self):
        model=toy();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.,fused=False)
        def update(m,o):
            o.zero_grad();m(torch.randn(4,3)).square().mean().backward();o.step();o.zero_grad()
        with tempfile.TemporaryDirectory() as tmp:
            saved.save_step(tmp,model,optimizer,step=0,world=1,rank_rng=[saved.rng_state("cpu")],receipt_hash="r",history=[])
            update(model,optimizer)
            saved.save_step(tmp,model,optimizer,step=1,world=1,rank_rng=[saved.rng_state("cpu")],receipt_hash="r",history=[{}])
            state,record=saved.load_latest(tmp,"r",1)
            update(model,optimizer);expected=saved.trainable_state(model)
            other=toy();opt=torch.optim.AdamW([p for p in other.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.,fused=False)
            saved.restore(other,opt,state,0,"cpu");update(other,opt)
            for n,v in saved.trainable_state(other).items():self.assertTrue(torch.equal(v,expected[n]))
            with self.assertRaises(ValueError):saved.load_latest(tmp,"r",2)
            with self.assertRaises(ValueError):saved.load_latest(tmp,"different",1)
            (Path(tmp)/record["file"]).write_bytes(b"broken")
            with self.assertRaises(ValueError):saved.load_latest(tmp,"r",1)

    def test_checkpoint_gap_and_immutable_records_rejected(self):
        model=toy();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            saved.save_step(tmp,model,opt,step=0,world=1,rank_rng=[saved.rng_state("cpu")],receipt_hash="r",history=[])
            with self.assertRaises(ValueError):
                saved.save_step(tmp,model,opt,step=0,world=1,rank_rng=[saved.rng_state("cpu")],receipt_hash="r",history=[])
            c.write_new(Path(tmp)/"steps/step_000002.json",{"step":2})
            with self.assertRaises(ValueError):saved.load_latest(tmp,"r",1)

    def test_only_lora_is_saved(self):
        model=toy();values=saved.trainable_state(model)
        self.assertEqual(set(values),{"fc.lora_A.weight","fc.lora_B.weight"})
        model.fc.base.weight.requires_grad_(True)
        with self.assertRaises(ValueError):saved.trainable_state(model)

    def test_scope_and_shell(self):
        cfg=c.read(c.ROOT/contract.CONFIG);contract.validate_config(cfg)
        for k,v in (("steps",1000),("augmentation",True),("development_pixels_allowed",True),("global_batch",8)):
            with self.assertRaises(ValueError):contract.validate_config({**cfg,k:v})
        text=(c.ROOT/"smoke/run_foundir_rain_flow_adaptation_v1_gpu.sh").read_text()
        for fragment in ("--nnodes=4 --nproc_per_node=2","CUDA_VISIBLE_DEVICES=0,1","WALL_CAP=3600","29684","train_foundir_rain_flow_adaptation_v1.py"):
            self.assertIn(fragment,text)
        self.assertNotIn('"$SCRIPT" formal',text)

    def test_training_requires_matching_node_runtimes(self):
        from smoke.train_foundir_rain_flow_adaptation_v1 import check_runtimes
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            value={"gpu_name":"H200","torch":"test","metrics":{"head":"same"}}
            for n in range(4):c.write_new(root/f"runtime_node{n}.json",value)
            self.assertEqual(check_runtimes(root),value["metrics"])
            with mock.patch.object(c,"read",side_effect=[value,value,value,{**value,"torch":"different"}]):
                with self.assertRaises(ValueError):check_runtimes(root)

    def test_manifest_source_separation_and_fixed_selection(self):
        def row(i,train=True):
            return {"record_id":f"record{i}","source_group":f"group{i%5780}" if train else f"dev{i}",
                "gt_sha256":f"gt{i}","lq_sha256":f"lq{i}","category":"10Rain","composition_size":"1",
                "split":"train" if train else "validation","role":"direct_endpoint_train" if train else "direct_endpoint_validation",
                "width":"600","height":"700","lq_path":f"data/foundir/MillionIRData/Train/10Rain/LQ/{i}.png",
                "gt_path":f"data/foundir/MillionIRData/Train/10Rain/GT/{i}.png"}
        train=[row(i) for i in range(38334)];dev=[row(38334+i,False) for i in range(100)]
        paths=[mock.MagicMock(),mock.MagicMock()]
        with mock.patch.object(contract.csv,"DictReader",side_effect=[train+dev,train+dev]):
            selected=contract.select_rows(*paths,42)
        self.assertEqual(len(selected),32);self.assertEqual(len({r["source_group"] for r in selected}),32)
        self.assertTrue(all(r["crop_xy"]==[44,94] for r in selected))
        contaminant={**dev[0],"source_group":selected[0]["source_group"]}
        with mock.patch.object(contract.csv,"DictReader",side_effect=[train+dev,train+dev+[contaminant]]):
            with self.assertRaises(ValueError):contract.select_rows(*paths,42)


if __name__=="__main__":unittest.main()
