import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from lowlight_restoration.protocol_routed_flux import (ExpertProtocol, ProtocolRouter, RouteBoundFlux,
    install_protocol_bank, probabilities, restore_protocol_routed, validate_weights)
from lowlight_restoration.existing_flux_experts import FLOW_CONTRACT
from smoke import existing_flux_replay_contract_v1 as c
from smoke import protocol_routed_flux_contract_v1 as contract


class Toy(nn.Module):
    def __init__(self,dtype=torch.float32):
        super().__init__(); self.fc=nn.Linear(3,4,bias=False,dtype=dtype)
    def forward(self,x):
        return self.fc(x)


def make_model(dtype=torch.float32):
    torch.manual_seed(9)
    model=Toy(dtype)
    experts=[(name,{"fc.lora_A.weight":torch.randn(2,3,dtype=dtype),
                    "fc.lora_B.weight":torch.randn(4,2,dtype=dtype)},2.,FLOW_CONTRACT)
             for name in ("blur","noise","haze","lowlight")]
    names=install_protocol_bank(model,modules=["fc"],expert_states=experts)
    return model,names


def protocols():
    return ProtocolRouter([ExpertProtocol(n,"original "+n,a,n!="haze")
                           for n,a in (("blur",.75),("noise",.30),("haze",1.),("lowlight",.925))])


class RoutedFluxTests(unittest.TestCase):
    def test_one_hot_matches_original_bank_fp32_bf16(self):
        for dtype in (torch.float32,torch.bfloat16):
            model,names=make_model(dtype);x=torch.randn(2,5,3,dtype=dtype)
            for index in range(4):
                w=torch.eye(4)[index:index+1]
                result=RouteBoundFlux(model,names,w)(x)
                self.assertTrue(torch.equal(result,model.fc.bank.forward_expert(x,index)))
            self.assertTrue(torch.isnan(model.fc.route_weights).all())

    def test_routes_do_not_leak_between_calls(self):
        model,names=make_model();x=torch.randn(1,5,3)
        a=RouteBoundFlux(model,names,torch.tensor([[.9,.1,0.,0.]]))
        b=RouteBoundFlux(model,names,torch.tensor([[0.,0.,.9,.1]]))
        before=a(x); b(x)
        self.assertTrue(torch.equal(before,a(x)))
        with self.assertRaises(ValueError):
            model(x)  # Missing explicit route fails, never silently picks expert0.

    def test_explicit_checkpoint_recomputation_preserves_gate_gradient(self):
        model,names=make_model();x=torch.randn(1,5,3)
        logits=torch.randn(1,4,requires_grad=True)
        def f(value,scores):
            return RouteBoundFlux(model,names,probabilities(scores))(value)
        reference=torch.autograd.grad(f(x,logits).square().sum(),logits)[0]
        actual=torch.autograd.grad(checkpoint(f,x,logits,use_reentrant=False).square().sum(),logits)[0]
        self.assertTrue(torch.equal(reference,actual))
        self.assertGreater(float(actual.abs().sum()),0.)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_cfg_batch_order_and_invalid_layout(self):
        model,names=make_model()
        x=torch.randn(4,5,3);w=torch.tensor([[1.,0,0,0],[0,1.,0,0]])
        actual=RouteBoundFlux(model,names,w)(x)
        expected=model.fc.bank(x,w.repeat(2,1))
        self.assertTrue(torch.equal(actual,expected))
        with self.assertRaises(ValueError):
            RouteBoundFlux(model,names,w)(x[:3])

    def test_anchor_keeps_original_alpha_not_weighted_alpha(self):
        rules=protocols();w=torch.tensor([[.1,.8,.05,.05]])
        selected=rules.anchor(w)
        self.assertEqual(selected.name,"noise");self.assertEqual(selected.alpha,.30)
        self.assertNotAlmostEqual(selected.alpha,float(w@torch.tensor([.75,.30,1.,.925])))
        self.assertEqual(rules.anchor(torch.full((1,4),.25)).name,"blur")

    def test_quantization_matches_all_original_protocols(self):
        from smoke.replay_foundir_existing_flux_experts_v1 import quantize
        rules=protocols();lq,raw=[torch.rand(1,3,16,16)*2-1 for _ in range(2)]
        for i,p in enumerate(rules.protocols):
            self.assertTrue(torch.equal(rules.output(lq,raw,torch.eye(4)[i:i+1]),quantize(p.name,lq,raw,p.alpha)))

    def test_zero_vlm_delta_is_identity_and_invalid_routes_reject(self):
        a=torch.randn(3,4)
        self.assertTrue(torch.equal(probabilities(a),probabilities(a,torch.zeros_like(a))))
        for w in (torch.zeros(1,4),torch.tensor([[2.,-1,0,0]]),torch.full((1,4),float("nan"))):
            with self.assertRaises(ValueError):validate_weights(w,4)
        with self.assertRaises(ValueError):protocols().anchor(torch.ones(2,4)/4)

    def test_coupled_restore_uses_one_call_and_correct_protocol(self):
        model,names=make_model();rules=protocols();w=torch.tensor([[0.,1,0,0]])
        lq=torch.zeros(1,3,8,8);raw=torch.ones_like(lq)*.4
        prompts={p.name:{"cfg_context":p.prompt,"cfg_ids":p.name} for p in rules.protocols}
        calls=[]
        def fake_restore(bound,ae,image,context,ids,pack,**kwargs):
            calls.append((context,ids,kwargs));self.assertTrue(torch.equal(bound.weights,w));return raw
        _,image,plan=restore_protocol_routed(model,names,rules,w,None,lq,prompts,None,fake_restore,seed=17)
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][1],"noise")
        self.assertEqual(plan["alpha"],.30)
        self.assertTrue(torch.equal(image,rules.output(lq,raw,w)))
        wrong=ProtocolRouter(list(reversed(rules.protocols)))
        with self.assertRaises(ValueError):
            restore_protocol_routed(model,names,wrong,w,None,lq,prompts,None,fake_restore,seed=17)

    def test_caller_mutation_does_not_change_bound_plan(self):
        model,names=make_model();w=torch.tensor([[0.,1,0,0]])
        bound=RouteBoundFlux(model,names,w);w.zero_()
        self.assertEqual(float(bound.weights[0,1]),1.)

    def test_eight_source_shards_have_parent_rank_assignment(self):
        tasks=list(c.CATEGORIES);full={t:list(range(32)) for t in tasks};small={t:v[:2] for t,v in full.items()}
        seen=[]
        for rank in range(8):
            task,rows=c.assigned(tasks,small,rank,8)
            self.assertEqual(rows,c.assigned(tasks,full,rank,8)[1][:1])
            seen.append((task,rows[0]))
        self.assertEqual(len(set(seen)),8)

    def test_frozen_qualification_scope(self):
        cfg=c.read(c.ROOT/contract.CONFIG);contract.validate_config(cfg)
        for key,value in (("training",True),("near_one_hot_primary_weight",.5),("records_per_task",32)):
            with self.assertRaises(ValueError):contract.validate_config({**cfg,key:value})
        self.assertNotEqual(contract.run_path(Path("/tmp"),"x"),c.run_path(Path("/tmp"),"x"))

    def test_three_arm_summary_keeps_soft_harm_visible(self):
        rows=[]
        for task in c.CATEGORIES:
            for i in range(2):
                rows.append({"identity":{"key":f"{task}{i}","task":task},"metrics":{
                    "original":{"psnr":30.,"lpips":.1},"one_hot":{"psnr":30.,"lpips":.1},
                    "near_one_hot":{"psnr":29.,"lpips":.09}}})
        keys=[r["identity"]["key"] for r in rows]
        result=contract.aggregate(rows,keys)
        self.assertEqual(result["noise"]["near_one_hot_harmed_images"],2)
        self.assertEqual(result["noise"]["near_one_hot_minus_original"]["psnr"],-1.)
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):contract.aggregate(bad,keys)
        bad=copy.deepcopy(rows);bad[0]["metrics"]["one_hot"]["psnr"]+=.1
        with self.assertRaises(ValueError):contract.aggregate(bad,keys)
        bad=copy.deepcopy(rows);bad[0]["metrics"]["near_one_hot"]["lpips"]=float("nan")
        with self.assertRaises(ValueError):contract.aggregate(bad,keys)

    def test_tensor_parity_requires_uint8_not_just_float_tolerance(self):
        from smoke.qualify_foundir_protocol_routed_flux_v1 import parity,check_tensors
        a={"decoded":torch.zeros(1,3,2,2),"uint8":torch.zeros(1,3,2,2,dtype=torch.uint8)}
        self.assertTrue(parity(a,a,2)["pass"])
        b={k:v.clone() for k,v in a.items()};b["uint8"][0,0,0,0]=1
        self.assertFalse(parity(a,b,2)["pass"])
        b["decoded"][0,0,0,0]=float("nan")
        with self.assertRaises(ValueError):check_tensors(b,2)

    def test_four_node_launcher_scope_and_no_prompt_regeneration(self):
        text=(c.ROOT/"smoke/run_foundir_protocol_routed_flux_v1_gpu.sh").read_text()
        for fragment in ("--nnodes=4 --nproc_per_node=2","CUDA_VISIBLE_DEVICES=0,1","WALL_CAP=1800",
                         "foundir_protocol_routed_flux_v1/","$((2*elapsed))","publish done","wait verified"):
            self.assertIn(fragment,text)
        self.assertNotIn('"$SCRIPT" prompts',text)
        self.assertNotIn('"$SCRIPT" train',text)
        self.assertIn("29683",text)

    def test_verification_recomputes_tensors_and_rejects_forged_pass(self):
        from safetensors.torch import save_file
        from smoke import qualify_foundir_protocol_routed_flux_v1 as q
        cfg=c.read(c.ROOT/contract.CONFIG)
        receipt={"config":cfg,"experts":{p.name:{"prompt":p.prompt,"inference":{"frozen_alpha":p.alpha}}
                 for p in protocols().protocols},"rows":{t:[{"record_id":f"{t}{i}","gt_sha256":"fixed"} for i in range(2)]
                 for t in cfg["tasks"]}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/"new";parent=root/"parent"
            output.mkdir();parent.mkdir()
            c.write_new(output/"receipt.json",receipt)
            for n in range(4):c.write_new(output/f"runtime_node{n}.json",{"metrics":{}})
            raw=torch.zeros(1,3,512,512);u8=torch.zeros_like(raw,dtype=torch.uint8)
            def identify(out,rec,task,row,rank):
                return {"key":row["record_id"],"task":task,"rank":rank}
            old_entries={}
            for rank in range(8):
                task,rows=c.assigned(cfg["tasks"],receipt["rows"],rank,8);row=rows[0]
                ident=identify(output,receipt,task,row,rank);key=ident["key"]
                old_marker=parent/f"{key}.json";old_file=parent/f"{key}.safetensors"
                save_file({"decoded":raw,"uint8":u8},str(old_file))
                old={"tensor":old_file.name};c.write_new(old_marker,old);old_entries[key]=(parent,old_marker,old)
                markers={}
                for arm in cfg["arms"]:
                    tensor=output/f"{arm}_{key}.safetensors";save_file({"decoded":raw,"uint8":u8},str(tensor))
                    marker=output/arm/f"{key}.json";markers[arm]=marker
                    c.write_new(marker,{"identity":ident,"files":{tensor.name:c.sha(tensor)},"tensor":tensor.name,
                        "plan":q.expected_plan(receipt,task,arm),"original_receipt_sha256":c.sha(old_marker),
                        "parity":q.parity({"decoded":raw,"uint8":u8},{"decoded":raw,"uint8":u8}) if arm=="one_hot" else None})
                c.write_new(output/"metrics"/f"{key}.json",{"identity":ident,"training_fit_only":True,
                    "binding":q.metric_binding(old_marker,markers,row),"metrics":{a:{"psnr":30.,"lpips":.1}
                        for a in ("original","one_hot","near_one_hot")}})
            def original(rec,task,row,rank):return old_entries[row["record_id"]]
            with mock.patch.object(q,"identity",side_effect=identify),mock.patch.object(q,"parent_prompts"),\
                 mock.patch.object(q.previous,"original",side_effect=original):
                q.verify(output,receipt)
                self.assertEqual(c.read(output/"result.json")["status"],"protocol_routed_engineering_pass")
                real_parity=q.parity
                def mismatched(a,b,size=512):
                    result=real_parity(a,b,size);return {**result,"max_abs_rgb":.01,"pass":False}
                # Pass fields remain true, but independent tensor comparison disagrees.
                with mock.patch.object(q,"parity",side_effect=mismatched),self.assertRaises(ValueError):
                    q.verify(output,receipt)
                with mock.patch.object(q,"expected_plan",return_value={}),self.assertRaises(ValueError):
                    q.verify(output,receipt)


if __name__=="__main__":unittest.main()
