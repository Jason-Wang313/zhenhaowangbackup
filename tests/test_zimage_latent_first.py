import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lowlight_restoration.zimage_latent_first import (
    SNAPSHOTS, average_gradients, baseline_replay_gate, joint_phase, loss_terms, metric_gate, render_report, validate_settings,
)
from smoke.train_foundir_blur_only_zimage_latent_first_ddp import (
    LatentFirstStudent, distributed_rehearsal, update,
)

ROOT=Path(__file__).resolve().parents[1]


def settings():
    return json.loads((ROOT/'configs/foundir_blur_only_zimage_latent_first_v1.json').read_text())


def settings_v2():
    return json.loads((ROOT/'configs/foundir_blur_only_zimage_latent_first_v2.json').read_text())


class Metric(torch.nn.Module):
    def forward(self,a,b):
        return (a-b).square().mean()

    def reset(self):
        pass


class ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(2,2)
        self.frozen=torch.nn.Parameter(torch.ones(2),requires_grad=False)

    def forward(self,item,prompt,decode=True):
        z=self.linear(item['lq_latent'])
        return z,z.tanh() if decode else None


def items_for(rank):
    return [dict(lq_latent=torch.tensor([[.1+rank/10,.2+i/10]]),
                 gt_latent=torch.tensor([[.4,.5]]),gt_rgb=torch.tensor([[.3,.6]])) for i in range(2)]


def test_frozen_scope_and_schedule():
    s=settings()
    validate_settings(s)
    for key in ('teacher_imitation','decoder_finetuning','development_gt_optimized',
                'qwen_degradation_conditioning','mixed_degradations',
                'sealed_internal_accessed','official_foundir_test_accessed'):
        for value in (True,None):
            bad=dict(s,**{key:value})
            with pytest.raises(ValueError): validate_settings(bad)
    for key in ('steps','latent_only_steps','selection_step','global_batch_size','learning_rate','seed'):
        with pytest.raises(ValueError): validate_settings(dict(s,**{key:s[key]+1}))
    assert not joint_phase(1,s) and not joint_phase(800,s)
    assert joint_phase(801,s) and joint_phase(1200,s)
    for step in (0,1201):
        with pytest.raises(ValueError): joint_phase(step,s)
    validate_settings(settings_v2())
    bad=copy.deepcopy(settings_v2())
    bad['baseline_replay']['historical_per_record_max_abs_psnr_delta_db']=.051
    with pytest.raises(ValueError): validate_settings(bad)


def baseline_rows(psnr_offset=0.,lpips_offset=0.,digest='same'):
    rows=[]
    for role in ('tiny_overfit_train','visual_development'):
        for i in range(16):
            rows.append(dict(record_id=f'{role}_{i}',role=role,input_psnr=20.,input_lpips=.3,
                output_psnr=21.+psnr_offset,output_lpips=.2+lpips_offset,latent_mse=.1,raw_rgb_mse=.2,
                clipped_fraction=.01,latent_digest=digest,rgb_digest=digest,image_sha256=digest))
    return rows


def test_baseline_replay_exactness_and_bounded_historical_drift():
    current=baseline_rows()
    historical=baseline_rows(psnr_offset=.01,lpips_offset=.001)
    result=baseline_replay_gate(current,copy.deepcopy(current),historical,settings_v2())
    assert result['status']=='pass' and all(result['checks'].values())
    assert result['maxima']['per_record_abs_psnr_delta_db']==pytest.approx(.01)
    changed=copy.deepcopy(current)
    changed[3]['rgb_digest']='different'
    with pytest.raises(ValueError,match='replay differs'):
        baseline_replay_gate(current,changed,historical,settings_v2())
    historical[0]['output_psnr']-=.1
    result=baseline_replay_gate(current,copy.deepcopy(current),historical,settings_v2())
    assert result['status']=='failure' and not result['checks']['historical_per_record_psnr_bounded']


def test_latent_only_skips_decoder_and_metric():
    class ForbiddenDecoder:
        def __call__(self,*args,**kwargs):
            raise AssertionError('decoder/LPIPS must not run during warmup')
    mock=SimpleNamespace(velocity=lambda lq,*args:lq*.2,
                         pipe=SimpleNamespace(vae_decoder=ForbiddenDecoder()))
    item=items_for(0)[0]
    item['lq_latent'].requires_grad_(True)
    z,rgb=LatentFirstStudent.forward(mock,item,None,decode=False)
    assert rgb is None
    loss,terms=loss_terms(z,rgb,item,ForbiddenDecoder(),settings(),False)
    assert set(terms)=={'latent_mse'}
    loss.backward()
    assert item['lq_latent'].grad is not None


def test_joint_loss_formula_and_gradients():
    item=items_for(0)[0]
    z=torch.zeros_like(item['gt_latent'],requires_grad=True)
    rgb=torch.zeros_like(item['gt_rgb'],requires_grad=True)
    total,terms=loss_terms(z,rgb,item,Metric(),settings(),True)
    torch.testing.assert_close(total,terms['latent_mse']+terms['pixel_mse']+.05*terms['lpips'])
    total.backward()
    assert z.grad.abs().sum()>0 and rgb.grad.abs().sum()>0
    with pytest.raises(ValueError): loss_terms(z,None,item,None,settings(),True)


def snapshots():
    return [dict(step=step,role=role,metrics=dict(psnr_gain_db=1.,lpips_improvement=.1,
                 latent_mse=2. if step==0 else 1.)) for step in SNAPSHOTS
            for role in ('tiny_overfit_train','visual_development')]


def test_gate_requires_both_metrics_both_roles_and_visual_review():
    rows=snapshots()
    result=metric_gate(rows,1200)
    assert result['automatic_metric_gate_pass']
    assert not result['quality_gate_pass'] and not result['full_training_started']
    for role in ('tiny_overfit_train','visual_development'):
        for key in ('psnr_gain_db','lpips_improvement'):
            bad=copy.deepcopy(rows)
            next(r for r in bad if r['step']==1200 and r['role']==role)['metrics'][key]=0.
            assert not metric_gate(bad,1200)['automatic_metric_gate_pass']
    rows[0]['metrics']['latent_mse']=.5
    assert not metric_gate(rows,1200)['automatic_metric_gate_pass']


def test_gate_rejects_missing_duplicate_and_nonfinite_snapshots():
    for rows in (snapshots()[1:],snapshots()+[snapshots()[0]]):
        with pytest.raises(ValueError): metric_gate(rows,1200)
    for index in (0,4,13):
        bad=snapshots()
        bad[index]['metrics']['latent_mse']=float('nan')
        with pytest.raises(ValueError): metric_gate(bad,1200)


def test_report_keeps_diagnostic_and_visual_review_limits():
    rows=snapshots()
    for row in rows:
        row['metrics'].update(input_psnr=23.,input_lpips=.3,output_psnr=24.,output_lpips=.2)
    report=render_report(dict(config_sha256='config',checkpoint='checkpoint',checkpoint_sha256='hash',
                              snapshots=rows,**metric_gate(rows,1200)))
    assert 'NOT successful teacher distillation' in report
    assert 'Human visual review remains pending' in report
    assert 'Full training has NOT started' in report
    assert 'PSNR >=25.9' in report


def test_reduction_rejects_invalid_gradients_and_world():
    p=torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError): average_gradients([p],1)
    p.grad=torch.ones_like(p)
    average_gradients([p],1)
    torch.testing.assert_close(p.grad,torch.ones(2))
    with pytest.raises(ValueError): average_gradients([p],2)
    with pytest.raises(ValueError): average_gradients([],1)
    with pytest.raises(ValueError): average_gradients([p],0)


def distributed_worker(rank,init_file):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2)
    try:
        torch.manual_seed(12)
        model=ToyStudent()
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        initial={n:p.detach().clone() for n,p in named}
        local=items_for(rank)
        report=distributed_rehearsal(model,named,initial,local,None,Metric(),settings(),2)
        assert report['two_repeats_exact']
        assert all(p.grad is None and torch.equal(p,initial[n]) for n,p in named)
        # Compare both distributed updates to a single-process global-batch reference.
        reference=ToyStudent()
        reference.load_state_dict(model.state_dict())
        params=[p for _,p in named]
        ref_params=[p for p in reference.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=0)
        ref_opt=torch.optim.AdamW(ref_params,lr=5e-5,weight_decay=0)
        for i,joint in enumerate((False,True)):
            update(model,params,local[i],None,Metric(),settings(),opt,joint,2)
            batch={key:torch.cat([items_for(r)[i][key] for r in range(2)]) for key in local[i]}
            update(reference,ref_params,batch,None,Metric(),settings(),ref_opt,joint,1)
            for actual,expected in zip(params,ref_params):
                torch.testing.assert_close(actual.grad,expected.grad,rtol=1e-5,atol=1e-7)
                torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    finally:
        dist.destroy_process_group()


def test_two_process_rehearsal_reset_and_global_batch_equivalence(tmp_path):
    mp.spawn(distributed_worker,args=(str(tmp_path/'gloo_init'),),nprocs=2,join=True)
