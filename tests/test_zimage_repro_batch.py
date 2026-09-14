import copy
import json
from pathlib import Path

import pytest
import torch

from lowlight_restoration.zimage_repro_batch import (
    PHASES, SNAPSHOTS, assignment, compare_runs, seed_everything,
    summarize_comparisons, tensor_digest, validate_settings,
)


def settings():
    return json.loads(Path('configs/foundir_blur_only_zimage_repro_batch_v1.json').read_text())


def run():
    return dict(status='complete', initial_parameter_digest='init',
                first_gradient_digest='grad', first_updated_parameter_digest='first',
                final_parameter_digest='final', snapshots=[dict(step=s, psnr=30., lpips=.1, latent_digest='z') for s in SNAPSHOTS])


def test_matrix_has_two_records_two_modes_two_device_replicates():
    rows = [assignment(i) for i in range(8)]
    assert len({(r['record_index'], r['deterministic'], r['replicate']) for r in rows}) == 8
    assert PHASES == ['off_a', 'on', 'off_b']
    assert all(r['arm'] == ('lora_rgb' if r['record_index']==0 else 'lora_mixed') for r in rows)
    with pytest.raises(ValueError):
        assignment(8)


@pytest.mark.parametrize('key,value', [('steps',201), ('learning_rate',1e-5), ('decoder_finetuning',True),
                                     ('official_foundir_test_accessed',True), ('snapshot_steps',[0,200])])
def test_scope_and_schedule_fail_closed(key, value):
    s = settings()
    validate_settings(s)
    s[key] = value
    with pytest.raises(ValueError):
        validate_settings(s)


def test_comparison_separates_exactness_from_metric_tolerance():
    a, b = run(), run()
    assert compare_runs(a,b,settings())['final_parameters_exact']
    b['first_gradient_digest'] = 'different'
    b['snapshots'][1]['psnr'] += .01
    result = compare_runs(a,b,settings())
    assert result['within_metric_tolerance'] and not result['first_gradient_exact']
    b['snapshots'][-1]['psnr'] += 1
    assert not compare_runs(a,b,settings())['within_metric_tolerance']
    b['initial_parameter_digest'] = 'wrong'
    assert not compare_runs(a,b,settings())['comparable']


def test_incomplete_duplicate_and_nan_are_not_passes():
    a, b = run(), run()
    b['snapshots'].append(copy.deepcopy(b['snapshots'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        compare_runs(a,b,settings())
    b = run()
    b['snapshots'][0]['psnr'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        compare_runs(a,b,settings())
    assert not compare_runs(a,dict(status='failed'),settings())['comparable']


def test_decision_never_admits_training_and_requires_complete_controls():
    rows = [dict(rank=rank,kind=kind,deterministic=True,**compare_runs(run(),run(),settings()))
            for rank in (2,3,6,7) for kind in ('same_gpu_repeat','instrumentation')]
    assert summarize_comparisons(rows,23)['next_action']=='investigate_incomplete_controls_no_scaling'
    assert not summarize_comparisons(rows[:1],24)['deterministic_same_gpu_control_exact']
    result = summarize_comparisons(rows,24)
    assert result['deterministic_same_gpu_control_exact']
    assert not result['quality_pass'] and not result['full_training_authorized_by_this_result']
    rows[0]['first_gradient_exact'] = False
    assert not summarize_comparisons(rows,24)['deterministic_same_gpu_control_exact']


def test_tensor_digest_is_content_not_container_order():
    a, b = torch.tensor([1.,2.]), torch.tensor([3.],dtype=torch.bfloat16)
    assert tensor_digest([('a',a),('b',b)])==tensor_digest([('b',b),('a',a.clone())])
    assert tensor_digest([('a',a)])!=tensor_digest([('a',a+1)])
    with pytest.raises(ValueError):
        tensor_digest([('nan',torch.tensor([float('nan')]))])


def test_restored_optimizer_and_seed_repeat_exactly():
    seed_everything(123)
    layer = torch.nn.Linear(4,2)
    initial = copy.deepcopy(layer.state_dict())
    x = torch.randn(3,4)
    hashes = []
    for _ in PHASES:
        layer.load_state_dict(initial)
        seed_everything(123)
        opt = torch.optim.AdamW(layer.parameters(),lr=5e-5,weight_decay=0)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            layer(x).square().mean().backward()
            opt.step()
        hashes.append(tensor_digest(layer.named_parameters()))
    assert len(set(hashes))==1


def test_flash_wrapper_explicitly_sets_deterministic_and_counts(monkeypatch):
    import flash_attn
    from diffsynth.core.attention import attention as backend
    from lowlight_restoration.zimage_repro_batch import install_flash_control
    calls = []
    def fake(q,k,v,deterministic=False):
        calls.append(deterministic)
        return q+k+v
    monkeypatch.setattr(flash_attn,'flash_attn_func',fake)
    monkeypatch.setattr(backend,'ATTENTION_IMPLEMENTATION','flash_attention_2')
    monkeypatch.setattr(backend,'torch_sdpa',lambda *a, **kw: 'sdpa')
    counters = install_flash_control(True)
    assert flash_attn.flash_attn_func(1,2,3)==6
    assert calls==[True] and counters['flash_calls']==1
    assert backend.torch_sdpa()=='sdpa' and counters['sdpa_calls']==1


def test_missing_workers_generate_partial_report_not_hang(tmp_path,monkeypatch):
    from smoke import audit_foundir_blur_only_zimage_repro_batch as batch
    monkeypatch.setattr(batch,'ROOT',tmp_path)
    output = tmp_path/'output'
    output.mkdir()
    assert batch.aggregate(output,settings(),'config_hash',[2]*8)==2
    result = json.loads((output/'result.json').read_text())
    assert result['completed_fits']==0 and not result['decision']['quality_pass']
    assert (output/'report.md').is_file()


def test_complete_batch_aggregates_all_matched_controls(tmp_path,monkeypatch):
    from smoke import audit_foundir_blur_only_zimage_repro_batch as batch
    monkeypatch.setattr(batch,'ROOT',tmp_path)
    # Artifact checking is separately exercised below; focus here on group coverage.
    monkeypatch.setattr(batch,'verify_run_artifacts',lambda r,s: None)
    output = tmp_path/'output'
    output.mkdir()
    for rank in range(8):
        folder = output/f'rank_{rank}'
        folder.mkdir()
        rows = []
        for phase in PHASES:
            value = run()
            value.update(phase=phase,max_gradient=.2)
            rows.append(value)
        worker = dict(**assignment(rank),status='complete',config_sha256='config_hash',
                      cached_input_digest='same_input',runs=rows)
        (folder/'result.json').write_text(json.dumps(worker))
    assert batch.aggregate(output,settings(),'config_hash',[0]*8)==0
    result = json.loads((output/'result.json').read_text())
    assert result['completed_fits']==24 and len(result['comparisons'])==24
    assert result['decision']['deterministic_same_gpu_control_exact']
    assert not result['decision']['full_training_authorized_by_this_result']


def test_artifact_verification_rejects_tampering(tmp_path,monkeypatch):
    from smoke import audit_foundir_blur_only_zimage_repro_batch as batch
    from PIL import Image
    monkeypatch.setattr(batch,'ROOT',tmp_path)
    output = tmp_path/'batch'/'rank_0'/'off_a'
    output.mkdir(parents=True)
    s = dict(output_dir='batch',steps=1,snapshot_steps=[0])
    (output/'final.safetensors').write_bytes(b'test-only-checkpoint')
    (output/'training.jsonl').write_text(json.dumps(dict(step=1,loss=.1,grad_norm=.2,elapsed_seconds=1.))+'\n')
    (output/'optimizer_trace.jsonl').write_text('')
    snapshot = dict(step=0,psnr=30.,lpips=.1)
    (output/'step_000000.json').write_text(json.dumps(snapshot))
    Image.new('RGB',(4,4)).save(output/'step_000000.png')
    r = dict(phase='off_a',steps=1,max_gradient=.2,snapshots=[snapshot])
    for key, name in [('checkpoint','final.safetensors'),('training_log','training.jsonl'),('trace','optimizer_trace.jsonl')]:
        r[key]=str((output/name).relative_to(tmp_path))
        r[key+'_sha256']=batch.sha256(output/name)
    batch.verify_run_artifacts(r,s)
    (output/'final.safetensors').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash mismatch'):
        batch.verify_run_artifacts(r,s)
