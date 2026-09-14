import inspect
import json
from pathlib import Path

import pytest
import torch

from lowlight_restoration.zimage_padding_compat import (
    build_padding_function,padding_operator_probe,padding_where,
)


def test_padding_operator_exact_forward_backward_and_restores_flags():
    before=torch.are_deterministic_algorithms_enabled()
    result=padding_operator_probe('cpu')
    assert result['status']=='pass' and len(result['cases'])==6
    assert torch.are_deterministic_algorithms_enabled()==before


@pytest.mark.parametrize('bad', ['mask_shape','mask_dtype','token_shape','token_dtype','trainable_token'])
def test_padding_rejects_unsupported_contract(bad):
    x=torch.ones(8,4)
    mask=torch.ones(8,dtype=torch.bool)
    token=torch.ones(1,4)
    if bad=='mask_shape': mask=mask[:,None]
    if bad=='mask_dtype': mask=mask.float()
    if bad=='token_shape': token=token[0]
    if bad=='token_dtype': token=token.double()
    if bad=='trainable_token': token.requires_grad_(True)
    with pytest.raises(ValueError):
        padding_where(x,mask,token)


def test_patch_is_process_local_and_changes_only_two_assignments():
    from diffsynth.pipelines import z_image
    original,repaired,metadata=build_padding_function(z_image)
    assert z_image.model_fn_z_image_turbo is original
    assert metadata['replacement_count']==2
    assert original.__code__.co_argcount==repaired.__code__.co_argcount
    assert repaired.__globals__['_padding_where'] is padding_where
    assert repaired.__defaults__==original.__defaults__
    assert len(metadata['transformed_function_sha256'])==64
    # The original function and on-disk checkout remain untouched.
    assert 'cap_feats[torch.cat' in inspect.getsource(original)


def test_repair_config_reuses_exact_parent_and_only_failed_ranks(tmp_path):
    from smoke.repair_foundir_blur_only_zimage_repro_batch import load_repair
    path=Path('configs/foundir_blur_only_zimage_repro_repair_v1.json')
    repair,s,old,e,c,reused,patch=load_repair(path,verify_artifacts=False)
    assert set(reused)=={0,1,4,5} and s['padding_repair']
    assert repair['worker_ranks']==[2,3,6,7]
    bad=json.loads(path.read_text())
    bad['worker_ranks']=[0,1,2,3]
    invalid=tmp_path/'invalid.json'
    invalid.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='only failed'):
        load_repair(invalid,verify_artifacts=False)


@pytest.mark.parametrize('require_cuda_gates',[False,True])
def test_aggregate_retains_reused_config_and_provenance(tmp_path,monkeypatch,require_cuda_gates):
    from smoke import audit_foundir_blur_only_zimage_repro_batch as batch
    from lowlight_restoration.zimage_repro_batch import assignment,PHASES,SNAPSHOTS
    monkeypatch.setattr(batch,'ROOT',tmp_path)
    checked=[]
    monkeypatch.setattr(batch,'verify_run_artifacts',lambda r,s: checked.append(s['output_dir']))
    output=tmp_path/'new'
    output.mkdir()
    settings=dict(output_dir='new',snapshot_steps=SNAPSHOTS,psnr_tolerance=.05,lpips_tolerance=.005,
                  padding_repair=require_cuda_gates)
    reused={}
    for rank in range(8):
        runs=[dict(status='complete',phase=p,initial_parameter_digest='init',first_gradient_digest='grad',
                   first_updated_parameter_digest='first',final_parameter_digest='last',max_gradient=.2,
                   snapshots=[dict(step=s,psnr=30.,lpips=.1,latent_digest='z') for s in SNAPSHOTS]) for p in PHASES]
        digest='parent' if rank in (0,1,4,5) else 'new'
        w=dict(**assignment(rank),status='complete',runs=runs,config_sha256=digest,cached_input_digest='same')
        if digest=='parent':
            reused[rank]=(w,dict(settings,output_dir='parent'),'parent')
        else:
            folder=output/f'rank_{rank}'
            folder.mkdir()
            (folder/'result.json').write_text(json.dumps(w))
    assert batch.aggregate(output,settings,'new',[0]*8,reused=reused)==(2 if require_cuda_gates else 0)
    result=json.loads((output/'result.json').read_text())
    assert result['reused_worker_ranks']==[0,1,4,5]
    assert result['completed_new_fits']==(0 if require_cuda_gates else 12)
    assert result['completed_reused_fits']==12
    assert result['workers'][0]['config_sha256']=='parent'
    assert result['workers'][2]['config_sha256']=='new'
    assert checked.count('parent')==12 and checked.count('new')==(0 if require_cuda_gates else 12)
    if require_cuda_gates:
        assert 'missing successful CUDA repair gate' in result['workers'][2]['validation_error']
