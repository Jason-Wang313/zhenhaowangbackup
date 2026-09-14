import torch
import pytest

from lowlight_restoration.zimage_optimizer_trace import OptimizerTrace, update_statistics, trace_step
from smoke.audit_foundir_blur_only_zimage_optimization import load_stability


def test_instrumentation_does_not_change_forward_or_gradient():
    torch.manual_seed(0)
    model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.RMSNorm(4,eps=1e-6),torch.nn.Linear(4,1))
    x=torch.randn(3,4)
    base=model(x)
    base.square().sum().backward()
    expected=[p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    tracer=OptimizerTrace(model)
    tracer.start()
    result=model(x)
    tracer.stop_forward()
    result.square().sum().backward()
    assert torch.equal(result,base)
    assert all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),expected))
    data=tracer.collect_gradients(list(model.named_parameters()))
    assert data['normalization_inputs']['1']['denominator_min']>0
    before={k:v.clone() for k,v in tracer.normalizations.items()}
    model(x*100)
    assert all(torch.equal(v,tracer.normalizations[k]) for k,v in before.items())
    tracer.close()


def test_trace_reports_actual_updates_and_dense_tail():
    p=torch.nn.Parameter(torch.tensor([3.,4.]))
    before=[p.detach().clone()]
    with torch.no_grad():
        p.add_(torch.tensor([.3,.4]))
    data=update_statistics([p],before)
    assert data['update_l2']==pytest.approx(.5,abs=1e-6)
    assert data['relative_update_l2']==pytest.approx(.1,abs=1e-6)
    assert trace_step(1) and trace_step(20) and all(trace_step(i) for i in range(160,201))
    assert not trace_step(19)


def test_stability_config_rejects_changed_objective(tmp_path):
    import json
    from pathlib import Path
    path=Path('configs/foundir_blur_only_zimage_lora_stability_v1.json')
    s,e,c=load_stability(path)
    assert s['stability']['learning_rates']==[5e-5,2e-5,1e-5,5e-6]
    value=json.loads(path.read_text())
    value['record_objectives']=['lora_mixed','lora_rgb']
    invalid=tmp_path/'invalid.json'
    invalid.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='assignment'):
        load_stability(invalid)
