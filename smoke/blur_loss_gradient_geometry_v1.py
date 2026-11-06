"""Read-only gradient geometry; predictions concern raw SGD first-order slopes, not Adam."""
import math


def summarize(aa,bb,ab):
    na,nb=math.sqrt(max(0.,aa)),math.sqrt(max(0.,bb))
    return {'weighted_MSE_gradient_norm':na,'weighted_LPIPS_gradient_norm':nb,
      'LPIPS_to_MSE_norm_ratio':nb/na if na else None,
      'cosine':ab/(na*nb) if na*nb else None,'dot':ab,
      'mixed_gradient_norm':math.sqrt(max(0.,aa+bb+2*ab)),
      'MSE_dot_mixed':aa+ab,'LPIPS_dot_mixed':bb+ab,
      'raw_SGD_mixed_descent_first_order_increases_MSE':aa+ab<0,
      'raw_SGD_mixed_descent_first_order_increases_LPIPS':bb+ab<0}


def geometry(a,b,names):
    import torch
    if not(len(a)==len(b)==len(names)) or len(set(names))!=len(names):raise ValueError('gradient layout mismatch')
    entries=[]
    for name,x,y in zip(names,a,b):
        if x.shape!=y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('invalid gradient')
        x=x.detach().double();y=y.detach().double()
        stats=torch.stack((x.square().sum(),y.square().sum(),(x*y).sum())).cpu().tolist()
        entries.append((name,x.numel(),stats))
    groups={}
    for group,predicate in [('all',lambda n:True),('double_blocks',lambda n:n.startswith('double_blocks.')),
      ('single_blocks',lambda n:n.startswith('single_blocks.')),('lora_A',lambda n:'.lora_A.' in n),('lora_B',lambda n:'.lora_B.' in n)]:
        subset=[entry for entry in entries if predicate(entry[0])]
        if not subset:continue
        groups[group]={**summarize(*(math.fsum(e[2][i] for e in subset) for i in range(3))),
          'parameter_count':sum(e[1] for e in subset),'parameter_tensors':len(subset)}
    return groups
