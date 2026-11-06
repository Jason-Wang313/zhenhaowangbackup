"""Small LQ classification network, never an RGB generator or FLUX replacement."""
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


def views(image):
    image=image.convert('RGB');w,h=image.size
    if min(w,h)<128:raise ValueError('native128 image required')
    patches=[np.array(image.resize((128,128),Image.Resampling.BILINEAR),copy=True)]
    for x,y in [(0,0),(w-128,0),(0,h-128),(w-128,h-128),((w-128)//2,(h-128)//2)]:
        patches.append(np.array(image.crop((x,y,x+128,y+128)),copy=True))
    return np.stack(patches)


class TextureRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(6,24,5,2,2),nn.GroupNorm(6,24),nn.SiLU(),
          nn.Conv2d(24,48,3,2,1),nn.GroupNorm(6,48),nn.SiLU(),
          nn.Conv2d(48,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU())
        self.head=nn.Linear(128,1)

    def forward(self,x):
        if x.ndim!=5 or x.shape[1:]!=(6,3,128,128):raise ValueError('B x six views x RGB128 required')
        if not torch.isfinite(x).all():raise ValueError('nonfinite LQ input')
        n=len(x);z=x.flatten(0,1);high=z-F.avg_pool2d(z,5,1,2)
        z=self.encoder(torch.cat((z,high),dim=1));pooled=torch.cat((z.mean((2,3)),z.var((2,3),unbiased=False).clamp_min(1e-8).sqrt()),1)
        return self.head(pooled).view(n,6).mean(1)


def route(base_logits,margin):
    if base_logits.ndim!=2 or base_logits.shape[1]!=5 or margin.shape!=(len(base_logits),):raise ValueError('invalid route shape')
    if not torch.isfinite(base_logits).all() or not torch.isfinite(margin).all():raise ValueError('nonfinite route')
    old=base_logits.argmax(1)
    return torch.where((old==1)|(old==3),torch.where(margin>=0,1,3),old)


def sample_indices(labels,step,rank):
    if not 1<=step<=400 or rank not in (0,1):raise ValueError('fixed400/two ranks required')
    g=torch.Generator().manual_seed(2026110613+step*101+rank)
    ids=[]
    for label in (0,1):
        group=torch.where(labels==label)[0]
        if not len(group):raise ValueError('both classes required')
        ids.append(group[torch.randint(len(group),(8,),generator=g)])
    return torch.cat(ids)
