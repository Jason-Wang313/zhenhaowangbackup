"""LQ-only statistics and a bounded binary route correction, never an RGB restorer."""
import numpy as np
import torch
from torch import nn
from PIL import Image

FEATURE_NAMES=([f'rgb_mean_{i}' for i in range(3)]+[f'rgb_std_{i}' for i in range(3)]+
               [f'luma_p{p}' for p in (1,10,25,50,75,90,99)]+['dark_fraction','bright_fraction']+
               [f'native_haar_mad_{i}' for i in range(3)]+['native_abs_haar','native_abs_gradient'])


def features(image):
    """Only LQ RGB; global exposure plus five fixed native crops, no label/path/GT."""
    if image.mode!='RGB':image=image.convert('RGB')
    a=np.asarray(image.resize((512,512),Image.Resampling.BILINEAR),dtype=np.float32)/255
    luma=a@np.array([.2126,.7152,.0722],np.float32)
    values=[*a.mean((0,1),dtype=np.float64),*a.std((0,1),dtype=np.float64),
            *np.percentile(luma,[1,10,25,50,75,90,99]),float((luma<=8/255).mean()),float((luma>=247/255).mean())]
    w,h=image.size
    if min(w,h)<512:raise ValueError('native512 requires dimensions >=512')
    cuts=[]
    for x,y in [(0,0),(w-512,0),(0,h-512),(w-512,h-512),((w-512)//2,(h-512)//2)]:
        crop=np.asarray(image.crop((x,y,x+512,y+512)),dtype=np.float32)/255
        haar=(crop[::2,::2]-crop[1::2,::2]-crop[::2,1::2]+crop[1::2,1::2])/2
        centered=haar-np.median(haar,axis=(0,1))
        mad=np.median(np.abs(centered),axis=(0,1))/.67448975
        gradient=(np.abs(np.diff(crop,axis=0)).mean()+np.abs(np.diff(crop,axis=1)).mean())/2
        cuts.append([*mad,float(np.abs(haar).mean()),float(gradient)])
    values.extend(np.mean(cuts,axis=0).tolist());result=np.asarray(values,dtype=np.float32)
    if result.shape!=(len(FEATURE_NAMES),) or not np.isfinite(result).all():raise ValueError('invalid physical features')
    return result


class BinaryCorrection(nn.Module):
    """Keep Blur/Haze/Rain decisions exactly; only disambiguate existing Noise/Lowlight."""
    def __init__(self,mean,scale):
        super().__init__();self.register_buffer('mean',mean.float());self.register_buffer('scale',scale.float())
        self.linear=nn.Linear(len(FEATURE_NAMES),1);nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)

    def forward(self,base_logits,physical):
        if base_logits.shape[-1]!=5 or physical.shape[-1]!=len(FEATURE_NAMES):raise ValueError('shape mismatch')
        if not torch.isfinite(base_logits).all() or not torch.isfinite(physical).all():raise ValueError('nonfinite inputs')
        z=((physical.float()-self.mean)/self.scale.clamp_min(1e-6)).clamp(-8,8)
        delta=self.linear(z).squeeze(-1);margin=base_logits[:,1]-base_logits[:,3]+delta
        old=base_logits.argmax(-1);binary=torch.where(margin>=0,1,3)
        route=torch.where((old==1)|(old==3),binary,old)
        return margin,route,delta
