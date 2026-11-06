"""Train-only candidate loss aligned with mean per-image PSNR, not log of batch MSE.

No optimizer, data selection, inference change, or automatic training is included.
The scale is fixed from predeclared training probes, never official/validation GT.
"""
import math
import torch

EPSILON_RGB_MINUS1_1=(2/255)**2
MSE_WEIGHT=10.
LPIPS_WEIGHT=.05


def per_image_terms(prediction,target,lpips_values,reference_mse):
    if prediction.shape!=target.shape or prediction.ndim!=4 or len(prediction)==0:
        raise ValueError('matched nonempty NCHW RGB tensors required')
    if prediction.shape[1]!=3 or not math.isfinite(reference_mse) or reference_mse<=0:
        raise ValueError('RGB and positive fixed training reference required')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('nonfinite image')
    if lpips_values.shape!=(len(prediction),) or not torch.isfinite(lpips_values).all():
        raise ValueError('one finite LPIPS value per image required')
    if target.requires_grad:raise ValueError('training target must be fixed')
    mse=(prediction.float()-target.float()).square().mean((1,2,3))
    scale=reference_mse+EPSILON_RGB_MINUS1_1
    log_relative=((mse+EPSILON_RGB_MINUS1_1)/scale).log()
    # At mse==reference_mse the MSE derivative equals the original weight10.
    # A negative relative loss is valid: it means error below the fixed reference.
    per_image=MSE_WEIGHT*scale*log_relative+LPIPS_WEIGHT*lpips_values
    return per_image,{'gt_rgb_mse':mse,'gt_lpips':lpips_values,
                      'log_relative_mse':log_relative,
                      'effective_mse_weight':MSE_WEIGHT*scale/(mse.detach()+EPSILON_RGB_MINUS1_1)}


def objective(model,ae,inputs,gt,metric,reference_mse):
    """Same native Rain endpoint and LPIPS path; only the MSE aggregation changes."""
    from lowlight_restoration import rain_onpolicy_midpoint as prior
    if gt.shape[0]!=1:raise ValueError('inherited native endpoint is batch-one')
    decoded=prior.decode_endpoint(ae,prior.predict_endpoint(model,inputs),checkpointed=True)
    lpips=metric((decoded+1)*.5,(gt+1)*.5).reshape(1)
    values,terms=per_image_terms(decoded,gt,lpips,reference_mse)
    return values.mean(),{k:v.mean() for k,v in terms.items()}
