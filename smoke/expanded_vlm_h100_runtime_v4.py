"""V4 four-worker execution; preserve weights/dtypes and bound VAE workspace.

Native PyTorch convolutions replace cuDNN only inside VAE encode/decode. This
is a numerical runtime change, NOT a measured H100 qualification or a claim of
bitwise equality with the old H200 cache. No quantization/resize/offload is used.
"""
from functools import wraps

POLICY = 'vae_native_conv_fp32_query_chunk256_v1'


def shard(items, rank, world=4):
    if world != 4 or type(rank) is not int or not 0 <= rank < world:
        raise ValueError('two nodes x two local workers required')
    return items[rank::world]


def bounded_vae(ae):
    import torch
    if getattr(ae, '_expanded_v4_native_conv', False):
        return ae
    def wrap(original):
        @wraps(original)
        def call(*args, **kwargs):
            # Prevent cuDNN algorithm workspaces; leave FP32 weights and inputs
            # intact. Chunked attention is installed by the inherited loader.
            with torch.backends.cudnn.flags(enabled=False):
                return original(*args, **kwargs)
        return call
    ae.encode = wrap(ae.encode)
    ae.decode = wrap(ae.decode)
    ae._expanded_v4_native_conv = True
    return ae


def features(output, bound, rank, device):
    import gc
    import torch
    from PIL import Image
    from smoke import train_foundir_frozen_expert_router_mixed_driver_v1 as impl
    from smoke.train_qwen_multidegradation_lora_ddp import load_model
    from lowlight_restoration.qwen_lora import load_qwen_adapter
    from lowlight_restoration.qwen_five_class_gate import image_embedding_sequence
    from lowlight_restoration.frozen_expert_router import pooled_image_features
    from flux2.util import load_flow_model, load_ae
    assigned = shard(bound['items'], rank)
    missing = [i for i in assigned if impl.cached(output, 'vlm', i) is None]
    if missing:
        model, processor, _ = load_model(bound['qcfg'], variant='merger_trainable', device=device)
        load_qwen_adapter(model, impl.c.path(impl.ROOT, bound['qwen']['adapter']))
        model.eval().requires_grad_(False)
        for number, item in enumerate(missing):
            lq = impl.pixels(item, 'lq', 'cpu')
            array = ((lq[0]+1)*127.5).round().clamp(0,255).byte().permute(1,2,0).numpy()
            inputs = processor.image_processor(images=[Image.fromarray(array)], return_tensors='pt')
            with torch.no_grad():
                seq = image_embedding_sequence(model.get_image_features(
                    inputs.pixel_values.to(device), inputs.image_grid_thw.to(device)))
                if len(seq)!=1 or seq[0].ndim!=2 or seq[0].shape[-1]!=3584:
                    raise ValueError('unexpected visual merger tokens')
                value = seq[0].float().mean(0,keepdim=True)
            if not torch.isfinite(value).all(): raise ValueError('nonfinite visual feature')
            impl.commit(output,'vlm',item,{'vlm':value},values={'GT_input':False})
            if number%10==0: print(f'EXPANDED_V4_VLM rank={rank} completed={number+1}/{len(missing)}',flush=True)
        del model, processor, inputs, seq, value
        gc.collect(); torch.cuda.empty_cache()
    missing = [i for i in assigned if impl.cached(output,'features',i) is None]
    if missing:
        model=load_flow_model('flux.2-klein-base-4b',device=device).eval().requires_grad_(False)
        ae=bounded_vae(impl.native.hopper.configure_vae(
            load_ae('flux.2-klein-base-4b',device=device).eval().requires_grad_(False)))
        for number,item in enumerate(missing):
            lq=impl.pixels(item,'lq',device)
            with torch.no_grad():
                reference=ae.encode(lq).to(torch.bfloat16)
                image=pooled_image_features(model,reference)
            vlm=impl.cached(output,'vlm',item)[1]['vlm']
            if image.shape!=(1,6144) or not torch.isfinite(image).all(): raise ValueError('invalid image feature')
            impl.commit(output,'features',item,{'image':image,'vlm':vlm,'reference':reference},
                        values={'GT_input':False,'vlm_receipt_sha256':impl.c.sha(output/'vlm'/(item['key']+'.json'))})
            if number%10==0: print(f'EXPANDED_V4_FEATURE rank={rank} completed={number+1}/{len(missing)}',flush=True)
        del model,ae
        gc.collect();torch.cuda.empty_cache()
