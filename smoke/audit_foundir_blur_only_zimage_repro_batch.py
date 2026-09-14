#!/usr/bin/env python3
"""One launch, eight independent workers, matched off/on/off numerical controls."""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowlight_restoration.zimage_repro_batch import (
    PHASES, assignment, compare_runs, install_flash_control, seed_everything,
    summarize_comparisons, tensor_digest, validate_settings,
)
from lowlight_restoration.zimage_optimizer_trace import OptimizerTrace, trace_step, update_statistics
from smoke.audit_foundir_blur_only_zimage_optimization import (
    EndpointStudent, load_audit, load_aligned_pair, load_frozen_rows, loss_terms,
    lpips_value, objective, psnr, restore_student_fp32, save_triptych, sha256,
    validate_gpu_inventory, verify_model_artifacts, write_json,
)


def load_config(path):
    s = json.loads(path.read_text())
    validate_settings(s)
    parent = ROOT / s['parent_audit_config']
    if sha256(parent) != s['parent_audit_config_sha256']:
        raise ValueError('parent config hash changed')
    old, e, c = load_audit(parent)
    output = (ROOT / s['output_dir']).resolve()
    if output.parent != ROOT / 'checkpoints' or output.name != 'foundir_blur_only_zimage_repro_batch_v1':
        raise ValueError('unexpected output root')
    return s, old, e, c


def inventory(config):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError('eight exposed CUDA devices required')
    props = [torch.cuda.get_device_properties(i) for i in range(8)]
    return validate_gpu_inventory([p.name for p in props], [p.total_memory // 2**20 for p in props], config)


def run_phase(model, metric, item, prompt, lq, gt, named, initial, s, e,
              spec, phase, folder, seed, cpu_smoke=False):
    """Reuse frozen model but restore every trainable tensor and optimizer/RNG."""
    for _, parameter in named:
        parameter.grad = None
    with torch.no_grad():
        for name, parameter in named:
            parameter.copy_(initial[name].to(parameter.device))
    metric.reset()
    seed_everything(seed)
    params = [p for _, p in named]
    initial_digest = tensor_digest(named)
    optimizer = torch.optim.AdamW(params, lr=s['learning_rate'], weight_decay=0)
    tracer = OptimizerTrace(model) if phase == 'on' else None
    snapshots, history = [], []
    folder.mkdir(parents=True, exist_ok=False)
    first_gradient = first_update = None
    start = time.monotonic()

    def evaluate(step):
        with torch.no_grad():
            latent, rgb = model(item, prompt)
            image = model.pipe.vae_output_to_image(rgb).convert('RGB')
            row = dict(step=step, psnr=psnr(gt, image),
                       lpips=lpips_value(metric, gt, image, model.pipe.device),
                       raw_rgb_mse=float((rgb.float()-item['gt_rgb'].float()).square().mean()),
                       latent_mse=float((latent.float()-item['gt_latent'].float()).square().mean()),
                       clipped_fraction=float((rgb.abs()>1).float().mean()),
                       latent_digest=tensor_digest([('latent', latent)]))
            # write_json rejects nonfinite metric rows.
            write_json(folder / f'step_{step:06d}.json', row)
            save_triptych(folder / f'step_{step:06d}.png', f"{spec['record_id']} {phase} TRAIN ONLY", lq, image, gt)
            snapshots.append(row)
            print('REPRO_SNAPSHOT', json.dumps(dict(rank=spec['rank'], phase=phase, **row)), flush=True)

    try:
        evaluate(0)
        if not cpu_smoke:
            baseline = spec['baseline']
            if abs(snapshots[0]['psnr']-baseline['psnr']) > .02 or abs(snapshots[0]['lpips']-baseline['lpips']) > .002:
                raise RuntimeError('inherited endpoint baseline mismatch')
        with (folder / 'training.jsonl').open('x') as log, (folder / 'optimizer_trace.jsonl').open('x') as trace_log:
            for step in range(1, (2 if cpu_smoke else s['steps']) + 1):
                optimizer.zero_grad(set_to_none=True)
                tracing = tracer is not None and trace_step(step)
                if tracing:
                    tracer.start()
                latent, rgb = model(item, prompt)
                if spec['arm'] == 'lora_rgb':
                    loss = (rgb.float()-item['gt_rgb'].float()).square().mean()
                else:
                    _, terms = loss_terms(latent, rgb, item, metric, e)
                    loss = objective(terms, spec['arm'], e)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                if tracer is not None:
                    tracer.stop_forward()
                loss.backward()
                trace = tracer.collect_gradients(named) if tracing else None
                if step == 1:
                    first_gradient = tensor_digest([(name, p.grad) for name, p in named])
                    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                        raise RuntimeError('frozen model received gradients')
                norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                if tracing:
                    before = [p.detach().clone() for p in params]
                optimizer.step()
                metric.reset()
                if step == 1:
                    first_update = tensor_digest(named)
                    if first_update == initial_digest:
                        raise RuntimeError('no actual parameter update')
                    if any(optimizer.state[p]['exp_avg'].dtype != torch.float32 or
                           optimizer.state[p]['exp_avg_sq'].dtype != torch.float32 for p in params):
                        raise RuntimeError('optimizer moments not FP32')
                row = dict(step=step, loss=float(loss.detach()), grad_norm=float(norm),
                           elapsed_seconds=time.monotonic()-start)
                history.append(row)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                if tracing:
                    trace.update(**row, **update_statistics(params, before))
                    trace_log.write(json.dumps(trace, allow_nan=False)+'\n')
                    trace_log.flush()
                    del before
                if step in s['snapshot_steps'] or cpu_smoke:
                    evaluate(step)
        state = {n: p.detach().cpu().contiguous() for n, p in named}
        path = folder / 'final.safetensors'
        save_file(state, path, metadata={'diagnostic_only': 'true', 'phase': phase})
        result = dict(status='complete', phase=phase, initial_parameter_digest=initial_digest,
                      first_gradient_digest=first_gradient, first_updated_parameter_digest=first_update,
                      final_parameter_digest=tensor_digest(state.items()),
                      checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha256(path),
                      snapshots=snapshots, max_gradient=max(x['grad_norm'] for x in history),
                      max_gradient_step=max(history, key=lambda x: x['grad_norm'])['step'],
                      clipped_steps=sum(x['grad_norm']>1 for x in history),
                      wall_seconds=time.monotonic()-start, steps=len(history),
                      training_log=str((folder/'training.jsonl').relative_to(ROOT)),
                      training_log_sha256=sha256(folder/'training.jsonl'),
                      trace=str((folder/'optimizer_trace.jsonl').relative_to(ROOT)),
                      trace_sha256=sha256(folder/'optimizer_trace.jsonl'))
        write_json(folder/'result.json', result)
        return result
    finally:
        if tracer is not None:
            tracer.close()
        model.zero_grad(set_to_none=True)
        metric.reset()


def worker(args, s, old, e, c):
    spec = assignment(args.worker)
    spec['record_id'] = old['record_ids'][spec['record_index']]
    spec['config_sha256'] = sha256(args.config)
    spec['student_checkpoint_sha256'] = old['student_checkpoint_sha256']
    folder = ROOT / s['output_dir'] / f"rank_{args.worker}"
    folder.mkdir(parents=True, exist_ok=False)
    runs = []
    try:
        torch.set_num_threads(4)
        device = torch.device('cpu' if args.cpu_smoke else f'cuda:{args.worker}')
        if not args.cpu_smoke:
            inventory(c)
            torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = spec['deterministic']
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(spec['deterministic'], warn_only=False)
        counters = dict(cpu_smoke=True) if args.cpu_smoke else install_flash_control(spec['deterministic'])
        seed_everything(old['seed'])
        model = EndpointStudent(c, device, ROOT/old['student_checkpoint'])
        restore_student_fp32(model, ROOT/old['student_checkpoint'])
        for name, p in model.pipe.dit.named_parameters():
            p.requires_grad_('lora_' in name)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if len(named) != 476 or any(p.dtype != torch.float32 for _, p in named):
            raise RuntimeError('trainable scope/precision changed')
        initial = {n: p.detach().cpu().clone() for n, p in named}
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
        train, _ = load_frozen_rows(c, verify_images=True)
        row = next(r for r in train if r['record_id']==spec['record_id'])
        with torch.no_grad():
            prompt = model.pipe.units[1].process(model.pipe, c['model']['prompt'], None)['prompt_embeds']
            lq, gt = load_aligned_pair(row, 64 if args.cpu_smoke else 512)
            gt_rgb = model.pipe.preprocess_image(gt)
            item = dict(lq_latent=model.pipe.vae_encoder(model.pipe.preprocess_image(lq)),
                        gt_rgb=gt_rgb, gt_latent=model.pipe.vae_encoder(gt_rgb), teacher_usable=False)
            item['teacher_latent'] = torch.zeros_like(item['gt_latent'])
        # Prompt is a tensor in this pinned pipeline; fail closed if interface changes.
        prompt_tensors = [('prompt', prompt)] if isinstance(prompt, torch.Tensor) else [(f'prompt_{i}', t) for i, t in enumerate(prompt)]
        spec['cached_input_digest'] = tensor_digest([(k, v) for k, v in item.items() if isinstance(v, torch.Tensor)] + prompt_tensors)
        previous = json.loads((ROOT/old['student_result']).read_text())
        baseline = next(r for snap in previous['snapshots'] if snap['step']==400 and snap['role']=='tiny_overfit_train'
                        for r in snap['rows'] if r['record_id']==spec['record_id'])
        spec['baseline'] = dict(psnr=baseline['output_psnr'], lpips=baseline['output_lpips'])
        spec['input_metrics'] = dict(psnr=psnr(gt, lq), lpips=lpips_value(metric, gt, lq, device))
        spec['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                                   strict_deterministic=torch.are_deterministic_algorithms_enabled(),
                                   cudnn_deterministic=torch.backends.cudnn.deterministic,
                                   cudnn_benchmark=False, tf32=False,
                                   packages={k: importlib.metadata.version(k) for k in ('torchmetrics', 'peft')})
        if not args.cpu_smoke:
            props = torch.cuda.get_device_properties(device)
            spec['environment'].update(gpu=props.name, uuid=str(getattr(props, 'uuid', 'unavailable')),
                                       flash_attention=importlib.metadata.version('flash_attn'))
        write_json(folder/'identity.json', spec)
        for phase in PHASES:
            try:
                result = run_phase(model, metric, item, prompt, lq, gt, named, initial,
                                   s, e, spec, phase, folder/phase, old['seed'], args.cpu_smoke)
            except Exception as error:
                result = dict(status='failed', phase=phase, error=repr(error), traceback=traceback.format_exc())
                write_json(folder/f'{phase}_failure.json', result)
                traceback.print_exc()
            runs.append(result)
            gc.collect()
            if not args.cpu_smoke:
                torch.cuda.empty_cache()
            print('REPRO_PHASE_END', json.dumps(dict(rank=args.worker, phase=phase, status=result['status'])), flush=True)
        spec.update(runs=runs, attention_calls=counters,
                    status='complete' if all(r['status']=='complete' for r in runs) else 'partial_failure')
        if not args.cpu_smoke and counters['flash_calls'] == 0:
            raise RuntimeError('no FlashAttention calls observed; selected backend was not exercised')
    except Exception as error:
        spec.update(status='worker_failure', runs=runs, error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    write_json(folder/'result.json', spec)
    return 0 if spec['status']=='complete' else 2


def verify_run_artifacts(run, settings):
    """Independently verify outputs before including a completed fit in comparisons."""
    for field in ('checkpoint', 'training_log', 'trace'):
        path = (ROOT/run[field]).resolve()
        if not path.is_relative_to((ROOT/settings['output_dir']).resolve()):
            raise ValueError('artifact outside batch output')
        if sha256(path) != run[field+'_sha256']:
            raise ValueError(f'artifact hash mismatch: {field}')
    folder = (ROOT/run['checkpoint']).parent
    if [row['step'] for row in run['snapshots']] != settings['snapshot_steps']:
        raise ValueError('snapshot schedule incomplete')
    for row in run['snapshots']:
        if json.loads((folder/f"step_{row['step']:06d}.json").read_text()) != row:
            raise ValueError('snapshot sidecar mismatch')
        from PIL import Image
        with Image.open(folder/f"step_{row['step']:06d}.png") as image:
            image.verify()
    history = [json.loads(line) for line in (ROOT/run['training_log']).read_text().splitlines()]
    if run['steps'] != settings['steps'] or [r['step'] for r in history] != list(range(1,settings['steps']+1)):
        raise ValueError('incomplete optimizer steps')
    if not all(math.isfinite(r[k]) for r in history for k in ('loss','grad_norm','elapsed_seconds')):
        raise ValueError('nonfinite training log')
    if run['max_gradient'] != max(r['grad_norm'] for r in history):
        raise ValueError('gradient summary mismatch')
    trace = [json.loads(line) for line in (ROOT/run['trace']).read_text().splitlines()]
    expected = [i for i in range(1,settings['steps']+1) if trace_step(i)] if run['phase']=='on' else []
    if [r['step'] for r in trace] != expected:
        raise ValueError('trace steps incomplete')


def aggregate(output, settings, config_digest, exit_codes):
    workers = []
    for rank in range(8):
        path = output/f'rank_{rank}'/'result.json'
        value = json.loads(path.read_text()) if path.exists() else dict(rank=rank, status='missing_result', runs=[])
        if value['rank'] != rank:
            raise RuntimeError('worker rank mismatch')
        if path.exists() and value['config_sha256'] != config_digest:
            raise RuntimeError('worker config mismatch')
        try:
            if value['status']=='complete':
                expected = assignment(rank)
                if any(value[k] != v for k,v in expected.items()) or [r['phase'] for r in value['runs']] != PHASES:
                    raise ValueError('worker assignment/phases mismatch')
                if exit_codes[rank] != 0:
                    raise ValueError('nonzero worker exit despite complete result')
            for run in value['runs']:
                if run['status']=='complete':
                    verify_run_artifacts(run, settings)
        except Exception as error:
            value.update(status='artifact_validation_failure', validation_error=repr(error))
            for run in value['runs']:
                if run['status']=='complete':
                    run['status']='invalid_artifacts'
        workers.append(value)
    comparisons = []
    for w in workers:
        runs = {r['phase']: r for r in w['runs']}
        for left, right, kind in [('off_a','off_b','same_gpu_repeat'), ('off_a','on','instrumentation')]:
            if left in runs and right in runs:
                comparisons.append(dict(kind=kind, rank=w['rank'], deterministic=w['deterministic'],
                                        **compare_runs(runs[left], runs[right], settings)))
    for rank in range(4):
        a, b = workers[rank], workers[rank+4]
        if a['status']=='complete' and b['status']=='complete':
            if a['cached_input_digest'] != b['cached_input_digest']:
                comparisons.append(dict(kind='cross_device', rank=rank, comparable=False, reason='cached_inputs_differ'))
            else:
                comparisons.append(dict(kind='cross_device', rank=rank, deterministic=a['deterministic'],
                                        **compare_runs(a['runs'][0], b['runs'][0], settings)))
    for rank in (0, 1, 4, 5):
        a, b = workers[rank], workers[rank+2]
        if a['status']=='complete' and b['status']=='complete':
            if a['cached_input_digest'] != b['cached_input_digest']:
                comparisons.append(dict(kind='execution_bundle', rank=rank, comparable=False, reason='cached_inputs_differ'))
            else:
                comparisons.append(dict(kind='execution_bundle', rank=rank,
                                        **compare_runs(a['runs'][0], b['runs'][0], settings)))
    complete = sum(r['status']=='complete' for w in workers for r in w['runs'])
    healthy_workers = all(w['status']=='complete' for w in workers) and all(c==0 for c in exit_codes)
    decision = summarize_comparisons(comparisons, complete)
    if not healthy_workers:
        decision.update(next_action='investigate_incomplete_controls_no_scaling')
    result = dict(status='repro_batch_complete' if complete==24 and healthy_workers else 'repro_batch_partial_failure',
                  config_sha256=config_digest, completed_fits=complete, expected_fits=24,
                  workers=workers, worker_exit_codes=exit_codes, comparisons=comparisons,
                  decision=decision, diagnostic_only=True,
                  independent_optimizers=True, sealed_internal_accessed=False,
                  official_foundir_test_accessed=False, development_gt_optimized=False, decoder_finetuned=False)
    write_json(output/'result.json', result)
    lines = ['# Consolidated reproducibility batch', '', 'DIAGNOSTIC ONLY / 非主线.', '',
             f'Completed fits: {complete}/24. Result SHA-256: `{sha256(output / "result.json")}`.', '',
             f'Next action: `{result["decision"]["next_action"]}`. No automatic quality or full-training pass.', '',
             '| Rank | Execution | Phase | Final PSNR / LPIPS | Max preclip gradient |',
             '|---:|---|---|---|---:|']
    for w in workers:
        for r in w['runs']:
            if r['status']=='complete':
                last = r['snapshots'][-1]
                lines.append(f'| {w["rank"]} | {w["deterministic"]} | {r["phase"]} | {last["psnr"]:.4f} / {last["lpips"]:.4f} | {r["max_gradient"]:.6g} |')
            else:
                lines.append(f'| {w["rank"]} | {w.get("deterministic")} | {r["phase"]} | FAILED | n/a |')
    lines += ['', 'See result.json for exact first-gradient/update/final equality, matched trajectories,',
              'cross-device controls, branch errors, identities and artifact hashes. Inspect fixed images.',
              'Both records are training-only fits; no generalization or SOW claim.']
    with (output/'report.md').open('x') as handle:
        handle.write('\n'.join(lines)+'\n')
    print('REPRO_BATCH_RESULT', json.dumps(dict(completed_fits=complete, result_sha256=sha256(output/'result.json'), decision=result['decision'])), flush=True)
    return 0 if result['status']=='repro_batch_complete' else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--cpu-smoke', action='store_true')
    args = parser.parse_args()
    s, old, e, c = load_config(args.config)
    if args.preflight:
        load_frozen_rows(c, verify_images=True)
        verify_model_artifacts(c, large_hashes=True)
        for name, mem in [('NVIDIA H100 80GB HBM3',81920), ('NVIDIA H200',143000)]:
            validate_gpu_inventory([name]*8, [mem]*8, c)
        print('REPRO_BATCH_PREFLIGHT_PASS', sha256(args.config), flush=True)
        return 0
    if args.cpu_smoke:
        if args.worker is None:
            raise ValueError('CPU smoke requires an explicit worker')
        # Isolated unique workspace-local temporary evidence; never touch GPU output.
        import tempfile
        s['output_dir'] = str(Path(tempfile.mkdtemp(prefix='zimage_repro_cpu_', dir=ROOT/'logs')).relative_to(ROOT))
    if args.worker is not None:
        return worker(args, s, old, e, c)
    inventory(c)
    verify_model_artifacts(c, large_hashes=True)
    output = ROOT/s['output_dir']
    output.mkdir(parents=True, exist_ok=False)
    sources = [str(Path(__file__).relative_to(ROOT)), 'lowlight_restoration/zimage_repro_batch.py',
               'lowlight_restoration/zimage_optimizer_trace.py',
               'smoke/train_foundir_blur_only_zimage_endpoint_distill_ddp.py',
               'smoke/train_foundir_blur_only_zimage_decoder_tail_ddp.py',
               'third_party/diffsynth-studio/diffsynth/core/attention/attention.py',
               'third_party/diffsynth-studio/diffsynth/models/z_image_dit.py',
               'third_party/diffsynth-studio/diffsynth/models/general_modules.py']
    write_json(output/'execution_identity.json', dict(config_sha256=sha256(args.config),
               student_checkpoint_sha256=old['student_checkpoint_sha256'],
               sources={p:sha256(ROOT/p) for p in sources},
               command=sys.argv, executable=sys.executable,
               git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    processes = []
    with contextlib.ExitStack() as stack:
        for rank in range(8):
            log = stack.enter_context((output/f'worker_{rank}.log').open('x'))
            cmd = [sys.executable, str(Path(__file__).resolve()), '--config', str(args.config), '--worker', str(rank)]
            processes.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        previous = None
        while any(p.poll() is None for p in processes):
            status = [p.poll() for p in processes]
            if status != previous:
                print('REPRO_WORKERS', json.dumps(status), flush=True)
                previous = status
            time.sleep(5)
        codes = [p.wait() for p in processes]
    return aggregate(output, s, sha256(args.config), codes)


if __name__ == '__main__':
    raise SystemExit(main())
