import copy
from datetime import timedelta
import json
from pathlib import Path
import random
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lowlight_restoration.lora import inject_lora
from lowlight_restoration.blur_delta_velocity_v1 import sample_indices
from lowlight_restoration.rain_global_batch_v1 import sum_gradients
from smoke import blur_singleframe_state_v1 as s
from smoke import blur_delta_state_v1 as old


def model_and_adam():
    torch.manual_seed(12)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3))
    inject_lora(model, ['0'], 2, 2.)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.)
    return model, optimizer


def parent_value():
    model, optimizer = model_and_adam()
    for _ in range(1792):
        optimizer.zero_grad(); model(torch.ones(1, 3)).square().mean().backward(); optimizer.step()
    with tempfile.TemporaryDirectory() as folder:
        # Construct the real step1792 state through the production serializer.
        r = old.save(folder, model, optimizer, step=1792, world=4, rank_rng=[old.rng_state('cpu')]*4,
                     receipt_sha='parent', history=[{'step': i+1} for i in range(1792)])
        return torch.load(Path(folder)/r['file'], map_location='cpu', weights_only=True)


def two_rank_worker(rank, directory, value):
    torch.set_num_threads(1); root = Path(directory)
    dist.init_process_group('gloo', init_method='file://'+str(root/'rendezvous'), rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        model, optimizer = model_and_adam(); reference, serial = model_and_adam()
        s.restore_parameters_adam(model, optimizer, copy.deepcopy(value))
        s.restore_parameters_adam(reference, serial, copy.deepcopy(value))
        torch.manual_seed(800+rank); random.seed(900+rank)
        parameters = [p for p in model.parameters() if p.requires_grad]
        reference_parameters = [p for p in reference.parameters() if p.requires_grad]
        history = copy.deepcopy(value['history'])
        def loss(m, index):
            x = torch.tensor([[index/1024., .3, -1.]])
            target = x.square()*.2
            return (m(x)-target).square().mean()
        def update(step):
            optimizer.zero_grad(set_to_none=True)
            for index in s.indices(step, rank): (loss(model, index)/8).backward()
            sum_gradients(parameters); torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
        def save(step):
            rngs = [None]*2; dist.all_gather_object(rngs, old.rng_state('cpu'))
            if rank == 0: old.save(root, model, optimizer, step=step, world=2, rank_rng=rngs,
                                    receipt_sha='new', history=history)
            dist.barrier()
        save(1792)
        update(1792)
        serial.zero_grad(set_to_none=True)
        (sum(loss(reference, i) for i in s.indices(1792, 0, world=1))/8).backward()
        torch.nn.utils.clip_grad_norm_(reference_parameters, 1.); serial.step()
        for a,b in zip(parameters, reference_parameters): torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
        for a,b in zip(optimizer.state.values(), serial.state.values()):
            for key in ('exp_avg','exp_avg_sq'): torch.testing.assert_close(a[key],b[key],rtol=1e-5,atol=1e-7)
        history.append({'step':1793}); save(1793)
        expected_random = torch.rand(3); expected_python = random.random()
        update(1793); expected = [p.detach().clone() for p in parameters]
        checkpoint, _ = s.latest(root, 'new')
        s.restore(model, optimizer, checkpoint, rank, 'cpu')
        assert torch.equal(torch.rand(3), expected_random) and random.random() == expected_python
        update(1793)
        for a,b in zip(parameters, expected): assert torch.equal(a,b)
        if rank == 0: (root/'pass.json').write_text(json.dumps({'global8_Adam_matches_serial': True,'two_rank_exact_resume': True}))
        dist.barrier()
    finally: dist.destroy_process_group()


class ExposureTests(unittest.TestCase):
    def test_all16_views_exactly100_exposures_and_no_epoch_replacement(self):
        counts = [0]*16
        for epoch in range(100):
            union = []
            for step in range(1792+2*epoch,1794+2*epoch):
                assigned = [s.indices(step,rank) for rank in range(2)]
                self.assertEqual([len(v) for v in assigned],[4,4])
                batch = sum(assigned,[])
                self.assertEqual(len(set(batch)),8)
                self.assertEqual(sorted(batch), sorted(s.indices(step,0,world=1)))
                union.extend(batch)
                for index in batch: counts[index]+=1
            self.assertEqual(sorted(union),list(range(16)))
        self.assertEqual(counts,[100]*16)
        for step in (1791,1992):
            with self.assertRaises(ValueError):s.indices(step,0)

    def test_model_adam_migration_does_not_claim_to_restore_old_rng(self):
        value = parent_value(); model, optimizer = model_and_adam()
        torch.manual_seed(99); before = torch.get_rng_state().clone()
        s.restore_parameters_adam(model, optimizer, value)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(old.parameter_hash(old.trainable_state(model)), value['parameter_hash'])
        self.assertTrue(all(float(v['step'])==1792 for v in optimizer.state.values()))

    def test_two_process_global8_Adam_and_exact_same_world_resume(self):
        value = parent_value()
        with tempfile.TemporaryDirectory() as folder:
            mp.start_processes(two_rank_worker, args=(folder,value), nprocs=2, start_method='fork', join=True)
            self.assertTrue(all(json.loads((Path(folder)/'pass.json').read_text()).values()))
            with self.assertRaises(ValueError): s.latest(folder,'new',world=4)
            with self.assertRaises(ValueError): s.latest(folder,'wrong')
            marker=json.loads((Path(folder)/'steps/step_001793.json').read_text())
            with (Path(folder)/marker['file']).open('ab') as handle: handle.write(b'corrupt')
            with self.assertRaises(ValueError): s.latest(folder,'new')


if __name__ == '__main__': unittest.main()
