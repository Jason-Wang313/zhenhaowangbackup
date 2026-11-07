import copy
from datetime import timedelta
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lowlight_restoration.rain_global_batch_v1 import sum_gradients
from smoke import blur_delta_state_v1 as serializer
from smoke import blur_fullgrid_state_v1 as parent
from smoke import blur_fullgrid_1x2_state_v1 as state
from smoke import blur_fullgrid_1x2_snapshots_v1 as snapshots
from smoke.evaluate_foundir_blur_fullgrid_1x2_official_v1 import assigned, metric_summary
from tests.test_blur_fullgrid_state_v1 import model_and_adam
from tests import test_blur_delta_official_v1 as official_fixture


def parent_fixture():
    model, optimizer = model_and_adam()
    model(torch.ones(1, 3)).square().mean().backward(); optimizer.step()
    for value in optimizer.state.values(): value['step'].fill_(state.START)
    rngs = []
    for rank in range(4):
        random.seed(901+rank); torch.manual_seed(801+rank)
        rngs.append(serializer.rng_state('cpu'))
    with tempfile.TemporaryDirectory() as folder:
        marker = serializer.save(folder, model, optimizer, step=state.START, world=4,
            rank_rng=rngs, receipt_sha='old', history=[{'step':i+1} for i in range(state.START)])
        return parent.load_step(folder, state.START, 'old')[0]


def two_rank_worker(rank, directory, value):
    torch.set_num_threads(1); root = Path(directory)
    dist.init_process_group('gloo', init_method='file://'+str(root/'rendezvous'), rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        model, optimizer = model_and_adam(); reference, serial = model_and_adam()
        state.migrate(model, optimizer, copy.deepcopy(value), rank, 'cpu')
        assert torch.equal(torch.get_rng_state(), value['rank_rng'][rank+2]['torch_cpu'])
        assert random.getstate() == value['rank_rng'][rank+2]['python']
        assert serializer.parameter_hash(serializer.trainable_state(model)) == value['parameter_hash']
        state.restore_parameters_adam(reference, serial, copy.deepcopy(value))
        parameters = [p for p in model.parameters() if p.requires_grad]
        reference_parameters = [p for p in reference.parameters() if p.requires_grad]
        history = copy.deepcopy(value['history'])
        def loss(m, index):
            x = torch.tensor([[index/15360., .3, -1.]])
            return (m(x)-x.square()*.2).square().mean()
        def update(step):
            optimizer.zero_grad(set_to_none=True)
            for index in state.indices(step, rank): (loss(model, index)/8).backward()
            sum_gradients(parameters); torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
        def save(step):
            rngs = [None]*2; dist.all_gather_object(rngs, serializer.rng_state('cpu'))
            if rank == 0: serializer.save(root, model, optimizer, step=step, world=2,
                rank_rng=rngs, receipt_sha='new', history=history)
            dist.barrier()
        save(state.START)
        update(state.START)
        serial.zero_grad(set_to_none=True)
        for index in parent.indices(state.START, 0, world=1): (loss(reference,index)/8).backward()
        torch.nn.utils.clip_grad_norm_(reference_parameters, 1.); serial.step()
        for a,b in zip(parameters,reference_parameters): torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
        for a,b in zip(optimizer.state.values(),serial.state.values()):
            for key in ('exp_avg','exp_avg_sq'): torch.testing.assert_close(a[key],b[key],rtol=1e-5,atol=1e-7)
        history.append({'step':state.START+1}); save(state.START+1)
        random_expected = (torch.rand(3),random.random()); update(state.START+1)
        expected = [p.detach().clone() for p in parameters]
        checkpoint,_ = state.latest(root,'new'); state.restore(model,optimizer,checkpoint,rank,'cpu')
        assert torch.equal(torch.rand(3),random_expected[0]) and random.random()==random_expected[1]
        update(state.START+1)
        for a,b in zip(parameters,expected): assert torch.equal(a,b)
        if rank == 0: (root/'pass.json').write_text(json.dumps({'global8_matches_serial':True,'same_world2_resume_exact':True}))
        dist.barrier()
    finally: dist.destroy_process_group()


class MigrationTests(unittest.TestCase):
    def test_every_remaining_step_has_identical_eight_view_set(self):
        for step in range(state.START,state.END):
            two=[state.indices(step,r) for r in range(2)]
            four=[parent.indices(step,r) for r in range(4)]
            self.assertEqual([len(g) for g in two],[4,4])
            self.assertEqual(sorted(sum(two,[])),sorted(sum(four,[])))
            self.assertEqual(two[0],[i for pair in zip(four[0],four[2]) for i in pair])
            self.assertEqual(two[1],[i for pair in zip(four[1],four[3]) for i in pair])
            self.assertEqual(len(set(sum(two,[]))),8)
        with self.assertRaises(ValueError):state.indices(state.START-1,0)
        with self.assertRaises(ValueError):state.indices(state.START,2)

    def test_real_two_rank_Adam_migration_and_restart(self):
        value=parent_fixture()
        with tempfile.TemporaryDirectory() as folder:
            mp.start_processes(two_rank_worker,args=(folder,value),nprocs=2,start_method='fork',join=True)
            self.assertTrue(all(json.loads((Path(folder)/'pass.json').read_text()).values()))
            (Path(folder)/'steps/state_003689_orphan.pt').write_bytes(b'uncommitted')
            self.assertEqual(state.latest(folder,'new')[0]['step'],3688)
            with self.assertRaises(ValueError):state.latest(folder,'wrong')

    def test_snapshot_keeps_same_eight_sources_and_rejects_bad_identity(self):
        b={'rows':[{'row':{'record_id':f'train_{i}'}} for i in range(8)]}
        rows=[{'source_index':i,'item':item,'rank':i%2,'step':3712,'parameter_hash':'model',
               'receipt_sha256':'receipt','scope':'training_fit_only',
               'metrics':{'lq':{'psnr':20.,'lpips':.4,'ssim':.5},'candidate':{'psnr':21.,'lpips':.3,'ssim':.6}}}
              for i,item in enumerate(b['rows'])]
        with patch.object(snapshots.data,'receipt_sha',return_value='receipt'):
            result=snapshots.summarize(rows,b,3712,'model')
            self.assertFalse(result['quality_gate']);self.assertEqual(result['means']['candidate']['psnr'],21.)
            for bad in (rows[:-1],rows[:-1]+rows[:1]):
                with self.assertRaises((ValueError,RuntimeError)):snapshots.summarize(bad,b,3712,'model')
            mixed=copy.deepcopy(rows);mixed[0]['rank']=1
            with self.assertRaises((ValueError,RuntimeError)):snapshots.summarize(mixed,b,3712,'model')

    def test_complete_official150_same_joint_gate(self):
        fixture=official_fixture.OfficialTests();rows=fixture.rows();groups=[assigned(rows,r) for r in range(2)]
        self.assertEqual([len(g) for g in groups],[75,75]);self.assertEqual(len({i['row']['record_id'] for g in groups for i in g}),150)
        for bad in (rows[:-1],rows[:-1]+rows[:1]):
            with self.assertRaises(ValueError):assigned(bad,0)
        self.assertTrue(metric_summary(fixture.records())['official_Blur_pass'])
        self.assertFalse(metric_summary(fixture.records(psnr=25.8))['official_Blur_pass'])
        self.assertFalse(metric_summary(fixture.records(lpips=.181))['official_Blur_pass'])


if __name__=='__main__':unittest.main()
