"""CPU-only layout, six-rank Gloo, sample weighting and exact-resume tests.

Gloo on one CPU host is NOT three physical hosts or NCCL/CUDA qualification.
"""
from datetime import timedelta
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from smoke import flux_native_energy_contract_v1 as contract
from smoke.train_foundir_flux_native_energy_v1 import (
    sample_slots,average_gradients,validate_execution,shared_storage_check,save_step,load_step,
)
from smoke.flux_native_energy_runtime_v1 import atomic_json


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([.3,-.2],dtype=torch.float64))

    def trainable_state(self):return {'weight':self.weight.detach().cpu().clone()}
    def load_trainable_state(self,state):
        with torch.no_grad():self.weight.copy_(state['weight'])


def six_rank_worker(rank,directory):
    torch.set_num_threads(1)
    root=Path(directory)
    dist.init_process_group('gloo',init_method='file://'+str(root/'rendezvous'),
                            rank=rank,world_size=6,timeout=timedelta(seconds=90))
    try:
        receipt={'spec':{'global_batch':8},'protocol_sha256':'cpu_six_rank_test_only'}
        shared_storage_check(root,receipt,rank,6)
        model=Toy();reference=Toy()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        serial=torch.optim.AdamW(reference.parameters(),lr=1e-3)
        def term(weight,slot):
            x=torch.tensor([slot/17.,1.],dtype=torch.float64)
            return ((weight*x).sum()-.07*slot)**2
        def update(step):
            optimizer.zero_grad(set_to_none=True)
            for slot in sample_slots(step,rank,6):
                (term(model.weight,slot)*(6/8)).backward()
            average_gradients(list(model.parameters()),6)
            optimizer.step()
        for step in (1,2):
            update(step)
            serial.zero_grad(set_to_none=True)
            sum(term(reference.weight,s)/8 for s in range((step-1)*8,step*8)).backward()
            serial.step()
            torch.testing.assert_close(model.weight,reference.weight,rtol=1e-12,atol=1e-12)
            if step==1:
                random.seed(100+rank);torch.manual_seed(200+rank)
                save_step(root,model,optimizer,1,receipt,rank,6,'cpu',1.,[{'step':1}])
                expected_rng=(random.random(),torch.rand(2))
        expected=model.weight.detach().clone()
        restored,_,_=load_step(root,model,optimizer,receipt,rank,6,'cpu')
        assert restored==1 and random.random()==expected_rng[0]
        assert torch.equal(torch.rand(2),expected_rng[1])
        update(2)
        assert torch.equal(model.weight,expected)
        if rank==0:atomic_json(root/'six_rank_pass.json',{'sample_mean':True,'adam_rng_resume':True,'shared_storage':True})
        dist.barrier()
    finally:dist.destroy_process_group()


def example_devices():
    return [dict(rank=r,local_rank=r%2,node_rank=r//2,host=f'node{r//2}',uuid=f'gpu{r}',
        name='NVIDIA H100' if r<2 else 'NVIDIA H200',torch='test',cuda='test',python='test',
        packages={},root='/shared/project',transport={},launch_mode='fresh') for r in range(6)]


class TopologyTests(unittest.TestCase):
    def test_default_and_layout_contract(self):
        self.assertEqual(contract.load_spec()['default_topology'],'3x2')
        self.assertEqual(contract.topology(6),(3,2))
        for world in (2,4,8):self.assertEqual(contract.topology(world),(1,world))
        with self.assertRaises(ValueError):contract.topology(3)

    def test_exact_sample_coverage_and_rotating_work(self):
        for world in (2,4,6,8):
            for step in range(1,13):
                slots=[s for r in range(world) for s in sample_slots(step,r,world)]
                self.assertEqual(sorted(slots),list(range((step-1)*8,step*8)))
        totals=[sum(len(sample_slots(s,r,6)) for s in range(1,7)) for r in range(6)]
        self.assertEqual(totals,[8]*6)

    def test_mixed_nodes_and_reject_bad_identity(self):
        devices=example_devices();validate_execution(devices,6)
        for field,value in (('uuid','gpu0'),('host','node0'),('torch','different'),('launch_mode','resume')):
            bad=copy.deepcopy(devices);bad[2][field]=value
            with self.assertRaises(RuntimeError):validate_execution(bad,6)
        bad=copy.deepcopy(devices);bad[1]['name']='NVIDIA H200'
        with self.assertRaises(RuntimeError):validate_execution(bad,6)

    def launcher(self,args=(),changes=None,tmux=False):
        env=dict(os.environ)
        for key in ('NNODES','NODE_RANK','MASTER_ADDR','MASTER_PORT','CUDA_VISIBLE_DEVICES'):
            env.pop(key,None)
        env.update(FLUX_NATIVE_DRY_RUN='1')
        env.update(changes or {})
        filename='start_foundir_flux_native_energy_v1_tmux.sh' if tmux else 'run_foundir_flux_native_energy_v1_gpu.sh'
        return subprocess.run(['bash',str(contract.ROOT/'smoke'/filename),*args],env=env,
                              text=True,capture_output=True,timeout=10)

    def test_default_needs_explicit_node_address(self):
        self.assertNotEqual(self.launcher().returncode,0)
        for rank in range(3):
            env={'NODE_RANK':str(rank),'MASTER_ADDR':'10.1.2.3'}
            result=self.launcher(changes=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('--nnodes=3',result.stdout)
            self.assertIn('--nproc_per_node=2',result.stdout)
            self.assertIn('--world-size 6',result.stdout)
            tmux=self.launcher(changes=env,tmux=True)
            self.assertEqual(tmux.returncode,0,tmux.stdout+tmux.stderr)
            self.assertIn('NODE_RANK='+str(rank),tmux.stdout)
            self.assertIn('MASTER_ADDR=10.1.2.3',tmux.stdout)

    def test_invalid_topology_does_not_launch(self):
        base={'NODE_RANK':'0','MASTER_ADDR':'10.1.2.3'}
        for patch in ({'NNODES':'2'},{'NODE_RANK':'3'},{'MASTER_ADDR':'127.0.0.1'},
                      {'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5'},{'CUDA_VISIBLE_DEVICES':'0,0'},
                      {'MASTER_PORT':'99999'}):
            self.assertNotEqual(self.launcher(changes={**base,**patch}).returncode,0)
        for world in (2,4,8):
            result=self.launcher(('light_explicit',str(world),'fresh'),{'NODE_RANK':'2','MASTER_ADDR':'old-node','NNODES':'3'})
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('--standalone',result.stdout)
            self.assertNotIn('old-node',result.stdout)

    def test_node_prepare_only_node_zero_writes(self):
        from unittest.mock import patch
        from smoke.train_foundir_flux_native_energy_v1 import prepare
        with tempfile.TemporaryDirectory(prefix='native-prepare-') as directory:
            output=Path(directory)/'run'
            with patch.object(contract,'preflight',return_value={'protocol_sha256':'test'}):
                prepare(output,6,False,0)
                before=(output/'preflight.json').read_bytes()
                prepare(output,6,False,1);prepare(output,6,False,2)
                self.assertEqual(before,(output/'preflight.json').read_bytes())
                with self.assertRaises(FileExistsError):prepare(output,6,False,0)

    def test_six_rank_sample_weighting_and_resume(self):
        with tempfile.TemporaryDirectory(prefix='native-six-rank-') as directory:
            atomic_json(Path(directory)/'execution.json',{'cpu_test_only':True})
            mp.start_processes(six_rank_worker,args=(directory,),nprocs=6,start_method='fork',join=True)
            result=json.loads((Path(directory)/'six_rank_pass.json').read_text())
            self.assertTrue(all(result.values()))


if __name__=='__main__':unittest.main()
