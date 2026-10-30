import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest

from smoke import flux_tail_distributed_v3 as cluster

ROOT=Path(__file__).resolve().parents[1]


class ClusterTests(unittest.TestCase):
    def test_pair_stream_and_weights_preserve_all_32_pairs(self):
        for step in (1,4,5,8,51,200):
            for world in (2,4,6,8):
                shards=[cluster.training_slots(step,i,world) for i in range(world)]
                self.assertEqual(sorted(sum(shards,[])),list(range((step-1)*32,step*32)))
                # DDP rank mean of weighted local sums is the true sample mean.
                actual=sum(sum(s*s*cluster.pair_loss_weight(world) for s in slots) for slots in shards)/world
                expected=sum(s*s for s in range((step-1)*32,step*32))/32
                self.assertAlmostEqual(actual,expected)
        self.assertEqual([len(cluster.training_slots(1,i,6)) for i in range(6)],[6,6,5,5,5,5])

    def test_full_cache_and_probe_coverage(self):
        for count in (24,8):
            for world in (2,4,6,8):
                self.assertEqual(sorted(sum([cluster.assigned(range(count),i,world) for i in range(world)],[])),list(range(count)))

    def identities(self,world):
        nodes,local=cluster.topology(world)
        return [dict(rank=i,node_rank=i//local,local_rank=i%local,local_world_size=local,
                     hostname=f'host{i//local}',uuid=f'gpu{i}',family='H100' if i%2 else 'H200',
                     memory_bytes=80*1024**3,capability=[9,0],runtime={'torch':'same'}) for i in range(world)]

    def test_mixed_h100_h200_on_all_supported_layouts(self):
        for world in (2,4,6,8):cluster.validate_identities(self.identities(world),world)

    def test_bad_identity_runtime_or_topology_fails_closed(self):
        for key,value in [('uuid','gpu0'),('hostname','host0'),('family','A100'),
                          ('memory_bytes',40*1024**3),('capability',[8,0]),('local_rank',7),
                          ('runtime',{'torch':'other'})]:
            rows=self.identities(6);rows[-1][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):cluster.validate_identities(rows,6)

    def launch(self,world,rank=0,mode='qualification',**updates):
        env={k:v for k,v in os.environ.items() if k not in
             ('NNODES','NODE_RANK','MASTER_ADDR','MASTER_PORT','CUDA_VISIBLE_DEVICES','NCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME')}
        env.update(FLUX_TAIL_V3_PYTHON=sys.executable,FLUX_TAIL_V3_DRY_RUN='1',
                   MASTER_ADDR='10.247.94.142',NODE_RANK=str(rank))
        env.update(updates)
        return subprocess.run(['bash',str(ROOT/'smoke/run_foundir_flux_tail_relative_energy_v3_gpu.sh'),mode,'test_run',str(world)],
                              env=env,cwd='/tmp',capture_output=True,text=True)

    def test_launch_plans_use_global_and_local_counts_separately(self):
        for world in (2,4,6,8):
            nodes,local=cluster.topology(world)
            for rank in range(nodes):
                r=self.launch(world,rank);self.assertEqual(r.returncode,0,r.stderr)
                self.assertIn(f'--nnodes={nodes}',r.stdout)
                self.assertIn(f'--nproc_per_node={local}',r.stdout)
                self.assertIn(f'--node_rank={rank}',r.stdout)
                self.assertIn(f'--gpus {world}',r.stdout)
                self.assertIn(f'WALL_CAP={43200//world}',r.stdout)
                self.assertNotIn('--standalone',r.stdout)

    def test_bad_launch_layout_rank_devices_and_master_rejected(self):
        for values in [dict(NNODES='2'),dict(NODE_RANK='3'),dict(CUDA_VISIBLE_DEVICES='0,0'),
                       dict(CUDA_VISIBLE_DEVICES='0'),dict(MASTER_ADDR='127.0.0.1'),dict(MASTER_PORT='99999')]:
            with self.subTest(values=values):self.assertNotEqual(self.launch(6,**values).returncode,0)

    def test_connectivity_does_not_select_training(self):
        r=self.launch(6,mode='connectivity');self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('--connectivity-only',r.stdout);self.assertIn('WALL_CAP=300',r.stdout)

    def test_tmux_explicitly_transports_topology_and_network(self):
        env=dict(os.environ,FLUX_TAIL_V3_PYTHON=sys.executable,FLUX_TAIL_V3_DRY_RUN='1',
                 NNODES='3',NODE_RANK='2',MASTER_ADDR='10.247.94.142',MASTER_PORT='29683',
                 CUDA_VISIBLE_DEVICES='1,0',NCCL_SOCKET_IFNAME='eth0',NCCL_IB_DISABLE='1')
        r=subprocess.run(['bash',str(ROOT/'smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh'),
                          'qualification','qualification_3x2_direct_v1','6'],
                         env=env,cwd='/tmp',capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);args=shlex.split(r.stdout)
        for expected in ['NODE_RANK=2','NNODES=3','CUDA_VISIBLE_DEVICES=1,0',
                         'MASTER_ADDR=10.247.94.142','MASTER_PORT=29683','NCCL_SOCKET_IFNAME=eth0']:
            self.assertIn(expected,args)
        self.assertEqual(args[-4:],['qualification','qualification_3x2_direct_v1','6','qualification_3x2_direct_v1'])

    def test_single_two_gpu_ignores_stale_cluster_rendezvous(self):
        r=self.launch(2,NNODES='3',NODE_RANK='2',MASTER_ADDR='10.0.0.99')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('--nnodes=1',r.stdout)
        self.assertIn('--node_rank=0',r.stdout)
        self.assertIn('--master_addr=127.0.0.1',r.stdout)
        self.assertIn('WALL_CAP=21600',r.stdout)
        self.assertEqual([len(cluster.training_slots(1,i,2)) for i in range(2)],[16,16])


if __name__=='__main__':unittest.main()
