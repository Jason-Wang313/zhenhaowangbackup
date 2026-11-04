import ast
import copy
from pathlib import Path
import unittest
from smoke.frozen_router_runtime_policy_v1 import check_runtimes


def runtimes():
    values = [dict(gpu_name=name, nvidia_driver='NVRM version: NVIDIA UNIX x86_64 Kernel Module '+driver,
                 torch='2.7.0+cu128', torch_cuda='12.8', cudnn=90701,
                 transformers='5.14.1', world=8, local_gpus=2,
                 metrics={'LPIPS': 'fixed'}, memory_policy={'query_chunk': 256})
            for name, driver in [('NVIDIA H100 80GB HBM3', '570.124.06'),
                                 ('NVIDIA H100 80GB HBM3', '570.124.06'),
                                 ('NVIDIA H200', '595.58.03'),
                                 ('NVIDIA H100 80GB HBM3', '595.58.03')]]
    for i in (2, 3):
        values[i]['nvidia_driver'] = 'NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.58.03  Release Build \nGCC version: gcc 11.4'
    return values


class MixedDriverTests(unittest.TestCase):
    def test_actual_cluster_releases_allowed_and_records_preserved(self):
        values = runtimes(); before = copy.deepcopy(values)
        check_runtimes(values)
        self.assertEqual(values, before)

    def test_software_scoring_policy_and_topology_mismatch_rejected(self):
        for field, value in [('torch', 'other'), ('torch_cuda', '12.9'),
                             ('cudnn', 1), ('transformers', 'other'),
                             ('metrics', {}), ('memory_policy', {}), ('world', 4)]:
            with self.subTest(field=field):
                values = runtimes(); values[2][field] = value
                with self.assertRaises(ValueError): check_runtimes(values)

    def test_invalid_gpu_driver_and_missing_node_rejected(self):
        for field, value in [('gpu_name', 'NVIDIA A100'), ('nvidia_driver', '')]:
            values = runtimes(); values[2][field] = value
            with self.assertRaises(ValueError): check_runtimes(values)
        with self.assertRaises(ValueError): check_runtimes(runtimes()[:3])

    def test_training_body_unchanged(self):
        old = ast.parse(Path('smoke/train_foundir_frozen_expert_router_v1.py').read_text())
        new = ast.parse(Path('smoke/train_foundir_frozen_expert_router_mixed_driver_v1.py').read_text())
        def body(tree):
            return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name not in ('main', 'runtime', 'check_cluster_runtimes')}
        self.assertEqual(body(old), body(new))


if __name__ == '__main__':
    unittest.main()
